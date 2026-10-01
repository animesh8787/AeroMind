"""Command line: train, export-onnx, demo, evaluate, dashboard, federated, cmapss."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import FAULT_MODES, HEALTHY, HOURS_PER_WINDOW
from .evaluate import evaluate
from .federated import federated_demo
from .pipeline import EdgePipeline
from .simulator import simulate_run
from .train import ModelBundle, TrainConfig, train

DEFAULT_MODEL = "artifacts/bundle.joblib"
DEFAULT_ONNX = "artifacts/onnx"


def _load_backend(a):
    """The models the pipeline runs on: the joblib bundle, or its ONNX export."""
    if a.backend == "onnx":
        from .onnx_export import OnnxBundle

        return OnnxBundle(a.onnx_dir)
    return ModelBundle.load(a.model)


def _add_backend_args(p) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help="joblib bundle (sklearn backend)")
    p.add_argument("--backend", choices=("sklearn", "onnx"), default="sklearn")
    p.add_argument("--onnx-dir", default=DEFAULT_ONNX, help="exported ONNX bundle (onnx backend)")


def _cmd_train(a) -> None:
    t0 = time.time()
    bundle = train(TrainConfig(runs_per_mode=a.runs_per_mode, healthy_runs=a.healthy_runs, seed=a.seed, rul_model=a.rul,
                                phases=a.phases))
    bundle.save(a.model)
    print(f"trained in {time.time() - t0:.0f}s -> {a.model}")


def _cmd_export_onnx(a) -> None:
    from .onnx_export import OnnxBundle, check_parity, export_onnx
    from .train import collect_run, rul_inputs

    bundle = ModelBundle.load(a.model)
    a.out = a.out or (DEFAULT_ONNX if a.trees == "onnx-ml" else f"{DEFAULT_ONNX}-trt")
    manifest = export_onnx(bundle, a.out, trees=a.trees, strategy=a.strategy)
    print(f"trees: {manifest['trees']}")
    for key, info in manifest["files"].items():
        print(f"{info['file']:<16} {info['bytes'] / 1024:7.0f} KB  sha256 {info['sha256'][:12]}")
    # Parity on fresh simulated windows from every mode (seeds disjoint from training and evaluation).
    runs = [collect_run(m, 300, 8_000_000 + i) for i, m in enumerate((HEALTHY, *FAULT_MODES))]
    X = np.vstack([r.X for r in runs])
    T = np.concatenate([rul_inputs(bundle.rul, r.X, bundle.anomaly.score(r.X)) for r in runs])
    print(json.dumps(check_parity(bundle, OnnxBundle(a.out), X, T), indent=2))
    print(f"-> {a.out}")


def _cmd_demo(a) -> None:
    pipe = EdgePipeline(_load_backend(a))
    first = None
    for w, truth in simulate_run(a.mode, a.life, a.seed):
        adv = pipe.process(w)
        if adv is None:
            continue
        first = first or (adv, truth)
        print(adv.to_json() if a.json else (
            f"[t={adv.flight_hours:6.1f}h] {adv.priority:<8} {adv.fault_type:<20} conf={adv.confidence:.2f} "
            f"RUL p50={adv.rul_hours_p50:6.1f}h (p10 {adv.rul_hours_p10:.1f} / p90 {adv.rul_hours_p90:.1f}) "
            f"| true RUL={'-' if truth.rul_windows is None else f'{truth.rul_windows * HOURS_PER_WINDOW:.1f}h'} "
            f"| signals: {', '.join(adv.contributing_signals)}"))
    s = pipe.stats
    print(f"\n{s.windows} windows, {s.advisories} advisories")
    if first:
        print(f"first advisory at window {first[0].window} ({first[1].degradation:.0%} degraded)")
    elif a.mode != HEALTHY:
        print("no advisory raised before failure")
    ratio = f" (~{s.reduction_factor:,.0f}x less)" if s.advisory_bytes else ""
    print(f"downlink: {s.advisory_bytes} B vs {s.raw_bytes / 1e6:.1f} MB raw{ratio}")
    print(f"per-window latency on this host: mean {s.latency_ms_mean:.1f} ms, max {s.latency_ms_max:.1f} ms")


def _cmd_evaluate(a) -> None:
    print(json.dumps(evaluate(_load_backend(a), a.runs_per_mode, a.healthy_runs, a.seed, a.phases), indent=2))


def _cmd_dashboard(a) -> None:
    from .dashboard import backend_label, build_dashboard

    t0 = time.time()
    out = build_dashboard(_load_backend(a), a.out, backend_label(a.backend))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s; open it in a browser")


def _cmd_serve(a) -> None:
    import uvicorn

    from .server.app import create_app

    app = create_app(a.onnx_dir, a.workdir, a.speed)
    print(f"AeroMind ground station: http://{a.host}:{a.port}  (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def _cmd_roi(a) -> None:
    from .roi import Assumptions, sensitivity, simulate

    ev = evaluate(_load_backend(a), a.runs_per_mode, a.healthy_runs, 0, a.phases)
    out = simulate(ev, Assumptions(fleet_size=a.fleet, horizon_fh=a.horizon, seed=a.seed))
    out["sensitivity"] = sensitivity(ev)
    print(json.dumps(out, indent=2))


def _cmd_federated(a) -> None:
    print(json.dumps(federated_demo(a.clients, a.seed), indent=2))


def _cmd_cmapss(a) -> None:
    from . import cmapss

    cmapss.download(a.data_dir)
    out = {}
    for sub in a.subsets:
        runs = [cmapss.run_subset(a.data_dir, sub, seed=s, rul=a.rul) for s in range(a.seeds)]
        agg = {}
        for key in ("aeromind", "no_anomaly_features", "constant_baseline"):
            agg[key] = {
                m: {"mean": round(float(np.mean([r[key][m] for r in runs])), 3),
                    "std": round(float(np.std([r[key][m] for r in runs])), 3)}
                for m in runs[0][key]
            }
        agg["n_test_engines"] = runs[0]["n_test_engines"]
        out[sub] = agg
        print(f"{sub} done", flush=True)
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aeromind", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train models on simulated data")
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--runs-per-mode", type=int, default=12)
    t.add_argument("--healthy-runs", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--rul", choices=("hgb", "lstm"), default="hgb",
                   help="RUL model: gradient-boosted quantile trees, or a PyTorch LSTM over the last 30 windows")
    t.add_argument("--phases", action="store_true", help="train on flight-phase simulation (taxi, climb, cruise ...)")
    t.set_defaults(fn=_cmd_train)

    x = sub.add_parser("export-onnx", help="export the trained bundle to ONNX and check parity")
    x.add_argument("--model", default=DEFAULT_MODEL)
    x.add_argument("--out", default=None, help=f"default {DEFAULT_ONNX} (onnx-ml) or {DEFAULT_ONNX}-trt (hummingbird)")
    x.add_argument("--trees", choices=("onnx-ml", "hummingbird"), default="onnx-ml",
                   help="onnx-ml: compact, ONNX Runtime only; hummingbird: tensor ops for TensorRT")
    x.add_argument("--strategy", choices=("gemm", "tree_trav"), default="gemm", help="Hummingbird tree implementation")
    x.set_defaults(fn=_cmd_export_onnx)

    d = sub.add_parser("demo", help="stream one simulated run through the edge pipeline")
    _add_backend_args(d)
    d.add_argument("--mode", choices=(HEALTHY, *FAULT_MODES), default="bearing_wear")
    d.add_argument("--life", type=int, default=350, help="windows until failure (or run length if healthy)")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--json", action="store_true", help="print raw advisory JSON")
    d.set_defaults(fn=_cmd_demo)

    e = sub.add_parser("evaluate", help="closed-loop metrics on fresh simulated runs")
    _add_backend_args(e)
    e.add_argument("--runs-per-mode", type=int, default=6)
    e.add_argument("--healthy-runs", type=int, default=6)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--phases", action="store_true", help="evaluate on flight-phase simulation")
    e.set_defaults(fn=_cmd_evaluate)

    h = sub.add_parser("dashboard", help="write a self-contained HTML replay of simulated runs")
    _add_backend_args(h)
    h.add_argument("--out", default="artifacts/dashboard.html")
    h.set_defaults(fn=_cmd_dashboard, backend="onnx")

    v = sub.add_parser("serve", help="live ground station: simulated fleet, fault injection, OTA (needs .[app])")
    v.add_argument("--onnx-dir", default="artifacts/onnx-phases", help="phase-aware ONNX model (trained if missing)")
    v.add_argument("--workdir", default="artifacts/ground", help="keys, signed packages and installed models")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--speed", type=float, default=4.0, help="windows per second per aircraft")
    v.set_defaults(fn=_cmd_serve)

    r = sub.add_parser("roi", help="fleet maintenance simulation: reactive vs fixed-interval vs AeroMind")
    _add_backend_args(r)
    r.add_argument("--phases", action="store_true")
    r.add_argument("--runs-per-mode", type=int, default=6)
    r.add_argument("--healthy-runs", type=int, default=6)
    r.add_argument("--fleet", type=int, default=30)
    r.add_argument("--horizon", type=float, default=3000.0, help="flight hours per aircraft")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=_cmd_roi)

    f = sub.add_parser("federated", help="FedAvg autoencoder demo across simulated aircraft")
    f.add_argument("--clients", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(fn=_cmd_federated)

    c = sub.add_parser("cmapss", help="RUL benchmark on NASA C-MAPSS (downloads the data if missing)")
    c.add_argument("--data-dir", default="data/cmapss")
    c.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    c.add_argument("--seeds", type=int, default=3)
    c.add_argument("--rul", choices=("hgb", "lstm"), default="hgb")
    c.set_defaults(fn=_cmd_cmapss)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
