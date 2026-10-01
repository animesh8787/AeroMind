"""Command line: train, demo, evaluate, federated."""

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


def _cmd_train(a) -> None:
    t0 = time.time()
    bundle = train(TrainConfig(runs_per_mode=a.runs_per_mode, healthy_runs=a.healthy_runs, seed=a.seed))
    bundle.save(a.model)
    print(f"trained in {time.time() - t0:.0f}s -> {a.model}")


def _cmd_demo(a) -> None:
    pipe = EdgePipeline(ModelBundle.load(a.model))
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
    print(json.dumps(evaluate(ModelBundle.load(a.model), a.runs_per_mode, a.healthy_runs, a.seed), indent=2))


def _cmd_federated(a) -> None:
    print(json.dumps(federated_demo(a.clients, a.seed), indent=2))


def _cmd_cmapss(a) -> None:
    from . import cmapss

    cmapss.download(a.data_dir)
    out = {}
    for sub in a.subsets:
        runs = [cmapss.run_subset(a.data_dir, sub, seed=s) for s in range(a.seeds)]
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
    t.set_defaults(fn=_cmd_train)

    d = sub.add_parser("demo", help="stream one simulated run through the edge pipeline")
    d.add_argument("--model", default=DEFAULT_MODEL)
    d.add_argument("--mode", choices=(HEALTHY, *FAULT_MODES), default="bearing_wear")
    d.add_argument("--life", type=int, default=350, help="windows until failure (or run length if healthy)")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--json", action="store_true", help="print raw advisory JSON")
    d.set_defaults(fn=_cmd_demo)

    e = sub.add_parser("evaluate", help="closed-loop metrics on fresh simulated runs")
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.add_argument("--runs-per-mode", type=int, default=6)
    e.add_argument("--healthy-runs", type=int, default=6)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(fn=_cmd_evaluate)

    f = sub.add_parser("federated", help="FedAvg autoencoder demo across simulated aircraft")
    f.add_argument("--clients", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(fn=_cmd_federated)

    c = sub.add_parser("cmapss", help="RUL benchmark on NASA C-MAPSS (downloads the data if missing)")
    c.add_argument("--data-dir", default="data/cmapss")
    c.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    c.add_argument("--seeds", type=int, default=3)
    c.set_defaults(fn=_cmd_cmapss)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
