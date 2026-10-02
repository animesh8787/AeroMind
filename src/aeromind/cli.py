"""AeroMind command line: train, export-onnx, demo, evaluate, serve, report, cmapss, ims, roi, bench,
federated, dashboard, edge-agent, copilot, llm doctor."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .core.config import FAULT_MODES, HEALTHY, HOURS_PER_WINDOW
from .evaluation.evaluate import evaluate
from .learning.federated import federated_demo
from .edge.pipeline import EdgePipeline
from .core.simulator import simulate_run
from .core.train import ModelBundle, TrainConfig, train

DEFAULT_MODEL = "artifacts/bundle.joblib"
DEFAULT_ONNX = "artifacts/onnx"


def _load_backend(a):
    """The models the pipeline runs on: the joblib bundle, or its ONNX export."""
    if a.backend == "onnx":
        from .edge.onnx_export import OnnxBundle

        return OnnxBundle(a.onnx_dir)
    return ModelBundle.load(a.model)


def _add_backend_args(p) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help="joblib bundle (sklearn backend)")
    p.add_argument("--backend", choices=("sklearn", "onnx"), default="sklearn")
    p.add_argument("--onnx-dir", default=DEFAULT_ONNX, help="exported ONNX bundle (onnx backend)")


def _cmd_train(a) -> None:
    t0 = time.time()
    bundle = train(TrainConfig(runs_per_mode=a.runs_per_mode, healthy_runs=a.healthy_runs, seed=a.seed, rul_model=a.rul,
                                phases=a.phases, conformal=a.conformal))
    bundle.save(a.model)
    print(f"trained in {time.time() - t0:.0f}s -> {a.model}")


def _cmd_export_onnx(a) -> None:
    from .edge.onnx_export import OnnxBundle, check_parity, export_onnx
    from .core.train import collect_run, rul_inputs

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
    from .report.dashboard import backend_label, build_dashboard

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
    from .maintenance.roi import Assumptions, sensitivity, simulate

    ev = evaluate(_load_backend(a), a.runs_per_mode, a.healthy_runs, 0, a.phases)
    out = simulate(ev, Assumptions(fleet_size=a.fleet, horizon_fh=a.horizon, seed=a.seed))
    out["sensitivity"] = sensitivity(ev)
    print(json.dumps(out, indent=2))


def _cmd_ims(a) -> None:
    from .datasets import ims

    try:
        test = ims.download(a.data_dir)
    except ims.DatasetUnavailable as e:
        print(f"IMS data unavailable: {e}")
        return
    hours, X = ims.load_features(test)
    print(json.dumps(ims.run(hours, X), indent=2))


def _cmd_bench(a) -> None:
    from .edge.bench import benchmark, compare, run_target

    if a.compare:
        print(compare(a.compare))
        return
    if a.target:
        print(json.dumps(run_target(a.target, a.onnx_dir, a.n, a.int8_dir, a.out_dir), indent=2))
        print(f"saved to {a.out_dir}/{a.target}.json")
        return
    print(json.dumps(benchmark(a.onnx_dir, a.n, phases=True, int8_dir=a.int8_dir), indent=2))


def _cmd_edge_agent(a) -> None:
    from .edge.agent import parse_sensor_fault, run_agent
    from .server.fleet import ensure_model

    if a.sensor_fault:
        parse_sensor_fault(a.sensor_fault)
    model = ensure_model(a.onnx_dir)
    where = a.ground or "no ground station (advisories printed" + (f" and written to {a.out})" if a.out else ")")
    print(f"AeroMind edge agent {a.tail}: model {model}, publishing to {where}. Sensor input is SIMULATED. Ctrl+C to stop.",
          flush=True)
    try:
        st = run_agent(model, a.tail, a.ground, rate=a.rate, max_windows=a.max_windows, inject=a.inject,
                       inject_after=a.inject_after, life=a.life, sensor_fault=a.sensor_fault, out=a.out, seed=a.seed)
    except KeyboardInterrupt:
        print("\nstopped")
        return
    lat = np.asarray(st.latencies_ms)
    print(f"{st.windows} windows, {st.advisories} advisories, {st.sent} sent, {st.failed_posts} failed posts, "
          f"{st.backlog} queued; per-window latency on this host: mean {lat.mean():.1f} ms, max {lat.max():.1f} ms")


def _show_copilot(r) -> None:
    print(f"\n=== {r.deterministic['title']} (computed by AeroMind, not by an LLM) ===")
    print("\n".join(r.deterministic["lines"]))
    head = (f"AI COPILOT ({r.provider} / {r.model})" if r.ai_generated
            else "RULE-BASED TEMPLATE (no LLM was used for this answer)")
    print(f"\n=== {head} ===")
    print(r.text)
    print(f"\n[{r.label}]")
    for w in r.warnings:
        print(f"[note] {w}")
    print(f"[{r.note}]")


def _cmd_copilot(a) -> None:
    from .llm import Copilot, CopilotRequest, load_dotenv
    from .llm.copilot import UnknownAircraft, UnknownTask
    from .llm.schemas import TASKS
    from .edge.onnx_export import OnnxBundle
    from .server.fleet import Fleet, FleetSource, ensure_model

    load_dotenv()
    fleet = Fleet(OnnxBundle(ensure_model(a.onnx_dir)))
    if a.aircraft not in fleet.aircraft:
        raise SystemExit(f"unknown aircraft {a.aircraft}; choose one of {', '.join(fleet.aircraft)}")
    for _ in range(a.warmup):
        fleet.tick()
    if a.inject:
        fleet.inject_component(a.aircraft, a.inject, a.life)
        for _ in range(a.steps):
            fleet.tick()
    cp, src, tail = Copilot(), FleetSource(fleet), a.aircraft
    st = cp.status()
    print(f"AeroMind Maintenance Copilot | LLM STATUS: {st['status']} ({st['provider']}) | aircraft {tail} (SIMULATED)")
    if st["status"] == "FALLBACK":
        print("No LLM provider is reachable: answers are rule-based templates. Run `aeromind llm doctor`.")

    def ask(task: str, question: str) -> None:
        try:
            _show_copilot(cp.ask(CopilotRequest(task=task, tail=tail, question=question), src))
        except (UnknownAircraft, UnknownTask) as e:
            print(f"error: {e}")

    if a.task or a.ask:
        ask(a.task or "", a.ask or "")
        return
    print(f"Ask a question, or: /task NAME ({', '.join(TASKS)}), /aircraft TAIL, /inject FAULT, /step N, /quit")
    while True:
        try:
            line = input(f"\n{tail}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        cmd, _, rest = line.partition(" ")
        if cmd == "/aircraft" and rest in fleet.aircraft:
            tail = rest
        elif cmd == "/step":
            for _ in range(int(rest or 20)):
                fleet.tick()
            print("advanced the simulation")
        elif cmd == "/inject" and rest in FAULT_MODES:
            fleet.inject_component(tail, rest, a.life)
            print(f"injected simulated {rest} on {tail}; use /step 150 to let it develop")
        elif cmd == "/task":
            ask(rest.strip(), "")
        elif line.startswith("/"):
            print("unknown command")
        else:
            ask("", line)


def _cmd_llm_doctor(a) -> None:
    import os

    from .llm import LLMRouter, load_dotenv
    from .llm.groq_provider import GroqProvider
    from .llm.ollama_provider import OllamaProvider
    from .llm.router import parse_priority

    load_dotenv()
    prio_raw = os.environ.get("AEROMIND_LLM_PROVIDER", "auto")
    try:
        prio = parse_priority(prio_raw)
    except ValueError as e:
        print(f"configuration error: {e}")
        return
    groq, ollama = GroqProvider(), OllamaProvider()
    print("AeroMind LLM doctor (secrets are never shown)")
    print(f"  AEROMIND_LLM_PROVIDER : {prio_raw}  -> priority: {' > '.join(prio) or 'deterministic only'}")
    print(f"  Groq    model         : {groq.model}")
    print(f"  Groq    API key       : {'set' if groq.configured() else 'NOT set'}")
    if a.ping and groq.configured():
        try:
            groq.complete("Reply with a JSON object.", '{"ping": true} -> reply {"ok": true}')
            print("  Groq    live request  : OK")
        except Exception as e:  # ProviderError text never contains the key
            print(f"  Groq    live request  : FAILED ({e})")
    elif groq.configured():
        print("  Groq    live request  : not tried (add --ping to send one tiny request)")
    models = ollama.installed_models()
    print(f"  Ollama  URL           : {ollama.base_url}")
    print(f"  Ollama  model         : {ollama.model}")
    print(f"  Ollama  server        : {'reachable' if models is not None else 'NOT reachable'}")
    if models is not None:
        print(f"  Ollama  installed     : {', '.join(models) or 'none'}")
        print(f"  Ollama  model pulled  : {'yes' if ollama.available() else f'NO (run: ollama pull {ollama.model})'}")
    st = LLMRouter(priority=prio_raw).status()
    print(f"  Active provider       : {st['provider']}  (LLM STATUS: {st['status']})")
    print("  Deterministic fallback: always available (rule-based templates, labelled as such)")


def _cmd_report(a) -> None:
    from .report.evidence_report import generate

    generate(a.out, a.cmapss_dir, a.ims_dir, a.cmapss_seeds, lstm=a.lstm, log=lambda m: print(m, flush=True))


def _cmd_federated(a) -> None:
    if a.rare_fault:
        from .learning.federated import rare_fault_demo

        print(json.dumps(rare_fault_demo(a.clients, a.seed), indent=2))
        return
    print(json.dumps(federated_demo(a.clients, a.seed), indent=2))


def _cmd_cmapss(a) -> None:
    from .datasets import cmapss

    cmapss.download(a.data_dir)
    out = {}
    for sub in a.subsets:
        runs = [cmapss.run_subset(a.data_dir, sub, seed=s, rul=a.rul, conformal=a.conformal) for s in range(a.seeds)]
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
    for stream in (sys.stdout, sys.stderr):  # evidence text contains symbols (σ, ×) a Windows console may not encode
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
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
    t.add_argument("--conformal", action="store_true", help="conformally calibrate the RUL interval on held-out runs")
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
    v.add_argument("--onnx-dir", default="artifacts/onnx-fleet", help="phase-aware ONNX model (trained if missing)")
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

    m = sub.add_parser("ims", help="NASA IMS real bearing run-to-failure (downloads ~1.1 GB on first use)")
    m.add_argument("--data-dir", default="data/ims")
    m.set_defaults(fn=_cmd_ims)

    k = sub.add_parser("bench", help="edge benchmark of an ONNX bundle: size, load time, latency, throughput, memory, INT8")
    k.add_argument("--onnx-dir", default="artifacts/onnx-fleet")
    k.add_argument("--int8-dir", default="artifacts/onnx-fleet-int8")
    k.add_argument("-n", type=int, default=1000)
    k.add_argument("--target", choices=("laptop", "raspberry-pi"),
                   help="label and save the result to --out-dir; raspberry-pi is refused unless this machine is one")
    k.add_argument("--out-dir", default="artifacts/bench")
    k.add_argument("--compare", nargs="+", metavar="FILE", help="print saved benchmark files side by side")
    k.set_defaults(fn=_cmd_bench)

    g = sub.add_parser("edge-agent", help="run the edge pipeline as a service and send advisories to a ground station")
    g.add_argument("--onnx-dir", default="artifacts/onnx-fleet")
    g.add_argument("--tail", default="VT-EDGE01", help="fictional registration reported to the ground station")
    g.add_argument("--ground", default=None, help="ground-station URL, e.g. http://192.168.1.20:8000 (omit to run offline)")
    g.add_argument("--rate", type=float, default=4.0, help="windows per second (0 = as fast as possible)")
    g.add_argument("--max-windows", type=int, default=None)
    g.add_argument("--inject", choices=FAULT_MODES, default=None, help="inject a simulated component fault")
    g.add_argument("--inject-after", type=int, default=40, help="window at which --inject / --sensor-fault start")
    g.add_argument("--life", type=int, default=320)
    g.add_argument("--sensor-fault", default=None, metavar="KIND:CHANNEL", help="e.g. stuck:temperature")
    g.add_argument("--out", default=None, help="also append advisories to this JSONL file")
    g.add_argument("--seed", type=int, default=4242)
    g.set_defaults(fn=_cmd_edge_agent)

    cp = sub.add_parser("copilot", help="ground-side LLM maintenance copilot in the terminal (simulated fleet)")
    cp.add_argument("--aircraft", default="VT-AMA01")
    cp.add_argument("--onnx-dir", default="artifacts/onnx-fleet")
    cp.add_argument("--inject", choices=FAULT_MODES, default=None, help="inject a simulated fault first")
    cp.add_argument("--life", type=int, default=320)
    cp.add_argument("--warmup", type=int, default=40, help="windows to simulate before asking")
    cp.add_argument("--steps", type=int, default=170, help="windows to simulate after --inject")
    cp.add_argument("--ask", default=None, help="ask one question and exit")
    cp.add_argument("--task", default=None, help="run one task (e.g. explain_alert, work_order) and exit")
    cp.set_defaults(fn=_cmd_copilot)

    ll = sub.add_parser("llm", help="LLM provider tools")
    llsub = ll.add_subparsers(dest="llm_cmd", required=True)
    dr = llsub.add_parser("doctor", help="show provider configuration and availability (never prints secrets)")
    dr.add_argument("--ping", action="store_true", help="send one tiny live request to Groq")
    dr.set_defaults(fn=_cmd_llm_doctor)

    q = sub.add_parser("report", help="regenerate every metric into artifacts/report (results.json, report.md)")
    q.add_argument("--out", default="artifacts/report")
    q.add_argument("--cmapss-dir", default="data/cmapss")
    q.add_argument("--ims-dir", default="data/ims")
    q.add_argument("--cmapss-seeds", type=int, default=3)
    q.add_argument("--lstm", action="store_true", help="also train/evaluate the PyTorch LSTM (needs .[lstm]; ~25 min)")
    q.set_defaults(fn=_cmd_report)

    f = sub.add_parser("federated", help="FedAvg autoencoder demo across simulated aircraft")
    f.add_argument("--clients", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--rare-fault", action="store_true",
                   help="non-IID fleet: can an aircraft recognise fault types only other aircraft have seen?")
    f.set_defaults(fn=_cmd_federated)

    c = sub.add_parser("cmapss", help="RUL benchmark on NASA C-MAPSS (downloads the data if missing)")
    c.add_argument("--data-dir", default="data/cmapss")
    c.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    c.add_argument("--seeds", type=int, default=3)
    c.add_argument("--rul", choices=("hgb", "lstm"), default="hgb")
    c.add_argument("--conformal", action="store_true", help="conformally calibrate the RUL interval")
    c.set_defaults(fn=_cmd_cmapss)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
