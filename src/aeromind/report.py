"""One-command evidence: ``python -m aeromind report`` regenerates every metric used in the README,
technical report and pitch, and writes ``artifacts/report/results.json`` and ``report.md``.

Nothing is copied from earlier runs: models are retrained, evaluations re-run. Public datasets are
used if present (IMS: ``data/ims``; C-MAPSS: ``data/cmapss``) and reported as unavailable otherwise.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .acars import MAX_CHARS, encode
from .config import FAULT_MODES, HEALTHY
from .evaluate import evaluate
from .onnx_export import OnnxBundle, export_onnx
from .pipeline import EdgePipeline, PipelineConfig
from .sensor_health import SensorFault, SensorFaultInjector
from .simulator import PHASES, simulate_run
from .train import TrainConfig, train

SENSOR_CASES = [("stuck", "temperature"), ("flatline", "pressure"), ("drift", "vibration"), ("dropout", "acoustic"),
                ("spike", "pressure"), ("out_of_range", "temperature"), ("dropout", "oil_debris")]


def healthy_gate_rates(models, phases: bool, runs: int = 6) -> dict:
    over = gate = n = 0
    by = {}
    for i in range(runs):
        p = EdgePipeline(models)
        for w, _ in simulate_run(HEALTHY, 300, 9_000_000 + i, phases=phases):
            p.process(w)
            n += 1
            o = p.last_score > 1.0
            over += o
            gate += p.last_flags >= 5
            by.setdefault(w.phase, []).append(o)
    return {"windows": n, "over_threshold": round(over / n, 4), "gate_open": round(gate / n, 4),
            "over_threshold_by_phase": {k: round(float(np.mean(v)), 3) for k, v in by.items()}}


def sensor_fault_metrics(models, phases: bool = True, repeats: int = 3) -> dict:
    clean_false, clean_windows = 0, 0
    for i, m in enumerate((HEALTHY, *FAULT_MODES)):
        for k in range(repeats):
            p = EdgePipeline(models)
            for w, _ in simulate_run(m, 300, 6_000_000 + 10 * i + k, phases=phases):
                p.process(w)
            clean_false += len(p.sensor_advisories)
            clean_windows += p.stats.windows
    cases = {}
    for kind, ch in SENSOR_CASES:
        comp, delays, gate = {True: 0, False: 0}, [], []
        for k in range(repeats):
            for on in (True, False):
                inj = SensorFaultInjector(SensorFault(kind, ch, start=60), seed=k)
                p = EdgePipeline(models, cfg=PipelineConfig(sensor_health=on))
                for w, _ in simulate_run(HEALTHY, 250, 6_100_000 + k, phases=phases):
                    if p.process(inj.apply(w)) is not None:
                        comp[on] += 1
                    if on and w.t > 60:
                        gate.append(p.last_flags >= 5)
                if on:
                    hit = [s.window for s in p.sensor_advisories if s.channel == ch and s.status == "FAULTY"]
                    delays.append(hit[0] - 60 if hit else None)
        cases[f"{kind}:{ch}"] = {"detection_delay_windows": delays, "component_advisories_with_health": comp[True],
                                 "component_advisories_without_health": comp[False],
                                 "gate_open_after_fault_with_health": round(float(np.mean(gate)), 4)}
    return {"false_sensor_faults_on_clean_runs": clean_false, "clean_windows": clean_windows, "cases": cases}


def acars_sizes(models, phases: bool = True) -> dict:
    lens = []
    for i, m in enumerate(FAULT_MODES):
        p = EdgePipeline(models)
        inj = SensorFaultInjector(SensorFault("stuck", "temperature", start=50))
        for w, _ in simulate_run(m, 300, 5_500_000 + i, phases=phases):
            a = p.process(inj.apply(w))
            if a is not None:
                lens.append(len(encode(a, "VT-AMA01")))
        lens += [len(encode(s, "VT-AMA01")) for s in p.sensor_advisories]
    return {"messages": len(lens), "max_chars": max(lens), "mean_chars": round(float(np.mean(lens)), 1),
            "limit": MAX_CHARS, "all_fit": max(lens) <= MAX_CHARS}


def _summary_eval(r: dict) -> dict:
    pm = r["per_mode"]
    return {"false_advisories_per_1000_windows": r["false_advisories_per_1000_windows"],
            "detected": {m: f"{v['detected']}/{v['runs']}" for m, v in pm.items()},
            "median_lead_time_h": {m: v["median_lead_time_hours_first_alert"] for m, v in pm.items()},
            "first_classified_correct": {m: v["first_classified_alert_correct"] for m, v in pm.items()},
            "rul_mae_h": {m: v["rul_mae_hours"] for m, v in pm.items()},
            "rul_p10_p90_coverage": {m: v["rul_p10_p90_coverage"] for m, v in pm.items()},
            "bandwidth_reduction_factor": r["bandwidth_reduction_factor"],
            "latency_ms_mean": r["latency_ms_mean_per_window"], "latency_ms_max": r["latency_ms_max_per_window"]}


def generate(out_dir: str | Path = "artifacts/report", cmapss_dir: str | Path = "data/cmapss",
             ims_dir: str | Path = "data/ims", cmapss_seeds: int = 3, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res: dict = {"generated": time.strftime("%Y-%m-%d %H:%M"), "note":
                 "Simulator results use synthetic data; IMS and C-MAPSS are public datasets. All values are measured "
                 "by this command except items listed as assumptions."}

    log("training models (steady simulator; flight phases + conformal RUL)...")
    steady = train(TrainConfig())
    fleet = train(TrainConfig(phases=True, conformal=True))
    export_onnx(steady, out / "models/steady")
    export_onnx(fleet, out / "models/fleet")
    m_steady, m_fleet = OnnxBundle(out / "models/steady"), OnnxBundle(out / "models/fleet")

    log("simulator evaluation...")
    ev_fleet = evaluate(m_fleet, phases=True)
    res["simulator"] = {
        "phase_aware_model_on_flight_phases": _summary_eval(ev_fleet),
        "steady_model_on_steady_runs": _summary_eval(evaluate(m_steady, phases=False)),
        "steady_model_on_flight_phases": _summary_eval(evaluate(m_steady, phases=True)),
        "conformal_q_hours": round(fleet.rul.q, 2),
    }
    log("flight-phase false alarms...")
    res["flight_phases"] = {"phases": list(PHASES),
                            "healthy_steady_model": healthy_gate_rates(m_steady, True),
                            "healthy_phase_aware_model": healthy_gate_rates(m_fleet, True)}
    log("sensor faults...")
    res["sensor_health"] = sensor_fault_metrics(m_fleet)
    res["acars"] = acars_sizes(m_fleet)

    log("fleet ROI...")
    from .roi import Assumptions, sensitivity, simulate

    roi = simulate(ev_fleet, Assumptions())
    res["roi"] = {"policies": roi["policies"], "assumptions": roi["assumptions"], "sensitivity": sensitivity(ev_fleet)}

    log("edge benchmark...")
    from .bench import benchmark

    res["edge_benchmark"] = benchmark(out / "models/fleet", n=500, int8_dir=out / "models/fleet-int8")

    log("NASA IMS...")
    from .datasets import ims

    test = Path(ims_dir) / "2nd_test"
    if test.is_dir():
        hours, X = ims.load_features(test)
        res["ims"] = ims.run(hours, X)
    else:
        res["ims"] = {"unavailable": f"{test} not found; run python -m aeromind ims to download (~1.1 GB)"}

    log("NASA C-MAPSS...")
    from . import cmapss

    if (Path(cmapss_dir) / "train_FD001.txt").exists():
        cm = {}
        for sub in cmapss.SUBSETS:
            runs = [cmapss.run_subset(cmapss_dir, sub, seed=s, rul="hgb", conformal=True) for s in range(cmapss_seeds)]
            a = [r["aeromind"] for r in runs]
            cm[sub] = {k: round(float(np.mean([x[k] for x in a])), 3) for k in a[0]}
            cm[sub]["constant_baseline_rmse"] = round(float(np.mean([r["constant_baseline"]["rmse"] for r in runs])), 1)
        res["cmapss_hgb_conformal"] = cm
    else:
        res["cmapss_hgb_conformal"] = {"unavailable": "run python -m aeromind cmapss once to download"}

    res["runtime_s"] = round(time.time() - t0)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "report.md").write_text(to_markdown(res))
    log(f"wrote {out / 'results.json'} and {out / 'report.md'} in {res['runtime_s']} s")
    return res


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def to_markdown(r: dict) -> str:
    s = r["simulator"]["phase_aware_model_on_flight_phases"]
    md = [f"# AeroMind evidence report\n\nGenerated {r['generated']} by `python -m aeromind report` ({r['runtime_s']} s). "
          + r["note"] + "\n"]
    md.append("## Simulator, flight phases (phase-aware model, conformal RUL)\n")
    md.append(_table(["Fault", "Detected", "Median lead (h)", "First classified correct", "RUL MAE (h)", "p10-p90 coverage"],
                     [[m, s["detected"][m], s["median_lead_time_h"][m], s["first_classified_correct"][m],
                       s["rul_mae_h"][m], s["rul_p10_p90_coverage"][m]] for m in s["detected"]]))
    md.append(f"\nFalse advisories: {s['false_advisories_per_1000_windows']} per 1000 healthy windows. "
              f"Downlink reduction: {s['bandwidth_reduction_factor']}x. Per-window latency: "
              f"{s['latency_ms_mean']} ms mean, {s['latency_ms_max']} ms max (CPU).\n")
    fp = r["flight_phases"]
    md.append("## Flight-phase awareness (healthy flights)\n")
    md.append(_table(["Model", "Windows over threshold", "Anomaly gate open"],
                     [["trained without flight phases", fp["healthy_steady_model"]["over_threshold"], fp["healthy_steady_model"]["gate_open"]],
                      ["phase-aware", fp["healthy_phase_aware_model"]["over_threshold"], fp["healthy_phase_aware_model"]["gate_open"]]]))
    sh = r["sensor_health"]
    md.append(f"\n## Sensor health\n\nFalse sensor faults on clean runs: {sh['false_sensor_faults_on_clean_runs']} "
              f"in {sh['clean_windows']} windows.\n")
    md.append(_table(["Injected fault", "Detection delay (windows)", "Component advisories with / without sensor health",
                      "Gate open after fault (with)"],
                     [[k, v["detection_delay_windows"], f"{v['component_advisories_with_health']} / {v['component_advisories_without_health']}",
                       v["gate_open_after_fault_with_health"]] for k, v in sh["cases"].items()]))
    a = r["acars"]
    md.append(f"\n## ACARS\n\n{a['messages']} messages, max {a['max_chars']} / mean {a['mean_chars']} characters "
              f"(limit {a['limit']}); all fit: {a['all_fit']}.\n")
    md.append("## Fleet ROI (simulation; costs and failure rates are assumptions)\n")
    md.append(_table(["Policy", "Unscheduled", "Scheduled", "Early groundings", "AOG h", "Parts", "Cost vs reactive"],
                     [[k, v["unscheduled_removals"], v["scheduled_removals"], v["early_groundings"], v["aog_hours"],
                       v["parts_used"], f"{v['cost_vs_reactive_pct']}%"] for k, v in r["roi"]["policies"].items()]))
    md.append("\nSensitivity:\n")
    md.append(_table(["Scenario", "AeroMind vs reactive", "Fixed interval vs reactive"],
                     [[x["scenario"], f"{x['aeromind_cost_vs_reactive_pct']}%", f"{x['fixed_interval_cost_vs_reactive_pct']}%"]
                      for x in r["roi"]["sensitivity"]]))
    b = r["edge_benchmark"]
    md.append(f"\n## Edge benchmark ({b['host']}; {b['note']})\n\nModels {b['total_kb']} KB; process RSS after load "
              f"{b['rss_mb_after_load']} MB. Inference median: anomaly {b['latency']['anomaly']['median_ms']} ms, classifier "
              f"{b['latency']['classifier']['median_ms']} ms, RUL {b['latency']['rul']['median_ms']} ms; full window "
              f"{b['latency']['full_window_process']['mean_ms']} ms mean.")
    if "int8" in b:
        q = b["int8"]
        md.append(f" INT8 (dynamic): {q['total_kb']} KB, anomaly decisions agree on {q['threshold_decisions_agree']:.1%} "
                  f"of windows (max score shift {q['anomaly_score_max_abs_diff']}).\n")
    ims_r = r["ims"]
    md.append("\n## NASA IMS bearings (real data, test 2)\n")
    if "unavailable" in ims_r:
        md.append(ims_r["unavailable"] + "\n")
    else:
        md.append("Lead columns are hours from the first alarm to the end of the test; only bearing 1 failed, so on "
                  "bearings 2-4 they measure alarms caused by the shared shaft, not warnings of their own failure.\n")
        md.append(_table(["Bearing", "Failed", "Gate first open (h)", "Hours before end of test", "BPFO evidence: hours before end",
                          "Gate open, first half"],
                         [[k, v["failed_at_end"], v["first_gate_open_hours"], v["lead_time_hours"], v["bpfo_evidence_lead_hours"],
                           v["gate_open_fraction_first_half"]] for k, v in ims_r["bearings"].items()]))
        if "localisation" in ims_r:
            md.append(f"\nStrongest BPFO line after first evidence: {ims_r['localisation']['share_of_snapshots_strongest']}\n")
    cm = r["cmapss_hgb_conformal"]
    md.append("\n## NASA C-MAPSS (gradient boosting + conformal interval, official test split)\n")
    if "unavailable" in cm:
        md.append(cm["unavailable"] + "\n")
    else:
        md.append(_table(["Subset", "RMSE", "MAE", "NASA score", "p10-p90 coverage", "Constant baseline RMSE"],
                         [[k, v["rmse"], v["mae"], round(v["nasa_score"]), v["p10_p90_coverage"], v["constant_baseline_rmse"]]
                          for k, v in cm.items()]))
    return "\n".join(md) + "\n"
