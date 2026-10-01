"""Closed-loop evaluation on freshly simulated runs (seeds disjoint from training)."""

from __future__ import annotations

import numpy as np

from .config import FAULT_MODES, HEALTHY, HOURS_PER_WINDOW, RAW_BYTES_PER_WINDOW, RUL_CAP_WINDOWS
from .pipeline import EdgePipeline
from .simulator import simulate_run
from .train import ModelBundle

EVAL_SEED_BASE = 9_000_000


def _run(bundle: ModelBundle, mode: str, life: int, seed: int):
    pipe = EdgePipeline(bundle)
    adv = []
    for w, truth in simulate_run(mode, life, seed):
        a = pipe.process(w)
        if a is not None:
            adv.append((a, truth))
    return pipe, adv


def evaluate(bundle: ModelBundle, runs_per_mode: int = 6, healthy_runs: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(EVAL_SEED_BASE + seed)
    results: dict = {"per_mode": {}}
    windows_total = advisory_bytes = lat_total = 0.0
    lat_max = 0.0

    # False alarms on healthy runs.
    fa, h_windows = 0, 0
    for i in range(healthy_runs):
        pipe, adv = _run(bundle, HEALTHY, 300, EVAL_SEED_BASE + seed * 1000 + i)
        fa += len(adv)
        h_windows += pipe.stats.windows
        windows_total += pipe.stats.windows
        advisory_bytes += pipe.stats.advisory_bytes
        lat_total += pipe.stats.latency_ms_total
        lat_max = max(lat_max, pipe.stats.latency_ms_max)
    results["false_advisories_per_1000_windows"] = round(1000 * fa / max(h_windows, 1), 2)

    for m, mode in enumerate(FAULT_MODES):
        detected, first_ok, first_n, cls_ok, cls_n = 0, 0, 0, 0, 0
        leads, cls_leads, abs_err, covered, n_rul = [], [], [], 0, 0
        for i in range(runs_per_mode):
            life = int(rng.integers(250, 450))
            pipe, adv = _run(bundle, mode, life, EVAL_SEED_BASE + seed * 1000 + 100 * (m + 1) + i)
            windows_total += pipe.stats.windows
            advisory_bytes += pipe.stats.advisory_bytes
            lat_total += pipe.stats.latency_ms_total
            lat_max = max(lat_max, pipe.stats.latency_ms_max)
            if not adv:
                continue
            detected += 1
            leads.append((life - adv[0][0].window) * HOURS_PER_WINDOW)
            classified = [a for a, _ in adv if a.fault_type != "unclassified_anomaly"]
            if classified:
                first_n += 1
                first_ok += classified[0].fault_type == mode
                cls_leads.append((life - classified[0].window) * HOURS_PER_WINDOW)
            cls_n += len(classified)
            cls_ok += sum(a.fault_type == mode for a in classified)
            for a, truth in adv:
                true_h = min(truth.rul_windows, RUL_CAP_WINDOWS) * HOURS_PER_WINDOW
                abs_err.append(abs(a.rul_hours_p50 - true_h))
                covered += a.rul_hours_p10 <= true_h <= a.rul_hours_p90
                n_rul += 1
        results["per_mode"][mode] = {
            "runs": runs_per_mode,
            "detected": detected,
            "median_lead_time_hours_first_alert": round(float(np.median(leads)), 1) if leads else None,
            "median_lead_time_hours_first_classified": round(float(np.median(cls_leads)), 1) if cls_leads else None,
            "first_classified_alert_correct": f"{first_ok}/{first_n}",
            "classified_advisory_accuracy": round(cls_ok / cls_n, 3) if cls_n else None,
            "rul_mae_hours": round(float(np.mean(abs_err)), 1) if abs_err else None,
            "rul_p10_p90_coverage": round(covered / n_rul, 2) if n_rul else None,
        }

    results["bandwidth_reduction_factor"] = round(windows_total * RAW_BYTES_PER_WINDOW / max(advisory_bytes, 1))
    results["latency_ms_mean_per_window"] = round(lat_total / windows_total, 2)
    results["latency_ms_max_per_window"] = round(lat_max, 2)
    return results
