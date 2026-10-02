"""Self-contained HTML dashboard that replays simulated runs through the edge pipeline.

Every window of every scenario is processed by ``EdgePipeline`` (with whichever model
backend is passed in) and the per-window internals are embedded in one HTML file. The
page needs no server: open it in a browser and press Play.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..core.config import FAULT_MODES, HEALTHY, HOURS_PER_WINDOW, RAW_BYTES_PER_WINDOW
from ..core.features import FEATURE_NAMES
from ..edge.pipeline import EdgePipeline, PipelineConfig
from ..core.simulator import simulate_run

# Seeds are disjoint from training (0..~6000) and from ``evaluate`` (9,000,000+).
DASHBOARD_SEED_BASE = 7_000_000
WAVE_SAMPLES = 128  # first 125 ms of each 1 s vibration window
SENSOR_FEATURES = ("vib_rms", "ac_rms", "cur_thd", "volt_mean", "temp_mean", "press_mean", "oil_mean")
PRIORITIES = ("ADVISORY", "URGENT", "CRITICAL")
FAULT_LABELS = (*FAULT_MODES, "unclassified_anomaly")


@dataclass(frozen=True)
class Scenario:
    mode: str
    life: int
    seed: int


def default_scenarios() -> list[Scenario]:
    runs = [Scenario(m, 350, DASHBOARD_SEED_BASE + i) for i, m in enumerate(FAULT_MODES)]
    return runs + [Scenario(HEALTHY, 300, DASHBOARD_SEED_BASE + len(FAULT_MODES))]


def _r(v: float, nd: int = 3) -> float:
    return float(round(float(v), nd))


def record_scenario(bundle, sc: Scenario) -> dict:
    """Run one scenario window by window and keep everything the dashboard draws."""
    pipe = EdgePipeline(bundle)
    idx = [FEATURE_NAMES.index(f) for f in SENSOR_FEATURES]
    cols: dict[str, list] = {k: [] for k in (
        "load", "score", "flags", "fault", "conf", "p10", "p50", "p90", "prio",
        "truth_rul", "degradation", "latency", "wave")}
    sensors: dict[str, list] = {f: [] for f in SENSOR_FEATURES}
    advisories = []
    for w, truth in simulate_run(sc.mode, sc.life, sc.seed):
        adv = pipe.process(w)
        x = pipe.last_features
        for f, i in zip(SENSOR_FEATURES, idx):
            sensors[f].append(_r(x[i] * (100 if f == "cur_thd" else 1)))
        a = pipe.last_assessment
        cols["load"].append(_r(w.load))
        cols["score"].append(_r(pipe.last_score))
        cols["flags"].append(pipe.last_flags)
        cols["fault"].append(FAULT_LABELS.index(a.fault) if a else -1)
        cols["conf"].append(_r(a.confidence) if a else None)
        for q, v in zip(("p10", "p50", "p90"), a.rul_hours if a else (None,) * 3):
            cols[q].append(None if v is None else _r(v, 1))
        cols["prio"].append(PRIORITIES.index(a.priority) if a else -1)
        cols["truth_rul"].append(None if truth.rul_windows is None else _r(truth.rul_windows * HOURS_PER_WINDOW, 1))
        cols["degradation"].append(_r(truth.degradation, 4))
        cols["latency"].append(_r(pipe.last_latency_ms, 2))
        cols["wave"].append([int(round(v * 100)) for v in w.vibration[:WAVE_SAMPLES]])
        if adv is not None:
            advisories.append({"i": w.t, "bytes": adv.size_bytes, "json": adv.to_dict()})

    s = pipe.stats
    first = advisories[0]["i"] if advisories else None
    return {
        "mode": sc.mode,
        "life": sc.life,
        "seed": sc.seed,
        "n": s.windows,
        **cols,
        "sensors": sensors,
        "advisories": advisories,
        "summary": {
            "advisories": s.advisories,
            "advisory_bytes": s.advisory_bytes,
            "raw_bytes": s.raw_bytes,
            "latency_ms_mean": _r(s.latency_ms_mean, 2),
            "latency_ms_max": _r(s.latency_ms_max, 2),
            "first_alert_window": first,
            "first_alert_degradation": None if first is None else cols["degradation"][first],
        },
    }


def build_dashboard(bundle, out_path: str | Path, backend: str, scenarios: list[Scenario] | None = None) -> Path:
    scenarios = scenarios or default_scenarios()
    cfg = PipelineConfig()
    data = {
        "meta": {
            "backend": backend,
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "host": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
            "hours_per_window": HOURS_PER_WINDOW,
            "raw_bytes_per_window": RAW_BYTES_PER_WINDOW,
            "threshold": cfg.anomaly_threshold,
            "persist_n": cfg.persist_n,
            "persist_of": cfg.persist_of,
            "faults": list(FAULT_LABELS),
            "priorities": list(PRIORITIES),
            "wave_samples": WAVE_SAMPLES,
            "component": EdgePipeline.__dataclass_fields__["component"].default,
        },
        "scenarios": [record_scenario(bundle, sc) for sc in scenarios],
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = resources.files("aeromind.report").joinpath("dashboard_template.html").read_text(encoding="utf-8")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.replace("/*__AEROMIND_DATA__*/null", payload), encoding="utf-8")
    return out


def backend_label(kind: str) -> str:
    if kind == "onnx":
        import onnxruntime

        return f"ONNX Runtime {onnxruntime.__version__} · CPU · 1 thread"
    import sklearn

    return f"scikit-learn {sklearn.__version__} · CPU"
