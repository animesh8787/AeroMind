"""Edge benchmark of an exported ONNX bundle: size, latency, memory, and INT8 vs FP32.

Measured on whatever CPU runs it (single thread, as ``OnnxBundle`` configures sessions). These are
NOT Jetson measurements; see deploy/jetson for the hardware kit (not yet validated on hardware).

INT8: ONNX Runtime dynamic quantisation (weights to int8) of the graphs that contain MatMul/Gemm/LSTM
weights (autoencoder in anomaly.onnx, LSTM RUL). Tree-ensemble operators are not quantised.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from .config import FAULT_MODES, HEALTHY
from .features import extract_features
from .onnx_export import FILES, OnnxBundle
from .pipeline import EdgePipeline
from .simulator import simulate_run


def _rss_mb() -> float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB
    except Exception:
        return None


def quantize_int8(src: str | Path, dst: str | Path) -> dict:
    """Copy ``src`` to ``dst`` with dynamically quantised anomaly/RUL graphs; manifest re-hashed."""
    import hashlib

    from onnxruntime.quantization import QuantType, quantize_dynamic

    src, dst = Path(src), Path(dst)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    (dst / "manifest.sig").unlink(missing_ok=True)
    manifest = json.loads((dst / "manifest.json").read_text())
    done = []
    for key in ("anomaly", "rul"):
        f = dst / FILES[key]
        tmp = f.with_suffix(".int8.onnx")
        try:
            quantize_dynamic(str(f), str(tmp), weight_type=QuantType.QInt8)
        except Exception as e:  # e.g. a graph with nothing to quantise
            manifest.setdefault("int8_skipped", {})[key] = str(e)[:200]
            continue
        os.replace(tmp, f)
        data = f.read_bytes()
        manifest["files"][key].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        done.append(key)
    manifest["dtype"] = f"int8 weights ({', '.join(done)}), float32 elsewhere"
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _latency(fn, n: int) -> dict:
    fn()  # warm-up
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(t)
    return {"median_ms": round(float(np.median(t)), 3), "p99_ms": round(float(np.percentile(t, 99)), 3)}


def benchmark(onnx_dir: str | Path, n: int = 1000, phases: bool = True, int8_dir: str | Path | None = None) -> dict:
    onnx_dir = Path(onnx_dir)
    rss0 = _rss_mb()
    ob = OnnxBundle(onnx_dir)
    rss1 = _rss_mb()
    pipe = EdgePipeline(ob)
    windows = [w for w, _ in simulate_run("bearing_wear", 300, 31, phases=phases)]
    for w in windows:
        pipe.process(w)
    x = pipe.last_features
    rul_in = np.zeros(ob.rul._row_shape, dtype=np.float32)
    out = {
        "host": f"{platform.system()} {platform.machine()} ({platform.processor() or 'cpu'}), 1 thread",
        "note": "CPU measurements on this host; not Jetson measurements",
        "files_kb": {k: round(v["bytes"] / 1024, 1) for k, v in ob.manifest["files"].items()},
        "total_kb": round(sum(v["bytes"] for v in ob.manifest["files"].values()) / 1024, 1),
        "rss_mb_after_load": None if rss1 is None else round(rss1, 1),
        "rss_mb_load_increase": None if rss0 is None or rss1 is None else round(rss1 - rss0, 1),
        "latency": {
            "anomaly": _latency(lambda: ob.anomaly.score(x), n),
            "classifier": _latency(lambda: ob.classifier.predict(x), n),
            "rul": _latency(lambda: ob.rul.predict(rul_in), n),
            "full_window_process": {"mean_ms": round(pipe.stats.latency_ms_mean, 3),
                                    "max_ms": round(pipe.stats.latency_ms_max, 3)},
        },
    }
    if int8_dir is not None:
        q = quantize_int8(onnx_dir, int8_dir)
        oq = OnnxBundle(int8_dir)
        X = np.vstack([np.stack([extract_features(w) for w, _ in simulate_run(m, 150, 77 + i, phases=phases)])
                       for i, m in enumerate((HEALTHY, *FAULT_MODES))])
        s32, s8 = ob.anomaly.score(X), oq.anomaly.score(X)
        out["int8"] = {
            "dtype": q["dtype"], "skipped": q.get("int8_skipped", {}),
            "files_kb": {k: round(v["bytes"] / 1024, 1) for k, v in q["files"].items()},
            "total_kb": round(sum(v["bytes"] for v in q["files"].values()) / 1024, 1),
            "anomaly_score_max_abs_diff": round(float(np.max(np.abs(s32 - s8))), 4),
            "anomaly_score_median_abs_diff": round(float(np.median(np.abs(s32 - s8))), 4),
            "threshold_decisions_agree": round(float(np.mean((s32 > 1) == (s8 > 1))), 4),
            "latency_anomaly": _latency(lambda: oq.anomaly.score(x), n),
            "latency_rul": _latency(lambda: oq.rul.predict(rul_in), n),
        }
    return out
