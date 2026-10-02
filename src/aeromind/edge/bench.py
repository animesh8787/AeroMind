"""Edge benchmark of an exported ONNX bundle: size, latency, memory, and INT8 vs FP32.

Measured on whatever CPU runs it (single thread, as ``OnnxBundle`` configures sessions). These are
NOT Jetson measurements; see deploy/jetson (not validated on hardware) and deploy/raspberry-pi (hardware results PENDING).

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

from ..core.config import FAULT_MODES, HEALTHY
from ..core.features import extract_features
from .onnx_export import FILES, OnnxBundle
from .pipeline import EdgePipeline
from ..core.simulator import simulate_run


def detect_platform() -> dict:
    """What machine this is. ``is_raspberry_pi`` is true only when the device tree or cpuinfo says so."""
    pi_model = None
    for f in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            txt = Path(f).read_bytes().decode("utf-8", "ignore").strip("\x00 \n")
            if "raspberry pi" in txt.lower():
                pi_model = txt
                break
        except OSError:
            pass
    try:
        import onnxruntime as ort

        ort_version = ort.__version__
    except ImportError:
        ort_version = None
    return {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu": platform.processor() or None, "cores": os.cpu_count(), "onnxruntime": ort_version,
            "is_raspberry_pi": pi_model is not None, "raspberry_pi_model": pi_model}


def run_target(target: str, onnx_dir: str | Path, n: int, int8_dir: str | Path | None, out_dir: str | Path) -> dict:
    """Benchmark for a named target and save it. ``raspberry-pi`` is refused on any other machine."""
    plat = detect_platform()
    if target == "raspberry-pi" and not plat["is_raspberry_pi"]:
        raise SystemExit("refusing to label this run 'raspberry-pi': this machine does not identify as a Raspberry Pi "
                         f"({plat['system']} {plat['machine']}). Run this command on the Pi; use --target laptop here.")
    res = benchmark(onnx_dir, n, phases=True, int8_dir=int8_dir)
    res.update(target=target, status="MEASURED", platform=plat)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{target}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def compare(paths: list[str | Path]) -> str:
    """Side-by-side table of saved benchmark files. Only files that exist and say MEASURED are shown."""
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    rows = [("target", [r.get("target", "?") for r in runs]),
            ("machine", [f"{r['platform']['system']} {r['platform']['machine']}" for r in runs]),
            ("model size (KB)", [r["total_kb"] for r in runs]),
            ("model load (ms)", [r.get("model_load_ms") for r in runs]),
            ("anomaly median (ms)", [r["latency"]["anomaly"]["median_ms"] for r in runs]),
            ("classifier median (ms)", [r["latency"]["classifier"]["median_ms"] for r in runs]),
            ("RUL median (ms)", [r["latency"]["rul"]["median_ms"] for r in runs]),
            ("full window mean (ms)", [r["latency"]["full_window_process"]["mean_ms"] for r in runs]),
            ("throughput (windows/s)", [r.get("throughput_windows_per_s") for r in runs]),
            ("RSS after load (MB)", [r.get("rss_mb_after_load") for r in runs])]
    w = max(len(k) for k, _ in rows) + 2
    return "\n".join(k.ljust(w) + "".join(str(v).rjust(18) for v in vals) for k, vals in rows)


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
    t_load = time.perf_counter()
    ob = OnnxBundle(onnx_dir)
    load_ms = (time.perf_counter() - t_load) * 1000
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
        "platform": detect_platform(),
        "files_kb": {k: round(v["bytes"] / 1024, 1) for k, v in ob.manifest["files"].items()},
        "total_kb": round(sum(v["bytes"] for v in ob.manifest["files"].values()) / 1024, 1),
        "model_load_ms": round(load_ms, 1),
        "throughput_windows_per_s": round(1000.0 / pipe.stats.latency_ms_mean, 1) if pipe.stats.latency_ms_mean else None,
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
