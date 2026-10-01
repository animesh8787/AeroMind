"""NASA IMS bearing run-to-failure data (real vibration), test 2.

Source: NSF I/UCR Center for Intelligent Maintenance Systems, via the NASA Prognostics Data
Repository ("4. Bearings"). Test 2: four Rexnord ZA-2115 bearings on one shaft at 2000 RPM,
one accelerometer per bearing, a 1 s snapshot (20,480 samples at 20 kHz) every 10 minutes for
about 7 days (984 files). At the end, bearing 1 had an outer race failure; bearings 2-4 did not fail.

Experiment (same idea as the edge pipeline, on real signals):
  * per snapshot and bearing: vibration features (RMS, kurtosis, crest, band energies) and the
    envelope-spectrum SNR at the bearing's outer-race defect frequency (BPFO);
  * anomaly detector (``models.AnomalyDetector``) fitted on that bearing's first ``fit_frac`` of
    the test and calibrated on the next ``cal_frac`` (assumed healthy: early life);
  * persistence gate (5 of the last 8 snapshots over the 99th-percentile score);
  * lead time = time from the gate first opening to the end of the test (the documented failure);
  * bearings 2-4 are controls: any gate opening on them is a false alarm.

Download needs ~1.1 GB and ``tar``/``bsdtar`` able to read 7z and RAR (libarchive: built into
Windows 10+ as ``tar``, ``libarchive-tools`` on Linux). Extracted test 2 needs ~0.5 GB.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from ..models import AnomalyDetector
from ..physics import BearingGeometry, envelope_spectrum

URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
FS = 20_000
SHAFT_HZ = 2000 / 60
# Rexnord ZA-2115 (Qiu, Lee & Lin 2006, the dataset's reference paper): 16 rollers per row,
# pitch diameter 2.815 in, roller diameter 0.331 in, contact angle 15.17 deg -> BPFO ~236.4 Hz.
GEOMETRY = BearingGeometry(n_balls=16, d_over_D=0.331 / 2.815, contact_deg=15.17)
FEATURES = ["rms", "kurtosis", "crest", "peak", "band_0_1k", "band_1_3k", "band_3_6k", "band_6_10k", "bpfo_snr"]


class DatasetUnavailable(RuntimeError):
    pass


def _tar() -> str:
    for name in ("bsdtar", "tar"):
        exe = shutil.which(name)
        if exe and "libarchive" in subprocess.run([exe, "--version"], capture_output=True, text=True).stdout:
            return exe
    raise DatasetUnavailable("need a libarchive tar (bsdtar) to unpack 7z/RAR: on Linux install "
                             "libarchive-tools; Windows 10+ 'tar' works")


def download(data_dir: str | Path = "data/ims") -> Path:
    """Fetch and unpack test 2 into ``data_dir/2nd_test`` (skipped if present)."""
    d = Path(data_dir)
    test = d / "2nd_test"
    if test.is_dir() and len(list(test.iterdir())) >= 984:
        return test
    d.mkdir(parents=True, exist_ok=True)
    tar = _tar()
    z = d / "bearings.zip"
    try:
        if not z.exists():
            urllib.request.urlretrieve(URL, z)
        with zipfile.ZipFile(z) as zf:
            inner = next(n for n in zf.namelist() if n.endswith("IMS.7z"))
            zf.extract(inner, d)
        seven = d / inner
        subprocess.run([tar, "-xf", str(seven), "-C", str(d), "2nd_test.rar"], check=True)
        subprocess.run([tar, "-xf", str(d / "2nd_test.rar"), "-C", str(d)], check=True)
    except Exception as e:
        raise DatasetUnavailable(f"could not download/unpack IMS data: {e}") from e
    finally:
        for p in (z, d / "2nd_test.rar"):
            p.unlink(missing_ok=True)
        shutil.rmtree(d / "4. Bearings", ignore_errors=True)
    return test


def _features(x: np.ndarray) -> list[float]:
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    kurt = float(np.mean(x**4) / (np.mean(x**2) ** 2 + 1e-12))
    peak = float(np.max(np.abs(x)))
    mag = np.abs(np.fft.rfft(x)) * 2 / len(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    bands = [float(np.log10(np.sum(mag[(f >= lo) & (f < hi)] ** 2) + 1e-12))
             for lo, hi in ((0, 1000), (1000, 3000), (3000, 6000), (6000, 10000))]
    return [rms, kurt, peak / (rms + 1e-12), peak, *bands, bpfo_snr(x)]


def bpfo_snr(x: np.ndarray, band=(2000.0, 8000.0), slip: float = 0.04) -> float:
    """Outer-race evidence: envelope peak within ±``slip`` of the theoretical BPFO, over the local
    noise floor, counted only if the 2x harmonic is also present (within 2 Hz of twice the peak).

    The rig is belt driven, so the true shaft speed is a little below the nominal 2000 RPM; on test 2
    the outer-race line appears near 230 Hz rather than the theoretical 236.4 Hz.
    """
    f, amp = envelope_spectrum(x, FS, band)
    target = GEOMETRY.orders()["BPFO"] * SHAFT_HZ

    def peak_snr(lo, hi, guard=40.0):
        m = (f >= lo) & (f <= hi)
        i = np.flatnonzero(m)[np.argmax(amp[m])]
        near = (np.abs(f - f[i]) <= guard) & ~m
        return float(f[i]), float(amp[i] / (np.median(amp[near]) + 1e-12))

    hz, snr = peak_snr(target * (1 - slip), target * (1 + slip))
    _, snr2 = peak_snr(2 * hz - 2, 2 * hz + 2)
    return snr if snr2 >= 2.0 else 0.0


def load_features(test_dir: str | Path, cache: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(times in hours since start, features (n_files, 4 bearings, n_features)); cached as .npz."""
    test_dir = Path(test_dir)
    cache = Path(cache) if cache else test_dir.parent / "ims_test2_features.npz"
    if cache.exists():
        z = np.load(cache)
        return z["hours"], z["X"]
    files = sorted(test_dir.iterdir())
    stamps = [datetime.strptime(p.name, "%Y.%m.%d.%H.%M.%S") for p in files]
    hours = np.array([(s - stamps[0]).total_seconds() / 3600 for s in stamps])
    X = np.empty((len(files), 4, len(FEATURES)))
    for i, p in enumerate(files):
        data = np.fromstring(p.read_text(), sep=" ").reshape(-1, 4)  # noqa: NPY201 (fast ASCII parse)
        for b in range(4):
            X[i, b] = _features(data[:, b])
    np.savez(cache, hours=hours, X=X)
    return hours, X


def gate_series(scores: np.ndarray, n: int = 5, of: int = 8, thr: float = 1.0) -> np.ndarray:
    over = (scores > thr).astype(int)
    return np.array([over[max(0, i - of + 1): i + 1].sum() >= n for i in range(len(over))])


def run(hours: np.ndarray, X: np.ndarray, fit_frac: float = 0.3, cal_frac: float = 0.1, seed: int = 0) -> dict:
    n = len(hours)
    n_fit, n_cal = int(fit_frac * n), int(cal_frac * n)
    out = {"files": n, "test_hours": round(float(hours[-1]), 1), "fit_files": n_fit, "calibration_files": n_cal,
           "assumption": f"first {fit_frac:.0%} of the test is healthy (fit), next {cal_frac:.0%} calibrates",
           "bearings": {}}
    for b in range(4):
        F = X[:, b, :]
        det = AnomalyDetector(seed=seed).fit(F[:n_fit], F[n_fit:n_fit + n_cal])
        s = det.score(F)
        gate = gate_series(s)
        after = np.arange(n) >= n_fit + n_cal
        first = int(np.flatnonzero(gate & after)[0]) if (gate & after).any() else None
        bpfo = gate_series(F[:, FEATURES.index("bpfo_snr")], thr=4.0)
        first_bpfo = int(np.flatnonzero(bpfo & after)[0]) if (bpfo & after).any() else None
        half = after & (hours <= hours[-1] / 2)
        out["bearings"][f"bearing_{b + 1}"] = {
            "failed_at_end": b == 0,
            "first_gate_open_hours": None if first is None else round(float(hours[first]), 1),
            "lead_time_hours": None if first is None else round(float(hours[-1] - hours[first]), 1),
            "gate_open_fraction_first_half": round(float(gate[half].mean()), 3) if half.any() else None,
            "gate_open_fraction_after_calibration": round(float(gate[after].mean()), 3),
            "bpfo_evidence_first_hours": None if first_bpfo is None else round(float(hours[first_bpfo]), 1),
            "bpfo_evidence_lead_hours": None if first_bpfo is None else round(float(hours[-1] - hours[first_bpfo]), 1),
        }
    out["bpfo_hz"] = round(GEOMETRY.orders()["BPFO"] * SHAFT_HZ, 1)
    # Localisation: which bearing shows the strongest outer-race line once any bearing shows it.
    snr = X[:, :, FEATURES.index("bpfo_snr")]
    ev = [v["bpfo_evidence_first_hours"] for v in out["bearings"].values() if v["bpfo_evidence_first_hours"]]
    if ev:
        late = hours >= min(ev)
        strongest = np.argmax(snr[late], axis=1)
        out["localisation"] = {
            "bpfo_snr_max": {f"bearing_{b + 1}": round(float(snr[late, b].max()), 1) for b in range(4)},
            "share_of_snapshots_strongest": {f"bearing_{b + 1}": round(float(np.mean(strongest == b)), 3)
                                             for b in range(4)},
        }
    return out
