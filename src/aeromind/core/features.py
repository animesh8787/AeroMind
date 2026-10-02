"""Multi-modal feature extraction (sensor fusion) and trend features."""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import ELEC_HZ, FS, N_ELEC, N_HIGH, TREND_WINDOW
from .simulator import SensorWindow

_VIB_BANDS = ((0, 100), (100, 250), (250, 400), (400, 513))
_FREQS = np.fft.rfftfreq(N_HIGH, 1 / FS)
_BAND_MASKS = [(_FREQS >= lo) & (_FREQS < hi) for lo, hi in _VIB_BANDS]
_AC_HF = _FREQS >= 200
_ELEC_BIN = {h: int(round(h * ELEC_HZ * N_ELEC / FS)) for h in (1, 3, 5)}

FEATURE_NAMES = [
    "load",
    "vib_rms", "vib_kurtosis", "vib_crest",
    "vib_band_0_100", "vib_band_100_250", "vib_band_250_400", "vib_band_400_512",
    "ac_rms", "ac_hf_energy",
    "cur_rms", "cur_thd",
    "volt_mean", "volt_std",
    "temp_mean",
    "press_mean", "press_std",
    "oil_mean", "oil_max",
    "ambient_c", "altitude_kft",  # operating context (air data)
]
N_FEATURES = len(FEATURE_NAMES)


def _log_energy(mag: np.ndarray, mask: np.ndarray) -> float:
    return float(np.log10(np.sum(mag[mask] ** 2) + 1e-6))


def extract_features(w: SensorWindow) -> np.ndarray:
    """Fuse one window of all six sensor families into a flat feature vector."""
    v = w.vibration
    v_rms = np.sqrt(np.mean(v**2))
    centred = v - v.mean()
    kurt = np.mean(centred**4) / (np.mean(centred**2) ** 2 + 1e-12)
    crest = np.max(np.abs(v)) / (v_rms + 1e-12)
    v_mag = np.abs(np.fft.rfft(v)) * 2 / N_HIGH
    bands = [_log_energy(v_mag, m) for m in _BAND_MASKS]

    a = w.acoustic
    a_mag = np.abs(np.fft.rfft(a)) * 2 / N_HIGH

    i_spec = np.abs(np.fft.rfft(w.current)) * 2 / N_ELEC
    fund = i_spec[_ELEC_BIN[1]]
    thd = np.sqrt(i_spec[_ELEC_BIN[3]] ** 2 + i_spec[_ELEC_BIN[5]] ** 2) / (fund + 1e-12)

    feats = [
        w.load,
        v_rms, kurt, crest, *bands,
        np.sqrt(np.mean(a**2)), _log_energy(a_mag, _AC_HF),
        np.sqrt(np.mean(w.current**2)), thd,
        w.voltage.mean(), w.voltage.std(),
        w.temperature.mean(),
        w.pressure.mean(), w.pressure.std(),
        w.oil_debris.mean(), w.oil_debris.max(),
        w.ambient_c, w.altitude_ft / 1000.0,
    ]
    return np.asarray(feats, dtype=np.float64)


def trend_features(x_hist: np.ndarray, s_hist: np.ndarray) -> np.ndarray:
    """Summarise recent history: mean features, then [mean, slope, last] of the anomaly score."""
    n = len(s_hist)
    slope = float(np.polyfit(np.arange(n), s_hist, 1)[0]) if n >= 3 else 0.0
    return np.concatenate([x_hist.mean(axis=0), [s_hist.mean(), slope, s_hist[-1]]])


N_TREND_FEATURES = N_FEATURES + 3


def trend_matrix(X: np.ndarray, scores: np.ndarray, k: int = TREND_WINDOW) -> np.ndarray:
    """Offline equivalent of running ``TrendTracker`` over a whole run."""
    return np.stack(
        [trend_features(X[max(0, t - k + 1) : t + 1], scores[max(0, t - k + 1) : t + 1]) for t in range(len(X))]
    )


def sequence_matrix(X: np.ndarray, scores: np.ndarray, length: int) -> np.ndarray:
    """(T, length, F+1) windows of [features, anomaly score] ending at each step.

    Steps before the start of the run are filled with the first window (left padding), so
    early windows still give a full-length sequence. Offline twin of ``SequenceTracker``.
    """
    Z = np.column_stack([X, scores])
    idx = np.arange(len(Z))[:, None] + np.arange(-length + 1, 1)[None, :]
    return Z[np.maximum(idx, 0)]


class SequenceTracker:
    """Streaming version of ``sequence_matrix``: identical output for identical history."""

    def __init__(self, length: int):
        self._z: deque[np.ndarray] = deque(maxlen=length)
        self.length = length

    def update(self, x: np.ndarray, score: float) -> np.ndarray:
        self._z.append(np.append(x, score))
        pad = [self._z[0]] * (self.length - len(self._z))
        return np.stack(pad + list(self._z))


class TrendTracker:
    """Streaming version of ``trend_matrix``: identical output for identical history."""

    def __init__(self, k: int = TREND_WINDOW):
        self._x: deque[np.ndarray] = deque(maxlen=k)
        self._s: deque[float] = deque(maxlen=k)

    def update(self, x: np.ndarray, score: float) -> np.ndarray:
        self._x.append(x)
        self._s.append(score)
        return trend_features(np.stack(self._x), np.asarray(self._s))
