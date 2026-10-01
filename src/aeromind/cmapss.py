"""RUL prediction on the NASA C-MAPSS turbofan degradation benchmark.

Reuses the AeroMind components (calibrated anomaly score, trend features,
quantile RUL regression) on real benchmark data. The data are *simulated*
turbofan run-to-failure trajectories published by NASA, not flight data.

Reference: A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation
Data Set", NASA Prognostics Data Repository, NASA Ames Research Center.

Protocol (fixed in advance, not tuned on the test set):
  * 14 commonly used informative sensors (1-indexed: 2,3,4,7,8,9,11,12,13,14,15,17,20,21)
  * per-operating-regime standardisation (KMeans on the 3 settings, k=6 for FD002/FD004)
  * anomaly detector fitted on the first 20% of life of 75% of training engines,
    calibrated on the same early-life slice of the other 25%
  * trend window of 30 cycles; RUL target capped at 125 cycles
  * evaluation on the official test split: one prediction per engine at its last cycle
  * RUL model: quantile gradient boosting on trend features ("hgb"), or a PyTorch LSTM over the
    last 30 cycles of [sensors, anomaly score] ("lstm"); the LSTM's size was chosen on held-out
    training engines only
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .features import sequence_matrix, trend_matrix
from .models import AnomalyDetector, LSTMRULEstimator, RULEstimator

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]  # 1-indexed
RUL_CAP = 125
TREND_K = 30
HEALTHY_FRACTION = 0.2
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
LSTM_HIDDEN = 64  # lowest mean RMSE on held-out training engines (FD001, FD004) of 16/32/64; test split not used


def download(data_dir: str | Path) -> None:
    """Fetch the official NASA archive into ``data_dir`` (skipped if already present)."""
    d = Path(data_dir)
    if (d / "train_FD001.txt").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    inner.extractall(d)


@dataclass
class Engines:
    """Per-engine arrays. ``settings`` is (T,3); ``sensors`` is (T,14)."""

    settings: list[np.ndarray]
    sensors: list[np.ndarray]


def _load(path: Path) -> Engines:
    raw = np.loadtxt(path)
    units = raw[:, 0].astype(int)
    settings, sensors = [], []
    for u in np.unique(units):
        rows = raw[units == u]
        settings.append(rows[:, 2:5])
        sensors.append(rows[:, [4 + s for s in SENSORS]])
    return Engines(settings, sensors)


def load_subset(data_dir: str | Path, subset: str) -> tuple[Engines, Engines, np.ndarray]:
    d = Path(data_dir)
    train = _load(d / f"train_{subset}.txt")
    test = _load(d / f"test_{subset}.txt")
    rul = np.loadtxt(d / f"RUL_{subset}.txt")
    return train, test, rul


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Asymmetric PHM08 score: late predictions (pred > true) are penalised more. Lower is better."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


class RegimeNormaliser:
    def __init__(self, k: int, seed: int = 0):
        self.k, self.seed = k, seed

    def fit(self, eng: Engines) -> "RegimeNormaliser":
        S, X = np.vstack(eng.settings), np.vstack(eng.sensors)
        self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(S) if self.k > 1 else None
        labels = self._labels(S)
        self.mean = np.stack([X[labels == r].mean(0) for r in range(self.k)])
        self.std = np.stack([X[labels == r].std(0) for r in range(self.k)]) + 1e-6
        return self

    def _labels(self, S: np.ndarray) -> np.ndarray:
        return self.km.predict(S) if self.km is not None else np.zeros(len(S), dtype=int)

    def transform(self, settings: np.ndarray, sensors: np.ndarray) -> np.ndarray:
        r = self._labels(settings)
        return (sensors - self.mean[r]) / self.std[r]


class CmapssRUL:
    """AeroMind pipeline for run-to-failure RUL on C-MAPSS."""

    def __init__(self, subset: str, seed: int = 0, rul: str = "hgb", lstm_hidden: int = LSTM_HIDDEN,
                 conformal: bool = False):
        if rul not in ("hgb", "lstm"):
            raise ValueError(f"unknown rul model {rul!r}")
        self.subset, self.seed, self.rul_kind, self.lstm_hidden = subset, seed, rul, lstm_hidden
        self.conformal = conformal

    def _normalise(self, eng: Engines) -> list[np.ndarray]:
        return [self.norm.transform(s, x) for s, x in zip(eng.settings, eng.sensors)]

    def _inputs(self, Xs: list[np.ndarray]) -> list[np.ndarray]:
        """Per engine: trend rows (hgb) or sequences of [sensors, anomaly score] (lstm)."""
        build = sequence_matrix if self.rul_kind == "lstm" else trend_matrix
        return [build(X, self.anomaly.score(X), TREND_K) for X in Xs]

    def _lstm(self) -> LSTMRULEstimator:
        return LSTMRULEstimator(seq_len=TREND_K, hidden=self.lstm_hidden, seed=self.seed)

    def fit(self, train: Engines) -> "CmapssRUL":
        cal = None
        if self.conformal:
            # Hold out 20% of engines; calibrate at 10 random truncation points per engine, which
            # matches how the official test split cuts trajectories (all-cycle calibration does not).
            rng = np.random.default_rng(self.seed + 7)
            idx = rng.permutation(len(train.sensors))
            k = max(2, len(idx) // 5)
            pick = lambda ids: Engines([train.settings[i] for i in ids], [train.sensors[i] for i in ids])  # noqa: E731
            cal, train = pick(idx[:k]), pick(idx[k:])
        self.norm = RegimeNormaliser(N_REGIMES[self.subset], self.seed).fit(train)
        Xs = self._normalise(train)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(Xs))
        n_cal = max(1, len(Xs) // 4)
        early = lambda X: X[: max(10, int(HEALTHY_FRACTION * len(X)))]
        X_fit = np.vstack([early(Xs[i]) for i in order[n_cal:]])
        X_cal = np.vstack([early(Xs[i]) for i in order[:n_cal]])
        self.anomaly = AnomalyDetector(seed=self.seed).fit(X_fit, X_cal)

        T = np.concatenate(self._inputs(Xs))
        y = np.concatenate([np.minimum(np.arange(len(X), 0, -1), RUL_CAP) for X in Xs]).astype(float)
        self.n_sensor_feats = len(SENSORS)
        # Ablation: same model without the anomaly-score inputs (sensors only).
        if self.rul_kind == "lstm":
            groups = np.concatenate([np.full(len(X), k) for k, X in enumerate(Xs)])
            self.rul = self._lstm().fit(T, y, groups)
            self.rul_no_anomaly = self._lstm().fit(T[:, :, : self.n_sensor_feats], y, groups)
        else:
            self.rul = RULEstimator(seed=self.seed).fit(T, y)
            self.rul_no_anomaly = RULEstimator(seed=self.seed).fit(T[:, : self.n_sensor_feats], y)
        self.train_median = float(np.median(y))
        if cal is not None:
            from .models.conformal import ConformalRUL

            rng = np.random.default_rng(self.seed + 8)
            Tc, yc = [], []
            for T, x in zip(self._inputs(self._normalise(cal)), cal.sensors):
                n = len(x)
                for cut in rng.integers(max(TREND_K + 1, n // 4), n, 10):
                    Tc.append(T[cut - 1])
                    yc.append(min(n - cut + 1, RUL_CAP))
            self.rul = ConformalRUL(self.rul).calibrate(np.array(Tc), np.array(yc, dtype=float))
        return self

    def predict_last(self, test: Engines) -> dict[str, np.ndarray]:
        """Prediction for each test engine at its last observed cycle."""
        last = np.stack([t[-1] for t in self._inputs(self._normalise(test))])
        sensors_only = last[..., : self.n_sensor_feats]
        return {
            "aeromind": self.rul.predict_batch(last),  # (N,3): p10, p50, p90
            "no_anomaly_features": self.rul_no_anomaly.predict_batch(sensors_only),
            "constant": np.full(len(last), self.train_median),
        }


def _metrics(pred: np.ndarray, true_capped: np.ndarray, true_raw: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean((pred - true_capped) ** 2))),
        "mae": float(np.mean(np.abs(pred - true_capped))),
        "rmse_vs_uncapped_truth": float(np.sqrt(np.mean((pred - true_raw) ** 2))),
        "nasa_score": nasa_score(pred, true_capped),
    }


def run_subset(data_dir: str | Path, subset: str, seed: int = 0, rul: str = "hgb", conformal: bool = False) -> dict:
    train, test, rul_true = load_subset(data_dir, subset)
    model = CmapssRUL(subset, seed, rul, conformal=conformal).fit(train)
    preds = model.predict_last(test)
    capped = np.minimum(rul_true, RUL_CAP)
    q = preds["aeromind"]
    return {
        "aeromind": {
            **_metrics(q[:, 1], capped, rul_true),
            "p10_p90_coverage": float(np.mean((q[:, 0] <= capped) & (capped <= q[:, 2]))),
        },
        "no_anomaly_features": _metrics(preds["no_anomaly_features"][:, 1], capped, rul_true),
        "constant_baseline": _metrics(preds["constant"], capped, rul_true),
        "n_train_engines": len(train.sensors),
        "n_test_engines": len(test.sensors),
    }
