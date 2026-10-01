"""Offline training on simulated run-to-failure data. Produces a ``ModelBundle``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .config import FAULT_MODES, HEALTHY, HOURS_PER_WINDOW, RUL_CAP_WINDOWS, TREND_WINDOW
from .features import N_FEATURES, extract_features, trend_matrix
from .models import AnomalyDetector, FaultClassifier, RULEstimator
from .simulator import simulate_run


@dataclass
class RunData:
    mode: str
    X: np.ndarray  # (T, N_FEATURES)
    degradation: np.ndarray  # (T,)
    rul_windows: np.ndarray  # (T,), NaN for healthy


@dataclass
class ModelBundle:
    anomaly: AnomalyDetector
    classifier: FaultClassifier
    rul: RULEstimator

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        return joblib.load(path)


@dataclass
class TrainConfig:
    healthy_runs: int = 20
    healthy_len: int = 250
    runs_per_mode: int = 12
    life_range: tuple[int, int] = (250, 450)
    seed: int = 0


def collect_run(mode: str, life: int, seed: int) -> RunData:
    X, d, rul = [], [], []
    for w, truth in simulate_run(mode, life, seed):
        X.append(extract_features(w))
        d.append(truth.degradation)
        rul.append(np.nan if truth.rul_windows is None else truth.rul_windows)
    return RunData(mode, np.stack(X), np.asarray(d), np.asarray(rul, dtype=float))


def collect_dataset(cfg: TrainConfig) -> list[RunData]:
    rng = np.random.default_rng(cfg.seed)
    runs = []
    for i in range(cfg.healthy_runs):
        runs.append(collect_run(HEALTHY, cfg.healthy_len, cfg.seed * 100_003 + i))
    for m, mode in enumerate(FAULT_MODES):
        for i in range(cfg.runs_per_mode):
            life = int(rng.integers(*cfg.life_range))
            runs.append(collect_run(mode, life, cfg.seed * 100_003 + 1000 * (m + 1) + i))
    return runs


def train(cfg: TrainConfig | None = None, runs: list[RunData] | None = None) -> ModelBundle:
    cfg = cfg or TrainConfig()
    runs = runs if runs is not None else collect_dataset(cfg)
    healthy = [r for r in runs if r.mode == HEALTHY]
    faulted = [r for r in runs if r.mode != HEALTHY]

    # 1. Anomaly detector: healthy runs only; the last quarter of the runs calibrates the score.
    n_cal = max(1, len(healthy) // 4)
    X_fit = np.vstack([r.X for r in healthy[:-n_cal]])
    X_cal = np.vstack([r.X for r in healthy[-n_cal:]])
    anomaly = AnomalyDetector(seed=cfg.seed).fit(X_fit, X_cal)

    # 2. Trend features per run, exactly as the streaming pipeline computes them.
    trends = {id(r): trend_matrix(r.X, anomaly.score(r.X), TREND_WINDOW) for r in runs}

    # 3. Fault classifier: healthy windows plus faulted windows past 30% degradation.
    Xc, yc = [], []
    for r in healthy:
        Xc.append(trends[id(r)][:, :N_FEATURES])
        yc += [HEALTHY] * len(r.X)
    for r in faulted:
        m = r.degradation >= 0.3
        Xc.append(trends[id(r)][m, :N_FEATURES])
        yc += [r.mode] * int(m.sum())
    classifier = FaultClassifier(seed=cfg.seed).fit(np.vstack(Xc), np.asarray(yc))

    # 4. RUL: faulted windows the deployed gate could plausibly reach (elevated anomaly score).
    Xr = np.vstack([trends[id(r)] for r in faulted])
    yr = np.concatenate([np.minimum(r.rul_windows, RUL_CAP_WINDOWS) * HOURS_PER_WINDOW for r in faulted])
    gated = Xr[:, N_FEATURES] >= 0.8  # mean anomaly score over the trend window
    if gated.sum() >= 200:
        Xr, yr = Xr[gated], yr[gated]
    rul = RULEstimator(seed=cfg.seed).fit(Xr, yr)

    return ModelBundle(anomaly, classifier, rul)
