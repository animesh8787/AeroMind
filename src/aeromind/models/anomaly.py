"""Unsupervised anomaly detection, trained on healthy data only.

Two complementary detectors are fused: an Isolation Forest and a small
bottleneck autoencoder (reconstruction error). Scores are calibrated on held-out
healthy data so that **1.0 is approximately the 99th percentile of healthy windows**.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..features import FEATURE_NAMES


class AutoencoderModel:
    """Tiny tanh bottleneck autoencoder over standardised features."""

    def __init__(self, hidden=(12, 4, 12), seed: int = 0, max_iter: int = 400):
        self.net = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="tanh",
            alpha=1e-3,
            learning_rate_init=3e-3,
            max_iter=max_iter,
            random_state=seed,
        )

    def fit(self, Xs: np.ndarray) -> "AutoencoderModel":
        self.net.fit(Xs, Xs)
        return self

    def recon_error(self, Xs: np.ndarray) -> np.ndarray:
        return np.mean((self.net.predict(Xs) - Xs) ** 2, axis=1)


class AnomalyDetector:
    def __init__(self, seed: int = 0, n_trees: int = 60):
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(n_estimators=n_trees, random_state=seed)
        self.autoencoder = AutoencoderModel(seed=seed)
        self._norm1: tuple[np.ndarray, np.ndarray] | None = None
        self._norm2: tuple[float, float] | None = None

    def _raw(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return np.column_stack([-self.iforest.score_samples(Xs), np.log(self.autoencoder.recon_error(Xs) + 1e-9)])

    @staticmethod
    def _norm_params(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        med, p99 = np.median(v, axis=0), np.percentile(v, 99, axis=0)
        return med, np.maximum(p99 - med, 1e-9)

    def fit(self, X_fit: np.ndarray, X_cal: np.ndarray) -> "AnomalyDetector":
        """``X_fit`` trains the models; ``X_cal`` (also healthy, disjoint) calibrates the score."""
        Xs = self.scaler.fit_transform(X_fit)
        self.iforest.fit(Xs)
        self.autoencoder.fit(Xs)
        raw = self._raw(X_cal)
        self._norm1 = self._norm_params(raw)
        combined = ((raw - self._norm1[0]) / self._norm1[1]).mean(axis=1)
        med, scale = self._norm_params(combined[:, None])
        self._norm2 = (float(med[0]), float(scale[0]))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score per row. ~0 is typical healthy; >1 is beyond 99% of healthy windows."""
        X = np.atleast_2d(X)
        raw = self._raw(X)
        combined = ((raw - self._norm1[0]) / self._norm1[1]).mean(axis=1)
        return (combined - self._norm2[0]) / self._norm2[1]

    def explain_z(self, x: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """The ``k`` features furthest from their healthy mean, with signed z-scores."""
        z = self.scaler.transform(np.atleast_2d(x))[0]
        return [(FEATURE_NAMES[i], float(z[i])) for i in np.argsort(np.abs(z))[::-1][:k]]

    def explain(self, x: np.ndarray, k: int = 3) -> list[str]:
        """Names of the ``k`` features furthest from their healthy mean (in std devs)."""
        z = np.abs(self.scaler.transform(np.atleast_2d(x))[0])
        return [FEATURE_NAMES[i] for i in np.argsort(z)[::-1][:k]]
