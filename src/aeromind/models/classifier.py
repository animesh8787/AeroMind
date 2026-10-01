"""Fault-mode classification (gradient-boosted trees; XGBoost-class model, no extra dependency)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

CONTEXT = ("load", "ambient_c", "altitude_kft")


class ContextResidual:
    """Each feature's deviation from its healthy value at the current operating context.

    A linear model per feature on [1, load, OAT, altitude, load*OAT], fitted on healthy windows.
    Output: residuals of all features followed by the three context values. Under flight phases the
    outside air temperature and load move temperature and oil readings as much as an early fault does;
    residuals remove that, so the classifier sees the fault signature.
    """

    def __init__(self, feature_names):
        self.idx = [list(feature_names).index(n) for n in CONTEXT]
        self.W: np.ndarray | None = None  # (5, n_features)

    def design(self, X: np.ndarray) -> np.ndarray:
        c = X[:, self.idx]
        return np.column_stack([np.ones(len(X)), c, c[:, 0] * c[:, 1]])

    def fit(self, X_healthy: np.ndarray) -> "ContextResidual":
        self.W = np.linalg.lstsq(self.design(X_healthy), X_healthy, rcond=None)[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.column_stack([X - self.design(X) @ self.W, X[:, self.idx]])


class FaultClassifier:
    def __init__(self, seed: int = 0, context: ContextResidual | None = None):
        self.model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)
        self.context = context

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.context.transform(X) if getattr(self, "context", None) is not None else X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FaultClassifier":
        self.model.fit(self._inputs(X), y)
        return self

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        """Return (label, probability of that label) for one feature vector."""
        proba = self.model.predict_proba(self._inputs(x))[0]
        i = int(np.argmax(proba))
        return str(self.model.classes_[i]), float(proba[i])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._inputs(X))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._inputs(X))
