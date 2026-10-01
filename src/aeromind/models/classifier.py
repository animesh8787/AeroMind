"""Fault-mode classification (gradient-boosted trees; XGBoost-class model, no extra dependency)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class FaultClassifier:
    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FaultClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        """Return (label, probability of that label) for one feature vector."""
        proba = self.model.predict_proba(np.atleast_2d(x))[0]
        i = int(np.argmax(proba))
        return str(self.model.classes_[i]), float(proba[i])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
