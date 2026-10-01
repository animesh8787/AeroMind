"""Remaining Useful Life with an uncertainty band (quantile gradient boosting)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

QUANTILES = (0.1, 0.5, 0.9)


class RULEstimator:
    """Predicts RUL in flight hours as (p10, p50, p90) from trend features."""

    def __init__(self, seed: int = 0):
        self.models = {
            q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150, random_state=seed)
            for q in QUANTILES
        }

    def fit(self, X: np.ndarray, rul_hours: np.ndarray) -> "RULEstimator":
        for m in self.models.values():
            m.fit(X, rul_hours)
        return self

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """(N, 3) array of monotone p10 <= p50 <= p90 in hours."""
        out = np.column_stack([self.models[q].predict(X) for q in QUANTILES])
        return np.maximum(np.sort(out, axis=1), 0.0)

    def predict(self, x: np.ndarray) -> tuple[float, float, float]:
        p10, p50, p90 = self.predict_batch(np.atleast_2d(x))[0]
        return float(p10), float(p50), float(p90)
