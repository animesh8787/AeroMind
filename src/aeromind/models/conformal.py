"""Conformalized quantile regression (CQR) for the RUL interval.

Wraps a fitted RUL estimator (``RULEstimator`` or ``LSTMRULEstimator``). On a calibration set
held out from training, the conformity score is ``max(p10 - y, y - p90)``; its
``ceil((n + 1)(1 - alpha)) / n`` empirical quantile ``q`` widens (q > 0) or narrows (q < 0) the
p10-p90 interval to ``[p10 - q, p90 + q]``. p50 is unchanged.

Guarantee and its limits: marginal coverage >= 1 - alpha holds when calibration and test examples
are exchangeable. Windows from the same run are correlated and test conditions can differ from
calibration (e.g. the official C-MAPSS test truncation), so treat the result as a calibration
correction to be measured, not a guarantee.
"""

from __future__ import annotations

import numpy as np


class ConformalRUL:
    def __init__(self, base, alpha: float = 0.2):
        self.base, self.alpha = base, alpha
        self.q: float | None = None
        self.n_cal = 0

    # expose the base model's input contract (pipeline, ONNX export)
    def __getattr__(self, name):
        if name in ("base", "alpha", "q", "n_cal"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalRUL":
        P = self.base.predict_batch(X_cal)
        scores = np.maximum(P[:, 0] - y_cal, y_cal - P[:, 2])
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q, self.n_cal = float(np.quantile(scores, level, method="higher")), n
        return self

    def adjust(self, P: np.ndarray) -> np.ndarray:
        out = P.copy()
        out[:, 0] -= self.q
        out[:, 2] += self.q
        out = np.sort(out, axis=1)  # a negative q can cross p50
        return np.maximum(out, 0.0)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.adjust(self.base.predict_batch(X))

    def predict(self, x: np.ndarray) -> tuple[float, float, float]:
        p10, p50, p90 = self.predict_batch(np.asarray(x)[None])[0]
        return float(p10), float(p50), float(p90)
