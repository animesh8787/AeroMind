"""Remaining Useful Life from a sequence of windows with a small PyTorch LSTM.

Input: the last ``seq_len`` windows of [features, anomaly score] (see ``features.sequence_matrix``).
Output: (p10, p50, p90) RUL in hours, ordered by construction (p50 and p90 add non-negative gaps).
Trained with the pinball (quantile) loss; early stopping on held-out runs, never held-out windows
of a run that is also in training.

The fitted weights are stored as numpy arrays, so a pickled estimator loads without PyTorch;
PyTorch is needed only to fit, predict or export.
"""

from __future__ import annotations

import numpy as np

from .rul import QUANTILES


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError('the LSTM RUL model needs PyTorch: pip install -e ".[lstm]"') from e
    return torch


def _net_class():
    torch = _torch()
    nn = torch.nn

    class LSTMQuantileNet(nn.Module):
        def __init__(self, n_in: int, hidden: int, layers: int, dropout: float = 0.0):
            super().__init__()
            self.register_buffer("mean", torch.zeros(n_in))
            self.register_buffer("std", torch.ones(n_in))
            self.register_buffer("y_scale", torch.ones(1))
            self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, len(QUANTILES))

        def raw(self, seq):
            """Ordered quantiles in units of ``y_scale``, before clipping at zero."""
            out, _ = self.lstm((seq - self.mean) / self.std)
            a = self.head(self.drop(out[:, -1, :]))
            gaps = nn.functional.softplus(a[:, 1:])
            p10 = a[:, :1]
            p50 = p10 + gaps[:, :1]
            return torch.cat([p10, p50, p50 + gaps[:, 1:]], dim=1)

        def forward(self, seq):
            """(N, L, F) -> (N, 3) RUL hours, p10 <= p50 <= p90, >= 0."""
            return torch.relu(self.raw(seq)) * self.y_scale

    return LSTMQuantileNet


class LSTMRULEstimator:
    """Drop-in for ``RULEstimator`` that reads a window sequence instead of trend features."""

    input_kind = "sequence"

    def __init__(self, seq_len: int = 30, hidden: int = 16, layers: int = 1, dropout: float = 0.2,
                 weight_decay: float = 1e-4, epochs: int = 60, batch_size: int = 256, lr: float = 1e-3,
                 patience: int = 8, val_fraction: float = 0.15, seed: int = 0):
        self.seq_len, self.hidden, self.layers, self.dropout = seq_len, hidden, layers, dropout
        self.weight_decay = weight_decay
        self.epochs, self.batch_size, self.lr, self.patience = epochs, batch_size, lr, patience
        self.val_fraction, self.seed = val_fraction, seed
        self.n_in: int | None = None
        self.state: dict[str, np.ndarray] | None = None
        self.history: list[tuple[float, float]] = []  # (train, validation) pinball loss per epoch
        self._net = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_net"] = None
        return d

    def network(self):
        """The fitted network (eval mode, float32), for prediction or ONNX export."""
        torch = _torch()

        if self._net is None:
            net = _net_class()(self.n_in, self.hidden, self.layers, self.dropout)
            net.load_state_dict({k: torch.from_numpy(v) for k, v in self.state.items()})
            self._net = net.eval()
        return self._net

    def fit(self, S: np.ndarray, rul_hours: np.ndarray, groups: np.ndarray) -> "LSTMRULEstimator":
        """``S`` (N, seq_len, F) sequences, ``rul_hours`` (N,) targets, ``groups`` (N,) run ids."""
        torch = _torch()

        if S.shape[1] != self.seq_len:
            raise ValueError(f"expected sequences of length {self.seq_len}, got {S.shape[1]}")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        S = np.asarray(S, dtype=np.float32)
        y = np.asarray(rul_hours, dtype=np.float32)
        self.n_in = S.shape[2]

        ids = np.unique(groups)
        n_val = max(1, int(round(self.val_fraction * len(ids))))
        val = np.isin(groups, rng.choice(ids, n_val, replace=False))
        if val.all() or not val.any():
            raise ValueError("need at least two groups to hold some out for early stopping")

        net = _net_class()(self.n_in, self.hidden, self.layers, self.dropout)
        flat = S[~val].reshape(-1, self.n_in)
        net.mean.copy_(torch.from_numpy(flat.mean(0)))
        net.std.copy_(torch.from_numpy(np.maximum(flat.std(0), 1e-6)))
        net.y_scale.fill_(float(max(y.max(), 1.0)))

        q = torch.tensor(QUANTILES, dtype=torch.float32)

        def pinball(pred, target):
            e = target[:, None] - pred
            return torch.maximum(q * e, (q - 1) * e).mean()

        Xt, yt = torch.from_numpy(S[~val]), torch.from_numpy(y[~val]) / net.y_scale
        Xv, yv = torch.from_numpy(S[val]), torch.from_numpy(y[val]) / net.y_scale
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, stale = float("inf"), None, 0
        self.history = []
        for _ in range(self.epochs):
            net.train()
            order = torch.from_numpy(rng.permutation(len(Xt)))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                b = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = pinball(net.raw(Xt[b]), yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(b)
            net.eval()
            with torch.no_grad():
                v = float(pinball(net.raw(Xv), yv))
            self.history.append((total / len(Xt), v))
            if v < best - 1e-5:
                best, stale = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.state = {k: t.numpy() for k, t in best_state.items()}
        self._net = None
        return self

    def predict_batch(self, S: np.ndarray) -> np.ndarray:
        """(N, 3) array of p10 <= p50 <= p90 in hours."""
        torch = _torch()

        with torch.no_grad():
            out = self.network()(torch.from_numpy(np.asarray(S, dtype=np.float32).reshape(-1, self.seq_len, self.n_in)))
        return out.numpy().astype(np.float64)

    def predict(self, s: np.ndarray) -> tuple[float, float, float]:
        p10, p50, p90 = self.predict_batch(s[None])[0]
        return float(p10), float(p50), float(p90)
