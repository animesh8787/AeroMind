"""Federated averaging (FedAvg) of the autoencoder across simulated aircraft.

Each aircraft keeps its raw healthy data local; only scaler statistics and
network weights are shared. Demonstration of the *mechanism*: the "fleet" is a
handful of simulated tails and there is no secure aggregation or transport here.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..core.config import FAULT_MODES, HEALTHY
from ..core.features import extract_features
from ..models.anomaly import AutoencoderModel
from ..core.simulator import TailProfile, simulate_run


def fedavg(client_weights: Sequence[Sequence[np.ndarray]], counts: Sequence[float]) -> list[np.ndarray]:
    """Sample-count-weighted average of per-client weight lists."""
    w = np.asarray(counts, dtype=float)
    w = w / w.sum()
    return [sum(wi * layer for wi, layer in zip(w, layers)) for layers in zip(*client_weights)]


def federated_scaler(stats: Sequence[tuple[int, np.ndarray, np.ndarray]]) -> StandardScaler:
    """Combine per-client (n, mean, var) into a global StandardScaler without pooling data."""
    n = np.array([s[0] for s in stats], dtype=float)
    mean = sum(ni * s[1] for ni, s in zip(n, stats)) / n.sum()
    var = sum(ni * (s[2] + (s[1] - mean) ** 2) for ni, s in zip(n, stats)) / n.sum()
    sc = StandardScaler()
    sc.mean_, sc.var_ = mean, var
    sc.scale_ = np.sqrt(np.maximum(var, 1e-12))
    sc.n_features_in_ = len(mean)
    sc.n_samples_seen_ = int(n.sum())
    return sc


class FederatedAutoencoder:
    def __init__(self, rounds: int = 10, local_epochs: int = 5, seed: int = 0):
        self.rounds, self.local_epochs, self.seed = rounds, local_epochs, seed
        self.scaler: StandardScaler | None = None
        self.model = AutoencoderModel(seed=seed)

    def fit(self, client_data: Sequence[np.ndarray]) -> "FederatedAutoencoder":
        self.scaler = federated_scaler([(len(X), X.mean(0), X.var(0)) for X in client_data])
        data = [self.scaler.transform(X) for X in client_data]
        net = self.model.net
        net.partial_fit(data[0][:64], data[0][:64])  # allocate weights / optimiser state
        for _ in range(self.rounds):
            updates = []
            for Xs in data:
                local = copy.deepcopy(net)
                for _ in range(self.local_epochs):
                    local.partial_fit(Xs, Xs)
                updates.append(local.coefs_ + local.intercepts_)
            merged = fedavg(updates, [len(X) for X in data])
            for param, new in zip(net.coefs_ + net.intercepts_, merged):
                param[...] = new  # in place so the optimiser keeps its references
        return self

    def recon_error(self, X: np.ndarray) -> np.ndarray:
        return self.model.recon_error(self.scaler.transform(X))


def _local_only(X: np.ndarray, epochs: int, seed: int) -> tuple[StandardScaler, AutoencoderModel]:
    sc = StandardScaler().fit(X)
    ae = AutoencoderModel(seed=seed)
    Xs = sc.transform(X)
    ae.net.partial_fit(Xs[:64], Xs[:64])
    for _ in range(epochs):
        ae.net.partial_fit(Xs, Xs)
    return sc, ae


def _features(
    mode: str, life: int, seed: int, tail: TailProfile, d_range: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    rows = [
        extract_features(w)
        for w, truth in simulate_run(mode, life, seed, tail)
        if d_range[0] <= truth.degradation <= d_range[1]
    ]
    return np.stack(rows)


def federated_demo(
    n_clients: int = 5, seed: int = 0, rounds: int = 10, local_epochs: int = 5, spread: float = 2.0,
    cold_start_windows: int = 40,
) -> dict:
    """Compare a local-only autoencoder with a federated one on an *unseen* aircraft.

    Metric: AUROC separating healthy windows from windows with 0.1 <= degradation <= 0.3 (early faults),
    by reconstruction error, on a tail neither model trained on. ``spread`` scales how
    different the aircraft's sensor baselines are from one another.
    """
    rng = np.random.default_rng(seed)
    clients = [_features(HEALTHY, 250, seed * 7919 + i, TailProfile.random(rng, spread)) for i in range(n_clients)]

    new_tail = TailProfile.random(rng, spread)
    healthy_eval = _features(HEALTHY, 250, seed + 555_000, new_tail)
    faulted_eval = np.vstack(
        [_features(mode, 300, seed + 556_000 + i, new_tail, (0.1, 0.3)) for i, mode in enumerate(FAULT_MODES)]
    )
    X_eval = np.vstack([healthy_eval, faulted_eval])
    y_eval = np.r_[np.zeros(len(healthy_eval)), np.ones(len(faulted_eval))]

    def auroc_local(X_train: np.ndarray) -> float:
        sc, ae = _local_only(X_train, epochs=rounds * local_epochs, seed=seed)
        return round(float(roc_auc_score(y_eval, ae.recon_error(sc.transform(X_eval)))), 3)

    fed = FederatedAutoencoder(rounds=rounds, local_epochs=local_epochs, seed=seed).fit(clients)
    return {
        "clients": n_clients,
        # Global model; saw nothing from the new aircraft.
        "auroc_federated": round(float(roc_auc_score(y_eval, fed.recon_error(X_eval))), 3),
        # Trained on one other aircraft only.
        "auroc_local_one_other_tail": auroc_local(clients[0]),
        # Cold start: the new aircraft's own first `cold_start_windows` healthy windows only.
        "auroc_local_cold_start": auroc_local(healthy_eval[:cold_start_windows]),
    }


# --------------------------------------------------------------------------- rare-fault sharing


def _labelled(mode: str, life: int, seed: int, tail: TailProfile, min_d: float = 0.3) -> tuple[np.ndarray, list]:
    X, y = [], []
    for w, truth in simulate_run(mode, life, seed, tail):
        if mode == HEALTHY or truth.degradation >= min_d:
            X.append(extract_features(w))
            y.append(mode)
    return np.stack(X), y


def rare_fault_demo(n_clients: int = 5, seed: int = 0, rounds: int = 15, local_epochs: int = 3,
                    faults_per_client: int = 2) -> dict:
    """Can an aircraft recognise a fault type it has never experienced, if the fleet has?

    Non-IID fleet: aircraft k has seen healthy operation and only ``faults_per_client`` fault types.
    Three MLP classifiers per aircraft, tested on fresh runs of ALL fault types on that aircraft:
      local        trained on its own data only (raw data never leaves, but no fleet knowledge);
      federated    FedAvg of MLP weights across the fleet (only weights and scaler statistics shared);
      centralised  trained on everyone's pooled raw data (upper bound; not privacy preserving).
    Reported: accuracy on fault types the aircraft had seen vs never seen, and healthy windows wrongly
    called faulty.
    """
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(seed)
    classes = np.array([HEALTHY, *FAULT_MODES])
    tails = [TailProfile.random(rng, 1.0) for _ in range(n_clients)]
    seen = [[FAULT_MODES[(k + j) % len(FAULT_MODES)] for j in range(faults_per_client)] for k in range(n_clients)]
    train, test = [], []
    for k, tail in enumerate(tails):
        parts = [_labelled(HEALTHY, 250, seed * 7919 + 100 * k, tail)]
        parts += [_labelled(m, 300, seed * 7919 + 100 * k + 1 + i, tail) for i, m in enumerate(seen[k])]
        train.append((np.vstack([p[0] for p in parts]), sum((p[1] for p in parts), [])))
        tparts = [_labelled(HEALTHY, 200, seed + 900_000 + 100 * k, tail)]
        tparts += [_labelled(m, 300, seed + 900_001 + 100 * k + i, tail) for i, m in enumerate(FAULT_MODES)]
        test.append((np.vstack([p[0] for p in tparts]), np.array(sum((p[1] for p in tparts), []))))

    def mlp():
        return MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, learning_rate_init=3e-3, random_state=seed)

    def fit_epochs(model, Xs, y, epochs):
        for _ in range(epochs):
            model.partial_fit(Xs, y, classes=classes)
        return model

    epochs = rounds * local_epochs
    local = []
    for X, y in train:
        sc = StandardScaler().fit(X)
        local.append((sc, fit_epochs(mlp(), sc.transform(X), y, epochs)))

    fed_sc = federated_scaler([(len(X), X.mean(0), X.var(0)) for X, _ in train])
    data = [(fed_sc.transform(X), y) for X, y in train]
    glob = mlp()
    glob.partial_fit(data[0][0][:16], data[0][1][:16], classes=classes)
    for _ in range(rounds):
        updates = []
        for Xs, y in data:
            m = copy.deepcopy(glob)
            fit_epochs(m, Xs, y, local_epochs)
            updates.append(m.coefs_ + m.intercepts_)
        merged = fedavg(updates, [len(y) for _, y in data])
        for param, new in zip(glob.coefs_ + glob.intercepts_, merged):
            param[...] = new

    Xall = np.vstack([X for X, _ in train])
    yall = sum((y for _, y in train), [])
    cen_sc = StandardScaler().fit(Xall)
    central = fit_epochs(mlp(), cen_sc.transform(Xall), yall, epochs)

    def scores(predict):
        seen_acc, unseen_acc, false_fault = [], [], []
        for k, (X, y) in enumerate(test):
            p = predict(k, X)
            fault = y != HEALTHY
            s = np.isin(y, seen[k]) & fault
            u = ~np.isin(y, seen[k]) & fault
            seen_acc.append(np.mean(p[s] == y[s]))
            unseen_acc.append(np.mean(p[u] == y[u]))
            false_fault.append(np.mean(p[~fault] != HEALTHY))
        return {"seen_fault_accuracy": round(float(np.mean(seen_acc)), 3),
                "unseen_fault_accuracy": round(float(np.mean(unseen_acc)), 3),
                "healthy_called_faulty": round(float(np.mean(false_fault)), 3)}

    return {
        "clients": n_clients, "faults_seen_per_client": faults_per_client, "seen": seen,
        "local": scores(lambda k, X: local[k][1].predict(local[k][0].transform(X))),
        "federated": scores(lambda k, X: glob.predict(fed_sc.transform(X))),
        "centralised_upper_bound": scores(lambda k, X: central.predict(cen_sc.transform(X))),
    }
