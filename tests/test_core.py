import json

import numpy as np
import pytest

from aeromind.core.config import FAULT_MODES, HEALTHY, N_ELEC, N_HIGH, N_SLOW
from aeromind.core.features import FEATURE_NAMES, N_FEATURES, TrendTracker, extract_features, trend_matrix
from aeromind.learning.federated import fedavg, federated_scaler
from aeromind.edge.pipeline import EdgePipeline
from aeromind.core.simulator import simulate_run
from aeromind.core.train import TrainConfig, train


def _features(mode, life, seed):
    return np.stack([extract_features(w) for w, _ in simulate_run(mode, life, seed)])


def test_window_shapes_and_determinism():
    (w1, _), = list(simulate_run(HEALTHY, 1, seed=3))
    (w2, _), = list(simulate_run(HEALTHY, 1, seed=3))
    assert w1.vibration.shape == (N_HIGH,) and w1.current.shape == (N_ELEC,)
    assert w1.temperature.shape == (N_SLOW,)
    np.testing.assert_array_equal(w1.vibration, w2.vibration)


def test_truth_degradation_and_rul():
    truths = [t for _, t in simulate_run("bearing_wear", 100, seed=1)]
    assert truths[0].degradation == 0.0 and truths[-1].degradation < 1.0
    assert truths[0].rul_windows == 100 and truths[-1].rul_windows == 1
    assert all(t.rul_windows is None for _, t in simulate_run(HEALTHY, 5, seed=1))


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        list(simulate_run("gremlins", 5, seed=0))


def test_features_finite_and_named():
    X = _features("electrical_fault", 20, 2)
    assert X.shape == (20, N_FEATURES) == (20, len(FEATURE_NAMES))
    assert np.isfinite(X).all()


@pytest.mark.parametrize("mode,feature", [
    ("bearing_wear", "vib_rms"),
    ("oil_contamination", "oil_mean"),
    ("overheating", "temp_mean"),
    ("electrical_fault", "cur_thd"),
    ("pressure_leak", "press_mean"),
])
def test_fault_moves_its_signature_feature(mode, feature):
    # Same seed for load/tail so only the fault differs between early and late windows.
    X = _features(mode, 200, 5)
    i = FEATURE_NAMES.index(feature)
    early, late = X[:20, i].mean(), X[-20:, i].mean()
    if mode == "pressure_leak":
        assert late < early - 5
    else:
        assert late > early * 1.2, (early, late)


def test_trend_tracker_matches_offline_matrix():
    rng = np.random.default_rng(0)
    X, s = rng.normal(size=(30, N_FEATURES)), rng.normal(size=30)
    tracker = TrendTracker()
    online = np.stack([tracker.update(x, v) for x, v in zip(X, s)])
    np.testing.assert_allclose(online, trend_matrix(X, s))


def test_fedavg_weighted_mean():
    a = [np.array([0.0, 0.0]), np.array([[1.0]])]
    b = [np.array([4.0, 8.0]), np.array([[5.0]])]
    out = fedavg([a, b], counts=[1, 3])
    np.testing.assert_allclose(out[0], [3.0, 6.0])
    np.testing.assert_allclose(out[1], [[4.0]])


def test_federated_scaler_equals_pooled_scaler():
    rng = np.random.default_rng(1)
    parts = [rng.normal(loc=i, scale=1 + i, size=(50 * (i + 1), 4)) for i in range(3)]
    sc = federated_scaler([(len(p), p.mean(0), p.var(0)) for p in parts])
    pooled = np.vstack(parts)
    np.testing.assert_allclose(sc.mean_, pooled.mean(0))
    np.testing.assert_allclose(sc.scale_, pooled.std(0))


@pytest.fixture(scope="module")
def small_bundle():
    return train(TrainConfig(healthy_runs=8, healthy_len=200, runs_per_mode=4, life_range=(200, 300), seed=11))


def _run(bundle, mode, life, seed):
    pipe = EdgePipeline(bundle)
    adv = [a for w, _ in simulate_run(mode, life, seed) if (a := pipe.process(w)) is not None]
    return pipe, adv


def test_pipeline_flags_fault_and_emits_valid_json(small_bundle):
    pipe, adv = _run(small_bundle, "pressure_leak", 260, seed=424242)
    assert adv, "expected at least one advisory before failure"
    first = json.loads(adv[0].to_json())
    assert first["fault_type"] in (*FAULT_MODES, "unclassified_anomaly")
    assert first["rul_hours_p10"] <= first["rul_hours_p50"] <= first["rul_hours_p90"]
    assert pipe.stats.advisory_bytes < pipe.stats.raw_bytes / 100


def test_pipeline_quiet_on_healthy_run(small_bundle):
    _, adv = _run(small_bundle, HEALTHY, 250, seed=424243)
    assert adv == []


def test_federated_rare_fault_sharing_transfers_unseen_faults():
    from aeromind.learning.federated import rare_fault_demo

    r = rare_fault_demo(n_clients=5, seed=3, rounds=6, local_epochs=2)
    assert r["local"]["unseen_fault_accuracy"] == 0.0  # it never saw those labels
    assert r["federated"]["unseen_fault_accuracy"] > 0.3  # transfer happens (full run: 89-93%)
    assert r["federated"]["healthy_called_faulty"] < 0.05
