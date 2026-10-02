import pickle

import numpy as np
import pytest

from aeromind.core.features import N_FEATURES, SequenceTracker, sequence_matrix


def test_sequence_tracker_matches_offline_matrix_and_pads_left():
    rng = np.random.default_rng(0)
    X, s = rng.normal(size=(12, 4)), rng.normal(size=12)
    tracker = SequenceTracker(5)
    online = np.stack([tracker.update(x, v) for x, v in zip(X, s)])
    offline = sequence_matrix(X, s, 5)
    np.testing.assert_array_equal(online, offline)
    assert offline.shape == (12, 5, 5)
    np.testing.assert_array_equal(offline[0], np.tile(np.append(X[0], s[0]), (5, 1)))  # first window repeated
    np.testing.assert_array_equal(offline[-1, -1], np.append(X[-1], s[-1]))


torch = pytest.importorskip("torch")

from aeromind.models import LSTMRULEstimator  # noqa: E402
from aeromind.edge.pipeline import EdgePipeline  # noqa: E402
from aeromind.core.simulator import simulate_run  # noqa: E402
from aeromind.core.train import TrainConfig, train  # noqa: E402


def _toy(n_runs=12, length=60, seq_len=8, seed=0):
    """Runs whose single feature drifts up as RUL falls."""
    rng = np.random.default_rng(seed)
    S, y, g = [], [], []
    for k in range(n_runs):
        rul = np.arange(length, 0, -1, dtype=float)
        x = (1 - rul / length)[:, None] + 0.05 * rng.normal(size=(length, 1))
        S.append(sequence_matrix(x, np.zeros(length), seq_len))
        y.append(rul)
        g.append(np.full(length, k))
    return np.concatenate(S), np.concatenate(y), np.concatenate(g)


def test_lstm_learns_ordered_quantiles_and_pickles_without_its_network():
    S, y, g = _toy()
    m = LSTMRULEstimator(seq_len=8, hidden=8, epochs=40, seed=1).fit(S, y, g)
    P = m.predict_batch(S)
    assert P.shape == (len(S), 3)
    assert (P >= 0).all() and (P[:, 0] <= P[:, 1]).all() and (P[:, 1] <= P[:, 2]).all()
    assert np.corrcoef(P[:, 1], y)[0, 1] > 0.8
    clone = pickle.loads(pickle.dumps(m))
    assert clone._net is None and isinstance(next(iter(clone.state.values())), np.ndarray)
    np.testing.assert_allclose(clone.predict_batch(S[:20]), P[:20], rtol=1e-6)


def test_lstm_rejects_wrong_length_and_single_group():
    S, y, g = _toy(seq_len=8)
    with pytest.raises(ValueError, match="length"):
        LSTMRULEstimator(seq_len=10).fit(S, y, g)
    with pytest.raises(ValueError, match="groups"):
        LSTMRULEstimator(seq_len=8, val_fraction=1.0).fit(S, y, g)


@pytest.fixture(scope="module")
def lstm_bundle():
    cfg = TrainConfig(healthy_runs=8, healthy_len=200, runs_per_mode=4, life_range=(200, 300), seed=11, rul_model="lstm")
    return train(cfg)


def test_pipeline_runs_lstm_rul(lstm_bundle):
    pipe = EdgePipeline(lstm_bundle)
    adv = [a for w, _ in simulate_run("pressure_leak", 260, 424242) if (a := pipe.process(w)) is not None]
    assert adv
    assert all(a.rul_hours_p10 <= a.rul_hours_p50 <= a.rul_hours_p90 for a in adv)
    assert adv[-1].rul_hours_p50 < adv[0].rul_hours_p50  # RUL falls as the fault develops


def test_lstm_onnx_export_matches_torch(lstm_bundle, tmp_path):
    pytest.importorskip("onnxruntime")
    import onnx

    from aeromind.edge.onnx_export import TENSORRT_OPERATORS, OnnxBundle, export_onnx, operators
    from aeromind.core.train import collect_run, rul_inputs

    man = export_onnx(lstm_bundle, tmp_path)
    assert man["rul_input"] == {"name": "sequence", "kind": "sequence", "shape": [30, N_FEATURES + 1]}
    rul_graph = onnx.load(tmp_path / "rul.onnx")
    assert "LSTM" in operators(rul_graph) and set(operators(rul_graph)) <= TENSORRT_OPERATORS
    r = collect_run("overheating", 200, 8_400_000)
    S = rul_inputs(lstm_bundle.rul, r.X, lstm_bundle.anomaly.score(r.X))
    ob = OnnxBundle(tmp_path)
    full = ob.rul.predict_batch(S)
    np.testing.assert_allclose(full, lstm_bundle.rul.predict_batch(S), atol=1e-3)
    np.testing.assert_allclose(np.array([ob.rul.predict(s) for s in S[:10]]), full[:10], atol=1e-3)
    # The pipeline gives the same advisories on both runtimes.
    def advisories(models):
        pipe = EdgePipeline(models)
        return [(a.window, a.fault_type, a.priority)
                for w, _ in simulate_run("overheating", 220, 434343) if (a := pipe.process(w)) is not None]

    assert advisories(OnnxBundle(tmp_path)) == advisories(lstm_bundle)
