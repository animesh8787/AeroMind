import numpy as np
import pytest

from aeromind.models import RULEstimator
from aeromind.models.conformal import ConformalRUL


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 50 + 10 * X[:, 0] + rng.normal(0, 8, n)  # noise the quantile model under-covers on few points
    return X, y


def test_conformal_restores_coverage_and_keeps_order():
    Xf, yf = _data(150, 0)
    Xc, yc = _data(2000, 1)
    Xt, yt = _data(4000, 2)
    base = RULEstimator().fit(Xf, yf)
    c = ConformalRUL(base, alpha=0.2).calibrate(Xc, yc)
    cov = lambda P: np.mean((P[:, 0] <= yt) & (yt <= P[:, 2]))  # noqa: E731
    P = c.predict_batch(Xt)
    assert abs(cov(P) - 0.8) < 0.05 and cov(P) > cov(base.predict_batch(Xt))
    assert (P[:, 0] <= P[:, 1]).all() and (P[:, 1] <= P[:, 2]).all() and (P >= 0).all()
    np.testing.assert_allclose(P[:, 1], base.predict_batch(Xt)[:, 1])
    assert c.predict(Xt[0]) == pytest.approx(tuple(P[0]))


def test_conformal_onnx_matches(tmp_path):
    pytest.importorskip("onnxruntime")
    from aeromind.onnx_export import OnnxBundle, export_onnx
    from aeromind.train import TrainConfig, collect_run, rul_inputs, train

    b = train(TrainConfig(healthy_runs=8, healthy_len=200, runs_per_mode=5, life_range=(200, 300), seed=11,
                          conformal=True))
    assert isinstance(b.rul, ConformalRUL) and b.rul.q is not None
    export_onnx(b, tmp_path)
    r = collect_run("overheating", 250, 8_500_000)
    T = rul_inputs(b.rul, r.X, b.anomaly.score(r.X))
    np.testing.assert_allclose(OnnxBundle(tmp_path).rul.predict_batch(T), b.rul.predict_batch(T), atol=0.05)
