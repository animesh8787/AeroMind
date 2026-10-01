import numpy as np

from aeromind.features import FEATURE_NAMES
from aeromind.models.classifier import CONTEXT, ContextResidual


def test_context_residual_removes_operating_context():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    load, oat, alt = (FEATURE_NAMES.index(c) for c in CONTEXT)
    X[:, load] = rng.uniform(0.45, 1.0, n)
    X[:, oat] = rng.uniform(-55, 40, n)
    X[:, alt] = rng.choice([0.0, 15.0, 36.0], n)
    t = FEATURE_NAMES.index("temp_mean")
    X[:, t] = 60 + 25 * X[:, load] + 0.35 * (X[:, oat] - 15) + 0.1 * rng.normal(size=n)
    ctx = ContextResidual(FEATURE_NAMES).fit(X)
    R = ctx.transform(X)
    assert R.shape == (n, len(FEATURE_NAMES) + 3)
    assert np.std(R[:, t]) < 0.15 < np.std(X[:, t])  # context explained away
    shifted = X[:5].copy()
    shifted[:, t] += 8.0  # a real temperature rise at the same context
    np.testing.assert_allclose(ctx.transform(shifted)[:, t] - R[:5, t], 8.0, atol=1e-9)
    np.testing.assert_array_equal(R[:, -3:], X[:, [load, oat, alt]])
