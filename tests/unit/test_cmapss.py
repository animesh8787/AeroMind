import numpy as np

from aeromind.datasets.cmapss import SENSORS, CmapssRUL, RegimeNormaliser, load_subset, nasa_score


def _write_fake_subset(d, subset="FD001", n_units=6, seed=0):
    """Write C-MAPSS-format files (26 columns) with a slow sensor drift toward failure."""
    rng = np.random.default_rng(seed)
    train_rows, test_rows, rul = [], [], []
    for u in range(1, n_units + 1):
        life = int(rng.integers(150, 200))
        for c in range(1, life + 1):
            row = [u, c, *rng.normal(0, 1e-3, 3), *(100 + rng.normal(0, 0.5, 21) + 0.05 * c * np.arange(21) / 21)]
            train_rows.append(row)
        cut = life - int(rng.integers(10, 60))
        for c in range(1, cut + 1):
            test_rows.append([u, c, *rng.normal(0, 1e-3, 3), *(100 + rng.normal(0, 0.5, 21) + 0.05 * c * np.arange(21) / 21)])
        rul.append(life - cut)
    np.savetxt(d / f"train_{subset}.txt", np.array(train_rows))
    np.savetxt(d / f"test_{subset}.txt", np.array(test_rows))
    np.savetxt(d / f"RUL_{subset}.txt", np.array(rul))
    return rul


def test_nasa_score_is_asymmetric_and_zero_when_exact():
    t = np.array([50.0])
    assert nasa_score(t, t) == 0.0
    assert nasa_score(t + 10, t) > nasa_score(t - 10, t)  # late is worse than early
    np.testing.assert_allclose(nasa_score(t + 10, t), np.exp(1.0) - 1)
    np.testing.assert_allclose(nasa_score(t - 13, t), np.exp(1.0) - 1)


def test_loader_splits_units_and_selects_sensors(tmp_path):
    rul = _write_fake_subset(tmp_path)
    train, test, rul_loaded = load_subset(tmp_path, "FD001")
    assert len(train.sensors) == len(test.sensors) == 6
    assert train.sensors[0].shape[1] == len(SENSORS) and train.settings[0].shape[1] == 3
    np.testing.assert_array_equal(rul_loaded, rul)


def test_regime_normaliser_standardises_each_regime():
    rng = np.random.default_rng(0)
    from aeromind.datasets.cmapss import Engines

    settings = np.vstack([np.tile([a, 0.0, 0.0], (300, 1)) for a in (0.0, 10.0, 20.0)])
    sensors = np.vstack([rng.normal(loc, 2.0, size=(300, 4)) for loc in (0.0, 50.0, 100.0)])
    norm = RegimeNormaliser(k=3).fit(Engines([settings], [sensors]))
    z = norm.transform(settings, sensors)
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(z.std(0), 1, atol=0.05)


def test_lstm_end_to_end_on_fake_data(tmp_path):
    import pytest

    pytest.importorskip("torch")
    _write_fake_subset(tmp_path, n_units=12)
    train, test, _ = load_subset(tmp_path, "FD001")
    model = CmapssRUL("FD001", rul="lstm", lstm_hidden=8).fit(train)
    preds = model.predict_last(test)
    for key in ("aeromind", "no_anomaly_features"):
        q = preds[key]
        assert q.shape == (12, 3) and np.isfinite(q).all()
        assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()


def test_pipeline_end_to_end_on_fake_data(tmp_path):
    _write_fake_subset(tmp_path, n_units=12)
    train, test, _ = load_subset(tmp_path, "FD001")
    preds = CmapssRUL("FD001").fit(train).predict_last(test)
    q = preds["aeromind"]
    assert q.shape == (12, 3) and np.isfinite(q).all()
    assert (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
