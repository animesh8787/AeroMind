"""IMS loader logic on synthetic signals (the real 1.1 GB dataset is not needed in CI)."""

import numpy as np

from aeromind.datasets import ims


def _impacts(rate_hz, n=20480, fs=20000, amp=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = 0.3 * rng.standard_normal(n)
    ring = np.exp(-np.arange(60) / 8) * np.sin(2 * np.pi * 4000 * np.arange(60) / fs)
    pulses = np.zeros(n)
    pulses[(np.arange(0, 1, 1 / rate_hz) * fs).astype(int)] = amp
    return x + np.convolve(pulses, ring)[:n], t


def test_bpfo_evidence_found_with_speed_slip_and_absent_in_noise():
    bpfo = ims.GEOMETRY.orders()["BPFO"] * ims.SHAFT_HZ
    assert abs(bpfo - 236.4) < 0.1
    x, _ = _impacts(bpfo * 0.975)  # 2.5% slower shaft, as on the real rig
    assert ims.bpfo_snr(x) > 4
    noise = 0.3 * np.random.default_rng(1).standard_normal(20480)
    assert ims.bpfo_snr(noise) < 4


def test_gate_series_and_run_on_synthetic_degradation():
    assert list(ims.gate_series(np.array([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]), n=5, of=8)) == \
        [False] * 4 + [True] * 4 + [False] * 3
    rng = np.random.default_rng(0)
    n = 300
    hours = np.arange(n) / 6.0
    X = rng.normal(size=(n, 4, len(ims.FEATURES)))
    X[220:, 0, 0] += np.linspace(5, 20, n - 220)  # bearing 1 degrades late
    X[:, :, -1] = 1.0
    X[230:, 0, -1] = 30.0  # strong outer-race line on bearing 1 only
    r = ims.run(hours, X)
    b1 = r["bearings"]["bearing_1"]
    assert b1["first_gate_open_hours"] is not None and 30 <= b1["first_gate_open_hours"] <= 40
    assert b1["bpfo_evidence_first_hours"] is not None
    assert r["localisation"]["share_of_snapshots_strongest"]["bearing_1"] == 1.0
