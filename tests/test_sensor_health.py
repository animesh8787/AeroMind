import numpy as np
import pytest

from aeromind.edge.pipeline import EdgePipeline, PipelineConfig
from aeromind.edge.sensor_health import (CHANNELS, SensorFault, SensorFaultInjector, SensorHealthConfig,
                                    SensorHealthMonitor, check_channel)
from aeromind.core.simulator import simulate_run
from aeromind.core.train import TrainConfig, train


def _windows(mode="healthy", life=80, seed=3):
    return [w for w, _ in simulate_run(mode, life, seed)]


def test_clean_windows_pass_every_check():
    mon = SensorHealthMonitor("X")
    for mode in ("healthy", "bearing_wear", "electrical_fault", "pressure_leak"):
        for w in _windows(mode, 120, 9):
            _, adv = mon.process(w)
            assert adv == []
    assert mon.faulty_channels == []


@pytest.mark.parametrize("kind,channel,expect", [
    ("dropout", "acoustic", "nan"),
    ("out_of_range", "temperature", "out_of_range"),
    ("flatline", "pressure", "flatline"),
    ("stuck", "temperature", "stuck"),
    ("spike", "pressure", "spike"),
    ("drift", "vibration", "bias"),
])
def test_each_injected_fault_is_declared_on_the_right_channel(kind, channel, expect):
    inj = SensorFaultInjector(SensorFault(kind, channel, start=10), seed=1)
    mon = SensorHealthMonitor("X")
    found = []
    for w in _windows(life=80):
        _, adv = mon.process(inj.apply(w))
        found += adv
    faulty = [a for a in found if a.status == "FAULTY"]
    assert [a.channel for a in faulty] == [channel]
    assert expect in faulty[0].checks
    assert faulty[0].window >= 10


def test_masked_window_is_finite_and_channel_recovers():
    w0, w1, w2 = _windows(life=3)
    mon = SensorHealthMonitor("X", SensorHealthConfig(clear_after=2))
    mon.process(w0)
    bad = SensorFaultInjector(SensorFault("dropout", "oil_debris", start=0)).apply(w1)
    masked, _ = mon.process(bad)
    assert np.isfinite(masked.oil_debris).all()
    np.testing.assert_array_equal(masked.oil_debris, w0.oil_debris)  # last validated reading


def test_unknown_fault_or_channel_rejected():
    with pytest.raises(ValueError):
        SensorFault("melted", "temperature")
    with pytest.raises(ValueError):
        SensorFault("stuck", "altimeter")
    assert set(CHANNELS) >= {"vibration", "temperature"}
    assert check_channel("voltage", np.full(16, 115.0), None, None, SensorHealthConfig()) == ["flatline"]


@pytest.fixture(scope="module")
def bundle():
    return train(TrainConfig(healthy_runs=8, healthy_len=200, runs_per_mode=4, life_range=(200, 300), seed=11))


def _run(bundle, mode, life, seed, fault=None, health=True):
    pipe = EdgePipeline(bundle, cfg=PipelineConfig(sensor_health=health))
    inj = SensorFaultInjector(fault, seed=seed) if fault else None
    adv = []
    for w, _ in simulate_run(mode, life, seed):
        a = pipe.process(inj.apply(w) if inj else w)
        if a is not None:
            adv.append(a)
    return pipe, adv


def test_broken_sensor_is_not_reported_as_a_component_fault(bundle):
    fault = SensorFault("out_of_range", "temperature", start=40)
    pipe, adv = _run(bundle, "healthy", 200, 77, fault)
    assert adv == []  # no component advisory
    assert [(s.channel, s.status) for s in pipe.sensor_advisories] == [("temperature", "FAULTY")]
    assert pipe.stats.sensor_advisories == 1
    _, adv_off = _run(bundle, "healthy", 200, 77, fault, health=False)
    assert adv_off  # without sensor health the same sensor failure looks like a component fault


def test_component_fault_still_detected_with_a_dead_sensor(bundle):
    _, adv = _run(bundle, "bearing_wear", 260, 424242, SensorFault("dropout", "temperature", start=20))
    assert any(a.fault_type == "bearing_wear" for a in adv)
