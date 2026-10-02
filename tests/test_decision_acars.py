import pytest

from aeromind.communications.acars import MAX_CHARS, decode, encode
from aeromind.core.alerts import Advisory
from aeromind.core.config import FAULT_MODES
from aeromind.maintenance.decision import (DEFER, GROUND_NOW, REPLACE_AT_NEXT_CHECK, Schedule, decide, failure_cdf,
                               work_order)
from aeromind.edge.pipeline import EdgePipeline
from aeromind.edge.sensor_health import SensorFault, SensorFaultInjector
from aeromind.core.simulator import simulate_run
from aeromind.core.train import TrainConfig, train


def _adv(p10, p50, p90, fault="bearing_wear"):
    return Advisory("ENG1-GEARBOX", 100, 50.0, fault, 0.97, 4.2, p10, p50, p90, "URGENT", "x", "y",
                    ("vib_rms", "oil_mean", "ac_rms"), ("Envelope peak 115 Hz ≈ BPFO (3.57× shaft 32.2 Hz), SNR 6.1",))


def test_failure_cdf_hits_the_quantiles_and_is_monotone():
    assert failure_cdf(10, 10, 20, 40) == pytest.approx(0.1)
    assert failure_cdf(20, 10, 20, 40) == pytest.approx(0.5)
    assert failure_cdf(40, 10, 20, 40) == pytest.approx(0.9)
    ts = [0, 5, 10, 15, 20, 30, 40, 50, 100]
    vals = [failure_cdf(t, 10, 20, 40) for t in ts]
    assert vals == sorted(vals) and vals[0] == 0.0 and vals[-1] == 1.0
    assert 0 <= failure_cdf(1, 0, 0, 0) <= 1  # degenerate quantiles stay valid


def test_decisions_escalate_as_rul_shrinks():
    s = Schedule(hours_to_next_check=10, leg_hours=2)
    assert decide(_adv(120, 150, 160), s).action == DEFER
    assert decide(_adv(15, 30, 45), s).action == REPLACE_AT_NEXT_CHECK
    d = decide(_adv(0.5, 2, 6), s)
    assert d.action == GROUND_NOW and d.p_fail_next_leg >= 0.05
    assert "not a certified" in d.policy_note


def test_acars_round_trip_and_size():
    a = _adv(15.2, 30.4, 45.9)
    msg = encode(a, "VT-AMA01")
    assert len(msg) <= MAX_CHARS
    d = decode(msg)
    assert d["fault_type"] == "bearing_wear" and d["priority"] == "URGENT" and d["confidence"] == 0.97
    assert d["rul_hours"] == (15.0, 30.0, 46.0)
    assert d["contributing_signals"] == ["vib_rms", "oil_mean", "ac_rms"]
    assert d["evidence"].startswith("ENVELOPE PEAK 115 HZ = BPFO")
    wo = work_order("VT-AMA01", a, decide(a))
    assert wo.acars == msg and wo.part and wo.number.startswith("WO-VT-AMA01-")


@pytest.fixture(scope="module")
def bundle():
    return train(TrainConfig(healthy_runs=8, healthy_len=200, runs_per_mode=4, life_range=(200, 300), seed=11))


def test_every_pipeline_advisory_fits_one_acars_block(bundle):
    n = 0
    for i, mode in enumerate(FAULT_MODES):
        pipe = EdgePipeline(bundle)
        inj = SensorFaultInjector(SensorFault("stuck", "temperature", start=30))
        for w, _ in simulate_run(mode, 260, 515 + i):
            a = pipe.process(inj.apply(w))
            if a is not None:
                d = decode(encode(a, "VT-AMA01"))
                assert d["fault_type"] == a.fault_type and d["priority"] == a.priority
                n += 1
        for sa in pipe.sensor_advisories:
            d = decode(encode(sa, "VT-AMA01"))
            assert d["channel"] == sa.channel and d["status"] == sa.status
            n += 1
    assert n > 20


def test_bearing_advisory_carries_physics_evidence(bundle):
    pipe = EdgePipeline(bundle)
    adv = [a for w, _ in simulate_run("bearing_wear", 300, 123) if (a := pipe.process(w)) is not None]
    bearing = [a for a in adv if a.fault_type == "bearing_wear"]
    assert bearing
    assert any("BPFO" in e for a in bearing for e in a.evidence)
    assert not any("BPFO" in e for a in adv if a.fault_type != "bearing_wear" for e in a.evidence)
