import numpy as np
import pytest

from aeromind.features import FEATURE_NAMES, extract_features
from aeromind.pipeline import EdgePipeline
from aeromind.simulator import FLIGHT_PROFILE, PHASES, LiveAircraft, phase_context, simulate_run
from aeromind.train import TrainConfig, train


def test_flight_profile_cycles_and_sets_context():
    ws = [w for w, _ in simulate_run("healthy", 2 * sum(n for _, n, _, _ in FLIGHT_PROFILE), 4, phases=True)]
    assert [w.phase for w in ws[:len(PHASES) + 6]][:3] == ["taxi_out", "taxi_out", "takeoff"]
    assert set(w.phase for w in ws) == set(PHASES)
    cruise = [w for w in ws if w.phase == "cruise"]
    ground = [w for w in ws if w.phase == "takeoff"]
    assert all(w.altitude_ft == 36_000 for w in cruise) and all(w.altitude_ft == 0 for w in ground)
    assert np.mean([w.ambient_c for w in cruise]) < np.mean([w.ambient_c for w in ground]) - 50
    assert np.mean([w.load for w in ground]) > np.mean([w.load for w in cruise])
    x = extract_features(cruise[0])
    assert x[FEATURE_NAMES.index("altitude_kft")] == 36.0


def test_default_simulation_is_unchanged_without_phases():
    w = next(iter(simulate_run("healthy", 1, 3)))[0]
    assert (w.phase, w.ambient_c, w.altitude_ft) == ("steady", 15.0, 0.0)
    assert phase_context("cruise", 30.0)[2] == pytest.approx(30.0 - 1.98 * 36)


def test_live_aircraft_injects_faults_at_any_time():
    ac = LiveAircraft(seed=1)
    for _ in range(20):
        ac.step()
    ac.inject("bearing_wear", life=100)
    _, truth = ac.step()
    assert truth.mode == "bearing_wear" and truth.degradation == 0.0 and truth.rul_windows == 100
    for _ in range(50):
        _, truth = ac.step()
    assert truth.degradation == pytest.approx(0.25) and truth.rul_windows == 50
    ac.phase_override = "cruise"
    assert ac.step()[0].phase == "cruise"
    with pytest.raises(ValueError):
        ac.inject("gremlins")


@pytest.fixture(scope="module")
def phase_bundle():
    return train(TrainConfig(healthy_runs=8, healthy_len=200, runs_per_mode=4, life_range=(200, 300), seed=11,
                             phases=True))


def test_phase_aware_model_stays_quiet_on_healthy_flights_and_catches_faults(phase_bundle):
    pipe = EdgePipeline(phase_bundle)
    gate = [pipe.process(w) or pipe.last_flags >= 5 for w, _ in simulate_run("healthy", 250, 31, phases=True)]
    assert np.mean(gate) < 0.02
    pipe = EdgePipeline(phase_bundle)
    adv = [a for w, _ in simulate_run("pressure_leak", 260, 32, phases=True) if (a := pipe.process(w))]
    assert any(a.fault_type == "pressure_leak" for a in adv)
