"""Ground-station acceptance test at the API level (no browser): the live demo scenario."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cryptography")
pytest.importorskip("onnxruntime")

from aeromind.onnx_export import export_onnx  # noqa: E402
from aeromind.server.app import GroundStation  # noqa: E402
from aeromind.train import TrainConfig, train  # noqa: E402


@pytest.fixture(scope="module")
def gs(tmp_path_factory):
    d = tmp_path_factory.mktemp("gs")
    export_onnx(train(TrainConfig(healthy_runs=10, healthy_len=200, runs_per_mode=5, life_range=(200, 300),
                                  seed=11, phases=True)), d / "model")
    return GroundStation(d / "model", d / "work")


def test_live_demo_scenario(gs):
    fleet = gs.fleet
    assert len(fleet.aircraft) == 6 and len({a.sim.t for a in fleet.aircraft.values()}) == 1
    for _ in range(30):
        fleet.tick()
    assert len({ac.window.phase for ac in fleet.aircraft.values()}) > 1  # staggered flight phases

    # Bearing wear on one aircraft: classified, RUL falls, physics evidence, maintenance decision, ACARS.
    fleet.inject_component("VT-AMA01", "bearing_wear", life=200)
    # Temperature sensor failure on another: SENSOR_FAULT, no component alarm.
    fleet.inject_sensor("VT-AMA02", "stuck", "temperature")
    ruls = []
    for _ in range(170):
        fleet.tick()
        a = fleet.aircraft["VT-AMA01"].pipe.last_assessment
        if a is not None:
            ruls.append(a.rul_hours[1])
    ac1, ac2 = fleet.aircraft["VT-AMA01"], fleet.aircraft["VT-AMA02"]
    adv = ac1.last_advisory
    assert adv is not None and adv.fault_type == "bearing_wear"
    assert ruls and ruls[-1] < ruls[0]
    assert any("BPFO" in e for ev in ac1.events if ev["kind"] == "component" for e in ev["advisory"]["evidence"])
    assert ac1.decision is not None and ac1.work_orders and len(ac1.work_orders[0]["acars"]) <= 220
    assert [e["advisory"]["channel"] for e in ac2.events if e["kind"] == "sensor"] == ["temperature"]
    assert not any(e["kind"] == "component" for e in ac2.events)
    for tail in ("VT-AMA03", "VT-AMA04", "VT-AMA05", "VT-AMA06"):
        assert not any(e["kind"] == "component" for e in fleet.aircraft[tail].events)

    # OTA: signed package accepted, tampered package rejected, rollback restores the previous model.
    assert gs.ota("valid") == {"accepted": True, "active": "v2"}
    r = gs.ota("tampered")
    assert r["accepted"] is False and "SHA-256" in r["reason"] and fleet.model_version == "v2"
    fleet.tick()  # fleet keeps running on v2
    assert gs.rollback()["active"] == "v1" and fleet.model_version == "v1"
    snap = fleet.snapshot()
    assert {e["kind"] for e in snap["events"]} >= {"component", "sensor", "ota"}


def test_http_api(gs):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(_app_for(gs))
    assert client.get("/").status_code == 200 and "AeroMind Ground Station" in client.get("/").text
    assert len(client.get("/api/fleet").json()["aircraft"]) == 6
    assert client.post("/api/aircraft/VT-AMA03/inject", json={"fault": "overheating"}).status_code == 200
    assert client.post("/api/aircraft/VT-AMA03/inject", json={"fault": "pressure_spike"}).status_code == 200
    assert client.post("/api/aircraft/VT-AMA03/inject", json={"fault": "nonsense"}).status_code == 400
    assert client.post("/api/aircraft/VT-XXX/reset").status_code == 404
    assert client.post("/api/aircraft/VT-AMA03/phase", json={"phase": "cruise"}).status_code == 200
    assert client.get("/api/aircraft/VT-AMA03").json()["tail"] == "VT-AMA03"


def _app_for(gs):
    """The FastAPI app bound to an existing GroundStation (avoids retraining in tests)."""
    from aeromind.server import app as app_module

    original = app_module.GroundStation
    app_module.GroundStation = lambda *a, **k: gs
    try:
        return app_module.create_app()
    finally:
        app_module.GroundStation = original


def test_roi_and_whatif_endpoints(gs):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(_app_for(gs))
    assert client.get("/api/roi").json()["status"] in ("disabled", "computing", "ready")
    gs.fleet.reset("VT-AMA06")
    assert client.post("/api/aircraft/VT-AMA06/whatif", json={"hours_to_next_check": 5}).status_code == 409
    gs.fleet.inject_component("VT-AMA06", "pressure_leak", life=150)
    for _ in range(140):
        gs.fleet.tick()
    near = client.post("/api/aircraft/VT-AMA06/whatif", json={"hours_to_next_check": 1}).json()
    far = client.post("/api/aircraft/VT-AMA06/whatif", json={"hours_to_next_check": 200}).json()
    assert near["p_fail_before_check"] <= far["p_fail_before_check"]
