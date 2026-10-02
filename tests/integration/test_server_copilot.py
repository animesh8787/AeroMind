"""Copilot and remote-ingest API on a real ground station (no network, no API key)."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cryptography")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient  # noqa: E402

from aeromind.edge.onnx_export import export_onnx  # noqa: E402
from aeromind.core.train import TrainConfig, train  # noqa: E402
from aeromind.llm import Copilot, LLMRouter  # noqa: E402
from aeromind.llm.base import Completion, LLMProvider  # noqa: E402
from aeromind.llm.knowledge import KnowledgeBase  # noqa: E402
from aeromind.server.app import create_app  # noqa: E402


class ContextEchoProvider(LLMProvider):
    """A fake model that answers from the CONTEXT block it is given (like a well-behaved LLM)."""

    name, status = "ollama", "LOCAL"

    def configured(self):
        return True

    def available(self):
        return True

    def complete(self, system, user, *, json_mode=True):
        ctx = json.loads(user.split("CONTEXT (JSON, the only source of facts):\n", 1)[1].split("\n\nOUTPUT JSON SCHEMA", 1)[0])
        if "recommended_inspection" in user.split("OUTPUT JSON SCHEMA:")[1]:
            body = {"recommended_inspection": ["Inspect per approved maintenance data"], "notes": "Draft only."}
        else:
            f = ctx.get("fault")
            if "aircraft_total" in ctx:
                f, ctx = None, {"aircraft": "fleet", "status": f"{ctx['aircraft_total']} aircraft", "physics_evidence": []}
            summary = (f"{ctx['aircraft']} shows {f['type'].replace('_', ' ')} with confidence {f['confidence']}."
                       if f else f"{ctx['aircraft']} has no component fault; status {ctx['status']}.")
            body = {"summary": summary, "evidence": list(ctx["physics_evidence"])[:2], "interpretation": [],
                    "limitations": ["Simulated data."]}
        return Completion(json.dumps(body), self.name, "echo-model")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("cp")
    export_onnx(train(TrainConfig(healthy_runs=10, healthy_len=200, runs_per_mode=5, life_range=(200, 300),
                                  seed=11, phases=True)), d / "model")
    cp = Copilot(LLMRouter(providers=[ContextEchoProvider("echo-model")]), KnowledgeBase(None))
    app = create_app(d / "model", d / "work", compute_roi=False, copilot=cp)
    with TestClient(app) as c:
        c.app.state.gs.fleet.paused = True  # the test drives the fleet by hand
        yield c


def drive(gs, n):
    for _ in range(n):
        gs.fleet.tick()


def test_llm_status_and_fallback_mode(client, tmp_path):
    st = client.get("/api/llm/status").json()
    assert st["status"] == "LOCAL" and st["provider"] == "ollama"
    gs = client.app.state.gs
    saved, gs.copilot = gs.copilot, Copilot(LLMRouter(providers=[]), KnowledgeBase(None))
    try:
        assert client.get("/api/llm/status").json()["status"] == "FALLBACK"
        r = client.post("/api/copilot", json={"tail": "VT-AMA04", "task": "summarize_aircraft"}).json()
        assert r["ai_generated"] is False and r["provider"] == "deterministic" and r["llm_status"] == "FALLBACK"
    finally:
        gs.copilot = saved


def test_bearing_wear_flow_copilot_explains_deterministic_output(client):
    gs = client.app.state.gs
    drive(gs, 30)
    gs.fleet.inject_component("VT-AMA01", "bearing_wear", life=200)
    gs.fleet.inject_sensor("VT-AMA02", "stuck", "temperature")
    drive(gs, 170)
    ac = gs.fleet.aircraft["VT-AMA01"]
    assert ac.last_advisory is not None and ac.last_advisory.fault_type == "bearing_wear"

    r = client.post("/api/copilot", json={"tail": "VT-AMA01", "question": "Why is this aircraft at risk?"}).json()
    assert r["task"] == "explain_alert" and r["ai_generated"] and r["provider"] == "ollama" and r["llm_status"] == "LOCAL"
    det = "\n".join(r["deterministic"]["lines"])
    assert "Fault: bearing_wear" in det and "RUL p10 / p50 / p90" in det and "Decision:" in det
    assert "bearing wear" in r["text"] and r["label"].startswith("AI-generated assistance")

    wo = client.post("/api/copilot", json={"tail": "VT-AMA01", "task": "work_order"}).json()
    assert wo["structured"]["status"] == "DRAFT" and wo["structured"]["fault"] == "bearing_wear"
    assert wo["structured"]["required_part"] == "bearing assembly kit"

    wi = client.post("/api/copilot", json={"tail": "VT-AMA01", "task": "what_if", "hours_to_next_check": 24}).json()
    assert any(x.startswith("What-if") and "24 flight hours" in x for x in wi["deterministic"]["lines"])

    # sensor failure is explained as a sensor fault, not a component fault
    s = client.post("/api/copilot", json={"tail": "VT-AMA02", "task": "why_fault"}).json()
    assert "no component fault" in s["text"].lower() and "SENSOR_FAULT on temperature" in "\n".join(s["deterministic"]["lines"])

    flr = client.post("/api/copilot", json={"task": "fleet_summary"})
    assert flr.status_code == 200, flr.text
    fl = flr.json()
    assert fl["structured"]["aircraft_total"] == 6
    assert "VT-AMA01" in {a["aircraft"] for a in fl["structured"]["needs_attention"]}
    assert fl["structured"]["sensor_health_problems"][0]["aircraft"] == "VT-AMA02"

    # the copilot never changed the deterministic state
    assert gs.fleet.aircraft["VT-AMA01"].last_advisory.fault_type == "bearing_wear"


def test_copilot_errors(client):
    assert client.post("/api/copilot", json={"tail": "VT-NOPE", "task": "explain_alert"}).status_code == 404
    assert client.post("/api/copilot", json={"tail": "VT-AMA01", "task": "fly_the_plane"}).status_code == 400


def test_remote_edge_ingest_and_copilot(client, monkeypatch):
    gs = client.app.state.gs
    adv = gs.fleet.aircraft["VT-AMA01"].last_advisory.to_dict()
    r = client.post("/api/ingest", json={"tail": "VT-PI01", "kind": "component", "advisory": adv, "model_version": "v1"})
    assert r.status_code == 200 and len(r.json()["acars"]) <= 220 and r.json()["acars"].startswith("AMD1/VT-PI01")
    assert [x["tail"] for x in client.get("/api/remote").json()] == ["VT-PI01"]
    assert any(e["tail"] == "VT-PI01" and e.get("remote") for e in client.get("/api/fleet").json()["events"])
    c = client.post("/api/copilot", json={"tail": "VT-PI01", "task": "explain_alert"}).json()
    assert c["tail"] == "VT-PI01" and "Fault: bearing_wear" in "\n".join(c["deterministic"]["lines"])

    bad = client.post("/api/ingest", json={"tail": "x", "kind": "component", "advisory": adv})
    assert bad.status_code == 400
    assert client.post("/api/ingest", json={"tail": "VT-PI01", "kind": "component", "advisory": {"fault_type": "x"}}).status_code == 400

    monkeypatch.setenv("AEROMIND_INGEST_TOKEN", "s3cret")
    body = {"tail": "VT-PI01", "kind": "component", "advisory": adv}
    assert client.post("/api/ingest", json=body).status_code == 401
    assert client.post("/api/ingest", json=body, headers={"X-AeroMind-Token": "wrong"}).status_code == 401
    assert client.post("/api/ingest", json=body, headers={"X-AeroMind-Token": "s3cret"}).status_code == 200


def test_dashboard_has_copilot_panel(client):
    html = client.get("/").text
    for needle in ("AI Maintenance Copilot", "DETERMINISTIC", "llm-status", "Draft Work Order", "What If We Defer?",
                   "Explain RUL", "Why This Fault?", "Summarize Aircraft", "Explain Alert", "What Should Maintenance Inspect?"):
        assert needle in html
