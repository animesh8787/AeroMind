"""Edge agent: same EdgePipeline, advisories to a (fake) ground station, store-and-forward, bench labelling."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("onnxruntime")

from aeromind.communications.acars import MAX_CHARS  # noqa: E402
from aeromind.core.train import TrainConfig, train  # noqa: E402
from aeromind.edge import bench  # noqa: E402
from aeromind.edge.agent import GroundLink, run_agent  # noqa: E402
from aeromind.edge.onnx_export import export_onnx  # noqa: E402


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("edge") / "model"
    export_onnx(train(TrainConfig(healthy_runs=10, healthy_len=200, runs_per_mode=5, life_range=(200, 300),
                                  seed=11, phases=True)), d)
    return d


class Collector(BaseHTTPRequestHandler):
    got: list = []
    status = 200

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Collector.got.append((self.headers.get("X-AeroMind-Token"), body))
        self.send_response(Collector.status)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


@pytest.fixture
def ground():
    Collector.got, Collector.status = [], 200
    srv = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_agent_publishes_advisories_from_the_same_pipeline(model, ground, monkeypatch):
    monkeypatch.setenv("AEROMIND_INGEST_TOKEN", "tok")
    st = run_agent(model, "VT-PI01", ground, rate=0, max_windows=200, inject="bearing_wear", inject_after=5, life=200,
                   log=lambda *_: None)
    assert st.windows == 200 and st.advisories >= 1 and st.sent == st.advisories and st.failed_posts == 0
    token, body = Collector.got[0]
    assert token == "tok" and body["tail"] == "VT-PI01" and body["kind"] == "component"
    assert body["advisory"]["fault_type"] in ("bearing_wear", "unclassified_anomaly")
    assert len(json.dumps(body["advisory"])) < 600  # advisories, not sensor data, cross the link


def test_agent_reports_sensor_fault_and_nothing_else(model, ground):
    run_agent(model, "VT-PI01", ground, rate=0, max_windows=80, sensor_fault="stuck:temperature", inject_after=10,
              log=lambda *_: None)
    kinds = [b["kind"] for _, b in Collector.got]
    assert kinds and set(kinds) == {"sensor"} and Collector.got[0][1]["advisory"]["channel"] == "temperature"


def test_store_and_forward_when_ground_is_down(model):
    st = run_agent(model, "VT-PI01", "http://127.0.0.1:9", rate=0, max_windows=120, inject="bearing_wear",
                   inject_after=5, life=200, log=lambda *_: None)
    assert st.sent == 0 and st.failed_posts >= 1 and st.backlog >= 1 and st.windows == 120  # the agent keeps running


def test_rejected_posts_do_not_block_the_queue(ground):
    Collector.status = 401
    link, st = GroundLink(ground, token="x"), type("S", (), {"failed_posts": 0, "sent": 0, "backlog": 0})()
    link.send({"tail": "VT-PI01"}, st)
    assert st.failed_posts == 1 and st.backlog == 0  # a 401 will never succeed, so it is dropped


def test_agent_writes_jsonl_offline(model, tmp_path):
    out = tmp_path / "adv.jsonl"
    run_agent(model, "VT-PI01", None, rate=0, max_windows=120, inject="bearing_wear", inject_after=5, life=200,
              out=out, log=lambda *_: None)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all(r["advisory"] for r in rows)


def test_bench_refuses_to_label_a_non_pi_as_raspberry_pi(model, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "detect_platform", lambda: {"system": "Windows", "machine": "AMD64",
                                                          "is_raspberry_pi": False, "raspberry_pi_model": None})
    with pytest.raises(SystemExit) as e:
        bench.run_target("raspberry-pi", model, 5, None, tmp_path)
    assert "refusing" in str(e.value) and not (tmp_path / "raspberry-pi.json").exists()


def test_bench_laptop_target_saves_measured_result_and_compare(model, tmp_path):
    res = bench.run_target("laptop", model, 20, None, tmp_path)
    assert res["target"] == "laptop" and res["status"] == "MEASURED" and res["model_load_ms"] > 0
    assert res["throughput_windows_per_s"] > 0 and (tmp_path / "laptop.json").exists()
    table = bench.compare([tmp_path / "laptop.json", tmp_path / "laptop.json"])
    assert "throughput (windows/s)" in table and "laptop" in table
