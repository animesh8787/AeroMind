"""The real Ollama provider over real HTTP against a small fake Ollama server (no model, no internet)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aeromind.llm.copilot import Copilot
from aeromind.llm.ollama_provider import OllamaProvider
from aeromind.llm.router import LLMRouter
from aeromind.llm.schemas import CopilotRequest
from conftest import GOOD_EXPLANATION


class FakeOllama(BaseHTTPRequestHandler):
    reply = GOOD_EXPLANATION
    requests: list = []

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"models": [{"name": "llama3.2:latest"}]} if self.path == "/api/tags" else {}, 200 if self.path == "/api/tags" else 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        self._send({"message": {"role": "assistant", "content": FakeOllama.reply}})


@pytest.fixture
def server():
    FakeOllama.requests = []
    FakeOllama.reply = GOOD_EXPLANATION
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_over_http_gives_local_status_and_ai_answer(server, source, kb):
    prov = OllamaProvider(base_url=server, model="llama3.2")
    assert prov.available() and prov.installed_models() == ["llama3.2:latest"]
    cp = Copilot(LLMRouter(providers=[prov]), kb)
    st = cp.status()
    assert st["status"] == "LOCAL" and st["provider"] == "ollama"
    r = cp.ask(CopilotRequest(task="explain_alert", tail="VT-AMA03"), source)
    assert r.ai_generated and r.provider == "ollama" and r.llm_status == "LOCAL" and "bearing wear" in r.text
    sent = FakeOllama.requests[0]
    assert sent["format"] == "json" and sent["stream"] is False
    assert "Ignore" not in json.dumps(sent) and "degradation" not in json.dumps(sent)


def test_ollama_server_down_means_fallback_not_fake_llm(source, kb):
    prov = OllamaProvider(base_url="http://127.0.0.1:9", model="llama3.2")  # nothing listens on port 9
    cp = Copilot(LLMRouter(providers=[prov]), kb)
    assert cp.status()["status"] == "FALLBACK"
    r = cp.ask(CopilotRequest(task="explain_alert", tail="VT-AMA03"), source)
    assert not r.ai_generated and r.provider == "deterministic"


def test_ollama_garbage_reply_is_repaired_then_falls_back(server, source, kb):
    FakeOllama.reply = "I am a chatty model, not JSON."
    cp = Copilot(LLMRouter(providers=[OllamaProvider(base_url=server, model="llama3.2")]), kb)
    r = cp.ask(CopilotRequest(task="explain_alert", tail="VT-AMA03"), source)
    assert not r.ai_generated and len(FakeOllama.requests) == 2  # one attempt + one repair
