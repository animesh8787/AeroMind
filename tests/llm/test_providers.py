"""Provider abstraction, Groq and Ollama with mocked HTTP, and router fallback order."""

import io
import json
import urllib.error

import pytest

from aeromind.llm import groq_provider, ollama_provider
from aeromind.llm.base import ProviderError
from aeromind.llm.groq_provider import GroqProvider
from aeromind.llm.ollama_provider import OllamaProvider
from aeromind.llm.router import AllProvidersFailed, LLMRouter, load_dotenv, parse_priority
from conftest import ScriptedProvider


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, module, payload=None, exc=None, capture=None):
    def fake(req, timeout=None):
        if capture is not None:
            capture.append(req)
        if exc:
            raise exc
        return _Resp(json.dumps(payload).encode())
    monkeypatch.setattr(module.urllib.request, "urlopen", fake)


def test_groq_mocked_response_and_key_never_in_body(monkeypatch):
    seen = []
    _patch_urlopen(monkeypatch, groq_provider, {"choices": [{"message": {"content": '{"summary": "ok"}'}}]}, capture=seen)
    p = GroqProvider(api_key="gsk_test_secret", model="m1")
    out = p.complete("sys", "user")
    assert (out.text, out.provider, out.model) == ('{"summary": "ok"}', "groq", "m1")
    req = seen[0]
    assert req.get_header("Authorization") == "Bearer gsk_test_secret"
    assert b"gsk_test_secret" not in req.data
    body = json.loads(req.data)
    assert body["response_format"] == {"type": "json_object"} and body["messages"][0]["role"] == "system"


def test_groq_without_key_is_unconfigured_and_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    p = GroqProvider()
    assert not p.configured() and not p.available()
    with pytest.raises(ProviderError):
        p.complete("s", "u")


def test_groq_http_error_does_not_leak_key(monkeypatch):
    err = urllib.error.HTTPError("https://x", 401, "Unauthorized", {}, io.BytesIO(b"bad key gsk_test_secret"))
    _patch_urlopen(monkeypatch, groq_provider, exc=err)
    with pytest.raises(ProviderError) as e:
        GroqProvider(api_key="gsk_test_secret").complete("s", "u")
    assert "gsk_test_secret" not in str(e.value) and "401" in str(e.value)


def test_groq_unexpected_reply_is_provider_error(monkeypatch):
    _patch_urlopen(monkeypatch, groq_provider, {"unexpected": True})
    with pytest.raises(ProviderError):
        GroqProvider(api_key="k").complete("s", "u")


def test_ollama_mocked_response(monkeypatch):
    seen = []
    _patch_urlopen(monkeypatch, ollama_provider, {"message": {"content": '{"summary": "local"}'}}, capture=seen)
    out = OllamaProvider(base_url="http://localhost:11434", model="llama3.2").complete("s", "u")
    assert out.provider == "ollama" and out.text == '{"summary": "local"}'
    assert json.loads(seen[0].data)["format"] == "json" and json.loads(seen[0].data)["stream"] is False


def test_ollama_unreachable(monkeypatch):
    _patch_urlopen(monkeypatch, ollama_provider, exc=urllib.error.URLError("refused"))
    p = OllamaProvider()
    assert p.installed_models() is None and not p.available()
    with pytest.raises(ProviderError):
        p.complete("s", "u")


def test_ollama_available_requires_model_pulled(monkeypatch):
    _patch_urlopen(monkeypatch, ollama_provider, {"models": [{"name": "llama3.2:latest"}]})
    assert OllamaProvider(model="llama3.2").available()
    assert not OllamaProvider(model="mistral").available()


def test_priority_parsing():
    assert parse_priority(None) == ["groq", "ollama"]
    assert parse_priority("auto") == ["groq", "ollama"]
    assert parse_priority("ollama,groq") == ["ollama", "groq"]
    assert parse_priority("off") == [] and parse_priority("fallback") == []
    with pytest.raises(ValueError):
        parse_priority("openai")


def test_router_prefers_first_provider_then_falls_through():
    a, b = ScriptedProvider(["A"]), ScriptedProvider(["B"])
    a.name, b.name, b.status = "groq", "ollama", "LOCAL"
    r = LLMRouter(providers=[a, b])
    assert r.complete("s", "u").text == "A" and r.status()["status"] == "ONLINE"
    a.replies = [ProviderError("down")]
    assert r.complete("s", "u").text == "B"  # groq fails -> ollama answers
    assert r.status()["status"] == "LOCAL" and r.status()["provider"] == "ollama"  # groq is cooling down


def test_router_all_failed_raises_and_status_is_fallback():
    a = ScriptedProvider([ProviderError("down")])
    r = LLMRouter(providers=[a])
    with pytest.raises(AllProvidersFailed):
        r.complete("s", "u")
    assert r.status()["status"] == "FALLBACK" and r.status()["provider"] == "deterministic"


def test_router_with_no_providers_is_fallback():
    assert LLMRouter(providers=[]).status()["status"] == "FALLBACK"


def test_dotenv_does_not_override_environment_or_print(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "already-set")
    f = tmp_path / ".env"
    f.write_text("# c\nGROQ_MODEL=from-file\nOLLAMA_MODEL=\"qwen\"\nGROQ_API_KEY=\n", encoding="utf-8")
    load_dotenv(f)
    import os
    assert os.environ["GROQ_MODEL"] == "already-set" and os.environ["OLLAMA_MODEL"] == "qwen"
    assert capsys.readouterr().out == ""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
