"""Ollama (local LLM server) over its HTTP API; no extra dependency."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import Completion, LLMProvider, ProviderError

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2"  # override with OLLAMA_MODEL


class OllamaProvider(LLMProvider):
    name = "ollama"
    status = "LOCAL"

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 90.0):
        super().__init__(model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL, timeout)
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_URL).rstrip("/")

    def configured(self) -> bool:
        return True  # no secret needed; whether it is running is a separate question

    def installed_models(self, timeout: float = 1.5) -> list[str] | None:
        """Model names the server has pulled, or None when the server is not reachable."""
        try:
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=timeout) as r:  # noqa: S310
                data = json.loads(r.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None

    def available(self) -> bool:
        names = self.installed_models()
        if names is None:
            return False
        want = self.model.split(":")[0]
        return any(n == self.model or n.split(":")[0] == want for n in names)

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> Completion:
        body = {"model": self.model, "stream": False, "options": {"temperature": 0.1},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(self.base_url + "/api/chat", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310
                data = json.loads(r.read().decode("utf-8"))
            text = data["message"]["content"]
        except urllib.error.HTTPError as e:
            raise ProviderError(f"Ollama HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(f"Ollama unreachable ({type(e).__name__})") from None
        except (KeyError, ValueError, TypeError):
            raise ProviderError("Ollama returned an unexpected reply") from None
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Ollama returned an empty reply")
        return Completion(text, self.name, self.model)
