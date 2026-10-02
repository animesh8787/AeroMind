"""Groq (OpenAI-compatible chat completions API) over plain HTTPS; no extra dependency.

The API key is read from ``GROQ_API_KEY`` and is only ever placed in the Authorization header.
It is never logged, printed or included in exceptions.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import Completion, LLMProvider, ProviderError

DEFAULT_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # override with GROQ_MODEL; Groq retires models over time


class GroqProvider(LLMProvider):
    name = "groq"
    status = "ONLINE"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None,
                 timeout: float = 30.0):
        super().__init__(model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL, timeout)
        self._key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self.base_url = (base_url or os.environ.get("GROQ_BASE_URL") or DEFAULT_URL).rstrip("/")

    def configured(self) -> bool:
        return bool(self._key.strip())

    def available(self) -> bool:
        # A key is necessary; reachability is learned from the first real request (no startup ping).
        return self.configured()

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> Completion:
        if not self.configured():
            raise ProviderError("GROQ_API_KEY is not set")
        body = {"model": self.model, "temperature": 0.1, "max_tokens": 900,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}",
                     "User-Agent": "aeromind-copilot/2"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (fixed https URL)
                data = json.loads(r.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            raise ProviderError(f"Groq HTTP {e.code}") from None  # no body: it may echo request data
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(f"Groq unreachable ({type(e).__name__})") from None
        except (KeyError, IndexError, ValueError, TypeError):
            raise ProviderError("Groq returned an unexpected reply") from None
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("Groq returned an empty reply")
        return Completion(text, self.name, self.model)
