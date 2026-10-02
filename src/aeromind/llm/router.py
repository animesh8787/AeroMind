"""Provider selection: Groq -> Ollama -> deterministic fallback (configurable).

``AEROMIND_LLM_PROVIDER`` is ``auto`` (groq, then ollama), ``groq``, ``ollama``, a comma-separated
priority list such as ``ollama,groq``, or ``off`` / ``fallback`` for deterministic responses only.
The router never invents a completion: when every provider fails it raises ``AllProvidersFailed``
and the caller uses the deterministic fallback and says so.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .base import Completion, LLMProvider, ProviderError
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
from .schemas import FALLBACK, LOCAL, ONLINE

COOLDOWN_S = 30.0  # after a provider fails, skip it for this long
AVAIL_TTL_S = 10.0  # cache availability probes (they can hit the network)
KNOWN = {"groq": GroqProvider, "ollama": OllamaProvider}


class AllProvidersFailed(ProviderError):
    pass


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win. Never prints values."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def parse_priority(value: str | None) -> list[str]:
    v = (value or "auto").strip().lower()
    if v in ("off", "none", "fallback", "deterministic", ""):
        return []
    if v == "auto":
        return ["groq", "ollama"]
    names = [n.strip() for n in v.split(",") if n.strip()]
    unknown = [n for n in names if n not in KNOWN]
    if unknown:
        raise ValueError(f"unknown LLM provider {unknown[0]!r} (use groq, ollama, auto or off)")
    return names


class LLMRouter:
    def __init__(self, providers: list[LLMProvider] | None = None, priority: str | None = None):
        if providers is None:
            providers = [KNOWN[n]() for n in parse_priority(priority or os.environ.get("AEROMIND_LLM_PROVIDER"))]
        self.providers = list(providers)
        self._down_until: dict[str, float] = {}
        self._avail: dict[str, tuple[float, bool]] = {}
        self.last_error: str | None = None

    def _is_available(self, p: LLMProvider) -> bool:
        if time.monotonic() < self._down_until.get(p.name, 0.0):
            return False
        hit = self._avail.get(p.name)
        if hit and time.monotonic() - hit[0] < AVAIL_TTL_S:
            return hit[1]
        ok = p.available()
        self._avail[p.name] = (time.monotonic(), ok)
        return ok

    def active(self) -> LLMProvider | None:
        for p in self.providers:
            if self._is_available(p):
                return p
        return None

    def status(self) -> dict:
        """What the UI shows. FALLBACK means no LLM is reachable and rule-based text is used."""
        p = self.active()
        state = FALLBACK if p is None else (ONLINE if p.status == "ONLINE" else LOCAL)
        return {"status": state, "provider": p.name if p else "deterministic",
                "model": p.model if p else "", "priority": [x.name for x in self.providers],
                "last_error": self.last_error}

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> Completion:
        errors = []
        for p in self.providers:
            if not self._is_available(p):
                continue
            try:
                out = p.complete(system, user, json_mode=json_mode)
                self.last_error = None
                return out
            except ProviderError as e:
                self._down_until[p.name] = time.monotonic() + COOLDOWN_S
                errors.append(f"{p.name}: {e}")
        self.last_error = "; ".join(errors) or "no LLM provider available"
        raise AllProvidersFailed(self.last_error)
