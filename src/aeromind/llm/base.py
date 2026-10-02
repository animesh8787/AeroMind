"""LLM provider interface. Providers only move text; they never see raw sensor history."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """A provider could not produce a completion (network, auth, timeout, bad reply)."""


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str


class LLMProvider(ABC):
    name: str = "base"
    #: UI status when this provider answers
    status: str = "ONLINE"

    def __init__(self, model: str, timeout: float = 30.0):
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def configured(self) -> bool:
        """True when the provider has what it needs (e.g. an API key). Never touches the network."""

    @abstractmethod
    def available(self) -> bool:
        """True when a request is likely to succeed right now."""

    @abstractmethod
    def complete(self, system: str, user: str, *, json_mode: bool = True) -> Completion:
        """One chat completion. Raises ``ProviderError`` on any failure."""

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model, "configured": self.configured()}
