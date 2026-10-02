"""Ground-side LLM maintenance copilot (explains deterministic AeroMind output; never replaces it)."""

from .base import Completion, LLMProvider, ProviderError
from .copilot import Copilot, infer_task
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider
from .router import AllProvidersFailed, LLMRouter, load_dotenv
from .schemas import (AdvisoryExplanation, CopilotRequest, CopilotResponse, FleetSummary, WorkOrderDraft)

__all__ = ["AdvisoryExplanation", "AllProvidersFailed", "Completion", "Copilot", "CopilotRequest", "CopilotResponse",
           "FleetSummary", "GroqProvider", "LLMProvider", "LLMRouter", "OllamaProvider", "ProviderError",
           "WorkOrderDraft", "infer_task", "load_dotenv"]
