"""The maintenance copilot: structured context in, validated explanation out.

Flow for every request:
  1. build the controlled context from deterministic outputs (and compute any what-if with the
     deterministic decision engine);
  2. ask an LLM provider for a JSON explanation (Groq -> Ollama);
  3. validate the structure and run the safety checks; on failure retry once with a repair prompt;
  4. if there is no provider, or the reply is still rejected, return the rule-based template.

The response always carries the deterministic facts separately from the explanation text, and says
whether the text was written by a model.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from . import fallback
from .base import ProviderError
from .context import (add_what_if, build_aircraft_context, build_fleet_context, facts_lines, fleet_facts_lines,
                      parse_delay_hours)
from .knowledge import KnowledgeBase
from .prompts import SYSTEM, build_repair_prompt, build_user_prompt
from .router import LLMRouter
from .safety import check_text, collect_numbers
from .schemas import (AI_LABEL, FALLBACK, TASKS, TEMPLATE_LABEL, UNAVAILABLE, AdvisoryExplanation, CopilotRequest,
                      CopilotResponse, FleetSummary, SchemaError, parse_work_order_parts)

_FAULT_WORDS = ("bearing_wear", "oil_contamination", "overheating", "electrical_fault", "pressure_leak")


class FleetSource(Protocol):
    """What the copilot needs from the ground station (kept as a protocol so tests can fake it)."""

    def tails(self) -> list[str]: ...
    def detail(self, tail: str) -> dict: ...
    def events(self, tail: str) -> list[dict]: ...


def infer_task(question: str) -> str:
    q = (question or "").lower()
    rules = (
        ("work_order", r"work ?order|draft (a )?(wo|order)"),
        ("what_if", r"what if|defer|delay|postpone|push (it|back)"),
        ("fleet_summary", r"\bfleet\b|all aircraft|which aircraft"),
        ("maintenance_assist", r"inspect|check|replace|maintenance should|what should|repair"),
        ("explain_rul", r"\brul\b|useful life|how (long|much).*(life|left|remain)|p10|p50|p90"),
        ("why_fault", r"why .*(fault|bearing|sensor)|rather than|instead of|sensor fault|classif"),
        ("summarize_aircraft", r"summar|what('?s| is) happening|status|overview"),
    )
    for task, pat in rules:
        if re.search(pat, q):
            return task
    return "explain_alert"


def _parse_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except ValueError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                pass
    raise SchemaError("reply is not valid JSON")


class Copilot:
    def __init__(self, router: LLMRouter | None = None, knowledge: KnowledgeBase | None = None):
        self.router = router if router is not None else LLMRouter()
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase()

    # ------------------------------------------------------------------ public
    def status(self) -> dict:
        return self.router.status()

    def ask(self, req: CopilotRequest, source: FleetSource) -> CopilotResponse:
        task = req.task or infer_task(req.question)
        if task not in TASKS:
            raise ValueError(f"unknown copilot task {task!r}")
        try:
            return self._answer(task, req, source)
        except (KeyError, ValueError):
            raise
        except Exception as e:  # the copilot must never take the ground station down
            return CopilotResponse(task, req.tail, {"title": "DETERMINISTIC AEROMIND OUTPUT", "lines": []},
                                   f"The copilot hit an internal error ({type(e).__name__}). The deterministic "
                                   "AeroMind output is unaffected.", False, "deterministic", "", UNAVAILABLE,
                                   TEMPLATE_LABEL, warnings=[f"internal error: {type(e).__name__}"])

    # ------------------------------------------------------------------ internals
    def _answer(self, task: str, req: CopilotRequest, source: FleetSource) -> CopilotResponse:
        question = req.cleaned_question()
        tails = list(source.tails())
        if task == "fleet_summary":
            ctxs = [build_aircraft_context(source.detail(t), source.events(t)) for t in tails]
            context = build_fleet_context(ctxs)
            facts = fleet_facts_lines(context)
            title, tail = "DETERMINISTIC FLEET STATE", "FLEET"
            faults = {a["fault"] for a in context["needs_attention"] if a["fault"]}
            actions = {a["decision"] for a in context["needs_attention"] if a["decision"]}
        else:
            if req.tail not in tails:
                raise KeyError(f"unknown aircraft {req.tail!r}")
            context = build_aircraft_context(source.detail(req.tail), source.events(req.tail))
            if task == "what_if":
                delay = req.hours_to_next_check if req.hours_to_next_check is not None else parse_delay_hours(question)
                add_what_if(context, max(0.0, float(delay)))
            facts = facts_lines(context)
            title, tail = "DETERMINISTIC AEROMIND OUTPUT", req.tail
            faults = {context["fault"]["type"]} if context["fault"] else set()
            faults |= {e["fault_type"] for e in context["recent_advisories"] if e["kind"] == "component"}
            if context["latest_advisory"]:
                faults.add(context["latest_advisory"]["fault_type"])
            actions = {context["maintenance"]["decision"]} if context["maintenance"] else set()
            actions |= {e["decision"] for e in context["recent_advisories"] if e.get("decision")}
            if context.get("what_if"):
                actions |= {context["what_if"]["baseline"]["decision"], context["what_if"]["scenario"]["decision"]}
        deterministic = {"title": title, "lines": facts}
        ref = ""
        if task != "fleet_summary" and context.get("maintenance"):
            ref = f"{req.tail}-DRAFT"

        notes = [f"[{c.source} / {c.heading}] {c.text}" for c in self.knowledge.retrieve(
            f"{question} {task.replace('_', ' ')} {' '.join(faults)}", k=2)]
        # Cost-assumption numbers (e.g. 12 AOG hours) are only valid on cost-related tasks; elsewhere they
        # would let an invented "12 hours" pass the grounding check.
        numeric = {k: v for k, v in context.items()
                   if k != "cost_assumptions" or task in ("what_if", "maintenance_assist", "work_order")}
        allowed = collect_numbers(numeric) | collect_numbers(question) | collect_numbers(notes)
        rul = context.get("rul_hours") if task != "fleet_summary" else None
        rul_values = collect_numbers(rul) if rul else None
        if task == "fleet_summary":
            rul_values = {a["rul_p50"] for a in context["needs_attention"] if a["rul_p50"] is not None}
        guard = dict(allowed_numbers=allowed, known_tails=set(tails), allowed_faults=faults, allowed_actions=actions,
                     rul_values=rul_values)

        warnings: list[str] = []
        result = None
        if self.router.active() is not None:
            result, warnings = self._ask_llm(task, question, context, notes, guard)
        else:
            warnings.append("No LLM provider available; showing the rule-based template.")

        st = self.router.status()
        if result is not None:
            expl, parts, provider, model = result
            ai, label, prov = True, AI_LABEL, provider
        else:
            expl, parts, provider, model = None, None, "deterministic", ""
            ai, label, prov = False, TEMPLATE_LABEL, "deterministic"
        return self._package(task, tail, deterministic, context, expl, parts, ref, ai, prov, model,
                             st["status"], label, warnings)

    def _ask_llm(self, task, question, context, notes, guard):
        user = build_user_prompt(task, question, context, notes)
        problems: list[str] = []
        for attempt in range(2):
            prompt = user if attempt == 0 else build_repair_prompt(user, problems)
            try:
                comp = self.router.complete(SYSTEM, prompt, json_mode=True)
            except ProviderError as e:
                return None, [f"LLM unavailable: {e}. Showing the rule-based template."]
            try:
                data = _parse_json(comp.text)
                if task == "work_order":
                    steps, note = parse_work_order_parts(data)
                    texts, expl, parts = steps + [note], None, (steps, note)
                else:
                    expl = AdvisoryExplanation.from_dict(data)
                    texts, parts = expl.texts(), None
                problems = check_text(texts, **guard)
            except SchemaError as e:
                problems = [f"malformed output: {e}"]
            if not problems:
                return (expl, parts, comp.provider, comp.model), []
        return None, ["Model output was rejected after one repair attempt (" + "; ".join(problems[:3]) +
                      "). Showing the rule-based template."]

    def _package(self, task, tail, deterministic, context, expl, parts, ref, ai, provider, model, status, label,
                 warnings) -> CopilotResponse:
        structured = None
        if task == "work_order":
            wo = (fallback.work_order_with(context, *parts, reference=ref) if parts
                  else fallback.work_order(context, reference=ref))
            if wo is None:
                text = "No component advisory with a maintenance decision exists, so no work order can be drafted."
            else:
                structured, text = wo.to_dict(), wo.render()
        elif task == "fleet_summary":
            expl = expl or fallback.fleet_summary(context)
            structured = FleetSummary(context["aircraft_total"], context["status_counts"], context["needs_attention"],
                                      context["sensor_health_problems"], context["maintenance_workload"],
                                      context["recurring_patterns"], expl).to_dict()
            text = expl.render()
        else:
            expl = expl or getattr(fallback, task)(context)
            text = expl.render()
        return CopilotResponse(task, tail, deterministic, text, ai, provider, model, status, label,
                               structured=structured, warnings=warnings)
