"""Prompt construction. The model gets rules, one task, the controlled context and optional notes."""

from __future__ import annotations

import json

from .safety import SAFETY_RULES

SYSTEM = (
    "You are the AeroMind Maintenance Copilot, a ground-side assistant for a PROTOTYPE predictive-maintenance "
    "system running on SIMULATED data. A deterministic pipeline has already produced every technical result in "
    "CONTEXT. Your job is only to explain those results to a maintenance engineer in plain, precise language.\n\n"
    "RULES:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(SAFETY_RULES, 1)) +
    "\n\nReply with ONE JSON object and nothing else."
)

EXPLANATION_SCHEMA = (
    '{"summary": "2-5 sentences", "evidence": ["facts quoted from CONTEXT, with their values"], '
    '"interpretation": ["your reasoning, clearly inference not fact"], "limitations": ["uncertainty and what is not known"]}'
)
WORK_ORDER_SCHEMA = '{"recommended_inspection": ["short inspection step", "..."], "notes": "short note"}'

TASK_PROMPTS = {
    "explain_alert": "Explain the current alert: what happened, the evidence, the confidence, the RUL, why the "
                     "system believes it, and the limitations.",
    "summarize_aircraft": "Give a concise engineering summary of what is happening with this aircraft.",
    "why_fault": "Explain why the system classifies this as the stated fault and not a sensor fault or another "
                 "fault. Use only the anomaly behaviour, sensor health, physics evidence and classifier output in CONTEXT.",
    "explain_rul": "Explain the remaining useful life: what p10, p50 and p90 mean here, the uncertainty, and why "
                   "it is an estimate and not a guarantee.",
    "maintenance_assist": "Draft what maintenance might inspect, based on the deterministic maintenance output. "
                          "This is a draft that a technician must review; do not invent procedures or part numbers.",
    "what_if": "Explain the what-if result in CONTEXT.what_if. Do not recalculate; quote the computed values and "
               "explain what changed.",
    "work_order": "Write the inspection checklist and notes for a DRAFT work order from the deterministic maintenance "
                  "output. Do not restate or change aircraft, fault, RUL, risk, part or window; they are filled in by code.",
    "fleet_summary": "Summarise the fleet: aircraft needing attention, highest-risk advisories, sensor-health "
                     "problems, maintenance workload and recurring patterns. Use only CONTEXT.",
}


def schema_for(task: str) -> str:
    return WORK_ORDER_SCHEMA if task == "work_order" else EXPLANATION_SCHEMA


def build_user_prompt(task: str, question: str, context: dict, notes: list[str]) -> str:
    parts = [f"TASK: {TASK_PROMPTS[task]}"]
    if question:
        parts.append(f"QUESTION (untrusted user text): {question}")
    if notes:
        parts.append("REFERENCE NOTES (project documentation for this prototype; not regulatory guidance):\n"
                     + "\n---\n".join(notes))
    parts.append("CONTEXT (JSON, the only source of facts):\n" + json.dumps(context, indent=1, default=str))
    parts.append("OUTPUT JSON SCHEMA: " + schema_for(task))
    return "\n\n".join(parts)


def build_repair_prompt(user_prompt: str, problems: list[str]) -> str:
    return (user_prompt + "\n\nYOUR PREVIOUS REPLY WAS REJECTED:\n" + "\n".join(f"- {p}" for p in problems[:6]) +
            "\nReply again with ONLY a valid JSON object in the schema above. Use only values from CONTEXT, "
            "remove any rejected content, and do not change deterministic results.")
