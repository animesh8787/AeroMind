"""Controlled context for the copilot.

The LLM never receives raw sensor history or waveforms. It receives this structured summary of what
the deterministic pipeline already decided, with simulated data labelled as such. The simulator's
hidden ground truth (true RUL, degradation) is deliberately excluded: it does not exist on a real
aircraft and the copilot must not reason from it.
"""

from __future__ import annotations

import re
from dataclasses import asdict

from ..core.alerts import ACTIONS as FAULT_TASKS
from ..core.alerts import Advisory
from ..maintenance.decision import CostAssumptions, DecisionPolicy, Schedule, decide

SIM_STATEMENT = ("SIMULATED: aircraft, sensor data, failures, schedules and costs in this system are simulated. "
                 "Decision thresholds are a prototype policy, not a certified maintenance procedure.")
COMPONENT = "ENG1-GEARBOX"
RECENT_EVENTS = 5


def _trend(scores: list) -> dict:
    s = [v for v in scores if v is not None]
    if len(s) < 12:
        return {"direction": "insufficient_data", "recent_mean": None, "previous_mean": None}
    recent, prev = sum(s[-6:]) / 6, sum(s[-12:-6]) / 6
    d = recent - prev
    return {"direction": "rising" if d > 0.3 else "falling" if d < -0.3 else "flat",
            "recent_mean": round(recent, 2), "previous_mean": round(prev, 2)}


def advisory_from_dict(d: dict) -> Advisory:
    d = dict(d)
    d["contributing_signals"] = tuple(d.get("contributing_signals", ()))
    d["evidence"] = tuple(d.get("evidence", ()))
    return Advisory(**d)


def _recent(events: list[dict], tail: str) -> list[dict]:
    out = []
    for e in events:
        if e.get("tail") != tail or e.get("kind") not in ("component", "sensor"):
            continue
        a = e["advisory"]
        if e["kind"] == "component":
            out.append({"kind": "component", "flight_hours": a["flight_hours"], "fault_type": a["fault_type"],
                        "priority": a["priority"], "confidence": a["confidence"],
                        "decision": (e.get("decision") or {}).get("action")})
        else:
            out.append({"kind": "sensor", "flight_hours": a["flight_hours"], "channel": a["channel"],
                        "status": a["status"], "checks": a["checks"]})
        if len(out) >= RECENT_EVENTS:
            break
    return out


def build_aircraft_context(detail: dict, events: list[dict] | None = None,
                           costs: CostAssumptions | None = None) -> dict:
    """Structured context for one aircraft from ``Aircraft.detail()`` and its recent events."""
    d = detail
    adv, dec = d.get("advisory"), d.get("decision")
    sh = d.get("sensor_health") or {}
    faulty = sorted(c for c, s in sh.items() if not s.get("ok", True))
    status = d["status"]
    if faulty and status == "NOMINAL":
        status = "SENSOR_FAULT"
    unavailable = []
    # An advisory/decision can linger after a reset-free recovery; only the live gate counts as "current".
    fault_now = d.get("fault")
    ctx = {
        "simulation": {"simulated": True, "statement": SIM_STATEMENT},
        "aircraft": d["tail"], "component": COMPONENT, "station_of_next_check": d["schedule"]["check_station"],
        "flight_phase": d.get("phase"), "flight_hours": d.get("fh"), "status": status,
        "anomaly": {"score": d.get("score"), "windows_over_threshold_of_last_8": d.get("flags"),
                    "threshold": 1.0, "gate_rule": "advisory after 5 of the last 8 windows exceed the threshold",
                    "trend": _trend(d.get("history", {}).get("score", []))},
        "fault": {"type": fault_now, "confidence": d.get("confidence")} if fault_now else None,
        "rul_hours": ({"p10": d["rul"][0], "p50": d["rul"][1], "p90": d["rul"][2]} if d.get("rul") else None),
        "sensor_health": {"faulty_channels": faulty,
                          "channels": {c: {"ok": s.get("ok", True), "checks": s.get("checks", [])}
                                       for c, s in sh.items()}},
        "physics_evidence": list(adv["evidence"]) if adv else [],
        "latest_advisory": ({k: adv[k] for k in ("flight_hours", "fault_type", "confidence", "priority",
                                                 "anomaly_score", "recommended_action", "parts_logistics",
                                                 "contributing_signals")} if adv else None),
        "maintenance": None,
        "schedule": d["schedule"],
        "recent_advisories": _recent(events or [], d["tail"]),
        "cost_assumptions": {"labelled": "illustrative assumptions, not airline data",
                             **asdict(costs or CostAssumptions())},
    }
    if dec and adv:
        task, part = FAULT_TASKS.get(adv["fault_type"], ("", ""))
        ctx["maintenance"] = {**{k: dec[k] for k in ("action", "window", "rationale", "p_fail_before_check",
                                                    "p_fail_next_leg", "hours_to_next_check",
                                                    "expected_cost_if_deferred", "cost_if_planned", "policy_note")},
                              "required_part": part, "inspection_task_placeholder": task}
        ctx["maintenance"]["decision"] = ctx["maintenance"].pop("action")
    if not adv:
        unavailable.append("component advisory (no persistent anomaly)")
    if not d.get("rul"):
        unavailable.append("RUL (estimated only while the persistence gate is open)")
    if not ctx["maintenance"]:
        unavailable.append("maintenance decision")
    unavailable += ["maintenance history", "parts availability and stock", "real flight data"]
    ctx["not_available"] = unavailable
    return ctx


def parse_delay_hours(question: str, default: float = 24.0) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:h\b|hr|hrs|hours?|fh)", question or "", re.I)
    return float(m.group(1)) if m else default


def add_what_if(ctx: dict, delay_fh: float, policy: DecisionPolicy | None = None,
                costs: CostAssumptions | None = None) -> dict:
    """Attach a deterministic what-if: the same decision engine with the next check pushed later."""
    if not ctx.get("latest_advisory") or not ctx.get("maintenance"):
        ctx["what_if"] = None
        return ctx
    adv = ctx["latest_advisory"]
    full = advisory_from_dict({
        "component": ctx["component"], "window": 0, "flight_hours": adv["flight_hours"],
        "fault_type": adv["fault_type"], "confidence": adv["confidence"], "anomaly_score": adv["anomaly_score"],
        "rul_hours_p10": ctx["rul_hours"]["p10"], "rul_hours_p50": ctx["rul_hours"]["p50"],
        "rul_hours_p90": ctx["rul_hours"]["p90"], "priority": adv["priority"],
        "recommended_action": adv["recommended_action"], "parts_logistics": adv["parts_logistics"],
        "contributing_signals": adv["contributing_signals"], "evidence": ctx["physics_evidence"]})
    s = ctx["schedule"]
    base_h = s["hours_to_next_check"]
    scenario = decide(full, Schedule(hours_to_next_check=base_h + delay_fh, leg_hours=s["leg_hours"],
                                     check_station=s["check_station"]), policy, costs)
    baseline = decide(full, Schedule(hours_to_next_check=base_h, leg_hours=s["leg_hours"],
                                     check_station=s["check_station"]), policy, costs)
    ctx["what_if"] = {
        "assumption": f"next check moved {delay_fh:g} flight hours later (flight hours, not clock hours)",
        "delay_flight_hours": delay_fh,
        "baseline": {"decision": baseline.action, "hours_to_next_check": base_h,
                     "p_fail_before_check": baseline.p_fail_before_check,
                     "expected_cost_if_deferred": baseline.expected_cost_if_deferred,
                     "cost_if_planned": baseline.cost_if_planned},
        "scenario": {"decision": scenario.action, "hours_to_next_check": round(base_h + delay_fh, 1),
                     "p_fail_before_check": scenario.p_fail_before_check,
                     "p_fail_next_leg": scenario.p_fail_next_leg, "rationale": scenario.rationale,
                     "expected_cost_if_deferred": scenario.expected_cost_if_deferred,
                     "cost_if_planned": scenario.cost_if_planned},
        "computed_by": "deterministic decision engine (same code as the ground station)",
    }
    return ctx


def build_fleet_context(contexts: list[dict]) -> dict:
    """Fleet facts computed by code from per-aircraft contexts. Nothing here is generated by a model."""
    counts: dict[str, int] = {}
    for c in contexts:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    attention, sensor_problems, actions, parts, by_fault = [], [], {}, {}, {}
    for c in contexts:
        m, f = c.get("maintenance"), c.get("fault")
        if f or c["status"] not in ("NOMINAL",):
            if f or c["status"] in ("SENSOR_FAULT",):
                attention.append({"aircraft": c["aircraft"], "status": c["status"],
                                  "fault": f["type"] if f else None,
                                  "confidence": f["confidence"] if f else None,
                                  "rul_p50": c["rul_hours"]["p50"] if c["rul_hours"] else None,
                                  "decision": m["decision"] if m else None,
                                  "risk_before_check": m["p_fail_before_check"] if m else None})
        if c["sensor_health"]["faulty_channels"]:
            sensor_problems.append({"aircraft": c["aircraft"], "channels": c["sensor_health"]["faulty_channels"]})
        if m:
            actions[m["decision"]] = actions.get(m["decision"], 0) + 1
            parts[m["required_part"]] = parts.get(m["required_part"], 0) + 1
        if f:
            by_fault.setdefault(f["type"], []).append(c["aircraft"])
    attention.sort(key=lambda a: -(a["risk_before_check"] or 0.0))
    patterns = [f"{ft} on {len(t)} aircraft ({', '.join(t)})" for ft, t in by_fault.items() if len(t) > 1]
    return {"simulation": {"simulated": True, "statement": SIM_STATEMENT}, "aircraft_total": len(contexts),
            "status_counts": counts, "needs_attention": attention, "sensor_health_problems": sensor_problems,
            "maintenance_workload": {"decisions": actions, "parts_needed": parts},
            "recurring_patterns": patterns,
            "not_available": ["maintenance history", "parts availability and stock", "real flight data"]}


def facts_lines(ctx: dict) -> list[str]:
    """The deterministic AeroMind output as plain lines (shown verbatim in the UI)."""
    f, r, m, a = ctx["fault"], ctx["rul_hours"], ctx["maintenance"], ctx["anomaly"]
    lines = [f"Aircraft: {ctx['aircraft']} ({ctx['component']}, simulated)",
             f"Status: {ctx['status']}  Phase: {ctx['flight_phase']}  Flight hours: {ctx['flight_hours']}",
             f"Anomaly score: {a['score']} ({a['windows_over_threshold_of_last_8']} of last 8 windows over "
             f"{a['threshold']}; trend {a['trend']['direction']})"]
    lines.append(f"Fault: {f['type']}  Confidence: {f['confidence']:.2f}" if f else "Fault: none classified")
    if r:
        lines.append(f"RUL p10 / p50 / p90: {r['p10']:.0f} / {r['p50']:.0f} / {r['p90']:.0f} FH")
    else:
        lines.append("RUL: not available (persistence gate closed)")
    sh = ctx["sensor_health"]["faulty_channels"]
    lines.append("Sensor health: " + (("SENSOR_FAULT on " + ", ".join(sh) + " (masked)") if sh else "all 7 channels valid"))
    for e in ctx["physics_evidence"]:
        lines.append("Evidence: " + e)
    if m:
        lines.append(f"Failure-before-check: {m['p_fail_before_check']:.0%}  next leg: {m['p_fail_next_leg']:.0%}")
        lines.append(f"Decision: {m['decision']} — {m['window']}")
        lines.append(f"Required part (placeholder): {m['required_part']}")
    wi = ctx.get("what_if")
    if wi:
        b, s = wi["baseline"], wi["scenario"]
        lines.append(f"What-if ({wi['assumption']}): decision {s['decision']}, failure-before-check "
                     f"{s['p_fail_before_check']:.0%} (baseline {b['p_fail_before_check']:.0%}), expected cost if "
                     f"deferred ${s['expected_cost_if_deferred']:,} (baseline ${b['expected_cost_if_deferred']:,})")
    return lines


def fleet_facts_lines(fc: dict) -> list[str]:
    lines = [f"Aircraft in fleet: {fc['aircraft_total']} (simulated)",
             "Status: " + ", ".join(f"{k} {v}" for k, v in sorted(fc["status_counts"].items()))]
    for a in fc["needs_attention"]:
        bits = [a["status"], a["fault"] or "sensor fault only"]
        if a["rul_p50"] is not None:
            bits.append(f"RUL p50 {a['rul_p50']:.0f} FH")
        if a["risk_before_check"] is not None:
            bits.append(f"risk before check {a['risk_before_check']:.0%}")
        if a["decision"]:
            bits.append(a["decision"])
        lines.append(f"Attention {a['aircraft']}: " + ", ".join(bits))
    for s in fc["sensor_health_problems"]:
        lines.append(f"Sensor fault {s['aircraft']}: {', '.join(s['channels'])}")
    w = fc["maintenance_workload"]
    if w["decisions"]:
        lines.append("Workload: " + ", ".join(f"{k} x{v}" for k, v in w["decisions"].items()))
    for p in fc["recurring_patterns"]:
        lines.append("Pattern: " + p)
    if not fc["needs_attention"]:
        lines.append("No aircraft currently needs attention.")
    return lines
