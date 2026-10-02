"""Deterministic (rule-based) responses used when no LLM is available or its output is rejected.

These are templates over the same structured context. They are labelled as templates in the UI and
in the response object; they are never presented as LLM output.
"""

from __future__ import annotations

from ..core.alerts import ACTIONS as FAULT_TASKS
from .schemas import AdvisoryExplanation, WorkOrderDraft

SIM_LIMIT = "All data is simulated; this is a prototype policy, not a certified maintenance procedure."


def _fault_name(ctx) -> str:
    return ctx["fault"]["type"].replace("_", " ") if ctx.get("fault") else "no component fault"


def explain_alert(ctx: dict) -> AdvisoryExplanation:
    f, r, m, a = ctx["fault"], ctx["rul_hours"], ctx["maintenance"], ctx["anomaly"]
    sh = ctx["sensor_health"]["faulty_channels"]
    if not f:
        if sh:
            return AdvisoryExplanation(
                f"{ctx['aircraft']} has a sensor fault on {', '.join(sh)}. The affected channel is masked and the "
                "system continues in degraded monitoring. No component fault is classified.",
                [f"Faulty channels: {', '.join(sh)}"], [], [SIM_LIMIT])
        return AdvisoryExplanation(
            f"{ctx['aircraft']} has no component advisory. Anomaly score {a['score']}, "
            f"{a['windows_over_threshold_of_last_8']} of the last 8 windows over the threshold; the advisory gate "
            "opens at 5 of 8.", [f"Status: {ctx['status']}"], [], [SIM_LIMIT])
    evidence = [f"Fault classified: {f['type']} (confidence {f['confidence']:.2f})",
                f"Anomaly score {a['score']}, trend {a['trend']['direction']}", *ctx["physics_evidence"]]
    if r:
        evidence.append(f"RUL p10/p50/p90: {r['p10']:.0f}/{r['p50']:.0f}/{r['p90']:.0f} FH")
    if m:
        evidence.append(f"Failure-before-check probability {m['p_fail_before_check']:.0%}; decision {m['decision']}")
    lim = [SIM_LIMIT, "RUL is an estimate with uncertainty, not a guarantee."]
    if f["type"] == "unclassified_anomaly":
        lim.append("The fault type is not yet clear; confidence is low.")
    return AdvisoryExplanation(
        f"{ctx['aircraft']} shows {_fault_name(ctx)} with confidence {f['confidence']:.2f}. The AeroMind pipeline "
        "raised this after a persistent anomaly and classified it from the sensor features.", evidence, [], lim)


def summarize_aircraft(ctx: dict) -> AdvisoryExplanation:
    base = explain_alert(ctx)
    base.summary = f"{ctx['aircraft']} is {ctx['status']} in {ctx['flight_phase']} at {ctx['flight_hours']} FH. " + base.summary
    return base


def why_fault(ctx: dict) -> AdvisoryExplanation:
    f, sh = ctx["fault"], ctx["sensor_health"]["faulty_channels"]
    if not f:
        return AdvisoryExplanation("No component fault is classified for this aircraft, so there is no "
                                   "classification to explain.", [f"Sensor-health faults: {', '.join(sh) or 'none'}"],
                                   [], [SIM_LIMIT])
    ev = [f"Classifier output: {f['type']} (confidence {f['confidence']:.2f})",
          "Sensor health: " + (f"SENSOR_FAULT on {', '.join(sh)}" if sh else "all channels passed validity checks"),
          f"Anomaly score {ctx['anomaly']['score']}, trend {ctx['anomaly']['trend']['direction']}", *ctx["physics_evidence"]]
    interp = ["Because every sensor channel passed its validity checks, the anomaly is treated as a component "
              "problem rather than a sensor failure." if not sh else
              "A sensor fault is present on a channel; that channel is masked, so the classification relies on the "
              "remaining channels."]
    if not ctx["physics_evidence"]:
        interp.append("No independent physics evidence was computed for this advisory.")
    return AdvisoryExplanation(f"The system classifies this as {_fault_name(ctx)} based on the classifier output, "
                               "sensor health and the evidence listed.", ev, interp, [SIM_LIMIT])


def explain_rul(ctx: dict) -> AdvisoryExplanation:
    r = ctx["rul_hours"]
    if not r:
        return AdvisoryExplanation("No RUL estimate is available: it is produced only while the persistence gate "
                                   "is open.", [], [], [SIM_LIMIT])
    return AdvisoryExplanation(
        f"The median estimate of remaining useful life is {r['p50']:.0f} flight hours (p50). A conservative estimate "
        f"is {r['p10']:.0f} FH (p10) and an optimistic one is {r['p90']:.0f} FH (p90).",
        [f"p10 {r['p10']:.0f} FH, p50 {r['p50']:.0f} FH, p90 {r['p90']:.0f} FH"],
        ["The band p10 to p90 is a nominal 80% interval; planning against p10 is the cautious choice."],
        ["RUL is an estimate with uncertainty, not a guarantee.", SIM_LIMIT])


def maintenance_assist(ctx: dict) -> AdvisoryExplanation:
    m, f = ctx["maintenance"], ctx["fault"]
    if not m or not f:
        return AdvisoryExplanation("There is no component advisory, so there is nothing for maintenance to inspect "
                                   "from the deterministic output.", [], [], [SIM_LIMIT])
    task, _ = FAULT_TASKS.get(f["type"], ("", ""))
    return AdvisoryExplanation(
        f"Deterministic decision: {m['decision']} ({m['window']}). Draft inspection focus for {_fault_name(ctx)}: {task}.",
        [f"Required part (placeholder): {m['required_part']}", *ctx["physics_evidence"]],
        ["Parts availability and maintenance history are not available to this system."],
        ["Draft only; a qualified technician must review and use approved maintenance data.", SIM_LIMIT])


def what_if(ctx: dict) -> AdvisoryExplanation:
    w = ctx.get("what_if")
    if not w:
        return AdvisoryExplanation("A what-if needs a component advisory and a maintenance decision; none exists "
                                   "for this aircraft yet.", [], [], [SIM_LIMIT])
    b, s = w["baseline"], w["scenario"]
    return AdvisoryExplanation(
        f"With the next check moved {w['delay_flight_hours']:g} flight hours later, the failure-before-check "
        f"probability changes from {b['p_fail_before_check']:.0%} to {s['p_fail_before_check']:.0%} and the "
        f"decision is {s['decision']}.",
        [f"Baseline: {b['decision']}, expected cost if deferred ${b['expected_cost_if_deferred']:,}",
         f"Scenario: {s['decision']}, expected cost if deferred ${s['expected_cost_if_deferred']:,}",
         s["rationale"]],
        [], [w["assumption"], "Costs are labelled illustrative assumptions.", SIM_LIMIT])


def fleet_summary(fc: dict) -> AdvisoryExplanation:
    n = len(fc["needs_attention"])
    ev = [f"{a['aircraft']}: {a['status']}" + (f", {a['fault']}" if a["fault"] else "") for a in fc["needs_attention"]]
    ev += [f"Sensor fault {s['aircraft']}: {', '.join(s['channels'])}" for s in fc["sensor_health_problems"]]
    return AdvisoryExplanation(
        f"{n} of {fc['aircraft_total']} simulated aircraft need attention." if n else
        f"All {fc['aircraft_total']} simulated aircraft are without a component advisory or sensor fault.",
        ev, list(fc["recurring_patterns"]), [SIM_LIMIT])


def work_order(ctx: dict, reference: str = "") -> WorkOrderDraft | None:
    m, f, r = ctx["maintenance"], ctx["fault"], ctx["rul_hours"]
    if not (m and f and r):
        return None
    return work_order_with(ctx, [m["inspection_task_placeholder"]],
                           "Draft generated from the deterministic AeroMind advisory; technician review required.",
                           reference)


def work_order_with(ctx: dict, steps: list[str], notes: str, reference: str = "") -> WorkOrderDraft | None:
    m, f, r = ctx["maintenance"], ctx["fault"], ctx["rul_hours"]
    if not (m and f and r):
        return None
    return WorkOrderDraft(
        aircraft=ctx["aircraft"], component=ctx["component"], fault=f["type"], confidence=f["confidence"],
        evidence=list(ctx["physics_evidence"]), rul_hours=dict(r), risk_before_check=m["p_fail_before_check"],
        decision=m["decision"], maintenance_window=m["window"], required_part=m["required_part"],
        recommended_inspection=steps, notes=notes, reference=reference)
