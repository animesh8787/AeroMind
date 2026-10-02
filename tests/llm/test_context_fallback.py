"""Controlled context construction and the deterministic fallback responses."""

import json

import pytest

from aeromind.llm import fallback
from aeromind.llm.context import (add_what_if, build_aircraft_context, build_fleet_context, facts_lines,
                                  fleet_facts_lines, parse_delay_hours)
from aeromind.llm.safety import check_text, collect_numbers
from conftest import RISK_PCT, FakeSource, make_detail


def test_context_has_structured_fields_and_labels_simulation():
    c = build_aircraft_context(make_detail())
    assert c["simulation"]["simulated"] is True and "SIMULATED" in c["simulation"]["statement"]
    assert c["fault"] == {"type": "bearing_wear", "confidence": 0.94}
    assert c["rul_hours"] == {"p10": 38.0, "p50": 61.0, "p90": 90.0}
    assert c["maintenance"]["decision"] == "REPLACE_AT_NEXT_CHECK" and c["maintenance"]["required_part"]
    assert c["anomaly"]["trend"]["direction"] == "rising"
    assert c["cost_assumptions"]["labelled"].startswith("illustrative")


def test_context_excludes_ground_truth_and_raw_history():
    s = json.dumps(build_aircraft_context(make_detail()))
    assert "truth" not in s and "degradation" not in s and "waveform" not in s
    assert '"history"' not in s  # only the trend summary, never the raw score history


def test_context_states_unavailable_data():
    c = build_aircraft_context(make_detail(fault=None, advisory=False))
    assert c["fault"] is None and c["rul_hours"] is None and c["maintenance"] is None
    na = " ".join(c["not_available"])
    assert "maintenance history" in na and "parts availability" in na and "RUL" in na


def test_sensor_fault_status_and_masked_channel():
    c = build_aircraft_context(make_detail(fault=None, advisory=False, faulty_channels=("temperature",)))
    assert c["status"] == "SENSOR_FAULT" and c["sensor_health"]["faulty_channels"] == ["temperature"]


def test_what_if_uses_the_deterministic_decision_engine():
    c = add_what_if(build_aircraft_context(make_detail()), 24.0)
    w = c["what_if"]
    assert w["delay_flight_hours"] == 24.0 and w["baseline"]["hours_to_next_check"] == 40.0
    assert w["scenario"]["hours_to_next_check"] == 64.0
    assert w["scenario"]["p_fail_before_check"] >= w["baseline"]["p_fail_before_check"]
    assert w["scenario"]["decision"] in ("GROUND_NOW", "REPLACE_AT_NEXT_CHECK", "DEFER_AND_MONITOR")
    assert "deterministic" in w["computed_by"]


def test_what_if_without_advisory_is_none():
    assert add_what_if(build_aircraft_context(make_detail(fault=None, advisory=False)), 24)["what_if"] is None


@pytest.mark.parametrize("q,expected", [("what if the check is delayed by 24 hours", 24.0), ("defer 12.5 FH", 12.5),
                                        ("what if we defer?", 24.0)])
def test_parse_delay(q, expected):
    assert parse_delay_hours(q) == expected


def test_fleet_context_is_computed_by_code():
    src = FakeSource([make_detail("VT-AMA03"), make_detail("VT-AMA04"), make_detail("VT-AMA01", None, advisory=False),
                      make_detail("VT-AMA02", None, ("temperature",), advisory=False)])
    fc = build_fleet_context([build_aircraft_context(src.detail(t)) for t in src.tails()])
    assert fc["aircraft_total"] == 4 and fc["status_counts"]["URGENT"] == 2
    assert {a["aircraft"] for a in fc["needs_attention"]} == {"VT-AMA03", "VT-AMA04", "VT-AMA02"}
    assert fc["sensor_health_problems"] == [{"aircraft": "VT-AMA02", "channels": ["temperature"]}]
    assert fc["maintenance_workload"]["decisions"] == {"REPLACE_AT_NEXT_CHECK": 2}
    assert fc["recurring_patterns"] and "bearing_wear on 2 aircraft" in fc["recurring_patterns"][0]


def test_facts_lines_are_deterministic_text():
    lines = facts_lines(build_aircraft_context(make_detail()))
    text = "\n".join(lines)
    assert "Fault: bearing_wear  Confidence: 0.94" in text
    assert "RUL p10 / p50 / p90: 38 / 61 / 90 FH" in text and f"Failure-before-check: {RISK_PCT}" in text
    assert "BPFO" in text


@pytest.mark.parametrize("task", ["explain_alert", "summarize_aircraft", "why_fault", "explain_rul",
                                  "maintenance_assist"])
@pytest.mark.parametrize("kw", [dict(), dict(fault=None, advisory=False),
                                dict(fault=None, advisory=False, faulty_channels=("temperature",))])
def test_every_fallback_passes_the_safety_check(task, kw):
    ctx = build_aircraft_context(make_detail(**kw))
    e = getattr(fallback, task)(ctx)
    faults = {ctx["fault"]["type"]} if ctx["fault"] else set()
    acts = {ctx["maintenance"]["decision"]} if ctx["maintenance"] else set()
    assert not check_text(e.texts(), allowed_numbers=collect_numbers(ctx), known_tails={"VT-AMA03"},
                          allowed_faults=faults, allowed_actions=acts)


def test_fallback_what_if_and_fleet_and_work_order():
    ctx = add_what_if(build_aircraft_context(make_detail()), 24)
    assert "changes from" in fallback.what_if(ctx).summary
    wo = fallback.work_order(ctx)
    assert wo.status == "DRAFT" and wo.aircraft == "VT-AMA03" and wo.rul_hours["p50"] == 61.0
    assert "technician review" in wo.notes.lower()
    assert fallback.work_order(build_aircraft_context(make_detail(fault=None, advisory=False))) is None
    fc = build_fleet_context([ctx])
    assert "need attention" in fallback.fleet_summary(fc).summary
    assert fleet_facts_lines(fc)
