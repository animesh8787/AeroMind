"""Safety rules: invented values, certification claims, control commands, overridden decisions."""

import pytest

from aeromind.llm.safety import SAFETY_RULES, check_text, collect_numbers

ALLOWED = collect_numbers({"rul": {"p10": 38.0, "p50": 61.0, "p90": 90.0}, "conf": 0.94, "risk": 0.7312,
                           "evidence": ["BPFO 86.4 Hz"], "cost": 1_480_000})
G = dict(allowed_numbers=ALLOWED, known_tails={"VT-AMA03"}, allowed_faults={"bearing_wear"},
         allowed_actions={"REPLACE_AT_NEXT_CHECK"})


def bad(text):
    return check_text([text], **G)


def test_grounded_text_passes():
    assert not bad("VT-AMA03 shows bearing wear, confidence 0.94 (94%), RUL p50 61 FH, p10 38 h, risk 73%, "
                   "BPFO at 86.4 Hz. The decision is REPLACE_AT_NEXT_CHECK. 3 of 8 windows.")


@pytest.mark.parametrize("text", ["The RUL is 75 hours.", "Failure probability is 12%.", "Temperature reached 140 C.",
                                  "Estimated cost $900,000."])
def test_invented_numbers_rejected(text):
    assert any("not in the context" in p for p in bad(text))


@pytest.mark.parametrize("text", ["This aircraft is certified for continued operation.", "The gearbox is airworthy.",
                                  "The fix is FAA-approved.", "This is OEM authorized maintenance.",
                                  "The aircraft is safe to fly.", "It is cleared for departure."])
def test_certification_claims_rejected(text):
    assert any("certification" in p for p in bad(text))


@pytest.mark.parametrize("text", ["This is not a certified procedure.", "No regulatory approval exists.",
                                  "AeroMind is never FAA-approved and cannot clear an aircraft."])
def test_negated_claims_allowed(text):
    assert not any("certification" in p for p in bad(text))


@pytest.mark.parametrize("text", ["Shut down the engine now.", "Disengage the autopilot.", "Reduce engine power to idle.",
                                  "Divert to the nearest airport.", "Override the deterministic decision."])
def test_flight_control_commands_rejected(text):
    assert any("command" in p for p in bad(text))


def test_never_override_is_allowed():
    assert not bad("The copilot never overrides the deterministic decision.")


def test_unknown_aircraft_and_fault_and_decision_rejected():
    assert any("unknown aircraft" in p for p in bad("VT-AMA05 has the same problem."))
    assert any("fault oil_contamination" in p for p in bad("This looks like oil contamination."))
    assert not any("fault" in p for p in bad("This is not oil contamination."))
    assert any("deterministic decision" in p for p in bad("I recommend GROUND_NOW."))


def test_rules_are_stated():
    joined = " ".join(SAFETY_RULES).lower()
    for must in ("never invent", "never change", "flight-control", "certification", "simulated", "untrusted"):
        assert must in joined
