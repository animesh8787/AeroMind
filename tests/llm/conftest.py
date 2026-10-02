"""Fixtures for the copilot tests: a fake fleet source and scripted providers. No network, no API key."""

import copy

import pytest

from aeromind.llm.context import advisory_from_dict
from aeromind.maintenance.decision import Schedule, decide
from aeromind.llm.base import Completion, LLMProvider, ProviderError
from aeromind.llm.copilot import Copilot
from aeromind.llm.knowledge import KnowledgeBase
from aeromind.llm.router import LLMRouter

CHANNELS = ("vibration", "acoustic", "current", "voltage", "temperature", "pressure", "oil_debris")

ADVISORY = {
    "component": "ENG1-GEARBOX", "window": 120, "flight_hours": 60.0, "fault_type": "bearing_wear",
    "confidence": 0.94, "anomaly_score": 4.2, "rul_hours_p10": 38.0, "rul_hours_p50": 61.0, "rul_hours_p90": 90.0,
    "priority": "URGENT", "recommended_action": "Inspect bearing (borescope / chip detector); plan replacement",
    "parts_logistics": "bearing assembly kit", "contributing_signals": ["vib_kurtosis", "oil_mean", "temp_mean"],
    "evidence": ["envelope spectrum peak at BPFO 86.4 Hz (3.1x noise floor)"],
}
SCHEDULE = {"hours_to_next_check": 40.0, "leg_hours": 2.0, "check_station": "BLR"}
DECISION = decide(advisory_from_dict(ADVISORY), Schedule(**SCHEDULE)).to_dict()  # same engine as the ground station
RISK_PCT = f"{DECISION['p_fail_before_check']:.0%}"


def make_detail(tail="VT-AMA03", fault="bearing_wear", faulty_channels=(), advisory=True):
    sh = {c: {"ok": c not in faulty_channels, "checks": ["stuck"] if c in faulty_channels else []} for c in CHANNELS}
    adv = dict(ADVISORY, fault_type=fault) if advisory else None
    has = advisory and fault is not None
    return {
        "tail": tail, "station": "BLR", "status": "URGENT" if has else "NOMINAL", "phase": "cruise", "fh": 60.0,
        "score": 4.2 if has else 0.3, "flags": 8 if has else 0, "fault": fault if has else None,
        "confidence": 0.94 if has else None, "rul": [38.0, 61.0, 90.0] if has else None,
        "sensor_faults": list(faulty_channels),
        "history": {"score": [0.4] * 12 + [3.0] * 6 + [4.2] * 6},
        "sensor_health": sh, "advisory": adv if has else None, "decision_detail": None,
        "schedule": dict(SCHEDULE),
        "work_orders": [], "truth": {"mode": "bearing_wear", "degradation": 0.4, "rul_hours": 59.0},
        "decision": dict(DECISION) if has else None,
    }


class FakeSource:
    def __init__(self, details):
        self._d = {d["tail"]: d for d in details}

    def tails(self):
        return list(self._d)

    def detail(self, tail):
        return copy.deepcopy(self._d[tail])

    def events(self, tail):
        d = self._d[tail]
        if not d.get("advisory"):
            return []
        return [{"tail": tail, "kind": "component", "advisory": d["advisory"], "decision": d["decision"]}]


class ScriptedProvider(LLMProvider):
    """Returns queued replies; records every prompt it was given."""

    name, status = "groq", "ONLINE"

    def __init__(self, replies, model="mock-model"):
        super().__init__(model)
        self.replies, self.calls = list(replies), []

    def configured(self):
        return True

    def available(self):
        return True

    def complete(self, system, user, *, json_mode=True):
        self.calls.append((system, user))
        r = self.replies.pop(0) if self.replies else None
        if isinstance(r, Exception):
            raise r
        if r is None:
            raise ProviderError("no scripted reply")
        return Completion(r, self.name, self.model)


GOOD_EXPLANATION = (
    '{"summary": "VT-AMA03 shows bearing wear with confidence 0.94. The decision is REPLACE_AT_NEXT_CHECK.",'
    ' "evidence": ["Fault classified: bearing_wear, confidence 0.94", "RUL p10/p50/p90 is 38/61/90 FH",'
    ' "Envelope spectrum peak at BPFO 86.4 Hz"], "interpretation": ["The BPFO line supports a bearing outer-race'
    ' defect; this is inference from the stated evidence."], "limitations": ["Simulated data; RUL is an estimate,'
    ' not a guarantee."]}')


@pytest.fixture
def source():
    return FakeSource([make_detail("VT-AMA03"), make_detail("VT-AMA01", fault=None, advisory=False),
                       make_detail("VT-AMA02", fault=None, advisory=False, faulty_channels=("temperature",))])


@pytest.fixture
def kb(tmp_path):
    (tmp_path / "rul.md").write_text("# RUL\n\nRUL p10 p50 p90 are quantiles of remaining useful life in flight hours. "
                                     "It is an estimate, not a guarantee.\n", encoding="utf-8")
    return KnowledgeBase(tmp_path)


def make_copilot(replies, kb, provider_cls=ScriptedProvider):
    prov = provider_cls(replies)
    return Copilot(LLMRouter(providers=[prov]), kb), prov


@pytest.fixture
def copilot_factory(kb):
    return lambda replies: make_copilot(replies, kb)
