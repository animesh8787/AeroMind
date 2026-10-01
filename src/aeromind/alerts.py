"""Structured maintenance advisories: the only thing that leaves the aircraft."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

PRIORITY_RANK = {"ADVISORY": 0, "URGENT": 1, "CRITICAL": 2}

# Illustrative placeholders; a real deployment maps these to the operator's MRO / parts system.
ACTIONS = {
    "bearing_wear": ("Inspect bearing (borescope / chip detector); plan replacement", "bearing assembly kit"),
    "oil_contamination": ("Sample and flush lubrication system; inspect filter", "oil filter + lubricant"),
    "overheating": ("Check cooling path and thermal sensors; reduce load if possible", "cooling-path components"),
    "electrical_fault": ("Inspect power feed and generator/controller harmonics", "power-electronics module"),
    "pressure_leak": ("Leak-check pressure line, seals and fittings", "seal / line kit"),
    "unclassified_anomaly": ("Persistent anomaly, mode unclear: schedule inspection", "none pre-positioned"),
}


@dataclass(frozen=True)
class Advisory:
    component: str
    window: int
    flight_hours: float
    fault_type: str
    confidence: float
    anomaly_score: float
    rul_hours_p10: float
    rul_hours_p50: float
    rul_hours_p90: float
    priority: str
    recommended_action: str
    parts_logistics: str
    contributing_signals: tuple[str, ...]
    evidence: tuple[str, ...] = ()  # computed from the signals (physics.evidence)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contributing_signals"] = list(self.contributing_signals)
        d["evidence"] = list(self.evidence)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @property
    def size_bytes(self) -> int:
        return len(self.to_json().encode())
