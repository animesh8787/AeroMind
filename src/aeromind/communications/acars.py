"""Compact advisory text that fits one ACARS downlink block (220 characters).

Deterministic and round-trip decodable for the fields that matter on the ground; the full
JSON advisory remains available for ground systems when a richer link exists.

Component advisory:
  AMD1/<tail>/<comp>/F<fault>/C<conf%>/P<prio>/R<p10>-<p50>-<p90>/T<flight h>/A<score>/S<sig,sig,sig>/E<evidence>
Sensor advisory:
  AMD1/<tail>/<comp>/SNS/<channel>/<status>/K<check,check>/T<flight h>
"""

from __future__ import annotations

import re

from ..core.alerts import Advisory
from ..core.features import FEATURE_NAMES
from ..edge.sensor_health import CHANNELS, SensorAdvisory

MAX_CHARS = 220
VERSION = "AMD1"

FAULT_CODES = {"bearing_wear": "BRG", "oil_contamination": "OIL", "overheating": "OHT",
               "electrical_fault": "ELC", "pressure_leak": "PRL", "unclassified_anomaly": "UNK"}
PRIO_CODES = {"ADVISORY": "A", "URGENT": "U", "CRITICAL": "C"}
CHANNEL_CODES = {"vibration": "VIB", "acoustic": "ACU", "current": "CUR", "voltage": "VLT",
                 "temperature": "TMP", "pressure": "PRS", "oil_debris": "ODB"}
CHECK_CODES = {"nan": "NAN", "out_of_range": "RNG", "flatline": "FLT", "stuck": "STK",
               "bias": "BIA", "spike": "SPK", "rate": "RTE"}
_inv = lambda d: {v: k for k, v in d.items()}  # noqa: E731
_SAFE = re.compile(r"[^A-Z0-9 .,=+\-%()]")


def _compact(text: str, limit: int) -> str:
    t = text.upper().replace("≈", "=").replace("×", "X").replace("σ", "SD").replace("—", "-")
    return _SAFE.sub("", t)[:limit]


def _comp(component: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", component.upper())[:10]


def encode(adv: Advisory | SensorAdvisory, tail: str) -> str:
    tail = re.sub(r"[^A-Z0-9-]", "", tail.upper())[:8]
    if isinstance(adv, SensorAdvisory):
        checks = ",".join(CHECK_CODES[c] for c in adv.checks)
        msg = (f"{VERSION}/{tail}/{_comp(adv.component)}/SNS/{CHANNEL_CODES[adv.channel]}/"
               f"{adv.status[:3]}/K{checks}/T{adv.flight_hours:.1f}")
        assert len(msg) <= MAX_CHARS
        return msg
    sig = ",".join(str(FEATURE_NAMES.index(s)) for s in adv.contributing_signals[:3])
    head = (f"{VERSION}/{tail}/{_comp(adv.component)}/F{FAULT_CODES[adv.fault_type]}/C{round(adv.confidence * 100)}"
            f"/P{PRIO_CODES[adv.priority]}/R{adv.rul_hours_p10:.0f}-{adv.rul_hours_p50:.0f}-{adv.rul_hours_p90:.0f}"
            f"/T{adv.flight_hours:.1f}/A{adv.anomaly_score:.1f}/S{sig}")
    room = MAX_CHARS - len(head) - 2
    ev = _compact(adv.evidence[0], room) if adv.evidence and room > 0 else ""
    msg = head + (f"/E{ev}" if ev else "")
    assert len(msg) <= MAX_CHARS
    return msg


def decode(msg: str) -> dict:
    parts = msg.split("/")
    if parts[0] != VERSION:
        raise ValueError(f"not an {VERSION} message")
    out = {"tail": parts[1], "component": parts[2]}
    if parts[3] == "SNS":
        out.update(kind="sensor", channel=_inv(CHANNEL_CODES)[parts[4]],
                   status={"FAU": "FAULTY", "REC": "RECOVERED"}[parts[5]],
                   checks=[_inv(CHECK_CODES)[c] for c in parts[6][1:].split(",") if c],
                   flight_hours=float(parts[7][1:]))
        return out
    f = {p[0]: p[1:] for p in parts[3:]}
    p10, p50, p90 = (float(v) for v in f["R"].split("-"))
    out.update(kind="component", fault_type=_inv(FAULT_CODES)[f["F"]], confidence=int(f["C"]) / 100,
               priority=_inv(PRIO_CODES)[f["P"]], rul_hours=(p10, p50, p90), flight_hours=float(f["T"]),
               anomaly_score=float(f["A"]),
               contributing_signals=[FEATURE_NAMES[int(i)] for i in f["S"].split(",") if i],
               evidence=f.get("E", ""))
    return out


assert set(CHANNEL_CODES) == set(CHANNELS)
