"""Advisories from a remote edge agent (e.g. Raspberry Pi 5) reaching the ground station.

The edge device runs the same ``EdgePipeline`` and sends the advisory JSON it already produces.
The ground station then does what it does for simulated aircraft: maintenance decision, work-order
draft, ACARS-sized text and an entry in the event log. The remote device is shown in the log and is
available to the copilot; it is not part of the six simulated aircraft.

``AEROMIND_INGEST_TOKEN``: when set, requests must send it in ``X-AeroMind-Token``.
"""

from __future__ import annotations

import re
import time

from ..communications.acars import encode
from ..edge.sensor_health import CHANNELS, SensorAdvisory
from ..llm.context import advisory_from_dict
from ..maintenance.decision import Schedule, decide, work_order

CHECK_INTERVAL_FH = 12.0
TAIL = re.compile(r"^[A-Z0-9-]{3,12}$")


class IngestError(ValueError):
    pass


class RemoteAircraft:
    """The last known state of one remote edge device, in the shape the copilot context expects."""

    def __init__(self, tail: str):
        self.tail = tail
        self.advisory: dict | None = None
        self.decision: dict | None = None
        self.sensor_faults: dict[str, list[str]] = {}
        self.fh = 0.0
        self.events: list[dict] = []
        self.last_seen = 0.0
        self.model_version = ""

    def detail(self) -> dict:
        a, d = self.advisory, self.decision
        faulty = self.sensor_faults
        status = a["priority"] if a else "NOMINAL"
        return {
            "tail": self.tail, "station": "BLR", "status": status, "phase": "remote (not reported)", "fh": self.fh,
            "score": a["anomaly_score"] if a else 0.0, "flags": 8 if a else 0,
            "fault": a["fault_type"] if a else None, "confidence": a["confidence"] if a else None,
            "rul": [a["rul_hours_p10"], a["rul_hours_p50"], a["rul_hours_p90"]] if a else None,
            "sensor_faults": sorted(faulty), "history": {"score": []},
            "sensor_health": {c: {"ok": c not in faulty, "checks": faulty.get(c, [])} for c in CHANNELS},
            "advisory": a, "decision": d,
            "schedule": {"hours_to_next_check": round(CHECK_INTERVAL_FH - self.fh % CHECK_INTERVAL_FH, 1),
                         "leg_hours": 2.0, "check_station": "BLR"},
            "work_orders": [], "remote": True,
        }

    def summary(self) -> dict:
        return {"tail": self.tail, "status": (self.advisory or {}).get("priority", "NOMINAL"),
                "fh": self.fh, "fault": (self.advisory or {}).get("fault_type"),
                "sensor_faults": sorted(self.sensor_faults), "model_version": self.model_version,
                "last_seen_s_ago": round(time.time() - self.last_seen, 1) if self.last_seen else None}


class RemoteRegistry:
    def __init__(self):
        self.aircraft: dict[str, RemoteAircraft] = {}

    def ingest(self, body: dict) -> dict:
        """Validate one advisory from an edge agent. Returns the event to log."""
        tail = str(body.get("tail", "")).upper()
        if not TAIL.match(tail):
            raise IngestError("tail must be 3-12 characters of A-Z, 0-9 or -")
        kind, adv = body.get("kind"), body.get("advisory")
        if kind not in ("component", "sensor") or not isinstance(adv, dict):
            raise IngestError("kind must be 'component' or 'sensor' and advisory an object")
        ac = self.aircraft.setdefault(tail, RemoteAircraft(tail))
        ac.last_seen, ac.model_version = time.time(), str(body.get("model_version", ""))[:32]
        now = time.time()
        try:
            if kind == "sensor":
                sa = SensorAdvisory(**{**adv, "checks": tuple(adv.get("checks", ()))})
                if sa.channel not in CHANNELS:
                    raise IngestError(f"unknown channel {sa.channel!r}")
                ac.fh = max(ac.fh, sa.flight_hours)
                if sa.status == "FAULTY":
                    ac.sensor_faults[sa.channel] = list(sa.checks)
                else:
                    ac.sensor_faults.pop(sa.channel, None)
                ev = {"time": now, "tail": tail, "kind": "sensor", "advisory": sa.to_dict(), "acars": encode(sa, tail),
                      "remote": True}
            else:
                full = advisory_from_dict(adv)
                ac.fh = max(ac.fh, full.flight_hours)
                dec = decide(full, Schedule(hours_to_next_check=round(CHECK_INTERVAL_FH - ac.fh % CHECK_INTERVAL_FH, 1),
                                            leg_hours=2.0, check_station="BLR"))
                wo = work_order(tail, full, dec)
                ac.advisory, ac.decision = full.to_dict(), dec.to_dict()
                ev = {"time": now, "tail": tail, "kind": "component", "advisory": ac.advisory,
                      "decision": ac.decision, "work_order": wo.number, "acars": wo.acars, "remote": True}
        except (TypeError, KeyError, ValueError, AssertionError) as e:
            if isinstance(e, IngestError):
                raise
            raise IngestError(f"invalid advisory ({type(e).__name__})") from None
        ac.events.insert(0, ev)
        del ac.events[30:]
        return ev
