"""Simulated fleet for the live ground station: one LiveAircraft + EdgePipeline per tail."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..acars import encode
from ..config import FAULT_MODES, HEALTHY, HOURS_PER_WINDOW
from ..decision import Schedule, decide, work_order
from ..features import FEATURE_NAMES
from ..physics import GEOMETRY, envelope_spectrum, shaft_hz
from ..pipeline import EdgePipeline
from ..sensor_health import CHANNELS, SENSOR_FAULT_KINDS, SensorFault, SensorFaultInjector
from ..simulator import PHASES, LiveAircraft

TAILS = tuple(f"VT-AMA{i:02d}" for i in range(1, 7))  # fictional registrations
HISTORY = 240
CHECK_INTERVAL_FH = 12.0  # an overnight check every 12 flight hours (prototype schedule)
STATIONS = ("BLR", "DEL", "BOM", "HYD", "MAA", "CCU")
SENSOR_SUMMARY = {"vib_rms": "g", "ac_rms": "RMS", "cur_thd": "%", "volt_mean": "V",
                  "temp_mean": "°C", "press_mean": "psi", "oil_mean": "ppm"}

# Demo shortcuts for the four sensor-failure buttons.
SENSOR_PRESETS = {
    "temperature_failure": ("stuck", "temperature"),
    "vibration_drift": ("drift", "vibration"),
    "pressure_spike": ("spike", "pressure"),
    "sensor_dropout": ("dropout", "oil_debris"),
}


def _r(v, nd=3):
    return None if v is None else float(round(float(v), nd))


@dataclass
class Aircraft:
    tail: str
    seed: int
    bundle: object
    station: str
    phase_offset: int = 0
    sim: LiveAircraft = field(init=False)
    pipe: EdgePipeline = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.sim = LiveAircraft(self.seed, phase_offset=self.phase_offset)
        self.pipe = EdgePipeline(self.bundle, component="ENG1-GEARBOX")
        self.injectors: list[SensorFaultInjector] = []
        self.hist = {k: deque(maxlen=HISTORY) for k in
                     ("t", "score", "flags", "p10", "p50", "p90", "truth", "phase", *SENSOR_SUMMARY)}
        self.events: deque = deque(maxlen=30)
        self.work_orders: deque = deque(maxlen=10)
        self.decision = None
        self.last_advisory = None
        self.window = None
        self.truth = None

    def swap_bundle(self, bundle):
        """New model version: keep the aircraft and its history, restart the pipeline state."""
        self.bundle = bundle
        self.pipe = EdgePipeline(bundle, component="ENG1-GEARBOX")

    def schedule(self) -> Schedule:
        fh = self.sim.t * HOURS_PER_WINDOW
        return Schedule(hours_to_next_check=round(CHECK_INTERVAL_FH - fh % CHECK_INTERVAL_FH, 1),
                        leg_hours=2.0, check_station=self.station)

    def step(self) -> list[dict]:
        """Advance one window; return the new events (advisories) it produced."""
        w, truth = self.sim.step()
        for inj in self.injectors:
            w = inj.apply(w)
        adv = self.pipe.process(w)
        self.window, self.truth = w, truth
        a, x, h = self.pipe.last_assessment, self.pipe.last_features, self.hist
        h["t"].append(w.t)
        h["score"].append(_r(self.pipe.last_score, 2))
        h["flags"].append(self.pipe.last_flags)
        for k, q in zip(("p10", "p50", "p90"), a.rul_hours if a else (None,) * 3):
            h[k].append(_r(q, 1))
        h["truth"].append(None if truth.rul_windows is None else truth.rul_windows * HOURS_PER_WINDOW)
        h["phase"].append(w.phase)
        for k in SENSOR_SUMMARY:
            v = x[FEATURE_NAMES.index(k)]
            h[k].append(_r(v * 100 if k == "cur_thd" else v, 2))
        now, new = time.time(), []
        for sa in self.pipe.last_sensor_advisories:
            new.append({"time": now, "tail": self.tail, "kind": "sensor", "advisory": sa.to_dict(),
                        "acars": encode(sa, self.tail)})
        if adv is not None:
            self.last_advisory = adv
            self.decision = decide(adv, self.schedule())
            wo = work_order(self.tail, adv, self.decision)
            self.work_orders.appendleft(wo.to_dict())
            new.append({"time": now, "tail": self.tail, "kind": "component", "advisory": adv.to_dict(),
                        "decision": self.decision.to_dict(), "work_order": wo.number, "acars": wo.acars})
        elif a is None and not self.pipe.last_flags:
            self.decision = None
        self.events.extendleft(new)
        return new

    def status(self) -> str:
        a = self.pipe.last_assessment
        if a is not None:
            return a.priority
        # WATCH = an anomaly building (3+ of the last 8 windows over threshold); isolated
        # single-window excursions (about 2% of healthy windows) stay NOMINAL.
        return "WATCH" if self.pipe.last_flags >= 3 else "NOMINAL"

    def summary(self) -> dict:
        a, w = self.pipe.last_assessment, self.window
        return {
            "tail": self.tail, "station": self.station, "status": self.status(),
            "phase": w.phase if w else None, "fh": _r(self.sim.t * HOURS_PER_WINDOW, 1),
            "score": _r(self.pipe.last_score, 2), "flags": self.pipe.last_flags,
            "fault": a.fault if a else None, "confidence": _r(a.confidence, 3) if a else None,
            "rul": [_r(q, 1) for q in a.rul_hours] if a else None,
            "sensor_faults": self.pipe.health.faulty_channels if self.pipe.health else [],
            "decision": self.decision.action if self.decision else None,
            "injected": {"component": self.sim.mode if self.sim.fault_life else None,
                         "sensors": [f"{i.fault.kind}:{i.fault.channel}" for i in self.injectors]},
        }

    def detail(self) -> dict:
        w, s = self.window, self.summary()
        f, amp = envelope_spectrum(w.vibration) if w is not None else (np.zeros(1), np.zeros(1))
        keep = f <= 300
        shaft = shaft_hz(w.load) if w is not None else 0.0
        s.update(
            history={k: list(v) for k, v in self.hist.items()},
            waveform=[_r(v, 2) for v in (w.vibration[:256] if w is not None else [])],
            envelope={"hz": [_r(v, 1) for v in f[keep][::2]], "amp": [_r(v, 4) for v in amp[keep][::2]],
                      "markers": {k: _r(o * shaft, 1) for k, o in GEOMETRY.orders().items() if k != "FTF"},
                      "shaft_hz": _r(shaft, 2)},
            context={"load": _r(w.load, 3), "ambient_c": _r(w.ambient_c, 1), "altitude_ft": _r(w.altitude_ft, 0)}
            if w is not None else None,
            sensor_health=self.pipe.health.status() if self.pipe.health else {},
            advisory=self.last_advisory.to_dict() if self.last_advisory else None,
            decision=self.decision.to_dict() if self.decision else None,
            work_orders=list(self.work_orders),
            schedule=self.schedule().__dict__,
            truth={"mode": self.truth.mode, "degradation": _r(self.truth.degradation, 3),
                   "rul_hours": None if self.truth.rul_windows is None else self.truth.rul_windows * HOURS_PER_WINDOW}
            if self.truth else None,
        )
        return s


class Fleet:
    def __init__(self, bundle, n: int = 6, seed: int = 4242, speed: float = 4.0, model_version: str = "v1"):
        self.bundle, self.speed, self.model_version = bundle, speed, model_version
        self.aircraft = {t: Aircraft(t, seed + 97 * i, bundle, STATIONS[i % len(STATIONS)], phase_offset=3 * i)
                         for i, t in enumerate(TAILS[:n])}
        self.events: deque = deque(maxlen=60)
        self.paused = False

    def tick(self):
        for ac in self.aircraft.values():
            self.events.extendleft(ac.step())

    def log(self, kind: str, text: str, **extra):
        self.events.appendleft({"time": time.time(), "tail": "FLEET", "kind": kind, "text": text, **extra})

    # ---- controls
    def _ac(self, tail: str) -> Aircraft:
        if tail not in self.aircraft:
            raise KeyError(f"unknown aircraft {tail}")
        return self.aircraft[tail]

    def inject_component(self, tail: str, mode: str, life: int = 240):
        if mode not in FAULT_MODES:
            raise ValueError(f"unknown fault {mode!r}")
        self._ac(tail).sim.inject(mode, life)

    def inject_sensor(self, tail: str, kind: str, channel: str):
        if kind not in SENSOR_FAULT_KINDS or channel not in CHANNELS:
            raise ValueError(f"unknown sensor fault {kind}:{channel}")
        ac = self._ac(tail)
        ac.injectors.append(SensorFaultInjector(SensorFault(kind, channel, start=ac.sim.t), seed=ac.sim.t))

    def clear_sensor_faults(self, tail: str):
        self._ac(tail).injectors.clear()

    def reset(self, tail: str):
        self._ac(tail).reset()

    def set_phase(self, tail: str, phase: str | None):
        if phase not in (None, *PHASES):
            raise ValueError(f"unknown phase {phase!r}")
        self._ac(tail).sim.phase_override = phase

    def set_speed(self, speed: float):
        self.speed = float(np.clip(speed, 0.5, 30.0))

    def swap_bundle(self, bundle, version: str):
        self.bundle, self.model_version = bundle, version
        for ac in self.aircraft.values():
            ac.swap_bundle(bundle)

    def snapshot(self) -> dict:
        return {"type": "fleet", "speed": self.speed, "paused": self.paused, "model_version": self.model_version,
                "aircraft": [ac.summary() for ac in self.aircraft.values()], "events": list(self.events)[:25]}


__all__ = ["Fleet", "TAILS", "SENSOR_PRESETS", "HEALTHY"]
