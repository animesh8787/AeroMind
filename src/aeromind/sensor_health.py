"""Sensor health: tell a failing *sensor* from a failing *component*.

Every window is checked per channel before feature extraction. A channel that is declared
faulty is reported once as a ``SENSOR_FAULT`` advisory and then masked (replaced by its last
validated reading) so the component models keep running in a degraded mode instead of
mistaking the broken sensor for a fault in the gearbox.

Simulated sensor faults are injected with ``SensorFaultInjector``; the simulator itself never
produces them.

Checks (all per window, cheap enough for the edge):
  nan           any NaN/inf sample (dropout, open circuit)
  out_of_range  a sample outside the channel's physical range
  flatline      no variation inside the window (every real sensor here has noise)
  stuck         the window repeats the previous one exactly (frozen data path)
  bias          a DC offset on an AC-coupled channel (vibration, acoustic, current),
                which healthy signals and genuine faults never produce
  spike         a slow-channel sample far from the window median (robust z on the MAD)
  rate          a slow-channel mean jumping faster than the physics allows

Not detected: slow drift of a slow channel (e.g. temperature creeping up) that stays in
range. Telling that from a genuine overheating trend needs a cross-sensor model and is
listed as a limitation.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .simulator import SensorWindow

CHANNELS = ("vibration", "acoustic", "current", "voltage", "temperature", "pressure", "oil_debris")
AC_CHANNELS = ("vibration", "acoustic", "current")
SLOW_CHANNELS = ("voltage", "temperature", "pressure", "oil_debris")

# Physical plausibility limits (sensor full-scale), in the simulator's nominal units.
RANGES = {
    "vibration": (-40.0, 40.0),
    "acoustic": (-20.0, 20.0),
    "current": (-80.0, 80.0),
    "voltage": (60.0, 160.0),
    "temperature": (-60.0, 250.0),
    "pressure": (0.0, 120.0),
    "oil_debris": (0.0, 500.0),
}
# Largest plausible change of a slow channel's window mean from one window to the next.
MAX_STEP = {"voltage": 30.0, "temperature": 30.0, "pressure": 25.0, "oil_debris": 40.0}
UNITS = {"vibration": "g", "acoustic": "RMS", "current": "A", "voltage": "V",
         "temperature": "°C", "pressure": "psi", "oil_debris": "ppm"}


# --------------------------------------------------------------------------- injection


SENSOR_FAULT_KINDS = ("stuck", "flatline", "drift", "dropout", "spike", "out_of_range")


@dataclass(frozen=True)
class SensorFault:
    """A simulated sensor failure starting at window ``start``."""

    kind: str
    channel: str
    start: int = 0
    magnitude: float = 1.0

    def __post_init__(self):
        if self.kind not in SENSOR_FAULT_KINDS:
            raise ValueError(f"unknown sensor fault {self.kind!r}")
        if self.channel not in CHANNELS:
            raise ValueError(f"unknown channel {self.channel!r}")


class SensorFaultInjector:
    """Applies one ``SensorFault`` to a stream of windows (stateful: stuck needs a frozen copy)."""

    def __init__(self, fault: SensorFault, seed: int = 0):
        self.fault = fault
        self.rng = np.random.default_rng(seed)
        self._frozen: np.ndarray | None = None

    def apply(self, w: SensorWindow) -> SensorWindow:
        f = self.fault
        if w.t < f.start:
            return w
        x = np.array(getattr(w, f.channel), dtype=float)
        lo, hi = RANGES[f.channel]
        if f.kind == "stuck":
            if self._frozen is None:
                self._frozen = x.copy()
            x = self._frozen.copy()
        elif f.kind == "flatline":
            x = np.full_like(x, float(np.mean(x)))
        elif f.kind == "drift":
            # Bias that grows with time: an accelerometer or amplifier drifting off zero.
            scale = np.std(x) if f.channel in AC_CHANNELS else max(abs(np.mean(x)), 1.0)
            x = x + f.magnitude * 0.02 * (w.t - f.start + 1) * scale
        elif f.kind == "dropout":
            x[:] = np.nan
        elif f.kind == "spike":
            n = int(self.rng.integers(1, 3))
            idx = self.rng.choice(len(x), n, replace=False)
            x[idx] += f.magnitude * 0.3 * (hi - lo) * self.rng.choice([-1.0, 1.0], n)
            x = np.clip(x, lo, hi)
        elif f.kind == "out_of_range":
            x = np.full_like(x, hi + 0.5 * (hi - lo))
        return replace(w, **{f.channel: x})


# --------------------------------------------------------------------------- detection


@dataclass(frozen=True)
class SensorAdvisory:
    """Downlinked when a sensor is declared faulty (or recovers)."""

    component: str
    window: int
    flight_hours: float
    channel: str
    status: str  # "FAULTY" or "RECOVERED"
    checks: tuple[str, ...]
    action: str
    fault_type: str = "SENSOR_FAULT"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = list(self.checks)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @property
    def size_bytes(self) -> int:
        return len(self.to_json().encode())


@dataclass
class SensorHealthConfig:
    confirm_n: int = 2  # failed checks in this many of the last ``confirm_of`` windows declare a fault
    confirm_of: int = 3
    clear_after: int = 5  # consecutive clean windows before a faulty channel is restored
    spike_z: float = 12.0  # robust z (|x - median| / 1.4826 MAD) for a slow-channel spike
    bias_ratio: float = 0.5  # |mean| / std for AC channels
    bias_min: float = 0.3


@dataclass
class ChannelState:
    history: deque = field(default_factory=lambda: deque(maxlen=3))
    faulty: bool = False
    clean_run: int = 0
    last_checks: tuple[str, ...] = ()
    last_good: np.ndarray | None = None
    since: int | None = None


def check_channel(name: str, x: np.ndarray, prev: np.ndarray | None, prev_mean: float | None,
                  cfg: SensorHealthConfig) -> list[str]:
    """Names of the checks this channel fails in this window (empty list: healthy)."""
    failed = []
    if not np.all(np.isfinite(x)):
        return ["nan"]
    lo, hi = RANGES[name]
    if x.min() < lo or x.max() > hi:
        failed.append("out_of_range")
    if np.ptp(x) == 0.0:
        failed.append("flatline")
    elif prev is not None and prev.shape == x.shape and np.array_equal(prev, x):
        failed.append("stuck")
    if name in AC_CHANNELS:
        m, s = abs(float(np.mean(x))), float(np.std(x))
        if m > cfg.bias_min and m > cfg.bias_ratio * s:
            failed.append("bias")
    else:
        med = float(np.median(x))
        mad = 1.4826 * float(np.median(np.abs(x - med)))
        if mad > 0 and np.max(np.abs(x - med)) / mad > cfg.spike_z:
            failed.append("spike")
        if prev_mean is not None and abs(float(np.mean(x)) - prev_mean) > MAX_STEP[name]:
            failed.append("rate")
    return failed


class SensorHealthMonitor:
    """Per-channel validation, fault confirmation, masking and recovery."""

    def __init__(self, component: str, cfg: SensorHealthConfig | None = None, hours_per_window: float = 0.5):
        self.component, self.cfg, self.hours_per_window = component, cfg or SensorHealthConfig(), hours_per_window
        self.state = {c: ChannelState() for c in CHANNELS}
        self._prev: dict[str, np.ndarray] = {}
        self._prev_mean: dict[str, float] = {}

    @property
    def faulty_channels(self) -> list[str]:
        return [c for c, s in self.state.items() if s.faulty]

    def status(self) -> dict[str, dict]:
        return {c: {"ok": not s.faulty, "checks": list(s.last_checks), "since": s.since} for c, s in self.state.items()}

    def process(self, w: SensorWindow) -> tuple[SensorWindow, list[SensorAdvisory]]:
        """Validate ``w``; return it with faulty channels masked, plus any new sensor advisories."""
        cfg, out, masked = self.cfg, [], {}
        for c in CHANNELS:
            x = np.asarray(getattr(w, c), dtype=float)
            st = self.state[c]
            checks = check_channel(c, x, self._prev.get(c), self._prev_mean.get(c), cfg)
            st.history.append(bool(checks))
            st.last_checks = tuple(checks)
            self._prev[c] = x
            if checks:
                st.clean_run = 0
            else:
                st.clean_run += 1
                self._prev_mean[c] = float(np.mean(x))
            if not st.faulty and sum(st.history) >= cfg.confirm_n:
                st.faulty, st.since = True, w.t
                out.append(self._advisory(w, c, "FAULTY", checks or st.last_checks))
            elif st.faulty and st.clean_run >= cfg.clear_after:
                st.faulty, st.since = False, None
                out.append(self._advisory(w, c, "RECOVERED", ()))
            if st.faulty or checks:
                # Mask: suspect data never reaches the models. Before confirmation a single bad
                # window is masked too, so one glitch cannot move the anomaly score.
                lo, hi = RANGES[c]
                masked[c] = st.last_good if st.last_good is not None else np.clip(np.nan_to_num(x), lo, hi)
            else:
                st.last_good = x
        return (replace(w, **masked) if masked else w), out

    def _advisory(self, w: SensorWindow, channel: str, status: str, checks) -> SensorAdvisory:
        action = (f"Inspect/replace {channel} sensor and harness; component monitoring continues without it"
                  if status == "FAULTY" else f"{channel} sensor readings valid again")
        return SensorAdvisory(self.component, w.t, round(w.t * self.hours_per_window, 1), channel, status,
                              tuple(checks), action)
