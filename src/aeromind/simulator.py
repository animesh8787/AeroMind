"""Synthetic multi-modal sensor simulator with run-to-failure fault injection.

This stands in for real flight data. Every fault mode is driven by a hidden
degradation level ``d`` in [0, 1] that grows as ``(t / life) ** 2`` and reaches
1.0 at failure. The simulator exposes ``d`` and the true remaining life only as
ground truth (``Truth``); the edge pipeline never sees them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import ELEC_HZ, FAULT_MODES, FS, HEALTHY, N_ELEC, N_HIGH, N_SLOW


@dataclass(frozen=True)
class SensorWindow:
    """One snapshot from all six sensor families."""

    t: int
    load: float  # normalised engine/gearbox load (0.45 - 1.0), an operating-context input
    vibration: np.ndarray  # (N_HIGH,)
    acoustic: np.ndarray  # (N_HIGH,)
    current: np.ndarray  # (N_ELEC,)
    voltage: np.ndarray  # (N_SLOW,)
    temperature: np.ndarray  # (N_SLOW,)
    pressure: np.ndarray  # (N_SLOW,)
    oil_debris: np.ndarray  # (N_SLOW,)


@dataclass(frozen=True)
class Truth:
    """Simulator-only ground truth. Never fed to the pipeline."""

    mode: str
    degradation: float
    rul_windows: int | None  # None for healthy runs
    life: int | None


@dataclass(frozen=True)
class TailProfile:
    """Per-aircraft sensor offsets, so different tails look slightly different."""

    temp_offset: float = 0.0
    pressure_offset: float = 0.0
    vib_gain: float = 1.0
    oil_offset: float = 0.0

    @classmethod
    def random(cls, rng: np.random.Generator, spread: float = 1.0) -> "TailProfile":
        return cls(
            temp_offset=float(rng.normal(0, 3.0 * spread)),
            pressure_offset=float(rng.normal(0, 1.5 * spread)),
            vib_gain=float(np.clip(rng.normal(1.0, 0.08 * spread), 0.7, 1.3)),
            oil_offset=float(rng.normal(0, 0.3 * spread)),
        )


_T_HIGH = np.arange(N_HIGH) / FS
_T_ELEC = np.arange(N_ELEC) / FS
_KERNEL = np.exp(-np.arange(48) / 10.0) * np.sin(2 * np.pi * 300.0 * np.arange(48) / FS)


def _bearing_impulses(rng: np.random.Generator, f_shaft: float) -> np.ndarray:
    """Periodic impacts at the (simulated) outer-race defect frequency, ringing at 300 Hz."""
    period = FS / (3.57 * f_shaft)
    idx = np.round(np.arange(rng.uniform(0, period), N_HIGH, period)).astype(int)
    impulses = np.zeros(N_HIGH)
    impulses[idx[idx < N_HIGH]] = 1.0 + 0.2 * rng.standard_normal(np.sum(idx < N_HIGH))
    return np.convolve(impulses, _KERNEL)[:N_HIGH]


def _make_window(
    rng: np.random.Generator, t: int, load: float, d: float, mode: str, tail: TailProfile
) -> SensorWindow:
    f_shaft = 25.0 + 10.0 * load
    phase = rng.uniform(0, 2 * np.pi)

    # Vibration: shaft + 2x harmonic + noise.
    vib = load * (
        1.0 * np.sin(2 * np.pi * f_shaft * _T_HIGH + phase)
        + 0.5 * np.sin(2 * np.pi * 2 * f_shaft * _T_HIGH + 2 * phase)
    )
    vib_noise = 0.35
    # Acoustic: broadband noise.
    ac_noise = 0.2 + 0.1 * load
    ac = np.zeros(N_HIGH)
    # Electrical.
    i0 = 10.0 + 8.0 * load
    harm3 = harm5 = 0.0
    v_mean, v_ripple = 115.0, 0.3
    # Slow channels.
    temp = 60.0 + 25.0 * load + tail.temp_offset
    press = 40.0 + 10.0 * load + tail.pressure_offset
    press_noise = 0.4
    oil = 2.0 + 0.5 * load + tail.oil_offset

    if mode == "bearing_wear":
        vib = vib + 3.0 * d * _bearing_impulses(rng, f_shaft)
        vib_noise += 0.3 * d
        ac_noise += 0.5 * d
        oil += 12.0 * d
        temp += 8.0 * d
    elif mode == "oil_contamination":
        oil += 30.0 * d
        temp += 12.0 * d
        vib_noise += 0.15 * d
        press -= 2.0 * d
    elif mode == "overheating":
        temp += 50.0 * d
        press += 3.0 * d
        oil += 4.0 * d
    elif mode == "electrical_fault":
        harm3, harm5 = 0.3 * d, 0.15 * d
        v_mean -= 10.0 * d
        v_ripple += 1.5 * d
        temp += 6.0 * d
        vib = vib + 0.4 * d * np.sin(2 * np.pi * 120.0 * _T_HIGH)  # electromagnetic hum
        ac = ac + 0.3 * d * np.sin(2 * np.pi * 120.0 * _T_HIGH)
    elif mode == "pressure_leak":
        press -= 18.0 * d
        press_noise *= 1.0 + 4.0 * d
        ac_noise += 0.7 * d
        temp += 3.0 * d
    elif mode != HEALTHY:
        raise ValueError(f"unknown mode {mode!r}")

    vib = tail.vib_gain * vib + vib_noise * rng.standard_normal(N_HIGH)
    ac = ac + ac_noise * rng.standard_normal(N_HIGH)

    w0 = 2 * np.pi * ELEC_HZ * _T_ELEC
    current = i0 * (np.sin(w0) + harm3 * np.sin(3 * w0) + harm5 * np.sin(5 * w0))
    current = current + 0.2 * rng.standard_normal(N_ELEC)

    return SensorWindow(
        t=t,
        load=load,
        vibration=vib,
        acoustic=ac,
        current=current,
        voltage=v_mean + v_ripple * rng.standard_normal(N_SLOW),
        temperature=temp + 0.5 * rng.standard_normal(N_SLOW),
        pressure=press + press_noise * rng.standard_normal(N_SLOW),
        oil_debris=np.abs(oil + 0.4 * rng.standard_normal(N_SLOW)),
    )


def simulate_run(
    mode: str,
    life: int,
    seed: int,
    tail: TailProfile | None = None,
) -> Iterator[tuple[SensorWindow, Truth]]:
    """Yield ``life`` consecutive windows. For a fault mode the component fails at ``life``;
    for ``healthy`` it never degrades (``life`` is just the run length)."""
    if mode != HEALTHY and mode not in FAULT_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed)
    tail = tail if tail is not None else TailProfile.random(rng)
    load_phase = rng.uniform(0, 2 * np.pi)
    ar = 0.0
    for t in range(life):
        ar = 0.9 * ar + 0.03 * rng.standard_normal()
        load = float(np.clip(0.75 + 0.15 * np.sin(2 * np.pi * t / 60 + load_phase) + ar, 0.45, 1.0))
        if mode == HEALTHY:
            d, rul = 0.0, None
        else:
            d, rul = (t / life) ** 2, life - t
        truth = Truth(mode=mode, degradation=d, rul_windows=rul, life=None if mode == HEALTHY else life)
        yield _make_window(rng, t, load, d, mode, tail), truth
