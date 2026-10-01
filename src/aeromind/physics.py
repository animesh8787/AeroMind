"""Physics-backed evidence for advisories, computed from the raw signals.

Bearing faults: envelope analysis. Band-pass the vibration around the structural resonance
excited by the defect impacts, take the Hilbert envelope, and look for peaks in the envelope
spectrum at the bearing's characteristic defect frequencies (BPFO / BPFI / BSF, from the
bearing geometry and shaft speed) and their harmonics.

Electrical faults: current harmonic distortion (3rd and 5th harmonic of the supply frequency).

Shaft speed comes from the operating context (load), as a tachometer would provide it on an
engine; the relation matches the simulator's drive model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from .config import ELEC_HZ, FS, N_ELEC
from .simulator import SensorWindow


@dataclass(frozen=True)
class BearingGeometry:
    """Rolling-element bearing: ball count, ball/pitch diameter ratio, contact angle (deg)."""

    n_balls: int = 9
    d_over_D: float = 0.2066
    contact_deg: float = 0.0

    def orders(self) -> dict[str, float]:
        """Defect frequencies as multiples of shaft speed."""
        r = self.d_over_D * np.cos(np.radians(self.contact_deg))
        return {
            "BPFO": self.n_balls / 2 * (1 - r),  # outer-race defect
            "BPFI": self.n_balls / 2 * (1 + r),  # inner-race defect
            "BSF": (1 / (2 * self.d_over_D)) * (1 - r**2),  # rolling-element (ball) defect
            "FTF": 0.5 * (1 - r),  # cage
        }


GEOMETRY = BearingGeometry()
RESONANCE_BAND = (200.0, 450.0)  # Hz, band containing the impact ringing


def shaft_hz(load: float) -> float:
    """Shaft speed from the operating context (same drive model as the simulator)."""
    return 25.0 + 10.0 * load


def envelope_spectrum(x: np.ndarray, fs: float = FS, band=RESONANCE_BAND) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, amplitude) of the envelope of ``x`` band-passed to ``band``."""
    n = len(x)
    spec = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[(f < band[0]) | (f > band[1])] = 0
    env = np.abs(hilbert(np.fft.irfft(spec, n)))
    amp = np.abs(np.fft.rfft(env - env.mean())) * 2 / n
    return f, amp


@dataclass(frozen=True)
class BearingFinding:
    defect: str  # BPFO / BPFI / BSF
    expected_hz: float
    peak_hz: float
    order: float
    snr: float
    harmonics: int  # how many of the 2x, 3x harmonics also stand out

    def text(self) -> str:
        h = f", {self.harmonics} harmonic{'s' if self.harmonics != 1 else ''}" if self.harmonics else ""
        return (f"Envelope peak {self.peak_hz:.0f} Hz ≈ {self.defect} ({self.order:.2f}× shaft "
                f"{self.peak_hz / self.order:.1f} Hz), SNR {self.snr:.1f}{h}")


def bearing_findings(vibration: np.ndarray, load: float, min_snr: float = 4.0, tol_hz: float = 2.0,
                     geometry: BearingGeometry = GEOMETRY) -> list[BearingFinding]:
    """Defect frequencies whose envelope-spectrum peak stands ``min_snr`` above the noise floor."""
    f, amp = envelope_spectrum(vibration)
    shaft = shaft_hz(load)

    def peak_near(hz):
        """Peak within ``tol_hz`` of ``hz`` and its ratio to the local noise floor.

        The envelope spectrum of band-limited noise is not flat (it falls with frequency), so the
        floor is the median of a ±20 Hz neighbourhood, excluding the peak region itself.
        """
        m = np.abs(f - hz) <= tol_hz
        if not m.any():
            return None, 0.0
        near = (np.abs(f - hz) <= 20) & ~(np.abs(f - hz) <= tol_hz + 1)
        floor = float(np.median(amp[near])) + 1e-12
        i = np.flatnonzero(m)[np.argmax(amp[m])]
        return float(f[i]), float(amp[i] / floor)

    out = []
    for name in ("BPFO", "BPFI", "BSF"):
        order = geometry.orders()[name]
        hz, snr = peak_near(order * shaft)
        if hz is None or snr < min_snr:
            continue
        harmonics = sum(1 for k in (2, 3) if peak_near(k * order * shaft)[1] >= min_snr / 2)
        if harmonics == 0 and snr < 1.5 * min_snr:
            continue  # a lone, weak peak is too often noise
        out.append(BearingFinding(name, order * shaft, hz, order, snr, harmonics))
    return sorted(out, key=lambda b: -b.snr)


def current_thd(current: np.ndarray) -> tuple[float, float, float]:
    """(THD, 3rd harmonic ratio, 5th harmonic ratio) of the supply current."""
    spec = np.abs(np.fft.rfft(current)) * 2 / N_ELEC
    k = lambda h: int(round(h * ELEC_HZ * N_ELEC / FS))  # noqa: E731
    fund = spec[k(1)] + 1e-12
    h3, h5 = spec[k(3)] / fund, spec[k(5)] / fund
    return float(np.hypot(h3, h5)), float(h3), float(h5)


def evidence(w: SensorWindow, fault: str, z_top: list[tuple[str, float]] | None = None) -> tuple[str, ...]:
    """Human-readable evidence for an advisory about ``fault``, computed from window ``w``.

    ``z_top`` (feature name, z-score vs the healthy baseline) adds the statistical evidence the
    models used; physics findings come first.
    """
    out = []
    if fault in ("bearing_wear", "unclassified_anomaly"):
        out += [b.text() for b in bearing_findings(w.vibration, w.load)[:1]]
    if fault in ("electrical_fault", "unclassified_anomaly"):
        thd, h3, h5 = current_thd(w.current)
        if thd > 0.05:
            out.append(f"Current THD {thd * 100:.1f}% (3rd {h3 * 100:.1f}%, 5th {h5 * 100:.1f}% of {ELEC_HZ:.0f} Hz)")
    for name, z in (z_top or [])[: 3 - len(out) if out else 3]:
        out.append(f"{name} {z:+.1f}σ vs healthy baseline")
    return tuple(out)
