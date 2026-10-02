# Sensors

AeroMind monitors one component per aircraft in the demo: the engine accessory gearbox
(`ENG1-GEARBOX`). All sensor data is **simulated** by the in-repo simulator.

Each window is a short snapshot taken once per 0.5 flight hours: 1 s of vibration and acoustic
data at 1024 Hz, 0.5 s of current at 1024 Hz, and 16 samples each of voltage, temperature,
pressure and oil-debris readings. Operating context (load, outside air temperature, altitude,
flight phase) accompanies every window.

| Channel | Unit | Notes |
|---|---|---|
| vibration | g | High-rate; source of the envelope-spectrum bearing evidence |
| acoustic | RMS | High-rate |
| current | A | AC-coupled; 3rd and 5th harmonic give the THD feature |
| voltage | V | Slow channel |
| temperature | °C | Slow channel |
| pressure | psi | Slow channel |
| oil_debris | ppm | Slow channel |

## Sensor health layer

Every window is checked per channel *before* feature extraction: NaN/inf (`nan`), outside the
physical range (`out_of_range`), no variation (`flatline`), a window identical to the previous one
(`stuck`), a DC offset on an AC-coupled channel (`bias`), a sample far from the window median
(`spike`), and a mean that jumps faster than physics allows (`rate`).

A channel declared faulty is reported once as a `SENSOR_FAULT` advisory and then **masked** (replaced
by its last validated reading) so the component models continue in degraded mode. A failed sensor is
therefore reported as a sensor problem, not as a component fault.

Not detected: slow drift of a slow channel that stays within range. Distinguishing it from a
genuine overheating trend needs a cross-sensor model that does not exist yet.
