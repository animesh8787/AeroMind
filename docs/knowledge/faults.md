# Fault modes

The simulator injects five component fault modes. Each is driven by a hidden degradation
`d = (t / life)^2` that reaches 1.0 at failure.

| Fault type | Dominant signature (simulated) | Draft inspection task (prototype placeholder) |
|---|---|---|
| `bearing_wear` | Periodic vibration impulses (kurtosis, envelope lines at bearing defect frequencies); oil debris and temperature also rise | Inspect bearing (borescope / chip detector); plan replacement |
| `oil_contamination` | Oil debris rises strongly; temperature rises, pressure dips slightly | Sample and flush lubrication system; inspect filter |
| `overheating` | Temperature rises strongly; pressure and oil debris rise slightly | Check cooling path and thermal sensors |
| `electrical_fault` | Current harmonic distortion rises; mean voltage drops, voltage ripple rises | Inspect power feed and generator/controller harmonics |
| `pressure_leak` | Pressure falls and becomes noisier; acoustic noise rises | Leak-check pressure line, seals and fittings |
| `unclassified_anomaly` | Persistent anomaly, mode not yet clear | Schedule inspection |

Early in a degradation the system often reports `unclassified_anomaly` at low confidence and later
switches to a typed fault once the signature is clear. The inspection tasks and part names are
illustrative placeholders, not maintenance-manual content.

`SENSOR_FAULT` is **not** a component fault. It means a sensor channel failed its validity checks.
