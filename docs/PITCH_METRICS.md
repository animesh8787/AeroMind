# Pitch metrics (measured; source: artifacts/report/report.md)

| Claim on a slide | Number | Source / caveat |
|---|---|---|
| Early warning | 75–163 h median before failure (simulator); **74.8 h on a real bearing (NASA IMS)** | Simulator is synthetic; IMS is one test rig |
| No false alarms on healthy flights | 0 false advisories per 1,000 windows; anomaly gate 0.0% vs 98.6% without phase awareness | Simulator |
| Knows a bad sensor from a bad part | 0 false component advisories for 7 injected sensor failures (up to 38 each without sensor health); detected in 1–11 windows | Simulator |
| Explains itself with physics | BPFO line identifies the failing bearing in 98.9% of real IMS snapshots | NASA IMS |
| Bandwidth | 545× less than raw; every advisory fits one ACARS block (max 141/220 characters) | Simulator |
| Edge-ready | 1.8 MB of models, ~1.2 ms per window on one CPU thread | Laptop/cloud CPU, **not Jetson** |
| RUL accuracy | C-MAPSS RMSE 14.6–16.1 (LSTM), 17.7–22.3 (trees) vs 48–63 constant baseline | NASA C-MAPSS |
| Names the fault | Correct at the first classified alert in 29/30 runs (oil contamination 5/6) | Simulator |
| Fleet learning | 91% accuracy on fault types an aircraft never saw, via federated averaging (local-only 0%) | Simulator |
| Honest uncertainty | Conformal interval coverage 0.74–0.99 simulator, 0.71–0.87 C-MAPSS (nominal 0.80) | |
| Business case | −34% to −99% maintenance cost vs reactive (30 aircraft × 3,000 FH) | Assumed costs and failure rates; sensitivity shown |
| Secure updates | Tampered model rejected, last valid model kept, rollback | Demo |

Deck corrections: keep slide 7 and 10 wording from `AEROMIND-corrected.pptx` (no "early trials", no
"flight-tested"). Do not claim certification, real flight data or Jetson benchmarks.
