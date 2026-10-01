# Architecture

```mermaid
flowchart LR
  subgraph Aircraft["Aircraft (edge, one per tail)"]
    S["6 sensor families<br/>vibration, acoustic, current/voltage,<br/>temperature, pressure, oil debris<br/>+ air data (OAT, altitude, phase)"]
    H["Sensor health<br/>NaN, range, flatline, stuck,<br/>bias, spike, rate"]
    F["Feature fusion<br/>21 features + trend / 30-window sequence"]
    A["Anomaly score<br/>Isolation Forest + autoencoder"]
    G{"Persistence gate<br/>5 of 8 windows"}
    C["Fault classifier<br/>gradient-boosted trees"]
    R["RUL p10/p50/p90<br/>quantile trees or LSTM<br/>+ conformal interval"]
    P["Physics evidence<br/>envelope BPFO/BPFI/BSF,<br/>current THD"]
    ADV["Advisory<br/>JSON + 220-char ACARS"]
    S --> H --> F --> A --> G
    G -- open --> C --> R --> P --> ADV
    H -- "SENSOR_FAULT (channel masked)" --> ADV
  end
  subgraph Ground["Ground station"]
    D["Decision engine<br/>P(fail before next check)<br/>ground / replace / defer"]
    W["Work order + parts"]
    ROI["Fleet ROI simulation"]
    OTA["Signed model packages<br/>Ed25519 + SHA-256, rollback"]
  end
  ADV -- downlink --> D --> W
  D --> ROI
  OTA -- "verified update" --> A
```

Models run in ONNX Runtime (FP32) on the edge; the tree ensembles can also be exported as tensor
operations (Hummingbird) for TensorRT. Code map: `simulator.py` (incl. flight phases, `LiveAircraft`),
`sensor_health.py`, `features.py`, `models/`, `pipeline.py`, `physics.py`, `acars.py`, `decision.py`,
`roi.py`, `signing.py`, `onnx_export.py`, `server/` + `web/`, `datasets/ims.py`, `cmapss.py`, `report.py`.
