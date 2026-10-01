# Feature matrix

Status as of the current `main`. "Tested" means covered by `pytest`; "Measured" means a number in
`artifacts/report/report.md` comes from `python -m aeromind report`.

| Feature | Status | Tested | Measured | Notes |
|---|---|---|---|---|
| Six sensor families, fusion features | Exists | Yes | Yes | Simulated; plus air-data context (OAT, altitude) |
| Flight phases (taxi ... cruise ... taxi) | Exists | Yes | Yes | Gate open on healthy flights 98.6% without, 0.0% with |
| Sensor health (stuck, flatline, drift, dropout, spike, range) | Exists | Yes | Yes | 0 false sensor faults in 5,400 clean windows |
| Anomaly detection (Isolation Forest + autoencoder), persistence gate | Exists | Yes | Yes | |
| Fault classification | Exists | Yes | Yes | Oil contamination 2/6 first-classified on flight phases |
| RUL quantiles (trees), LSTM (PyTorch), conformal interval | Exists | Yes | Yes | LSTM optional (`.[lstm]`) |
| Physics evidence (envelope BPFO/BPFI/BSF, THD) | Exists | Yes | Yes | Also on real IMS data |
| Maintenance decision engine + work orders | Exists | Yes | n/a | Prototype policy, not certified |
| ACARS-sized advisory (<=220 chars) | Exists | Yes | Yes | Max 141 chars measured |
| Live ground station, fault injection | Exists | Yes (API + browser) | n/a | `python -m aeromind serve` |
| Signed OTA updates + rollback | Exists | Yes | n/a | Ed25519 over a SHA-256 manifest |
| Fleet ROI simulation + sensitivity | Exists | Yes | Yes | Costs and failure rates are assumptions |
| ONNX export / ONNX Runtime backend | Exists | Yes | Yes | |
| TensorRT-compatible export (Hummingbird) | Exists | Yes | Operators only | Not built with TensorRT (no GPU) |
| Edge benchmark, INT8 | Exists | Partly | Yes | CPU only; INT8 not beneficial |
| Jetson kit | Partial | No | No | NOT YET VALIDATED ON HARDWARE |
| NASA IMS real bearing data | Exists | Yes (synthetic) | Yes | Needs ~1.1 GB download |
| NASA C-MAPSS benchmark | Exists | Yes | Yes | |
| Federated learning | Partial | Yes | Yes | Mechanism only; no benefit shown |
| CI (GitHub Actions) | Exists | n/a | n/a | Torch/Hummingbird/browser tests skip |
| Transformer-lite, TensorFlow, ACARS/SWIM connectivity, MRO integration | Missing | | | Roadmap |
