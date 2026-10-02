# Feature matrix (AeroMind V2)

Status values: **IMPLEMENTED** (code exists), **PARTIAL**, **MISSING**. "Tested" means covered by `pytest`;
"Measured" means a number comes from `python -m aeromind report` (see [evidence-report.md](evidence-report.md)).
Everything is simulated unless the notes say REAL DATASET. Limitations are in [limitations.md](limitations.md).

| Feature | Status | Tested | Measured | Notes |
|---|---|---|---|---|
| Seven sensor channels, multi-modal fusion (21 features) | IMPLEMENTED | Yes | Yes | Simulated; includes air-data context (OAT, altitude) |
| Flight phases (taxi ... cruise ... taxi) | IMPLEMENTED | Yes | Yes | Gate open on 98.6% of healthy windows without phases, 0.0% with |
| Sensor health (stuck, flatline, drift, dropout, spike, range) | IMPLEMENTED | Yes | Yes | 0 false sensor faults in 5,400 clean windows; slow in-range drift not detected |
| Anomaly detection (Isolation Forest + autoencoder), persistence gate | IMPLEMENTED | Yes | Yes | |
| Fault classification (context residuals) | IMPLEMENTED | Yes | Yes | 29/30 first classified alerts correct (published report) |
| RUL quantiles (trees), PyTorch LSTM, conformal interval | IMPLEMENTED | Yes | Yes | LSTM is optional (`.[lstm]`) |
| Physics evidence (envelope BPFO/BPFI/BSF, current THD) | IMPLEMENTED | Yes | Yes | Also on REAL DATASET NASA IMS |
| Maintenance decision engine + work-order drafts | IMPLEMENTED | Yes | n/a | Prototype policy, not certified |
| ACARS-sized advisory (220 characters) | IMPLEMENTED | Yes | Yes | Encoding only; no ACARS connectivity |
| Ground station: fleet, fault injection, what-if, ROI panel | IMPLEMENTED | Yes (API + browser) | n/a | `python -m aeromind serve` |
| Signed OTA model updates + rollback | IMPLEMENTED | Yes | n/a | Ed25519 over a SHA-256 manifest; in-process demo |
| Fleet ROI simulation + sensitivity | IMPLEMENTED | Yes | Yes | Costs and failure rates are assumptions |
| ONNX export and ONNX Runtime backend | IMPLEMENTED | Yes | Yes | |
| TensorRT-compatible export (Hummingbird) | PARTIAL | Yes | Operators only | Never built with TensorRT |
| Edge benchmark (size, load time, latency, throughput, memory, INT8) | IMPLEMENTED | Yes | Yes | CPU only; `--target raspberry-pi` refuses non-Pi machines |
| NASA IMS real bearing data | IMPLEMENTED | Yes (synthetic) | Yes | REAL DATASET, about 1.1 GB download |
| NASA C-MAPSS benchmark | IMPLEMENTED | Yes | Yes | REAL DATASET (simulated engines) |
| Federated learning | IMPLEMENTED | Yes | Yes | Rare-fault sharing helps (91% vs 0%); no benefit for anomaly detection |
| LLM provider abstraction (Groq, Ollama, template fallback) | IMPLEMENTED | Yes (mocked + fake Ollama server) | n/a | Live-model quality not evaluated |
| LLM copilot: 8 tasks, typed schemas, safety checks, repair retry | IMPLEMENTED | Yes | n/a | Ground-side only |
| Copilot in the ground station (panel, LLM STATUS, API) | IMPLEMENTED | Yes (API) | n/a | Browser flow checked manually |
| Knowledge base for the copilot (`docs/knowledge`) | IMPLEMENTED | Yes | n/a | Keyword retrieval; project notes, not OEM or regulatory data |
| Edge agent + remote ingest to the ground station | IMPLEMENTED | Yes (loopback) | n/a | Not run across two devices |
| Raspberry Pi 5 deployment kit | PARTIAL | x86 only | **PENDING** | Prepared; no hardware result |
| Jetson kit | PARTIAL | No | **UNVALIDATED** | Never run on a Jetson |
| Real sensor adapter for the edge agent | MISSING | | | Roadmap |
| Authentication for the ground station | MISSING | | | Roadmap |
| ACARS / SWIM connectivity, MRO integration | MISSING | | | Roadmap; needs a partner |
| FPGA | MISSING | | | Not planned here |
| CWRU, N-CMAPSS | MISSING | | | Not run |
| Transformer-lite, TensorFlow | MISSING | | | Not planned |
| Evidence deck and pre-read built from the report | IMPLEMENTED | No | n/a | `tools/deck`, `tools/preread`, `docs/presentation/` |
| CI (GitHub Actions) | IMPLEMENTED | n/a | n/a | No API key needed; torch/Hummingbird/browser tests skip |
