# Limitations

AeroMind V2 is a research prototype. This page is the single list of what it does **not** do. Other
documents link here rather than repeat it.

## Data and results

- The simulator, the six aircraft and every sensor stream in the demo are **synthetic**. The simulator's faults are strong and cleanly separable, so detection and classification are easier than on real equipment.
- Real-data evidence is limited to two public datasets: **NASA IMS** (one bearing test rig; only one bearing failed) and **NASA C-MAPSS** (simulated turbofan engines, RUL only; no fault labels, no raw signals).
- There is **no flight data**, no airline data and no hardware-in-the-loop test.
- Sample sizes are small (6 runs per fault in the standard evaluation), so lead times and coverage figures are noisy.
- RUL intervals are nominal 80% intervals with measured coverage from 0.70 to 1.0 depending on dataset and model; they are not guarantees.
- Slow in-range drift of a slow sensor channel (for example temperature creeping up) is not detected as a sensor fault.
- Federated learning showed **no benefit for anomaly detection** in the tested setup; it helped only in the rare-fault classification experiment, on simulated aircraft.
- INT8 quantisation saved no space and shifted anomaly decisions, so FP32 is used.
- CWRU bearing data and N-CMAPSS were not run.

## Decisions and costs

- Decision thresholds and the cost figures (AOG cost per hour, parts, labour) are **prototype assumptions**, not airline data or certified procedures. The AOG figure comes from the project deck and needs a citation.
- Work orders are in-memory **drafts**. Inspection tasks and part names are illustrative placeholders, not maintenance-manual content.
- The fleet ROI figures depend on those assumptions; the sensitivity table shows the range.

## Connectivity and integration

- **ACARS:** advisories are encoded to fit one 220-character block and decoded again in tests. There is **no ACARS connection**.
- **SWIM, MRO systems, OEM systems, airline operations:** not connected. Nothing is integrated with real aviation infrastructure.
- Over-the-air updates are demonstrated between the ground-station process and in-process aircraft (Ed25519 signature over a SHA-256 manifest, rollback). There is no real aircraft datalink or key-management service.
- The ground station has no user login. Keep it on localhost or a trusted network; when it listens on a LAN set `AEROMIND_INGEST_TOKEN`.

## Hardware

- **No Jetson, FPGA or Raspberry Pi measurement exists in this repository.** All latency, memory and throughput figures come from a laptop or cloud CPU.
- The TensorRT-compatible export was never parsed by TensorRT or run on a GPU.
- The Raspberry Pi 5 deployment path (`deploy/raspberry-pi`, `aeromind edge-agent`, `aeromind bench --target raspberry-pi`) is prepared and unit-tested on x86; **Pi results are PENDING**.
- Sensor input on any edge device is the simulator. A real sensor adapter does not exist.

## LLM copilot

- The copilot explains deterministic results. It does not diagnose, decide or approve anything, and its text can be wrong. Technician review is required.
- Answer quality with a live Groq or Ollama model has not been evaluated; only the validation, safety and fallback layers are tested (with mocked providers).
- The safety checks are conservative text filters, not a guarantee; see [llm.md](llm.md).
- Free-text routing to a task uses keywords, and conversation history is not sent to the model.
- `docs/knowledge/` is project documentation. It is not OEM data, not regulatory guidance and not maintenance data.

## Certification

No certification, regulatory approval, OEM approval, airline deployment or partnership exists. A real system would need
assurance work (for example DO-178C / DO-254 style processes and the regulator's AI guidance) that has not been started.
