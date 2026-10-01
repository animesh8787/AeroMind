# AeroMind: technical report

Project FY27-605244, Tata InnoVent 2027, "AI at the Edge Solutions for Aerospace".
Every number below is produced by `python -m aeromind report` (see `artifacts/report/report.md`) unless
marked as an assumption. Simulator results use synthetic data; NASA IMS and C-MAPSS are public datasets.

## 1. Problem

Maintenance is scheduled or reactive; degradation is found late, causing aircraft-on-ground (AOG)
events; and raw sensor streams are too large to send to the ground. AeroMind runs on the aircraft,
turns sensor windows into a few high-confidence advisories, and on the ground turns each advisory
into a maintenance decision.

## 2. System

See `ARCHITECTURE.md`. Per 1-second window: sensor health checks → feature fusion (21 features incl.
outside air temperature and altitude) → anomaly score (Isolation Forest + autoencoder, calibrated so 1.0
is the 99th percentile of healthy windows) → persistence gate (5 of 8 windows) → fault classifier →
RUL p10/p50/p90 (quantile gradient boosting or a PyTorch LSTM, with a conformal interval) → physics
evidence → advisory (JSON and a 220-character ACARS message). The ground station adds the decision
engine, work orders, fleet ROI and signed over-the-air model updates.

## 3. Results

**Simulator, flight phases** (6 runs per fault, phase-aware model, conformal RUL): every fault detected
(6/6 each), median warning 75–126 h before failure, 0 false advisories per 1,000 healthy windows, 607×
less downlink than raw data, RUL interval coverage 0.81–0.99 (nominal 0.80). Weakness: oil contamination
is classified correctly at the first classified alert in only 2 of 6 runs.

**Flight-phase awareness:** a model trained without flight phases has its anomaly gate open on 98.6% of
healthy flight windows; the phase-aware model on 0.0%.

**Sensor health:** 0 false sensor faults in 5,400 clean windows; injected sensor failures detected in
1–2 windows (vibration drift 10–11), with 0 component advisories and the anomaly gate closed (0%).

**NASA IMS bearings (real data):** on the bearing that failed (outer race), the anomaly gate first opens
89.0 h into the 163.8 h test (74.8 h before the end) and the envelope BPFO signature persists from 89.3 h;
no gate opened in the first half of the test on any bearing. The anomaly score alone does not identify
which bearing (others on the same shaft alarm later, one slightly earlier), but the BPFO line does: bearing
1 has the strongest line in 98.9% of snapshots once it appears.

**NASA C-MAPSS** (official test split, 3 seeds): gradient boosting + conformal interval RMSE 19.4 / 17.7 /
22.3 / 20.1 (constant baseline 48.5–62.9), interval coverage 0.72–0.84. The optional LSTM reaches RMSE
14.8–15.6 (README).

**Edge:** 1.8 MB of models; ~0.02 ms per model call and ~1.2 ms per window on one CPU thread (laptop/cloud
CPU, not Jetson). INT8 quantisation saved no space and shifted 3% of anomaly decisions, so FP32 is kept.

**Fleet ROI** (30 aircraft × 3,000 FH; costs and failure rates assumed): AeroMind −98.6% maintenance cost vs
reactive as measured on the simulator; −34.4% under harsh assumptions (67% detection, 10% of the warning
time, 5 false advisories per 1,000 windows, $10k/h AOG). Fixed-interval replacement: −24.0% and +10.4%.

## 4. Limitations

Simulated faults are cleaner than real ones; IMS is one test rig and C-MAPSS is simulated engines; no flight
data, no hardware-in-the-loop, no Jetson or TensorRT run, no certification work. Decision thresholds and
cost figures are prototype assumptions. Federated learning is implemented as a mechanism only and showed
no benefit in our setup.

## 5. Next steps

1. Run the Jetson kit (`deploy/jetson`) and measure latency and power. 2. Add real multi-sensor data
(e.g. N-CMAPSS, more bearing sets). 3. Improve oil-contamination classification under flight phases.
4. Partner data and an MRO integration pilot.
