# AeroMind: proof-of-concept codebase

Onboard (edge) predictive aircraft health and maintenance intelligence, scaffolded from the
AeroMind deck (*Project FY27-605244, AI at the Edge Solutions for Aerospace*).

**Everything here runs on simulated data**: an in-house sensor simulator plus NASA's simulated C-MAPSS turbofan benchmark. There is no real flight data, no
hardware-in-the-loop, no Jetson/TensorRT deployment, and no regulatory work. The numbers below
describe how the software behaves on a synthetic simulator, not how it would perform on aircraft.

## What it does

```
simulator ──> features ──> anomaly score ──> persistence gate ──> fault classifier ──> RUL (p10/p50/p90) ──> advisory JSON
 (6 sensor     (multi-modal   (IsolationForest +  (5 of last 8       (gradient-boosted    (quantile gradient     (the only thing
  families)     fusion, 19)    autoencoder)        windows alarm)     trees)               boosting)              sent to ground)
```

| Deck claim | Where | Status |
|---|---|---|
| Six sensor families, multi-modal fusion | `simulator.py`, `features.py` | Implemented (synthetic) |
| Anomaly detection (autoencoder, Isolation Forest) | `models/anomaly.py` | Implemented |
| Autonomous fault classification | `models/classifier.py` | Implemented (HistGradientBoosting as the XGBoost-class model) |
| RUL with confidence | `models/rul.py` | Implemented (quantile regression) |
| Actionable alerts only, bandwidth reduction | `alerts.py`, `pipeline.py` | Implemented, measured |
| Federated fleet learning | `federated.py` | FedAvg *mechanism* only; see results |
| LSTM / transformer-lite, PyTorch/TF | | Not implemented (scikit-learn only) |
| ONNX / TensorRT, Jetson, FPGA DSP, OTA updates | | Not implemented |
| ACARS / SWIM, MRO connectors | | Not implemented (advisories are plain JSON) |

## Quick start

```bash
pip install -e ".[dev]"
python -m aeromind train                         # ~20 s, writes artifacts/bundle.joblib
python -m aeromind demo --mode bearing_wear      # stream one run-to-failure through the pipeline
python -m aeromind demo --mode healthy --life 300
python -m aeromind evaluate                      # closed-loop metrics (~3 min)
python -m aeromind federated                     # FedAvg autoencoder demo
python -m aeromind cmapss                        # NASA C-MAPSS RUL benchmark (~2.5 min, downloads ~12 MB)
pytest
```

Fault modes: `bearing_wear`, `oil_contamination`, `overheating`, `electrical_fault`, `pressure_leak`.
Each is driven by a hidden degradation `d = (t/life)^2` that reaches 1.0 at failure. Each window
represents one 1 s snapshot per 0.5 flight hours (`config.py`).

## Results on the simulator

`python -m aeromind evaluate` (6 fresh runs per mode, 6 healthy runs; seeds disjoint from training):

| Fault | Detected | Lead time, first alert (h) | Lead time, first *classified* alert (h) | First classified alert correct | RUL MAE (h) | p10-p90 coverage |
|---|---|---|---|---|---|---|
| bearing_wear | 6/6 | 118.0 | 96.0 | 6/6 | 6.4 | 0.87 |
| oil_contamination | 6/6 | 96.8 | 82.5 | 4/6 | 8.5 | 0.47 |
| overheating | 6/6 | 95.8 | 95.8 | 6/6 | 9.8 | 0.63 |
| electrical_fault | 6/6 | 112.5 | 111.8 | 6/6 | 6.5 | 0.58 |
| pressure_leak | 6/6 | 125.2 | 125.2 | 6/6 | 8.5 | 0.57 |

- **False advisories:** 0 per 1000 healthy windows (about 1,800 windows; small sample).
- **Downlink:** about 860x fewer bytes than streaming raw float32 samples.
- **Latency:** mean about 13 ms, max about 75 ms per window on this *cloud CPU*. Not representative of Jetson hardware, and not a deterministic-latency guarantee.

How to read this honestly:

- The simulator's faults are strong and cleanly separable, so detection and classification are easy. Real data will be much harder.
- Early alerts (around 5-10% degradation) are reported as `unclassified_anomaly` at low confidence, then become a typed fault once the signature is clear. That is why "first classified" lead time is shorter than "first alert".
- The RUL p10-p90 interval is nominally 80% but **under-covers** (0.47-0.87). The uncertainty bands are over-confident.
- With only 6 runs per mode these figures are noisy. They are a smoke test, not a benchmark.

### Federated learning: no benefit shown

`python -m aeromind federated` trains an autoencoder with FedAvg across 5 simulated aircraft (only weights
and scaler statistics are shared) and tests it on an unseen aircraft, on early faults (10-30% degradation).
Across 8 runs (4 seeds x 2 baseline-spread settings) the federated model scored AUROC 0.96-0.99. A local
model trained on just 40 windows of the new aircraft scored 0.99-1.00, at least as high as the federated
model in every run. A local model trained on one *other* aircraft was mixed (0.96-1.00; federated was ahead in 2 of 8).
So in this setup federation showed **no measurable advantage**. The module demonstrates the mechanism; the
benefit claimed in the deck remains unproven here.

## Results on NASA C-MAPSS (public benchmark)

`python -m aeromind cmapss` downloads the official NASA turbofan degradation dataset
(Saxena & Goebel, 2008, NASA Prognostics Data Repository) and runs the same components
(calibrated anomaly score, trend features, quantile RUL regression) on it. These are NASA's
*simulated* run-to-failure engines, not flight data, but unlike the in-house simulator the
benchmark is independent of this codebase.

Protocol, fixed before looking at test results: 14 standard informative sensors; per-operating-regime
standardisation (6 regimes for FD002/FD004); anomaly detector fitted on the first 20% of life of 75% of
training engines and calibrated on the rest; 30-cycle trend window; RUL target capped at 125 cycles;
one prediction per test engine at its last observed cycle on the **official test split**. Mean ± std over 3 model seeds
(the test set is the same; the seeds vary model initialisation and the fit/calibration split).

| Subset | Test engines | RMSE (cycles) | MAE | NASA score (lower is better) | p10-p90 coverage | RMSE without anomaly features | RMSE, constant baseline |
|---|---|---|---|---|---|---|---|
| FD001 | 100 | **20.0 ± 0.7** | 14.4 ± 0.6 | 1,607 | 0.75 ± 0.02 | 26.4 ± 0.1 | 49.8 |
| FD002 | 259 | **17.8 ± 0.2** | 11.9 ± 0.2 | 2,801 | 0.80 ± 0.01 | 20.7 ± 0.1 | 52.6 |
| FD003 | 100 | **21.3 ± 0.4** | 14.0 ± 0.5 | 3,627 | 0.70 ± 0.04 | 25.9 ± 0.2 | 63.7 |
| FD004 | 248 | **20.5 ± 0.2** | 13.3 ± 0.0 | 6,954 | 0.72 ± 0.02 | 22.8 ± 0.3 | 62.3 |

- RMSE and MAE are against the RUL **capped at 125**, the common convention. Against uncapped true RUL, RMSE is 21.0 / 28.0 / 22.1 / 29.4 for FD001-FD004, because some test engines have far more than 125 cycles left.
- The anomaly-score features reduce RMSE in all four subsets (for example FD001: 26.4 to 20.0). That is the one ablation run.
- The p10-p90 interval is nominally 80%. Coverage is 0.70-0.80, so it is still somewhat over-confident, though much closer than on the in-house simulator.
- This is a simple tree-based model on hand-built window features. It is **not** compared against published deep-learning results here, and no claim is made that it is competitive with them.
- Only the RUL part of the pipeline is exercised: C-MAPSS has no labelled fault types, no raw vibration/acoustic signals and no edge hardware, so it says nothing about fault classification, bandwidth or on-aircraft latency.

## Layout

```
src/aeromind/
  config.py        sensor/window constants
  simulator.py     synthetic sensors, fault injection, per-aircraft profiles
  features.py      feature extraction + streaming/offline trend features
  models/          anomaly.py, classifier.py, rul.py
  train.py         simulate -> train -> ModelBundle
  pipeline.py      EdgePipeline: window in, advisory out (gating, de-duplication, priority)
  alerts.py        Advisory schema, recommended actions
  evaluate.py      closed-loop evaluation
  federated.py     FedAvg autoencoder + demo
  cmapss.py        NASA C-MAPSS loader and RUL benchmark
  cli.py           train | demo | evaluate | federated | cmapss
tests/            test_core.py, test_cmapss.py
```

## Suggested next steps

1. Add public bearing run-to-failure data (e.g. NASA IMS / FEMTO) to exercise the vibration and classification path on real signals.
2. Re-calibrate the RUL intervals (e.g. conformal prediction); coverage is currently below nominal.
3. Export models (ONNX) and benchmark on target hardware.
4. Re-test federation with realistic, non-IID fleet data before claiming a benefit.
