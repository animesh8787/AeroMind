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
| LSTM, PyTorch | `models/lstm.py` | Implemented for RUL (optional; quantile LSTM over the last 30 windows) |
| Transformer-lite, TensorFlow | | Not implemented |
| ONNX optimized inference | `onnx_export.py` | Implemented: ONNX export + ONNX Runtime backend (CPU); see below |
| TensorRT | `onnx_export.py` | TensorRT-compatible ONNX export (Hummingbird); **not yet built or run with TensorRT** |
| Jetson, FPGA DSP, OTA updates | | Not implemented |
| ACARS / SWIM, MRO connectors | | Not implemented (advisories are plain JSON) |

## Quick start

```bash
pip install -e ".[dev]"
python -m aeromind train                         # ~20 s, writes artifacts/bundle.joblib
python -m aeromind demo --mode bearing_wear      # stream one run-to-failure through the pipeline
python -m aeromind demo --mode healthy --life 300
python -m aeromind evaluate                      # closed-loop metrics (~3 min)
python -m aeromind export-onnx                   # writes artifacts/onnx/ and checks parity with scikit-learn
python -m aeromind export-onnx --trees hummingbird   # TensorRT-compatible graphs in artifacts/onnx-trt/ (needs .[hummingbird])
python -m aeromind demo --backend onnx           # same pipeline, models run in ONNX Runtime
python -m aeromind dashboard                     # writes artifacts/dashboard.html (open in a browser)
python -m aeromind federated                     # FedAvg autoencoder demo
python -m aeromind cmapss                        # NASA C-MAPSS RUL benchmark (~2.5 min, downloads ~12 MB)

# Optional PyTorch LSTM for remaining useful life (pip install -e ".[lstm]")
python -m aeromind train --rul lstm --model artifacts/bundle-lstm.joblib    # ~30 s
python -m aeromind evaluate --model artifacts/bundle-lstm.joblib
python -m aeromind cmapss --rul lstm             # ~20 min on 4 CPU cores
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

## LSTM model for remaining useful life (PyTorch)

`models/lstm.py` adds an optional RUL model: a single-layer PyTorch LSTM that reads the last 30
windows of [features, anomaly score] and outputs p10/p50/p90. The head adds non-negative gaps to p10,
so the quantiles are always ordered. It is trained with the pinball (quantile) loss, with dropout,
weight decay and early stopping on held-out *runs* (never held-out windows of a training run). It
replaces only the RUL step: detection, the persistence gate and fault classification are unchanged.
The pipeline keeps a rolling 30-window history when the bundle's RUL model is an LSTM.

Model size was chosen on held-out training data only, never on the evaluation runs or the C-MAPSS
test split: 16 units on the simulator (lowest pinball loss over three held-out splits of training runs)
and 64 units on C-MAPSS (lowest mean RMSE on held-out FD001/FD004 training engines, of 16/32/64).

**C-MAPSS** (same protocol as above, official test split, mean ± std over 3 seeds):

| Subset | RMSE, LSTM | RMSE, gradient boosting (above) | MAE, LSTM | NASA score, LSTM | p10-p90 coverage, LSTM | RMSE, LSTM without anomaly score |
|---|---|---|---|---|---|---|
| FD001 | **15.6 ± 0.7** | 20.0 ± 0.7 | 11.0 ± 0.3 | 766 | 0.78 ± 0.04 | 14.3 ± 0.3 |
| FD002 | **14.8 ± 0.1** | 17.8 ± 0.2 | 9.8 ± 0.1 | 1,630 | 0.82 ± 0.05 | 15.3 ± 0.7 |
| FD003 | **15.6 ± 0.3** | 21.3 ± 0.4 | 10.5 ± 0.4 | 1,279 | 0.77 ± 0.03 | 14.8 ± 0.5 |
| FD004 | **15.5 ± 0.3** | 20.5 ± 0.2 | 10.2 ± 0.1 | 1,656 | 0.76 ± 0.05 | 16.0 ± 0.3 |

- The LSTM is better than gradient boosting on every subset, by 3 to 6 cycles RMSE, and roughly halves
  the NASA score. Its p10-p90 intervals cover 0.76-0.82, close to the nominal 0.80.
- The anomaly score **does not help the LSTM**: without it, RMSE is better on FD001 and FD003 and worse
  on FD002 and FD004. The LSTM reads the raw sensor sequence and gets little from the extra input. (For
  gradient boosting the anomaly features did help; see the ablation above.)
- Against uncapped true RUL, RMSE is 16.5 / 25.2 / 16.6 / 26.2.
- Published deep models on C-MAPSS report FD001 RMSE of roughly 11-16. These figures are in that range but
  were not produced under identical conditions, so no ranking is claimed.

**Simulator** (`evaluate`, same 6 runs per mode as above):

| Fault | RUL MAE, gradient boosting (h) | RUL MAE, LSTM (h) | p10-p90 coverage, gradient boosting | p10-p90 coverage, LSTM |
|---|---|---|---|---|
| bearing_wear | 6.4 | 8.1 | 0.87 | 0.99 |
| oil_contamination | 8.5 | 8.3 | 0.47 | 0.89 |
| overheating | 9.8 | 9.8 | 0.63 | 0.84 |
| electrical_fault | 6.5 | 5.8 | 0.58 | 0.96 |
| pressure_leak | 8.5 | 8.9 | 0.57 | 0.91 |

Detection, lead times, classification and false advisories are identical, since the LSTM only replaces
RUL. Median error is about the same; the gain is in the intervals, which no longer under-cover (on
bearing wear they are now wider than needed). Per-window latency on one CPU thread: 7.7 ms mean with
PyTorch, **0.57 ms with ONNX Runtime**, against 9.6 ms for the scikit-learn pipeline.

The LSTM exports with the rest of the bundle (`export-onnx --model artifacts/bundle-lstm.joblib`):
`rul.onnx` is then a 14 KB graph with input `sequence` (N, 30, 20), recorded in `manifest.json`. It
matches PyTorch to within 5e-5 h at every batch size, uses the standard ONNX `LSTM` operator (on
TensorRT's supported list), and gives identical `evaluate` results to the PyTorch model.

## ONNX export

`python -m aeromind export-onnx` writes three float32 graphs and a manifest to `artifacts/onnx/`:

| File | Input | Output | Contents |
|---|---|---|---|
| `anomaly.onnx` (~260 KB) | features (N,19) | score (N), z (N,19) | scaler, Isolation Forest, autoencoder and the healthy calibration, all in-graph |
| `classifier.onnx` (~680 KB) | features (N,19) | probabilities (N,6) | gradient-boosted trees + softmax; class names in `manifest.json` |
| `rul.onnx` (~875 KB) | trend features (N,22) | rul_hours (N,3) | the three quantile models in one ensemble, sorted and clipped at 0 |

Feature extraction, trend tracking, the persistence gate and advisory formatting stay in Python
(`pipeline.py`); only the learned models are exported. `manifest.json` records SHA-256 hashes,
and `OnnxBundle` refuses to load a file that does not match. `OnnxBundle` is a drop-in for
`ModelBundle`, so `demo`, `evaluate` and `dashboard` all take `--backend onnx`.

The graphs are written directly with `onnx.helper` rather than skl2onnx: skl2onnx 1.20 fails on
scikit-learn 1.9's HistGradientBoosting models.

Parity with scikit-learn on 1,800 fresh simulated windows (all modes):

- Class probabilities differ by at most 7e-7; the predicted class agrees on every window.
- Anomaly score: median difference 5e-7. One window in 1,800 landed within float32 rounding of an
  Isolation Forest split and moved by 0.02. Every threshold decision agrees.
- RUL: median difference 2e-5 h. Four windows in 1,800 took the other branch of a split and moved by up to 0.4 h.

End to end, `python -m aeromind evaluate --backend onnx` gives **identical** per-mode results to the
scikit-learn backend (detection, lead times, classification, RUL error and coverage, false advisories,
downlink). Only latency changes. Both runs used one thread (`OMP_NUM_THREADS=1`) on the same cloud CPU,
and the times cover the whole `process()` call, feature extraction included:

| Backend | Mean ms / window | Max ms / window | Full `evaluate` run |
|---|---|---|---|
| scikit-learn | 9.55 | 30.0 | 1 min 59 s |
| ONNX Runtime | 0.53 | 5.6 | 10 s |

Most of the gain is per-call overhead: scikit-learn is slow at predicting one row at a time. These
are cloud-CPU numbers, not Jetson numbers.

These default graphs express the tree models with the ONNX-ML `TreeEnsembleRegressor` operator,
which ONNX Runtime runs but **TensorRT does not support**. For TensorRT, use the Hummingbird export below.

### TensorRT-compatible export (Hummingbird)

```bash
pip install -e ".[hummingbird]"     # adds torch and hummingbird-ml (export machine only)
python -m aeromind export-onnx --trees hummingbird            # GEMM trees -> artifacts/onnx-trt/
python -m aeromind export-onnx --trees hummingbird --strategy tree_trav --out artifacts/onnx-trt-tt
```

[Hummingbird](https://github.com/microsoft/hummingbird) compiles the Isolation Forest, the fault
classifier and the three RUL quantile models into ordinary tensor operations (matrix multiplies and
comparisons for `gemm`, gathers for `tree_trav`). Each converted ensemble is spliced into the same
three graphs as before, so inputs, outputs, `manifest.json` and `OnnxBundle` are unchanged. The
exporter refuses to write a graph containing any operator outside a list of operators TensorRT's ONNX
parser supports (`TENSORRT_OPERATORS`), and the manifest records the operators used by each file.

Three problems had to be fixed to make Hummingbird 0.4.12 work here:

- **Wrong results.** Its scikit-learn HistGradientBoosting parser stores any leaf value or threshold
  of exactly 0 as -1. This model has 57 zero-valued classifier leaves and one zero-valued RUL leaf, so
  the stock conversion gave saturated probabilities (17% of predicted classes wrong) and RUL errors of
  exactly 1 h. `onnx_export.py` swaps in a corrected parser during conversion.
- **No export on current PyTorch.** PyTorch 2.9+ exports through `torch.export`, which cannot trace
  Hummingbird's modules. The exporter uses PyTorch's TorchScript exporter (`dynamo=False`); tested with
  torch 2.14. A future PyTorch that drops it will need a newer Hummingbird.
- **Batch of one broke.** Hummingbird's `squeeze()` exports as a `Squeeze` with no axes, which also
  drops the batch dimension when N = 1, the pipeline's case. The exporter pins each `Squeeze` to the
  axes it removes at batch size 3; TensorRT wants static axes anyway.

Hummingbird also only converts classifiers with integer labels, so the classifier is converted with
labels 0–5. The class names stay in `manifest.json`.

Results on this cloud CPU (ONNX Runtime, one thread). `evaluate` gives **identical** per-mode results on
all four backends, and parity with scikit-learn is the same as for the ONNX-ML graphs above:

| Backend | Files | Models only, one row (ms) | Full `process()` per window, mean / max (ms) |
|---|---|---|---|
| scikit-learn | joblib | n/a | 9.55 / 30.0 |
| ONNX-ML trees | 1.8 MB | 0.15 | 0.53 / 5.6 |
| Hummingbird GEMM | 9.4 MB | 1.09 | 1.31 / 8.0 |
| Hummingbird tree traversal | 2.5 MB | 1.87 | 1.29 / 7.7 |

On a CPU the tensor-op graphs are slower than ONNX-ML trees: GEMM evaluates every split of every tree
as one dense matrix product, which suits a GPU rather than a CPU. **These graphs have not been parsed
by TensorRT, built into an engine, or run on a Jetson.** The export environment had no GPU, and
NVIDIA's package index was not reachable. On the device:

```bash
trtexec --onnx=artifacts/onnx-trt/classifier.onnx --minShapes=features:1x19 \
        --optShapes=features:1x19 --maxShapes=features:64x19 --saveEngine=classifier.plan
```

and the same for `anomaly.onnx` (`features`, 19 columns) and `rul.onnx` (`trend_features`, 22 columns).
Compare GEMM against `tree_trav` there and keep the faster one, then run `check_parity` against the
engine outputs before trusting them. `hummingbird-ml` 0.4.12 pins `onnx<=1.16.1`, which has wheels up
to Python 3.12, so use Python 3.12 or older for this export step; the device needs neither package.

## Demo dashboard

`python -m aeromind dashboard` runs five fault scenarios and one healthy run through the edge
pipeline (ONNX backend by default; seeds disjoint from training and evaluation) and writes
`artifacts/dashboard.html`, a single self-contained file (~1.2 MB) that needs no server. It shows:

- crew/MRO status (NOMINAL, WATCH, ADVISORY, URGENT, CRITICAL), fault type and confidence;
- the anomaly score against its threshold, with the persistence gate shaded;
- predicted RUL (p50 with p10–p90 band) against the simulator's ground truth;
- each advisory as it is sent, with its exact JSON payload and size, and the running downlink total vs raw;
- the six sensor families and the raw vibration waveform of the current window;
- per-window inference latency, measured on the machine that generated the page.

Press Play to replay a run, drag the slider to scrub, or click a chart or advisory to jump to that window.

## Layout

```
src/aeromind/
  config.py        sensor/window constants
  simulator.py     synthetic sensors, fault injection, per-aircraft profiles
  features.py      feature extraction + streaming/offline trend features
  models/          anomaly.py, classifier.py, rul.py, lstm.py
  train.py         simulate -> train -> ModelBundle
  pipeline.py      EdgePipeline: window in, advisory out (gating, de-duplication, priority)
  alerts.py        Advisory schema, recommended actions
  evaluate.py      closed-loop evaluation
  onnx_export.py   ONNX graphs, ONNX Runtime bundle, parity check
  dashboard.py     self-contained HTML replay (+ dashboard_template.html)
  federated.py     FedAvg autoencoder + demo
  cmapss.py        NASA C-MAPSS loader and RUL benchmark
  cli.py           train | export-onnx | demo | evaluate | dashboard | federated | cmapss
tests/            test_core.py, test_cmapss.py, test_onnx_dashboard.py, test_lstm.py
```

## Suggested next steps

1. Add public bearing run-to-failure data (e.g. NASA IMS / FEMTO) to exercise the vibration and classification path on real signals.
2. Re-calibrate the RUL intervals (e.g. conformal prediction): the gradient-boosting intervals under-cover; the LSTM's are close to nominal on C-MAPSS but too wide for bearing wear on the simulator.
3. Build the Hummingbird graphs with TensorRT on a Jetson, check parity on the device, and benchmark GEMM against tree traversal.
4. Re-test federation with realistic, non-IID fleet data before claiming a benefit.
