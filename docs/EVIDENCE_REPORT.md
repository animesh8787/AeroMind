# AeroMind evidence report

Generated 2026-10-01 20:44 by `python -m aeromind report` (918 s). Simulator results use synthetic data; IMS and C-MAPSS are public datasets. All values are measured by this command except items listed as assumptions.

## Simulator, flight phases (phase-aware model, conformal RUL)

| Fault | Detected | Median lead (h) | First classified correct | RUL MAE (h) | p10-p90 coverage |
|---|---|---|---|---|---|
| bearing_wear | 6/6 | 120.5 | 6/6 | 6.4 | 0.99 |
| oil_contamination | 6/6 | 126.2 | 5/6 | 11.6 | 0.8 |
| overheating | 6/6 | 74.8 | 6/6 | 11.1 | 0.81 |
| electrical_fault | 6/6 | 163.0 | 6/6 | 10.7 | 0.74 |
| pressure_leak | 6/6 | 123.8 | 6/6 | 9.2 | 0.86 |

False advisories: 0.0 per 1000 healthy windows. Downlink reduction: 545x. Per-window latency: 1.11 ms mean, 34.58 ms max (CPU).

## Flight-phase awareness (healthy flights)

| Model | Windows over threshold | Anomaly gate open |
|---|---|---|
| trained without flight phases | 0.9628 | 0.9856 |
| phase-aware | 0.0194 | 0.0 |

## Sensor health

False sensor faults on clean runs: 0 in 5400 windows.

| Injected fault | Detection delay (windows) | Component advisories with / without sensor health | Gate open after fault (with) |
|---|---|---|---|
| stuck:temperature | [2, 2, 2] | 0 / 0 | 0.0 |
| flatline:pressure | [1, 1, 1] | 0 / 15 | 0.0 |
| drift:vibration | [10, 11, 11] | 0 / 21 | 0.0 |
| dropout:acoustic | [1, 1, 1] | 0 / 0 | 0.0 |
| spike:pressure | [1, 1, 1] | 0 / 36 | 0.0 |
| out_of_range:temperature | [1, 1, 1] | 0 / 38 | 0.0 |
| dropout:oil_debris | [1, 1, 1] | 0 / 0 | 0.0 |

## ACARS

68 messages, max 141 / mean 105.4 characters (limit 220); all fit: True.

## Fleet ROI (simulation; costs and failure rates are assumptions)

| Policy | Unscheduled | Scheduled | Early groundings | AOG h | Parts | Cost vs reactive |
|---|---|---|---|---|---|---|
| reactive | 44 | 0 | 0 | 528.0 | 44 | 0.0% |
| fixed_interval | 32 | 90 | 0 | 384.0 | 122 | -24.0% |
| aeromind | 0 | 44 | 0 | 0.0 | 44 | -98.6% |

Sensitivity:

| Scenario | AeroMind vs reactive | Fixed interval vs reactive |
|---|---|---|
| as measured on the simulator | -98.6% | -24.0% |
| detection 67%, warning x0.1, 5 false advisories/1000 windows | -60.4% | -24.0% |
| as measured, AOG cost $10k/h instead of $150k/h | -84.2% | 10.4% |
| detection 67%, warning x0.1, 5 FA/1000, AOG $10k/h | -34.4% | 10.4% |

## Edge benchmark (Linux x86_64 (x86_64), 1 thread; CPU measurements on this host; not Jetson measurements)

Models 1831.6 KB; process RSS after load 226.9 MB. Inference median: anomaly 0.015 ms, classifier 0.02 ms, RUL 0.016 ms; full window 1.211 ms mean.
 INT8 (dynamic): 1832.6 KB, anomaly decisions agree on 97.0% of windows (max score shift 1.0375).


## NASA IMS bearings (real data, test 2)

Lead columns are hours from the first alarm to the end of the test; only bearing 1 failed, so on bearings 2-4 they measure alarms caused by the shared shaft, not warnings of their own failure.

| Bearing | Failed | Gate first open (h) | Hours before end of test | BPFO evidence: hours before end | Gate open, first half |
|---|---|---|---|---|---|
| bearing_1 | True | 89.0 | 74.8 | 74.5 | 0.0 |
| bearing_2 | False | 117.3 | 46.5 | 46.3 | 0.0 |
| bearing_3 | False | 140.5 | 23.3 | 22.8 | 0.0 |
| bearing_4 | False | 85.3 | 78.5 | 55.2 | 0.0 |

Strongest BPFO line after first evidence: {'bearing_1': 0.989, 'bearing_2': 0.007, 'bearing_3': 0.002, 'bearing_4': 0.002}


## NASA C-MAPSS (gradient boosting + conformal interval, official test split)

| Subset | RMSE | MAE | NASA score | p10-p90 coverage | Constant baseline RMSE |
|---|---|---|---|---|---|
| FD001 | 19.377 | 14.218 | 1524 | 0.837 | 48.5 |
| FD002 | 17.745 | 11.946 | 2983 | 0.83 | 52.4 |
| FD003 | 22.255 | 15.047 | 4049 | 0.72 | 62.9 |
| FD004 | 20.1 | 13.326 | 5599 | 0.78 | 61.9 |

## Federated learning: rare-fault sharing (simulator, 3 seeds)

| Model | Fault types seen locally | Fault types never seen locally | Healthy called faulty |
|---|---|---|---|
| local | 0.996 | 0.0 | 0.0 |
| federated | 0.923 | 0.913 | 0.0 |
| centralised_upper_bound | 1.0 | 1.0 | 0.0 |

## LSTM RUL (PyTorch, conformal interval)

Simulator, flight phases: RUL MAE {'bearing_wear': 7.7, 'oil_contamination': 9.3, 'overheating': 10.0, 'electrical_fault': 10.5, 'pressure_leak': 8.1} h; coverage {'bearing_wear': 1.0, 'oil_contamination': 0.9, 'overheating': 0.81, 'electrical_fault': 0.91, 'pressure_leak': 0.95}.

| Subset | RMSE | RMSE std | MAE | NASA score | p10-p90 coverage |
|---|---|---|---|---|---|
| FD001 | 16.003 | 0.5 | 11.567 | 849 | 0.807 |
| FD002 | 14.645 | 0.38 | 9.804 | 1513 | 0.867 |
| FD003 | 16.11 | 2.67 | 10.721 | 1481 | 0.71 |
| FD004 | 15.781 | 0.83 | 10.546 | 1821 | 0.8 |

## Not run

CWRU bearing data: host blocked by this environment's network policy (HTTP 403). N-CMAPSS: 15.8 GB archive, not downloaded. Ground-side LLM maintenance copilot: needs an API key.

