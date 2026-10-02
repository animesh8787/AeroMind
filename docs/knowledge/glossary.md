# Glossary

- **ACARS**: Aircraft Communications Addressing and Reporting System, a low-bandwidth datalink. AeroMind only produces ACARS-sized text (≤ 220 characters); it has no ACARS connectivity.
- **Advisory**: the structured record an aircraft sends to the ground (fault, confidence, RUL, evidence).
- **AOG**: aircraft on ground (unscheduled grounding); here a cost assumption.
- **BPFO / BPFI / BSF**: bearing outer-race / inner-race / ball-spin defect frequencies.
- **C-MAPSS**: NASA's simulated turbofan run-to-failure benchmark.
- **Conformal prediction**: a calibration method used to make the RUL interval match its nominal coverage.
- **Edge**: the onboard compute running the AeroMind pipeline.
- **FH**: flight hours.
- **IMS**: NASA Intelligent Maintenance Systems bearing run-to-failure dataset (real vibration data).
- **Isolation Forest / autoencoder**: the two anomaly detectors combined into one score.
- **Masked channel**: a sensor channel declared faulty and replaced by its last valid reading.
- **MRO**: maintenance, repair and overhaul.
- **OTA**: over-the-air model update; packages are Ed25519-signed with SHA-256-pinned files.
- **Persistence gate**: requires 5 of the last 8 windows over the anomaly threshold before an advisory.
- **RUL p10/p50/p90**: remaining useful life quantiles, in flight hours.
- **SENSOR_FAULT**: a sensor channel failed validity checks; not a component fault.
