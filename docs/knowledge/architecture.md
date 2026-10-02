# Architecture (one page)

Edge (per aircraft): sensor streams → sensor health → feature extraction → anomaly detection
(Isolation Forest + autoencoder) → persistence gate (5 of the last 8 windows over threshold) →
fault classification → RUL (p10/p50/p90) → physics-backed evidence → advisory.

Ground station: maintenance decision engine (probability of failure before the next check, then
GROUND_NOW / REPLACE_AT_NEXT_CHECK / DEFER_AND_MONITOR) → work-order draft → fleet ROI simulation.
Signed model packages (Ed25519 over a SHA-256 manifest) are pushed to the edge with rollback.

The advisory is the only thing that leaves the aircraft. A compact ACARS-sized text form (at most
220 characters) is generated from it. No real ACARS link exists in this project; the encoding is
only checked for length and round-trip decoding.

The LLM copilot is **ground-side only**. It reads structured outputs of the deterministic pipeline
and explains them. It does not change them and it never runs on the aircraft or the edge device.
