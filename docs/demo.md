# Live demo run sheet (about 6 minutes)

Everything in the demo is **simulated**: six fictional aircraft (`VT-AMA01` to `VT-AMA06`), simulated sensors and
simulated failures. Say so on screen and out loud.

## Before the demo

```bash
pip install -e ".[app]"
python -m aeromind serve          # first start trains a flight-phase model (about 20 s); then open http://127.0.0.1:8000
```

Set speed to 8 windows/s. Wi-Fi can stay off: the demo needs no internet. The **LLM STATUS** chip in the header shows
what the copilot will use:

- `ONLINE` / `LOCAL`: a real LLM answers (Groq or Ollama; see [llm.md](llm.md)). Check with `aeromind llm doctor`.
- `FALLBACK`: no LLM is reachable. The copilot still answers, from rule-based templates, and says so on screen. This is a valid way to demo.

Optional: show the edge device path with `aeromind edge-agent --ground http://127.0.0.1:8000 --tail VT-PI01 --inject bearing_wear`
(a second process standing in for a Raspberry Pi; simulated sensors).

## Run sheet

| # | Step | What to say and show |
|---|---|---|
| 1 | Launch the ground station | "Six simulated aircraft, each with its own edge pipeline." |
| 2 | Healthy fleet | All NOMINAL, even through take-off: flight-phase awareness. |
| 3 | Select **VT-AMA03** | The aircraft detail panel. |
| 4 | Inject **bearing wear** | Button in the right column. |
| 5 | Sensor stream changes | Vibration waveform and envelope spectrum change. |
| 6 | Anomaly detector reacts | Anomaly score rises over the threshold. |
| 7 | Persistence gate triggers | Shaded region: 5 of 8 windows over threshold. |
| 8 | Fault classifier responds | `unclassified anomaly`, then `bearing wear`. |
| 9 | RUL changes | p50 with a p10 to p90 band that falls over time. |
| 10 | Physics evidence appears | "Envelope peak at BPFO, the outer-race defect frequency", computed from the vibration. |
| 11 | Maintenance decision appears | Action, risk before the next check, cost if planned vs deferred (labelled assumptions). |
| 12 | ACARS-sized advisory appears | The 220-character line in the work-order panel and the event log. |
| 13 | Open the **AI Maintenance Copilot** panel | Note the two kinds of block: DETERMINISTIC AEROMIND OUTPUT and AI COPILOT. |
| 14 | Ask "Why is this aircraft at risk?" | Or click *Explain Alert*. |
| 15 | Copilot explains using the actual evidence | Point at the deterministic block first, then the explanation, then the "technician review required" label. |
| 16 | Click *What Should Maintenance Inspect?* | A draft checklist from the deterministic decision. |
| 17 | Click *Draft Work Order* | Status DRAFT; fault, RUL, risk, part and window come from the pipeline, not the model. Also try *What If We Defer?*: the numbers come from the decision engine. |
| 18 | On another aircraft (e.g. **VT-AMA05**) inject **Temperature sensor failure** | Right column, sensor failures. |
| 19 | `SENSOR_FAULT` appears | Event log and the sensor tile. |
| 20 | The channel is isolated and masked | "A broken probe is not a broken gearbox"; status stays nominal, no component alarm. |
| 21 | Degraded monitoring continues | The remaining six channels keep feeding the models. |
| 22 | Ask the copilot "Why is this a sensor fault, not a bearing problem?" on that aircraft | It explains from the sensor-health output; there is no component fault to explain. |
| 23 | **Push signed model** | Accepted; model chip changes to v2. |
| 24 | Verify accepted | Event log: "verified (Ed25519 + SHA-256) and installed". |
| 25 | **Push tampered model** | One byte flipped after signing. |
| 26 | Verify rejected | Event log: REJECTED with the SHA-256 reason. |
| 27 | Verify the previous valid model stays active | The model chip still shows v2; the fleet keeps running. *Roll back* returns to v1. |

## Evidence slide (optional, 30 s)

Open `docs/evidence-report.md`: real NASA IMS bearing (outer-race failure warned 74.8 h ahead) and the fleet ROI range
with its assumptions. Say that Raspberry Pi and Jetson figures do not exist yet.

## Do not say

"Flight-tested", "certified", "approved", "integrated with ACARS / SWIM / an airline", or any Raspberry Pi or Jetson speed.
The copilot is "AI-generated assistance" and does not make the maintenance decision.

## Automated check

`pytest tests/integration/test_server_copilot.py tests/integration/test_server.py` runs this scenario headlessly
(fault injection, advisory, copilot, sensor fault, OTA accept, tamper reject, rollback).
