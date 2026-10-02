# AeroMind demo video script (about 4 minutes)

Format: screen recording of the ground station with voice-over, 1080p. Flow follows `docs/demo.md`.
Every number is from `docs/evidence-report.md`. Keep the "simulated" wording: the aircraft and faults are simulated.

## Setup before recording

```bash
pip install -e ".[app]"
python -m aeromind serve        # first start trains a flight-phase model (~20 s)
```

- Open http://127.0.0.1:8000 in a 1500x1000 window. Set speed to 8 windows/s. Wi-Fi off (shows it needs no internet).
- Reset all aircraft between takes. Record in OBS at 1080p, 30 fps. Record voice-over separately for clean audio.
- Have `docs/preread/figures/cmapss_rmse.png` and `roi.png` ready for the last scene (full screen images).

## Script

| Time | Scene | On screen | Voice-over |
|---|---|---|---|
| 0:00-0:20 | Hook | Title card: "AeroMind: AI that runs on the aircraft". Cut to the ground station, six aircraft nominal | "Most aircraft faults are found after the flight, when the aircraft is already on the ground and costing thousands of dollars an hour. AeroMind moves the intelligence onto the aircraft. This is a proof of concept running on simulated data." |
| 0:20-0:45 | Fleet and flight phases | Fleet panel, phase labels cycling through taxi, take-off, climb, cruise | "Six simulated aircraft, each running its own edge pipeline through taxi, take-off, climb, cruise and descent. All nominal, even during take-off, because the models know the flight phase. Without that, the anomaly gate was open on 98.6% of healthy windows. With it, 0.0%." |
| 0:45-1:50 | Component fault | Select VT-AMA03, click bearing wear. Zoom on score, gate shading, classifier, RUL band, evidence line, work order, ACARS line | "I inject a bearing fault. The anomaly score rises, and the persistence gate opens after five of eight windows. The classifier names the fault: bearing wear. Remaining life appears as a band, p10 to p90, and it keeps falling. Here is the evidence: the envelope spectrum peaks at the outer-race defect frequency, computed from the vibration. The decision engine recommends an action and creates a work order. The message to the ground is one ACARS line, around 140 of 220 characters. Raw data would have been about 10 kilobytes for every second. That is 545 times less." |
| 1:50-2:30 | Sensor failure | Select VT-AMA05, click temperature sensor failure. Show the SENSOR_FAULT event, masked channel, status staying nominal | "A broken probe is not a broken part. I fail a temperature sensor. AeroMind flags a sensor fault within a couple of windows, masks the channel, and the aircraft stays nominal. Across seven kinds of injected sensor failure there were zero false component advisories. Without this check there were up to 38." |
| 2:30-3:10 | Secure updates | Click push signed model, then tampered model, then roll back | "Models are updated over the air. A signed model is accepted. A tampered one is rejected, and the fleet keeps the last valid model. Rollback is one click. Packages are signed with Ed25519 and checked against a SHA-256 manifest." |
| 3:10-3:40 | Real-data proof | Full-screen: IMS result from `docs/evidence-report.md`, then `cmapss_rmse.png` | "This is not only simulation. On a real bearing run-to-failure test from NASA, the gate opened 74.8 hours before the failure, and the defect-frequency line identified the failing bearing in 98.9% of snapshots. On NASA's C-MAPSS benchmark, the LSTM's error is 14.6 to 16.1 cycles, against 48 to 63 for a naive baseline." |
| 3:40-4:00 | Business case and close | Full-screen `roi.png`, then end card with repo link | "In a fleet simulation, maintenance cost falls between 34 and 99 percent versus reactive, depending on assumptions that I list openly. What is next: Raspberry Pi 5 and Jetson bench validation, then hardware-in-the-loop. AeroMind: from scheduled maintenance to intelligent aviation foresight." |

## V2 addition: copilot scene (about 30 s)

After the component fault, open the AI Maintenance Copilot panel and click *Explain Alert*, then *Draft Work Order*. Say: "A ground-side assistant
explains what the pipeline already decided. The numbers come from AeroMind, not from the language model, and the work order is only a draft."
Show the two separate blocks (deterministic output, then the explanation with its label). If no LLM is configured the header says FALLBACK
and the answer is a rule-based template; say that instead of calling it AI. Trim the business-case scene to keep to four minutes.

## Notes

- Do not say "flight-tested", "certified" or quote Jetson or Raspberry Pi speeds. None exist yet.
- Say "simulated" at least three times. Judges trust the numbers more when the data source is clear.
- If a take shows a wrong classification, reset the aircraft and re-record; the first classified alert is correct in 29 of 30 runs, so one miss is possible but not a story for the video.
- The voice-over is about 360 words (about 2 min 45 s at a calm pace), which leaves room for pauses while the screen action plays.
