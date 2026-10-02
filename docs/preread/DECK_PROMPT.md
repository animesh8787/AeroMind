# Prompt for building the AeroMind presentation

How to use: paste everything below the line into your slide tool (Claude, Gamma, Copilot, Canva Magic, etc.).
Attach the four PNGs from `docs/preread/figures/` and, if the tool accepts it, `AeroMind-PreRead.pdf`.
Every number in the prompt comes from `docs/EVIDENCE_REPORT.md`. Tell the tool not to change them.

---

You are a presentation designer and technical storyteller. Build a **13-slide, 16:9 pitch deck** for **AeroMind**, my entry in **Tata InnoVent 2027** (Project ID FY27-605244, category "AI at the Edge Solutions for Aerospace", team leader Animesh). The audience is a panel of senior engineers and business leaders in aerospace and aviation. They will see a live demo after the deck, so the deck must be short, visual and honest. The talk is about 6 minutes, so keep each slide to one idea.

## Product in one sentence
AeroMind is an AI system that runs on the aircraft. It reads six sensor families (vibration, temperature, pressure, oil debris, acoustic, electrical) once per second, detects degrading components, names the fault, estimates remaining useful life (RUL) with an uncertainty band, explains the call with physics, and sends the ground one short message instead of raw data.

## Hard rules
1. **Use only the numbers below. Do not invent, round up or add statistics, logos, customer names, certifications or quotes.**
2. Next to every result, show a small tag for its data source: `REAL DATA`, `PUBLIC BENCHMARK`, `SIMULATOR` or `ASSUMPTION`.
3. Do not claim flight testing, certification, real flight data or Jetson/TensorRT measurements. None exist yet. Say "proof of concept" on slides 1 and 12.
4. Speaker notes: 2 to 4 plain sentences per slide, in first person, no jargon without a one-line explanation.

## Design direction
- Look: clean, modern aerospace. Deep navy `#0B2545` for text and dark slides, teal `#13A89E` as the accent, amber `#E09F3E` only to highlight one thing per slide, light blue-grey `#EAF2F7` for panels, white backgrounds.
- Font: Calibri or Inter. Titles are a full sentence stating the takeaway (not a topic label), 28 to 32 pt. Body text at least 18 pt. Maximum 3 bullets or 40 words per slide.
- Big numbers: show the key figure at 60 pt or larger, with a one-line caption.
- Use the attached images for slides 4, 7, 9 and 10. Do not redraw the charts or change their numbers.
- No stock photos of planes with text over them. Simple line icons are fine.

## Slide outline

**1. Title.** "AeroMind: AI that tells you which part will fail, while the aircraft is still flying". Subtitle: "Real-time onboard AI for predictive aircraft health and maintenance | Tata InnoVent 2027 | FY27-605244 | Team AeroMind, led by Animesh". Small tag: "Proof of concept".

**2. The problem.** Title: "Maintenance is scheduled or reactive, so failures are found too late". Three points: critical faults are often found after the flight; raw sensor streams cannot be sent to the ground in real time; static thresholds cannot tell a take-off transient from a fault or a failed probe from a failed part. Big number: **$150k+ per AOG hour** with tag `ASSUMPTION` (industry-style benchmark, varied down to $10k later).

**3. Why current systems fall short.** Four gaps as a 2x2: latency (minutes to hours), cost (bandwidth and ground processing), intelligence (static rules do not learn), prediction (no usable RUL or confidence).

**4. The solution.** Title: "Intelligence moves onto the aircraft; only a short advisory goes to the ground". Show `architecture.png` full width. Caption: "No connectivity needed on board. Only the advisory is downlinked."

**5. How it works, step by step.** One line per stage: sensor health check, features that know the flight phase, anomaly score, persistence gate (5 of the last 8 windows), fault classifier, RUL with p10 / p50 / p90, physics evidence, advisory. Animate or number the steps.

**6. What sets it apart.** Six cards, one line each: flight-phase aware; tells a bad sensor from a bad part; explains itself with physics (bearing defect frequencies); calibrated uncertainty (conformal prediction); decision engine with work orders; signed updates plus federated fleet learning.

**7. Proof on a real bearing.** Title: "On a real bearing, it warned 74.8 hours before failure". Big number **74.8 h** with tag `REAL DATA: NASA IMS bearing test 2`. Second figure: **98.9%** of snapshots show the outer-race defect frequency (BPFO) line on the failing bearing. Footnote: "One test rig, one failed bearing."

**8. No false alarms, and sensor faults are not mistaken for part faults.** Two big numbers, both tagged `SIMULATOR`: anomaly gate open on **0.0%** of healthy flight windows (vs **98.6%** without flight-phase awareness); **0** false component advisories on 7 injected sensor failures (up to 38 each without sensor health). Also: 0 false advisories per 1,000 healthy windows.

**9. Honest RUL accuracy.** Show `cmapss_rmse.png`. Tag `PUBLIC BENCHMARK: NASA C-MAPSS, official test split`. Caption: "LSTM RMSE 16.0 / 14.6 / 16.1 / 15.8 (FD001 to FD004) vs 48.5 to 62.9 for a constant baseline. Uncertainty bands are conformal: coverage 0.71 to 0.87 against a nominal 0.80."

**10. Business case.** Show `roi.png`. Tag `ASSUMPTION: simulation, not a forecast`. Caption: "Maintenance cost vs reactive: -98.6% as measured on the simulator, -34.4% under harsh assumptions (67% detection, warnings 10x shorter, 5 false advisories per 1,000 windows, AOG at $10k/h). Fixed-interval maintenance: -24.0%, and +10.4% when AOG is cheap."

**11. Fits the real world.** Four stats, tagged `SIMULATOR` or `CPU`: **545x** less downlink than raw data; every advisory fits one ACARS block (max **141** of **220** characters); **1.2 ms** per window on one CPU thread with **1.8 MB** of models (tag "CPU, not Jetson"); updates are Ed25519-signed, a tampered model is rejected and rollback works.

**12. Where we are and what is next.** Title: "A working proof of concept, with an honest path to hardware". Left: what exists (live ground-station demo, tested code, evidence report, federated learning: **91.3%** on fault types never seen locally vs **0%** local-only, tag `SIMULATOR`). Right: roadmap in three steps: 3 months, more public run-to-failure data (CWRU, N-CMAPSS); 3 to 6 months, Jetson bench validation; 6 to 12 months, hardware-in-the-loop and a shadow-mode trial beside an existing health-monitoring system. Add a line: "Not yet done: flight data, Jetson measurements, certification."

**13. Live demo.** Dark navy slide with `ground-station.png`. Text: "Live demo: inject a fault, watch AeroMind catch it". Three small labels: bearing fault, sensor failure, signed update.

## Output
Return the finished deck (or a slide-by-slide structure with exact text, layout and speaker notes if you cannot build slides). End with a short list of any place where you were tempted to add a number that is not in this prompt.
