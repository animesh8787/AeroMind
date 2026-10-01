# Live demo run sheet (about 4 minutes)

Before judges arrive: `python -m aeromind serve`, open http://127.0.0.1:8000, Wi-Fi off, speed 8 windows/s.

1. **Fleet (20 s).** "Six simulated aircraft, each with its own edge pipeline, flying taxi, take-off, climb,
   cruise, descent. All nominal, even through take-off: that is the flight-phase awareness."
2. **Component fault (75 s).** Select VT-AMA03 → *bearing wear*. Narrate: anomaly score rises → persistence
   gate opens (shaded) → classifier names bearing wear → RUL band appears and falls → evidence: "Envelope peak
   ≈ BPFO, the outer-race defect frequency, computed from the vibration" → decision and work order →
   ACARS line (~140 of 220 characters). "Raw data would be 3.7 MB; we send one short message."
3. **Sensor failure (45 s).** Select VT-AMA05 → *Temperature sensor failure*. "A broken probe is not a broken
   gearbox": SENSOR_FAULT event, channel masked, status stays nominal, no component alarm.
4. **Over-the-air update (40 s).** *Push signed model* → accepted (v2). *Push tampered model* → rejected,
   fleet stays on v2. *Roll back* → v1.
5. **Evidence (30 s).** Show `artifacts/report/report.md`: real NASA IMS bearing, 74.8 h warning before the
   outer-race failure; ROI range −34% to −99% with the assumptions listed.

Recording the video: same flow, 1500×1000 window, speed 8; reset aircraft between takes.
