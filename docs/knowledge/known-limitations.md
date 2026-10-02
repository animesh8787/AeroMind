# Known limitations

- All aircraft, sensor streams and failures in the demo are simulated. Results describe software
  behaviour on a synthetic simulator, not performance on aircraft.
- No certification, regulatory approval, OEM approval or airline deployment exists. Decision policies
  are prototype policies.
- ACARS: messages are only sized and encoded to fit 220 characters; there is no real datalink.
  There is no SWIM connection and no real MRO integration; work orders are in-memory drafts.
- Edge hardware: no Jetson, FPGA or Raspberry Pi measurement exists in this repository unless a
  result file says otherwise. Latency figures are from a development CPU.
- RUL intervals are estimates; coverage is not guaranteed. Slow in-range sensor drift is not detected.
- Federated learning showed no advantage for anomaly detection in the tested setup; it helps only in
  the rare-fault classification experiment.
- The LLM copilot only explains deterministic outputs. Its text is AI-generated, can be wrong, and
  needs technician review. With no LLM available a deterministic template response is used.
