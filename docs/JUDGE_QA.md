# Judge Q&A

**Is any of this on real aircraft data?** No. The live demo is simulated. Real-data evidence: NASA IMS
bearing run-to-failure vibration (74.8 h warning before the outer-race failure) and NASA C-MAPSS turbofan
RUL (simulated engines, public benchmark).

**Why should we trust a 98% cost saving?** Don't take it alone: it uses the simulator's near-perfect
detection and the deck's $150k/h AOG figure. Under harsh assumptions (67% detection, 10% of the warning,
5 false alarms per 1,000 windows, $10k/h AOG) it is −34%; the table shows the range.

**What happens when a sensor fails?** It is flagged SENSOR_FAULT within 1–2 windows (drift ~10), masked
using the same-flight-phase reading or a context model, and the component models keep running. 0 false
component advisories in our tests.

**Take-off looks nothing like cruise; how do you avoid false alarms?** Flight phase, outside air temperature
and altitude are model inputs. Without them the anomaly gate is open 98.6% of the time on healthy flights;
with them 0.0%.

**Has it run on a Jetson / TensorRT?** Not yet. The graphs use only TensorRT-supported operators and the
build kit is ready; all timings so far are CPU (~1.2 ms per window).

**How do you explain a prediction to an engineer?** Each advisory carries computed evidence: e.g. the
envelope-spectrum peak at the bearing outer-race defect frequency, plus the most abnormal signals in σ.

**Certification?** Not addressed. Decisions follow a configurable prototype policy; a real system would go
through DO-178C/DO-254-style assurance and the regulator's AI guidance.

**Weaknesses?** One real bearing test rig only; no flight data; one oil-contamination run in six is still
misnamed at its first classified alert; RUL intervals for bearing wear are wider than needed (coverage 0.99).

**Does federated learning actually help?** For anomaly detection it did not in our tests. For fault
classification it does: an aircraft that has seen only 2 of 5 fault types recognises the other 3 with 91%
accuracy after federated averaging, without any raw data leaving the aircraft (simulated fleet).
