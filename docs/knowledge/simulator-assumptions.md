# Simulator assumptions

- Every number the system sees comes from a synthetic simulator or from public datasets used offline
  (NASA C-MAPSS, NASA IMS). There is no airline data and no flight data.
- Fault signatures are strong and cleanly separable, so detection is easier than it would be on real
  hardware.
- One window represents 1 s of data per 0.5 flight hours. RUL targets are capped at 300 windows
  (150 flight hours); beyond that the component counts as healthy enough.
- Flight phases (taxi, take-off, climb, cruise, descent, taxi) change load and ambient context;
  the anomaly gate is trained to use them.
- Aircraft registrations `VT-AMA01`..`VT-AMA06` are fictional.
- Costs (AOG cost per hour, part cost, labour) are labelled illustrative assumptions, not airline data.
