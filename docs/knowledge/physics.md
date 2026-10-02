# Physics-backed evidence

Evidence strings in an advisory are **computed from the raw signals**, independently of the
classifier's output.

- **Bearing faults:** envelope analysis. The vibration signal is band-passed around the structural
  resonance, the Hilbert envelope is taken, and the envelope spectrum is searched for peaks at the
  bearing's characteristic defect frequencies (BPFO outer race, BPFI inner race, BSF ball spin) and
  harmonics. These frequencies follow from the assumed bearing geometry (9 balls, d/D = 0.2066,
  0° contact angle) and the shaft speed. Shaft speed is derived from the load in the operating
  context, as a tachometer would provide on an engine.
- **Electrical faults:** total harmonic distortion of the current from its 3rd and 5th harmonic
  relative to the 60 Hz fundamental.

The bearing geometry is a prototype assumption, not any real engine's bearing. On the NASA IMS
dataset the same envelope technique is applied to real vibration records.
