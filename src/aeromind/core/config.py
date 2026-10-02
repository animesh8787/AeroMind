"""Shared constants for the simulated sensor suite and the edge pipeline."""

FS = 1024  # Hz: vibration / acoustic / electrical sample rate
N_HIGH = 1024  # samples per window for vibration and acoustic (1 s)
N_ELEC = 512  # samples per window for the current channel (0.5 s)
N_SLOW = 16  # samples per window for voltage, temperature, pressure, oil debris
ELEC_HZ = 60.0  # electrical fundamental

# Each simulated window is one 1 s snapshot taken once per HOURS_PER_WINDOW of operation.
HOURS_PER_WINDOW = 0.5
RUL_CAP_WINDOWS = 300  # RUL targets are clipped here; beyond it "healthy enough"

HEALTHY = "healthy"
FAULT_MODES = (
    "bearing_wear",
    "oil_contamination",
    "overheating",
    "electrical_fault",
    "pressure_leak",
)

# What a ground-link would carry if raw data were streamed (float32 samples).
RAW_SAMPLES_PER_WINDOW = 2 * N_HIGH + N_ELEC + 4 * N_SLOW
RAW_BYTES_PER_WINDOW = 4 * RAW_SAMPLES_PER_WINDOW

TREND_WINDOW = 10  # windows of history used for trend features
