# =============================================================================
# config.py — OptoScan System Configuration
# =============================================================================
# All hardware and software constants live here.
# Change COM_PORT to match the port assigned to your HC-05 Bluetooth adapter.
# =============================================================================

# ── Serial / Bluetooth ────────────────────────────────────────────────────────
COM_PORT        = "COM3"        # Default HC-05 pairing port — override in sidebar
BAUD_RATE       = 9600          # HC-05 default baud rate
READ_TIMEOUT    = 1.0           # seconds — serial read timeout
RECONNECT_BASE  = 1.0           # seconds — first reconnect wait
RECONNECT_MAX   = 30.0          # seconds — maximum reconnect back-off ceiling
QUEUE_MAX_SIZE  = 200           # samples buffered between reader and UI

# ── ADC / Sensor ─────────────────────────────────────────────────────────────
ADC_MIN         = 0             # Arduino ADC floor
ADC_MAX         = 1023          # Arduino ADC ceiling (10-bit)
SENSOR_HZ       = 10            # Expected sample rate from Arduino sketch (Hz)

# ── Heuristic Classification Thresholds ──────────────────────────────────────
# These map raw reflectance (ADC) to tissue type.
# Low ADC  → high absorption  → Hematoma (dense vascular / blood pooling)
# Mid ADC  → normal scatter   → Normal tissue
# High ADC → high reflectance → Fibrous / Scar tissue (collagenous)
HEMATOMA_UPPER  = 340
NORMAL_UPPER    = 680
# Anything above NORMAL_UPPER is classified as Fibrous/Scar

# ── ML Model ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = ""         # Leave empty to use heuristic mode
                                # Set to your .pkl / .h5 / .onnx path to enable ML

# ── UI / Dashboard ───────────────────────────────────────────────────────────
UI_REFRESH_MS       = 200       # Dashboard polling interval (milliseconds)
ROLLING_WINDOW      = 60        # Number of recent samples shown in time-series
APP_TITLE           = "OptoScan — Optical Biopsy Diagnostic System"
APP_VERSION         = "v1.0.0"

# ── Diagnosis Labels ─────────────────────────────────────────────────────────
LABEL_NORMAL    = "Normal"
LABEL_HEMATOMA  = "Hematoma"
LABEL_FIBROUS   = "Fibrous / Scar"

# ── Colour Palette (used in Plotly + CSS) ────────────────────────────────────
COLOR_NORMAL    = "#00C896"     # Teal-green
COLOR_HEMATOMA  = "#FF4C6A"     # Clinical red
COLOR_FIBROUS   = "#F5A623"     # Amber/orange
COLOR_GAUGE_BG  = "#0D1B2A"     # Deep navy
COLOR_ACCENT    = "#00D4FF"     # Cyan accent
