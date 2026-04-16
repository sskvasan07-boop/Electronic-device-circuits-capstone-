# =============================================================================
# diagnostic_dashboard.py — OptoScan Standalone Diagnostic Dashboard
# =============================================================================
# Fully self-contained. No external API calls. Uses an embedded expert knowledge
# base (OPTO_SCAN_LOGIC) to interpret every IR reflectance ADC reading.
#
# Run: streamlit run diagnostic_dashboard.py
# =============================================================================

from __future__ import annotations
import os
import time
import queue
import threading
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go

# ── Optional serial import ────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

# =============================================================================
# EXPERT KNOWLEDGE BASE — hard-coded, no API dependency
# =============================================================================
OPTO_SCAN_LOGIC = [
    {
        "range":      (0, 150),
        "result":     "SIGNAL ERROR",
        "physics":    "Zero to low photon return detected by the receiver.",
        "pathology":  "Indicates an optical disconnect or poor coupling with the skin surface.",
        "suggestion": "Re-position the probe head; ensure the sensor is flush against the tissue.",
        "color":      "#9CA3AF",
        "glow":       "rgba(156,163,175,0.15)",
        "bg":         "linear-gradient(135deg,#111827,#1F2937)",
        "border":     "#374151",
        "icon":       "⚠️",
        "severity":   "error",
    },
    {
        "range":      (151, 350),
        "result":     "SUSPECT: HEMATOMA",
        "physics":    "High IR absorption by subsurface chromophores (hemoglobin).",
        "pathology":  "Blood pooled outside vessels causes localized pressure and localized trauma signature.",
        "suggestion": "Apply a cold compress to the area; recommend monitoring for expanding swelling.",
        "color":      "#EF4444",
        "glow":       "rgba(239,68,68,0.20)",
        "bg":         "linear-gradient(135deg,#1A0A0A,#2D0D0D)",
        "border":     "#7F1D1D",
        "icon":       "🔴",
        "severity":   "critical",
    },
    {
        "range":      (351, 450),
        "result":     "VASCULAR CONGESTION",
        "physics":    "Moderate absorption due to high capillary density in the sample volume.",
        "pathology":  "Suggested mild localized inflammation or active tissue irritation response.",
        "suggestion": "Compare the reading with a control site (e.g., opposite arm); monitor for local heat.",
        "color":      "#F97316",
        "glow":       "rgba(249,115,22,0.18)",
        "bg":         "linear-gradient(135deg,#1A0F05,#2A1800)",
        "border":     "#7C2D12",
        "icon":       "🟠",
        "severity":   "warning",
    },
    {
        "range":      (451, 600),
        "result":     "NORMAL SKIN",
        "physics":    "Balanced, healthy scattering of photons off organized dermal collagen.",
        "pathology":  "Skin architecture is healthy, providing appropriate structural support and elasticity.",
        "suggestion": "No subsurface anomaly detected; system is calibrated. Routine monitoring only.",
        "color":      "#10B981",
        "glow":       "rgba(16,185,129,0.18)",
        "bg":         "linear-gradient(135deg,#021A0D,#053D1C)",
        "border":     "#065F46",
        "icon":       "✅",
        "severity":   "normal",
    },
    {
        "range":      (601, 750),
        "result":     "EARLY FIBROSIS",
        "physics":    "Increased backscatter detected, indicating higher collagen density.",
        "pathology":  "Initial tissue thickening is reducing elasticity; monitor for potential hardening.",
        "suggestion": "Document the site texture and any changes. Moisturize to maintain skin elasticity.",
        "color":      "#F59E0B",
        "glow":       "rgba(245,158,11,0.18)",
        "bg":         "linear-gradient(135deg,#1A1205,#2A1E00)",
        "border":     "#78350F",
        "icon":       "🟡",
        "severity":   "moderate",
    },
    {
        "range":      (751, 950),
        "result":     "SUSPECT: DENSE FIBROMA/SCAR",
        "physics":    "Intense backscatter from disorganized, highly dense collagen bundles.",
        "pathology":  "Dense subsurface bundles (scars/keloids) can restrict tissue movement and cause itching.",
        "suggestion": "High probability of dense anomaly. Dermatological consult recommended for characterization.",
        "color":      "#F43F5E",
        "glow":       "rgba(244,63,94,0.22)",
        "bg":         "linear-gradient(135deg,#1A080B,#2D0E14)",
        "border":     "#9F1239",
        "icon":       "🔶",
        "severity":   "high",
    },
    {
        "range":      (951, 1023),
        "result":     "OPTICAL SATURATION",
        "physics":    "Specular reflection or direct glare is overwhelming the receiver diode.",
        "pathology":  "Surface artifacts (sweat/oil/shiny surface) are masking subsurface data.",
        "suggestion": "Error condition. Clean the probe tip and the skin surface thoroughly with alcohol.",
        "color":      "#A855F7",
        "glow":       "rgba(168,85,247,0.20)",
        "bg":         "linear-gradient(135deg,#0D0A1A,#1A1030)",
        "border":     "#6D28D9",
        "icon":       "💜",
        "severity":   "error",
    },
]


def get_logic(adc: int) -> dict:
    """Return the diagnostic entry matching an ADC value 0–1023."""
    adc = max(0, min(1023, int(adc)))
    for entry in OPTO_SCAN_LOGIC:
        lo, hi = entry["range"]
        if lo <= adc <= hi:
            return entry
    return OPTO_SCAN_LOGIC[0]   # fallback


# =============================================================================
# SERIAL BACKGROUND THREAD
# =============================================================================
def _serial_worker(port: str, baud: int, q: queue.Queue, stop: threading.Event) -> None:
    """Background thread: reads lines from serial and enqueues them."""
    try:
        with serial.Serial(port, baud, timeout=1.0) as ser:
            q.put("__CONNECTED__")
            while not stop.is_set():
                try:
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if raw:
                        q.put(raw)
                except Exception:
                    pass
    except Exception as exc:
        q.put(f"__ERROR__{exc}")


def start_serial(port: str, baud: int) -> tuple[queue.Queue, threading.Event, threading.Thread]:
    q     = queue.Queue(maxsize=50)
    stop  = threading.Event()
    t     = threading.Thread(target=_serial_worker, args=(port, baud, q, stop), daemon=True)
    t.start()
    return q, stop, t


def stop_serial() -> None:
    if "serial_stop" in st.session_state and st.session_state.serial_stop:
        st.session_state.serial_stop.set()
    for key in ("serial_thread", "serial_stop", "serial_queue"):
        st.session_state.pop(key, None)


def parse_serial_line(line: str) -> int | None:
    """Parse 'ADC,Label' or plain integer from serial line."""
    parts = line.split(",")
    try:
        return int(parts[0].strip())
    except ValueError:
        return None


# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="OptoScan — Diagnostic Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #070B14 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0B1120 !important;
    border-right: 1px solid #1E2D44;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
.stSlider > div { padding: 0 !important; }
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
[data-testid="stMetric"] {
    background: #0F1729;
    border: 1px solid #1E2D44;
    border-radius: 12px;
    padding: 14px !important;
}
.block-container { padding: 1.2rem 2rem 2rem !important; }
hr { border-color: #1E2D44 !important; }

/* ── Header ─────────────────────────────────────────────────── */
.opto-header {
    display: flex; align-items: center; gap: 14px;
    border-bottom: 1px solid #1E2D44;
    padding-bottom: 14px; margin-bottom: 6px;
}
.opto-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #00D4FF, #7B2FFF);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
}
.opto-title-block h1 {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(120deg, #00D4FF, #7B2FFF 80%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1;
}
.opto-title-block p {
    font-size: 0.7rem; color: #475569;
    letter-spacing: 0.14em; text-transform: uppercase; margin: 2px 0 0;
}
.opto-badge {
    margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}

/* ── Result Banner ───────────────────────────────────────────── */
.result-banner {
    border-radius: 16px; padding: 22px 28px;
    margin: 18px 0 8px;
    display: flex; align-items: center; justify-content: space-between;
    position: relative; overflow: hidden;
    border: 1px solid;
}
.result-banner::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    opacity: 0.065;
    background: radial-gradient(circle at 20% 50%, white, transparent 70%);
}
.result-icon   { font-size: 2.4rem; line-height: 1; }
.result-label  { font-size: 0.62rem; color: #64748B; letter-spacing: 0.18em;
                 text-transform: uppercase; margin-bottom: 2px; }
.result-text   { font-size: 1.75rem; font-weight: 900; letter-spacing: 0.06em;
                 font-family: 'Inter', sans-serif; }
.result-meta   { font-size: 0.7rem; color: #94A3B8; margin-top: 4px; }
.result-adc    { text-align: right; }
.result-adc-val {
    font-size: 3.5rem; font-weight: 900; line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.result-adc-label { font-size: 0.65rem; color: #64748B; letter-spacing: 0.12em;
                    text-transform: uppercase; margin-top: 2px; }

/* ── The 4 Diagnostic Cards ──────────────────────────────────── */
.diag-card {
    border-radius: 14px; padding: 20px 22px;
    border: 1px solid;
    position: relative; overflow: hidden;
    height: 100%;
}
.diag-card::after {
    content: ''; position: absolute;
    bottom: -30px; right: -30px; width: 90px; height: 90px;
    border-radius: 50%; opacity: 0.08;
    background: radial-gradient(circle, white, transparent);
}
.card-label {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.card-dot {
    width: 6px; height: 6px; border-radius: 50%; display: inline-block;
}
.card-text {
    font-size: 0.88rem; color: #CBD5E1; line-height: 1.75;
    font-weight: 400;
}

/* ── Severity Badge ──────────────────────────────────────────── */
.sev-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; border: 1px solid;
}

/* ── History Log ─────────────────────────────────────────────── */
.history-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 14px; border-radius: 8px;
    border: 1px solid #1E2D44; margin-bottom: 5px;
    background: #0B1120;
    font-size: 0.75rem; color: #94A3B8;
}
.history-adc  { font-family: 'JetBrains Mono', monospace; font-weight: 700; min-width: 40px; }
.history-result { font-weight: 600; flex: 1; }
.history-time { color: #475569; font-size: 0.67rem; white-space: nowrap; }

/* ── Scale Bar ───────────────────────────────────────────────── */
.scale-bar {
    height: 8px; border-radius: 4px; width: 100%;
    background: linear-gradient(to right,
        #374151 0%,         /* 0–150   signal error  */
        #374151 14.6%,
        #7F1D1D 14.6%,      /* 151–350 hematoma    */
        #7F1D1D 34.2%,
        #7C2D12 34.2%,      /* 351–450 vascular    */
        #7C2D12 43.9%,
        #065F46 43.9%,      /* 451–600 normal      */
        #065F46 58.7%,
        #78350F 58.7%,      /* 601–750 fibrosis    */
        #78350F 73.3%,
        #9F1239 73.3%,      /* 751–950 fibroma     */
        #9F1239 92.9%,
        #6D28D9 92.9%,      /* 951–1023 saturation */
        #6D28D9 100%
    );
    margin-bottom: 4px;
}
.scale-labels {
    display: flex; justify-content: space-between;
    font-size: 0.58rem; color: #475569; font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INIT
# =============================================================================
if "adc_value"      not in st.session_state: st.session_state.adc_value      = 512
if "manual_slider"  not in st.session_state: st.session_state.manual_slider   = 512
if "exact_input"    not in st.session_state: st.session_state.exact_input     = 512
if "history"        not in st.session_state: st.session_state.history         = []
if "serial_active"  not in st.session_state: st.session_state.serial_active   = False
if "serial_status"  not in st.session_state: st.session_state.serial_status   = "Disconnected"
if "auto_refresh"   not in st.session_state: st.session_state.auto_refresh    = False
if "refresh_ms"     not in st.session_state: st.session_state.refresh_ms      = 500
if "reading_count"  not in st.session_state: st.session_state.reading_count   = 0


# =============================================================================
# HELPER: record a reading into history
# =============================================================================
def _add_reading(val: int, src: str = "manual") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.insert(0, (val, get_logic(val)["result"], ts, src))
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]
    st.session_state.reading_count += 1


# on_change callbacks — fire BEFORE main body re-renders
def _on_slider_change() -> None:
    """Slider dragged → propagate to adc_value and sync number box."""
    val = int(st.session_state.manual_slider)
    st.session_state.adc_value  = val
    st.session_state.exact_input = val   # keep number box in sync
    _add_reading(val, "manual")


def _on_input_change() -> None:
    """Number box edited → propagate to adc_value and sync slider."""
    val = int(st.session_state.exact_input)
    st.session_state.adc_value   = val
    st.session_state.manual_slider = val  # keep slider in sync
    _add_reading(val, "manual")


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 0 8px;'>
      <div style='font-size:1rem;font-weight:800;color:#E2E8F0;letter-spacing:.04em;'>
        🔬 OptoScan
      </div>
      <div style='font-size:0.62rem;color:#475569;letter-spacing:.14em;text-transform:uppercase;margin-top:2px;'>
        Diagnostic Dashboard v2.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Manual Input ─────────────────────────────────────────────────────────
    st.markdown("**📐 Manual ADC Input**")

    # Callbacks keep slider + number box in perfect sync via session_state.
    # on_change fires FIRST → adc_value is always fresh by the time the
    # main body renders the gauge and cards.
    st.slider(
        "ADC Value", 0, 1023,
        key="manual_slider",
        on_change=_on_slider_change,
        disabled=st.session_state.serial_active,
        help="Drag to set a sensor reading (0–1023). Disabled while serial is live.",
    )
    st.number_input(
        "Or type exact value", 0, 1023,
        key="exact_input",
        on_change=_on_input_change,
        disabled=st.session_state.serial_active,
        help="Type any value 0–1023 and press Enter.",
    )

    st.markdown("---")

    # ── Serial Port ───────────────────────────────────────────────────────────
    st.markdown("**🔌 HC-05 Bluetooth Serial**")

    if not _SERIAL_OK:
        st.warning("Install pyserial:\n`pip install pyserial`")
    else:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            ports = ["(No ports found)"]

        col_p, col_b = st.columns([2, 1])
        with col_p:
            selected_port = st.selectbox("COM Port", ports, key="com_port")
        with col_b:
            baud_rate = st.selectbox("Baud", [9600, 19200, 38400, 57600, 115200], index=0, key="baud")

        if not st.session_state.serial_active:
            if st.button("⚡ Connect HC-05", key="serial_connect_btn", type="primary",
                         use_container_width=True):
                if selected_port and selected_port != "(No ports found)":
                    q, stop_ev, t = start_serial(selected_port, baud_rate)
                    st.session_state.serial_queue  = q
                    st.session_state.serial_stop   = stop_ev
                    st.session_state.serial_thread = t
                    st.session_state.serial_active = True
                    st.session_state.serial_status = f"Connecting → {selected_port}"
                    st.session_state.auto_refresh  = True
                    st.rerun()
        else:
            st.success(f"● {st.session_state.serial_status}")
            if st.button("✕ Disconnect", key="serial_disconnect_btn",
                         use_container_width=True):
                stop_serial()
                st.session_state.serial_active = False
                st.session_state.serial_status = "Disconnected"
                st.session_state.auto_refresh  = False
                st.rerun()

    st.markdown("---")

    # ── Auto-Refresh ──────────────────────────────────────────────────────────
    st.markdown("**🔄 Auto-Refresh**")
    ar_toggle = st.toggle("Enable auto-refresh", value=st.session_state.auto_refresh,
                          key="ar_toggle_widget")
    st.session_state.auto_refresh = ar_toggle
    if ar_toggle:
        st.session_state.refresh_ms = st.select_slider(
            "Interval (ms)", [250, 500, 750, 1000, 2000],
            value=st.session_state.refresh_ms,
        )

    st.markdown("---")

    # ── Reference Scale ───────────────────────────────────────────────────────
    st.markdown("**📊 ADC Range Legend**")
    for entry in OPTO_SCAN_LOGIC:
        lo, hi = entry["range"]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;"
            f"font-size:0.69rem;color:#94A3B8;margin-bottom:3px;'>"
            f"<span style='width:10px;height:10px;border-radius:2px;"
            f"background:{entry['color']};display:inline-block;flex-shrink:0;'></span>"
            f"<span style='color:{entry['color']};font-weight:600;min-width:70px;'>"
            f"{lo}–{hi}</span>{entry['result']}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("🔬 For research and prototyping only.\nNot approved for clinical diagnosis.")


# =============================================================================
# DRAIN SERIAL QUEUE → update session state
# =============================================================================
if st.session_state.serial_active and "serial_queue" in st.session_state:
    q: queue.Queue = st.session_state.serial_queue
    latest_val = None
    latest_raw = None
    while not q.empty():
        raw = q.get_nowait()
        if raw == "__CONNECTED__":
            st.session_state.serial_status = f"Live · {st.session_state.get('com_port','')}"
        elif raw.startswith("__ERROR__"):
            st.session_state.serial_status = f"Error: {raw[9:]}"
            st.session_state.serial_active = False
            stop_serial()
        else:
            val = parse_serial_line(raw)
            if val is not None:
                latest_val = val
                latest_raw = raw

    if latest_val is not None:
        st.session_state.adc_value = latest_val
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.history.insert(0, (latest_val, get_logic(latest_val)["result"], ts, "serial"))
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[:20]
        st.session_state.reading_count += 1


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
adc     = int(st.session_state.adc_value)
logic   = get_logic(adc)
pct     = adc / 1023 * 100

# ── Header ────────────────────────────────────────────────────────────────────
serial_badge = (
    f"<span class='sev-badge' style='color:#10B981;border-color:#065F46;"
    f"background:#021A0D;'>● Serial Live</span>"
    if st.session_state.serial_active else
    f"<span class='sev-badge' style='color:#6B7280;border-color:#374151;"
    f"background:#111827;'>○ Serial Off</span>"
)
readings_badge = (
    f"<span class='sev-badge' style='color:#60A5FA;border-color:#1D4ED8;"
    f"background:#0C1A3D;'>📈 {st.session_state.reading_count} readings</span>"
)
st.markdown(f"""
<div class='opto-header'>
  <div class='opto-logo'>🔬</div>
  <div class='opto-title-block'>
    <h1>OptoScan Diagnostic Dashboard</h1>
    <p>850 nm IR Reflectance · BPW34 Photodiode · Expert Knowledge Base v2.0</p>
  </div>
  <div class='opto-badge'>
    {readings_badge}
    {serial_badge}
  </div>
</div>
""", unsafe_allow_html=True)


# ── Result Banner ─────────────────────────────────────────────────────────────
sev_colors = {
    "normal":   ("#10B981", "#021A0D"),
    "moderate": ("#F59E0B", "#1A1205"),
    "warning":  ("#F97316", "#1A0F05"),
    "high":     ("#F43F5E", "#1A080B"),
    "critical": ("#EF4444", "#1A0A0A"),
    "error":    ("#9CA3AF", "#111827"),
}
sev_c, sev_bg = sev_colors.get(logic["severity"], ("#9CA3AF", "#111827"))

st.markdown(f"""
<div class='result-banner'
     style='background:{logic['bg']};border-color:{logic['border']};
            box-shadow: 0 0 40px {logic['glow']};'>
  <div style='display:flex;align-items:center;gap:18px;'>
    <span class='result-icon'>{logic['icon']}</span>
    <div>
      <div class='result-label'>Primary Diagnosis</div>
      <div class='result-text' style='color:{logic['color']};'>{logic['result']}</div>
      <div class='result-meta'>
        ADC range {logic['range'][0]}–{logic['range'][1]} &nbsp;|&nbsp;
        Severity: <span style='color:{sev_c};font-weight:700;'>{logic['severity'].upper()}</span>
      </div>
    </div>
  </div>
  <div class='result-adc' style='color:{logic['color']};'>
    <div class='result-adc-val'>{adc}</div>
    <div class='result-adc-label'>ADC counts / {pct:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── ADC Scale Bar ─────────────────────────────────────────────────────────────
pointer_pct = adc / 1023 * 100
st.markdown(f"""
<div style='margin-bottom:16px;'>
  <div style='position:relative;height:14px;margin-bottom:2px;'>
    <div class='scale-bar'></div>
    <div style='position:absolute;top:-2px;left:{pointer_pct}%;
                transform:translateX(-50%);
                width:3px;height:14px;background:{logic['color']};
                border-radius:2px;box-shadow:0 0 8px {logic['color']};'></div>
  </div>
  <div class='scale-labels'>
    <span>0</span><span>150</span><span>350</span><span>450</span>
    <span>600</span><span>750</span><span>950</span><span>1023</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Plotly Gauge + 4 Cards ────────────────────────────────────────────────────
gauge_col, cards_col = st.columns([1, 1.6], gap="large")

with gauge_col:
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = adc,
        domain= {"x": [0, 1], "y": [0, 1]},
        title = {
            "text": (
                "RAW ADC VALUE<br>"
                "<span style='font-size:0.65em;color:#64748B;'>"
                "850nm · BPW34 · 0–1023</span>"
            ),
            "font": {"size": 15, "color": "#94A3B8", "family": "Inter"},
        },
        number= {
            "font": {"size": 64, "color": logic["color"], "family": "JetBrains Mono"},
        },
        gauge = {
            "axis": {
                "range": [0, 1023],
                "nticks": 8,
                "tickwidth": 1,
                "tickcolor": "#334155",
                "tickfont": {"color": "#475569", "size": 9, "family": "JetBrains Mono"},
            },
            "bar":  {"color": logic["color"], "thickness": 0.24},
            "bgcolor": "#0F1729",
            "borderwidth": 2,
            "bordercolor": "#1E2D44",
            "steps": [
                {"range": [0,   150],  "color": "#111827"},
                {"range": [151, 350],  "color": "#1A0A0A"},
                {"range": [351, 450],  "color": "#1A0F05"},
                {"range": [451, 600],  "color": "#021A0D"},
                {"range": [601, 750],  "color": "#1A1205"},
                {"range": [751, 950],  "color": "#1A080B"},
                {"range": [951, 1023], "color": "#0D0A1A"},
            ],
            "threshold": {
                "line": {"color": logic["color"], "width": 5},
                "thickness": 0.82,
                "value": adc,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#070B14",
        plot_bgcolor ="#070B14",
        font         = {"family": "Inter", "color": "#E2E8F0"},
        height       = 340,
        margin       = {"t": 70, "b": 10, "l": 30, "r": 30},
    )
    st.plotly_chart(fig, use_container_width=True, key="main_gauge")

    # Mini metrics below gauge
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Reflectance", f"{adc/1023:.4f}", help="Normalized 0.0–1.0")
    with m2:
        lo, hi = logic["range"]
        st.metric("Range", f"{lo}–{hi}")
    with m3:
        st.metric("Severity", logic["severity"].title())


# ── Four Diagnostic Cards ─────────────────────────────────────────────────────
CARD_DEFS = [
    {
        "label":   "Primary Result",
        "icon":    "🎯",
        "key":     "result",
        "color":   logic["color"],
        "dot":     logic["color"],
        "bg":      logic["bg"],
        "border":  logic["border"],
        "glow":    logic["glow"],
        "bold":    True,
    },
    {
        "label":   "Mechanism — What is Happening?",
        "icon":    "⚛️",
        "key":     "physics",
        "color":   "#60A5FA",
        "dot":     "#3B82F6",
        "bg":      "linear-gradient(135deg,#060D1F,#0C1A3D)",
        "border":  "#1E3A5F",
        "glow":    "rgba(96,165,250,0.12)",
        "bold":    False,
    },
    {
        "label":   "Clinical Impact — Tissue Effect",
        "icon":    "🩺",
        "key":     "pathology",
        "color":   "#FB923C",
        "dot":     "#F97316",
        "bg":      "linear-gradient(135deg,#0D0900,#1E1100)",
        "border":  "#431407",
        "glow":    "rgba(251,146,60,0.12)",
        "bold":    False,
    },
    {
        "label":   "Recommended Action",
        "icon":    "💊",
        "key":     "suggestion",
        "color":   "#34D399",
        "dot":     "#10B981",
        "bg":      "linear-gradient(135deg,#00110A,#001F12)",
        "border":  "#064E3B",
        "glow":    "rgba(52,211,153,0.10)",
        "bold":    False,
    },
]

with cards_col:
    top_l, top_r = st.columns(2)
    bot_l, bot_r = st.columns(2)
    card_cols    = [top_l, top_r, bot_l, bot_r]

    for col, cdef in zip(card_cols, CARD_DEFS):
        text = logic[cdef["key"]]
        txt_style = (
            f"font-size:1rem;font-weight:800;color:{cdef['color']};letter-spacing:0.04em;"
            if cdef["bold"] else
            "font-size:0.86rem;color:#CBD5E1;line-height:1.75;"
        )
        with col:
            st.markdown(f"""
            <div class='diag-card'
                 style='background:{cdef['bg']};
                        border-color:{cdef['border']};
                        box-shadow:0 0 24px {cdef['glow']}; height:160px;'>
              <div class='card-label' style='color:{cdef['color']};'>
                <span class='card-dot' style='background:{cdef['dot']};'></span>
                {cdef['icon']} {cdef['label']}
              </div>
              <div class='{('card-text' if not cdef['bold'] else '')}'>
                <span style='{txt_style}'>{text}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# READING HISTORY LOG
# =============================================================================
st.markdown("---")
hist_col, spacer = st.columns([3, 1])
with hist_col:
    st.markdown(
        "<div style='font-size:0.75rem;color:#64748B;font-weight:700;"
        "letter-spacing:0.14em;text-transform:uppercase;margin-bottom:8px;'>"
        "📁 Reading History (last 20)</div>",
        unsafe_allow_html=True,
    )
    if not st.session_state.history:
        st.caption("No readings yet. Move the slider or connect the HC-05 module.")
    else:
        for val, result, ts, src in st.session_state.history[:20]:
            entry_logic = get_logic(val)
            src_icon    = "📡" if src == "serial" else "🖱️"
            st.markdown(f"""
            <div class='history-row' style='border-color:{entry_logic['border']};'>
              <span class='history-adc' style='color:{entry_logic['color']};'>{val}</span>
              <span class='history-result' style='color:{entry_logic['color']};'>{result}</span>
              <span style='color:#475569;font-size:0.7rem;'>{src_icon} {src}</span>
              <span class='history-time'>{ts}</span>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.history:
        if st.button("🗑️ Clear History", key="clear_hist"):
            st.session_state.history       = []
            st.session_state.reading_count = 0
            st.rerun()

with spacer:
    # Quick stats
    if st.session_state.history:
        vals = [h[0] for h in st.session_state.history]
        st.metric("Max ADC", max(vals))
        st.metric("Min ADC", min(vals))
        st.metric("Avg ADC", f"{sum(vals)//len(vals)}")


# =============================================================================
# AUTO-REFRESH
# =============================================================================
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_ms / 1000)
    st.rerun()
