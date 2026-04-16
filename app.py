# =============================================================================
# app.py — OptoScan Clinical Monitoring Dashboard
# =============================================================================
# Run with:  streamlit run app.py
#
# The page auto-refreshes every UI_REFRESH_MS milliseconds using
# st.rerun() inside a time-controlled loop, continuously pulling the
# latest ADC sample from the BluetoothReader queue.
# =============================================================================

from __future__ import annotations
import time
import math
import os
import logging
import tempfile
from collections import deque

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from config import (
    BAUD_RATE, ADC_MIN, ADC_MAX,
    UI_REFRESH_MS, ROLLING_WINDOW, APP_TITLE, APP_VERSION,
    COLOR_NORMAL, COLOR_HEMATOMA, COLOR_FIBROUS,
    COLOR_GAUGE_BG, COLOR_ACCENT,
    LABEL_NORMAL, LABEL_HEMATOMA, LABEL_FIBROUS,
)
from bluetooth_reader import BluetoothReader
from inference_engine import InferenceEngine
from gemini_agent import GeminiAgent

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="OptoScan",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: #060D18;
      color: #E2E8F0;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0A1628 0%, #0D1F3C 100%);
      border-right: 1px solid #1E3A5F;
  }
  section[data-testid="stSidebar"] label {
      color: #94A3B8 !important;
      font-size: 0.78rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
  }

  /* ── Metric cards ── */
  .metric-card {
      background: linear-gradient(135deg, #0F1E35 0%, #0A192F 100%);
      border: 1px solid #1E3A5F;
      border-radius: 12px;
      padding: 20px 24px;
      text-align: center;
      position: relative;
      overflow: hidden;
  }
  .metric-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, #00D4FF, #0080FF);
  }
  .metric-card .label {
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #64748B;
      margin-bottom: 6px;
  }
  .metric-card .value {
      font-family: 'Space Mono', monospace;
      font-size: 2.8rem;
      font-weight: 700;
      color: #00D4FF;
      line-height: 1;
  }
  .metric-card .unit {
      font-size: 0.8rem;
      color: #64748B;
      margin-top: 4px;
  }

  /* ── Diagnosis card ── */
  .dx-card {
      border-radius: 12px;
      padding: 24px;
      text-align: center;
      border: 1px solid rgba(255,255,255,0.08);
      margin-bottom: 16px;
  }
  .dx-label {
      font-size: 1.8rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
  }
  .dx-sub {
      font-size: 0.75rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #94A3B8;
  }

  /* ── Status badge ── */
  .badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
  }
  .badge-connected  { background: #064E3B; color: #34D399; border: 1px solid #059669; }
  .badge-disconnected { background: #450A0A; color: #FCA5A5; border: 1px solid #DC2626; }
  .badge-simulation { background: #1E1B4B; color: #A78BFA; border: 1px solid #7C3AED; }
  .badge-heuristic  { background: #1C1917; color: #A8A29E; border: 1px solid #78716C; }
  .badge-ml         { background: #0C4A6E; color: #38BDF8; border: 1px solid #0369A1; }

  /* ── Section headers ── */
  .section-header {
      font-size: 0.68rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #475569;
      padding-bottom: 8px;
      border-bottom: 1px solid #1E293B;
      margin-bottom: 16px;
  }

  /* ── Footer ── */
  .footer {
      font-size: 0.70rem;
      color: #334155;
      text-align: center;
      padding-top: 24px;
      letter-spacing: 0.05em;
  }

  /* ── Streamlit overrides ── */
  div[data-testid="stMetric"] { background: transparent; }
  div.stButton > button {
      width: 100%;
      border-radius: 8px;
      font-weight: 600;
      letter-spacing: 0.05em;
  }
  div.stButton > button:hover { border-color: #00D4FF; color: #00D4FF; }
  .stSlider > div > div > div > div { background: #00D4FF; }
  div[data-testid="stProgress"] > div { background: #00D4FF; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialisation
# =============================================================================

def _init_state() -> None:
    defaults = {
        "reader":               None,
        "engine":               InferenceEngine(),
        "connected":            False,
        "baud_rate":            BAUD_RATE,
        "history":              deque(maxlen=ROLLING_WINDOW),
        "labels":               deque(maxlen=ROLLING_WINDOW),
        "timestamps":           deque(maxlen=ROLLING_WINDOW),
        "last_result":          None,
        "session_start":        time.time(),
        "total_samples":        0,
        "drop_count":           0,
        "last_value":           512,
        "sim_mode":             False,
        # ── Port scanning ────────────────────────────────────────────────
        "available_ports":      [],     # list of (device, description, hwid)
        "selected_port_device": None,   # currently selected port string
        "ports_scanned":        False,  # whether a scan has been run
        # ── Auto-training ─────────────────────────────────────────────────
        "auto_trained":         False,
        "training_results":     None,
        # ── Manual entry ───────────────────────────────────────────────
        "manual_adc":           512,
        "manual_result":        None,
        # ── Gemini clinical agent ─────────────────────────────────────────
        "gemini_agent":         GeminiAgent(),   # auto-reads .env on init
        "gemini_insight":       None,            # last GeminiInsight object
        "gemini_loading":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# =============================================================================
# Sidebar — Device Controls
# =============================================================================

def render_sidebar() -> int:
    with st.sidebar:
        # ── Logo / branding ────────────────────────────────────────────────
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width='stretch')
        else:
            st.markdown("""
            <div style='text-align:center; padding: 20px 0 10px;'>
              <div style='font-size:2.2rem;'>🔬</div>
              <div style='font-size:1.2rem; font-weight:700; color:#00D4FF; letter-spacing:0.1em;'>
                OptoScan
              </div>
              <div style='font-size:0.65rem; color:#475569; letter-spacing:0.15em; text-transform:uppercase;'>
                Optical Biopsy System
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── SECTION 1: Device Connection ─────────────────────────────────
        st.markdown("<div class='section-header'>Device Connection</div>",
                    unsafe_allow_html=True)

        if st.button("🔍  Scan Available Devices"):
            _scan_ports()
            st.rerun()

        if st.session_state.ports_scanned:
            if st.session_state.available_ports:
                port_labels = [
                    f"{desc}  ·  {dev}"
                    if (desc and desc.lower() not in ("n/a", dev.lower()))
                    else dev
                    for dev, desc, _hwid in st.session_state.available_ports
                ]
                sel_idx = st.selectbox(
                    f"Available Devices  ({len(port_labels)} found)",
                    range(len(port_labels)),
                    format_func=lambda i: port_labels[i],
                    key="port_selectbox",
                )
                chosen = st.session_state.available_ports[sel_idx][0]
                st.session_state.selected_port_device = chosen
                st.caption(f"Port: `{chosen}`")
            else:
                st.warning("⚠ No serial devices found. Check connections and retry.")
                st.session_state.selected_port_device = None
        else:
            st.info("Click **Scan Available Devices** to detect connected hardware.")

        baud = st.selectbox(
            "Baud Rate", [4800, 9600, 19200, 38400, 57600, 115200], index=1
        )

        no_port = st.session_state.selected_port_device is None
        if not st.session_state.connected:
            if st.button(
                "⚡  Connect",
                type="primary",
                disabled=no_port,
                help="Select a device above first" if no_port else None,
            ):
                _connect(st.session_state.selected_port_device, baud)
        else:
            if st.button("⛔  Disconnect", type="secondary"):
                _disconnect()

        _render_connection_badge()

        # ── SECTION 2: Inference Status ────────────────────────────────
        st.markdown("---")
        st.markdown("<div class='section-header'>Inference Engine</div>",
                    unsafe_allow_html=True)

        if st.session_state.engine.model_loaded:
            res = st.session_state.training_results
            st.markdown(
                "<span class='badge badge-ml'>🤖 ML Model Active</span>",
                unsafe_allow_html=True,
            )
            if res:
                st.caption(
                    f"✅ Accuracy: **{res['accuracy']*100:.1f}%** · "
                    f"{res['n_samples']} samples · "
                    f"{res['n_features']} spectral features"
                )
                st.caption(
                    f"Classes: {', '.join(str(c) for c in res['class_names'])}"
                )
        else:
            st.markdown(
                "<span class='badge badge-heuristic'>📐 Heuristic Mode</span>",
                unsafe_allow_html=True,
            )
            st.caption("Add CSV files to the project folder to enable ML.")

        # ── SECTION 3: Gemini Clinical Agent ─────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
          <div class='section-header' style='margin:0;'>Gemini AI Agent</div>
          <span style='font-size:1rem;'>✨</span>
        </div>
        """, unsafe_allow_html=True)

        agent: GeminiAgent = st.session_state.gemini_agent

        if agent.is_ready:
            st.markdown(
                "<span class='badge badge-ml' style='background:#1a0a2e;"
                "color:#C084FC;'>✨ Gemini Connected</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Model: `{agent.model_name}`")
            if st.button("Disconnect Gemini", key="gemini_disconnect_btn"):
                st.session_state.gemini_agent   = GeminiAgent()
                st.session_state.gemini_insight = None
                st.rerun()
        else:
            st.markdown(
                "<span class='badge badge-heuristic' style='color:#F59E0B;'>"
                "⚙ Not Connected</span>",
                unsafe_allow_html=True,
            )
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIza…",
                key="gemini_key_field",
                help="https://aistudio.google.com/app/apikey",
            )
            if st.button("✨ Connect Gemini", key="gemini_connect_btn", type="primary"):
                if api_key_input:
                    ok = st.session_state.gemini_agent.configure(api_key_input)
                    if ok:
                        st.session_state.gemini_insight = None   # clear any old error
                        st.toast("✨ Gemini AI connected!", icon="✅")
                        st.rerun()
                    else:
                        st.error("❌ Invalid API key or network error.")
                else:
                    st.warning("Please enter your Gemini API key.")
            st.caption("🔑 Or set `GEMINI_API_KEY` in a `.env` file to auto-connect.")

        # ── SECTION 4: Display Settings ────────────────────────────────
        st.markdown("---")
        st.markdown("<div class='section-header'>Display Settings</div>",
                    unsafe_allow_html=True)
        refresh_ms = st.slider(
            "Refresh Interval (ms)", 100, 2000, value=UI_REFRESH_MS, step=100
        )

        st.markdown("---")
        st.markdown(
            f"<div class='footer'>{APP_VERSION} · OptoScan"
            f"<br>Non-Invasive Optical Biopsy</div>",
            unsafe_allow_html=True,
        )

    return refresh_ms


def _scan_ports() -> None:
    """
    Enumerate all available serial / COM ports and store them in session state.
    Uses pyserial's list_ports utility (included with pyserial >= 3.0).
    """
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        st.session_state.available_ports = [
            (p.device, p.description, p.hwid)
            for p in sorted(ports, key=lambda x: x.device)
        ]
    except ImportError:
        # pyserial not installed — simulate with empty list
        st.session_state.available_ports = []
        logger.warning("pyserial not installed; cannot scan ports.")
    st.session_state.ports_scanned = True
    logger.info(
        "Port scan complete: %d devices found",
        len(st.session_state.available_ports),
    )


def _run_training(csv_path: str, model_path: str) -> None:
    """
    Train a RandomForestClassifier on *csv_path* and save to *model_path*.
    On success, auto-loads the model into the inference engine.
    """
    with st.spinner(f"Training on {os.path.basename(csv_path)} …"):
        try:
            import train_model as tm
            results = tm.train_and_save(csv_path, model_path)
            st.session_state.training_results = results

            # Auto-load the freshly trained model
            ok = st.session_state.engine.load_model(model_path)
            if ok:
                st.toast(
                    f"✅ Trained & loaded — Accuracy: {results['accuracy']*100:.1f}%",
                    icon="🎯",
                )
            else:
                st.warning("Model saved but could not be auto-loaded. Try uploading manually.")
        except Exception as exc:
            st.error(f"❌ Training failed: {exc}")
            logger.exception("Training error")


def _auto_train() -> None:
    """
    Runs once per browser session (guarded by session_state.auto_trained).

    Logic:
      1. If trained_model.pkl already exists → load it, done.
      2. Else if CSV files exist in the project folder → train on all of them
         combined (each file = one tissue class) and save as trained_model.pkl.
      3. Else → remain in heuristic mode silently.
    """
    if st.session_state.get("auto_trained"):
        return

    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(project_dir, "trained_model.pkl")

    # ── Step 1: existing model ────────────────────────────────────────────────
    if os.path.exists(model_path):
        ok = st.session_state.engine.load_model(model_path)
        if ok:
            logger.info("Auto-loaded existing model from %s", model_path)
            st.session_state.auto_trained = True
            return

    # ── Step 2: train on all CSVs ─────────────────────────────────────────────
    csv_files = sorted(
        os.path.join(project_dir, f)
        for f in os.listdir(project_dir)
        if f.lower().endswith(".csv")
    )
    if csv_files:
        try:
            import train_model as tm
            logger.info("Auto-training on %d CSV files …", len(csv_files))
            results = tm.train_and_save_merged(csv_files, model_path)
            st.session_state.training_results = results
            st.session_state.engine.load_model(model_path)
            logger.info(
                "Auto-training complete — accuracy=%.3f  classes=%s",
                results["accuracy"], results["class_names"],
            )
        except Exception as exc:
            logger.error("Auto-training failed: %s", exc)

    st.session_state.auto_trained = True


def _scan_ports() -> None:
    """Enumerate all available serial / COM ports and store in session state."""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        st.session_state.available_ports = [
            (p.device, p.description, p.hwid)
            for p in sorted(ports, key=lambda x: x.device)
        ]
    except ImportError:
        st.session_state.available_ports = []
        logger.warning("pyserial not installed; cannot scan ports.")
    st.session_state.ports_scanned = True
    logger.info("Port scan: %d devices found", len(st.session_state.available_ports))


def _render_connection_badge() -> None:

    if st.session_state.sim_mode:
        badge = "<span class='badge badge-simulation'>⚡ Simulation Mode</span>"
    elif st.session_state.connected:
        badge = "<span class='badge badge-connected'>● Connected</span>"
    else:
        badge = "<span class='badge badge-disconnected'>● Disconnected</span>"
    st.markdown(f"<div style='margin-top:12px;'>{badge}</div>", unsafe_allow_html=True)


def _connect(port: str, baud: int) -> None:
    if st.session_state.reader is not None:
        st.session_state.reader.stop()

    reader = BluetoothReader(port=port, baud=baud)
    reader.start()
    time.sleep(0.6)   # allow thread to attempt first connection

    st.session_state.reader    = reader
    st.session_state.baud_rate = baud
    st.session_state.connected = True

    # Detect simulation mode (pyserial absent)
    try:
        import serial  # noqa: F401
        st.session_state.sim_mode = False
    except ImportError:
        st.session_state.sim_mode = True

    st.rerun()


def _disconnect() -> None:
    if st.session_state.reader:
        st.session_state.reader.stop()
    st.session_state.reader    = None
    st.session_state.connected = False
    st.session_state.sim_mode  = False
    st.rerun()


def _load_uploaded_model(file) -> None:
    """Save uploaded model to a temp file and load it into the engine."""
    suffix = os.path.splitext(file.name)[1]
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.read())
    tmp.close()
    success = st.session_state.engine.load_model(tmp.name)
    if success:
        st.success(f"✅ Model loaded: {file.name}")
    else:
        st.error("❌ Failed to load model. Check format and try again.")


# =============================================================================
# Gauge Chart
# =============================================================================

def build_gauge(value: int, label: str) -> go.Figure:
    color_map = {
        LABEL_NORMAL:   COLOR_NORMAL,
        LABEL_HEMATOMA: COLOR_HEMATOMA,
        LABEL_FIBROUS:  COLOR_FIBROUS,
    }
    needle_color = color_map.get(label, COLOR_ACCENT)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "font": {"size": 52, "color": needle_color, "family": "Space Mono"},
            "suffix": "",
        },
        gauge={
            "axis": {
                "range": [ADC_MIN, ADC_MAX],
                "tickwidth": 1,
                "tickcolor": "#1E3A5F",
                "tickfont": {"size": 10, "color": "#475569"},
                "nticks": 11,
            },
            "bar": {"color": needle_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   340],  "color": "#1A0A0D"},   # Hematoma zone
                {"range": [340, 680],  "color": "#0A1A12"},   # Normal zone
                {"range": [680, 1023], "color": "#1A1205"},   # Fibrous zone
            ],
            "threshold": {
                "line": {"color": needle_color, "width": 4},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(t=20, b=0, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
    )
    return fig


# =============================================================================
# Confidence Bar
# =============================================================================

def build_confidence_bar(confidence: float, label: str) -> go.Figure:
    color_map = {
        LABEL_NORMAL:   COLOR_NORMAL,
        LABEL_HEMATOMA: COLOR_HEMATOMA,
        LABEL_FIBROUS:  COLOR_FIBROUS,
    }
    bar_color = color_map.get(label, COLOR_ACCENT)
    pct       = round(confidence * 100, 1)

    fig = go.Figure(go.Bar(
        x=[pct],
        y=["Confidence"],
        orientation="h",
        marker=dict(color=bar_color, line=dict(width=0)),
        text=[f"{pct}%"],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="white", size=13, family="Space Mono"),
    ))
    fig.add_trace(go.Bar(
        x=[100 - pct],
        y=["Confidence"],
        orientation="h",
        marker=dict(color="#0D1B2A"),
        showlegend=False,
    ))
    fig.update_layout(
        barmode="stack",
        height=70,
        margin=dict(t=5, b=5, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
    )
    return fig


# =============================================================================
# Rolling Time-Series
# =============================================================================

def build_timeseries(history: deque, labels: deque, timestamps: deque) -> go.Figure:
    if not history:
        return go.Figure()

    xs     = list(range(len(history)))
    values = list(history)
    lbls   = list(labels)

    color_map = {
        LABEL_NORMAL:   COLOR_NORMAL,
        LABEL_HEMATOMA: COLOR_HEMATOMA,
        LABEL_FIBROUS:  COLOR_FIBROUS,
    }

    # Build per-label scatter traces for coloured dots
    traces: list[go.BaseTraceType] = []

    # Main line (always on top)
    traces.append(go.Scatter(
        x=xs, y=values,
        mode="lines",
        line=dict(color=COLOR_ACCENT, width=1.5, dash="dot"),
        name="ADC Readings",
        hovertemplate="Sample %{x}<br>ADC: %{y}<extra></extra>",
    ))

    # Per-label scatter markers
    for lbl, col in color_map.items():
        idx = [i for i, l in enumerate(lbls) if l == lbl]
        if idx:
            traces.append(go.Scatter(
                x=idx,
                y=[values[i] for i in idx],
                mode="markers",
                marker=dict(color=col, size=6, symbol="circle"),
                name=lbl,
                hovertemplate=f"{lbl}<br>ADC: %{{y}}<extra></extra>",
            ))

    # Horizontal zone guide lines
    zone_shapes = [
        dict(type="line", x0=0, x1=len(history), y0=340, y1=340,
             line=dict(color=COLOR_HEMATOMA, width=1, dash="dash")),
        dict(type="line", x0=0, x1=len(history), y0=680, y1=680,
             line=dict(color=COLOR_FIBROUS, width=1, dash="dash")),
    ]

    fig = go.Figure(traces)
    fig.update_layout(
        height=220,
        margin=dict(t=10, b=30, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#060D18",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=dict(text="Samples (most recent →)", font=dict(size=10, color="#475569")),
            color="#475569", gridcolor="#0F1E35",
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="ADC (0–1023)", font=dict(size=10, color="#475569")),
            color="#475569",
            gridcolor="#0F1E35",
            range=[ADC_MIN - 30, ADC_MAX + 30],
        ),
        shapes=zone_shapes,
    )
    return fig


# =============================================================================
# Diagnosis Display Card
# =============================================================================

def _dx_card_html(label: str, confidence: float, mode: str) -> str:
    color_map = {
        LABEL_NORMAL:   (COLOR_NORMAL,   "#064E3B", "#D1FAE5"),
        LABEL_HEMATOMA: (COLOR_HEMATOMA, "#450A0A", "#FECACA"),
        LABEL_FIBROUS:  (COLOR_FIBROUS,  "#451A03", "#FDE68A"),
    }
    border, bg, text = color_map.get(label, (COLOR_ACCENT, "#0C4A6E", "#BAE6FD"))

    icons = {
        LABEL_NORMAL:   "✅",
        LABEL_HEMATOMA: "🩸",
        LABEL_FIBROUS:  "🔶",
    }
    icon = icons.get(label, "🔬")

    mode_badge = (
        f"<span style='background:#0C4A6E;color:#38BDF8;border:1px solid #0369A1;"
        f"border-radius:999px;padding:2px 10px;font-size:0.65rem;letter-spacing:0.1em;"
        f"text-transform:uppercase;'>🤖 {mode}</span>"
    )

    return f"""
    <div class='dx-card' style='background:{bg}20;border-color:{border}60;'>
      <div style='font-size:2.5rem;margin-bottom:6px;'>{icon}</div>
      <div class='dx-label' style='color:{text};'>{label}</div>
      <div class='dx-sub' style='margin-bottom:12px;'>Tissue Classification</div>
      {mode_badge}
    </div>
    """


# =============================================================================
# Session Statistics
# =============================================================================

def _session_stats() -> dict:
    hist = list(st.session_state.history)
    if not hist:
        return {"min": 0, "max": 0, "avg": 0, "count": st.session_state.total_samples}
    return {
        "min":   min(hist),
        "max":   max(hist),
        "avg":   round(sum(hist) / len(hist), 1),
        "count": st.session_state.total_samples,
    }


# =============================================================================
# Header
# =============================================================================

def render_header() -> None:
    conn_html = (
        "<span class='badge badge-connected'>● LIVE</span>"
        if st.session_state.connected
        else "<span class='badge badge-disconnected'>● OFFLINE</span>"
    )
    sim_html = (
        " &nbsp;<span class='badge badge-simulation'>⚡ SIM</span>"
        if st.session_state.sim_mode else ""
    )
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;
                padding:12px 4px 20px;border-bottom:1px solid #1E293B;margin-bottom:24px;'>
      <div>
        <div style='font-size:1.4rem;font-weight:700;letter-spacing:0.06em;color:#E2E8F0;'>
          🔬 OptoScan
          <span style='font-size:0.75rem;color:#475569;margin-left:8px;font-weight:400;'>
            Optical Biopsy Diagnostic System
          </span>
        </div>
      </div>
      <div style='text-align:right;'>
        {conn_html}{sim_html}
        <div style='font-size:0.68rem;color:#475569;margin-top:4px;
                    font-family:"Space Mono",monospace;'>
          {ts}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)



# =============================================================================
# Manual Probe Entry Panel
# =============================================================================

# ── on_change callbacks (must be module-level for Streamlit) ─────────────────
def _on_manual_slider() -> None:
    """Slider moved → push value to manual_adc AND sync the number input key."""
    val = st.session_state["_m_slider"]
    st.session_state.manual_adc        = val
    st.session_state["_m_number"]      = val   # keeps number field in sync


def _on_manual_number() -> None:
    """Number typed → push value to manual_adc AND sync the slider key."""
    val = int(st.session_state["_m_number"])
    st.session_state.manual_adc        = val
    st.session_state["_m_slider"]      = val   # keeps slider in sync


def render_manual_entry(placeholder) -> None:
    """Render the offline manual ADC entry + classification panel."""
    with placeholder.container():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>
          <div class='section-header' style='margin:0;'>Manual Probe Entry</div>
          <span style='font-size:0.68rem; color:#475569; letter-spacing:0.08em;
                       background:#1E293B; border-radius:20px; padding:3px 10px;'>
            ⌨ Offline Mode — No Device Required
          </span>
        </div>
        <div style='height:1px; background:linear-gradient(90deg,#00D4FF33,transparent);
                    margin-bottom:20px;'></div>
        """, unsafe_allow_html=True)

        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("""
            <div style='font-size:0.68rem; color:#475569; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:8px;'>
              ADC Reading Input (0 – 1023)
            </div>""", unsafe_allow_html=True)

            # ── Initialise widget session keys on first render ───────────────
            if "_m_slider" not in st.session_state:
                st.session_state["_m_slider"] = st.session_state.manual_adc
            if "_m_number" not in st.session_state:
                st.session_state["_m_number"] = st.session_state.manual_adc

            # Slider — on_change keeps number field in sync
            st.slider(
                "ADC Slider",
                min_value=0, max_value=1023,
                step=1,
                label_visibility="collapsed",
                key="_m_slider",
                on_change=_on_manual_slider,
            )

            # Number input — on_change keeps slider in sync
            st.number_input(
                "Or type exact value:",
                min_value=0, max_value=1023,
                step=1,
                key="_m_number",
                on_change=_on_manual_number,
            )

            # Single source of truth for everything downstream
            final_adc = int(st.session_state.manual_adc)

            # Visual zone indicator
            if final_adc <= 340:
                zone_col, zone_name = "#FF4C6A", "🩸 Hematoma Zone"
            elif final_adc <= 680:
                zone_col, zone_name = "#00C896", "✅ Normal Zone"
            else:
                zone_col, zone_name = "#F5A623", "🔶 Fibrous / Scar Zone"

            st.markdown(f"""
            <div style='margin-top:12px; padding:12px 16px; border-radius:10px;
                        background:#0F172A; border-left:3px solid {zone_col};
                        display:flex; justify-content:space-between; align-items:center;'>
              <div>
                <div style='font-size:0.65rem; color:#475569; letter-spacing:0.1em;
                            text-transform:uppercase;'>Heuristic Zone</div>
                <div style='font-size:0.9rem; color:{zone_col}; font-weight:600;
                            margin-top:2px;'>{zone_name}</div>
              </div>
              <div style='font-size:2rem; font-weight:700; color:{zone_col};
                          font-family:"Space Mono",monospace;'>{final_adc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Reflectance display
            reflectance = final_adc / 1023.0
            st.markdown(f"""
            <div style='margin-top:10px; padding:10px 14px; border-radius:8px;
                        background:#0F172A; border:1px solid #1E3A5F;'>
              <div style='font-size:0.62rem; color:#475569; letter-spacing:0.1em;
                          text-transform:uppercase;'>Normalised Reflectance</div>
              <div style='font-size:1.1rem; color:#00D4FF; font-weight:600;
                          font-family:"Space Mono",monospace;'>{reflectance:.4f}</div>
              <div style='font-size:0.62rem; color:#334155; margin-top:2px;'>
                Broadcast across all spectral bands for ML inference
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            run_clicked = st.button(
                "🔬  Run Diagnosis",
                type="primary",
                key="manual_run_btn",
                help="Classify the entered ADC value using the loaded ML model",
            )

            if run_clicked:
                result = st.session_state.engine.run(final_adc)
                st.session_state.manual_result = result
                # Also fire Gemini insight for the manual value
                agent: GeminiAgent = st.session_state.gemini_agent
                if agent.is_ready:
                    insight = agent.get_insight(final_adc, force=True)
                    st.session_state.gemini_insight = insight

        with col_result:
            st.markdown("""
            <div style='font-size:0.68rem; color:#475569; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:8px;'>
              Classification Result
            </div>""", unsafe_allow_html=True)

            res = st.session_state.manual_result

            if res is None:
                st.markdown("""
                <div style='height:260px; border-radius:16px; border:1px dashed #1E3A5F;
                            display:flex; flex-direction:column; align-items:center;
                            justify-content:center; color:#334155;'>
                  <div style='font-size:2rem;'>🔬</div>
                  <div style='font-size:0.8rem; margin-top:8px;'>
                    Enter a value and click <strong style="color:#00D4FF">Run Diagnosis</strong>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                COLOR_MAP = {
                    LABEL_NORMAL:   COLOR_NORMAL,
                    LABEL_HEMATOMA: COLOR_HEMATOMA,
                    LABEL_FIBROUS:  COLOR_FIBROUS,
                }
                label_color = COLOR_MAP.get(res.label, COLOR_ACCENT)
                ICON_MAP = {
                    LABEL_NORMAL:   "✅",
                    LABEL_HEMATOMA: "🩸",
                    LABEL_FIBROUS:  "🔶",
                }
                icon = ICON_MAP.get(res.label, "🔬")
                mode_badge = (
                    "<span style='background:#1E3A5F;color:#00D4FF;font-size:0.6rem;"
                    "padding:2px 8px;border-radius:10px;letter-spacing:0.08em;'>ML MODEL</span>"
                    if res.mode == "ML Model"
                    else
                    "<span style='background:#1A2810;color:#84CC16;font-size:0.6rem;"
                    "padding:2px 8px;border-radius:10px;letter-spacing:0.08em;'>HEURISTIC</span>"
                )

                st.markdown(f"""
                <div style='border-radius:16px; border:1px solid {label_color}44;
                            background:linear-gradient(135deg,#0F172A 0%,#1E293B 100%);
                            padding:24px; text-align:center;'>
                  <div style='font-size:2.8rem; margin-bottom:8px;'>{icon}</div>
                  <div style='font-size:1.6rem; font-weight:700; color:{label_color};
                              letter-spacing:0.05em; font-family:"Space Mono",monospace;'>
                    {res.label}
                  </div>
                  <div style='font-size:0.65rem; color:#475569; letter-spacing:0.15em;
                              text-transform:uppercase; margin:6px 0 12px;'>
                    Tissue Classification
                  </div>
                  {mode_badge}
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div style='font-size:0.65rem;color:#475569;letter-spacing:0.1em;
                            text-transform:uppercase;margin-bottom:6px;'>
                  Confidence Score
                </div>""", unsafe_allow_html=True)
                fig_manual_conf = build_confidence_bar(res.confidence, res.label)
                st.plotly_chart(
                    fig_manual_conf,
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key="manual_conf_chart",
                )

                # Summary row
                c1, c2, c3 = st.columns(3)
                for col, lbl, val in [
                    (c1, "ADC Value",    str(st.session_state.manual_adc)),
                    (c2, "Reflectance",  f"{st.session_state.manual_adc/1023:.3f}"),
                    (c3, "Confidence",   f"{res.confidence*100:.1f}%"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class='metric-card' style='padding:10px 8px;'>
                          <div class='label'>{lbl}</div>
                          <div class='value' style='font-size:1.2rem;'>{val}</div>
                        </div>""", unsafe_allow_html=True)

        # ── Gemini Insight below manual result ─────────────────────────────
            if st.session_state.gemini_insight is not None:
                render_gemini_insight(st.session_state.gemini_insight)


# =============================================================================
# Gemini Clinical Insight Panel
# =============================================================================

def render_gemini_insight(insight) -> None:
    """Renders the Gemini AI clinical interpretation card."""
    if insight is None:
        return

    if not insight.ok:
        if insight.is_key_invalid:
            # Agent has already reset itself — sidebar will return to key-entry
            st.session_state.gemini_insight = None
            st.markdown("""
            <div style='margin-top:16px; padding:20px 24px; border-radius:12px;
                        background:#1A0010; border:1px solid #EC489955;'>
              <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>
                <span style='font-size:1.2rem;'>🔑</span>
                <span style='font-size:0.76rem;color:#EC4899;font-weight:700;
                             letter-spacing:0.12em;text-transform:uppercase;'>
                  Invalid API Key
                </span>
              </div>
              <div style='font-size:0.8rem;color:#CBD5E1;line-height:1.7;'>
                The key was rejected by Google (<strong style='color:#F472B6;'>400 API_KEY_INVALID</strong>).
                This usually means the key was copied incorrectly.<br><br>
                <strong style='color:#E2E8F0;'>To fix:</strong><br>
              </div>
              <ol style='font-size:0.78rem;color:#94A3B8;margin:8px 0 0 16px;line-height:2.2;'>
                <li>Go to <strong style='color:#60A5FA;'>aistudio.google.com/app/apikey</strong></li>
                <li>Click the <strong>copy icon</strong> next to your key (don\'t retype it)</li>
                <li>Paste it into the <strong>Gemini API Key</strong> field in the sidebar</li>
              </ol>
            </div>
            """, unsafe_allow_html=True)
            return
        elif insight.is_quota_zero:
            # ── Project quota set to 0 — configuration issue, not temp limit ──
            st.markdown("""
            <div style='margin-top:16px; padding:20px 24px; border-radius:12px;
                        background:#1A0010; border:1px solid #EC489955;'>
              <div style='display:flex;align-items:center;gap:8px;margin-bottom:12px;'>
                <span style='font-size:1.2rem;'>🔑</span>
                <span style='font-size:0.76rem;color:#EC4899;font-weight:700;
                             letter-spacing:0.12em;text-transform:uppercase;'>
                  API Key Needs to Be Replaced
                </span>
              </div>
              <div style='font-size:0.8rem;color:#CBD5E1;line-height:1.7;'>
                Your current key's Google Cloud project has the
                <strong style='color:#F472B6;'>free-tier quota set to 0</strong>.
                This is a project configuration issue — waiting will not fix it.<br><br>
                <strong style='color:#E2E8F0;'>✅ Fix (2 minutes):</strong>
              </div>
              <ol style='font-size:0.78rem;color:#94A3B8;margin:10px 0 0 16px;line-height:2;'>
                <li>Open
                  <strong style='color:#60A5FA;'>
                    aistudio.google.com/app/apikey
                  </strong>
                  (NOT Google Cloud Console)
                </li>
                <li>Click <strong>Create API Key → Create API key in new project</strong></li>
                <li>Copy the new key</li>
                <li>In the OptoScan sidebar:
                  click <strong>Disconnect Gemini</strong>,
                  paste the new key, click <strong>✨ Connect Gemini</strong>
                </li>
              </ol>
              <div style='margin-top:12px;font-size:0.65rem;color:#475569;'>
                💡 AI Studio keys have 15 RPM / 1,500 RPD free tier built-in.
              </div>
            </div>
            """, unsafe_allow_html=True)
        elif insight.is_rate_limited:
            elapsed   = int(time.time() - insight.timestamp)
            remaining = max(0, (insight.retry_after or 60) - elapsed)
            if remaining > 0:
                status_txt   = f"⏳ Retry available in ~{remaining}s"
                status_color = "#F59E0B"
            else:
                status_txt   = "✅ Ready — click \u2728 Get AI Insight"
                status_color = "#10B981"
            st.markdown(f"""
            <div style='margin-top:16px; padding:16px 20px; border-radius:12px;
                        background:#1A1000; border:1px solid #F59E0B55;'>
              <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
                <span style='font-size:1.1rem;'>⚡</span>
                <span style='font-size:0.72rem;color:#F59E0B;font-weight:600;
                             letter-spacing:0.1em;text-transform:uppercase;'>
                  Free Tier Rate Limit
                </span>
              </div>
              <div style='font-size:0.78rem;color:#94A3B8;line-height:1.6;'>
                All Gemini models exhausted their per-minute quota
                (<strong style='color:#FCD34D;'>15 RPM free tier</strong>).<br>
                <strong style='color:{status_color};margin-top:6px;display:block;'>{status_txt}</strong>
              </div>
              <div style='margin-top:8px;font-size:0.65rem;color:#475569;'>
                💡 Wait 60 seconds, then click <strong>✨ Get AI Insight</strong> again.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='margin-top:16px;padding:14px 18px;border-radius:12px;
                        background:#1A0A0D;border:1px solid #EF444455;
                        font-size:0.78rem;color:#F87171;'>
              ⚠ Gemini error: {insight.error}
            </div>
            """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div style='
        margin-top:20px;
        padding:20px 24px;
        border-radius:16px;
        background: linear-gradient(135deg, #0D0620 0%, #1A0A3A 60%, #0F172A 100%);
        border: 1px solid #9333EA44;
        box-shadow: 0 0 24px #9333EA22;
    '>
      <div style='display:flex; align-items:center; gap:10px; margin-bottom:14px;'>
        <span style='font-size:1.3rem;'>✨</span>
        <div>
          <div style='font-size:0.62rem; letter-spacing:0.15em; text-transform:uppercase;
                      color:#9333EA; font-weight:600;'>Gemini Clinical Insight</div>
          <div style='font-size:0.58rem; color:#475569; margin-top:1px;'>
            ADC {insight.adc_value} · Reflectance {insight.adc_value/1023:.4f}
          </div>
        </div>
      </div>
      <div style='font-size:0.82rem; color:#E2E8F0; line-height:1.7;
                  font-family:"Inter",sans-serif; font-style:italic;
                  border-left:3px solid #9333EA; padding-left:14px;'>
        {insight.text}
      </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main App Loop
# =============================================================================

def main() -> None:

    _auto_train()
    refresh_ms = render_sidebar()
    render_header()

    # ── Placeholder containers ─────────────────────────────────────────────
    ph_gauge  = st.empty()
    ph_dx     = st.empty()
    ph_chart  = st.empty()
    ph_stats  = st.empty()
    ph_gemini = st.empty()
    ph_manual = st.empty()
    ph_footer = st.empty()

    # ── Main render ────────────────────────────────────────────────────────
    _render_frame(ph_gauge, ph_dx, ph_chart, ph_stats, ph_gemini, ph_footer)
    render_manual_entry(ph_manual)

    # ── Auto-refresh ───────────────────────────────────────────────────────
    if st.session_state.connected:
        time.sleep(refresh_ms / 1000.0)
        st.rerun()


def _render_frame(ph_gauge, ph_dx, ph_chart, ph_stats, ph_gemini, ph_footer) -> None:
    # Pull latest value from reader thread
    raw_value = st.session_state.last_value

    if st.session_state.reader is not None:
        new_val = st.session_state.reader.latest_value()
        if new_val is not None:
            raw_value                       = new_val
            st.session_state.last_value     = new_val
            st.session_state.total_samples  = st.session_state.reader.total_samples
            st.session_state.drop_count     = st.session_state.reader.drop_count

    # Run ML inference
    result = st.session_state.engine.run(raw_value)
    st.session_state.last_result = result
    # NOTE: Gemini is called only via the explicit button below (not auto)
    # to respect free-tier rate limits (15 RPM / 1500 RPD).

    # Update rolling history
    st.session_state.history.append(raw_value)
    st.session_state.labels.append(result.label)
    st.session_state.timestamps.append(time.time())

    # ── Row 1: Gauge + Diagnosis ──────────────────────────────────────────
    with ph_gauge.container():
        col_g, col_dx = st.columns([1, 1], gap="large")

        with col_g:
            st.markdown("<div class='section-header'>Live Sensor Reading</div>",
                        unsafe_allow_html=True)
            fig_gauge = build_gauge(raw_value, result.label)
            st.plotly_chart(fig_gauge, use_container_width=True,
                            config={"displayModeBar": False})

            # Numeric readout card
            st.markdown(f"""
            <div class='metric-card'>
              <div class='label'>ADC Raw Value</div>
              <div class='value'>{raw_value:4d}</div>
              <div class='unit'>counts / 1023 · 850nm IR</div>
            </div>
            """, unsafe_allow_html=True)

        with col_dx:
            st.markdown("<div class='section-header'>Tissue Classification</div>",
                        unsafe_allow_html=True)

            # Diagnosis card
            st.markdown(
                _dx_card_html(result.label, result.confidence, result.mode),
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.markdown("<div style='font-size:0.68rem;color:#475569;letter-spacing:0.1em;"
                        "text-transform:uppercase;margin:12px 0 6px;'>Confidence Score</div>",
                        unsafe_allow_html=True)
            fig_conf = build_confidence_bar(result.confidence, result.label)
            st.plotly_chart(fig_conf, use_container_width=True,
                            config={"displayModeBar": False})

            # Zone reference legend
            st.markdown("""
            <div style='margin-top:16px;'>
              <div style='font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                          color:#475569;margin-bottom:10px;'>Tissue Zone Reference</div>
              <div style='display:flex;gap:8px;flex-wrap:wrap;'>
                <div style='flex:1;background:#1A0A0D;border:1px solid #FF4C6A55;
                            border-radius:8px;padding:8px 10px;text-align:center;'>
                  <div style='color:#FF4C6A;font-size:0.75rem;font-weight:600;'>🩸 Hematoma</div>
                  <div style='color:#64748B;font-size:0.65rem;'>ADC 0–340</div>
                </div>
                <div style='flex:1;background:#0A1A12;border:1px solid #00C89655;
                            border-radius:8px;padding:8px 10px;text-align:center;'>
                  <div style='color:#00C896;font-size:0.75rem;font-weight:600;'>✅ Normal</div>
                  <div style='color:#64748B;font-size:0.65rem;'>ADC 341–680</div>
                </div>
                <div style='flex:1;background:#1A1205;border:1px solid #F5A62355;
                            border-radius:8px;padding:8px 10px;text-align:center;'>
                  <div style='color:#F5A623;font-size:0.75rem;font-weight:600;'>🔶 Fibrous</div>
                  <div style='color:#64748B;font-size:0.65rem;'>ADC 681–1023</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Row 2: Time-Series ────────────────────────────────────────────────
    with ph_chart.container():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Real-Time Reflectance Trend</div>",
                    unsafe_allow_html=True)
        fig_ts = build_timeseries(
            st.session_state.history,
            st.session_state.labels,
            st.session_state.timestamps,
        )
        st.plotly_chart(fig_ts, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Row 3: Session Statistics ─────────────────────────────────────────
    with ph_stats.container():
        stats = _session_stats()
        elapsed = int(time.time() - st.session_state.session_start)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)

        st.markdown("<div class='section-header'>Session Statistics</div>",
                    unsafe_allow_html=True)

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        for col, label, val in [
            (sc1, "Min ADC",    f"{stats['min']}"),
            (sc2, "Max ADC",    f"{stats['max']}"),
            (sc3, "Avg ADC",    f"{stats['avg']}"),
            (sc4, "Samples",    f"{stats['count']:,}"),
            (sc5, "Data Drops", f"{st.session_state.drop_count}"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card' style='padding:14px 10px;'>
                  <div class='label'>{label}</div>
                  <div class='value' style='font-size:1.6rem;'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Row 4: Gemini Clinical Insight ────────────────────────────────────
    with ph_gemini.container():
        agent_live: GeminiAgent = st.session_state.gemini_agent
        if agent_live.is_ready:
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            col_g1, col_g2 = st.columns([3, 1], gap="small")
            with col_g1:
                st.markdown("""
                <div style='font-size:0.68rem;color:#9333EA;letter-spacing:0.12em;
                            text-transform:uppercase;font-weight:600;'>
                  ✨ Gemini Clinical Insight
                </div>
                <div style='font-size:0.62rem;color:#475569;margin-top:2px;'>
                  AI-generated interpretation of the current sensor reading
                </div>
                """, unsafe_allow_html=True)
            with col_g2:
                if st.button(
                    "✨ Get AI Insight",
                    key="live_gemini_btn",
                    help="Call Gemini to analyse the current ADC reading",
                ):
                    with st.spinner("Asking Gemini..."):
                        ins = agent_live.get_insight(raw_value, force=True)
                        if ins is not None:
                            st.session_state.gemini_insight = ins
                        st.rerun()

            insight = st.session_state.gemini_insight
            if insight is not None:
                render_gemini_insight(insight)
            else:
                st.markdown("""
                <div style='padding:14px 18px; border-radius:12px;
                            background:#0D0620; border:1px dashed #9333EA44;
                            color:#475569; font-size:0.75rem; text-align:center;'>
                  Press <strong style='color:#C084FC'>✨ Get AI Insight</strong>
                  to generate a clinical interpretation
                </div>
                """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────
    with ph_footer.container():
        uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
        infer_mode = result.mode if result else "—"
        st.markdown(f"""
        <div class='footer' style='margin-top:32px;'>
          <span style='color:#1E3A5F;'>──────────────────────────────────────────</span><br>
          OptoScan {APP_VERSION} &nbsp;·&nbsp; Session Uptime: {uptime_str}
          &nbsp;·&nbsp; Inference: {infer_mode}
          &nbsp;·&nbsp; Non-Invasive Optical Biopsy Prototype<br>
          <span style='color:#1E293B;font-size:0.6rem;'>
            ⚠ For research and prototyping only. Not approved for clinical diagnosis.
          </span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
