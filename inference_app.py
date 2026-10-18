# =========================================================
# AEGIS // TACTICAL DRONE DETECTION CONSOLE
# =========================================================
# A glass/blueprint split-screen ops console: video viewport
# on the left, scrolling telemetry feed on the right. Runs a
# custom-trained YOLO model over uploaded video to detect
# drones. Built with Streamlit + OpenCV + Ultralytics.
# =========================================================

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import shutil
from pathlib import Path
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AEGIS · Tactical Detection Console",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONFIG — point this at your trained weights
# =========================================================

MODEL_PATH = r"C:\ComputerVisionProjects\Drone_Detection\best_model.pt"
MODEL_CONF_THRESHOLD_DEFAULT = 0.45
MODEL_ACCURACY_LABEL = "97.1"              # reported validation accuracy (%)

# =========================================================
# GLASS / BLUEPRINT CSS
# =========================================================
# Palette
#   bg          #070B14   blueprint navy
#   grid        rgba(80,180,255,0.04)   etched schematic grid
#   glass       rgba(18,28,46,0.55)     blurred panel surface
#   border      rgba(110,190,255,0.16)
#   text        #E8EEF5
#   text-dim    #7C8BA0
#   accent      #5BC8FF   cool cyan-blue
#   contact     #FF9D3D   amber-orange, reserved for live detections
#   error       #FF5C5C
# Type
#   Inter — headings/labels/body
#   JetBrains Mono — coordinates, timestamps, telemetry numbers
# =========================================================

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>

:root {
    --bg: #070B14;
    --glass: rgba(18,28,46,0.55);
    --glass-raised: rgba(24,36,58,0.6);
    --border: rgba(110,190,255,0.16);
    --border-strong: rgba(110,190,255,0.32);
    --text: #E8EEF5;
    --text-dim: #7C8BA0;
    --text-faint: #4C5A72;
    --accent: #5BC8FF;
    --accent-soft: rgba(91,200,255,0.12);
    --contact: #FF9D3D;
    --error: #FF5C5C;
    --sans: 'Inter', sans-serif;
    --mono: 'JetBrains Mono', monospace;
}

html, body, [class*="css"] { font-family: var(--sans); }

.stApp {
    background-color: var(--bg);
    background-image:
        linear-gradient(rgba(91,200,255,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(91,200,255,0.045) 1px, transparent 1px),
        radial-gradient(ellipse 1300px 700px at 50% -5%, rgba(91,200,255,0.07), transparent);
    background-size: 38px 38px, 38px 38px, 100% 100%;
}

#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1240px;
}

/* ---------------------------------------------------- */
/* GLASS PANEL PRIMITIVE                                  */
/* ---------------------------------------------------- */

.glass-panel {
    background: var(--glass);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* ---------------------------------------------------- */
/* STATUS BAR                                             */
/* ---------------------------------------------------- */

.statusbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 13px 20px;
    margin-bottom: 20px;
}

.statusbar-left { display: flex; align-items: center; gap: 12px; }

.statusbar-mark {
    width: 32px; height: 32px;
    border-radius: 7px;
    background: linear-gradient(135deg, var(--accent), #1E4A66);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; color: #05131D; font-size: 15px;
}

.statusbar-name { font-weight: 700; font-size: 14.5px; color: var(--text); letter-spacing: 0.02em; }
.statusbar-sub { font-family: var(--mono); font-size: 10.5px; color: var(--text-faint); letter-spacing: 0.05em; margin-top: 1px; }

.statusbar-right { display: flex; align-items: center; gap: 18px; font-family: var(--mono); font-size: 11.5px; color: var(--text-dim); }

.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 7px; }
.dot.armed { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 2.2s ease-in-out infinite; }
.dot.standby { background: var(--text-faint); }
.dot.contact { background: var(--contact); box-shadow: 0 0 8px var(--contact); animation: pulse-fast 0.8s ease-in-out infinite; }

@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }
@keyframes pulse-fast { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

/* ---------------------------------------------------- */
/* HERO                                                   */
/* ---------------------------------------------------- */

.hero { margin-bottom: 26px; padding: 0 4px; }

.hero-eyebrow {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 9px;
}

.hero-title {
    font-weight: 600;
    font-size: 28px;
    color: var(--text);
    letter-spacing: -0.01em;
    line-height: 1.3;
    margin-bottom: 8px;
}

.hero-desc { font-size: 13.5px; color: var(--text-dim); max-width: 580px; line-height: 1.6; }

/* ---------------------------------------------------- */
/* SECTION LABEL                                          */
/* ---------------------------------------------------- */

.sec-label {
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--text-faint);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 9px 2px;
    display: flex; align-items: center; gap: 8px;
}
.sec-label::after { content: ""; flex: 1; height: 1px; background: var(--border); }

/* ---------------------------------------------------- */
/* MODEL PANEL                                            */
/* ---------------------------------------------------- */

.model-panel {
    padding: 16px 20px;
    margin-bottom: 22px;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 12px;
}

.model-left { display: flex; align-items: center; gap: 13px; }

.model-icon {
    width: 36px; height: 36px; border-radius: 8px;
    border: 1px solid var(--border-strong);
    background: var(--accent-soft);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: var(--accent);
}

.model-name { font-size: 13px; color: var(--text); font-weight: 600; }
.model-path { font-family: var(--mono); font-size: 10.5px; color: var(--text-faint); margin-top: 2px; }

.spec-pill {
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--text-dim);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 5px;
    margin-left: 8px;
}

.badge { font-family: var(--mono); font-size: 10.5px; padding: 6px 12px; border-radius: 6px; letter-spacing: 0.04em; }
.badge.ready { color: var(--accent); border: 1px solid var(--border-strong); background: var(--accent-soft); }
.badge.idle { color: var(--text-dim); border: 1px solid var(--border); background: transparent; }
.badge.err { color: var(--error); border: 1px solid rgba(255,92,92,0.3); background: rgba(255,92,92,0.08); }

/* ---------------------------------------------------- */
/* SPLIT SCREEN                                           */
/* ---------------------------------------------------- */

.split-wrap { display: flex; gap: 16px; align-items: stretch; margin-bottom: 18px; }

.viewport-col { flex: 1.55; min-width: 0; }
.telemetry-col { flex: 1; min-width: 0; }

.viewport-frame {
    position: relative;
    padding: 0;
    overflow: hidden;
}

.viewport-titlebar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 10.5px;
    color: var(--text-dim); letter-spacing: 0.06em;
}

.viewport-body { position: relative; padding: 10px; min-height: 180px; }

.viewport-placeholder {
    display: flex; align-items: center; justify-content: center;
    height: 320px;
    color: var(--text-faint);
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: 0.05em;
    border: 1px dashed var(--border);
    border-radius: 6px;
}

/* corner brackets over the viewport */
.corner-bracket {
    position: absolute;
    width: 18px; height: 18px;
    border-color: var(--accent);
    opacity: 0.7;
    pointer-events: none;
}
.cb-tl { top: 10px; left: 10px; border-top: 2px solid; border-left: 2px solid; }
.cb-tr { top: 10px; right: 10px; border-top: 2px solid; border-right: 2px solid; }
.cb-bl { bottom: 10px; left: 10px; border-bottom: 2px solid; border-left: 2px solid; }
.cb-br { bottom: 10px; right: 10px; border-bottom: 2px solid; border-right: 2px solid; }

/* scanning sweep line */
.scan-sweep {
    position: absolute;
    left: 10px; right: 10px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    box-shadow: 0 0 10px var(--accent);
    animation: sweep 3.2s linear infinite;
    opacity: 0.65;
    pointer-events: none;
}
@keyframes sweep {
    0% { top: 10px; }
    100% { top: calc(100% - 10px); }
}

/* ---------------------------------------------------- */
/* TELEMETRY FEED — flight-recorder strip                 */
/* ---------------------------------------------------- */

.telemetry-frame { height: 100%; display: flex; flex-direction: column; }

.telemetry-body {
    flex: 1;
    overflow-y: auto;
    padding: 6px 4px;
    max-height: 380px;
}

.telemetry-row {
    display: flex;
    gap: 10px;
    padding: 7px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
}

.telemetry-time {
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--text-faint);
    min-width: 64px;
    padding-top: 1px;
}

.telemetry-msg { color: var(--text-dim); line-height: 1.5; }
.telemetry-msg .hit { color: var(--contact); font-weight: 600; }
.telemetry-msg .conf { font-family: var(--mono); color: var(--accent); }
.telemetry-msg .sys { color: var(--text-faint); font-style: italic; }

.telemetry-body::-webkit-scrollbar { width: 5px; }
.telemetry-body::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
.telemetry-body::-webkit-scrollbar-track { background: transparent; }

/* ---------------------------------------------------- */
/* TELEMETRY TILES (results strip)                        */
/* ---------------------------------------------------- */

.tile-row { display: flex; gap: 12px; margin: 18px 0 6px 0; flex-wrap: wrap; }

.tile {
    flex: 1; min-width: 130px;
    padding: 13px 16px;
}

.tile-val { font-family: var(--mono); font-size: 21px; font-weight: 700; color: var(--text); line-height: 1; }
.tile-val.accent { color: var(--accent); }
.tile-val.contact { color: var(--contact); }

.tile-label {
    font-family: var(--mono); font-size: 9.5px; color: var(--text-faint);
    letter-spacing: 0.07em; text-transform: uppercase; margin-top: 6px;
}

/* ---------------------------------------------------- */
/* FILE UPLOADER                                          */
/* ---------------------------------------------------- */

[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border-strong);
    border-radius: 10px;
    background: var(--glass);
    padding: 6px;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { fill: var(--accent); }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--text) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--text-faint) !important; }

/* ---------------------------------------------------- */
/* BUTTONS                                                */
/* ---------------------------------------------------- */

.stButton>button {
    width: 100%; border-radius: 8px; height: 44px;
    font-size: 13px; font-weight: 600; letter-spacing: 0.02em;
    background: var(--accent); color: #04131D; border: none;
    transition: all 0.15s ease;
}
.stButton>button:hover { background: #82D6FF; box-shadow: 0 0 0 3px rgba(91,200,255,0.16); }

.stDownloadButton>button {
    width: 100%; border-radius: 8px; height: 44px;
    font-size: 13px; font-weight: 600; letter-spacing: 0.02em;
    background: transparent; color: var(--accent); border: 1.5px solid var(--border-strong);
}
.stDownloadButton>button:hover { background: var(--accent-soft); border-color: var(--accent); }

.stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }

/* ---------------------------------------------------- */
/* PROGRESS                                               */
/* ---------------------------------------------------- */

.stProgress > div > div > div > div { background: linear-gradient(90deg, #1E4A66, var(--accent)); }
.stProgress > div > div > div { background: rgba(110,190,255,0.08); }

/* ---------------------------------------------------- */
/* SIDEBAR                                                */
/* ---------------------------------------------------- */

[data-testid="stSidebar"] {
    background: rgba(7,11,20,0.96);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.8rem; }

.side-title { font-family: var(--mono); font-size: 11px; color: var(--accent); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 16px; }
.side-heading { font-weight: 600; font-size: 12.5px; color: var(--text); margin-bottom: 8px; }

.spec-row { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 11px; color: var(--text-dim); padding: 7px 0; border-bottom: 1px solid var(--border); }
.spec-row span:last-child { color: var(--text); font-weight: 600; }

.format-chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
.format-chip { font-family: var(--mono); font-size: 10.5px; color: var(--text-dim); border: 1px solid var(--border); padding: 4px 9px; border-radius: 5px; }

.side-divider { height: 1px; background: var(--border); margin: 22px 0; border: none; }

/* ---------------------------------------------------- */
/* FOOTER                                                 */
/* ---------------------------------------------------- */

.footer-bar {
    margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    font-family: var(--mono); font-size: 10px; color: var(--text-faint); letter-spacing: 0.04em;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "model" not in st.session_state:
    st.session_state.model = None
if "model_status" not in st.session_state:
    st.session_state.model_status = "not_loaded"   # not_loaded | ready | error
if "model_error" not in st.session_state:
    st.session_state.model_error = None

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    """Load the trained YOLO weights via Ultralytics.
    Cached so the model is only loaded into memory once per session."""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return model


def ensure_model_loaded(weights_path: str):
    """Loads the model with a visible telemetry-style progress
    sequence so weight loading is never a silent freeze."""

    if st.session_state.model_status == "ready":
        return True

    feed = st.empty()
    progress = st.progress(0)
    rows = []

    stages = [
        ("Locating weights file", 15),
        ("Reading checkpoint header", 35),
        ("Building YOLO architecture", 60),
        ("Loading tensor weights", 85),
        ("Warming up inference engine", 100),
    ]

    def render():
        feed.markdown(
            f'<div class="telemetry-body" style="max-height:160px;">{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )

    try:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights file not found at '{weights_path}'. "
                f"Place your trained .pt file at this path or update MODEL_PATH."
            )

        for label, pct in stages:
            ts = datetime.now().strftime("%H:%M:%S")
            rows.append(
                f'<div class="telemetry-row"><div class="telemetry-time">{ts}</div>'
                f'<div class="telemetry-msg"><span class="sys">{label}...</span></div></div>'
            )
            render()
            progress.progress(pct)
            time.sleep(0.16)

        model = load_yolo_model(weights_path)

        ts = datetime.now().strftime("%H:%M:%S")
        rows.append(
            f'<div class="telemetry-row"><div class="telemetry-time">{ts}</div>'
            f'<div class="telemetry-msg"><span class="conf">Model armed — standing by</span></div></div>'
        )
        render()
        time.sleep(0.25)

        st.session_state.model = model
        st.session_state.model_status = "ready"
        feed.empty()
        progress.empty()
        return True

    except Exception as e:
        st.session_state.model_status = "error"
        st.session_state.model_error = str(e)
        feed.empty()
        progress.empty()
        return False

# =========================================================
# STATUS BAR
# =========================================================

armed = st.session_state.model_status == "ready"
dot_class = "dot armed" if armed else "dot standby"
status_label = "MODEL ARMED" if armed else "MODEL NOT LOADED"

st.markdown(f"""
<div class="glass-panel statusbar">
    <div class="statusbar-left">
        <div class="statusbar-mark">⬡</div>
        <div>
            <div class="statusbar-name">AEGIS</div>
            <div class="statusbar-sub">TACTICAL DETECTION CONSOLE</div>
        </div>
    </div>
    <div class="statusbar-right">
        <div><span class="{dot_class}"></span>{status_label}</div>
        <div>{datetime.now().strftime('%Y-%m-%d · %H:%M:%S')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">YOLO inference · custom-trained detector</div>
    <div class="hero-title">Track and identify drones across<br>uploaded video, in real time.</div>
    <div class="hero-desc">
        Upload a video feed. Aegis runs every frame through a custom-trained YOLO model,
        plots contacts on the viewport, and logs each detection to the telemetry feed.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="side-title">Console settings</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-heading">Model specification</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="spec-row"><span>Architecture</span><span>YOLO</span></div>
    <div class="spec-row"><span>Weights</span><span>.pt</span></div>
    <div class="spec-row"><span>Accuracy</span><span>{MODEL_ACCURACY_LABEL}%</span></div>
    <div class="spec-row"><span>Task</span><span>Drone detect</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="side-divider">', unsafe_allow_html=True)

    st.markdown('<div class="side-heading">Detection parameters</div>', unsafe_allow_html=True)
    confidence_threshold = st.slider(
        "Confidence threshold", min_value=0.05, max_value=0.95,
        value=MODEL_CONF_THRESHOLD_DEFAULT, step=0.05,
    )
    frame_skip = st.slider(
        "Process every Nth frame", min_value=1, max_value=10, value=1, step=1,
        help="Higher values process faster but may miss brief contacts.",
    )

    st.markdown('<hr class="side-divider">', unsafe_allow_html=True)

    st.markdown('<div class="side-heading">Accepted formats</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="format-chip-row">
        <div class="format-chip">.MP4</div>
        <div class="format-chip">.MOV</div>
        <div class="format-chip">.AVI</div>
        <div class="format-chip">.MKV</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MODEL PANEL
# =========================================================

st.markdown('<div class="sec-label">Model</div>', unsafe_allow_html=True)

badge_class = "badge ready" if armed else ("badge err" if st.session_state.model_status == "error" else "badge idle")
badge_text = "● READY" if armed else ("● ERROR" if st.session_state.model_status == "error" else "● NOT LOADED")

st.markdown(f"""
<div class="glass-panel model-panel">
    <div class="model-left">
        <div class="model-icon">⬡</div>
        <div>
            <div class="model-name">Custom YOLO Drone Detector</div>
            <div class="model-path">{MODEL_PATH}</div>
        </div>
    </div>
    <div class="{badge_class}">{badge_text}</div>
</div>
""", unsafe_allow_html=True)

if not armed:
    if st.button("Load model"):
        ensure_model_loaded(MODEL_PATH)
        st.rerun()

if st.session_state.model_status == "error":
    st.error(
        f"Model failed to load: {st.session_state.model_error}\n\n"
        f"Update MODEL_PATH at the top of this script to point to your trained .pt weights file."
    )

# =========================================================
# VIDEO UPLOAD
# =========================================================

st.markdown('<div class="sec-label">Feed upload</div>', unsafe_allow_html=True)

uploaded_video = st.file_uploader(
    "Upload a video file to scan for drones",
    type=["mp4", "mov", "avi", "mkv"],
    label_visibility="collapsed",
)

# =========================================================
# HUD-STYLE BOX DRAWING (clean targeting frame, not reticles)
# =========================================================

def draw_target_box(frame, x1, y1, x2, y2, conf, color=(255, 157, 61)):
    """Thin rectangle + corner ticks + label, matching the
    cyan/amber glass aesthetic (amber = active contact)."""

    thickness = 2
    tick = max(10, int(min(x2 - x1, y2 - y1) * 0.15))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

    for (cx, cy, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * tick, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * tick), color, thickness)

    tag = f"DRONE {conf*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (8, 14, 20), 1, cv2.LINE_AA)

    return frame

# =========================================================
# VIDEO PROCESSING — SPLIT SCREEN
# =========================================================

if uploaded_video is not None:

    if not armed:
        st.warning("Load the model above before running detection on a video.")

    else:
        st.markdown('<div class="sec-label">Run detection</div>', unsafe_allow_html=True)

        start_detection = st.button("Start detection")

        if start_detection:

            with tempfile.TemporaryDirectory() as temp_dir:

                input_path = os.path.join(temp_dir, "input_video")
                with open(input_path, "wb") as f:
                    f.write(uploaded_video.read())

                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                output_path = os.path.join(temp_dir, "annotated_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0)

                progress_bar = st.progress(0)
                progress_caption = st.empty()

                detections_total = 0
                frames_with_detection = 0
                frame_idx = 0
                t0 = time.time()

                model = st.session_state.model

                # ---- silent processing loop: no live frame-by-frame preview ----
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    run_inference = (frame_idx % frame_skip == 0)

                    if run_inference:
                        results = model.predict(frame, conf=confidence_threshold, verbose=False)[0]
                        boxes = results.boxes
                        frame_had_detection = False

                        if boxes is not None and len(boxes) > 0:
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                conf = float(box.conf[0])
                                frame = draw_target_box(frame, x1, y1, x2, y2, conf)
                                detections_total += 1
                                frame_had_detection = True

                        if frame_had_detection:
                            frames_with_detection += 1

                    writer.write(frame)
                    frame_idx += 1

                    if total_frames > 0 and frame_idx % 5 == 0:
                        pct = min(frame_idx / total_frames, 1.0)
                        progress_bar.progress(pct)
                        progress_caption.markdown(
                            f'<div class="term-dim" style="font-family: var(--mono); font-size:12px; text-align:center; margin-top:8px;">'
                            f'Scanning footage — <span style="color: var(--accent);">{int(pct*100)}%</span> complete '
                            f'({frame_idx} / {total_frames} frames)</div>',
                            unsafe_allow_html=True,
                        )

                cap.release()
                writer.release()
                progress_bar.progress(1.0)
                elapsed = time.time() - t0
                avg_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress_caption.empty()

                # -------------------------------------------------
                # RE-ENCODE for browser-safe playback (mp4v written
                # by OpenCV isn't always playable in <video> tags;
                # falls back to the raw file if ffmpeg is unavailable).
                # -------------------------------------------------

                final_output_path = os.path.join(temp_dir, "aegis_annotated_output.mp4")
                reencoded = False

                if shutil.which("ffmpeg"):
                    ffmpeg_cmd = (
                        f'ffmpeg -y -i "{output_path}" '
                        f'-vcodec libx264 -pix_fmt yuv420p -movflags +faststart '
                        f'"{final_output_path}" -loglevel error'
                    )
                    ret_code = os.system(ffmpeg_cmd)
                    reencoded = (ret_code == 0 and os.path.exists(final_output_path))

                if not reencoded:
                    final_output_path = output_path

                # -------------------------------------------------
                # RESULTS
                # -------------------------------------------------

                st.markdown('<div class="sec-label">Run summary</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="tile-row">
                    <div class="glass-panel tile">
                        <div class="tile-val contact">{detections_total}</div>
                        <div class="tile-label">Total contacts</div>
                    </div>
                    <div class="glass-panel tile">
                        <div class="tile-val accent">{frames_with_detection}</div>
                        <div class="tile-label">Frames w/ contact</div>
                    </div>
                    <div class="glass-panel tile">
                        <div class="tile-val">{frame_idx}</div>
                        <div class="tile-label">Frames processed</div>
                    </div>
                    <div class="glass-panel tile">
                        <div class="tile-val">{avg_fps:.1f}</div>
                        <div class="tile-label">Throughput (fps)</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="sec-label">Annotated feed</div>', unsafe_allow_html=True)

                with open(final_output_path, "rb") as f:
                    video_bytes = f.read()

                st.markdown('<div class="glass-panel" style="padding: 12px;">', unsafe_allow_html=True)
                st.video(video_bytes)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)

                st.download_button(
                    label="Download annotated video (.mp4)",
                    data=video_bytes,
                    file_name="aegis_annotated_output.mp4",
                    mime="video/mp4",
                )

# =========================================================
# FOOTER
# =========================================================

st.markdown(f"""
<div class="footer-bar">
    <div>AEGIS · TACTICAL DETECTION CONSOLE</div>
    <div>MODEL: CUSTOM YOLO · ACCURACY {MODEL_ACCURACY_LABEL}%</div>
</div>
""", unsafe_allow_html=True)