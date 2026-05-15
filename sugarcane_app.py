"""
app.py  (Streamlit frontend)
============================
Pure UI layer. All ML / XAI logic lives in backend.py.

Run with:
    streamlit run app.py
"""

import io
import os

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

import backend as be
from backend import MODEL_CONFIGS

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sugarcane Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg:           #f7f5f0;
        --bg2:          #eeeae2;
        --surface:      #ffffff;
        --surface2:     #f0ede6;
        --border:       #ddd8ce;
        --border2:      #ccc8bf;
        --sage:         #5a7a5e;
        --sage-light:   #7a9e7e;
        --sage-pale:    #c8dbc9;
        --sage-ultra:   #eaf2ea;
        --amber:        #c97c2e;
        --amber-pale:   #f5e6d0;
        --red:          #b85450;
        --red-pale:     #f5e0df;
        --ink:          #2c2c2c;
        --ink2:         #5a5751;
        --ink3:         #9a958e;
        --serif:        'Playfair Display', Georgia, serif;
        --sans:         'DM Sans', sans-serif;
        --mono:         'DM Mono', monospace;
        --radius:       14px;
        --radius-sm:    8px;
        --shadow:       0 2px 12px rgba(80,70,50,0.09);
        --shadow-lg:    0 8px 32px rgba(80,70,50,0.13);
    }

    html, body, [class*="css"] { font-family: var(--sans); background-color: var(--bg); color: var(--ink); }
    .stApp { background: var(--bg); }
    .stApp::before {
        content: '';
        position: fixed; inset: 0;
        background:
            radial-gradient(ellipse 120% 80% at 80% 10%, rgba(90,122,94,0.06) 0%, transparent 60%),
            radial-gradient(ellipse 80% 60% at 10% 90%, rgba(201,124,46,0.04) 0%, transparent 50%),
            repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(90,122,94,0.015) 40px, rgba(90,122,94,0.015) 41px);
        pointer-events: none; z-index: 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; box-shadow: 2px 0 20px rgba(80,70,50,0.06); }
    section[data-testid="stSidebar"] * { color: var(--ink) !important; }
    section[data-testid="stSidebar"] .stSelectbox > div > div { background: var(--surface2) !important; border-color: var(--border2) !important; border-radius: var(--radius-sm) !important; }
    section[data-testid="stSidebar"] .stTextInput > div > div > input { background: var(--surface2) !important; border-color: var(--border2) !important; border-radius: var(--radius-sm) !important; }

    /* Hero */
    .hero { background: linear-gradient(135deg, #ffffff 0%, #f3efe7 50%, #eaf2ea 100%); border: 1px solid var(--border); border-radius: 20px; padding: 2.4rem 2.8rem 2rem; margin-bottom: 2rem; position: relative; overflow: hidden; box-shadow: var(--shadow-lg); }
    .hero::before { content: ''; position: absolute; top: -40px; right: -40px; width: 220px; height: 220px; border-radius: 50%; background: radial-gradient(circle, rgba(90,122,94,0.12) 0%, transparent 70%); }
    .hero::after { content: '🌿'; position: absolute; right: 2.5rem; top: 50%; transform: translateY(-50%); font-size: 5rem; opacity: 0.12; filter: grayscale(0.3); }
    .hero-eyebrow { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 3px; text-transform: uppercase; color: var(--sage); opacity: 0.8; margin-bottom: 0.5rem; }
    .hero h1 { font-family: var(--serif); font-weight: 700; font-size: 2.4rem; color: var(--ink); margin: 0 0 0.4rem; letter-spacing: -0.5px; line-height: 1.15; }
    .hero h1 span { color: var(--sage); }
    .hero-sub { font-family: var(--sans); font-size: 0.88rem; color: var(--ink2); margin: 0; font-weight: 400; max-width: 500px; line-height: 1.6; }
    .hero-tags { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
    .hero-tag { font-family: var(--mono); font-size: 0.62rem; letter-spacing: 1.5px; background: var(--sage-ultra); border: 1px solid var(--sage-pale); color: var(--sage); padding: 0.22rem 0.7rem; border-radius: 999px; }

    /* Section headings */
    .section-heading { font-family: var(--serif); font-size: 1.2rem; font-weight: 500; color: var(--ink); margin: 0 0 1rem; display: flex; align-items: center; gap: 0.5rem; }
    .section-heading::after { content: ''; flex: 1; height: 1px; background: var(--border); margin-left: 0.5rem; }

    /* Cards */
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.6rem 1.8rem; margin-bottom: 1rem; box-shadow: var(--shadow); }
    .card-label { font-family: var(--mono); font-size: 0.62rem; letter-spacing: 2.5px; text-transform: uppercase; color: var(--ink3); margin-bottom: 0.55rem; }

    /* Badge */
    .badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1.4rem; border-radius: 999px; font-family: var(--sans); font-weight: 600; font-size: 1.1rem; letter-spacing: 0.2px; margin-bottom: 1.2rem; }
    .badge-healthy { background: var(--sage-ultra); border: 1.5px solid var(--sage-pale); color: var(--sage); }
    .badge-unhealthy { background: var(--red-pale); border: 1.5px solid #dda0a0; color: var(--red); }

    /* Confidence bar */
    .bar-track { background: var(--bg2); border-radius: 999px; height: 7px; margin: 0.6rem 0 0.4rem; overflow: hidden; border: 1px solid var(--border); }
    .bar-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }

    /* Stat pill */
    .stat-pill { display: inline-block; font-family: var(--mono); font-size: 0.7rem; background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.2rem 0.65rem; color: var(--ink2); margin-right: 0.35rem; margin-top: 0.3rem; }

    /* Legend */
    .legend-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.6rem 1.7rem; box-shadow: var(--shadow); display: flex; flex-direction: column; }
    .legend-title { font-family: var(--serif); font-weight: 500; font-size: 1.05rem; color: var(--ink); margin-bottom: 1.1rem; padding-bottom: 0.7rem; border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .legend-row { display: flex; align-items: flex-start; gap: 0.85rem; margin-bottom: 0.9rem; }
    .legend-row:last-child { margin-bottom: 0; }
    .legend-swatch { width: 34px; min-width: 34px; height: 34px; border-radius: var(--radius-sm); margin-top: 2px; border: 1px solid var(--border); flex-shrink: 0; }
    .legend-text strong { display: block; font-size: 0.84rem; font-weight: 600; color: var(--ink); margin-bottom: 0.2rem; }
    .legend-text span { font-size: 0.75rem; color: var(--ink2); line-height: 1.55; }
    .legend-panel-tag { display: inline-block; font-family: var(--mono); font-size: 0.6rem; background: var(--sage-ultra); border: 1px solid var(--sage-pale); border-radius: 4px; padding: 0.12rem 0.55rem; color: var(--sage); margin-bottom: 0.5rem; letter-spacing: 1.5px; }
    .legend-conclusion { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.9rem 1rem; margin-top: 0.6rem; flex-shrink: 0; }

    /* Upload */
    .upload-hint { background: var(--surface); border: 1.5px dashed var(--border2); border-radius: var(--radius); padding: 2rem 1.5rem; text-align: center; color: var(--ink3); font-size: 0.88rem; margin-bottom: 0.8rem; }
    .empty-state { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 3rem 1.5rem; text-align: center; box-shadow: var(--shadow); }

    /* Buttons */
    .stButton > button { background: var(--sage) !important; color: #ffffff !important; border: none !important; border-radius: var(--radius-sm) !important; font-family: var(--sans) !important; font-size: 0.88rem !important; font-weight: 600 !important; padding: 0.65rem 1.6rem !important; transition: all 0.2s ease !important; box-shadow: 0 2px 8px rgba(90,122,94,0.3) !important; }
    .stButton > button:hover { background: var(--sage-light) !important; box-shadow: 0 4px 16px rgba(90,122,94,0.4) !important; transform: translateY(-1px) !important; }
    .stDownloadButton > button { background: var(--surface2) !important; color: var(--ink2) !important; border: 1px solid var(--border2) !important; border-radius: var(--radius-sm) !important; font-family: var(--sans) !important; font-size: 0.82rem !important; font-weight: 500 !important; padding: 0.5rem 1.2rem !important; }
    .stDownloadButton > button:hover { background: var(--bg2) !important; border-color: var(--sage-pale) !important; color: var(--sage) !important; }

    /* Misc overrides */
    hr { border-color: var(--border); margin: 1.5rem 0; }
    .stInfo  { background: var(--sage-ultra) !important; border-color: var(--sage-pale) !important; border-radius: var(--radius-sm) !important; }
    .stSuccess { background: var(--sage-ultra) !important; border-color: var(--sage-pale) !important; border-radius: var(--radius-sm) !important; }
    .stError   { background: var(--red-pale) !important; border-radius: var(--radius-sm) !important; }
    .stWarning { background: var(--amber-pale) !important; border-radius: var(--radius-sm) !important; }
    .stSpinner > div { color: var(--sage) !important; }
    .stImage img { border-radius: var(--radius) !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow) !important; }

    /* Sidebar labels */
    .sidebar-section-title { font-family: var(--mono); font-size: 0.62rem; letter-spacing: 2px; text-transform: uppercase; color: var(--ink3); margin-bottom: 0.5rem; }
    .model-status-ok  { display: flex; align-items: center; gap: 0.5rem; background: var(--sage-ultra); border: 1px solid var(--sage-pale); border-radius: var(--radius-sm); padding: 0.5rem 0.8rem; font-size: 0.8rem; color: var(--sage); font-weight: 500; }
    .model-status-err { display: flex; align-items: center; gap: 0.5rem; background: var(--red-pale); border: 1px solid #dda0a0; border-radius: var(--radius-sm); padding: 0.5rem 0.8rem; font-size: 0.8rem; color: var(--red); font-weight: 500; }

    /* XAI header */
    .xai-header { background: linear-gradient(135deg, var(--surface) 0%, var(--sage-ultra) 100%); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.2rem 1.8rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 1rem; box-shadow: var(--shadow); }
    .xai-header-icon { font-size: 2rem; line-height: 1; }
    .xai-header-text h3 { font-family: var(--serif); font-size: 1.15rem; font-weight: 500; color: var(--ink); margin: 0 0 0.2rem; }
    .xai-header-text p { font-size: 0.8rem; color: var(--ink2); margin: 0; }

    /* Footer */
    .footer { text-align: center; font-family: var(--mono); font-size: 0.65rem; color: var(--ink3); padding: 1rem 0 0.5rem; letter-spacing: 1px; }
    .footer span { color: var(--sage); }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Cached wrappers (Streamlit caching lives in the UI layer)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def _cached_load_model(model_type: str, model_path: str):
    return be.load_model(model_type, model_path)


@st.cache_data(show_spinner=False)
def _cached_xai(img_bytes: bytes, model_type: str, model_path: str, img_size: int, n_steps: int):
    model, device = _cached_load_model(model_type, model_path)
    return be.generate_xai_attributions(img_bytes, model, device, img_size, n_steps)


# ─────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────

def _render_result_card(result: be.PredictionResult) -> None:
    bar_colour = "#5a7a5e" if result.is_healthy else "#b85450"
    badge_cls  = "badge-healthy" if result.is_healthy else "badge-unhealthy"
    icon       = "✅" if result.is_healthy else "⚠️"

    st.markdown(f"""
    <div class="card">
      <div class="card-label">Prediction</div>
      <div class="badge {badge_cls}">{icon} &nbsp;{result.pred_label}</div>

      <div class="card-label">Confidence Score</div>
      <div style="font-size:2.4rem;font-weight:700;color:{bar_colour};line-height:1;margin-bottom:0.3rem;font-family:'Playfair Display',serif;">
        {result.confidence*100:.1f}%
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{result.confidence*100:.1f}%;background:{bar_colour};"></div>
      </div>
      <div style="font-size:0.78rem;color:#9a958e;margin-bottom:1.2rem;">{result.confidence_text} in this result.</div>

      <div class="card-label">Diagnostics</div>
      <span class="stat-pill">Model: {result.model_type}</span>
      <span class="stat-pill">P(Unhealthy): {result.prob_unhealthy:.3f}</span>
      <span class="stat-pill">Input: {result.img_size}px</span>
    </div>
    """, unsafe_allow_html=True)


def _render_legend(result: be.PredictionResult, class_names: list) -> None:
    decision = (
        "Healthy visual features were detected — no significant disease markings were found in the leaf tissue."
        if result.is_healthy else
        "Visual features associated with disease were detected — the highlighted regions in the heatmap indicate affected areas."
    )
    icon = "✅" if result.is_healthy else "⚠️"

    st.markdown(f"""
    <div class="legend-wrap">
      <div class="legend-title">📖 Reading the Analysis</div>

      <div class="legend-panel-tag">PANEL 1 — ORIGINAL</div>
      <div class="legend-row">
        <div class="legend-swatch" style="background:linear-gradient(135deg,#8db08f,#5a7a5e);"></div>
        <div class="legend-text">
          <strong>Your uploaded image</strong>
          <span>The original leaf photo the model examined, unmodified.</span>
        </div>
      </div>

      <div class="legend-panel-tag">PANEL 2 — HEATMAP</div>
      <div class="legend-row">
        <div class="legend-swatch" style="background:linear-gradient(to bottom,#b22222,#e07040,#f5c842);"></div>
        <div class="legend-text">
          <strong>Model attention map</strong>
          <span>
            <span style="color:#b22222;font-weight:600;">■ Red</span> — strongest influence<br>
            <span style="color:#e07040;font-weight:600;">■ Orange</span> — moderate importance<br>
            <span style="color:#d4a020;font-weight:600;">■ Yellow</span> — low but present<br>
            <span style="color:#9a958e;">■ Dark</span> — largely ignored
          </span>
        </div>
      </div>

      <div class="legend-panel-tag">PANEL 3 — OVERLAY</div>
      <div class="legend-row">
        <div class="legend-swatch" style="background:linear-gradient(135deg,#b44030 40%,#5a7a5e 100%);"></div>
        <div class="legend-text">
          <strong>Heatmap on leaf</strong>
          <span>Same colours mapped directly onto the image, pinpointing which regions drove the prediction.</span>
        </div>
      </div>

      <div class="legend-conclusion">
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem;">
          <span style="font-size:1.15rem;">{icon}</span>
          <strong style="font-size:0.9rem;color:{'#5a7a5e' if result.is_healthy else '#b85450'};">
            {result.pred_label} &nbsp;·&nbsp; {result.confidence*100:.1f}%
          </strong>
        </div>
        <span style="font-size:0.76rem;color:var(--ink2);line-height:1.55;">{decision}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────

def _build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:0.5rem 0 1.2rem;'>
          <div style='font-family:"DM Mono",monospace;font-size:0.6rem;letter-spacing:3px;color:#9a958e;text-transform:uppercase;margin-bottom:0.3rem;'>Configuration</div>
          <div style='font-family:"Playfair Display",serif;font-size:1.3rem;color:#2c2c2c;font-weight:500;'>Settings</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">MODEL</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Select model",
            list(MODEL_CONFIGS.keys()),
            help="Custom CNN: faster & lighter (128×128). VGG16: more powerful (224×224).",
            label_visibility="collapsed",
        )
        cfg        = MODEL_CONFIGS[model_choice]
        img_size   = cfg["img_size"]
        model_path = cfg["path"]

        st.markdown(f"""
        <div style='display:flex;gap:0.4rem;margin:0.5rem 0 1.2rem;flex-wrap:wrap;'>
          <span class='stat-pill'>{model_choice}</span>
          <span class='stat-pill'>{img_size}×{img_size}px</span>
          <span class='stat-pill'>Binary</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">CLASS LABELS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        class_names = [
            c1.text_input("Class 0", value="Healthy"),
            c2.text_input("Class 1", value="Unhealthy"),
        ]

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">MODEL WEIGHTS</div>', unsafe_allow_html=True)
        if os.path.exists(model_path):
            kb = os.path.getsize(model_path) // 1024
            st.markdown(f"""
            <div class='model-status-ok'>
              ✓ &nbsp;<code style='font-size:0.75rem;background:none;color:inherit;'>{model_path}</code>
              <span style='margin-left:auto;opacity:0.7;font-size:0.72rem;'>{kb:,} KB</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='model-status-err'>
              ✗ &nbsp;<code style='font-size:0.75rem;background:none;color:inherit;'>{model_path}</code> not found
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">EXPLAINABILITY</div>', unsafe_allow_html=True)
        show_xai = st.checkbox("Show AI Explanation (XAI)", value=True)
        n_steps  = 30
        if show_xai:
            n_steps = st.slider(
                "Explanation Quality",
                min_value=10, max_value=80, value=30, step=10,
                help="Higher = more accurate, slower.",
            )
            st.caption("30 steps is a good default balance.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"DM Mono",monospace;font-size:0.62rem;color:#9a958e;line-height:1.6;padding-top:0.5rem;border-top:1px solid #ddd8ce;'>
          PyTorch · Captum · Streamlit<br>Binary leaf health classifier
        </div>
        """, unsafe_allow_html=True)

    return model_choice, img_size, model_path, class_names, show_xai, n_steps


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()

    # ── Session state ─────────────────────────────────────────────
    st.session_state.setdefault("has_analyzed",    False)
    st.session_state.setdefault("current_file_id", None)

    # ── Sidebar ───────────────────────────────────────────────────
    model_choice, img_size, model_path, class_names, show_xai, n_steps = _build_sidebar()

    # ── Hero ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">AI · PLANT PATHOLOGY · COMPUTER VISION</div>
      <h1>Sugarcane <span>Disease</span> Classifier</h1>
      <p class="hero-sub">
        Upload a leaf image to instantly detect disease using deep learning,
        with visual explainability showing exactly what the model examined.
      </p>
      <div class="hero-tags">
        <span class="hero-tag">PYTORCH</span>
        <span class="hero-tag">VGG16</span>
        <span class="hero-tag">CUSTOM CNN</span>
        <span class="hero-tag">XAI · CAPTUM</span>
        <span class="hero-tag">INTEGRATED GRADIENTS</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main columns ──────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.35], gap="large")

    with col_left:
        st.markdown('<div class="section-heading">📤 Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "leaf",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            # Reset analysis state when a new file is uploaded
            if st.session_state["current_file_id"] != uploaded.file_id:
                st.session_state["has_analyzed"]    = False
                st.session_state["current_file_id"] = uploaded.file_id

            pil_img   = Image.open(uploaded).convert("RGB")
            img_bytes = uploaded.getvalue()
            st.image(pil_img, use_container_width=True, caption="Ready to analyse")
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

            if st.button("🔍 Analyse Leaf", use_container_width=True):
                st.session_state["has_analyzed"] = True
        else:
            st.session_state["has_analyzed"]    = False
            st.session_state["current_file_id"] = None
            pil_img   = None
            img_bytes = None
            st.markdown("""
            <div class="upload-hint">
              <div style="font-size:2.2rem;margin-bottom:0.6rem;opacity:0.5;">🍃</div>
              <strong style="color:#5a5751;">Drop a sugarcane leaf photo here</strong><br>
              <span style="font-size:0.78rem;color:#9a958e;">JPG · PNG · WEBP supported</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Right column: results ─────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-heading">📊 Result</div>', unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div class="empty-state">
              <div style="font-size:3rem;margin-bottom:0.8rem;opacity:0.3;">🌱</div>
              <div style="color:#9a958e;font-size:0.88rem;line-height:1.6;">
                Upload a leaf image on the left<br>and click <strong>Analyse Leaf</strong> to see results.
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state["has_analyzed"]:
            if not os.path.exists(model_path):
                st.error(f"Model weights `{model_path}` not found. Check the sidebar.")
            else:
                # ── Inference ─────────────────────────────────────
                with st.spinner(f"Running {model_choice}…"):
                    model, device = _cached_load_model(model_choice, model_path)
                    result        = be.run_inference(
                        pil_img, model, device, img_size, class_names, model_choice
                    )

                _render_result_card(result)

                # ── XAI ───────────────────────────────────────────
                if show_xai:
                    with st.spinner("Computing AI explanation…"):
                        xai = _cached_xai(
                            img_bytes, model_choice, model_path, img_size, n_steps
                        )

                    if xai is None:
                        st.warning("Captum not installed — run `pip install captum`.")
                    else:
                        st.markdown(f"""
                        <div style='font-family:"DM Mono",monospace;font-size:0.68rem;
                        color:#7a9e7e;margin-top:0.2rem;display:flex;align-items:center;gap:0.4rem;'>
                          <span style='color:#5a7a5e;'>✓</span> Explanation ready &nbsp;·&nbsp;
                          convergence δ = {xai.convergence_delta:.5f}
                        </div>
                        """, unsafe_allow_html=True)

        else:  # uploaded but not yet analysed
            st.markdown("""
            <div class="empty-state">
              <div style="font-size:2.5rem;margin-bottom:0.8rem;">🔍</div>
              <div style="color:#5a5751;font-size:0.88rem;">
                Image ready. Click <strong>Analyse Leaf</strong> to run the model.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── XAI full-width panel ──────────────────────────────────────
    if (
        uploaded
        and st.session_state.get("has_analyzed", False)
        and show_xai
        and os.path.exists(model_path)
    ):
        try:
            xai_check = _cached_xai(img_bytes, model_choice, model_path, img_size, n_steps)
            if xai_check is not None and pil_img is not None:
                st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

                # ── Section header ────────────────────────────────
                st.markdown("""
                <div class="xai-header">
                  <div class="xai-header-icon">🔬</div>
                  <div class="xai-header-text">
                    <h3>AI Explanation — What Did the Model See?</h3>
                    <p>Integrated Gradients highlights which regions of the leaf most influenced the prediction.</p>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Balanced 2-column layout: figure left, legend right ──
                # Use [5, 2] so legend gets ~28 % — enough room to breathe
                fig_col, leg_col = st.columns([5, 2], gap="large")

                # Render figure once, reuse buffer for download
                fig = be.make_attr_figure(pil_img, xai_check.attr_np, img_size)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                buf.seek(0)
                plt.close(fig)

                with fig_col:
                    # Wrap image + download in a unified card so heights align
                    st.markdown("""
                    <div style="
                        background:var(--surface);
                        border:1px solid var(--border);
                        border-radius:var(--radius);
                        padding:1.2rem 1.2rem 0.8rem;
                        box-shadow:var(--shadow);
                    ">
                    """, unsafe_allow_html=True)
                    st.image(buf, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Download sits just below the card, left-aligned
                    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
                    st.download_button(
                        "⬇ Download Explanation Image",
                        data=buf.getvalue(),
                        file_name=f"explanation_{model_choice.replace(' ', '_')}.png",
                        mime="image/png",
                    )

                with leg_col:
                    # Re-use already-cached inference result for legend
                    model_for_legend, dev_for_legend = _cached_load_model(model_choice, model_path)
                    result_for_legend = be.run_inference(
                        pil_img, model_for_legend, dev_for_legend,
                        img_size, class_names, model_choice
                    )
                    # Stretch legend to full column height via flex wrapper
                    st.markdown("""
                    <style>
                    /* Make the legend card fill the column height */
                    div[data-testid="column"]:last-child .legend-wrap {
                        height: 100%;
                        box-sizing: border-box;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    _render_legend(result_for_legend, class_names)

        except Exception as exc:
            st.error(f"Error rendering explanation: {exc}")

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
      Sugarcane Disease Classifier &nbsp;·&nbsp;
      <span>PyTorch</span> + <span>Captum</span> + <span>Streamlit</span>
      &nbsp;·&nbsp; Integrated Gradients XAI
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()