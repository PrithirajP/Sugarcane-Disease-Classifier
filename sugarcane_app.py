"""
Sugarcane Disease Classifier — Streamlit App (Redesigned UI)
Supports Custom CNN and VGG16 with Captum Integrated Gradients explainability.
"""

import io
import os
from typing import Tuple, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────────
# Page config & CSS
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sugarcane Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_custom_css():
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

    html, body, [class*="css"] {
        font-family: var(--sans);
        background-color: var(--bg);
        color: var(--ink);
    }
    .stApp {
        background: var(--bg);
    }

    /* ── Background texture using CSS gradient ── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 120% 80% at 80% 10%, rgba(90,122,94,0.06) 0%, transparent 60%),
            radial-gradient(ellipse 80% 60% at 10% 90%, rgba(201,124,46,0.04) 0%, transparent 50%),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 40px,
                rgba(90,122,94,0.015) 40px,
                rgba(90,122,94,0.015) 41px
            );
        pointer-events: none;
        z-index: 0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        box-shadow: 2px 0 20px rgba(80,70,50,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: var(--surface2) !important;
        border-color: var(--border2) !important;
        color: var(--ink) !important;
        border-radius: var(--radius-sm) !important;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: var(--surface2) !important;
        border-color: var(--border2) !important;
        color: var(--ink) !important;
        border-radius: var(--radius-sm) !important;
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
        padding: 0 !important;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f3efe7 50%, #eaf2ea 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.4rem 2.8rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 220px; height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(90,122,94,0.12) 0%, transparent 70%);
    }
    .hero::after {
        content: '🌿';
        position: absolute;
        right: 2.5rem; top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.12;
        filter: grayscale(0.3);
    }
    .hero-eyebrow {
        font-family: var(--mono);
        font-size: 0.68rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--sage);
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }
    .hero h1 {
        font-family: var(--serif);
        font-weight: 700;
        font-size: 2.4rem;
        color: var(--ink);
        margin: 0 0 0.4rem;
        letter-spacing: -0.5px;
        line-height: 1.15;
    }
    .hero h1 span {
        color: var(--sage);
    }
    .hero-sub {
        font-family: var(--sans);
        font-size: 0.88rem;
        color: var(--ink2);
        margin: 0;
        font-weight: 400;
        max-width: 500px;
        line-height: 1.6;
    }
    .hero-tags {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .hero-tag {
        font-family: var(--mono);
        font-size: 0.62rem;
        letter-spacing: 1.5px;
        background: var(--sage-ultra);
        border: 1px solid var(--sage-pale);
        color: var(--sage);
        padding: 0.22rem 0.7rem;
        border-radius: 999px;
    }

    /* ── Section headings ── */
    .section-heading {
        font-family: var(--serif);
        font-size: 1.2rem;
        font-weight: 500;
        color: var(--ink);
        margin: 0 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-heading::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
        margin-left: 0.5rem;
    }

    /* ── Cards ── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
    }
    .card-label {
        font-family: var(--mono);
        font-size: 0.62rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--ink3);
        margin-bottom: 0.55rem;
    }

    /* ── Prediction badge ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.4rem;
        border-radius: 999px;
        font-family: var(--sans);
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.2px;
        margin-bottom: 1.2rem;
    }
    .badge-healthy {
        background: var(--sage-ultra);
        border: 1.5px solid var(--sage-pale);
        color: var(--sage);
    }
    .badge-unhealthy {
        background: var(--red-pale);
        border: 1.5px solid #dda0a0;
        color: var(--red);
    }

    /* ── Confidence bar ── */
    .bar-track {
        background: var(--bg2);
        border-radius: 999px;
        height: 7px;
        margin: 0.6rem 0 0.4rem;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    /* ── Stat pill ── */
    .stat-pill {
        display: inline-block;
        font-family: var(--mono);
        font-size: 0.7rem;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 0.2rem 0.65rem;
        color: var(--ink2);
        margin-right: 0.35rem;
        margin-top: 0.3rem;
    }

    /* ── Legend block ── */
    .legend-wrap {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem 1.7rem;
        margin-top: 0;
        box-shadow: var(--shadow);
    }
    .legend-title {
        font-family: var(--serif);
        font-weight: 500;
        font-size: 1.05rem;
        color: var(--ink);
        margin-bottom: 1.1rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid var(--border);
    }
    .legend-row {
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
        margin-bottom: 0.9rem;
    }
    .legend-swatch {
        width: 32px;
        min-width: 32px;
        height: 32px;
        border-radius: var(--radius-sm);
        margin-top: 2px;
        border: 1px solid var(--border);
    }
    .legend-text strong {
        display: block;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.18rem;
    }
    .legend-text span {
        font-size: 0.76rem;
        color: var(--ink2);
        line-height: 1.5;
    }
    .legend-panel-tag {
        display: inline-block;
        font-family: var(--mono);
        font-size: 0.6rem;
        background: var(--sage-ultra);
        border: 1px solid var(--sage-pale);
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        color: var(--sage);
        margin-bottom: 0.55rem;
        letter-spacing: 1.5px;
    }

    /* ── Upload area ── */
    .upload-hint {
        background: var(--surface);
        border: 1.5px dashed var(--border2);
        border-radius: var(--radius);
        padding: 2rem 1.5rem;
        text-align: center;
        color: var(--ink3);
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .upload-hint:hover {
        border-color: var(--sage-light);
    }

    /* ── Empty state ── */
    .empty-state {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 3rem 1.5rem;
        text-align: center;
        box-shadow: var(--shadow);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--sage) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--sans) !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.6rem !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 2px 8px rgba(90,122,94,0.3) !important;
    }
    .stButton > button:hover {
        background: var(--sage-light) !important;
        box-shadow: 0 4px 16px rgba(90,122,94,0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: var(--surface2) !important;
        color: var(--ink2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--sans) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stDownloadButton > button:hover {
        background: var(--bg2) !important;
        border-color: var(--sage-pale) !important;
        color: var(--sage) !important;
    }

    /* ── Streamlit overrides ── */
    hr { border-color: var(--border); margin: 1.5rem 0; }
    .stFileUploader label { color: var(--ink2) !important; }
    .stSelectbox label, .stSlider label, .stCheckbox label { color: var(--ink2) !important; font-size: 0.85rem !important; }
    .stInfo  { background: var(--sage-ultra) !important; border-color: var(--sage-pale) !important; border-radius: var(--radius-sm) !important; }
    .stSuccess { background: var(--sage-ultra) !important; border-color: var(--sage-pale) !important; border-radius: var(--radius-sm) !important; }
    .stError   { background: var(--red-pale) !important; border-radius: var(--radius-sm) !important; }
    .stWarning { background: var(--amber-pale) !important; border-radius: var(--radius-sm) !important; }
    .stSpinner > div { color: var(--sage) !important; }

    /* ── Sidebar controls ── */
    .sidebar-section {
        background: var(--bg);
        border-radius: var(--radius-sm);
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border: 1px solid var(--border);
    }
    .sidebar-section-title {
        font-family: var(--mono);
        font-size: 0.62rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--ink3);
        margin-bottom: 0.5rem;
    }

    /* ── Model status indicator ── */
    .model-status-ok {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--sage-ultra);
        border: 1px solid var(--sage-pale);
        border-radius: var(--radius-sm);
        padding: 0.5rem 0.8rem;
        font-size: 0.8rem;
        color: var(--sage);
        font-weight: 500;
    }
    .model-status-err {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--red-pale);
        border: 1px solid #dda0a0;
        border-radius: var(--radius-sm);
        padding: 0.5rem 0.8rem;
        font-size: 0.8rem;
        color: var(--red);
        font-weight: 500;
    }

    /* ── Image display ── */
    .stImage img {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow) !important;
    }

    /* ── XAI section header ── */
    .xai-header {
        background: linear-gradient(135deg, var(--surface) 0%, var(--sage-ultra) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: var(--shadow);
    }
    .xai-header-icon {
        font-size: 2rem;
        line-height: 1;
    }
    .xai-header-text h3 {
        font-family: var(--serif);
        font-size: 1.15rem;
        font-weight: 500;
        color: var(--ink);
        margin: 0 0 0.2rem;
    }
    .xai-header-text p {
        font-size: 0.8rem;
        color: var(--ink2);
        margin: 0;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        font-family: var(--mono);
        font-size: 0.65rem;
        color: var(--ink3);
        padding: 1rem 0 0.5rem;
        letter-spacing: 1px;
    }
    .footer span {
        color: var(--sage);
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ─────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_vgg16_model() -> nn.Module:
    m = models.vgg16(weights=None)
    m.classifier = nn.Sequential(
        nn.Linear(m.classifier[0].in_features, 512),
        nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1),
    )
    return m


@st.cache_resource
def load_model(model_type: str, model_path: str) -> Tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN() if model_type == "Custom CNN" else create_vgg16_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device


def preprocess(img: Image.Image, size: int) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return t(img.convert("RGB")).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────
# Core Logic & XAI
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_xai_attributions(img_bytes: bytes, model_type: str, model_path: str, size: int, n_steps: int):
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        return None, None

    model, device = load_model(model_type, model_path)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inp_tensor = preprocess(pil_img, size).to(device).requires_grad_(True)
    ig = IntegratedGradients(model)
    attrs, delta = ig.attribute(
        inp_tensor, target=None,
        return_convergence_delta=True,
        n_steps=n_steps,
        internal_batch_size=4,
    )
    attr_np = attrs.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    delta_val = delta.item()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return attr_np, delta_val


# ─────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────
def make_attr_figure(pil_img: Image.Image, attr_np: np.ndarray, img_size: int) -> plt.Figure:
    display = np.array(pil_img.convert("RGB").resize((img_size, img_size))) / 255.0
    attr_mag = np.abs(attr_np).sum(axis=2)
    lo, hi = np.percentile(attr_mag, 1), np.percentile(attr_mag, 99)
    attr_clipped = np.clip(attr_mag, lo, hi)
    span = attr_clipped.max() - attr_clipped.min() + 1e-8
    attr_norm = (attr_clipped - attr_clipped.min()) / span

    BG       = "#f7f5f0"
    SURFACE  = "#ffffff"
    INK2     = "#5a5751"
    SAGE     = "#5a7a5e"
    BORDER   = "#ddd8ce"

    fig = plt.figure(figsize=(24, 9), facecolor=BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.10,
                           left=0.01, right=0.95, top=0.90, bottom=0.04)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    panel_titles = ["Original Image", "Importance Heatmap", "Overlay"]
    for ax, title in zip(axes, panel_titles):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK2, fontsize=18,
                     fontfamily="sans-serif", pad=16, fontweight="600")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
            sp.set_linewidth(1.2)

    axes[0].imshow(display)

    im = axes[1].imshow(attr_norm, cmap="YlOrRd", vmin=0, vmax=1, interpolation="bilinear")
    cb = plt.colorbar(im, ax=axes[1], fraction=0.042, pad=0.04, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["Low", "Mid", "High"], color=INK2, fontsize=12, fontweight="500")
    cb.outline.set_edgecolor(BORDER)
    cb.outline.set_linewidth(1)
    cb.ax.yaxis.set_tick_params(color=BORDER, length=4, width=1)

    axes[2].imshow(display)
    axes[2].imshow(attr_norm, cmap="YlOrRd", alpha=0.58, vmin=0, vmax=1, interpolation="bilinear")

    fig.text(0.97, 0.02, "Integrated Gradients · Captum",
             ha="right", va="bottom", fontsize=9,
             color=INK2, alpha=0.5, style="italic")

    return fig


def render_legend(pred_label: str, confidence: float, class_names: List[str]):
    is_healthy = (pred_label == class_names[0])
    decision_text = (
        "Healthy visual features were detected — no significant disease markings were found in the leaf tissue."
        if is_healthy else
        "Visual features associated with disease were detected — the highlighted regions in the heatmap indicate affected areas."
    )

    legend_html = f"""<div class="legend-wrap">
<div class="legend-title">📖 Reading the Analysis</div>

<div class="legend-panel-tag">PANEL 1 — ORIGINAL</div>
<div class="legend-row" style="margin-bottom:1rem;">
<div class="legend-swatch" style="background:linear-gradient(135deg,#8db08f,#5a7a5e);"></div>
<div class="legend-text"><strong>Your uploaded image</strong><span>The original leaf photo the model examined, unmodified.</span></div>
</div>

<div class="legend-panel-tag">PANEL 2 — HEATMAP</div>
<div class="legend-row">
<div class="legend-swatch" style="background:linear-gradient(to bottom, #b22222, #e07040, #f5c842);"></div>
<div class="legend-text"><strong>Model attention map</strong><span>
<span style="color:#b22222;font-weight:600;">■ Red</span> — strongest influence on the decision<br>
<span style="color:#e07040;font-weight:600;">■ Orange</span> — moderate importance<br>
<span style="color:#d4a020;font-weight:600;">■ Yellow</span> — low but present influence<br>
<span style="color:#9a958e;">■ Dark</span> — largely ignored by the model
</span></div>
</div>

<div class="legend-panel-tag" style="margin-top:0.6rem;">PANEL 3 — OVERLAY</div>
<div class="legend-row" style="margin-bottom:0.8rem;">
<div class="legend-swatch" style="background:linear-gradient(135deg,#b44030 40%,#5a7a5e 100%);"></div>
<div class="legend-text"><strong>Heatmap on leaf</strong><span>Same colours mapped directly onto the image to pinpoint exactly which regions drove the prediction.</span></div>
</div>

<hr style="border:none;border-top:1px solid var(--border);margin:0.6rem 0 0.8rem;">

<div class="legend-row" style="align-items:center;margin-bottom:0;">
<div class="legend-swatch" style="background:{'#eaf2ea' if is_healthy else '#f5e0df'};border:1.5px solid {'#7a9e7e' if is_healthy else '#dda0a0'};display:flex;align-items:center;justify-content:center;font-size:1.1rem;">{'✅' if is_healthy else '⚠️'}</div>
<div class="legend-text"><strong>Conclusion: {pred_label} &nbsp;({confidence*100:.1f}%)</strong><span>{decision_text}</span></div>
</div>
</div>"""

    st.markdown(legend_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# App State
# ─────────────────────────────────────────────────────────────────
if 'has_analyzed' not in st.session_state:
    st.session_state['has_analyzed'] = False
if 'current_file_id' not in st.session_state:
    st.session_state['current_file_id'] = None

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.2rem;'>
        <div style='font-family:"DM Mono",monospace;font-size:0.6rem;letter-spacing:3px;color:#9a958e;text-transform:uppercase;margin-bottom:0.3rem;'>Configuration</div>
        <div style='font-family:"Playfair Display",serif;font-size:1.3rem;color:#2c2c2c;font-weight:500;'>Settings</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="sidebar-section-title">MODEL</div>""", unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Select model",
        ["Custom CNN", "VGG16 Transfer Learning"],
        help="Custom CNN: faster & lighter (128×128). VGG16: more powerful (224×224).",
        label_visibility="collapsed",
    )
    IMG_SIZE   = 128 if model_choice == "Custom CNN" else 224
    MODEL_PATH = "best_custom_cnn.pth" if model_choice == "Custom CNN" else "best_vgg16.pth"

    st.markdown(f"""
    <div style='display:flex;gap:0.4rem;margin:0.5rem 0 1.2rem;flex-wrap:wrap;'>
        <span class='stat-pill'>{model_choice}</span>
        <span class='stat-pill'>{IMG_SIZE}×{IMG_SIZE}px</span>
        <span class='stat-pill'>Binary</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="sidebar-section-title">CLASS LABELS</div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    CLASS_NAMES = [
        c1.text_input("Class 0", value="Healthy"),
        c2.text_input("Class 1", value="Unhealthy"),
    ]

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-section-title">MODEL WEIGHTS</div>""", unsafe_allow_html=True)
    if os.path.exists(MODEL_PATH):
        kb = os.path.getsize(MODEL_PATH) // 1024
        st.markdown(f"""
        <div class='model-status-ok'>
            ✓ &nbsp;<code style='font-size:0.75rem;background:none;color:inherit;'>{MODEL_PATH}</code>
            <span style='margin-left:auto;opacity:0.7;font-size:0.72rem;'>{kb:,} KB</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='model-status-err'>
            ✗ &nbsp;<code style='font-size:0.75rem;background:none;color:inherit;'>{MODEL_PATH}</code> not found
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-section-title">EXPLAINABILITY</div>""", unsafe_allow_html=True)
    show_xai = st.checkbox("Show AI Explanation (XAI)", value=True)
    if show_xai:
        n_steps = st.slider(
            "Explanation Quality",
            min_value=10, max_value=80, value=30, step=10,
            help="Higher = more accurate, slower.",
        )
        st.caption("30 steps is a good default balance.")
    else:
        n_steps = 30

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"DM Mono",monospace;font-size:0.62rem;color:#9a958e;line-height:1.6;padding-top:0.5rem;border-top:1px solid #ddd8ce;'>
        PyTorch · Captum · Streamlit<br>Binary leaf health classifier
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
# Main columns
# ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.35], gap="large")

with col_left:
    st.markdown('<div class="section-heading">📤 Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "leaf",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        if st.session_state['current_file_id'] != uploaded.file_id:
            st.session_state['has_analyzed'] = False
            st.session_state['current_file_id'] = uploaded.file_id

        pil_img   = Image.open(uploaded).convert("RGB")
        img_bytes = uploaded.getvalue()
        st.image(pil_img, use_container_width=True, caption="Ready to analyse")

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🔍 Analyse Leaf", use_container_width=True):
            st.session_state['has_analyzed'] = True
    else:
        st.session_state['has_analyzed'] = False
        st.session_state['current_file_id'] = None
        st.markdown("""
        <div class="upload-hint">
            <div style="font-size:2.2rem;margin-bottom:0.6rem;opacity:0.5;">🍃</div>
            <strong style="color:#5a5751;">Drop a sugarcane leaf photo here</strong><br>
            <span style="font-size:0.78rem;color:#9a958e;">JPG · PNG · WEBP supported</span>
        </div>
        """, unsafe_allow_html=True)

# ── Right column: results ─────────────────────────────────────────
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

    elif uploaded and st.session_state['has_analyzed']:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model weights `{MODEL_PATH}` not found. Check the sidebar.")
        else:
            with st.spinner(f"Running {model_choice}…"):
                model, device = load_model(model_choice, MODEL_PATH)
                inp_tensor = preprocess(pil_img, IMG_SIZE)
                with torch.no_grad():
                    logit  = model(inp_tensor.to(device))
                    prob1  = torch.sigmoid(logit).squeeze().item()

                pred_idx   = int(prob1 > 0.5)
                pred_label = CLASS_NAMES[pred_idx]
                confidence = prob1 if pred_idx == 1 else 1 - prob1

            is_healthy = pred_idx == 0
            badge_cls  = "badge-healthy"   if is_healthy else "badge-unhealthy"
            bar_colour = "#5a7a5e"          if is_healthy else "#b85450"
            icon       = "✅"               if is_healthy else "⚠️"
            conf_text  = "Very confident"  if confidence > 0.85 else "Moderately confident"

            card_html = f"""<div class="card">
<div class="card-label">Prediction</div>
<div class="badge {badge_cls}">{icon} &nbsp;{pred_label}</div>

<div class="card-label">Confidence Score</div>
<div style="font-size:2.4rem;font-weight:700;color:{bar_colour};line-height:1;margin-bottom:0.3rem;font-family:'Playfair Display',serif;">{confidence*100:.1f}%</div>
<div class="bar-track">
  <div class="bar-fill" style="width:{confidence*100:.1f}%;background:{bar_colour};"></div>
</div>
<div style="font-size:0.78rem;color:#9a958e;margin-bottom:1.2rem;">{conf_text} in this result.</div>

<div class="card-label">Diagnostics</div>
<span class="stat-pill">Model: {model_choice}</span>
<span class="stat-pill">P(Unhealthy): {prob1:.3f}</span>
<span class="stat-pill">Input: {IMG_SIZE}px</span>
</div>"""

            st.markdown(card_html, unsafe_allow_html=True)

            if show_xai:
                with st.spinner("Computing AI explanation…"):
                    attr_np, delta_val = generate_xai_attributions(
                        img_bytes, model_choice, MODEL_PATH, IMG_SIZE, n_steps
                    )
                    if attr_np is None:
                        st.warning("Captum not installed — run `pip install captum`.")
                    else:
                        st.markdown(f"""
                        <div style='font-family:"DM Mono",monospace;font-size:0.68rem;
                        color:#7a9e7e;margin-top:0.2rem;display:flex;align-items:center;gap:0.4rem;'>
                        <span style='color:#5a7a5e;'>✓</span> Explanation ready &nbsp;·&nbsp;
                        convergence δ = {delta_val:.5f}
                        </div>
                        """, unsafe_allow_html=True)

    elif uploaded and not st.session_state['has_analyzed']:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">🔍</div>
            <div style="color:#5a5751;font-size:0.88rem;">
                Image ready. Click <strong>Analyse Leaf</strong> to run the model.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# XAI Section — full width
# ─────────────────────────────────────────────────────────────────
if uploaded and st.session_state.get('has_analyzed', False) and show_xai:
    try:
        if 'attr_np' in locals() and attr_np is not None:
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="xai-header">
                <div class="xai-header-icon">🔬</div>
                <div class="xai-header-text">
                    <h3>AI Explanation — What Did the Model See?</h3>
                    <p>Integrated Gradients highlights which regions of the leaf most influenced the prediction.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_col, leg_col = st.columns([3, 1], gap="large")

            with fig_col:
                fig = make_attr_figure(pil_img, attr_np, IMG_SIZE)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=280, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close(fig)

                st.download_button(
                    "⬇ Download Explanation Image",
                    data=buf.getvalue(),
                    file_name=f"explanation_{model_choice.replace(' ','_')}.png",
                    mime="image/png",
                )

            with leg_col:
                render_legend(pred_label, confidence, CLASS_NAMES)

    except Exception as e:
        st.error(f"Error rendering explanation: {str(e)}")

# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    Sugarcane Disease Classifier &nbsp;·&nbsp;
    <span>PyTorch</span> + <span>Captum</span> + <span>Streamlit</span>
    &nbsp;·&nbsp; Integrated Gradients XAI
</div>
""", unsafe_allow_html=True)