"""
backend.py
==========
All ML model definitions, loading, inference, and XAI logic.
No Streamlit imports — fully independent and testable.
"""

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "Custom CNN": {
        "path": "best_custom_cnn.pth",
        "img_size": 128,
    },
    "VGG16 Transfer Learning": {
        "path": "best_vgg16.pth",
        "img_size": 224,
    },
}


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    pred_idx: int
    pred_label: str
    confidence: float
    prob_unhealthy: float
    model_type: str
    img_size: int

    @property
    def is_healthy(self) -> bool:
        return self.pred_idx == 0

    @property
    def confidence_text(self) -> str:
        return "Very confident" if self.confidence > 0.85 else "Moderately confident"


@dataclass
class XAIResult:
    attr_np: np.ndarray        # shape (H, W, 3)
    convergence_delta: float


# ─────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────

class CustomCNN(nn.Module):
    """Lightweight 3-layer CNN for 128×128 binary leaf classification."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _create_vgg16_model() -> nn.Module:
    """VGG16 with a custom binary head."""
    m = models.vgg16(weights=None)
    m.classifier = nn.Sequential(
        nn.Linear(m.classifier[0].in_features, 512),
        nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1),
    )
    return m


# ─────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────

def load_model(model_type: str, model_path: str) -> Tuple[nn.Module, torch.device]:
    """
    Load model weights from *model_path* onto the best available device.

    Parameters
    ----------
    model_type  : "Custom CNN" | "VGG16 Transfer Learning"
    model_path  : path to the .pth checkpoint file

    Returns
    -------
    (model, device)  Both already on *device*, model in eval mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module = (
        CustomCNN() if model_type == "Custom CNN" else _create_vgg16_model()
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model, device


# ─────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image, size: int) -> torch.Tensor:
    """
    Convert a PIL image to a normalised (1, 3, size, size) tensor.
    Uses ImageNet mean/std — appropriate for VGG16 and as a reasonable
    default for the custom CNN.
    """
    pipeline = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return pipeline(img.convert("RGB")).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────

def run_inference(
    pil_img: Image.Image,
    model: nn.Module,
    device: torch.device,
    img_size: int,
    class_names: List[str],
    model_type: str,
) -> PredictionResult:
    """
    Run a single forward pass and return a structured result.

    Parameters
    ----------
    pil_img     : PIL image (any mode — converted to RGB internally)
    model       : loaded, eval-mode nn.Module
    device      : target device
    img_size    : resize target (128 for CNN, 224 for VGG16)
    class_names : [class_0_name, class_1_name]
    model_type  : label string forwarded into the result

    Returns
    -------
    PredictionResult
    """
    inp = preprocess(pil_img, img_size).to(device)
    with torch.no_grad():
        logit     = model(inp)
        prob1     = torch.sigmoid(logit).squeeze().item()

    pred_idx   = int(prob1 > 0.5)
    pred_label = class_names[pred_idx]
    confidence = prob1 if pred_idx == 1 else 1.0 - prob1

    return PredictionResult(
        pred_idx       = pred_idx,
        pred_label     = pred_label,
        confidence     = confidence,
        prob_unhealthy = prob1,
        model_type     = model_type,
        img_size       = img_size,
    )


# ─────────────────────────────────────────────────────────────────
# XAI — Integrated Gradients via Captum
# ─────────────────────────────────────────────────────────────────

def generate_xai_attributions(
    img_bytes: bytes,
    model: nn.Module,
    device: torch.device,
    img_size: int,
    n_steps: int = 30,
) -> Optional[XAIResult]:
    """
    Compute Integrated Gradients attributions for *img_bytes*.

    Returns None (gracefully) if Captum is not installed.

    Parameters
    ----------
    img_bytes : raw image bytes (e.g. from uploaded_file.getvalue())
    model     : loaded, eval-mode nn.Module
    device    : target device
    img_size  : must match the model's expected input size
    n_steps   : IG approximation steps (higher → more accurate, slower)

    Returns
    -------
    XAIResult | None
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        return None

    pil_img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inp_tensor = preprocess(pil_img, img_size).to(device).requires_grad_(True)

    ig = IntegratedGradients(model)
    attrs, delta = ig.attribute(
        inp_tensor,
        target           = None,
        return_convergence_delta = True,
        n_steps          = n_steps,
        internal_batch_size = 4,
    )

    attr_np    = attrs.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    delta_val  = delta.item()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return XAIResult(attr_np=attr_np, convergence_delta=delta_val)


# ─────────────────────────────────────────────────────────────────
# Visualisation — XAI figure
# ─────────────────────────────────────────────────────────────────

def make_attr_figure(
    pil_img: Image.Image,
    attr_np: np.ndarray,
    img_size: int,
) -> plt.Figure:
    """
    Build and return a 3-panel matplotlib figure:
      [0] Original image
      [1] Importance heatmap
      [2] Overlay

    Parameters
    ----------
    pil_img  : original PIL image
    attr_np  : attribution array of shape (H, W, 3)
    img_size : resize target (used to normalise display image)

    Returns
    -------
    plt.Figure  (caller is responsible for calling plt.close(fig))
    """
    display  = np.array(pil_img.convert("RGB").resize((img_size, img_size))) / 255.0
    attr_mag = np.abs(attr_np).sum(axis=2)
    lo, hi   = np.percentile(attr_mag, 1), np.percentile(attr_mag, 99)
    clipped  = np.clip(attr_mag, lo, hi)
    span     = clipped.max() - clipped.min() + 1e-8
    attr_norm = (clipped - clipped.min()) / span

    BG, SURFACE, INK2, BORDER = "#f7f5f0", "#ffffff", "#5a5751", "#ddd8ce"

    fig = plt.figure(figsize=(24, 10), facecolor=BG)
    gs  = gridspec.GridSpec(
        1, 3, figure=fig, wspace=0.10,
        left=0.01, right=0.95, top=0.84, bottom=0.04,   # more headroom for titles
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, title in zip(axes, ["Original Image", "Importance Heatmap", "Overlay"]):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK2, fontsize=28, fontfamily="sans-serif",
                     pad=20, fontweight="700")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
            sp.set_linewidth(1.2)

    axes[0].imshow(display)

    im = axes[1].imshow(attr_norm, cmap="YlOrRd", vmin=0, vmax=1, interpolation="bilinear")
    cb = plt.colorbar(im, ax=axes[1], fraction=0.042, pad=0.04, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["Low", "Mid", "High"], color=INK2, fontsize=14, fontweight="500")
    cb.outline.set_edgecolor(BORDER)
    cb.outline.set_linewidth(1)
    cb.ax.yaxis.set_tick_params(color=BORDER, length=4, width=1)

    axes[2].imshow(display)
    axes[2].imshow(attr_norm, cmap="YlOrRd", alpha=0.58, vmin=0, vmax=1, interpolation="bilinear")

    fig.text(0.97, 0.02, "Integrated Gradients · Captum",
             ha="right", va="bottom", fontsize=9, color=INK2, alpha=0.5, style="italic")

    return fig