# 🌿 Sugarcane Disease Classifier

> Binary leaf-health classification with deep learning and pixel-level explainability — built on PyTorch and Captum.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Models](#models)
   - [Custom CNN](#1-custom-cnn)
   - [VGG16 Transfer Learning](#2-vgg16-transfer-learning)
   - [Model Comparison](#model-comparison)
4. [Training Pipeline](#training-pipeline)
5. [Explainability — Integrated Gradients](#explainability--integrated-gradients)
   - [Why XAI Matters Here](#why-xai-matters-here)
   - [How Integrated Gradients Works](#how-integrated-gradients-works)
   - [Reading the Output](#reading-the-output)
   - [Convergence Delta](#convergence-delta)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
8. [Running the App](#running-the-app)
9. [Streamlit Interface](#streamlit-interface)
10. [Results at a Glance](#results-at-a-glance)

---

## Project Overview

Sugarcane is one of the world's most economically important crops. Leaf diseases, if undetected, can devastate entire harvests. This project builds a **binary image classifier** — `Healthy` vs `Unhealthy` — that not only makes a prediction but also explains *where* in the leaf it found evidence of disease using **Integrated Gradients**, a gradient-based attribution method from the field of Explainable AI (XAI).

Two model variants are provided:

| Model | Input | Use case |
|---|---|---|
| Custom CNN | 128 × 128 px | Fast inference, lower resource footprint |
| VGG16 (Transfer Learning) | 224 × 224 px | Higher accuracy, production-grade |

---

## Dataset

📦 **Download:** [Sugarcane Disease Dataset (Google Drive)](https://drive.google.com/file/d/1Svp70KvoKT1u2b3E00BdUNU1NC8pg0W_/view?usp=drive_link)

After downloading, extract and place the folder as:

```
f_data/
├── Healthy/        ← class 0
└── Unhealthy/      ← class 1
```

**Total images:** 2,401  
**Split strategy** (seeded with `torch.manual_seed(42)` for reproducibility):

| Split | Proportion | Size |
|---|---|---|
| Train | 70 % | 1,680 images |
| Validation | 15 % | 360 images |
| Test | 15 % | 361 images (176 Healthy, 185 Unhealthy) |

The train set uses **data augmentation**; validation and test sets use clean transforms only. This prevents augmentation leakage into evaluation.

**Augmentation applied to training only:**

```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

All splits are normalised with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), which is appropriate for both VGG16 fine-tuning and as a stable baseline for the custom CNN.

---

## Models

### 1. Custom CNN

A lightweight 3-block convolutional network designed for 128 × 128 inputs.

**Architecture:**

```
Input (3 × 128 × 128)
│
├─ Block 1: Conv2d(3→32, 3×3) → BatchNorm → ReLU → MaxPool2d(2) → Dropout2d(0.25)
├─ Block 2: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool2d(2) → Dropout2d(0.25)
├─ Block 3: Conv2d(64→128, 3×3) → BatchNorm → ReLU → MaxPool2d(2)
│
│   Feature map: 128 × 16 × 16 = 32,768 features
│
├─ Flatten
├─ Linear(32768 → 256) → ReLU → Dropout(0.5)
└─ Linear(256 → 1)     ← raw logit (no sigmoid; BCEWithLogitsLoss handles it)
```

Each convolutional block follows the standard Conv → BN → ReLU → Pool pattern. **BatchNorm** stabilises training by normalising activations per mini-batch. **Spatial Dropout** (Dropout2d) drops entire feature maps rather than individual neurons, which is more effective for convolutional representations. The classifier head uses standard **Dropout(0.5)** to prevent overfitting on the dense layer.

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimiser | Adam (`lr=1e-3`, `weight_decay=1e-4`) |
| Loss | `BCEWithLogitsLoss` |
| Max epochs | 30 |
| Early stopping patience | 5 epochs |
| LR scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2) |
| Batch size | 32 |

---

### 2. VGG16 Transfer Learning

VGG16 pre-trained on ImageNet-1K is used as a frozen feature extractor. Only the classifier head is trained, dramatically reducing the number of learnable parameters and training time.

**Modification strategy:**

```python
# All feature layers frozen — ImageNet weights preserved
for param in model.features.parameters():
    param.requires_grad = False

# Custom binary head replaces the original 1000-class classifier
model.classifier = nn.Sequential(
    nn.Linear(25088 → 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512 → 1),   # binary logit
)
```

The frozen backbone (conv layers 1–13) acts as a universal feature extractor — edges, textures, shapes — that generalises well to leaf imagery. Only the 512-unit head (~13M fewer parameters than full fine-tuning) is updated.

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimiser | Adam (`lr=1e-4`, `weight_decay=1e-4`) |
| Loss | `BCEWithLogitsLoss` |
| Max epochs | 20 |
| Early stopping patience | 4 epochs |
| LR scheduler | `ReduceLROnPlateau` (factor 0.5, patience 2) |
| Batch size | 32 |

The lower learning rate (`1e-4` vs `1e-3`) reflects that only a small head is being trained — aggressive updates would destabilise the pre-trained representation boundary.

---

### Model Comparison

| Metric | Custom CNN | VGG16 |
|---|---|---|
| Input resolution | 128 × 128 | 224 × 224 |
| Trainable parameters | ~8.6M | ~13M (head only) |
| Epochs to converge | 21 (early stop) | 12 (early stop) |
| Best val loss | 0.0698 | 0.0757 |
| Best val accuracy | 96.11 % | 97.78 % |
| Test accuracy | **94 %** | **95 %** |
| Test F1 (macro) | **0.94** | **0.95** |
| Inference speed | Fast (128 px) | Slower (224 px) |
| Checkpoint file | `best_custom_cnn.pth` | `best_vgg16.pth` |

> **Custom CNN test results** (classification report, 361 images):
>
> ```
>               precision  recall  f1-score  support
>    Healthy       0.96     0.92     0.94      176
>  Unhealthy       0.93     0.96     0.94      185
>   accuracy                         0.94      361
>  macro avg       0.94     0.94     0.94      361
> ```

> **VGG16 test results** (classification report, 361 images):
>
> ```
>               precision  recall  f1-score  support
>    Healthy       0.93     0.98     0.95      176
>  Unhealthy       0.98     0.93     0.95      185
>   accuracy                         0.95      361
>  macro avg       0.95     0.95     0.95      361
> ```

Both models show balanced precision and recall across classes, indicating no systematic bias toward either `Healthy` or `Unhealthy`. VGG16 edges ahead on test accuracy (95 % vs 94 %) while converging in roughly half the epochs.

---

## Training Pipeline

The training loop in `Pytorch_models.ipynb` follows a clean separation of concerns:

```
make_transforms()   → build augmented / clean pipelines
build_loaders()     → train / val / test DataLoader objects
train_model()       → epoch loop with:
                        • BCEWithLogitsLoss forward pass
                        • Adam + ReduceLROnPlateau
                        • per-epoch val evaluation
                        • best-checkpoint saving
                        • early stopping
evaluate_model()    → classification report + confusion matrix
plot_history()      → loss & accuracy curves
```

**Early stopping** monitors validation loss. If it does not improve for `patience` consecutive epochs, training halts and the best checkpoint (lowest val loss) is kept — preventing overfitting without guessing the right number of epochs in advance.

**`ReduceLROnPlateau`** halves the learning rate whenever validation loss plateaus for 2 epochs. This allows the model to escape local flat regions late in training without requiring manual schedule tuning.

---

## Explainability — Integrated Gradients

### Why XAI Matters Here

A model that says "Unhealthy" is not enough in agricultural or medical contexts. A farmer, agronomist, or researcher needs to know *which part of the leaf* triggered that prediction — to verify the model is focusing on lesions, discolouration, or disease spots rather than background artefacts (soil, stems, lighting). Integrated Gradients provides exactly that: a pixel-level attribution map grounded in the model's own computation.

---

### How Integrated Gradients Works

Integrated Gradients (IG) is a **model-agnostic attribution method** that satisfies two desirable axioms: *Sensitivity* (if a feature affects the output, it gets non-zero attribution) and *Implementation Invariance* (two models that compute the same function assign the same attributions, regardless of implementation).

**The core idea** is to accumulate gradients along a straight path from a neutral *baseline* image (typically all zeros — a black image) to the actual input:

$$\text{IG}_i(\mathbf{x}) = (\mathbf{x}_i - \mathbf{x}'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial \mathbf{x}_i} \, d\alpha$$

Where:
- $\mathbf{x}$ = input image
- $\mathbf{x}'$ = baseline (black image)
- $F$ = model output (logit)
- $\alpha$ = interpolation step along the path
- $i$ = pixel index

In practice, the integral is approximated with a **Riemann sum** over `n_steps` interpolated images:

```python
ig = IntegratedGradients(model)
attrs, delta = ig.attribute(
    input_tensor,
    target=None,                    # binary model, no class index needed
    return_convergence_delta=True,
    n_steps=30,                     # configurable in the UI
    internal_batch_size=4,          # process 4 interpolation steps at a time
)
```

The result `attrs` has the same shape as the input `(1, 3, H, W)`. To get a single-channel spatial importance map:

```python
attr_magnitude = np.abs(attrs).sum(axis=channel_dim)   # sum over RGB
attr_normalised = (attr_magnitude - min) / (max - min) # scale to [0, 1]
```

Summing absolute values across the channel dimension gives the *total attribution magnitude* per pixel — how much that spatial location, across all colour channels, contributed to the prediction.

---

### Reading the Output

The visualisation produces three panels:

```
┌─────────────────┬──────────────────────┬─────────────────┐
│  Original Image │  Importance Heatmap  │     Overlay     │
│                 │                      │                 │
│  Unmodified     │  Red   = high attn   │  Heatmap fused  │
│  input leaf     │  Orange= mid attn    │  onto original  │
│                 │  Yellow= low attn    │  at α = 0.58    │
│                 │  Dark  = ignored     │                 │
└─────────────────┴──────────────────────┴─────────────────┘
```

**What to look for:**

- In a **healthy** leaf, high-attribution regions typically correspond to uniform leaf texture, vein structure, or normal colouration — the model finds no concentrated anomaly.
- In an **unhealthy** leaf, attributions concentrate tightly around **lesion clusters**, **discolouration patches**, or **necrotic spots** — the same regions a plant pathologist would highlight.
- If attributions concentrate on the background (soil, other plants), this is a signal of **shortcut learning** — the model is using context rather than leaf features. IG makes this visible so it can be corrected.

---

### Convergence Delta

After computing IG, Captum returns a `convergence_delta` value:

```
convergence δ = Σ(attributions) - (F(input) - F(baseline))
```

This measures how closely the sum of attributions satisfies the *completeness axiom* — the guarantee that attributions sum to the difference in model output between input and baseline. **A delta close to zero** means the approximation is faithful. A large delta indicates too few `n_steps` were used. The UI displays this value after each explanation is computed; `n_steps=30` provides a reliable balance between speed and accuracy.

---

## Project Structure

```
sugarcane-disease-classifier/
│
├── Pytorch_models.ipynb      # Full training notebook (data → models → eval → XAI)
│
├── backend.py                # All ML logic — model defs, inference, XAI, plotting
│   ├── CustomCNN             # Lightweight CNN architecture
│   ├── _create_vgg16_model() # VGG16 with frozen backbone
│   ├── load_model()          # Checkpoint loading
│   ├── preprocess()          # PIL → normalised tensor
│   ├── run_inference()       # Forward pass → PredictionResult dataclass
│   ├── generate_xai_attributions()  # Integrated Gradients via Captum
│   └── make_attr_figure()    # 3-panel matplotlib XAI figure
│
├── app.py                    # Streamlit UI — imports backend, zero ML code
│
├── best_custom_cnn.pth       # Saved Custom CNN weights
├── best_vgg16.pth            # Saved VGG16 weights
│
└── f_data/
    ├── Healthy/
    └── Unhealthy/
```

`backend.py` has **no Streamlit imports** — it is independently testable and can be imported in scripts, notebooks, or other frameworks.

---

## Setup & Installation

**Requirements:** Python 3.9+, CUDA optional (CPU works fine for inference).

```bash
# 1. Clone / download the project
git clone <your-repo-url>
cd sugarcane-disease-classifier

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install streamlit captum matplotlib pillow scikit-learn seaborn opencv-python
```

**Training from scratch** (optional — pre-trained `.pth` files provided):

```bash
# Open and run Pytorch_models.ipynb in Jupyter
jupyter notebook Pytorch_models.ipynb
```

Place your dataset under `f_data/Healthy/` and `f_data/Unhealthy/` before running.

---

## Running the App

```bash
streamlit run app.py
```

The app starts at `http://localhost:8501`.

---

## Streamlit Interface

The app (`app.py`) is a thin UI layer over `backend.py`. It provides:

- **Model selector** — switch between Custom CNN and VGG16 from the sidebar
- **Custom class labels** — rename `Healthy`/`Unhealthy` if applied to another crop
- **Image upload** — JPG, PNG, or WEBP
- **One-click inference** — confidence score, probability readout, diagnostic pills
- **XAI panel** — toggleable, with adjustable `n_steps` (10–80) for the IG approximation quality
- **Downloadable explanation** — saves the 3-panel figure as a high-resolution PNG

All `@st.cache_resource` and `@st.cache_data` decorators live in `app.py`, keeping caching concerns out of the ML backend.

---

## Results at a Glance

```
Custom CNN — Test Set (361 images)
────────────────────────────────────────
  Accuracy   :  94.0 %
  Precision  :  0.94  (macro)
  Recall     :  0.94  (macro)
  F1-score   :  0.94  (macro)

  Healthy    →  precision 0.96  recall 0.92
  Unhealthy  →  precision 0.93  recall 0.96

  Best Val Loss     :  0.0698  (epoch 16)
  Best Val Accuracy :  96.11 %
  Converged in      :  21 epochs

VGG16 — Test Set (361 images)
────────────────────────────────────────
  Accuracy   :  95.0 %
  Precision  :  0.95  (macro)
  Recall     :  0.95  (macro)
  F1-score   :  0.95  (macro)

  Healthy    →  precision 0.93  recall 0.98
  Unhealthy  →  precision 0.98  recall 0.93

  Best Val Loss     :  0.0757  (epoch 8)
  Best Val Accuracy :  97.78 %
  Converged in      :  12 epochs
```

Both models were trained on CUDA (`torch.device('cuda')`). Inference runs on CPU in the Streamlit app without any code changes.

---

*Built with PyTorch · Captum · Streamlit*