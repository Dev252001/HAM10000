<h1 align="center">HAM10000 Skin Lesion Classifier</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab" />
  <img src="https://img.shields.io/badge/Dataset-HAM10000-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Stage%201%20Complete-brightgreen?style=flat-square" />
</p>

<p align="center">
  A portfolio-grade dermoscopic image classifier built on the HAM10000 dataset — 10,015 images across 7 skin lesion classes.<br>
  Progresses from a from-scratch baseline CNN through transfer learning (ResNet18, EfficientNet-B0) to Grad-CAM interpretability,<br>
  with explicit handling of a 42:1 class imbalance and clinical-priority metrics throughout.
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Stages](#stages)
- [Results](#results)
- [How to Run](#how-to-run)
- [Environment](#environment)
- [License](#license)

---

## Project Overview

Skin lesion classification is a clinically meaningful computer vision problem:
the three malignant classes (melanoma, basal cell carcinoma, actinic keratosis)
are also the **minority** classes in the dataset. A naive model that always
predicts the majority class (*melanocytic nevi*) achieves ~67% accuracy while
completely failing to detect cancer — making accuracy a dangerously misleading
metric here.

This project is designed so every design decision can be defended in a technical
interview. Key principles:

- **Primary metrics are macro-F1 and recall on malignant classes** — not accuracy
- **Imbalance is handled explicitly** — class-weighted loss (choice justified in Stage 2)
- **Two architectures compared** on identical conditions — baseline CNN vs fine-tuned transfer models
- **Grad-CAM confirms the model looks at lesion structure**, not background artefacts
- **All tools are free and open-source** — PyTorch, torchvision, pytorch-grad-cam, Google Colab

---

## Dataset

**Source:** [Kaggle — Skin Cancer MNIST: HAM10000 by K Scott Mader](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
**Origin:** ISIC (International Skin Imaging Collaboration) Archive  
**Size:** 10,015 dermoscopy images · 7 classes · heavily imbalanced

| Code | Full Name | Malignant | Count | % |
|------|-----------|:---------:|------:|--:|
| `nv` | Melanocytic Nevi | — | ~6705 | 66.9% |
| `mel` | Melanoma | ⚠️ | ~1113 | 11.1% |
| `bkl` | Benign Keratosis | — | ~1099 | 11.0% |
| `bcc` | Basal Cell Carcinoma | ⚠️ | ~514 | 5.1% |
| `akiec` | Actinic Keratosis / IEC | ⚠️ | ~327 | 3.3% |
| `vasc` | Vascular Lesion | — | ~142 | 1.4% |
| `df` | Dermatofibroma | — | ~115 | 1.1% |

> **Imbalance ratio (nv : vasc) ≈ 42 : 1.**  
> Malignant classes account for only ~19.5% of the data combined.  
> Recall on `mel`, `bcc`, and `akiec` is always reported alongside accuracy.

---

## Project Structure

```
HAM10000/
├── README.md
├── requirements.txt
├── data/                        # ← images not tracked by git (.gitignore)
│   └── .gitkeep
├── notebooks/
│   └── exploration.ipynb        # Stage 1 EDA — run this in Google Colab
├── outputs/
│   ├── figures/                 # saved plots: distribution, confusion matrices, Grad-CAM
│   └── models/                  # saved checkpoints — not tracked by git
└── src/
    ├── data_loader.py            # Kaggle download + HAM10000Dataset class
    ├── preprocessing.py          # stratified split, normalization, augmentation
    ├── train.py                  # shared training loop (baseline + transfer)
    ├── evaluate.py               # accuracy, per-class F1, confusion matrix, malignant recall
    ├── gradcam.py                # Grad-CAM visualizations via pytorch-grad-cam
    └── models/
        ├── baseline_cnn.py       # from-scratch CNN architecture
        └── transfer_models.py    # ResNet18 / EfficientNet-B0 fine-tuning setup
```

The notebook imports from `src/` — functions and classes live in modules, not notebook cells.

---

## Stages

| # | Stage | Status |
|---|-------|--------|
| 1 | Data loading + EDA — class distribution, sample images, imbalance quantification | ✅ Complete |
| 2 | Preprocessing + split — stratified split, normalization, justified augmentation strategy | 🔄 In progress |
| 3 | Baseline CNN — from-scratch model, per-class F1, confusion matrix, malignant recall | ⏳ Pending |
| 4 | Transfer learning — ResNet18 + EfficientNet-B0, comparison against baseline | ⏳ Pending |
| 5 | Grad-CAM — visualizations for correct and incorrect predictions, both models | ⏳ Pending |
| 6 | Final writeup — structured comparison: accuracy vs interpretability vs training cost | ⏳ Pending |

---

## Results

> Filled in as stages complete.

| Stage | Model | Accuracy | Macro F1 | Recall — mel | Recall — bcc | Recall — akiec |
|-------|-------|:--------:|:--------:|:------------:|:------------:|:--------------:|
| 3 | Baseline CNN | — | — | — | — | — |
| 4 | ResNet18 | — | — | — | — | — |
| 4 | EfficientNet-B0 | — | — | — | — | — |

---

## How to Run

> **Requirements:** A Google account (for Colab) and a Kaggle account (for the dataset API key).  
> Everything else is free. No paid tiers, no paid APIs.

### Step 1 — Open the notebook in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/exploration.ipynb)

Or manually: [colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub → `Dev252001/HAM10000`

### Step 2 — Clone the repo and add `src/` to the path

```python
!git clone https://github.com/Dev252001/HAM10000.git /content/ham10000-classifier
import sys
sys.path.insert(0, "/content/ham10000-classifier/src")
```

### Step 3 — Install dependencies

```python
!pip install -q -r /content/ham10000-classifier/requirements.txt
```

### Step 4 — Configure Kaggle credentials and download the dataset

Get your `kaggle.json` from [kaggle.com/settings](https://www.kaggle.com/settings) → **API Tokens** tab → **Create Legacy API Key**.

> ⚠️ Use **"Create Legacy API Key"** (not "Generate New Token") — the Kaggle CLI requires the legacy `{"username":…,"key":…}` format.

Cell 3 of the notebook handles credentials and download automatically — it:
- Reads the uploaded file **directly from memory** (avoids Colab's duplicate-filename bug where `kaggle.json` becomes `kaggle (2).json`)
- Writes credentials to `/root/.kaggle/kaggle.json`
- Pulls the latest `src/` code before importing, so the correct dataset slug is always used
- Is **idempotent** — safe to re-run; skips download if data already exists

### Step 5 — Run all cells

Runtime → Run all. Expected output in Stage 1:
- `Total rows: 10015`
- `Any missing filepaths: 0`
- Class distribution bar chart saved to `outputs/figures/`
- Sample image grid (5 per class) saved to `outputs/figures/`

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.6.0 |
| torchvision | 0.21.0 |
| scikit-learn | 1.4.2 |
| grad-cam | 1.5.0 |
| pandas | 2.2.2 |
| matplotlib | 3.9.0 |
| Platform | Google Colab free tier (T4 GPU) |

Full pinned versions in [`requirements.txt`](requirements.txt).

---

## License

- **Dataset:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — HAM10000 / ISIC Archive
- **Code:** [MIT](https://opensource.org/licenses/MIT)
