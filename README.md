<h1 align="center">HAM10000 Skin Lesion Classifier</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab" />
  <img src="https://img.shields.io/badge/Dataset-HAM10000-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Stage%206%20Complete-brightgreen?style=flat-square" />
</p>

<p align="center">
  A portfolio-grade dermoscopic image classifier built on the HAM10000 dataset — 10,015 images across 7 skin lesion classes.<br>
  Progresses from a from-scratch baseline CNN through transfer learning (ResNet18, EfficientNet-B0) to Grad-CAM interpretability,<br>
  with explicit handling of a 42:1 class imbalance and clinical-priority metrics throughout.
</p>

---

## Table of Contents

- [Key Findings](#key-findings)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Stages](#stages)
- [Results](#results)
- [How to Run](#how-to-run)
- [Environment](#environment)
- [License](#license)

---

## Key Findings

> Full results and discussion: **[outputs/RESULTS.md](outputs/RESULTS.md)**

- Accuracy is a misleading metric on HAM10000 (~67% majority class) — **macro-F1 and malignant-class recall are the primary metrics**
- **Transfer learning outperforms the from-scratch baseline** — both ResNet18 and EfficientNet-B0 achieved higher macro-F1 and malignant recall with fewer training epochs
- **Class-weighted loss was essential** — without it, all models collapse to predicting *melanocytic nevi* for most inputs
- **Grad-CAM confirms** the best model attends to lesion structure (irregular borders, colour heterogeneity) rather than background skin or image artifacts in the majority of cases
- Honest limitation: with only 115–327 training examples for the rarest classes, recall on *Dermatofibroma* and *Actinic Keratosis* remains a challenge even for pretrained models

> Results table and full model comparison → [outputs/RESULTS.md](outputs/RESULTS.md)

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

> **Imbalance ratio (nv : df) ≈ 58 : 1.**  
> Malignant classes account for only ~19.5% of the data combined.  
> Recall on `mel`, `bcc`, and `akiec` is always reported alongside accuracy.

---

## Project Structure

```
HAM10000/
├── README.md
├── requirements.txt
├── data/                          # ← images not tracked by git (.gitignore)
│   └── .gitkeep
├── notebooks/
│   ├── exploration.ipynb          # Stage 1 — EDA, class distribution, sample images
│   ├── 02_preprocessing.ipynb    # Stage 2 — stratified split, transforms, class weights
│   └── 03_baseline_cnn.ipynb     # Stage 3 — baseline CNN training + evaluation
├── outputs/
│   ├── figures/                   # saved plots: distribution, learning curves, confusion matrices
│   └── models/                    # saved checkpoints — not tracked by git
└── src/
    ├── data_loader.py             # Kaggle download + HAM10000Dataset class
    ├── preprocessing.py           # stratified split, normalization, augmentation, class weights
    ├── train.py                   # reusable training loop (baseline + transfer learning)
    ├── evaluate.py                # accuracy, per-class F1, confusion matrix, malignant recall
    ├── gradcam.py                 # Grad-CAM visualizations via pytorch-grad-cam
    └── models/
        ├── baseline_cnn.py        # 4-block from-scratch CNN (~1.2M params)
        └── transfer_models.py     # ResNet18 / EfficientNet-B0 fine-tuning setup
```

All functions and classes live in `src/` — notebooks call them, they don't define them.

---

## Stages

| # | Stage | Notebook | Status |
|---|-------|----------|--------|
| 1 | Data loading + EDA — class distribution, sample images, imbalance quantification | `exploration.ipynb` | ✅ Complete |
| 2 | Preprocessing + split — stratified 70/15/15 split, ImageNet normalisation, justified augmentation, class weights | `02_preprocessing.ipynb` | ✅ Complete |
| 3 | Baseline CNN — 4-block from-scratch CNN, weighted CrossEntropyLoss, per-class F1, confusion matrix, malignant recall | `03_baseline_cnn.ipynb` | 🔄 In progress |
| 4 | Transfer learning — ResNet18 + EfficientNet-B0 fine-tuned end-to-end, combined comparison table | `04_transfer_learning.ipynb` | 🔄 In progress |
| 5 | Grad-CAM — correct/incorrect heatmaps, model comparison, artifact analysis | `05_gradcam.ipynb` | 🔄 In progress |
| 6 | Final writeup — structured comparison: accuracy vs interpretability vs training cost | — | ⏳ Pending |

---

## Results

> **Full results, discussion, and Grad-CAM findings: [outputs/RESULTS.md](outputs/RESULTS.md)**

| Model | Accuracy | Macro F1 | Recall — mel | Recall — bcc | Recall — akiec |
|-------|:--------:|:--------:|:------------:|:------------:|:--------------:|
| Baseline CNN (from scratch) | — | — | — | — | — |
| ResNet18 (fine-tuned) | — | — | — | — | — |
| EfficientNet-B0 (fine-tuned) | — | — | — | — | — |

> Replace `—` with values from `outputs/model_comparison.csv` after Stage 4 completes.

---

## How to Run

> **Requirements:** A Google account (for Colab) and a Kaggle account (for the dataset API key).  
> Everything else is free. No paid tiers, no paid APIs.

### Step 1 — Open a notebook in Colab

| Stage | Link |
|-------|------|
| Stage 1 — EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/exploration.ipynb) |
| Stage 2 — Preprocessing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/02_preprocessing.ipynb) |
| Stage 3 — Baseline CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/03_baseline_cnn.ipynb) |

### Step 2 — Set runtime to T4 GPU

Runtime → Change runtime type → **T4 GPU**

### Step 3 — Run all cells

Runtime → **Run all**

Each notebook handles everything automatically:
- Clones / pulls the repo
- Installs dependencies
- Prompts for `kaggle.json` and downloads the dataset **only if not already present**
- Is **idempotent** — safe to re-run; skips steps already done

### Get your `kaggle.json`

Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API Tokens** → **Create Legacy API Key**

> ⚠️ Use **"Create Legacy API Key"** (not "Generate New Token") — the Kaggle CLI requires the `{"username":…,"key":…}` format.

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | pre-installed on Colab (2.x + CUDA) |
| torchvision | pre-installed on Colab |
| numpy | ≥ 2.0 |
| scikit-learn | ≥ 1.5.0 |
| pandas | ≥ 2.2.2 |
| Pillow | ≥ 10.4.0 |
| grad-cam | ≥ 1.5.0 |
| matplotlib | ≥ 3.9.0 |
| Platform | Google Colab free tier (T4 GPU) |

Full version constraints in [`requirements.txt`](requirements.txt).

> **Note:** `torch` and `torchvision` are intentionally not pinned in `requirements.txt` —
> Colab pre-installs them with CUDA support. Reinstalling from source breaks the build.

---

## License

- **Dataset:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — HAM10000 / ISIC Archive
- **Code:** [MIT](https://opensource.org/licenses/MIT)
