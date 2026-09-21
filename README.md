<h1 align="center">HAM10000 Skin Lesion Classifier</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab" />
  <img src="https://img.shields.io/badge/Dataset-HAM10000-green?style=flat-square" />
</p>

Dermoscopic image classifier trained on HAM10000 — 10,015 images across 7 skin lesion classes. Covers a from-scratch baseline CNN, transfer learning (ResNet18, EfficientNet-B0), and Grad-CAM visualisations, with class imbalance handled throughout.

---

## Results

| Model | Accuracy | Macro F1 | Recall — mel | Recall — bcc | Recall — akiec |
|-------|:--------:|:--------:|:------------:|:------------:|:--------------:|
| Baseline CNN | 62.81% | 0.4475 | 0.6287 | 0.3377 ⚠️ | 0.6531 |
| ResNet18 | 73.72% | 0.6679 | 0.6766 | 0.7792 | 0.7551 |
| **EfficientNet-B0** | **79.44%** | **0.7715** | **0.8323** | **0.8182** | **0.8571** |

Accuracy alone is misleading here — a model that always predicts `nv` gets ~67%. Macro-F1 and malignant-class recall (mel, bcc, akiec) are the real metrics.

Full discussion: [outputs/RESULTS.md](outputs/RESULTS.md)

---

## Dataset

**Source:** [Kaggle — Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
**Size:** 10,015 images · 7 classes · heavily imbalanced

| Code | Full Name | Malignant | Count | % |
|------|-----------|:---------:|------:|--:|
| `nv` | Melanocytic Nevi | — | ~6705 | 66.9% |
| `mel` | Melanoma | ⚠️ | ~1113 | 11.1% |
| `bkl` | Benign Keratosis | — | ~1099 | 11.0% |
| `bcc` | Basal Cell Carcinoma | ⚠️ | ~514 | 5.1% |
| `akiec` | Actinic Keratosis / IEC | ⚠️ | ~327 | 3.3% |
| `vasc` | Vascular Lesion | — | ~142 | 1.4% |
| `df` | Dermatofibroma | — | ~115 | 1.1% |

Imbalance ratio: `nv` is ~58× larger than `df`.

---

## Project Structure

```
HAM10000/
├── notebooks/
│   ├── exploration.ipynb          # Stage 1 — EDA
│   ├── 02_preprocessing.ipynb    # Stage 2 — split, transforms, class weights
│   ├── 03_baseline_cnn.ipynb     # Stage 3 — baseline CNN
│   ├── 04_transfer_learning.ipynb # Stage 4 — ResNet18 + EfficientNet-B0
│   └── 05_gradcam.ipynb           # Stage 5 — Grad-CAM
├── outputs/
│   ├── RESULTS.md
│   └── figures/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── models/
│       ├── baseline_cnn.py
│       └── transfer_models.py
└── requirements.txt
```

---

## How to Run

Requirements: Google account (Colab) + Kaggle account (for `kaggle.json`).

### Open a notebook in Colab

| Stage | Link |
|-------|------|
| Stage 1 — EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/exploration.ipynb) |
| Stage 2 — Preprocessing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/02_preprocessing.ipynb) |
| Stage 3 — Baseline CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/03_baseline_cnn.ipynb) |
| Stage 4 — Transfer Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/04_transfer_learning.ipynb) |
| Stage 5 — Grad-CAM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dev252001/HAM10000/blob/main/notebooks/05_gradcam.ipynb) |

1. Runtime → Change runtime type → **T4 GPU**
2. Runtime → **Run all**

On first run, you'll be prompted to upload `kaggle.json`. The dataset is then saved to Google Drive so future runs skip the download.

### Get your `kaggle.json`

[kaggle.com/settings](https://www.kaggle.com/settings) → API Tokens → **Create Legacy API Key**

---

## Notes

- Split: 70/15/15 stratified on `dx`. The rarest class (`df`, 115 images) needs ~17 images per eval set for stable recall — 80/10/10 would leave only ~11.
- Class weights computed from training set only (`w_c = N / (C × n_c)`), passed to `CrossEntropyLoss`.
- Image-level split only — images of the same lesion can appear in different splits. Test performance is an optimistic upper bound on lesion-level generalisation.

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | pre-installed on Colab (2.x + CUDA) |
| numpy | ≥ 2.0 |
| scikit-learn | ≥ 1.5.0 |
| pandas | ≥ 2.2.2 |
| Pillow | ≥ 10.4.0 |
| grad-cam | ≥ 1.5.0 |

`torch` and `torchvision` are not pinned in `requirements.txt` — Colab pre-installs them with CUDA support.

---

## License

- Dataset: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — HAM10000 / ISIC Archive
- Code: [MIT](https://opensource.org/licenses/MIT)
