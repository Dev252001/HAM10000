# HAM10000 Skin Lesion Classifier

A portfolio-grade skin lesion classification project built on the
[HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
(10,015 dermoscopy images, 7 classes). The project progresses through four
modelling stages — baseline CNN, transfer learning, and Grad-CAM
interpretability — with explicit attention to class imbalance and clinical
relevance of metrics.

> **Status:** Stage 1 complete — EDA and class distribution analysis.

---

## Project Structure

```
ham10000-classifier/
├── README.md
├── requirements.txt
├── data/                    # ← not tracked by git (see .gitignore)
├── src/
│   ├── data_loader.py       # Kaggle download + dataset class
│   ├── preprocessing.py     # train/val/test split, augmentation pipeline
│   ├── models/
│   │   ├── baseline_cnn.py  # from-scratch CNN architecture
│   │   └── transfer_models.py  # ResNet18 / EfficientNet-B0 setup
│   ├── train.py             # shared training loop
│   ├── evaluate.py          # metrics: F1, confusion matrix, malignant recall
│   └── gradcam.py           # Grad-CAM visualizations
├── notebooks/
│   └── exploration.ipynb    # Stage 1 EDA (run this in Google Colab)
└── outputs/
    ├── figures/             # saved plots (tracked by git)
    └── models/              # saved checkpoints (not tracked by git)
```

---

## Dataset

**Source:** [Kaggle — Skin Lesion Analysis Toward Melanoma Detection](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)

| Code   | Full Name                     | Malignant | Count  | %     |
|--------|-------------------------------|-----------|--------|-------|
| nv     | Melanocytic Nevi              | No        | ~6705  | 66.9% |
| mel    | Melanoma                      | **Yes**   | ~1113  | 11.1% |
| bkl    | Benign Keratosis              | No        | ~1099  | 11.0% |
| bcc    | Basal Cell Carcinoma          | **Yes**   | ~514   | 5.1%  |
| akiec  | Actinic Keratosis / IEC       | **Yes**   | ~327   | 3.3%  |
| vasc   | Vascular Lesion               | No        | ~142   | 1.4%  |
| df     | Dermatofibroma                | No        | ~115   | 1.1%  |

**Imbalance ratio (nv : vasc) ≈ 42:1.** Accuracy alone is a misleading
metric — a model predicting "nv" for everything scores ~67%. We always
report macro-F1 and recall on the three malignant classes alongside accuracy.

---

## How to Run (Google Colab)

### 1. Clone the repo into your Colab session

```python
!git clone https://github.com/Dev252001/HAM10000.git /content/ham10000-classifier
import sys
sys.path.insert(0, "/content/ham10000-classifier/src")
```

### 2. Install dependencies

```python
!pip install -q -r /content/ham10000-classifier/requirements.txt
```

### 3. Set up Kaggle credentials and download the dataset

```python
from google.colab import files
files.upload()   # upload kaggle.json
import os, shutil
os.makedirs("/root/.kaggle", exist_ok=True)
shutil.copy("/content/kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)

!kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection \
    -p /content/ham10000-classifier/data --unzip
```

### 4. Open the notebook

Open `notebooks/exploration.ipynb` in Colab and run all cells.

---

## Results

> Results will be filled in as each stage completes.

| Stage | Model | Accuracy | Macro F1 | Mal. Recall (mel/bcc/akiec) |
|-------|-------|----------|----------|-----------------------------|
| 3     | Baseline CNN | — | — | — |
| 4     | ResNet18 | — | — | — |
| 4     | EfficientNet-B0 | — | — | — |

---

## Environment

- Python 3.10
- PyTorch 2.3.0 + torchvision 0.18.0
- Google Colab free tier (T4 GPU)
- See `requirements.txt` for full pinned dependencies

---

## License

Dataset: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (HAM10000 / ISIC).
Code: MIT.
