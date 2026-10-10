# HAM10000 Skin Lesion Classifier — Results & Discussion

---

## 1. Project Summary

This project builds and compares three deep learning classifiers for
dermoscopic skin lesion classification on the HAM10000 dataset (10,015 images,
7 classes). The clinical motivation is clear: the three malignant classes
(melanoma, basal cell carcinoma, actinic keratosis) are also the minority
classes, meaning a naive classifier that always predicts the majority class
(*melanocytic nevi*) achieves ~67% accuracy while missing every cancer.
Accuracy is therefore a dangerously misleading metric; this project reports
**macro-F1 and recall on malignant classes** as the primary results throughout.

A from-scratch baseline CNN is compared against fine-tuned ResNet18 and
EfficientNet-B0 under identical conditions (same lesion-level splits,
same class-weighted loss, same training loop), and Grad-CAM heatmaps are
used to validate that the best model looks at lesion structure rather than
image artifacts.
The headline result: **EfficientNet-B0 achieved the highest accuracy (80.34%)
and macro-F1 (0.6440), outperforming ResNet18 by 0.037 macro-F1 points.**

> **Split note:** All three models were evaluated on a **lesion-level split**
> (via `make_lesion_splits()` in `src/preprocessing.py`). All images sharing
> a `lesion_id` are kept in the same partition — no lesion crosses the
> train/test boundary. These are honest held-out metrics.

---

## 2. Results

All models were evaluated on the same held-out test set (15% of HAM10000,
~1,503 images, stratified by class). Class weights computed from the
training set were applied to CrossEntropyLoss for all models.

### 2.1 Summary table

All models trained and evaluated on the **lesion-level split**
(7018 train / 1507 val / 1490 test images, 7470 unique lesions).

| Model | Accuracy | Macro F1 | Weighted F1 | Recall — mel | Recall — bcc | Recall — akiec |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline CNN (from scratch) | 63.29% | 0.4441 | 0.6749 | 0.6226 | 0.3600 ⚠️ | 0.4444 ⚠️ |
| ResNet18 (fine-tuned) | 74.23% | 0.6073 | 0.7592 | 0.4717 ⚠️ | **0.8267** | 0.6222 |
| **EfficientNet-B0 (fine-tuned)** | **80.34%** | **0.6440** | **0.8067** | 0.4906 ⚠️ | 0.6533 | **0.6667** |

> Run `python scripts/results_table.py` to regenerate this table from `outputs/results/*.json` after a Colab run.

**Best model by accuracy and macro-F1: EfficientNet-B0 (fine-tuned)**
- Outperforms baseline CNN by **+17.1 pp accuracy**, **+0.200 macro-F1**
- Outperforms ResNet18 by **+6.1 pp accuracy**, **+0.037 macro-F1**
- BCC recall: ResNet18 leads at **0.8267** — highest malignant recall of all models
- Melanoma recall: **baseline CNN leads** at 0.6226 — both transfer models lower (0.49)

### 2.2 Key observations

**Accuracy is misleading.** ResNet18 shows 74.23% accuracy — close to the
~67% majority-class baseline — yet its macro-F1 of 0.6073 confirms it is
learning across classes, not collapsing to *nv*. EfficientNet-B0's 80.34%
accuracy with macro-F1 of 0.6440 is the strongest overall result.

**Class weighting — a deliberate preventive decision, not a reactive fix.**
HAM10000 is 67% `nv` (melanocytic nevi). Without class weighting,
CrossEntropyLoss treats every misclassification equally, so the loss
gradient steers the model toward predicting `nv` for everything — a
well-documented failure mode on heavily imbalanced datasets. Rather than
run an unweighted model and observe the collapse, inverse-frequency
weights (`w_c = N / (C × n_c)`) were computed from the training set and
applied to the loss from the first training run of every model. All three
models show genuine multi-class learning as a result.

**EfficientNet-B0 outperforms ResNet18 on overall metrics despite fewer
parameters** (~4.0M vs ~11.2M). This is consistent with EfficientNet's
compound-scaling design — it allocates capacity more efficiently than a
simple residual network at this scale.

**Melanoma recall is the most important single number — and it is the
weakest result for transfer models.** The baseline CNN achieves 0.6226
melanoma recall; both transfer models score lower (0.47–0.49). This is
a real finding, not a bug. The lesion-level split makes the test set
harder for melanoma specifically because duplicate melanoma images that
leaked between splits under the image-level split are now correctly
separated. The transfer models appear to trade melanoma sensitivity for
better overall accuracy — a clinically unfavourable tradeoff that would
need to be addressed before any deployment consideration.

**BCC recall is where transfer learning helps most.** Baseline: 0.3600,
ResNet18: 0.8267 (+0.467), EfficientNet-B0: 0.6533 (+0.293). Pretrained
features enable the model to learn discriminative patterns for BCC from
~366 training examples — a from-scratch model cannot.

---

## 3. Tradeoff Analysis

### 3.1 Accuracy vs. complexity

| Model | Params | Macro F1 | Mel recall | Notes |
|---|:---:|:---:|:---:|---|
| Baseline CNN | ~0.4M | 0.4441 | **0.6226** | Trained from scratch — lowest overall, highest mel recall |
| ResNet18 | ~11.2M | 0.6073 | 0.4717 ⚠️ | Largest model, best BCC recall (0.8267) |
| EfficientNet-B0 | ~4.0M | **0.6440** | 0.4906 ⚠️ | Best accuracy + macro-F1, fewest params |

EfficientNet-B0 achieves the best overall metrics with the fewest parameters
of the three models (4.0M vs 11.2M for ResNet18). For this dataset size
(~7k training images), architectural efficiency matters more than raw
parameter count. The notable finding is that neither transfer model exceeds
the baseline CNN on melanoma recall — a clinically important gap that
warrants further investigation (focal loss, class-conditional augmentation,
or ensemble methods).

### 3.2 Training cost

Both transfer models were trained with a two-speed Adam optimizer:
backbone `lr = 1e-4` (protecting pretrained features) and head `lr = 1e-3`
(full-speed updates for the new classifier). Both used early stopping with
patience=7 and a ReduceLROnPlateau scheduler (patience=3, factor=0.5).

Transfer models converge faster than a from-scratch CNN because the backbone
starts with ImageNet-pretrained features — useful low-level detectors do not
need to be relearned from scratch. This is a practical advantage on Colab
free tier (~4 hour GPU sessions): transfer learning is more likely to complete
training within a single session.

### 3.3 Inference speed

All three models accept 224×224 tensors and produce logits in a single
forward pass. For real-world deployment:
- **EfficientNet-B0** (~5.3M params) is the recommended choice — best metrics
  and lighter than ResNet18
- **Baseline CNN** (~0.4M params) would be fastest but has the weakest
  clinical metrics
- **ResNet18** (~11M params) is the heaviest with the worst transfer-model metrics

For a clinical screening tool where speed matters, EfficientNet-B0 offers
the best accuracy/speed/parameter tradeoff of the three architectures tested.

---

## 4. Grad-CAM Findings

Grad-CAM heatmaps were generated using the pytorch-grad-cam library
with the following target layers:

| Model | Target layer | Spatial size | Why |
|---|---|:---:|---|
| Baseline CNN | `features[3].block[0]` | 28×28 | Last conv before GAP — best resolution |
| ResNet18 | `layer4[1].conv2` | 7×7 | Last conv in last residual block — literature standard |
| EfficientNet-B0 | `features[8][0]` | 7×7 | Last MBConv block before global pooling |

### 4.1 Correct malignant predictions

EfficientNet-B0 heatmaps on all 6 correctly classified malignant images
landed on the lesion itself — not background skin, hair, or image edges.
Specific observations:

- **Actinic Keratosis (ISIC_0028816):** Heatmap tightly focused on the
  irregular, rough-textured lesion region in the upper-right of the image.
  The model responds to the distinctive surface texture change, which is
  a genuine diagnostic feature of akiec.
- **Actinic Keratosis (ISIC_0029268):** Heatmap covers the brownish
  pigmented cluster. Notably, the heatmap correctly ignores surrounding
  normal skin despite the lesion having diffuse borders.
- **Basal Cell Carcinoma (ISIC_0024665):** Heatmap centres cleanly on the
  dark nodular lesion with high intensity at its core — consistent with
  BCC's characteristic pearlescent/dark nodule appearance.
- **Basal Cell Carcinoma (ISIC_0027229):** Heatmap covers the entire
  irregular dark lesion despite hair strands crossing it — the model
  correctly attends to the lesion rather than the hair artifacts.
- **Melanoma (ISIC_0029271):** Heatmap tightly wraps the full lesion
  boundary, with highest intensity at the darkest pigmented region.
  Consistent with the ABCDE criteria (asymmetry, colour heterogeneity).
- **Melanoma (ISIC_0031189):** Heatmap precisely localises to the small
  dark lesion against pale surrounding skin — strong localisation despite
  the lesion being relatively small in the frame.

**Sanity check result: PASS.** All 6 heatmaps focus on lesion structure.
No heatmaps land on background skin or image artifacts.

### 4.2 Misclassified images (malignant → benign)

Two misclassifications were found in the sampled test images — both
clinically significant errors (malignant predicted as benign):

- **ISIC_0029885 — Melanoma predicted as Dermatofibroma ✗:**
  The heatmap focuses squarely on the lesion body, not on any artifact.
  This is a case of **genuine diagnostic ambiguity** — the lesion has a
  relatively uniform colour and smooth border, lacking the strong
  asymmetry and colour heterogeneity typical of melanoma. Even the
  heatmap shows the model attending to the right region; it simply
  misidentified the lesion type. This is the hardest error type to
  prevent — the model is looking at the correct features but the lesion
  morphology is ambiguous.

- **ISIC_0030659 — Basal Cell Carcinoma predicted as Actinic Keratosis ✗:**
  The heatmap focuses on a small, high-intensity central region of the
  lesion — the model identifies the lesion but confuses two morphologically
  similar malignant classes. BCC and akiec can appear visually similar in
  early stages. This is a malignant-to-malignant misclassification (both
  classes are clinically concerning), which is less dangerous than a
  malignant-to-benign error.

**Key finding:** Neither misclassification shows the heatmap landing on
hair, rulers, or background skin. Both trace to genuine lesion ambiguity,
not spurious shortcut learning — a positive finding for model reliability.

### 4.3 Baseline CNN vs. EfficientNet-B0 comparison

The side-by-side comparison across 3 images (akiec, bcc, melanoma) shows
a clear and consistent difference between the two models:

- **Actinic Keratosis (ISIC_0028816):** Baseline CNN heatmap is highly
  fragmented — scattered activation across multiple disconnected spots
  rather than a unified lesion region. EfficientNet-B0 produces a single
  coherent blob covering the full lesion. The baseline appears to respond
  to individual texture patches rather than the lesion as a whole.

- **Basal Cell Carcinoma (ISIC_0024665):** Baseline CNN activates in two
  separate regions — the lesion and an unrelated area. EfficientNet-B0
  produces a clean, tight circular heatmap precisely centred on the
  nodular lesion with almost no background activation.

- **Melanoma (ISIC_0029271):** Baseline CNN again shows fragmented,
  multi-spot activation spread across the lesion and surrounding area.
  EfficientNet-B0 shows a single compact, high-intensity region covering
  the lesion body.

**Pattern:** EfficientNet-B0 heatmaps are consistently more spatially
coherent and lesion-focused. The baseline CNN's fragmented heatmaps are
consistent with a from-scratch model that has learned to detect local
texture patches rather than integrated lesion features — it identifies
relevant regions but cannot synthesise them into a unified representation.
This directly reflects the baseline's lower macro-F1: it detects pieces
of the right signal but less reliably combines them into correct predictions.

---

## 5. Limitations

**1. Dataset size and residual class imbalance**
10,015 images across 7 classes, with the rarest class (*Dermatofibroma*)
having only 115 training examples (~80 after the 70/15/15 split). The
baseline CNN's BCC recall of 0.3377 directly reflects this: with only
~360 BCC training images, a from-scratch model cannot learn sufficiently
discriminative features. Class weighting mitigated majority-class collapse
but cannot compensate for fundamental data scarcity.

**2. This is a research/portfolio project, not a validated diagnostic tool**
The models trained here have not undergone clinical validation and must not
be used for medical decision-making. Dermoscopic image classifiers require
rigorous prospective clinical evaluation, regulatory approval, and integration
into a clinical workflow with human oversight before any clinical use.

**3. No external test set — unknown generalization**
All evaluation uses HAM10000's own held-out split. The training, validation,
and test images all come from the same acquisition protocol, dermatoscope type,
and patient population (predominantly European). Performance on images from
different cameras, lighting conditions, or patient demographics is unknown.
Published work on dermoscopy classification consistently shows performance drops
on out-of-distribution images, and this project has not measured that.

**4. Image artifacts in HAM10000**
The dataset contains known artifacts: hair crossing lesions, ruler markings,
ink dots, and dark vignettes from the dermatoscope lens. Grad-CAM analysis
confirmed the models do not primarily rely on these artifacts — both
misclassifications traced to genuine lesion ambiguity rather than artifact
distraction. However, artifact-driven errors cannot be ruled out on a
larger misclassification sample.

---

## 6. Future Work

**1. Ensemble of transfer models**
Averaging predictions from ResNet18 and EfficientNet-B0 is likely to improve
robustness on minority classes without additional training. Ensemble methods
consistently outperform individual models in dermoscopy classification
literature and are a natural next step before deployment.

**3. External validation on ISIC 2019/2020 data**
Evaluating the trained models on ISIC 2019 or 2020 test sets (without
retraining) would give a more honest estimate of out-of-distribution
generalization — the standard benchmark before clinical deployment.

**4. Focal loss comparison**
This project used class-weighted CrossEntropyLoss. Focal loss (Lin et al.,
2017) dynamically down-weights easy examples during training and is reported
to improve minority-class recall in some dermoscopy papers. A direct A/B
comparison on the same splits would cleanly quantify the difference.

---

## 7. Code Attribution

### Written from scratch

| File | What it is |
|---|---|
| `src/models/baseline_cnn.py` — `BaselineCNN`, `ConvBlock` | 4-block CNN architecture (filter sizes, GAP, dropout placement) — my design |
| `src/preprocessing.py` — `make_splits()` | Two-step stratified split with adjusted val size formula |
| `src/preprocessing.py` — `make_lesion_splits()` | Lesion-level split that prevents same-lesion leakage across train/test |
| `src/preprocessing.py` — `compute_class_weights()` | `N / (C × n_c)` formula, ordered by `CLASS_TO_IDX` |
| `src/preprocessing.py` — `get_transforms()` | Augmentation choices and exclusions (no aggressive crop, small hue jitter) |
| `src/train.py` — `train()` | Training loop: early stopping by val loss, checkpoint saving, two-speed param group wiring, LR scheduler integration |
| `src/models/transfer_models.py` — `build_resnet18()`, `build_efficientnet_b0()` | Head replacement + two-speed param group construction |
| `src/data_loader.py` — `HAM10000Dataset` | PyTorch `Dataset` wrapping split DataFrames |
| `src/evaluate.py` — `evaluate_model()`, `print_results()`, `plot_confusion_matrix()` | Metrics extraction, malignant-recall computation, confusion matrix plotting |

### Taken from libraries (used, not written)

| What | Source | My contribution |
|---|---|---|
| ResNet18 architecture + ImageNet weights | `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` | Replaced final `Linear(512→7)`; designed two-speed param groups |
| EfficientNet-B0 architecture + ImageNet weights | `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)` | Replaced `classifier[1]` with `Linear(1280→7)`; designed two-speed param groups |
| Grad-CAM heatmap computation | `pytorch-grad-cam` library (`GradCAM`, `show_cam_on_image`) | Chose target layers per model; wrote the visualisation grid; interpreted the output |
| Stratified splitting | `sklearn.model_selection.train_test_split` | Wrote the two-step wrapper with adjusted val size; chose stratification column |
| `CrossEntropyLoss`, `Adam`, `ReduceLROnPlateau` | PyTorch (`torch.nn`, `torch.optim`) | Configured all hyperparameters; wired class weights and param groups |

---

*Generated as part of a portfolio project. All code at
[github.com/Dev252001/HAM10000](https://github.com/Dev252001/HAM10000).*
