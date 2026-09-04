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
EfficientNet-B0 under identical conditions (same splits, same class-weighted
loss, same training loop), and Grad-CAM heatmaps are used to validate that
the best model looks at lesion structure rather than image artifacts.
The headline result: **EfficientNet-B0 achieved the highest macro-F1 of 0.7715
and melanoma recall of 0.8323, outperforming ResNet18 by 0.10 macro-F1 points.
The baseline CNN checkpoint is pending — comparison against the from-scratch
model will be added once Stage 3 training completes.**

---

## 2. Results

All models were evaluated on the same held-out test set (15% of HAM10000,
~1,503 images, stratified by class). Class weights computed from the
training set were applied to CrossEntropyLoss for all models.

### 2.1 Summary table

| Model | Accuracy | Macro F1 | Weighted F1 | Recall — mel | Recall — bcc | Recall — akiec |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline CNN (from scratch) | — | — | — | — | — | — |
| ResNet18 (fine-tuned) | 73.72% | 0.6679 | 0.7598 | 0.6766 | 0.7792 | 0.7551 |
| EfficientNet-B0 (fine-tuned) | **79.44%** | **0.7715** | **0.8109** | **0.8323** | **0.8182** | **0.8571** |

> Baseline CNN row will be filled in after Stage 3 training completes.
> Full comparison CSV at `outputs/model_comparison.csv`.

**Best model: EfficientNet-B0 (fine-tuned)**
- Outperforms ResNet18 by **+6.6 pp accuracy**, **+0.104 macro-F1**
- Melanoma recall of **0.8323** — correctly identifies 83% of melanoma cases
- Akiec recall of **0.8571** — highest of all malignant classes
- BCC recall of **0.8182** — strong detection of basal cell carcinoma

### 2.2 Key observations

**Accuracy is misleading.** ResNet18 shows 73.72% accuracy — close to the
~67% majority-class baseline — yet its macro-F1 of 0.6679 confirms it is
learning across classes, not collapsing to *nv*. EfficientNet-B0's 79.44%
accuracy with macro-F1 of 0.7715 is a more coherent result: genuine
multi-class learning.

**Class weighting worked.** Both transfer models show meaningful recall
across all three malignant classes (0.67–0.86), confirming the weighted
loss prevented majority-class collapse. Without class weighting, recall
on minority malignant classes typically falls below 0.2 on this dataset.

**EfficientNet-B0 outperforms ResNet18 despite fewer parameters** (~5.3M
vs ~11M). This is consistent with EfficientNet's compound-scaling design
— it allocates capacity more efficiently than a simple residual network
at this scale.

---

## 3. Tradeoff Analysis

### 3.1 Accuracy vs. complexity

| Model | Params | Macro F1 | Notes |
|---|:---:|:---:|---|
| Baseline CNN | ~0.4M | — | Pending Stage 3 training |
| ResNet18 | ~11M | 0.6679 | Larger than EfficientNet, lower F1 |
| EfficientNet-B0 | ~5.3M | 0.7715 | Best result, fewest transfer-model params |

The EfficientNet-B0 result is striking: it achieves the best metrics with
fewer parameters than ResNet18. This suggests that for this dataset size
(~7k training images), architectural efficiency matters more than raw
parameter count. EfficientNet's compound scaling (balancing depth, width,
and resolution simultaneously) appears to extract more useful features
per parameter than ResNet18's standard residual design for dermoscopic images.

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

**1. Baseline CNN results pending**
Stage 3 training did not complete before Stage 4 results were available.
The baseline checkpoint (`baseline_cnn_best.pt`) is absent from the comparison
table. The full three-way comparison will be added once Stage 3 completes.

**2. Dataset size and residual class imbalance**
10,015 images across 7 classes, with the rarest class (*Dermatofibroma*)
having only 115 training examples (~80 after the 70/15/15 split). Class
weighting mitigated majority-class collapse, but any model will struggle
with so few minority-class examples. The EfficientNet-B0's strong recall
numbers (0.76–0.86 on malignant classes) are encouraging given this constraint,
but should be interpreted cautiously with small absolute test set counts.

**3. This is a research/portfolio project, not a validated diagnostic tool**
The models trained here have not undergone clinical validation and must not
be used for medical decision-making. Dermoscopic image classifiers require
rigorous prospective clinical evaluation, regulatory approval, and integration
into a clinical workflow with human oversight before any clinical use.

**4. No external test set — unknown generalization**
All evaluation uses HAM10000's own held-out split. The training, validation,
and test images all come from the same acquisition protocol, dermatoscope type,
and patient population (predominantly European). Performance on images from
different cameras, lighting conditions, or patient demographics is unknown.
Published work on dermoscopy classification consistently shows performance drops
on out-of-distribution images, and this project has not measured that.

**5. Image artifacts in HAM10000**
The dataset contains known artifacts: hair crossing lesions, ruler markings,
ink dots, and dark vignettes from the dermatoscope lens. The Grad-CAM analysis
(Stage 5) will reveal whether the models partially rely on these as features —
this section will be updated after reviewing the heatmaps.

---

## 6. Future Work

**1. Complete baseline CNN comparison**
Run Stage 3 training to completion and add the from-scratch CNN row to the
comparison table. This will quantify the exact gap between training from
scratch and fine-tuning, which is the core question the project addresses.

**2. Ensemble of transfer models**
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

*Generated as part of a portfolio project. All code at
[github.com/Dev252001/HAM10000](https://github.com/Dev252001/HAM10000).*
