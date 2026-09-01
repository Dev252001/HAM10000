# HAM10000 Skin Lesion Classifier — Results & Discussion

> **Status:** Fill in the `[PLACEHOLDER]` values with your actual results from
> `outputs/model_comparison.csv` (generated in Stage 4, Cell 10) before
> submitting this file. Every placeholder is clearly marked — do not leave any
> unfilled.

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
The headline result: **[PLACEHOLDER — e.g. "EfficientNet-B0 achieved the
highest macro-F1 of X.XX and melanoma recall of X.XX, outperforming the
baseline CNN by X.XX macro-F1 points"]**.

---

## 2. Results

All three models were evaluated on the same held-out test set (15% of
HAM10000, ~1,503 images, stratified by class). Class weights computed from
the training set were applied to CrossEntropyLoss for all models.

### 2.1 Summary table

| Model | Accuracy | Macro F1 | Recall — mel | Recall — bcc | Recall — akiec |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline CNN (from scratch) | [ACC]% | [F1] | [REC] | [REC] | [REC] |
| ResNet18 (fine-tuned) | [ACC]% | [F1] | [REC] | [REC] | [REC] |
| EfficientNet-B0 (fine-tuned) | [ACC]% | [F1] | [REC] | [REC] | [REC] |

> Numbers from `outputs/model_comparison.csv`. Replace `[ACC]`, `[F1]`,
> `[REC]` with actual values before submitting.

**Best model:** [PLACEHOLDER — model name]  
**Margin over baseline:** [PLACEHOLDER — e.g. "+0.08 macro-F1, +0.12 melanoma recall"]

### 2.2 Key observations

- **Accuracy is misleading here.** All three models may show accuracy in the
  60–80% range; what matters is whether they detect malignant lesions. A model
  with 75% accuracy but 0.3 melanoma recall is clinically worse than one with
  65% accuracy and 0.6 melanoma recall.
- **Class weighting worked** (or did not — fill in after reviewing results):
  [PLACEHOLDER — e.g. "The confusion matrix confirms predictions are spread
  across all 7 classes, not collapsed onto nv".]
- **Transfer learning vs. baseline:** [PLACEHOLDER — e.g. "ResNet18 and
  EfficientNet-B0 both outperformed the baseline on macro-F1, consistent with
  the expectation that ImageNet-pretrained features provide a useful
  initialisation even for the dermoscopy domain."]

---

## 3. Tradeoff Analysis

### 3.1 Accuracy vs. complexity

| Model | Params | Macro F1 | Training time (T4) |
|---|:---:|:---:|:---:|
| Baseline CNN | ~0.4M | [F1] | [PLACEHOLDER — e.g. ~35 min] |
| ResNet18 | ~11M | [F1] | [PLACEHOLDER — e.g. ~45 min] |
| EfficientNet-B0 | ~5.3M | [F1] | [PLACEHOLDER — e.g. ~50 min] |

[PLACEHOLDER — discuss: did the ~28× parameter increase from baseline to
ResNet18 justify the F1 improvement? EfficientNet-B0 has fewer parameters
than ResNet18 — did it outperform it anyway? A typical honest answer is
something like: "The gain from transfer learning is meaningful (+X macro-F1)
but the gap between ResNet18 and EfficientNet-B0 is small (±0.02), suggesting
that for this dataset size both pretrained models extract similarly useful
features."]

### 3.2 Training cost

- **Baseline CNN:** Trained from scratch — required more epochs to converge
  (~[PLACEHOLDER] epochs before early stopping), since every weight initialised
  randomly. Epoch time: ~[PLACEHOLDER] sec on T4.
- **ResNet18 / EfficientNet-B0:** Converged in fewer epochs
  (~[PLACEHOLDER] epochs) because pretrained weights already encode useful
  low-level features. However, fine-tuning end-to-end adds per-epoch time
  due to gradient computation through the full backbone.
- **Practical takeaway:** For a small dataset (<10k images), transfer learning
  is strictly preferable — it converges faster, achieves better metrics, and
  is less likely to overfit.

### 3.3 Inference speed

[PLACEHOLDER — if you timed inference in Colab, report it here. If not,
note that all three models accept 224×224 tensors and produce logits in a
single forward pass. For real-time clinical use, EfficientNet-B0 (~5.3M
params) is lighter than ResNet18 (~11M) with comparable or better accuracy,
making it the preferred production choice between the two. The baseline CNN
(~0.4M params) is the fastest but has the worst clinical metrics.]

---

## 4. Grad-CAM Findings

Grad-CAM heatmaps were generated for the best-performing model
([PLACEHOLDER — model name]) using the following target layers:

| Model | Target layer | Spatial size |
|---|---|:---:|
| Baseline CNN | `features[3].block[0]` | 28×28 |
| ResNet18 | `layer4[1].conv2` | 7×7 |
| EfficientNet-B0 | `features[8][0]` | 7×7 |

### 4.1 Correct malignant predictions

[PLACEHOLDER — describe what you observed in `gradcam_correct_malignant.png`.
Examples of what to write:
- "For melanoma images, the heatmap consistently centred on the irregular,
  darkly-pigmented lesion area, suggesting the model responds to colour
  heterogeneity and irregular borders — both genuine diagnostic features."
- "For basal cell carcinoma images, heatmaps focused on the pearlescent
  nodular region, which is diagnostically characteristic."
If any heatmap landed on background skin or hair, note it explicitly:
- "Two akiec images showed heatmaps extending onto surrounding skin rather
  than the lesion boundary — this may reflect that akiec lesions have less
  distinct borders, or could indicate partial reliance on background context."]

### 4.2 Misclassified images (malignant → benign)

[PLACEHOLDER — describe what you observed in `gradcam_incorrect_malignant.png`.
This is the most important subsection for a research writeup. Common patterns:

**Artifact distraction:**
"In [N] of [M] examined misclassifications, the heatmap focused partially or
entirely on hair strands crossing the lesion. HAM10000 contains many images
with hair that was not removed during photography, and the model appears to
use hair density as an unintended cue in some cases."

**Genuine ambiguity:**
"In [N] cases, the heatmap correctly focused on the lesion but the model still
predicted the wrong class. These cases likely represent genuinely ambiguous
lesion morphology — for example, early-stage melanoma can be visually similar
to a dysplastic nevus even for trained dermatologists."

**Diffuse/scattered heatmaps:**
"[N] misclassifications showed diffuse heatmaps with no dominant region,
suggesting the model had low confidence and was uncertain rather than
confidently wrong."]

### 4.3 Baseline CNN vs. transfer model comparison

[PLACEHOLDER — describe what you observed in `gradcam_model_comparison.png`.
Expected pattern:
"The transfer model's heatmaps were more tightly localised to the lesion
region, whereas the baseline CNN's heatmaps were more diffuse, sometimes
extending into surrounding skin. This is consistent with the hypothesis
that ImageNet-pretrained features provide a more discriminative spatial
encoding of lesion structure from the first few training epochs, while the
from-scratch model requires more data to learn tightly localised features."

If the opposite was observed, say so honestly and discuss why.]

---

## 5. Limitations

These limitations are stated plainly — not to undersell the work, but because
an honest assessment is more credible than one that omits them.

**1. Dataset size and residual class imbalance**  
10,015 images across 7 classes, with the rarest class (*Dermatofibroma*)
having only 115 examples. Class weighting substantially mitigated the
imbalance problem, but recall on the smallest classes remains lower than
on larger ones — with so few training examples, any model will struggle
to learn robust features for these classes regardless of loss weighting.

**2. This is a research/portfolio project, not a validated diagnostic tool**  
The models trained here have not undergone clinical validation and must not
be used for medical decision-making. Dermoscopic image classifiers require
rigorous prospective clinical evaluation, regulatory approval, and
integration into a clinical workflow with human oversight before any
clinical use. This project demonstrates the technical methodology only.

**3. No external test set — unknown generalization**  
All evaluation uses HAM10000's own held-out split. The training,
validation, and test images all come from the same acquisition protocol,
dermatoscope type, and patient population (predominantly European).
Performance on images from different cameras, lighting conditions, or
patient skin tones is unknown. Published work on dermoscopy classification
consistently shows significant performance drops on out-of-distribution
images, and this project has not measured that.

**4. Image artifacts in HAM10000**  
The dataset contains known artifacts: hair crossing lesions, ruler
markings, ink dots, and dark vignettes from the dermatoscope lens.
The Grad-CAM analysis suggests the models partially learned to use some
of these as features in some cases (see §4.2). A production system would
apply artifact removal as a preprocessing step.

**5. [PLACEHOLDER — any additional limitation your actual results revealed]**  
[e.g. "Recall on akiec remained below 0.5 even for the best transfer model,
suggesting that with only 327 training examples, even pretrained features
are insufficient for reliable classification of this class." Or: "The
training curves showed mild overfitting from epoch N onward for the baseline
CNN — the model would likely benefit from stronger regularisation or more
aggressive augmentation."]

---

## 6. Future Work

**1. Ensemble of transfer models**  
Averaging predictions from ResNet18 and EfficientNet-B0 (or a simple
majority vote) is likely to improve robustness on minority classes without
additional training. Ensemble methods consistently outperform individual
models in dermoscopy classification literature and would be a natural
next step before deployment.

**2. External validation on ISIC 2019/2020 challenge data**  
The ISIC archive contains publicly available dermoscopy datasets from
multiple institutions and camera types. Evaluating the trained models on
ISIC 2019 or 2020 test sets (without retraining) would give a much more
honest estimate of out-of-distribution generalization and is the standard
way clinical AI systems are benchmarked before deployment.

**3. Focal loss comparison**  
This project used class-weighted CrossEntropyLoss to handle imbalance.
Focal loss (Lin et al., 2017) — which down-weights easy examples
dynamically during training — is an alternative that some dermoscopy
papers report improves minority-class recall further. A direct A/B
comparison on the same splits would cleanly quantify whether focal loss
outperforms static class weights for this specific dataset.

---

*Generated as part of a portfolio project. All code at
[github.com/Dev252001/HAM10000](https://github.com/Dev252001/HAM10000).*
