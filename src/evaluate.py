"""
src/evaluate.py
───────────────
Model evaluation utilities:
  - Overall accuracy
  - Per-class F1 score
  - Confusion matrix (with plot)
  - Recall on malignant classes specifically (mel, bcc, akiec)

Why not just use accuracy?
  HAM10000 is ~67% 'nv'. A model that always predicts 'nv' achieves 67%
  accuracy but zero recall on every malignant class — clinically catastrophic.
  Macro-F1 and malignant recall are the primary evaluation metrics here.

STAGE STATUS: stub — implemented in Stage 3.
"""


def evaluate_model(model, data_loader, device, class_names):
    """
    Run model on data_loader and return a dict containing:
      accuracy, per_class_f1, macro_f1, weighted_f1,
      malignant_recall (dict keyed by class code),
      confusion_matrix (numpy array)

    Implemented in Stage 3.
    """
    raise NotImplementedError("Implemented in Stage 3.")


def plot_confusion_matrix(cm, class_names, save_path: str = None):
    """Plot and optionally save a labelled confusion matrix heatmap."""
    raise NotImplementedError("Implemented in Stage 3.")
