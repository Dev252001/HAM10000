"""
src/evaluate.py
───────────────
Model evaluation utilities for HAM10000.

Why not just report accuracy?
  HAM10000 is ~67% 'nv'. A model that always predicts 'nv' achieves 67%
  accuracy but zero recall on every malignant class — clinically catastrophic.
  The metrics that matter here are:
    - Macro-F1 (treats all classes equally regardless of frequency)
    - Malignant-class recall (mel, bcc, akiec) — the cost of a false negative
      is a missed cancer diagnosis, which is far worse than a false positive.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from data_loader import CLASSES, LABEL_MAP, MALIGNANT_CLASSES


def evaluate_model(model, data_loader, device) -> dict:
    """
    Run model on data_loader and return a comprehensive metrics dict.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model. Must already be on `device`.
    data_loader : DataLoader
        Val or test DataLoader (no augmentation).
    device : torch.device

    Returns
    -------
    dict with keys:
        accuracy        : float
        macro_f1        : float
        weighted_f1     : float
        per_class_f1    : dict {class_code: f1_score}
        malignant_recall: dict {class_code: recall}  — mel, bcc, akiec only
        confusion_matrix: np.ndarray  shape (7, 7)
        report          : str         full sklearn classification_report
    """
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Overall metrics ───────────────────────────────────────────────────────
    accuracy    = accuracy_score(all_labels, all_preds)
    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm          = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    per_class_f1_arr = f1_score(all_labels, all_preds, average=None,
                                labels=list(range(len(CLASSES))), zero_division=0)
    per_class_f1 = {cls: per_class_f1_arr[i] for i, cls in enumerate(CLASSES)}

    # ── Malignant-class recall ─────────────────────────────────────────────────
    # For each malignant class: recall = TP / (TP + FN) = cm[i,i] / cm[i,:].sum()
    malignant_recall = {}
    for cls in MALIGNANT_CLASSES:
        idx = CLASSES.index(cls)
        row_sum = cm[idx].sum()
        malignant_recall[cls] = float(cm[idx, idx] / row_sum) if row_sum > 0 else 0.0

    # ── Full sklearn report ───────────────────────────────────────────────────
    target_names = [LABEL_MAP[c] for c in CLASSES]
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        zero_division=0,
    )

    return {
        "accuracy":         accuracy,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "per_class_f1":     per_class_f1,
        "malignant_recall": malignant_recall,
        "confusion_matrix": cm,
        "report":           report,
    }


def print_results(results: dict) -> None:
    """
    Print a clear, structured summary of evaluate_model() output.
    Malignant-class recall is printed prominently — it is the primary
    clinical metric.
    """
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Overall Accuracy : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  Macro F1         : {results['macro_f1']:.4f}")
    print(f"  Weighted F1      : {results['weighted_f1']:.4f}")
    print()

    print("── Malignant-class Recall (most clinically critical) ──")
    for cls, recall in results["malignant_recall"].items():
        flag = "⚠️  LOW" if recall < 0.5 else "✓"
        print(f"  {cls:<8} ({LABEL_MAP[cls]:<30}): {recall:.4f}  {flag}")
    print()

    print("── Per-class F1 ───────────────────────────────────────")
    for cls in CLASSES:
        f1  = results["per_class_f1"][cls]
        bar = "█" * int(f1 * 20)
        print(f"  {cls:<8} {LABEL_MAP[cls]:<30}: {f1:.4f}  {bar}")
    print()

    print("── Full Classification Report ─────────────────────────")
    print(results["report"])

    # ── Imbalance sanity check ────────────────────────────────────────────────
    if results["accuracy"] > 0.90 and results["macro_f1"] < 0.50:
        print("⚠️  WARNING: High accuracy but low macro-F1.")
        print("   The model may be predicting the majority class (nv) for most inputs.")
        print("   Check confusion matrix — verify class_weights are on the correct device.")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> None:
    """
    Plot a normalised confusion matrix heatmap and optionally save it.

    Parameters
    ----------
    cm : np.ndarray  shape (n_classes, n_classes)
        Raw (unnormalised) confusion matrix from evaluate_model().
    save_path : str, optional
        Full file path to save the figure (e.g. outputs/figures/confusion_matrix.png).
    """
    import os

    labels = [LABEL_MAP[c] for c in CLASSES]
    # Row-normalise so each cell shows recall (fraction of true class predicted as X)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title("Confusion Matrix (row-normalised = recall per class)", fontsize=11)

    # Annotate each cell with the raw count and normalised value
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
