import hashlib
import json
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from data_loader import CLASSES, LABEL_MAP, MALIGNANT_CLASSES


def evaluate_model(model, data_loader, device) -> dict:
    """Run model on data_loader and return a metrics dict."""
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy    = accuracy_score(all_labels, all_preds)
    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm          = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))

    per_class_f1_arr = f1_score(all_labels, all_preds, average=None,
                                labels=list(range(len(CLASSES))), zero_division=0)
    per_class_f1 = {cls: per_class_f1_arr[i] for i, cls in enumerate(CLASSES)}

    malignant_recall = {}
    for cls in MALIGNANT_CLASSES:
        idx     = CLASSES.index(cls)
        row_sum = cm[idx].sum()
        malignant_recall[cls] = float(cm[idx, idx] / row_sum) if row_sum > 0 else 0.0

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
    """Print a summary of evaluate_model() output."""
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Overall Accuracy : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  Macro F1         : {results['macro_f1']:.4f}")
    print(f"  Weighted F1      : {results['weighted_f1']:.4f}")
    print()

    print("── Malignant-class Recall ─────────────────────────────")
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

    if results["accuracy"] > 0.90 and results["macro_f1"] < 0.50:
        print("⚠️  WARNING: High accuracy but low macro-F1.")
        print("   The model may be predicting the majority class (nv) for most inputs.")
        print("   Check confusion matrix — verify class_weights are on the correct device.")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> None:
    """Plot a normalised confusion matrix and optionally save it."""
    import os

    labels  = [LABEL_MAP[c] for c in CLASSES]
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


def save_metrics_json(
    results: dict,
    model_name: str,
    split_mode: str,
    seed: int,
    n_params: int,
    train_time_s: float,
    out_dir: str,
    manifest_path: str = None,
) -> str:
    """
    Serialise evaluation results to a JSON file and return the path.

    File is named:  <out_dir>/<model_name>_<split_mode>_seed<seed>.json

    Fields written
    --------------
    model_name, split_mode, seed
    accuracy, macro_f1, weighted_f1
    per_class_precision, per_class_recall, per_class_f1   (dict keyed by class code)
    malignant_recall                                       (dict keyed by class code)
    confusion_matrix                                       (list-of-lists, int)
    n_params, train_time_s
    git_commit                                             (HEAD SHA, or "unknown")
    manifest_hash                                          (SHA-256 of manifest CSV, or null)
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- git commit ---
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    # --- manifest hash ---
    mhash = None
    if manifest_path and os.path.exists(manifest_path):
        h = hashlib.sha256()
        with open(manifest_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        mhash = h.hexdigest()

    # --- per-class precision / recall (not stored in results dict already) ---
    cm = results["confusion_matrix"]
    all_labels_flat = []
    all_preds_flat  = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            all_labels_flat.extend([i] * int(cm[i, j]))
            all_preds_flat.extend([j] * int(cm[i, j]))

    prec_arr = precision_score(all_labels_flat, all_preds_flat,
                               average=None, labels=list(range(len(CLASSES))),
                               zero_division=0)
    rec_arr  = recall_score(all_labels_flat, all_preds_flat,
                            average=None, labels=list(range(len(CLASSES))),
                            zero_division=0)
    f1_arr   = [results["per_class_f1"][cls] for cls in CLASSES]

    per_class_precision = {cls: float(prec_arr[i]) for i, cls in enumerate(CLASSES)}
    per_class_recall    = {cls: float(rec_arr[i])  for i, cls in enumerate(CLASSES)}
    per_class_f1        = {cls: float(f1_arr[i])   for i, cls in enumerate(CLASSES)}

    payload = {
        "model_name":          model_name,
        "split_mode":          split_mode,
        "seed":                seed,
        "accuracy":            float(results["accuracy"]),
        "macro_f1":            float(results["macro_f1"]),
        "weighted_f1":         float(results["weighted_f1"]),
        "per_class_precision": per_class_precision,
        "per_class_recall":    per_class_recall,
        "per_class_f1":        per_class_f1,
        "malignant_recall":    {k: float(v) for k, v in results["malignant_recall"].items()},
        "confusion_matrix":    cm.tolist(),
        "n_params":            int(n_params),
        "train_time_s":        float(train_time_s),
        "git_commit":          git_commit,
        "manifest_hash":       mhash,
    }

    filename = f"{model_name}_{split_mode}_seed{seed}.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Metrics saved → {path}")
    return path
