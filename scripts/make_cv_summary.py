"""
scripts/make_cv_summary.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Reads outputs/metrics/*.json (primary lesion-split results only) and writes
outputs/RESULTS_SUMMARY.md — a short CV-ready summary containing only numbers
that are backed by committed JSON files.

Rows whose JSON is missing are omitted entirely (not shown as "—"), so the
output file contains only verified numbers.

Usage (from repo root):
    python scripts/make_cv_summary.py

Output: outputs/RESULTS_SUMMARY.md
"""

import json
import os
import sys
from datetime import datetime

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(REPO_ROOT, "outputs", "metrics")
OUT_PATH    = os.path.join(REPO_ROOT, "outputs", "RESULTS_SUMMARY.md")

PRIMARY = [
    ("baseline_cnn",    "lesion", 42, "Baseline CNN (from scratch)"),
    ("resnet18",        "lesion", 42, "ResNet18 (fine-tuned, ImageNet)"),
    ("efficientnet_b0", "lesion", 42, "EfficientNet-B0 (fine-tuned, ImageNet)"),
]


def load(model_name, split_mode, seed):
    path = os.path.join(METRICS_DIR, f"{model_name}_{split_mode}_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    rows = []
    for model_name, split_mode, seed, label in PRIMARY:
        data = load(model_name, split_mode, seed)
        if data is None:
            print(f"  skipping (no JSON): {model_name}_{split_mode}_seed{seed}",
                  file=sys.stderr)
            continue
        rows.append((label, data))

    if not rows:
        print("No verified metrics found. Run the Colab notebooks and commit the JSON files.",
              file=sys.stderr)
        sys.exit(1)

    lines = []
    lines.append("# HAM10000 — Verified Results Summary")
    lines.append("")
    lines.append(
        f"_Generated {datetime.utcnow().strftime('%Y-%m-%d')} from "
        f"`outputs/metrics/*.json`. All numbers are from committed files._"
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append("- HAM10000 dermoscopy dataset — 10,015 images, 7 classes")
    lines.append("- Class imbalance ~58:1 (image-level nv/df ratio from published counts)")
    lines.append("- Split: lesion-grouped 70/15/15, stratified on class, seed 42")
    lines.append("- No lesion appears in more than one partition")
    lines.append("")
    lines.append("## Results (lesion-level split, seed 42)")
    lines.append("")
    lines.append(
        "| Model | Accuracy | Macro F1 | Recall — mel | Recall — bcc | "
        "Recall — akiec | Params |"
    )
    lines.append(
        "|-------|:--------:|:--------:|:------------:|:------------:|"
        ":--------------:|-------:|"
    )
    for label, d in rows:
        acc    = f"{d['accuracy']*100:.2f}%"
        f1     = f"{d['macro_f1']:.4f}"
        mel    = f"{d['malignant_recall'].get('mel',  float('nan')):.4f}"
        bcc    = f"{d['malignant_recall'].get('bcc',  float('nan')):.4f}"
        akiec  = f"{d['malignant_recall'].get('akiec', float('nan')):.4f}"
        params = f"{d['n_params']:,}"
        lines.append(
            f"| {label} | {acc} | {f1} | {mel} | {bcc} | {akiec} | {params} |"
        )

    lines.append("")
    lines.append("## Evaluation protocol")
    lines.append("")
    lines.append(
        "- All three models trained and evaluated on the **same** lesion-grouped split"
    )
    lines.append(
        "- Class-weighted CrossEntropyLoss (`w_c = N / (C × n_c)`), weights from training "
        "partition only"
    )
    lines.append(
        "- Checkpoint selected by lowest validation loss; test set evaluated **once** per model"
    )
    lines.append(
        "- Primary metric: macro-F1 (treats all 7 classes equally, not skewed by majority class)"
    )
    lines.append(
        "- Secondary: recall on mel, bcc, akiec (malignant classes — false negatives are "
        "the clinically costly errors)"
    )

    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    for model_name, split_mode, seed, label in PRIMARY:
        data = load(model_name, split_mode, seed)
        if data is None:
            continue
        commit = data.get("git_commit", "unknown")
        mhash  = data.get("manifest_hash") or "n/a"
        lines.append(f"- **{label}**: git `{commit[:12]}`, manifest SHA-256 `{mhash[:16]}…`")

    lines.append("")
    lines.append(
        "_This file is auto-generated. Do not edit by hand. "
        "Re-run `python scripts/make_cv_summary.py` after committing new metrics._"
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Written → {OUT_PATH}")
    print(f"Models included: {[r[0] for r in rows]}")


if __name__ == "__main__":
    main()
