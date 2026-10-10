"""
results_table.py — generate the README results table from outputs/results/*.json

Usage (from repo root):
    python scripts/results_table.py

Reads:
    outputs/results/baseline_cnn_results.json
    outputs/results/resnet18_results.json
    outputs/results/efficientnet_b0_results.json

Prints a Markdown table to stdout.  Paste it into README.md to update the
Results section.  Numbers are read from the JSON files produced by the
Colab training notebooks, so the table always matches the actual runs.
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "results")

MODELS = [
    ("baseline_cnn_results.json",     "Baseline CNN"),
    ("resnet18_results.json",         "ResNet18"),
    ("efficientnet_b0_results.json",  "EfficientNet-B0"),
]


def load(filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — run the corresponding Colab notebook first.",
              file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def fmt_f1(v: float) -> str:
    return f"{v:.4f}"


def recall_cell(data: dict, cls: str) -> str:
    v = data["malignant_recall"].get(cls, float("nan"))
    s = f"{v:.4f}"
    if v < 0.5:
        s += " ⚠️"
    return s


def main():
    rows = []
    for filename, display_name in MODELS:
        data = load(filename)
        if data is None:
            rows.append((display_name, "TBD", "TBD", "TBD", "TBD", "TBD"))
            continue
        rows.append((
            display_name,
            fmt_pct(data["accuracy"]),
            fmt_f1(data["macro_f1"]),
            recall_cell(data, "mel"),
            recall_cell(data, "bcc"),
            recall_cell(data, "akiec"),
        ))

    header = "| Model | Accuracy | Macro F1 | Recall — mel | Recall — bcc | Recall — akiec |"
    sep    = "|-------|:--------:|:--------:|:------------:|:------------:|:--------------:|"
    print(header)
    print(sep)
    for name, acc, mf1, mel, bcc, akiec in rows:
        print(f"| {name} | {acc} | {mf1} | {mel} | {bcc} | {akiec} |")

    print()
    print("# Source files:")
    for filename, _ in MODELS:
        path = os.path.join(RESULTS_DIR, filename)
        status = "present" if os.path.exists(path) else "MISSING"
        print(f"#   {filename}: {status}")


if __name__ == "__main__":
    main()
