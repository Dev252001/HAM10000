"""
Tests for HAM10000Dataset and the class-weight computation.

These tests do NOT require the real dataset on disk.
They build minimal fake DataFrames that match the schema
load_metadata() returns, and verify behaviour in isolation.

Run from the repo root:
    pytest tests/ -v
"""
import sys
import os

# Allow importing from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from data_loader import HAM10000Dataset, CLASSES, CLASS_TO_IDX
from preprocessing import compute_class_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_df(tmp_path, n_per_class: int = 3) -> pd.DataFrame:
    """
    Build a minimal DataFrame matching the schema of load_metadata().
    Creates tiny 4×4 RGB JPEG files so HAM10000Dataset.__getitem__ works.
    """
    rows = []
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for cls in CLASSES:
        for i in range(n_per_class):
            img_id = f"ISIC_{cls}_{i:04d}"
            fpath = str(img_dir / f"{img_id}.jpg")
            # Save a small solid-colour image (4×4 pixels)
            Image.new("RGB", (4, 4), color=(i * 20, 100, 200)).save(fpath)
            rows.append({
                "image_id":    img_id,
                "lesion_id":   f"HAM_{cls}_{i:04d}",
                "dx":          cls,
                "dx_type":     "histo",
                "age":         30.0,
                "sex":         "male",
                "localization": "back",
                "label":       cls,
                "class_idx":   CLASS_TO_IDX[cls],
                "filepath":    fpath,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# test_dataset.py
# ---------------------------------------------------------------------------

class TestHAM10000Dataset:

    def test_len(self, tmp_path):
        df = _make_fake_df(tmp_path, n_per_class=3)
        ds = HAM10000Dataset(df)
        assert len(ds) == len(CLASSES) * 3

    def test_getitem_returns_tensor_and_int(self, tmp_path):
        df = _make_fake_df(tmp_path, n_per_class=2)
        ds = HAM10000Dataset(df)
        image, label = ds[0]
        # Without a transform the dataset returns a PIL Image
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)

    def test_getitem_with_transform(self, tmp_path):
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        df = _make_fake_df(tmp_path, n_per_class=2)
        ds = HAM10000Dataset(df, transform=tf)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == torch.Size([3, 32, 32])
        assert 0 <= label < len(CLASSES)

    def test_label_matches_class_idx(self, tmp_path):
        df = _make_fake_df(tmp_path, n_per_class=1)
        ds = HAM10000Dataset(df)
        for idx in range(len(ds)):
            _, label = ds[idx]
            expected = CLASS_TO_IDX[df.iloc[idx]["dx"]]
            assert label == expected, (
                f"Row {idx}: got label {label}, expected {expected} "
                f"for class '{df.iloc[idx]['dx']}'"
            )
