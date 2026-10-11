"""
Tests for make_lesion_splits() in src/preprocessing.py.

Key invariant: no lesion_id appears in more than one split.
That is the whole point of make_lesion_splits() — verify it.

Run from the repo root:
    pytest tests/ -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from preprocessing import make_lesion_splits, make_splits, make_split
from data_loader import CLASSES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_metadata(n_lesions_per_class: int = 6,
                         images_per_lesion: int = 2) -> pd.DataFrame:
    """
    Build a fake metadata DataFrame with known lesion_id structure.
    Each lesion has `images_per_lesion` images.
    Total images = len(CLASSES) * n_lesions_per_class * images_per_lesion.
    """
    rows = []
    img_counter = 0
    for cls in CLASSES:
        for les_i in range(n_lesions_per_class):
            lesion_id = f"HAM_{cls}_{les_i:04d}"
            for img_i in range(images_per_lesion):
                rows.append({
                    "image_id":    f"ISIC_{img_counter:07d}",
                    "lesion_id":   lesion_id,
                    "dx":          cls,
                    "filepath":    f"/fake/{img_counter}.jpg",
                    "class_idx":   0,
                })
                img_counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests for make_lesion_splits()
# ---------------------------------------------------------------------------

class TestMakeLesionSplits:

    def test_no_lesion_in_multiple_splits(self):
        """Core invariant: every lesion_id is in exactly one split."""
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        train_df, val_df, test_df = make_lesion_splits(df, random_state=42)

        train_ids = set(train_df["lesion_id"])
        val_ids   = set(val_df["lesion_id"])
        test_ids  = set(test_df["lesion_id"])

        assert train_ids.isdisjoint(val_ids),  "Lesion leak: train ∩ val"
        assert train_ids.isdisjoint(test_ids), "Lesion leak: train ∩ test"
        assert val_ids.isdisjoint(test_ids),   "Lesion leak: val ∩ test"

    def test_all_images_assigned(self):
        """Every image ends up in exactly one split."""
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        train_df, val_df, test_df = make_lesion_splits(df, random_state=42)

        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df), f"Image count mismatch: {total} != {len(df)}"

    def test_approximate_split_ratio(self):
        """Each split is roughly 70 / 15 / 15 of total images (within 10 pp)."""
        df = _make_fake_metadata(n_lesions_per_class=10, images_per_lesion=2)
        train_df, val_df, test_df = make_lesion_splits(df, random_state=42)

        n = len(df)
        assert abs(len(train_df) / n - 0.70) < 0.10, "Train split far from 70%"
        assert abs(len(val_df)   / n - 0.15) < 0.10, "Val split far from 15%"
        assert abs(len(test_df)  / n - 0.15) < 0.10, "Test split far from 15%"

    def test_reproducible_with_same_seed(self):
        """Same random_state must produce the same split every time."""
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        train1, val1, test1 = make_lesion_splits(df, random_state=42)
        train2, val2, test2 = make_lesion_splits(df, random_state=42)

        assert list(train1["image_id"]) == list(train2["image_id"])
        assert list(val1["image_id"])   == list(val2["image_id"])
        assert list(test1["image_id"])  == list(test2["image_id"])

    def test_different_seeds_give_different_splits(self):
        """Different seeds should produce different splits (very likely)."""
        df = _make_fake_metadata(n_lesions_per_class=8, images_per_lesion=2)
        train1, _, _ = make_lesion_splits(df, random_state=0)
        train2, _, _ = make_lesion_splits(df, random_state=99)

        assert set(train1["image_id"]) != set(train2["image_id"]), (
            "Seeds 0 and 99 produced identical splits — check random_state wiring"
        )

    def test_class_proportions_within_tolerance(self):
        """Each class's share in train is within ±5 pp of its share in the full dataset."""
        df = _make_fake_metadata(n_lesions_per_class=10, images_per_lesion=2)
        train_df, val_df, test_df = make_lesion_splits(df, random_state=42)

        n_total = len(df)
        for cls in CLASSES:
            expected_frac = (df["dx"] == cls).sum() / n_total
            actual_frac   = (train_df["dx"] == cls).sum() / len(train_df)
            assert abs(actual_frac - expected_frac) < 0.05, (
                f"Class '{cls}': train fraction {actual_frac:.3f} deviates "
                f"more than 5 pp from overall {expected_frac:.3f}"
            )


class TestMakeSplitWrapper:

    def test_lesion_mode_dispatches_correctly(self):
        """make_split('lesion') must produce zero lesion overlap."""
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        train_df, val_df, test_df = make_split(df, split_mode="lesion", random_state=42)

        train_ids = set(train_df["lesion_id"])
        val_ids   = set(val_df["lesion_id"])
        test_ids  = set(test_df["lesion_id"])

        assert train_ids.isdisjoint(val_ids),  "Lesion leak via wrapper: train ∩ val"
        assert train_ids.isdisjoint(test_ids), "Lesion leak via wrapper: train ∩ test"
        assert val_ids.isdisjoint(test_ids),   "Lesion leak via wrapper: val ∩ test"

    def test_image_mode_dispatches_correctly(self):
        """make_split('image') must assign every image exactly once."""
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        train_df, val_df, test_df = make_split(df, split_mode="image", random_state=42)

        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df)

    def test_invalid_mode_raises(self):
        """make_split with unknown split_mode must raise ValueError."""
        import pytest
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        with pytest.raises(ValueError, match="split_mode must be"):
            make_split(df, split_mode="bogus")

    def test_manifest_saved_with_correct_schema(self, tmp_path):
        """Manifest CSV must exist with columns image_id/lesion_id/dx/split."""
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        make_split(df, split_mode="lesion", random_state=42,
                   manifest_dir=str(tmp_path))

        manifest_file = tmp_path / "manifest_lesion_seed42.csv"
        assert manifest_file.exists(), "Manifest file not created"

        import pandas as _pd
        manifest = _pd.read_csv(manifest_file)
        assert set(manifest.columns) >= {"image_id", "lesion_id", "dx", "split"}

    def test_manifest_every_image_assigned_once(self, tmp_path):
        """Every image_id appears exactly once in the manifest."""
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        make_split(df, split_mode="lesion", random_state=42,
                   manifest_dir=str(tmp_path))

        import pandas as _pd
        manifest = _pd.read_csv(tmp_path / "manifest_lesion_seed42.csv")
        assert manifest["image_id"].nunique() == len(df), (
            "Some image_id appears more than once or is missing from the manifest"
        )

    def test_manifest_split_values(self, tmp_path):
        """Manifest 'split' column must contain exactly train/val/test."""
        df = _make_fake_metadata(n_lesions_per_class=6, images_per_lesion=2)
        make_split(df, split_mode="lesion", random_state=42,
                   manifest_dir=str(tmp_path))

        import pandas as _pd
        manifest = _pd.read_csv(tmp_path / "manifest_lesion_seed42.csv")
        assert set(manifest["split"]) == {"train", "val", "test"}
