"""
Tests for compute_class_weights() in src/preprocessing.py.

Verifies the formula w_c = N / (C * n_c) on a known small input
where the answer can be computed by hand.

Run from the repo root:
    pytest tests/ -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pandas as pd

from preprocessing import compute_class_weights
from data_loader import CLASSES, CLASS_TO_IDX


class TestComputeClassWeights:

    def test_formula_on_known_input(self):
        """
        Build a DataFrame where every class has exactly `n` images.
        For a balanced dataset w_c = N / (C * n) = (C*n) / (C*n) = 1.0 for all classes.
        """
        n = 5
        rows = [{"dx": cls} for cls in CLASSES for _ in range(n)]
        df = pd.DataFrame(rows)

        weights = compute_class_weights(df)

        assert weights.shape == torch.Size([len(CLASSES)])
        # All weights should be 1.0 for a perfectly balanced dataset
        expected = torch.ones(len(CLASSES))
        assert torch.allclose(weights, expected, atol=1e-5), (
            f"Expected all-ones for balanced dataset, got {weights}"
        )

    def test_rare_class_gets_higher_weight(self):
        """
        A class with fewer samples must receive a higher weight than a class
        with more samples.  Concretely: if class A has 10 images and class B
        has 2 images, w_B > w_A.
        """
        # Make a df with only two classes, unequal counts
        cls_many = CLASSES[0]   # e.g. 'akiec'
        cls_few  = CLASSES[1]   # e.g. 'bcc'
        rows = (
            [{"dx": cls_many}] * 10 +
            [{"dx": cls_few}]  * 2  +
            # Fill remaining classes with equal counts to satisfy stratification
            [{"dx": cls} for cls in CLASSES[2:] for _ in range(5)]
        )
        df = pd.DataFrame(rows)
        weights = compute_class_weights(df)

        idx_many = CLASS_TO_IDX[cls_many]
        idx_few  = CLASS_TO_IDX[cls_few]
        assert weights[idx_few] > weights[idx_many], (
            f"Rare class weight ({weights[idx_few]:.4f}) should be > "
            f"common class weight ({weights[idx_many]:.4f})"
        )

    def test_output_is_float32_tensor(self):
        rows = [{"dx": cls} for cls in CLASSES for _ in range(3)]
        df = pd.DataFrame(rows)
        weights = compute_class_weights(df)

        assert isinstance(weights, torch.Tensor)
        assert weights.dtype == torch.float32

    def test_weight_order_matches_class_to_idx(self):
        """
        Weights must be ordered by CLASS_TO_IDX, not by DataFrame row order.
        Shuffle the DataFrame and verify weights are identical.
        """
        rows = [{"dx": cls} for cls in CLASSES for _ in range(4)]
        df_ordered  = pd.DataFrame(rows)
        df_shuffled = df_ordered.sample(frac=1, random_state=7).reset_index(drop=True)

        w1 = compute_class_weights(df_ordered)
        w2 = compute_class_weights(df_shuffled)

        assert torch.allclose(w1, w2, atol=1e-5), (
            "Weights changed when DataFrame row order changed — "
            "output is not indexed by CLASS_TO_IDX"
        )
