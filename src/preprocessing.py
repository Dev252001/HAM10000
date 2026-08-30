"""
src/preprocessing.py
────────────────────
Stratified train / val / test splits, image resizing, normalization,
and data augmentation pipelines.

STAGE STATUS: stub — implementation added in Stage 2.
All public function signatures are defined here so that imports in other
modules and the notebook don't break before Stage 2 is written.
"""


def make_splits(df, val_size=0.15, test_size=0.15, random_state=42):
    """
    Return (train_df, val_df, test_df) with stratification on 'dx'.
    Implemented in Stage 2.
    """
    raise NotImplementedError("Implemented in Stage 2.")


def get_transforms(split: str):
    """
    Return a torchvision transforms pipeline for 'train', 'val', or 'test'.
    Augmentation choices are justified in Stage 2.
    Implemented in Stage 2.
    """
    raise NotImplementedError("Implemented in Stage 2.")
