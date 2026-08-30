"""
src/preprocessing.py
────────────────────
Stage 2: Stratified splits, image transforms, DataLoader factory,
and class-weight computation for weighted CrossEntropyLoss.

Split rationale — 70 / 15 / 15
───────────────────────────────
With 10,015 images and severe class imbalance (the rarest class, df, has only
115 samples), we need enough minority-class images in val and test to get
stable metrics.  A 70/15/15 split gives:
  - df (115 total): ~80 train / ~17 val / ~17 test
  - vasc (142 total): ~99 train / ~21 val / ~21 test
An 80/10/10 split would leave only ~11 df images per eval set — too few for
reliable recall estimates.  70/15/15 is the safer choice here.

Augmentation rationale
──────────────────────
All augmentations are chosen to be *class-preserving*: they simulate natural
variation in how a lesion is photographed without destroying the diagnostic
features a dermatologist would use to classify it.

  RandomHorizontalFlip  — lesions have no left/right orientation; flipping
                          doubles effective training data at zero distortion cost.
  RandomVerticalFlip    — same reasoning; dermoscopic images can be taken from
                          any angle.
  RandomRotation(±20°)  — the camera can be rotated; ±20° is enough to cover
                          realistic variation without stretching the lesion.
  ColorJitter(bright=0.2, contrast=0.2, sat=0.2, hue=0.05)
                        — lighting and white-balance differ between devices.
                          Small perturbations improve generalisation.
                          hue is kept very small (0.05) — colour IS a
                          diagnostic feature (e.g. melanoma darkness), so we
                          must not alter it aggressively.

Explicitly excluded:
  RandomCrop / aggressive crop — could cut off the lesion boundary, which
                                 encodes diagnostic shape information.
  Grayscale / channel-drop    — colour is clinically meaningful.
  Large hue/saturation shifts — would change colour cues used for diagnosis.

ImageNet normalisation
──────────────────────
We will use pretrained ImageNet backbones in Stage 4.  Using ImageNet
mean/std now (even for the Stage 3 baseline) means the preprocessing
pipeline does not have to change between stages.

Class weights
─────────────
Weights are computed as:  w_c = N / (C × n_c)
  N  = total training samples
  C  = number of classes (7)
  n_c = number of training samples in class c
This is the standard inverse-frequency formula.  A class with fewer samples
gets a higher weight, so the loss treats every class as equally important
regardless of how many images it has.  The tensor is ordered to match
CLASS_TO_IDX from data_loader.py so it can be passed directly to
torch.nn.CrossEntropyLoss(weight=class_weights).
"""

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import CLASSES, CLASS_TO_IDX, HAM10000Dataset


# ── Constants ─────────────────────────────────────────────────────────────────

# Standard ImageNet statistics — used because we'll fine-tune pretrained
# backbones in Stage 4, and the pixel distribution must match what those
# models were trained on.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Input size expected by most ImageNet pretrained models (EfficientNet, ResNet, etc.)
IMAGE_SIZE = 224


# ── Split ─────────────────────────────────────────────────────────────────────

def make_splits(df, val_size=0.15, test_size=0.15, random_state=42):
    """
    Stratified train / val / test split on the 'dx' column.

    Parameters
    ----------
    df : pd.DataFrame
        Full metadata DataFrame from load_metadata().
    val_size : float
        Fraction of the full dataset reserved for validation.
    test_size : float
        Fraction of the full dataset reserved for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
        Each DataFrame has a reset index and retains all original columns.

    Notes
    -----
    We perform two successive stratified splits:
      1. Split off the test set first (test_size of the full data).
      2. Split the remainder into train and val  (val_size / (1 - test_size)
         of the remainder, which equals val_size of the full data).
    Stratifying on 'dx' ensures every class appears in all three splits in
    roughly the same proportion as the full dataset.
    """
    # Step 1: carve out test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["dx"],
        random_state=random_state,
    )

    # Step 2: carve out val set from the remaining train+val pool
    # Adjusted fraction so val is val_size of the *full* dataset
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df["dx"],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Return the torchvision transform pipeline for the given split.

    Parameters
    ----------
    split : str
        One of 'train', 'val', or 'test'.

    Returns
    -------
    transforms.Compose
        For 'train': resize → augmentation → tensor → normalise.
        For 'val'/'test': resize → tensor → normalise (no augmentation).
    """
    normalise = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),          # no orientation bias
            transforms.RandomVerticalFlip(),            # same reasoning
            transforms.RandomRotation(degrees=20),      # camera-angle variation
            transforms.ColorJitter(                     # lighting variation
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,                               # tiny — colour is diagnostic
            ),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        # val / test: deterministic pipeline — no stochastic transforms
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalise,
        ])


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(train_df) -> torch.Tensor:
    """
    Compute per-class weights for CrossEntropyLoss using the training set only.

    Formula:  w_c = N / (C × n_c)
      N   = total training samples
      C   = number of classes
      n_c = number of training samples in class c

    The returned tensor is ordered by CLASS_TO_IDX (alphabetical: akiec, bcc,
    bkl, df, mel, nv, vasc) so it can be passed directly to:
        nn.CrossEntropyLoss(weight=class_weights.to(device))

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (output of make_splits).

    Returns
    -------
    torch.Tensor of shape (n_classes,) with dtype float32.
    """
    n_total = len(train_df)
    n_classes = len(CLASSES)
    counts = train_df["dx"].value_counts()

    weights = []
    for cls in CLASSES:  # iterate in CLASS_TO_IDX order
        n_c = counts.get(cls, 1)  # fallback to 1 to avoid division by zero
        weights.append(n_total / (n_classes * n_c))

    return torch.tensor(weights, dtype=torch.float32)


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_dataloaders(train_df, val_df, test_df,
                     batch_size: int = 32,
                     num_workers: int = 2) -> dict:
    """
    Wrap the three split DataFrames in HAM10000Dataset + DataLoader.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Outputs of make_splits().
    batch_size : int
        Number of images per batch. 32 is a safe default for a T4 GPU.
    num_workers : int
        Number of worker processes for data loading. 2 works well on Colab.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each containing a DataLoader.
    """
    datasets = {
        "train": HAM10000Dataset(train_df, transform=get_transforms("train")),
        "val":   HAM10000Dataset(val_df,   transform=get_transforms("val")),
        "test":  HAM10000Dataset(test_df,  transform=get_transforms("test")),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,           # shuffle every epoch for training
            num_workers=num_workers,
            pin_memory=True,        # faster CPU→GPU transfer
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,          # deterministic eval
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders
