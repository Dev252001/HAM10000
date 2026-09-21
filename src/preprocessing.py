import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import CLASSES, CLASS_TO_IDX, HAM10000Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def make_splits(df, val_size=0.15, test_size=0.15, random_state=42):
    """Stratified train/val/test split on the 'dx' column (70/15/15)."""
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["dx"],
        random_state=random_state,
    )

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


def get_transforms(split: str) -> transforms.Compose:
    """Return the torchvision transform pipeline for the given split."""
    normalise = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalise,
        ])


def compute_class_weights(train_df) -> torch.Tensor:
    """
    Compute per-class weights for CrossEntropyLoss using the training set only.
    Formula: w_c = N / (C * n_c). Ordered by CLASS_TO_IDX.
    """
    n_total   = len(train_df)
    n_classes = len(CLASSES)
    counts    = train_df["dx"].value_counts()

    weights = []
    for cls in CLASSES:
        n_c = counts.get(cls, 1)
        weights.append(n_total / (n_classes * n_c))

    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(train_df, val_df, test_df,
                     batch_size: int = 32,
                     num_workers: int = 2) -> dict:
    """Wrap the three split DataFrames in HAM10000Dataset + DataLoader."""
    datasets = {
        "train": HAM10000Dataset(train_df, transform=get_transforms("train")),
        "val":   HAM10000Dataset(val_df,   transform=get_transforms("val")),
        "test":  HAM10000Dataset(test_df,  transform=get_transforms("test")),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
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
