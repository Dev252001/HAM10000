"""
src/data_loader.py
──────────────────
Handles two responsibilities:
  1. Downloading the HAM10000 dataset from Kaggle into a local directory.
  2. Loading the metadata CSV + resolving each image_id to its file path,
     then exposing a PyTorch Dataset class for use in DataLoaders.

Design note: keeping download logic and the Dataset class in the same file
makes the data pipeline self-contained — one import covers both concerns.
"""

import os
import glob
import subprocess

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# ── Class definitions ─────────────────────────────────────────────────────────

# Alphabetically sorted so the mapping is deterministic across all runs.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

LABEL_MAP = {
    "nv"    : "Melanocytic Nevi",
    "mel"   : "Melanoma",
    "bkl"   : "Benign Keratosis",
    "bcc"   : "Basal Cell Carcinoma",
    "akiec" : "Actinic Keratosis / IEC",
    "vasc"  : "Vascular Lesion",
    "df"    : "Dermatofibroma",
}

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
IDX_TO_LABEL = {idx: LABEL_MAP[cls] for cls, idx in CLASS_TO_IDX.items()}

# Kaggle dataset slug — "Skin Cancer MNIST: HAM10000" by K Scott Mader
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"


# ── Download helper ───────────────────────────────────────────────────────────

def download_dataset(dest_dir: str) -> None:
    """
    Download and unzip the HAM10000 dataset from Kaggle using the Kaggle CLI.

    Requires:
      - kaggle CLI installed  (pip install kaggle)
      - kaggle.json placed at ~/.kaggle/kaggle.json  (chmod 600)

    Parameters
    ----------
    dest_dir : str
        Directory where the dataset will be extracted, e.g. "/content/ham10000-classifier/data"

    Notes
    -----
    The function is idempotent: if the metadata CSV already exists in dest_dir,
    it skips the download so re-running the notebook doesn't re-download 2.5 GB.
    """
    csv_path = os.path.join(dest_dir, "HAM10000_metadata.csv")
    if os.path.exists(csv_path):
        print(f"Dataset already present at '{dest_dir}'. Skipping download.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading HAM10000 to '{dest_dir}' …")
    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", KAGGLE_DATASET,
         "-p", dest_dir,
         "--unzip"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)
    print("Download complete.")


# ── Metadata loading ──────────────────────────────────────────────────────────

def load_metadata(data_dir: str) -> pd.DataFrame:
    """
    Read HAM10000_metadata.csv and attach the resolved file path for each image.

    Parameters
    ----------
    data_dir : str
        Root directory that contains HAM10000_metadata.csv,
        HAM10000_images_part_1/, and HAM10000_images_part_2/.

    Returns
    -------
    pd.DataFrame
        One row per image with columns:
          image_id, dx, dx_type, age, sex, localization,
          label (full name), class_idx (integer), filepath (absolute path)

    Raises
    ------
    FileNotFoundError
        If the metadata CSV is missing — usually means download didn't complete.
    ValueError
        If any image_id in the CSV has no matching file on disk.
    """
    csv_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata CSV not found at '{csv_path}'. "
            "Run download_dataset() first."
        )

    df = pd.read_csv(csv_path)

    # Build image_id → filepath by scanning both image folders
    image_paths: dict[str, str] = {}
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        folder = os.path.join(data_dir, part)
        for fpath in glob.glob(os.path.join(folder, "*.jpg")):
            img_id = os.path.splitext(os.path.basename(fpath))[0]
            image_paths[img_id] = fpath

    df["filepath"] = df["image_id"].map(image_paths)

    missing = df["filepath"].isna().sum()
    if missing > 0:
        raise ValueError(
            f"{missing} images in the CSV have no matching file on disk. "
            "The download may be incomplete."
        )

    # Attach human-readable label and integer class index
    df["label"]     = df["dx"].map(LABEL_MAP)
    df["class_idx"] = df["dx"].map(CLASS_TO_IDX)

    return df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    Wraps a DataFrame (a train, val, or test split) and applies an optional
    torchvision transform to each image at load time.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain columns 'filepath' (str) and 'class_idx' (int).
        Typically produced by load_metadata() or one of the split functions
        in preprocessing.py.
    transform : callable, optional
        A torchvision transforms pipeline. If None, images are returned as
        raw PIL Images — not suitable for model input, but useful for debugging.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        # Reset index so integer indexing is contiguous (required by DataLoader)
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["class_idx"])

        if self.transform:
            image = self.transform(image)

        return image, label
