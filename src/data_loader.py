import os
import glob
import subprocess

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


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

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"


def download_dataset(dest_dir: str) -> None:
    """Download and unzip the HAM10000 dataset from Kaggle. Idempotent."""
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


def load_metadata(data_dir: str) -> pd.DataFrame:
    """
    Read HAM10000_metadata.csv and attach the resolved file path for each image.

    Returns a DataFrame with columns:
      image_id, dx, dx_type, age, sex, localization,
      label, class_idx, filepath

    Raises FileNotFoundError if the CSV is missing, ValueError if any image
    file is missing from disk.
    """
    csv_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Metadata CSV not found at '{csv_path}'. "
            "Run download_dataset() first."
        )

    df = pd.read_csv(csv_path)

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

    df["label"]     = df["dx"].map(LABEL_MAP)
    df["class_idx"] = df["dx"].map(CLASS_TO_IDX)

    return df


class HAM10000Dataset(Dataset):
    """PyTorch Dataset wrapping a train/val/test split DataFrame."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
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
