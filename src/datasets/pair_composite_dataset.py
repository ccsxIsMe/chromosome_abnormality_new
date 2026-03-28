import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


def load_grayscale_image(path):
    return Image.open(path).convert("L")


def build_pair_composite_tensor(left_path, right_path):
    left_img = load_grayscale_image(left_path)
    right_img = load_grayscale_image(right_path)

    if left_img.size != right_img.size:
        right_img = right_img.resize(left_img.size)

    left_t = torch.from_numpy(np.array(left_img, dtype="float32"))
    right_t = torch.from_numpy(np.array(right_img, dtype="float32"))
    diff_t = (left_t - right_t).abs()

    composite = torch.stack([left_t, right_t, diff_t], dim=0) / 255.0
    return composite


class PairCompositeDataset(Dataset):
    """
    Represent a homologous chromosome pair as a single 3-channel image:
    - channel 0: left grayscale chromosome
    - channel 1: right grayscale chromosome
    - channel 2: absolute difference |left - right|

    This keeps pair structure while allowing single-image anomaly frameworks
    to consume the data.
    """

    def __init__(self, csv_path, transform=None, return_metadata=True):
        self.csv_path = str(csv_path)
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.return_metadata = return_metadata

        required_cols = ["left_path", "right_path", "label", "chromosome_id"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _load_grayscale(path):
        return load_grayscale_image(path)

    def _build_composite(self, left_path, right_path):
        return build_pair_composite_tensor(left_path, right_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        composite = self._build_composite(row["left_path"], row["right_path"])

        sample = {
            "image": composite,
            "label": int(row["label"]),
        }

        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])

        if self.return_metadata:
            optional_cols = [
                "case_id",
                "pair_key",
                "chromosome_id",
                "abnormal_subtype_id",
                "subtype_status",
                "left_path",
                "right_path",
                "left_filename",
                "right_filename",
                "split",
            ]
            for col in optional_cols:
                if col in self.df.columns:
                    value = row[col]
                    sample[col] = "" if pd.isna(value) else str(value)

        return sample


def save_pair_composite_image(left_path, right_path, output_path):
    composite = build_pair_composite_tensor(left_path, right_path)
    image = (composite.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
