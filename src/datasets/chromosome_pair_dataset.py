import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChromosomePairDataset(Dataset):
    def __init__(self, csv_path, transform=None, chr_to_idx=None, use_chromosome_id=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.chr_to_idx = chr_to_idx
        self.use_chromosome_id = use_chromosome_id

        required_cols = ["left_path", "right_path", "label", "chromosome_id"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        left_img = Image.open(row["left_path"]).convert("RGB")
        right_img = Image.open(row["right_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        sample = {
            "left_image": left_img,
            "right_image": right_img,
            "label": label,
            "left_path": row["left_path"],
            "right_path": row["right_path"],
            "chromosome_id": str(row["chromosome_id"]),
        }

        if self.use_chromosome_id:
            if self.chr_to_idx is None:
                raise ValueError("chr_to_idx must be provided when use_chromosome_id=True")
            chr_idx = self.chr_to_idx[str(row["chromosome_id"])]
            sample["chr_idx"] = chr_idx

        optional_cols = [
            "case_id",
            "pair_key",
            "left_single_label",
            "right_single_label",
            "left_filename",
            "right_filename",
            "split",
        ]
        for col in optional_cols:
            if col in self.df.columns:
                sample[col] = row[col]

        return sample