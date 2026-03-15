import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.utils.chromosome_vocab import canonicalize_chromosome_id

class ChromosomeDataset(Dataset):
    def __init__(self, csv_path, transform=None, chr_to_idx=None, use_chromosome_id=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.chr_to_idx = chr_to_idx
        self.use_chromosome_id = use_chromosome_id

        required_cols = ["image_path", "label"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        if self.use_chromosome_id and "chromosome_id" not in self.df.columns:
            raise ValueError("use_chromosome_id=True but csv missing chromosome_id column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])

        sample = {
            "image": image,
            "label": label,
            "image_path": image_path,
        }

        if "chromosome_id" in self.df.columns:
            raw_chr = row["chromosome_id"]
            canon_chr = canonicalize_chromosome_id(raw_chr)
            sample["chromosome_id"] = raw_chr
            sample["chromosome_id_canonical"] = canon_chr

            if self.use_chromosome_id:
                if self.chr_to_idx is None:
                    raise ValueError("chr_to_idx must be provided when use_chromosome_id=True")

                chr_idx = self.chr_to_idx.get(canon_chr, self.chr_to_idx.get("UNK", 0))
                sample["chr_idx"] = chr_idx

        optional_cols = ["case_id", "case_dir", "split", "filename"]
        for col in optional_cols:
            if col in self.df.columns:
                sample[col] = row[col]

        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])

        return sample