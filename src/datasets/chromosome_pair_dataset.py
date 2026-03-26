import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.utils.inversion_attributes import get_inversion_attributes_by_chromosome


class ChromosomePairDataset(Dataset):
    def __init__(
        self,
        csv_path,
        transform=None,
        chr_to_idx=None,
        use_chromosome_id=False,
        return_style_view=False,
        style_transform=None,
        random_swap=False,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.chr_to_idx = chr_to_idx
        self.use_chromosome_id = use_chromosome_id
        self.return_style_view = return_style_view
        self.style_transform = style_transform
        self.random_swap = random_swap

        required_cols = ["left_path", "right_path", "label", "chromosome_id"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        left_path = row["left_path"]
        right_path = row["right_path"]
        left_single_label = int(row["left_single_label"]) if "left_single_label" in self.df.columns else 0
        right_single_label = int(row["right_single_label"]) if "right_single_label" in self.df.columns else 0
        left_filename = row["left_filename"] if "left_filename" in self.df.columns else left_path
        right_filename = row["right_filename"] if "right_filename" in self.df.columns else right_path

        if self.random_swap and random.random() < 0.5:
            left_path, right_path = right_path, left_path
            left_single_label, right_single_label = right_single_label, left_single_label
            left_filename, right_filename = right_filename, left_filename

        left_img_raw = Image.open(left_path).convert("RGB")
        right_img_raw = Image.open(right_path).convert("RGB")
        label = int(row["label"])
        side_label = -1
        if left_single_label == 1 and right_single_label == 0:
            side_label = 0
        elif left_single_label == 0 and right_single_label == 1:
            side_label = 1

        # main view
        left_img = left_img_raw.copy()
        right_img = right_img_raw.copy()
        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        sample = {
            "left_image": left_img,
            "right_image": right_img,
            "label": label,
            "left_path": left_path,
            "right_path": right_path,
            "chromosome_id": str(row["chromosome_id"]),
            "side_label": side_label,
        }

        if self.use_chromosome_id:
            if self.chr_to_idx is None:
                raise ValueError("chr_to_idx must be provided when use_chromosome_id=True")
            chr_idx = self.chr_to_idx[str(row["chromosome_id"])]
            sample["chr_idx"] = chr_idx

        # optional style-perturbed second view
        if self.return_style_view:
            left_img_style = left_img_raw.copy()
            right_img_style = right_img_raw.copy()

            if self.style_transform is not None:
                left_img_style = self.style_transform(left_img_style)
                right_img_style = self.style_transform(right_img_style)
            elif self.transform is not None:
                left_img_style = self.transform(left_img_style)
                right_img_style = self.transform(right_img_style)

            sample["left_image_style"] = left_img_style
            sample["right_image_style"] = right_img_style

        attr = get_inversion_attributes_by_chromosome(row["chromosome_id"])
        if label == 1 and attr is not None:
            sample["karyotype_text"] = attr["karyotype"]
            sample["pericentric_label"] = attr["pericentric_label"]
            sample["bp1_arm_label"] = attr["bp1_arm_label"]
            sample["bp2_arm_label"] = attr["bp2_arm_label"]
            sample["bp1_major_label"] = attr["bp1_major_label"]
            sample["bp2_major_label"] = attr["bp2_major_label"]
            sample["bp1_text"] = attr["bp1"]
            sample["bp2_text"] = attr["bp2"]
            sample["bp1_major_token"] = attr["bp1_major_token"]
            sample["bp2_major_token"] = attr["bp2_major_token"]
        else:
            sample["karyotype_text"] = ""
            sample["pericentric_label"] = -1
            sample["bp1_arm_label"] = -1
            sample["bp2_arm_label"] = -1
            sample["bp1_major_label"] = -1
            sample["bp2_major_label"] = -1
            sample["bp1_text"] = ""
            sample["bp2_text"] = ""
            sample["bp1_major_token"] = ""
            sample["bp2_major_token"] = ""

        optional_cols = [
            "case_id",
            "pair_key",
            "abnormal_subtype_id",
            "subtype_status",
            "left_single_label",
            "right_single_label",
            "left_filename",
            "right_filename",
            "split",
        ]
        text_optional_cols = {
            "case_id",
            "pair_key",
            "abnormal_subtype_id",
            "subtype_status",
            "left_filename",
            "right_filename",
            "split",
        }
        for col in optional_cols:
            if col in self.df.columns:
                if col == "left_single_label":
                    sample[col] = left_single_label
                elif col == "right_single_label":
                    sample[col] = right_single_label
                elif col == "left_filename":
                    sample[col] = left_filename
                elif col == "right_filename":
                    sample[col] = right_filename
                else:
                    value = row[col]
                    if col in text_optional_cols:
                        sample[col] = "" if pd.isna(value) else str(value)
                    else:
                        sample[col] = value

        return sample
