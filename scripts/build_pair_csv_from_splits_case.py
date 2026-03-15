import csv
from pathlib import Path
from collections import defaultdict
import re


ROOT = Path("/data5/chensx/MyProject/UAE/data/splits-case")
OUTPUT_DIR = Path("/data5/chensx/MyProject/chromosome_abnormality_new/data")


def parse_case_id_from_filename(filename: str) -> str:
    return filename.split(".")[0]


def parse_chr_and_side(filename: str):
    """
    例:
    75263.001A.K_1L.png -> ('1', 'L')
    75263.001A.K_22R.png -> ('22', 'R')
    75263.001A.K_XL.png -> ('X', 'L')
    """
    stem = Path(filename).stem
    token = stem.split("_")[-1]  # 1L / 22R / XL / YR

    if len(token) < 2:
        return None, None

    side = token[-1].upper()
    chrom = token[:-1]

    if side not in ["L", "R"]:
        return None, None

    chrom = chrom.upper()
    if chrom.isdigit():
        chrom = str(int(chrom))  # 01 -> 1

    return chrom, side


def collect_split(split_name: str):
    split_root = ROOT / split_name
    rows = []

    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    # pair_map[(case_id, chrom_id)] = {"L": {...}, "R": {...}}
    pair_map = defaultdict(dict)

    for case_dir in sorted(split_root.iterdir()):
        if not case_dir.is_dir():
            continue

        case_dir_name = case_dir.name

        for cls_name, label in [("normal", 0), ("abnormal", 1)]:
            cls_dir = case_dir / cls_name
            if not cls_dir.exists():
                continue

            for img_path in sorted(cls_dir.glob("*.png")):
                filename = img_path.name
                case_id = parse_case_id_from_filename(filename)
                chrom_id, side = parse_chr_and_side(filename)

                if chrom_id is None or side is None:
                    continue

                key = (case_id, chrom_id)
                pair_map[key][side] = {
                    "path": str(img_path.resolve()),
                    "single_label": label,
                    "filename": filename,
                    "case_dir": case_dir_name,
                }

    for (case_id, chrom_id), item in pair_map.items():
        if "L" not in item or "R" not in item:
            continue

        left_info = item["L"]
        right_info = item["R"]

        # pair-level label: 只要一边异常，就记为 abnormal pair
        pair_label = 1 if (left_info["single_label"] == 1 or right_info["single_label"] == 1) else 0

        rows.append({
            "left_path": left_info["path"],
            "right_path": right_info["path"],
            "label": pair_label,
            "chromosome_id": chrom_id,
            "case_id": case_id,
            "left_single_label": left_info["single_label"],
            "right_single_label": right_info["single_label"],
            "left_filename": left_info["filename"],
            "right_filename": right_info["filename"],
            "split": split_name,
        })

    return rows


def save_csv(rows, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "left_path",
        "right_path",
        "label",
        "chromosome_id",
        "case_id",
        "left_single_label",
        "right_single_label",
        "left_filename",
        "right_filename",
        "split",
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    for split in ["train", "val", "test"]:
        rows = collect_split(split)
        save_path = OUTPUT_DIR / f"{split}_pair.csv"
        save_csv(rows, save_path)

        total = len(rows)
        normal = sum(r["label"] == 0 for r in rows)
        abnormal = sum(r["label"] == 1 for r in rows)

        print(f"[{split}] saved to {save_path}")
        print(f"  total_pairs={total}, normal_pairs={normal}, abnormal_pairs={abnormal}")


if __name__ == "__main__":
    main()