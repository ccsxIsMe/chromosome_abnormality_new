import os
import csv
from pathlib import Path


ROOT = Path("/data5/chensx/MyProject/UAE/data/splits-case")
OUTPUT_DIR = Path("/data5/chensx/MyProject/chromosome_abnormality_new/data")


def parse_case_id_from_filename(filename: str) -> str:
    """
    例：
    75263.001A.K_1L.png -> 75263
    A75263.001A.K_1L.png -> A75263
    75263A.001A.K_1L.png -> 75263A
    """
    return filename.split(".")[0]


def parse_chromosome_id_from_filename(filename: str) -> str:
    """
    例：
    75263.001A.K_1L.png -> 1
    75263.001A.K_XL.png -> X
    75263.001A.K_YR.png -> Y
    """
    stem = Path(filename).stem  # 75263.001A.K_1L
    token = stem.split("_")[-1]  # 1L / XL / YR / 22R

    if len(token) < 2:
        return "unknown"

    side = token[-1].upper()
    chrom = token[:-1]

    if side not in ["L", "R"]:
        # 如果最后一位不是 L/R，就直接返回原 token
        return token

    return chrom


def collect_split(split_name: str):
    """
    扫描:
    /data5/chensx/MyProject/UAE/data/splits-case/train/<case_id>/normal/*.png
    /data5/chensx/MyProject/UAE/data/splits-case/train/<case_id>/abnormal/*.png
    """
    split_root = ROOT / split_name
    rows = []

    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    for case_dir in sorted(split_root.iterdir()):
        if not case_dir.is_dir():
            continue

        case_id_from_dir = case_dir.name

        for cls_name, label in [("normal", 0), ("abnormal", 1)]:
            cls_dir = case_dir / cls_name
            if not cls_dir.exists():
                continue

            for img_path in sorted(cls_dir.glob("*.png")):
                filename = img_path.name
                case_id_from_file = parse_case_id_from_filename(filename)
                chromosome_id = parse_chromosome_id_from_filename(filename)

                # 以文件名解析出的 case_id 为主，同时保留目录 case_id 做一致性检查
                row = {
                    "image_path": str(img_path.resolve()),
                    "label": label,
                    "chromosome_id": chromosome_id,
                    "case_id": case_id_from_file,
                    "case_dir": case_id_from_dir,
                    "split": split_name,
                    "filename": filename,
                }
                rows.append(row)

    return rows


def save_csv(rows, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "label",
        "chromosome_id",
        "case_id",
        "case_dir",
        "split",
        "filename",
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_stats = {}

    for split in ["train", "val", "test"]:
        rows = collect_split(split)
        save_path = OUTPUT_DIR / f"{split}.csv"
        save_csv(rows, save_path)

        total = len(rows)
        normal = sum(1 for r in rows if r["label"] == 0)
        abnormal = sum(1 for r in rows if r["label"] == 1)

        all_stats[split] = {
            "total": total,
            "normal": normal,
            "abnormal": abnormal,
        }

        print(f"[{split}] saved to {save_path}")
        print(f"  total={total}, normal={normal}, abnormal={abnormal}")

    print("\nSummary:")
    for split, stat in all_stats.items():
        print(f"{split}: {stat}")


if __name__ == "__main__":
    main()