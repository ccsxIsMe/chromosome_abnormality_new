import pandas as pd
from pathlib import Path


BASE_DIR = Path("/data5/chensx/MyProject/chromosome_abnormality_new/data")


def parse_pair_key_chr_side(filename: str):
    """
    例:
    75263.001A.K_1L.png  -> pair_key='75263.001A.K_1', chrom='1', side='L'
    75263.001A.K_1R.png  -> pair_key='75263.001A.K_1', chrom='1', side='R'
    75263.001A.K_22L.png -> pair_key='75263.001A.K_22', chrom='22', side='L'
    75263.001A.K_XR.png  -> pair_key='75263.001A.K_X', chrom='X', side='R'
    """
    stem = Path(filename).stem
    token = stem.split("_")[-1]   # 1L / 22R / XL / YR

    if len(token) < 2:
        return None, None, None

    side = token[-1].upper()
    chrom = token[:-1].upper()

    if side not in ["L", "R"]:
        return None, None, None

    if chrom.isdigit():
        chrom = str(int(chrom))   # 01 -> 1

    pair_key = stem[:-1]          # 去掉最后 L/R
    return pair_key, chrom, side


def is_autosome(chrom: str):
    return chrom.isdigit() and 1 <= int(chrom) <= 22


def build_pair_csv(split: str):
    in_csv = BASE_DIR / f"{split}.csv"
    out_csv = BASE_DIR / f"{split}_pair.csv"

    df = pd.read_csv(in_csv)

    records = []
    skipped_xy = 0
    skipped_invalid_parse = 0

    for _, row in df.iterrows():
        filename = row["filename"] if "filename" in row else Path(row["image_path"]).name
        pair_key, chrom, side = parse_pair_key_chr_side(filename)

        if pair_key is None:
            skipped_invalid_parse += 1
            continue

        # 只保留 1~22，排除 X/Y
        if not is_autosome(chrom):
            skipped_xy += 1
            continue

        records.append({
            "pair_key": pair_key,
            "case_id": row["case_id"],
            "chromosome_id": chrom,
            "side": side,
            "image_path": row["image_path"],
            "label": int(row["label"]),
            "filename": filename,
        })

    df2 = pd.DataFrame(records)

    pair_rows = []
    missing_groups = 0
    duplicate_groups = 0
    invalid_label_groups = 0

    for pair_key, grp in df2.groupby("pair_key"):
        left_grp = grp[grp["side"] == "L"]
        right_grp = grp[grp["side"] == "R"]

        if len(left_grp) == 0 or len(right_grp) == 0:
            missing_groups += 1
            continue

        if len(left_grp) > 1 or len(right_grp) > 1:
            duplicate_groups += 1

        left_row = left_grp.iloc[0]
        right_row = right_grp.iloc[0]

        left_label = int(left_row["label"])
        right_label = int(right_row["label"])

        # 只接受：
        # (0,0) -> normal pair
        # (0,1)/(1,0) -> abnormal pair
        # (1,1) -> 暂时视为异常数据，跳过
        if (left_label, right_label) == (0, 0):
            pair_label = 0
        elif (left_label, right_label) in [(0, 1), (1, 0)]:
            pair_label = 1
        else:
            invalid_label_groups += 1
            continue

        pair_rows.append({
            "left_path": left_row["image_path"],
            "right_path": right_row["image_path"],
            "label": pair_label,
            "chromosome_id": left_row["chromosome_id"],
            "case_id": left_row["case_id"],
            "pair_key": pair_key,
            "left_single_label": left_label,
            "right_single_label": right_label,
            "left_filename": left_row["filename"],
            "right_filename": right_row["filename"],
            "split": split,
        })

    out_df = pd.DataFrame(pair_rows)
    out_df.to_csv(out_csv, index=False)

    print(f"[{split}] saved to: {out_csv}")
    print(f"  total_pairs = {len(out_df)}")
    if len(out_df) > 0:
        print("  pair label counts:")
        print(out_df["label"].value_counts(dropna=False))
        print("  chromosome counts:")
        print(out_df["chromosome_id"].value_counts().sort_index())
    print(f"  skipped_xy = {skipped_xy}")
    print(f"  skipped_invalid_parse = {skipped_invalid_parse}")
    print(f"  missing_groups = {missing_groups}")
    print(f"  duplicate_groups = {duplicate_groups}")
    print(f"  invalid_label_groups = {invalid_label_groups}")
    print("-" * 60)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        build_pair_csv(split)