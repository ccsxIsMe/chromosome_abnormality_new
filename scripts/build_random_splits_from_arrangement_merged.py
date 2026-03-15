import os
import re
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd


ROOT_DIR = Path("/data5/chensx/MyProject/UAE/data/arrangement_merged")
OUT_DIR = Path("/data5/chensx/MyProject/chromosome_abnormality_new/data_randomized")

SEED = 42

# 你可以改这几个比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 如果想让 val/test 平衡，就启用这个
BALANCE_VAL_TEST = True

# 如果启用平衡，val/test 的 abnormal 数量由这里控制：
# None 表示自动取“能覆盖、且合理”的最大值
VAL_ABNORMAL_TARGET = None
TEST_ABNORMAL_TARGET = None


def set_seed(seed=42):
    random.seed(seed)


def normalize_chromosome_token(x: str):
    x = x.upper()
    if x.isdigit():
        return str(int(x))
    return x


def parse_folder_name(folder_name: str):
    """
    例:
    chr_01 -> chromosome_id='1', label=0, abnormal_type='normal'
    chr_1_inversion -> chromosome_id='1', label=1, abnormal_type='inversion'
    chr_x -> chromosome_id='X', label=0, abnormal_type='normal'
    chr_x_inversion -> chromosome_id='X', label=1, abnormal_type='inversion'
    """
    folder_name = folder_name.lower()

    m = re.match(r"^chr_([0-9]{1,2}|x|y)(?:_(.+))?$", folder_name)
    if not m:
        return None

    chrom = normalize_chromosome_token(m.group(1))
    suffix = m.group(2)

    if suffix is None:
        return {
            "chromosome_id": chrom,
            "label": 0,
            "abnormal_type": "normal",
            "folder_name": folder_name,
        }

    return {
        "chromosome_id": chrom,
        "label": 1,
        "abnormal_type": suffix.lower(),
        "folder_name": folder_name,
    }


def parse_filename_info(filename: str):
    """
    例:
    75263.001A.K_1L.png
    case_id = 75263
    pair_key = 75263.001A.K_1
    side = L
    side_token = 1L
    chromosome_from_name = 1
    """
    stem = Path(filename).stem

    if "." not in stem:
        case_id = stem.split("_")[0]
    else:
        case_id = stem.split(".")[0]

    token = stem.split("_")[-1]  # 1L / 22R / XL / YR

    side = None
    chrom = None
    pair_key = None

    if len(token) >= 2 and token[-1].upper() in ["L", "R"]:
        side = token[-1].upper()
        chrom = normalize_chromosome_token(token[:-1])
        pair_key = stem[:-1]  # 去掉最后 L/R
    else:
        side = "UNK"
        chrom = "UNK"
        pair_key = stem

    return {
        "filename": filename,
        "stem": stem,
        "case_id": case_id,
        "side": side,
        "chromosome_from_name": chrom,
        "pair_key": pair_key,
    }


def build_single_image_dataframe(root_dir: Path):
    rows = []

    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        folder_info = parse_folder_name(subdir.name)
        if folder_info is None:
            continue

        for img_path in sorted(subdir.glob("*.png")):
            file_info = parse_filename_info(img_path.name)

            row = {
                "image_path": str(img_path),
                "filename": img_path.name,
                "case_id": file_info["case_id"],
                "pair_key": file_info["pair_key"],
                "side": file_info["side"],
                "chromosome_id": folder_info["chromosome_id"],
                "chromosome_from_name": file_info["chromosome_from_name"],
                "label": folder_info["label"],
                "abnormal_type": folder_info["abnormal_type"],
                "source_folder": subdir.name,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # 一致性检查
    mismatch = df[
        (df["chromosome_from_name"] != "UNK") &
        (df["chromosome_id"].astype(str) != df["chromosome_from_name"].astype(str))
    ]
    if len(mismatch) > 0:
        print("[Warning] chromosome_id mismatch between folder and filename:")
        print(mismatch.head(20))

    return df


def print_single_split_stats(name, df):
    print(f"\n===== {name.upper()} =====")
    print("total:", len(df))
    print("label counts:")
    print(df["label"].value_counts(dropna=False).sort_index())

    print("\nchromosome distribution:")
    print(df["chromosome_id"].value_counts().sort_index())

    print("\nabnormal chromosome distribution:")
    print(df[df["label"] == 1]["chromosome_id"].value_counts().sort_index())

    print("\nabnormal type distribution:")
    print(df[df["label"] == 1]["abnormal_type"].value_counts().sort_index())

    print("\nunique case_id:", df["case_id"].nunique())
    print("unique pair_key:", df["pair_key"].nunique())


def stratified_split_indices(indices, train_ratio, val_ratio, test_ratio):
    n = len(indices)
    indices = indices[:]
    random.shuffle(indices)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def build_scheme1_single_image_random(df, out_dir):
    """
    方案1：完全单图级随机打散
    """
    print("\n==============================")
    print("SCHEME 1: fully single-image random split")
    print("==============================")

    strata = defaultdict(list)

    for idx, row in df.iterrows():
        key = (str(row["chromosome_id"]), int(row["label"]), str(row["abnormal_type"]))
        strata[key].append(idx)

    train_ids, val_ids, test_ids = [], [], []

    for key, idxs in strata.items():
        tr, va, te = stratified_split_indices(idxs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        train_ids.extend(tr)
        val_ids.extend(va)
        test_ids.extend(te)

    train_df = df.loc[train_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = df.loc[val_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = df.loc[test_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)

    scheme_dir = out_dir / "scheme1_single_image_random"
    scheme_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(scheme_dir / "train.csv", index=False)
    val_df.to_csv(scheme_dir / "val.csv", index=False)
    test_df.to_csv(scheme_dir / "test.csv", index=False)

    print_single_split_stats("train", train_df)
    print_single_split_stats("val", val_df)
    print_single_split_stats("test", test_df)


def build_group_df_by_pair_key(df):
    """
    用于方案2：同一 pair_key 必须进同一 split
    group 的标签用于分层：
    - chromosome_id
    - abnormal_type
    - group_label: 只要组内有 abnormal 就记为1，否则0
    """
    grouped_rows = []
    for pair_key, grp in df.groupby("pair_key"):
        chromosome_id = grp["chromosome_id"].iloc[0]
        case_id = grp["case_id"].iloc[0]

        group_label = int(grp["label"].max())
        abnormal_types = sorted(set(grp["abnormal_type"].tolist()))
        abnormal_type = abnormal_types[-1] if group_label == 1 else "normal"

        grouped_rows.append({
            "pair_key": pair_key,
            "case_id": case_id,
            "chromosome_id": chromosome_id,
            "group_label": group_label,
            "abnormal_type": abnormal_type,
            "member_indices": grp.index.tolist(),
        })

    return pd.DataFrame(grouped_rows)


def build_scheme2_grouped_by_pair(df, out_dir):
    """
    方案2：尽量分层打散，但避免同一左右同号对跨 split
    最小单位仍是 pair_key group，但最终输出 single-image csv
    """
    print("\n==============================")
    print("SCHEME 2: grouped-by-pair split (single-image output)")
    print("==============================")

    group_df = build_group_df_by_pair_key(df)

    strata = defaultdict(list)
    for idx, row in group_df.iterrows():
        key = (str(row["chromosome_id"]), int(row["group_label"]), str(row["abnormal_type"]))
        strata[key].append(idx)

    train_g, val_g, test_g = [], [], []

    for key, idxs in strata.items():
        tr, va, te = stratified_split_indices(idxs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        train_g.extend(tr)
        val_g.extend(va)
        test_g.extend(te)

    def expand_group_indices(group_indices):
        ids = []
        for gi in group_indices:
            ids.extend(group_df.loc[gi, "member_indices"])
        return ids

    train_ids = expand_group_indices(train_g)
    val_ids = expand_group_indices(val_g)
    test_ids = expand_group_indices(test_g)

    train_df = df.loc[train_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = df.loc[val_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = df.loc[test_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)

    scheme_dir = out_dir / "scheme2_grouped_by_pair_single_output"
    scheme_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(scheme_dir / "train.csv", index=False)
    val_df.to_csv(scheme_dir / "val.csv", index=False)
    test_df.to_csv(scheme_dir / "test.csv", index=False)

    print_single_split_stats("train", train_df)
    print_single_split_stats("val", val_df)
    print_single_split_stats("test", test_df)


def build_pair_dataframe(df):
    """
    构建 pair 级别数据
    规则：
    - 同一 pair_key 内，找 L 和 R
    - pair label = max(left_label, right_label)
    - 保留 left/right 单图标签
    """
    pair_rows = []

    missing_groups = 0
    duplicate_groups = 0

    for pair_key, grp in df.groupby("pair_key"):
        left_grp = grp[grp["side"] == "L"]
        right_grp = grp[grp["side"] == "R"]

        if len(left_grp) == 0 or len(right_grp) == 0:
            missing_groups += 1
            continue

        if len(left_grp) > 1 or len(right_grp) > 1:
            duplicate_groups += 1

        left_row = left_grp.iloc[0]
        right_row = right_grp.iloc[0]

        pair_label = int(max(left_row["label"], right_row["label"]))

        # abnormal_type：如果有 abnormal，用 abnormal 那一边的类型；否则 normal
        if pair_label == 1:
            abnormal_type = left_row["abnormal_type"] if left_row["label"] == 1 else right_row["abnormal_type"]
        else:
            abnormal_type = "normal"

        pair_rows.append({
            "left_path": left_row["image_path"],
            "right_path": right_row["image_path"],
            "label": pair_label,
            "chromosome_id": str(left_row["chromosome_id"]),
            "case_id": left_row["case_id"],
            "pair_key": pair_key,
            "left_single_label": int(left_row["label"]),
            "right_single_label": int(right_row["label"]),
            "left_filename": left_row["filename"],
            "right_filename": right_row["filename"],
            "abnormal_type": abnormal_type,
        })

    pair_df = pd.DataFrame(pair_rows)

    print("\n[Pair DataFrame Summary]")
    print("total_pairs:", len(pair_df))
    print("missing_groups:", missing_groups)
    print("duplicate_groups:", duplicate_groups)
    print(pair_df["label"].value_counts(dropna=False))

    return pair_df


def print_pair_split_stats(name, df):
    print(f"\n===== {name.upper()} =====")
    print("total pairs:", len(df))
    print("pair label counts:")
    print(df["label"].value_counts(dropna=False).sort_index())

    print("\nchromosome distribution:")
    print(df["chromosome_id"].value_counts().sort_index())

    print("\nabnormal chromosome distribution:")
    print(df[df["label"] == 1]["chromosome_id"].value_counts().sort_index())

    print("\nabnormal type distribution:")
    print(df[df["label"] == 1]["abnormal_type"].value_counts().sort_index())

    print("\nunique case_id:", df["case_id"].nunique())
    print("unique pair_key:", df["pair_key"].nunique())

    if "left_single_label" in df.columns and "right_single_label" in df.columns:
        print("\n(left_single_label, right_single_label) counts:")
        print(df.groupby(["left_single_label", "right_single_label"]).size())


def choose_balanced_val_test_from_pair_df(pair_df, val_abnormal_target=None, test_abnormal_target=None):
    """
    给方案3用：
    从 pair_df 中构建尽量平衡的 val/test：
    - val abnormal = val_abnormal_target
    - val normal 同样数量
    - test abnormal = test_abnormal_target
    - test normal 同样数量
    并尽量分层到 chromosome_id + abnormal_type
    """
    abnormal_df = pair_df[pair_df["label"] == 1].copy()
    normal_df = pair_df[pair_df["label"] == 0].copy()

    total_ab = len(abnormal_df)

    if val_abnormal_target is None:
        val_abnormal_target = max(1, int(total_ab * 0.15))
    if test_abnormal_target is None:
        test_abnormal_target = max(1, int(total_ab * 0.15))

    val_abnormal_target = min(val_abnormal_target, total_ab)
    test_abnormal_target = min(test_abnormal_target, total_ab - val_abnormal_target)

    strata = defaultdict(list)
    for idx, row in abnormal_df.iterrows():
        key = (str(row["chromosome_id"]), str(row["abnormal_type"]))
        strata[key].append(idx)

    val_ab_ids, test_ab_ids, remain_ab_ids = [], [], []

    # 先按 strata 比例分
    all_keys = list(strata.keys())
    for key in all_keys:
        idxs = strata[key][:]
        random.shuffle(idxs)

        n = len(idxs)
        n_val = int(round(n * val_abnormal_target / total_ab))
        n_test = int(round(n * test_abnormal_target / total_ab))

        if n_val + n_test > n:
            n_test = max(0, n - n_val)

        val_ab_ids.extend(idxs[:n_val])
        test_ab_ids.extend(idxs[n_val:n_val + n_test])
        remain_ab_ids.extend(idxs[n_val + n_test:])

    # 修正数量
    def fill_to_target(current_ids, remain_pool, target):
        current_ids = current_ids[:]
        remain_pool = remain_pool[:]
        random.shuffle(remain_pool)
        while len(current_ids) < target and len(remain_pool) > 0:
            current_ids.append(remain_pool.pop())
        return current_ids, remain_pool

    val_ab_ids, remain_ab_ids = fill_to_target(val_ab_ids, remain_ab_ids, val_abnormal_target)
    test_ab_ids, remain_ab_ids = fill_to_target(test_ab_ids, remain_ab_ids, test_abnormal_target)

    val_ab_df = abnormal_df.loc[val_ab_ids]
    test_ab_df = abnormal_df.loc[test_ab_ids]

    # 从 normal 中按 chromosome_id 匹配抽样
    def sample_matching_normals(target_df, n_target):
        chosen_ids = []
        by_chr = defaultdict(list)
        for idx, row in normal_df.iterrows():
            by_chr[str(row["chromosome_id"])].append(idx)

        target_chr_counts = target_df["chromosome_id"].value_counts().to_dict()

        for chr_id, cnt in target_chr_counts.items():
            pool = by_chr.get(str(chr_id), [])[:]
            random.shuffle(pool)
            chosen_ids.extend(pool[:cnt])

        # 如果没够，再补
        if len(chosen_ids) < n_target:
            remain = list(set(normal_df.index.tolist()) - set(chosen_ids))
            random.shuffle(remain)
            chosen_ids.extend(remain[:(n_target - len(chosen_ids))])

        return chosen_ids[:n_target]

    val_norm_ids = sample_matching_normals(val_ab_df, len(val_ab_df))
    used_norm = set(val_norm_ids)

    test_norm_candidates = normal_df.loc[list(set(normal_df.index.tolist()) - used_norm)]
    tmp_normal_df = normal_df
    normal_df_test_backup = normal_df.copy()

    # 临时替换 normal_df 供抽样
    normal_df_local = test_norm_candidates.copy()

    chosen_ids = []
    by_chr = defaultdict(list)
    for idx, row in normal_df_local.iterrows():
        by_chr[str(row["chromosome_id"])].append(idx)

    test_chr_counts = test_ab_df["chromosome_id"].value_counts().to_dict()
    for chr_id, cnt in test_chr_counts.items():
        pool = by_chr.get(str(chr_id), [])[:]
        random.shuffle(pool)
        chosen_ids.extend(pool[:cnt])

    if len(chosen_ids) < len(test_ab_df):
        remain = list(set(normal_df_local.index.tolist()) - set(chosen_ids))
        random.shuffle(remain)
        chosen_ids.extend(remain[:(len(test_ab_df) - len(chosen_ids))])

    test_norm_ids = chosen_ids[:len(test_ab_df)]

    val_df = pd.concat([val_ab_df, normal_df_test_backup.loc[val_norm_ids]], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = pd.concat([test_ab_df, normal_df_test_backup.loc[test_norm_ids]], axis=0).sample(frac=1, random_state=SEED).reset_index(drop=True)

    used_pair_keys = set(val_df["pair_key"].tolist()) | set(test_df["pair_key"].tolist())
    train_df = pair_df[~pair_df["pair_key"].isin(used_pair_keys)].copy().sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train_df, val_df, test_df


def build_scheme3_pair_level(df, out_dir):
    """
    方案3：打散，但是最小单位就是 pair
    """
    print("\n==============================")
    print("SCHEME 3: pair-level randomized split")
    print("==============================")

    pair_df = build_pair_dataframe(df)

    if BALANCE_VAL_TEST:
        train_df, val_df, test_df = choose_balanced_val_test_from_pair_df(
            pair_df,
            val_abnormal_target=VAL_ABNORMAL_TARGET,
            test_abnormal_target=TEST_ABNORMAL_TARGET,
        )
    else:
        strata = defaultdict(list)
        for idx, row in pair_df.iterrows():
            key = (str(row["chromosome_id"]), int(row["label"]), str(row["abnormal_type"]))
            strata[key].append(idx)

        train_ids, val_ids, test_ids = [], [], []

        for key, idxs in strata.items():
            tr, va, te = stratified_split_indices(idxs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
            train_ids.extend(tr)
            val_ids.extend(va)
            test_ids.extend(te)

        train_df = pair_df.loc[train_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)
        val_df = pair_df.loc[val_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)
        test_df = pair_df.loc[test_ids].sample(frac=1, random_state=SEED).reset_index(drop=True)

    scheme_dir = out_dir / "scheme3_pair_level"
    scheme_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(scheme_dir / "train_pair.csv", index=False)
    val_df.to_csv(scheme_dir / "val_pair.csv", index=False)
    test_df.to_csv(scheme_dir / "test_pair.csv", index=False)

    print_pair_split_stats("train", train_df)
    print_pair_split_stats("val", val_df)
    print_pair_split_stats("test", test_df)


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building single-image dataframe from arrangement_merged...")
    df = build_single_image_dataframe(ROOT_DIR)

    print("\n[Overall Single Image Summary]")
    print("total:", len(df))
    print(df["label"].value_counts(dropna=False))
    print(df["chromosome_id"].value_counts().sort_index())
    print(df[df["label"] == 1]["abnormal_type"].value_counts())

    # 方案1
    build_scheme1_single_image_random(df, OUT_DIR)

    # 方案2
    build_scheme2_grouped_by_pair(df, OUT_DIR)

    # 方案3
    build_scheme3_pair_level(df, OUT_DIR)

    print("\nAll done.")
    print(f"Saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()