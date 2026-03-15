import os
import re
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ================= 配置区 =================
ROOT_DIR = Path("/data5/chensx/MyProject/UAE/data/arrangement_merged")
OUT_DIR = Path("/data5/chensx/MyProject/chromosome_abnormality_new/data_randomized_balenced")

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

BALANCE_VAL_TEST = True # 开启后，方案1, 2, 3 的 Val/Test 均为 1:1
VAL_ABNORMAL_TARGET = None
TEST_ABNORMAL_TARGET = None
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def normalize_chromosome_token(x: str):
    x = x.upper()
    if x.isdigit(): return str(int(x))
    return x

def parse_folder_name(folder_name: str):
    folder_name = folder_name.lower()
    m = re.match(r"^chr_([0-9]{1,2}|x|y)(?:_(.+))?$", folder_name)
    if not m: return None
    chrom = normalize_chromosome_token(m.group(1))
    suffix = m.group(2)
    return {
        "chromosome_id": chrom,
        "label": 0 if suffix is None else 1,
        "abnormal_type": "normal" if suffix is None else suffix.lower(),
        "folder_name": folder_name,
    }

def parse_filename_info(filename: str):
    stem = Path(filename).stem
    case_id = stem.split(".")[0] if "." in stem else stem.split("_")[0]
    token = stem.split("_")[-1]
    side, chrom, pair_key = "UNK", "UNK", stem
    if len(token) >= 2 and token[-1].upper() in ["L", "R"]:
        side = token[-1].upper()
        chrom = normalize_chromosome_token(token[:-1])
        pair_key = stem[:-1]
    return {"filename": filename, "case_id": case_id, "side": side, "chromosome_from_name": chrom, "pair_key": pair_key}

def build_single_image_dataframe(root_dir: Path):
    rows = []
    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir(): continue
        folder_info = parse_folder_name(subdir.name)
        if folder_info is None: continue
        for img_path in sorted(subdir.glob("*.png")):
            file_info = parse_filename_info(img_path.name)
            rows.append({
                "image_path": str(img_path), "filename": img_path.name,
                "case_id": file_info["case_id"], "pair_key": file_info["pair_key"],
                "side": file_info["side"], "chromosome_id": folder_info["chromosome_id"],
                "label": folder_info["label"], "abnormal_type": folder_info["abnormal_type"]
            })
    return pd.DataFrame(rows)

# --- 核心新增：通用平衡划分逻辑 ---
def balance_split_dataframe(df, label_col="label"):
    """
    通用逻辑：
    1. 将异常样本按比例/目标数分配到 Val 和 Test。
    2. 从正常样本中为 Val 和 Test 各抽取同等数量的样本（尽量匹配 chromosome_id）。
    3. 剩下的所有样本（异常+正常）全部进 Train。
    """
    abnormal_df = df[df[label_col] == 1].copy()
    normal_df = df[df[label_col] == 0].copy()
    total_ab = len(abnormal_df)

    # 确定目标数量
    v_target = VAL_ABNORMAL_TARGET if VAL_ABNORMAL_TARGET else max(1, int(total_ab * VAL_RATIO))
    t_target = TEST_ABNORMAL_TARGET if TEST_ABNORMAL_TARGET else max(1, int(total_ab * TEST_RATIO))
    v_target = min(v_target, total_ab - 1)
    t_target = min(t_target, total_ab - v_target)

    # 1. 划分异常样本 (分层打散)
    ab_indices = abnormal_df.index.tolist()
    random.shuffle(ab_indices)
    val_ab_ids = ab_indices[:v_target]
    test_ab_ids = ab_indices[v_target:v_target + t_target]
    train_ab_ids = ab_indices[v_target + t_target:]

    # 2. 划分正常样本 (匹配染色体分布)
    def sample_normals(target_ab_df, count):
        chosen_ids = []
        # 按染色体 ID 归类正常样本索引
        norm_pool = defaultdict(list)
        for idx, row in normal_df.iterrows():
            norm_pool[str(row["chromosome_id"])].append(idx)
        
        # 尽量按异常样本的染色体比例抽
        chr_counts = target_ab_df["chromosome_id"].value_counts().to_dict()
        for c_id, c_num in chr_counts.items():
            pool = norm_pool.get(str(c_id), [])
            random.shuffle(pool)
            picked = pool[:c_num]
            chosen_ids.extend(picked)
            # 从全局 pool 中移除已选的
            for p in picked: normal_df.drop(p, inplace=True)
            
        # 如果还没凑够（比如某个染色体正常样本不够了），从剩余 normal 里随机补
        if len(chosen_ids) < count:
            remain_ids = normal_df.index.tolist()
            random.shuffle(remain_ids)
            extra = remain_ids[:(count - len(chosen_ids))]
            chosen_ids.extend(extra)
            for e in extra: normal_df.drop(e, inplace=True)
        return chosen_ids

    val_norm_ids = sample_normals(abnormal_df.loc[val_ab_ids], len(val_ab_ids))
    test_norm_ids = sample_normals(abnormal_df.loc[test_ab_ids], len(test_ab_ids))
    
    # 3. 组合结果
    val_df = pd.concat([df.loc[val_ab_ids], df.loc[val_norm_ids]]).sample(frac=1)
    test_df = pd.concat([df.loc[test_ab_ids], df.loc[test_norm_ids]]).sample(frac=1)
    
    # 训练集：剩余的所有
    used_indices = set(val_ab_ids) | set(test_ab_ids) | set(val_norm_ids) | set(test_norm_ids)
    train_df = df.loc[~df.index.isin(used_indices)].sample(frac=1)
    
    return train_df, val_df, test_df

# --- 方案 1 ---
def build_scheme1_single_image_random(df, out_dir):
    print("\n[Scheme 1] Processing...")
    if BALANCE_VAL_TEST:
        train_df, val_df, test_df = balance_split_dataframe(df)
    else:
        # 走原有的随机分层逻辑... (略)
        pass 
    
    save_path = out_dir / "scheme1_single_image_random"
    save_path.mkdir(parents=True, exist_ok=True)
    for name, d in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        d.to_csv(save_path / f"{name}.csv", index=False)
    print(f"Saved Scheme 1. Val size: {len(val_df)}, Test size: {len(test_df)}")

# --- 方案 2 ---
def build_scheme2_grouped_by_pair(df, out_dir):
    print("\n[Scheme 2] Processing...")
    # 先聚合 Pair
    group_rows = []
    for pk, grp in df.groupby("pair_key"):
        group_rows.append({
            "pair_key": pk, "chromosome_id": grp["chromosome_id"].iloc[0],
            "label": grp["label"].max(), "member_indices": grp.index.tolist()
        })
    group_df = pd.DataFrame(group_rows)

    if BALANCE_VAL_TEST:
        train_g, val_g, test_g = balance_split_dataframe(group_df)
    else:
        # 原逻辑... (略)
        pass

    def expand(g_df):
        all_idx = []
        for _, r in g_df.iterrows(): all_idx.extend(r["member_indices"])
        return df.loc[all_idx].sample(frac=1)

    train_df, val_df, test_df = expand(train_g), expand(val_g), expand(test_g)
    
    save_path = out_dir / "scheme2_grouped_by_pair"
    save_path.mkdir(parents=True, exist_ok=True)
    for name, d in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        d.to_csv(save_path / f"{name}.csv", index=False)
    print(f"Saved Scheme 2. Val size: {len(val_df)} imgs, Test size: {len(test_df)} imgs")

# --- 方案 3 ---
def build_scheme3_pair_level(df, out_dir):
    print("\n[Scheme 3] Processing...")
    # 构建 Pair 行
    pair_rows = []
    for pk, grp in df.groupby("pair_key"):
        L = grp[grp["side"]=="L"]; R = grp[grp["side"]=="R"]
        if len(L)>0 and len(R)>0:
            l_row = L.iloc[0]; r_row = R.iloc[0]
            pair_rows.append({
                "left_path": l_row["image_path"], "right_path": r_row["image_path"],
                "label": max(l_row["label"], r_row["label"]), "chromosome_id": l_row["chromosome_id"],
                "pair_key": pk, "abnormal_type": l_row["abnormal_type"] if l_row["label"]==1 else r_row["abnormal_type"]
            })
    pair_df = pd.DataFrame(pair_rows)

    if BALANCE_VAL_TEST:
        train_df, val_df, test_df = balance_split_dataframe(pair_df)
    else:
        # 原逻辑... (略)
        pass

    save_path = out_dir / "scheme3_pair_level"
    save_path.mkdir(parents=True, exist_ok=True)
    for name, d in zip(['train_pair', 'val_pair', 'test_pair'], [train_df, val_df, test_df]):
        d.to_csv(save_path / f"{name}.csv", index=False)
    print(f"Saved Scheme 3. Val size: {len(val_df)} pairs, Test size: {len(test_df)} pairs")

def main():
    set_seed(SEED)
    df = build_single_image_dataframe(ROOT_DIR)
    build_scheme1_single_image_random(df, OUT_DIR)
    build_scheme2_grouped_by_pair(df, OUT_DIR)
    build_scheme3_pair_level(df, OUT_DIR)
    print("\nAll done!")

if __name__ == "__main__":
    main()