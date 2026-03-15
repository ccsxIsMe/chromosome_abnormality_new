import re
import pandas as pd


def canonicalize_chromosome_id(raw_id):
    """
    把各种 chromosome_id 规范化为:
    '1' ~ '22', 'X', 'Y', 'UNK'

    示例:
    '01' -> '1'
    '1' -> '1'
    'chr_01' -> '1'
    'dup8' -> '8'
    'x' -> 'X'
    'xOnly' -> 'X'
    'yOnly' -> 'Y'
    """
    if raw_id is None:
        return "UNK"

    s = str(raw_id).strip()

    if s == "":
        return "UNK"

    s_lower = s.lower()

    # X / Y 优先判断
    if "x" in s_lower and "y" not in s_lower:
        return "X"
    if "y" in s_lower and "x" not in s_lower:
        return "Y"

    # 提取数字
    nums = re.findall(r"\d+", s_lower)
    if len(nums) > 0:
        n = int(nums[0])
        if 1 <= n <= 22:
            return str(n)

    return "UNK"


def build_chr_vocab_from_csv(train_csv_path):
    """
    只从 train.csv 构建 vocab，避免信息泄漏。
    返回:
        chr_to_idx: dict
        idx_to_chr: dict
    """
    df = pd.read_csv(train_csv_path)

    if "chromosome_id" not in df.columns:
        raise ValueError("train csv missing column: chromosome_id")

    canon_ids = df["chromosome_id"].apply(canonicalize_chromosome_id).tolist()
    unique_ids = sorted(set(canon_ids), key=lambda x: (x == "UNK", x))

    # 强制把常见类别排序得更直观一点
    ordered = []
    for i in range(1, 23):
        if str(i) in unique_ids:
            ordered.append(str(i))
    if "X" in unique_ids:
        ordered.append("X")
    if "Y" in unique_ids:
        ordered.append("Y")
    if "UNK" in unique_ids:
        ordered.append("UNK")

    chr_to_idx = {c: i for i, c in enumerate(ordered)}
    idx_to_chr = {i: c for c, i in chr_to_idx.items()}

    return chr_to_idx, idx_to_chr