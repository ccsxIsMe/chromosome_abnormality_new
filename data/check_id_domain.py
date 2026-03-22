import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torchvision import models, transforms


# =========================
# 0. 配置
# =========================
CSV_PATHS = [
    "/data5/chensx/MyProject/chromosome_abnormality_new/data/train.csv",
    "/data5/chensx/MyProject/chromosome_abnormality_new/data/val.csv",
    "/data5/chensx/MyProject/chromosome_abnormality_new/data/test.csv",
]

OUTPUT_DIR = "./fixed_class_case_classifier_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 你要控制的筛选条件
# -------------------------
# 模式 1：只固定 chromosome_id
USE_CHROMOSOME_FILTER = True
TARGET_CHROMOSOME_ID = 10   # 示例：10

# 模式 2：固定到某一个 label（更干净，推荐也做）
USE_LABEL_FILTER = False
TARGET_LABEL = 0            # 示例：0

# 参与 case_id 分类的 case，至少要有这么多张图
MIN_IMAGES_PER_CASE = 8

# 最多取多少个 case 参与分类
MAX_CASES = 20

# 每个 case 最多用多少张图，避免样本极不均衡
MAX_IMAGES_PER_CASE = 50

# 是否打印更详细信息
VERBOSE = True


# =========================
# 1. 读数据
# =========================
def load_csvs(csv_paths):
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            dfs.append(df)
        else:
            print(f"[Warning] CSV not found: {p}")
    if not dfs:
        raise FileNotFoundError("No valid CSV files found.")
    return pd.concat(dfs, ignore_index=True)


df = load_csvs(CSV_PATHS)

required_cols = ["image_path", "label", "chromosome_id", "case_id", "split"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

df = df[df["image_path"].apply(lambda x: isinstance(x, str) and os.path.exists(x))].copy()
df["case_id"] = df["case_id"].astype(str)

# 类型统一
df["label"] = df["label"].astype(int)
df["chromosome_id"] = df["chromosome_id"].astype(str)

print("Total valid images:", len(df))
print("Total unique cases:", df["case_id"].nunique())


# =========================
# 2. 固定类别筛选
# =========================
filtered_df = df.copy()

filter_desc = []

if USE_CHROMOSOME_FILTER:
    filtered_df = filtered_df[filtered_df["chromosome_id"] == TARGET_CHROMOSOME_ID]
    filter_desc.append(f"chromosome_id={TARGET_CHROMOSOME_ID}")

if USE_LABEL_FILTER:
    filtered_df = filtered_df[filtered_df["label"] == TARGET_LABEL]
    filter_desc.append(f"label={TARGET_LABEL}")

if len(filter_desc) == 0:
    raise ValueError("You must enable at least one filter: chromosome_id or label.")

filter_name = "__".join(filter_desc)
print("\n[Filter condition]:", filter_name)
print("Filtered images:", len(filtered_df))
print("Filtered unique cases:", filtered_df["case_id"].nunique())

if len(filtered_df) == 0:
    raise ValueError("No images left after filtering.")


# =========================
# 3. 选 case：保证每个 case 有足够样本
# =========================
case_counts = filtered_df["case_id"].value_counts()

eligible_cases = case_counts[case_counts >= MIN_IMAGES_PER_CASE].index.tolist()
if len(eligible_cases) < 2:
    raise ValueError(
        f"Not enough eligible cases after filtering. "
        f"Need at least 2 cases with >= {MIN_IMAGES_PER_CASE} images."
    )

selected_cases = eligible_cases[:MAX_CASES]
selected_df = filtered_df[filtered_df["case_id"].isin(selected_cases)].copy()

# 为了避免 case 间样本数差异太大，每个 case 最多抽 MAX_IMAGES_PER_CASE 张
balanced_rows = []
for case_id, sub_df in selected_df.groupby("case_id"):
    n = min(len(sub_df), MAX_IMAGES_PER_CASE)
    balanced_rows.append(sub_df.sample(n=n, random_state=RANDOM_SEED))

selected_df = pd.concat(balanced_rows, ignore_index=True)

print("\nSelected cases:", len(selected_cases))
print("Case image counts after balancing:")
print(selected_df["case_id"].value_counts())

# 保存筛选后的清单
selected_df.to_csv(os.path.join(OUTPUT_DIR, f"filtered_samples__{filter_name}.csv"), index=False)


# =========================
# 4. 特征提取模型：ResNet18
# =========================
class BackboneFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(model.children())[:-1])  # 去掉 fc

    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.flatten(1)
        return feat


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

feature_extractor = BackboneFeatureExtractor().to(DEVICE)
feature_extractor.eval()


def extract_feature(image_path):
    img = Image.open(image_path).convert("L")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feature_extractor(x).cpu().numpy()[0]
    return feat


# =========================
# 5. 提取特征
# =========================
feature_rows = []

for idx, row in selected_df.iterrows():
    try:
        feat = extract_feature(row["image_path"])
        rec = {
            "case_id": row["case_id"],
            "label": row["label"],
            "chromosome_id": row["chromosome_id"],
            "split": row["split"],
            "image_path": row["image_path"],
        }
        for i, v in enumerate(feat):
            rec[f"f_{i}"] = float(v)
        feature_rows.append(rec)
    except Exception as e:
        print(f"[Warning] feature extraction failed on {row['image_path']}: {e}")

feat_df = pd.DataFrame(feature_rows)
feat_df.to_csv(os.path.join(OUTPUT_DIR, f"embedding_features__{filter_name}.csv"), index=False)

feat_cols = [c for c in feat_df.columns if c.startswith("f_")]
X = feat_df[feat_cols].values
y = feat_df["case_id"].values

print("\nFinal samples used:", len(feat_df))
print("Final unique cases:", len(np.unique(y)))

if len(np.unique(y)) < 2:
    raise ValueError("Need at least 2 case_ids for classification.")


# =========================
# 6. 训练 / 测试划分
# =========================
# 注意：这里是图像级随机划分。
# 如果你想更严格，可以做“按病例内图像分层划分”或“重复多次不同随机种子”。
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_SEED,
    stratify=y
)

# 随机猜测的理论准确率
num_cases = len(np.unique(y))
random_acc = 1.0 / num_cases


# =========================
# 7. 简单分类器：Logistic Regression
# =========================
clf = LogisticRegression(
    max_iter=3000,
    multi_class="auto"
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

print("\n===== Fixed-class case_id classifier result =====")
print("Filter:", filter_name)
print("Num cases:", num_cases)
print(f"Random chance accuracy: {random_acc:.4f}")
print(f"Observed accuracy     : {acc:.4f}")
print("\nClassification report:\n")
print(report)

# 保存结果
result_txt = os.path.join(OUTPUT_DIR, f"case_classifier_report__{filter_name}.txt")
with open(result_txt, "w") as f:
    f.write("===== Fixed-class case_id classifier result =====\n")
    f.write(f"Filter: {filter_name}\n")
    f.write(f"Num cases: {num_cases}\n")
    f.write(f"Random chance accuracy: {random_acc:.4f}\n")
    f.write(f"Observed accuracy     : {acc:.4f}\n\n")
    f.write(report)

np.save(os.path.join(OUTPUT_DIR, f"confusion_matrix__{filter_name}.npy"), cm)

print(f"[Saved] {result_txt}")
print(f"[Saved] confusion_matrix__{filter_name}.npy")


# =========================
# 8. 简单判断建议
# =========================
print("\n===== Interpretation hint =====")
if acc <= random_acc * 1.3:
    print("结果接近随机，说明在固定类别后，case_id 不容易被识别。")
    print("这通常意味着之前的 case classifier 结果更多来自类别组成差异，而不是真正的病例域差异。")
elif acc <= random_acc * 2.0:
    print("结果略高于随机，说明存在一定 case-level 可识别信号，但不算很强。")
    print("你可以继续做多个 chromosome_id / label 重复实验，观察是否稳定。")
else:
    print("结果明显高于随机，说明在固定类别后仍存在较强的 case-level 可识别信号。")
    print("这时才更有理由考虑病例域偏移、风格差异或蒸馏/域泛化方法。")