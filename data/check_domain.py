import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

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
OUTPUT_DIR = "./domain_shift_check_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 每个 case 在 montage 里最多抽多少张
MONTAGE_IMAGES_PER_CASE = 8
# 最多展示多少个 case
MONTAGE_NUM_CASES = 12

# embedding 可视化时每个 case 最多取多少张
EMBED_PER_CASE = 30
# 至少多少张图的 case 才参与 patient-level classifier
MIN_IMAGES_PER_CASE_FOR_CASE_CLS = 10
# 最多选多少个 case 参与 patient-level classifier
MAX_CASES_FOR_CASE_CLS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1. 读取数据
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
    df = pd.concat(dfs, ignore_index=True)
    return df


df = load_csvs(CSV_PATHS)

required_cols = ["image_path", "label", "chromosome_id", "case_id", "split"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

df = df[df["image_path"].apply(lambda x: isinstance(x, str) and os.path.exists(x))].copy()
df["case_id"] = df["case_id"].astype(str)

print("Total valid images:", len(df))
print("Total unique cases:", df["case_id"].nunique())
print("Images per split:")
print(df["split"].value_counts())
print("Top case counts:")
print(df["case_id"].value_counts().head(10))


# =========================
# 2. 基础图像统计特征
# =========================
def load_gray_image(image_path):
    img = Image.open(image_path).convert("L")
    arr = np.array(img).astype(np.float32)
    return arr

def compute_edge_density(arr):
    # 简单梯度近似
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    g = np.pad(gx, ((0, 0), (0, 1)), mode="constant") + np.pad(gy, ((0, 1), (0, 0)), mode="constant")
    threshold = np.percentile(g, 75)
    edge_density = (g > threshold).mean()
    return float(edge_density)

def compute_contrast(arr):
    # RMS contrast
    return float(arr.std())

def compute_histogram(arr, bins=32):
    hist, _ = np.histogram(arr, bins=bins, range=(0, 255), density=True)
    return hist

def compute_image_stats(image_path):
    arr = load_gray_image(image_path)
    h, w = arr.shape[:2]
    stats = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "height": int(h),
        "width": int(w),
        "aspect_ratio": float(w / h),
        "edge_density": compute_edge_density(arr),
        "contrast": compute_contrast(arr),
    }
    hist = compute_histogram(arr, bins=32)
    for i, v in enumerate(hist):
        stats[f"hist_{i}"] = float(v)
    return stats


# =========================
# 3. 检查 1：montage 可视化
# =========================
def plot_case_montage(df, output_path, num_cases=12, images_per_case=8):
    case_counts = df["case_id"].value_counts()
    candidate_cases = case_counts.index.tolist()
    sampled_cases = random.sample(candidate_cases, min(num_cases, len(candidate_cases)))

    fig, axes = plt.subplots(len(sampled_cases), images_per_case, figsize=(2 * images_per_case, 2 * len(sampled_cases)))
    if len(sampled_cases) == 1:
        axes = np.array([axes])

    for row_idx, case_id in enumerate(sampled_cases):
        case_df = df[df["case_id"] == case_id]
        sampled_rows = case_df.sample(min(images_per_case, len(case_df)), random_state=RANDOM_SEED)
        sampled_rows = sampled_rows.reset_index(drop=True)

        for col_idx in range(images_per_case):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx < len(sampled_rows):
                img_path = sampled_rows.loc[col_idx, "image_path"]
                label = sampled_rows.loc[col_idx, "label"]
                chrom = sampled_rows.loc[col_idx, "chromosome_id"]
                split = sampled_rows.loc[col_idx, "split"]

                arr = load_gray_image(img_path)
                ax.imshow(arr, cmap="gray")
                ax.set_title(f"case={case_id}\nchr={chrom}, y={label}, {split}", fontsize=8)
            else:
                ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Saved] Montage: {output_path}")


plot_case_montage(
    df,
    output_path=os.path.join(OUTPUT_DIR, "case_montage.png"),
    num_cases=MONTAGE_NUM_CASES,
    images_per_case=MONTAGE_IMAGES_PER_CASE,
)


# =========================
# 4. 检查 2：统计特征
# =========================
stats_records = []
for idx, row in df.iterrows():
    try:
        stats = compute_image_stats(row["image_path"])
        stats["case_id"] = row["case_id"]
        stats["label"] = row["label"]
        stats["chromosome_id"] = row["chromosome_id"]
        stats["split"] = row["split"]
        stats_records.append(stats)
    except Exception as e:
        print(f"[Warning] failed on {row['image_path']}: {e}")

stats_df = pd.DataFrame(stats_records)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "image_stats.csv"), index=False)

case_stats = stats_df.groupby("case_id").agg({
    "mean": ["mean", "std"],
    "std": ["mean", "std"],
    "edge_density": ["mean", "std"],
    "contrast": ["mean", "std"],
    "aspect_ratio": ["mean", "std"],
})
case_stats.to_csv(os.path.join(OUTPUT_DIR, "case_level_stats.csv"))

print("[Saved] image_stats.csv")
print("[Saved] case_level_stats.csv")

# 画几个分布图
for feat in ["mean", "std", "edge_density", "contrast", "aspect_ratio"]:
    plt.figure(figsize=(8, 5))
    stats_df.boxplot(column=feat, by="case_id", rot=90)
    plt.title(f"{feat} by case_id")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{feat}_by_case.png"), dpi=200)
    plt.close()

print("[Saved] boxplots for basic features")


# =========================
# 5. 检查 3：embedding 可视化
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


# 每个 case 采样少量图片做 embedding 可视化
embed_rows = []
for case_id, case_df in df.groupby("case_id"):
    sampled = case_df.sample(min(EMBED_PER_CASE, len(case_df)), random_state=RANDOM_SEED)
    for _, row in sampled.iterrows():
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
            embed_rows.append(rec)
        except Exception as e:
            print(f"[Warning] feature extract failed on {row['image_path']}: {e}")

embed_df = pd.DataFrame(embed_rows)
embed_df.to_csv(os.path.join(OUTPUT_DIR, "embedding_features.csv"), index=False)
print("[Saved] embedding_features.csv")

feat_cols = [c for c in embed_df.columns if c.startswith("f_")]
X = embed_df[feat_cols].values

# 先 PCA 再 t-SNE，稳定些
if X.shape[0] >= 10:
    X_pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=RANDOM_SEED).fit_transform(X)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, max(5, X.shape[0] // 10)))
    X_2d = tsne.fit_transform(X_pca)

    plt.figure(figsize=(8, 6))
    unique_cases = embed_df["case_id"].unique().tolist()
    color_map = {cid: i for i, cid in enumerate(unique_cases)}
    colors = [color_map[cid] for cid in embed_df["case_id"]]
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=10, alpha=0.8)
    plt.title("t-SNE of image embeddings colored by case_id")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_by_case.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    unique_chr = sorted(embed_df["chromosome_id"].unique().tolist())
    color_map_chr = {cid: i for i, cid in enumerate(unique_chr)}
    colors_chr = [color_map_chr[cid] for cid in embed_df["chromosome_id"]]
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_chr, s=10, alpha=0.8)
    plt.title("t-SNE of image embeddings colored by chromosome_id")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_by_chromosome.png"), dpi=200)
    plt.close()

    print("[Saved] tsne_by_case.png")
    print("[Saved] tsne_by_chromosome.png")


# =========================
# 6. 检查 4：patient-level classifier
# =========================
case_counts = df["case_id"].value_counts()
eligible_cases = case_counts[case_counts >= MIN_IMAGES_PER_CASE_FOR_CASE_CLS].index.tolist()
selected_cases = eligible_cases[:MAX_CASES_FOR_CASE_CLS]

case_cls_df = df[df["case_id"].isin(selected_cases)].copy()

print("Selected cases for case_id classifier:", len(selected_cases))
print("Case counts:")
print(case_cls_df["case_id"].value_counts())

case_embed_rows = []
for _, row in case_cls_df.iterrows():
    try:
        feat = extract_feature(row["image_path"])
        rec = {"case_id": row["case_id"]}
        for i, v in enumerate(feat):
            rec[f"f_{i}"] = float(v)
        case_embed_rows.append(rec)
    except Exception as e:
        print(f"[Warning] case-cls feature extract failed on {row['image_path']}: {e}")

case_embed_df = pd.DataFrame(case_embed_rows)
feat_cols_case = [c for c in case_embed_df.columns if c.startswith("f_")]
X_case = case_embed_df[feat_cols_case].values
y_case = case_embed_df["case_id"].values

if len(np.unique(y_case)) >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X_case, y_case, test_size=0.3, random_state=RANDOM_SEED, stratify=y_case
    )

    clf = LogisticRegression(max_iter=3000, multi_class="auto")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    with open(os.path.join(OUTPUT_DIR, "case_classifier_report.txt"), "w") as f:
        f.write(f"case_id classification accuracy: {acc:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))

    print(f"[Saved] case_classifier_report.txt, acc={acc:.4f}")
else:
    print("[Info] Not enough unique cases for case classifier.")


print("\nAll done. Outputs saved to:", OUTPUT_DIR)