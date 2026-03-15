import pandas as pd
from collections import Counter

base = "/data5/chensx/MyProject/chromosome_abnormality_new/data"

for split in ["train", "val", "test"]:
    path = f"{base}/{split}_pair.csv"
    df = pd.read_csv(path)

    print(f"\n===== {split.upper()} =====")
    print("total pairs:", len(df))
    print("label counts:")
    print(df["label"].value_counts(dropna=False).sort_index())

    print("\nunique case_id:", df["case_id"].nunique())

    print("\nchromosome distribution:")
    print(df["chromosome_id"].value_counts().sort_index())

    print("\nabnormal chromosome distribution:")
    print(df[df["label"] == 1]["chromosome_id"].value_counts().sort_index())

    if "left_single_label" in df.columns and "right_single_label" in df.columns:
        print("\n(left_single_label, right_single_label) counts:")
        print(df.groupby(["left_single_label", "right_single_label"]).size())

    print("\nabnormal pairs per case (top 20):")
    abnormal_per_case = df[df["label"] == 1]["case_id"].value_counts().head(20)
    print(abnormal_per_case)