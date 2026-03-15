import pandas as pd


def check_csv(csv_path):
    df = pd.read_csv(csv_path)

    mismatch = df[df["case_id"].astype(str) != df["case_dir"].astype(str)]

    print(f"\nChecking {csv_path}")
    print(f"Total samples: {len(df)}")
    print(f"Mismatched case_id/case_dir: {len(mismatch)}")

    if len(mismatch) > 0:
        print(mismatch[["filename", "case_id", "case_dir"]].head(20))

    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nChromosome distribution:")
    print(df["chromosome_id"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    for path in ["/data5/chensx/MyProject/chromosome_abnormality_new/data/train.csv", "/data5/chensx/MyProject/chromosome_abnormality_new/data/val.csv", "/data5/chensx/MyProject/chromosome_abnormality_new/data/test.csv"]:
        check_csv(path)