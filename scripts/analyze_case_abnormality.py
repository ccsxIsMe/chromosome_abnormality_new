import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PAIR_CSVS = [
    Path("data_randomized/scheme3_pair_level/train_pair.csv"),
    Path("data_randomized/scheme3_pair_level/val_pair.csv"),
    Path("data_randomized/scheme3_pair_level/test_pair.csv"),
]


def load_pair_csvs(csv_paths):
    frames = []

    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        df["source_csv"] = str(path)

        required_cols = {"label", "chromosome_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        if "case_id" not in df.columns:
            if "pair_key" in df.columns:
                df["case_id"] = df["pair_key"].astype(str).str.split(".", n=1).str[0]
            else:
                raise ValueError(f"{path} missing case_id and pair_key; cannot aggregate by case")

        if "abnormal_type" not in df.columns:
            df["abnormal_type"] = "unknown"

        if "split" not in df.columns:
            split_name = path.stem.replace("_pair", "")
            df["split"] = split_name

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["case_id"] = merged["case_id"].astype(str)
    merged["chromosome_id"] = merged["chromosome_id"].astype(str)
    merged["abnormal_type"] = merged["abnormal_type"].fillna("unknown").astype(str)
    merged["split"] = merged["split"].astype(str)
    return merged


def summarize_dataset(df):
    total_cases = df["case_id"].nunique()
    abnormal_df = df[df["label"] == 1].copy()
    abnormal_cases = abnormal_df["case_id"].nunique()

    print("===== DATASET SUMMARY =====")
    print(f"total_rows: {len(df)}")
    print(f"total_cases: {total_cases}")
    print(f"abnormal_rows: {len(abnormal_df)}")
    print(f"abnormal_cases: {abnormal_cases}")
    print(f"normal_only_cases: {total_cases - abnormal_cases}")

    print("\nrows per split:")
    print(df["split"].value_counts().sort_index().to_string())

    print("\nabnormal cases per split:")
    split_case_df = abnormal_df.groupby("split")["case_id"].nunique().sort_index()
    print(split_case_df.to_string())


def summarize_abnormal_cases(df):
    abnormal_df = df[df["label"] == 1].copy()
    if abnormal_df.empty:
        print("\nNo abnormal rows found.")
        return

    case_summary = (
        abnormal_df.groupby("case_id")
        .agg(
            abnormal_pair_count=("label", "size"),
            abnormal_chromosome_count=("chromosome_id", "nunique"),
            abnormal_type_count=("abnormal_type", "nunique"),
            splits=("split", lambda s: ",".join(sorted(set(s)))),
            abnormal_chromosomes=("chromosome_id", lambda s: ",".join(sorted(set(map(str, s))))),
            abnormal_types=("abnormal_type", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .sort_values(
            by=["abnormal_pair_count", "abnormal_chromosome_count", "case_id"],
            ascending=[False, False, True],
        )
    )

    print("\n===== ABNORMAL CASE DISTRIBUTION =====")
    print("abnormal_pair_count distribution:")
    print(case_summary["abnormal_pair_count"].value_counts().sort_index().to_string())

    print("\nabnormal_chromosome_count distribution:")
    print(case_summary["abnormal_chromosome_count"].value_counts().sort_index().to_string())

    print("\nTop 20 abnormal cases:")
    print(case_summary.head(20).to_string())


def build_category_table(abnormal_df, category_cols):
    dedup = abnormal_df[["case_id", "split", *category_cols]].drop_duplicates()

    table = (
        dedup.groupby(category_cols)
        .agg(
            abnormal_case_count=("case_id", "nunique"),
            abnormal_split_count=("split", "nunique"),
            abnormal_splits=("split", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
        .sort_values(
            by=["abnormal_case_count", "abnormal_split_count", *category_cols],
            ascending=[True, True, *([True] * len(category_cols))],
        )
    )
    return table


def print_category_report(abnormal_df, category_cols, label, rare_threshold):
    table = build_category_table(abnormal_df, category_cols)

    print(f"\n===== {label} =====")
    print(f"total_categories: {len(table)}")
    print("case-count distribution:")
    print(table["abnormal_case_count"].value_counts().sort_index().to_string())

    rare = table[table["abnormal_case_count"] <= rare_threshold].copy()
    print(f"\nrare categories (abnormal_case_count <= {rare_threshold}): {len(rare)}")
    if rare.empty:
        print("None")
    else:
        print(rare.to_string(index=False))

    return table, rare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=[str(p) for p in DEFAULT_PAIR_CSVS],
        help="Pair-level CSVs to analyze",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=2,
        help="Flag categories whose unique abnormal case count is <= this threshold",
    )
    args = parser.parse_args()

    df = load_pair_csvs(args.csvs)
    summarize_dataset(df)
    summarize_abnormal_cases(df)

    abnormal_df = df[df["label"] == 1].copy()
    if abnormal_df.empty:
        return

    print_category_report(
        abnormal_df,
        category_cols=["chromosome_id"],
        label="ABNORMAL CASES BY CHROMOSOME",
        rare_threshold=args.rare_threshold,
    )

    if "abnormal_type" in abnormal_df.columns:
        print_category_report(
            abnormal_df,
            category_cols=["abnormal_type"],
            label="ABNORMAL CASES BY ABNORMAL TYPE",
            rare_threshold=args.rare_threshold,
        )

        print_category_report(
            abnormal_df,
            category_cols=["chromosome_id", "abnormal_type"],
            label="ABNORMAL CASES BY CHROMOSOME + TYPE",
            rare_threshold=args.rare_threshold,
        )


if __name__ == "__main__":
    main()
