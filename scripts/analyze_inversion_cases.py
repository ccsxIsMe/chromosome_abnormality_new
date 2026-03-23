import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PAIR_CSVS = [
    Path("data/train_pair.csv"),
    Path("data/val_pair.csv"),
    Path("data/test_pair.csv"),
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

        if "split" not in df.columns:
            df["split"] = path.stem.replace("_pair", "")

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["case_id"] = merged["case_id"].astype(str)
    merged["chromosome_id"] = merged["chromosome_id"].astype(str)
    merged["split"] = merged["split"].astype(str)
    return merged


def _normalize_abnormal_type(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"normal", "unknown", "nan", "none"}:
        return None
    return text


def build_inversion_type(row):
    abnormal_type = _normalize_abnormal_type(row.get("abnormal_type"))
    chromosome_id = str(row["chromosome_id"]).strip()

    if abnormal_type is None:
        abnormal_type = "inversion"

    return f"{abnormal_type}_chr{chromosome_id}"


def collect_abnormal_rows(df):
    abnormal_df = df[df["label"] == 1].copy()
    if abnormal_df.empty:
        return abnormal_df

    abnormal_df["inversion_type"] = abnormal_df.apply(build_inversion_type, axis=1)
    return abnormal_df


def build_inversion_type_table(abnormal_df):
    dedup = abnormal_df[["case_id", "split", "inversion_type"]].drop_duplicates()

    table = (
        dedup.groupby("inversion_type")
        .agg(
            case_count=("case_id", "nunique"),
            split_count=("split", "nunique"),
            splits=("split", lambda s: ",".join(sorted(set(map(str, s))))),
            case_ids=("case_id", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(by=["case_count", "inversion_type"], ascending=[False, True])
    )
    return table


def build_case_table(abnormal_df):
    dedup = abnormal_df[["case_id", "split", "inversion_type", "chromosome_id"]].drop_duplicates()

    case_table = (
        dedup.groupby("case_id")
        .agg(
            inversion_type_count=("inversion_type", "nunique"),
            chromosome_count=("chromosome_id", "nunique"),
            split_count=("split", "nunique"),
            splits=("split", lambda s: ",".join(sorted(set(map(str, s))))),
            inversion_types=("inversion_type", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(
            by=["inversion_type_count", "chromosome_count", "case_id"],
            ascending=[False, False, True],
        )
    )

    pair_counts = (
        abnormal_df.groupby("case_id")
        .agg(abnormal_pair_count=("label", "size"))
        .reset_index()
    )
    case_table = case_table.merge(pair_counts, on="case_id", how="left")

    ordered_cols = [
        "case_id",
        "abnormal_pair_count",
        "inversion_type_count",
        "chromosome_count",
        "split_count",
        "splits",
        "inversion_types",
    ]
    return case_table[ordered_cols]


def print_summary(df, abnormal_df, inversion_type_table, case_table):
    print("===== INVERSION DATASET SUMMARY =====")
    print(f"total_rows: {len(df)}")
    print(f"total_cases: {df['case_id'].nunique()}")
    print(f"abnormal_rows: {len(abnormal_df)}")
    print(f"abnormal_cases: {abnormal_df['case_id'].nunique()}")
    print(f"unique_inversion_types: {len(inversion_type_table)}")

    print("\n===== CASE COUNT PER INVERSION TYPE =====")
    if inversion_type_table.empty:
        print("No inversion rows found.")
    else:
        print(inversion_type_table.to_string(index=False))

    print("\n===== INVERSION TYPE COUNT PER CASE =====")
    if case_table.empty:
        print("No abnormal cases found.")
    else:
        distribution = case_table["inversion_type_count"].value_counts().sort_index()
        print("distribution:")
        print(distribution.to_string())
        print("\ncase details:")
        print(case_table.to_string(index=False))


def maybe_save_tables(output_dir, inversion_type_table, case_table):
    if output_dir is None:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inversion_type_csv = output_path / "inversion_type_case_counts.csv"
    case_table_csv = output_path / "case_inversion_summary.csv"

    inversion_type_table.to_csv(inversion_type_csv, index=False, encoding="utf-8-sig")
    case_table.to_csv(case_table_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved inversion type table to: {inversion_type_csv}")
    print(f"Saved case summary table to: {case_table_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inversion-type case counts from pair-level CSV files."
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=[str(p) for p in DEFAULT_PAIR_CSVS],
        help="One or more pair-level CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save CSV summaries",
    )
    args = parser.parse_args()

    df = load_pair_csvs(args.csvs)
    abnormal_df = collect_abnormal_rows(df)

    inversion_type_table = build_inversion_type_table(abnormal_df)
    case_table = build_case_table(abnormal_df)

    print_summary(df, abnormal_df, inversion_type_table, case_table)
    maybe_save_tables(args.output_dir, inversion_type_table, case_table)


if __name__ == "__main__":
    main()
