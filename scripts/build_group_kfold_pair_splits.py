import argparse
import random
from collections import Counter
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
        if "case_id" not in df.columns:
            if "pair_key" not in df.columns:
                raise ValueError(f"{path} missing both case_id and pair_key")
            df["case_id"] = df["pair_key"].astype(str).str.split(".", n=1).str[0]

        if "abnormal_type" not in df.columns:
            df["abnormal_type"] = "unknown"

        df["case_id"] = df["case_id"].astype(str)
        df["chromosome_id"] = df["chromosome_id"].astype(str)
        df["abnormal_type"] = df["abnormal_type"].fillna("unknown").astype(str)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged


def build_case_records(df):
    records = []

    for case_id, case_df in df.groupby("case_id"):
        abnormal_df = case_df[case_df["label"] == 1]
        chr_counter = Counter(abnormal_df["chromosome_id"].astype(str).tolist())
        chr_type_counter = Counter(
            [
                f"{row.chromosome_id}::{row.abnormal_type}"
                for row in abnormal_df[["chromosome_id", "abnormal_type"]].itertuples(index=False)
            ]
        )
        type_counter = Counter(abnormal_df["abnormal_type"].astype(str).tolist())

        records.append(
            {
                "case_id": str(case_id),
                "total_pairs": int(len(case_df)),
                "abnormal_pairs": int(len(abnormal_df)),
                "abnormal_chr_count": int(len(chr_counter)),
                "abnormal_type_count": int(len(type_counter)),
                "chr_counter": chr_counter,
                "chr_type_counter": chr_type_counter,
                "type_counter": type_counter,
            }
        )

    return records


def count_abnormal_cases_by_category(df, category_cols):
    abnormal_df = df[df["label"] == 1]
    if abnormal_df.empty:
        return pd.DataFrame(columns=[*category_cols, "abnormal_case_count"])

    dedup = abnormal_df[["case_id", *category_cols]].drop_duplicates()
    table = (
        dedup.groupby(category_cols)["case_id"]
        .nunique()
        .reset_index(name="abnormal_case_count")
        .sort_values(by=["abnormal_case_count", *category_cols], ascending=True)
        .reset_index(drop=True)
    )
    return table


def build_empty_fold():
    return {
        "cases": [],
        "total_pairs": 0,
        "abnormal_pairs": 0,
        "chr_counter": Counter(),
        "chr_type_counter": Counter(),
    }


def clone_fold(fold):
    return {
        "cases": fold["cases"][:],
        "total_pairs": fold["total_pairs"],
        "abnormal_pairs": fold["abnormal_pairs"],
        "chr_counter": Counter(fold["chr_counter"]),
        "chr_type_counter": Counter(fold["chr_type_counter"]),
    }


def add_case_to_fold(fold, case_record):
    fold["cases"].append(case_record["case_id"])
    fold["total_pairs"] += case_record["total_pairs"]
    fold["abnormal_pairs"] += case_record["abnormal_pairs"]
    fold["chr_counter"].update(case_record["chr_counter"])
    fold["chr_type_counter"].update(case_record["chr_type_counter"])


def score_folds(folds, target_total_pairs, target_abnormal_pairs, target_chr_counts, target_chr_type_counts):
    score = 0.0

    for fold in folds:
        score += 1.0 * ((fold["total_pairs"] - target_total_pairs) / max(target_total_pairs, 1.0)) ** 2
        score += 2.5 * ((fold["abnormal_pairs"] - target_abnormal_pairs) / max(target_abnormal_pairs, 1.0)) ** 2

    for category, target in target_chr_counts.items():
        denom = max(target, 1.0)
        for fold in folds:
            score += 1.5 * ((fold["chr_counter"].get(category, 0) - target) / denom) ** 2

    for category, target in target_chr_type_counts.items():
        denom = max(target, 1.0)
        for fold in folds:
            score += 1.0 * ((fold["chr_type_counter"].get(category, 0) - target) / denom) ** 2

    return score


def assign_cases_to_folds(case_records, num_folds, seed):
    rng = random.Random(seed)
    records = case_records[:]
    rng.shuffle(records)
    records.sort(
        key=lambda r: (
            r["abnormal_pairs"],
            r["abnormal_chr_count"],
            r["abnormal_type_count"],
            r["total_pairs"],
        ),
        reverse=True,
    )

    total_pairs = sum(r["total_pairs"] for r in records)
    total_abnormal_pairs = sum(r["abnormal_pairs"] for r in records)
    total_chr_counts = Counter()
    total_chr_type_counts = Counter()
    for record in records:
        total_chr_counts.update(record["chr_counter"])
        total_chr_type_counts.update(record["chr_type_counter"])

    target_total_pairs = total_pairs / num_folds
    target_abnormal_pairs = total_abnormal_pairs / num_folds
    target_chr_counts = {k: v / num_folds for k, v in total_chr_counts.items()}
    target_chr_type_counts = {k: v / num_folds for k, v in total_chr_type_counts.items()}

    folds = [build_empty_fold() for _ in range(num_folds)]

    for record in records:
        best_fold_idx = None
        best_score = None

        for fold_idx in range(num_folds):
            trial_folds = [clone_fold(fold) for fold in folds]
            add_case_to_fold(trial_folds[fold_idx], record)

            trial_score = score_folds(
                trial_folds,
                target_total_pairs=target_total_pairs,
                target_abnormal_pairs=target_abnormal_pairs,
                target_chr_counts=target_chr_counts,
                target_chr_type_counts=target_chr_type_counts,
            )

            if best_score is None or trial_score < best_score:
                best_score = trial_score
                best_fold_idx = fold_idx

        add_case_to_fold(folds[best_fold_idx], record)

    return folds


def summarize_fold_assignments(df, folds):
    print("\n===== FOLD SUMMARY =====")
    for fold_idx, fold in enumerate(folds):
        fold_df = df[df["case_id"].isin(fold["cases"])]
        abnormal_df = fold_df[fold_df["label"] == 1]
        abnormal_case_count = abnormal_df["case_id"].nunique()
        abnormal_chr = ",".join(sorted(set(abnormal_df["chromosome_id"].astype(str).tolist())))

        print(
            f"fold={fold_idx} "
            f"cases={fold_df['case_id'].nunique()} "
            f"pairs={len(fold_df)} "
            f"abnormal_pairs={len(abnormal_df)} "
            f"abnormal_cases={abnormal_case_count} "
            f"abnormal_chromosomes=[{abnormal_chr}]"
        )


def save_fold_outputs(df, folds, out_dir, with_val):
    out_dir.mkdir(parents=True, exist_ok=True)

    assignment_rows = []
    for fold_idx, fold in enumerate(folds):
        for case_id in fold["cases"]:
            case_df = df[df["case_id"] == case_id]
            abnormal_df = case_df[case_df["label"] == 1]
            assignment_rows.append(
                {
                    "case_id": case_id,
                    "fold_id": fold_idx,
                    "pair_count": len(case_df),
                    "abnormal_pair_count": len(abnormal_df),
                    "abnormal_chromosomes": ",".join(sorted(set(abnormal_df["chromosome_id"].astype(str).tolist()))),
                    "abnormal_types": ",".join(sorted(set(abnormal_df["abnormal_type"].astype(str).tolist()))),
                }
            )

    pd.DataFrame(assignment_rows).sort_values(by=["fold_id", "case_id"]).to_csv(
        out_dir / "fold_assignments.csv",
        index=False,
    )

    num_folds = len(folds)
    for test_fold in range(num_folds):
        fold_dir = out_dir / f"fold_{test_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if with_val:
            val_fold = (test_fold + 1) % num_folds
            train_folds = [i for i in range(num_folds) if i not in {test_fold, val_fold}]
            val_case_ids = folds[val_fold]["cases"]
            val_df = df[df["case_id"].isin(val_case_ids)].copy().reset_index(drop=True)
            val_df.to_csv(fold_dir / "val_pair.csv", index=False)
        else:
            train_folds = [i for i in range(num_folds) if i != test_fold]

        test_case_ids = folds[test_fold]["cases"]
        train_case_ids = []
        for fold_idx in train_folds:
            train_case_ids.extend(folds[fold_idx]["cases"])

        train_df = df[df["case_id"].isin(train_case_ids)].copy().reset_index(drop=True)
        test_df = df[df["case_id"].isin(test_case_ids)].copy().reset_index(drop=True)

        train_df.to_csv(fold_dir / "train_pair.csv", index=False)
        test_df.to_csv(fold_dir / "test_pair.csv", index=False)


def print_impossible_category_report(df, num_folds):
    print("\n===== IMPOSSIBLE-FOR-FULL-COVERAGE REPORT =====")

    for category_cols, name in [
        (["chromosome_id"], "chromosome"),
        (["abnormal_type"], "abnormal_type"),
        (["chromosome_id", "abnormal_type"], "chromosome+abnormal_type"),
    ]:
        table = count_abnormal_cases_by_category(df, category_cols)
        rare = table[table["abnormal_case_count"] < num_folds]

        print(f"\n{name}: total_categories={len(table)} categories_with_case_count_lt_{num_folds}={len(rare)}")
        if rare.empty:
            print("  None")
        else:
            print(rare.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=[str(p) for p in DEFAULT_PAIR_CSVS],
        help="Pair-level CSVs to merge before building grouped k-fold splits",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_cv/group_kfold_pair",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Only export train_pair.csv and test_pair.csv for each fold",
    )
    args = parser.parse_args()

    if args.num_folds < 2:
        raise ValueError("--num-folds must be >= 2")
    if not args.no_val and args.num_folds < 3:
        raise ValueError("Need at least 3 folds to export separate train/val/test files")

    df = load_pair_csvs(args.csvs)
    case_records = build_case_records(df)
    folds = assign_cases_to_folds(case_records, num_folds=args.num_folds, seed=args.seed)

    print(f"total_cases={df['case_id'].nunique()} total_pairs={len(df)}")
    print_impossible_category_report(df, num_folds=args.num_folds)
    summarize_fold_assignments(df, folds)

    save_fold_outputs(
        df,
        folds,
        out_dir=Path(args.out_dir),
        with_val=not args.no_val,
    )
    print(f"\nSaved grouped k-fold splits to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
