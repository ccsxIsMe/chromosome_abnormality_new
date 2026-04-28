import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite pair CSV left/right image paths to a new image root while preserving "
            "the original split, labels, case isolation, and metadata."
        )
    )
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument(
        "--source_root",
        default="",
        help="Required only when mode=prefix_replace.",
    )
    parser.add_argument("--target_root", required=True)
    parser.add_argument("--output_csv_dir", required=True)
    parser.add_argument(
        "--mode",
        default="prefix_replace",
        choices=["prefix_replace", "splits_case_layout"],
        help=(
            "prefix_replace: replace source_root prefix with target_root prefix. "
            "splits_case_layout: build paths as <target_root>/<split>/<case_dir>/<normal|abnormal>/<filename>."
        ),
    )
    parser.add_argument(
        "--case_dir_column",
        default="case_dir",
        help=(
            "Used only when mode=splits_case_layout. "
            "If this column is missing or empty, fall back to case_id."
        ),
    )
    parser.add_argument("--strict_exists", action="store_true")
    parser.add_argument(
        "--report_path",
        default=None,
        help="Optional markdown report path. Defaults to <output_csv_dir>/protocol_notes.md",
    )
    return parser.parse_args()


def normalize_posix(path_str: str) -> str:
    return str(path_str).replace("\\", "/")


def remap_single_path(path_str: str, source_root: Path, target_root: Path) -> str:
    src_root = normalize_posix(str(source_root)).rstrip("/")
    tgt_root = normalize_posix(str(target_root)).rstrip("/")
    path_norm = normalize_posix(str(path_str))

    if not path_norm.startswith(src_root + "/") and path_norm != src_root:
        raise ValueError(
            f"Path does not start with source_root.\n"
            f"path={path_norm}\nsource_root={src_root}"
        )

    suffix = path_norm[len(src_root):]
    remapped = tgt_root + suffix
    return remapped


def label_to_class_dir(label_value) -> str:
    return "abnormal" if int(label_value) == 1 else "normal"


def choose_case_dir(row: pd.Series, case_dir_column: str) -> str:
    if case_dir_column in row.index:
        value = str(row[case_dir_column]).strip()
        if value and value.lower() != "nan":
            return value
    if "case_id" not in row.index:
        raise ValueError("mode=splits_case_layout requires either case_dir_column or case_id.")
    return str(row["case_id"]).strip()


def build_split_case_path(row: pd.Series, target_root: Path, side: str, case_dir_column: str) -> str:
    if "split" not in row.index:
        raise ValueError("mode=splits_case_layout requires `split` column.")

    split_name = str(row["split"]).strip()
    case_dir = choose_case_dir(row, case_dir_column)
    filename_col = f"{side}_filename"
    label_col = f"{side}_single_label"

    if filename_col not in row.index:
        raise ValueError(f"mode=splits_case_layout requires `{filename_col}` column.")
    if label_col not in row.index:
        raise ValueError(f"mode=splits_case_layout requires `{label_col}` column.")

    filename = str(row[filename_col]).strip()
    class_dir = label_to_class_dir(row[label_col])
    return normalize_posix(str(Path(target_root) / split_name / case_dir / class_dir / filename))


def remap_pair_df(
    df: pd.DataFrame,
    source_root: Path,
    target_root: Path,
    strict_exists: bool,
    mode: str,
    case_dir_column: str,
) -> pd.DataFrame:
    out = df.copy()
    if mode == "prefix_replace":
        out["left_path"] = out["left_path"].astype(str).map(lambda p: remap_single_path(p, source_root, target_root))
        out["right_path"] = out["right_path"].astype(str).map(lambda p: remap_single_path(p, source_root, target_root))
    elif mode == "splits_case_layout":
        out["left_path"] = out.apply(
            lambda row: build_split_case_path(row, target_root=target_root, side="left", case_dir_column=case_dir_column),
            axis=1,
        )
        out["right_path"] = out.apply(
            lambda row: build_split_case_path(row, target_root=target_root, side="right", case_dir_column=case_dir_column),
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if strict_exists:
        left_exists = out["left_path"].map(lambda p: Path(str(p)).exists())
        right_exists = out["right_path"].map(lambda p: Path(str(p)).exists())
        missing_left = out.loc[~left_exists, ["left_path", "pair_key"] if "pair_key" in out.columns else ["left_path"]]
        missing_right = out.loc[
            ~right_exists,
            ["right_path", "pair_key"] if "pair_key" in out.columns else ["right_path"],
        ]
        if not missing_left.empty or not missing_right.empty:
            raise FileNotFoundError(
                "Some remapped image paths do not exist. "
                f"missing_left={len(missing_left)}, missing_right={len(missing_right)}"
            )

    return out


def summarize_split(df: pd.DataFrame, split_name: str) -> dict:
    labels = df["label"].astype(int)
    return {
        "split": split_name,
        "pairs": int(len(df)),
        "normal_pairs": int((labels == 0).sum()),
        "abnormal_pairs": int((labels == 1).sum()),
        "cases": int(df["case_id"].astype(str).nunique()) if "case_id" in df.columns else 0,
        "chromosomes": int(df["chromosome_id"].astype(str).nunique()) if "chromosome_id" in df.columns else 0,
    }


def write_report(report_path: Path, args, split_rows):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pair CSV Image-Root Remap",
        "",
        "Definition",
        "- preserve the original train/val/test split, labels, and metadata",
        "- only rewrite `left_path` and `right_path` from the old image root to the new image root",
        "- intended for switching an existing protocol to a new image version such as straightened chromosomes",
        "",
        "Settings",
        f"- mode: `{args.mode}`",
        f"- source_root: `{args.source_root}`",
        f"- target_root: `{args.target_root}`",
        f"- case_dir_column: `{args.case_dir_column}`",
        f"- strict_exists: `{bool(args.strict_exists)}`",
        "",
        "Split summary",
    ]
    for row in split_rows:
        lines.append(
            f"- {row['split']}: pairs={row['pairs']}, normal={row['normal_pairs']}, "
            f"abnormal={row['abnormal_pairs']}, cases={row['cases']}, chromosomes={row['chromosomes']}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    output_csv_dir = Path(args.output_csv_dir)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "prefix_replace" and not str(args.source_root).strip():
        raise ValueError("--source_root is required when mode=prefix_replace")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    train_out = remap_pair_df(
        train_df,
        source_root,
        target_root,
        args.strict_exists,
        mode=args.mode,
        case_dir_column=args.case_dir_column,
    )
    val_out = remap_pair_df(
        val_df,
        source_root,
        target_root,
        args.strict_exists,
        mode=args.mode,
        case_dir_column=args.case_dir_column,
    )
    test_out = remap_pair_df(
        test_df,
        source_root,
        target_root,
        args.strict_exists,
        mode=args.mode,
        case_dir_column=args.case_dir_column,
    )

    train_out.to_csv(output_csv_dir / "train.csv", index=False)
    val_out.to_csv(output_csv_dir / "val.csv", index=False)
    test_out.to_csv(output_csv_dir / "test.csv", index=False)

    split_rows = [
        summarize_split(train_out, "train"),
        summarize_split(val_out, "val"),
        summarize_split(test_out, "test"),
    ]
    pd.DataFrame(split_rows).to_csv(output_csv_dir / "split_summary.csv", index=False)

    report_path = Path(args.report_path) if args.report_path else (output_csv_dir / "protocol_notes.md")
    write_report(report_path, args, split_rows)

    print(f"Saved remapped pair CSVs to {output_csv_dir}")
    print(f"Saved split summary to {output_csv_dir / 'split_summary.csv'}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
