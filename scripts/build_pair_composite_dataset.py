import argparse
from pathlib import Path

import pandas as pd

from src.datasets.pair_composite_dataset import save_pair_composite_image


def process_split(csv_path, split_name, output_root):
    df = pd.read_csv(csv_path)
    rows = []

    for idx, row in df.iterrows():
        case_id = "" if pd.isna(row.get("case_id")) else str(row.get("case_id"))
        pair_key = "" if pd.isna(row.get("pair_key")) else str(row.get("pair_key"))
        chromosome_id = str(row["chromosome_id"])
        label = int(row["label"])

        file_stem_parts = [split_name, chromosome_id]
        if case_id:
            file_stem_parts.append(case_id)
        if pair_key:
            file_stem_parts.append(pair_key.replace("/", "_"))
        else:
            file_stem_parts.append(str(idx))

        file_name = "__".join(file_stem_parts) + ".png"
        rel_path = Path(split_name) / file_name
        abs_path = output_root / rel_path

        save_pair_composite_image(
            left_path=row["left_path"],
            right_path=row["right_path"],
            output_path=abs_path,
        )

        record = {
            "image_path": str(abs_path),
            "relative_path": str(rel_path).replace("\\", "/"),
            "label": label,
            "chromosome_id": chromosome_id,
            "case_id": case_id,
            "pair_key": pair_key,
            "abnormal_subtype_id": "" if pd.isna(row.get("abnormal_subtype_id")) else str(row.get("abnormal_subtype_id")),
            "subtype_status": "" if pd.isna(row.get("subtype_status")) else str(row.get("subtype_status")),
            "left_path": str(row["left_path"]),
            "right_path": str(row["right_path"]),
            "split": split_name,
        }
        rows.append(record)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_to_csv = {
        "train": args.train_csv,
        "val": args.val_csv,
        "test": args.test_csv,
    }

    summary_rows = []
    for split_name, csv_path in split_to_csv.items():
        manifest_df = process_split(csv_path, split_name, output_root)
        manifest_path = output_root / f"{split_name}_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)

        summary_rows.append(
            {
                "split": split_name,
                "num_images": int(len(manifest_df)),
                "num_cases": int(manifest_df["case_id"].nunique()) if "case_id" in manifest_df.columns else 0,
                "num_abnormal": int((manifest_df["label"].astype(int) == 1).sum()),
            }
        )

        print(f"Saved {split_name} manifest to {manifest_path}")

    pd.DataFrame(summary_rows).to_csv(output_root / "summary.csv", index=False)
    print(f"Saved summary to {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
