import argparse
import pandas as pd


def load_case_ids(csv_path: str):
    df = pd.read_csv(csv_path)

    if "case_id" not in df.columns:
        raise ValueError(f"{csv_path} 中不存在 'case_id' 列，无法检查病例隔离。")

    case_ids = set(df["case_id"].astype(str).dropna().unique().tolist())
    return df, case_ids


def summarize_overlap(name_a, ids_a, name_b, ids_b):
    overlap = sorted(ids_a & ids_b)
    return {
        "pair": f"{name_a} vs {name_b}",
        "num_overlap": len(overlap),
        "overlap_cases": overlap,
    }


def main(train_csv: str, val_csv: str, test_csv: str):
    train_df, train_ids = load_case_ids(train_csv)
    val_df, val_ids = load_case_ids(val_csv)
    test_df, test_ids = load_case_ids(test_csv)

    print("=" * 80)
    print("Split basic info")
    print("=" * 80)
    print(f"train: rows={len(train_df)}, unique_case_id={len(train_ids)}")
    print(f"val:   rows={len(val_df)}, unique_case_id={len(val_ids)}")
    print(f"test:  rows={len(test_df)}, unique_case_id={len(test_ids)}")

    overlaps = [
        summarize_overlap("train", train_ids, "val", val_ids),
        summarize_overlap("train", train_ids, "test", test_ids),
        summarize_overlap("val", val_ids, "test", test_ids),
    ]

    print("\n" + "=" * 80)
    print("Case overlap check")
    print("=" * 80)

    is_case_isolated = True
    for item in overlaps:
        print(f"{item['pair']}: overlap_cases = {item['num_overlap']}")
        if item["num_overlap"] > 0:
            is_case_isolated = False
            print("overlap case_id list:")
            print(item["overlap_cases"])
            print("-" * 80)

    print("\n" + "=" * 80)
    if is_case_isolated:
        print("结论：这三个文件是病例隔离的（train / val / test 的 case_id 没有重叠）。")
    else:
        print("结论：这三个文件不是病例隔离的（存在重复 case_id）。")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    args = parser.parse_args()

    main(args.train_csv, args.val_csv, args.test_csv)