from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / "data_protocol" / "p1_pair_case_isolated_v1"
    target_dir = project_root / "data_protocol" / "p12_pair_normal_only_v1"
    report_dir = project_root / "outputs" / "p12_pair_normal_only_v1"

    target_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(source_dir / "train.csv")
    val_df = pd.read_csv(source_dir / "val.csv")
    test_df = pd.read_csv(source_dir / "test.csv")

    train_normal_df = train_df[train_df["label"].astype(int) == 0].copy()

    train_normal_df.sort_values(["chromosome_id", "case_id", "pair_key"]).to_csv(
        target_dir / "train.csv", index=False
    )
    val_df.sort_values(["chromosome_id", "case_id", "pair_key"]).to_csv(
        target_dir / "val.csv", index=False
    )
    test_df.sort_values(["chromosome_id", "case_id", "pair_key"]).to_csv(
        target_dir / "test.csv", index=False
    )

    summary_df = pd.DataFrame(
        [
            {
                "split": "train",
                "total_pairs": int(len(train_normal_df)),
                "normal_pairs": int((train_normal_df["label"].astype(int) == 0).sum()),
                "abnormal_pairs": int((train_normal_df["label"].astype(int) == 1).sum()),
                "cases": int(train_normal_df["case_id"].nunique()),
            },
            {
                "split": "val",
                "total_pairs": int(len(val_df)),
                "normal_pairs": int((val_df["label"].astype(int) == 0).sum()),
                "abnormal_pairs": int((val_df["label"].astype(int) == 1).sum()),
                "cases": int(val_df["case_id"].nunique()),
            },
            {
                "split": "test",
                "total_pairs": int(len(test_df)),
                "normal_pairs": int((test_df["label"].astype(int) == 0).sum()),
                "abnormal_pairs": int((test_df["label"].astype(int) == 1).sum()),
                "cases": int(test_df["case_id"].nunique()),
            },
        ]
    )
    summary_df.to_csv(report_dir / "split_summary.csv", index=False)

    notes = [
        "# P12 Pair Normal-Only Protocol",
        "",
        "Definition",
        "- derived from data_protocol/p1_pair_case_isolated_v1",
        "- train keeps only normal homologous pairs",
        "- val/test stay unchanged and still contain both normal and abnormal pairs",
        "",
        "Goal",
        "- learn chromosome-conditional normal pair manifold only",
        "- detect abnormalities as deviation from normal pair structure",
        "",
        f"- train normal pairs = {summary_df.loc[0, 'normal_pairs']}",
        f"- val total pairs = {summary_df.loc[1, 'total_pairs']}",
        f"- test total pairs = {summary_df.loc[2, 'total_pairs']}",
    ]
    (report_dir / "protocol_notes.md").write_text("\n".join(notes), encoding="utf-8")

    print(f"Saved P12 normal-only protocol to: {target_dir}")
    print(f"Saved P12 report to: {report_dir}")


if __name__ == "__main__":
    main()
