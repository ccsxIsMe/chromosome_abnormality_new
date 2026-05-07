import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact high-value posthoc suite for a saved multi-prototype straightened experiment: "
            "saved_eval -> chromosome-conditioned variants -> gray-zone reranker variants."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--saved_eval_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tail_quantiles", default="0.90,0.95")
    parser.add_argument("--quantiles", default="0.95,0.975,0.99")
    parser.add_argument("--gray_zone_ratios", default="0.10,0.20,0.30")
    return parser.parse_args()


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if str(x).strip()]


def run_cmd(cmd):
    print("\n[Run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    saved_eval_dir = Path(args.saved_eval_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_predictions = str(saved_eval_dir / "train_predictions.csv")
    val_predictions = str(saved_eval_dir / "val_predictions.csv")
    test_predictions = str(saved_eval_dir / "test_predictions.csv")

    for path in [train_predictions, val_predictions, test_predictions]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing saved-eval prediction file: {path}")

    quantiles = parse_float_list(args.quantiles)
    tail_quantiles = parse_float_list(args.tail_quantiles)
    gray_zone_ratios = parse_float_list(args.gray_zone_ratios)

    chr_score_modes = [
        "chr_zscore",
        "chr_robust_zscore",
        "chr_percentile",
        "chr_tail_zscore",
    ]

    for tail_q in tail_quantiles:
        for score_mode in chr_score_modes:
            save_dir = output_root / f"{score_mode}_tailq{str(tail_q).replace('.', 'p')}"
            run_cmd(
                [
                    sys.executable,
                    "scripts/evaluate_p12_chr_conditioned_posthoc.py",
                    "--train_predictions",
                    train_predictions,
                    "--val_predictions",
                    val_predictions,
                    "--test_predictions",
                    test_predictions,
                    "--output_dir",
                    str(save_dir),
                    "--score_mode",
                    score_mode,
                    "--quantiles",
                    ",".join(str(x) for x in quantiles),
                    "--tail_quantile",
                    str(tail_q),
                ]
            )

    reranker_base_modes = [
        "chr_tail_zscore",
        "chr_percentile",
    ]

    for tail_q in tail_quantiles:
        for base_mode in reranker_base_modes:
            for gray_ratio in gray_zone_ratios:
                save_dir = output_root / (
                    f"grayzone_{base_mode}_tailq{str(tail_q).replace('.', 'p')}"
                    f"_g{str(gray_ratio).replace('.', 'p')}"
                )
                run_cmd(
                    [
                        sys.executable,
                        "scripts/run_grayzone_reranker_posthoc.py",
                        "--config",
                        args.config,
                        "--ckpt",
                        args.ckpt,
                        "--save_dir",
                        str(save_dir),
                        "--base_score_mode",
                        base_mode,
                        "--tail_quantile",
                        str(tail_q),
                        "--gray_zone_ratio",
                        str(gray_ratio),
                    ]
                )

    print(f"\nSaved posthoc suite outputs under {output_root}")


if __name__ == "__main__":
    main()
