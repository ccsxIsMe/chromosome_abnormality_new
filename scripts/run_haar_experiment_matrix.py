import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("\n[Run]", " ".join(str(item) for item in cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch a practical Haar experiment matrix covering raw/straightened, "
            "haar-only and P16+Haar fusion branches."
        )
    )
    parser.add_argument("--raw_train_csv", required=True)
    parser.add_argument("--raw_val_csv", required=True)
    parser.add_argument("--raw_test_csv", required=True)
    parser.add_argument("--straight_train_csv", required=True)
    parser.add_argument("--straight_val_csv", required=True)
    parser.add_argument("--straight_test_csv", required=True)
    parser.add_argument("--raw_p16_config", required=True)
    parser.add_argument("--raw_p16_ckpt", required=True)
    parser.add_argument("--straight_p16_config", required=True)
    parser.add_argument("--straight_p16_ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--profile_length", type=int, default=128)
    parser.add_argument("--band_width", type=int, default=32)
    parser.add_argument("--kernel_sizes", default="4,8,16,32,64")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def haar_only_command(train_csv, val_csv, test_csv, output_dir, representation_version, feature_set, feature_select_mode):
    return [
        sys.executable,
        "scripts/run_haar_adaboost_pair_baseline.py",
        "--train_csv",
        train_csv,
        "--val_csv",
        val_csv,
        "--test_csv",
        test_csv,
        "--output_dir",
        output_dir,
        "--profile_length",
        "128",
        "--band_width",
        "32",
        "--kernel_sizes",
        "4,8,16,32,64",
        "--representation_version",
        representation_version,
        "--pair_orientation_align",
        "--feature_set",
        feature_set,
        "--feature_select_mode",
        feature_select_mode,
        "--feature_topk",
        "128",
        "--n_estimators",
        "200",
        "--learning_rate",
        "0.5",
        "--max_depth",
        "1",
        "--seed",
        "42",
    ]


def fusion_command(config, ckpt, train_csv, val_csv, test_csv, output_dir, representation_version, feature_set, feature_select_mode):
    return [
        sys.executable,
        "scripts/run_p16_haar_boost_fusion.py",
        "--config",
        config,
        "--ckpt",
        ckpt,
        "--train_csv",
        train_csv,
        "--val_csv",
        val_csv,
        "--test_csv",
        test_csv,
        "--output_dir",
        output_dir,
        "--profile_length",
        "128",
        "--band_width",
        "32",
        "--kernel_sizes",
        "4,8,16,32,64",
        "--representation_version",
        representation_version,
        "--pair_orientation_align",
        "--feature_set",
        feature_set,
        "--haar_feature_select_mode",
        feature_select_mode,
        "--haar_feature_topk",
        "128",
        "--classifier",
        "gradient_boosting",
        "--n_estimators",
        "200",
        "--learning_rate",
        "0.05",
        "--max_depth",
        "2",
        "--subsample",
        "1.0",
        "--embedding_pca_dim",
        "64",
        "--seed",
        "42",
    ]


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_specs = [
        {
            "name": "haar_raw_1d_v2_all",
            "cmd": haar_only_command(
                args.raw_train_csv,
                args.raw_val_csv,
                args.raw_test_csv,
                str(output_root / "haar_raw_1d_v2_all"),
                representation_version="v2",
                feature_set="1d",
                feature_select_mode="all",
            ),
        },
        {
            "name": "haar_raw_2d_v2_all",
            "cmd": haar_only_command(
                args.raw_train_csv,
                args.raw_val_csv,
                args.raw_test_csv,
                str(output_root / "haar_raw_2d_v2_all"),
                representation_version="v2",
                feature_set="2d",
                feature_select_mode="all",
            ),
        },
        {
            "name": "haar_raw_1d2d_v2_all",
            "cmd": haar_only_command(
                args.raw_train_csv,
                args.raw_val_csv,
                args.raw_test_csv,
                str(output_root / "haar_raw_1d2d_v2_all"),
                representation_version="v2",
                feature_set="1d2d",
                feature_select_mode="all",
            ),
        },
        {
            "name": "haar_raw_1d_topk_v2",
            "cmd": haar_only_command(
                args.raw_train_csv,
                args.raw_val_csv,
                args.raw_test_csv,
                str(output_root / "haar_raw_1d_topk_v2"),
                representation_version="v2",
                feature_set="1d",
                feature_select_mode="topk",
            ),
        },
        {
            "name": "haar_straight_1d_v3_all",
            "cmd": haar_only_command(
                args.straight_train_csv,
                args.straight_val_csv,
                args.straight_test_csv,
                str(output_root / "haar_straight_1d_v3_all"),
                representation_version="v3",
                feature_set="1d",
                feature_select_mode="all",
            ),
        },
        {
            "name": "haar_straight_2d_v3_all",
            "cmd": haar_only_command(
                args.straight_train_csv,
                args.straight_val_csv,
                args.straight_test_csv,
                str(output_root / "haar_straight_2d_v3_all"),
                representation_version="v3",
                feature_set="2d",
                feature_select_mode="all",
            ),
        },
        {
            "name": "haar_straight_1d2d_v3_all",
            "cmd": haar_only_command(
                args.straight_train_csv,
                args.straight_val_csv,
                args.straight_test_csv,
                str(output_root / "haar_straight_1d2d_v3_all"),
                representation_version="v3",
                feature_set="1d2d",
                feature_select_mode="all",
            ),
        },
        {
            "name": "haar_straight_1d_topk_v3",
            "cmd": haar_only_command(
                args.straight_train_csv,
                args.straight_val_csv,
                args.straight_test_csv,
                str(output_root / "haar_straight_1d_topk_v3"),
                representation_version="v3",
                feature_set="1d",
                feature_select_mode="topk",
            ),
        },
        {
            "name": "fusion_raw_1d_all",
            "cmd": fusion_command(
                args.raw_p16_config,
                args.raw_p16_ckpt,
                args.raw_train_csv,
                args.raw_val_csv,
                args.raw_test_csv,
                str(output_root / "fusion_raw_1d_all"),
                representation_version="v2",
                feature_set="1d",
                feature_select_mode="all",
            ),
        },
        {
            "name": "fusion_raw_1d_topk",
            "cmd": fusion_command(
                args.raw_p16_config,
                args.raw_p16_ckpt,
                args.raw_train_csv,
                args.raw_val_csv,
                args.raw_test_csv,
                str(output_root / "fusion_raw_1d_topk"),
                representation_version="v2",
                feature_set="1d",
                feature_select_mode="topk",
            ),
        },
        {
            "name": "fusion_straight_1d_all",
            "cmd": fusion_command(
                args.straight_p16_config,
                args.straight_p16_ckpt,
                args.straight_train_csv,
                args.straight_val_csv,
                args.straight_test_csv,
                str(output_root / "fusion_straight_1d_all"),
                representation_version="v3",
                feature_set="1d",
                feature_select_mode="all",
            ),
        },
        {
            "name": "fusion_straight_1d_topk",
            "cmd": fusion_command(
                args.straight_p16_config,
                args.straight_p16_ckpt,
                args.straight_train_csv,
                args.straight_val_csv,
                args.straight_test_csv,
                str(output_root / "fusion_straight_1d_topk"),
                representation_version="v3",
                feature_set="1d",
                feature_select_mode="topk",
            ),
        },
    ]

    manifest_lines = ["name,results_path"]
    for spec in experiment_specs:
        run_cmd(spec["cmd"])
        manifest_lines.append(f"{spec['name']},{(output_root / spec['name'] / 'results.yaml').as_posix()}")

    manifest_path = output_root / "haar_experiment_manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"\nSaved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
