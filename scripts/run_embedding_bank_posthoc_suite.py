import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def parse_float_csv(text):
    return ",".join([item.strip() for item in str(text).split(",") if item.strip()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--analyses",
        default="memory,kmeans,gmm",
        help="Comma-separated subset of: memory,kmeans,gmm",
    )

    parser.add_argument("--memory_distance", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--memory_knn_k", type=int, default=1)
    parser.add_argument("--memory_quantiles", default="0.95,0.975,0.99")
    parser.add_argument("--memory_max_train_per_chr", type=int, default=0)
    parser.add_argument("--disable_memory_loo", action="store_true")

    parser.add_argument("--kmeans_distance", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--kmeans_num_prototypes", type=int, default=8)
    parser.add_argument("--kmeans_topk_prototypes", type=int, default=3)
    parser.add_argument("--kmeans_quantiles", default="0.95,0.975,0.99")
    parser.add_argument("--kmeans_max_train_per_chr", type=int, default=0)
    parser.add_argument("--kmeans_seed", type=int, default=42)

    parser.add_argument("--gmm_num_components", type=int, default=1)
    parser.add_argument("--gmm_covariance_type", default="diag", choices=["full", "tied", "diag", "spherical"])
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-5)
    parser.add_argument("--gmm_max_iter", type=int, default=200)
    parser.add_argument("--gmm_quantiles", default="0.95,0.975,0.99")
    parser.add_argument("--gmm_max_train_per_chr", type=int, default=0)
    parser.add_argument("--gmm_seed", type=int, default=42)
    args = parser.parse_args()

    analyses = {item.strip().lower() for item in str(args.analyses).split(",") if item.strip()}
    valid_analyses = {"memory", "kmeans", "gmm"}
    invalid = analyses - valid_analyses
    if invalid:
        raise ValueError(f"Unsupported analyses: {sorted(invalid)}")

    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [sys.executable]

    if "memory" in analyses:
        command = base_cmd + [
            str(script_dir / "evaluate_p12_embedding_chr_memory_bank.py"),
            "--config",
            args.config,
            "--ckpt",
            args.ckpt,
            "--save_dir",
            str(output_root / "memory_bank"),
            "--distance",
            args.memory_distance,
            "--knn_k",
            str(args.memory_knn_k),
            "--quantiles",
            parse_float_csv(args.memory_quantiles),
            "--max_train_per_chr",
            str(args.memory_max_train_per_chr),
        ]
        if args.disable_memory_loo:
            command.append("--no_train_leave_one_out")
        run_command(command)

    if "kmeans" in analyses:
        command = base_cmd + [
            str(script_dir / "evaluate_p12_embedding_chr_kmeans_bank.py"),
            "--config",
            args.config,
            "--ckpt",
            args.ckpt,
            "--save_dir",
            str(output_root / "kmeans_bank"),
            "--distance",
            args.kmeans_distance,
            "--num_prototypes",
            str(args.kmeans_num_prototypes),
            "--topk_prototypes",
            str(args.kmeans_topk_prototypes),
            "--quantiles",
            parse_float_csv(args.kmeans_quantiles),
            "--max_train_per_chr",
            str(args.kmeans_max_train_per_chr),
            "--seed",
            str(args.kmeans_seed),
        ]
        run_command(command)

    if "gmm" in analyses:
        command = base_cmd + [
            str(script_dir / "evaluate_p12_embedding_chr_gmm_bank.py"),
            "--config",
            args.config,
            "--ckpt",
            args.ckpt,
            "--save_dir",
            str(output_root / "gmm_bank"),
            "--num_components",
            str(args.gmm_num_components),
            "--covariance_type",
            args.gmm_covariance_type,
            "--reg_covar",
            str(args.gmm_reg_covar),
            "--max_iter",
            str(args.gmm_max_iter),
            "--quantiles",
            parse_float_csv(args.gmm_quantiles),
            "--max_train_per_chr",
            str(args.gmm_max_train_per_chr),
            "--seed",
            str(args.gmm_seed),
        ]
        run_command(command)


if __name__ == "__main__":
    main()
