import copy
import csv
import re
from pathlib import Path

import numpy as np
import yaml

from src.main import load_config, run_experiment, to_serializable


DEFAULT_CONFIGS = [
    "configs/local_pair_resnet18_weighted_ce.yaml",
    "configs/baseline_pair_resnet18_chrid_weighted_ce.yaml",
]

SUMMARY_METRIC_PATHS = [
    ("test_metrics_best", "auroc"),
    ("test_metrics_best", "auprc"),
    ("test_metrics_best", "f1"),
    ("test_metrics_best", "balanced_acc"),
    ("test_metrics_best", "loss"),
    ("test_metrics_best", "precision_abnormal"),
    ("test_metrics_best", "recall_abnormal"),
    ("test_metrics_05", "auroc"),
    ("test_metrics_05", "auprc"),
    ("test_metrics_05", "f1"),
    ("test_metrics_05", "balanced_acc"),
    ("test_metrics_05", "loss"),
    ("test_metrics_05", "precision_abnormal"),
    ("test_metrics_05", "recall_abnormal"),
]


def natural_fold_key(path):
    match = re.search(r"(\d+)$", path.name)
    number = int(match.group(1)) if match else 10**9
    return (number, path.name)


def list_fold_dirs(fold_root):
    fold_dirs = [path for path in fold_root.iterdir() if path.is_dir() and path.name.startswith("fold_")]
    return sorted(fold_dirs, key=natural_fold_key)


def get_nested_value(data, path):
    value = data
    for key in path:
        value = value[key]
    return value


def summarize_metric(values):
    normalized = [np.nan if value is None else value for value in values]
    arr = np.array(normalized, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=0)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def build_fold_config(base_cfg, fold_dir):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["train_csv"] = str((fold_dir / "train_pair.csv").as_posix())
    cfg["data"]["val_csv"] = str((fold_dir / "val_pair.csv").as_posix())
    cfg["data"]["test_csv"] = str((fold_dir / "test_pair.csv").as_posix())
    cfg["experiment_name"] = f"{base_cfg['experiment_name']}_{fold_dir.name}"
    return cfg


def aggregate_results(per_fold_results):
    summary = {}
    for path in SUMMARY_METRIC_PATHS:
        values = [get_nested_value(result, path) for result in per_fold_results]
        summary["/".join(path)] = summarize_metric(values)

    thresholds = [result["best_threshold"] for result in per_fold_results]
    summary["best_threshold"] = summarize_metric(thresholds)
    return summary


def save_per_fold_csv(save_path, per_fold_results):
    fieldnames = [
        "fold",
        "experiment_name",
        "best_threshold",
        "test_best_auroc",
        "test_best_auprc",
        "test_best_f1",
        "test_best_balanced_acc",
        "test_best_loss",
        "test_05_auroc",
        "test_05_auprc",
        "test_05_f1",
        "test_05_balanced_acc",
        "test_05_loss",
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in per_fold_results:
            writer.writerow(
                {
                    "fold": result["fold"],
                    "experiment_name": result["experiment_name"],
                    "best_threshold": result["best_threshold"],
                    "test_best_auroc": result["test_metrics_best"]["auroc"],
                    "test_best_auprc": result["test_metrics_best"]["auprc"],
                    "test_best_f1": result["test_metrics_best"]["f1"],
                    "test_best_balanced_acc": result["test_metrics_best"]["balanced_acc"],
                    "test_best_loss": result["test_metrics_best"]["loss"],
                    "test_05_auroc": result["test_metrics_05"]["auroc"],
                    "test_05_auprc": result["test_metrics_05"]["auprc"],
                    "test_05_f1": result["test_metrics_05"]["f1"],
                    "test_05_balanced_acc": result["test_metrics_05"]["balanced_acc"],
                    "test_05_loss": result["test_metrics_05"]["loss"],
                }
            )


def save_summary(summary_dir, base_experiment_name, base_config_path, per_fold_results, aggregate_summary):
    summary_dir.mkdir(parents=True, exist_ok=True)

    per_fold_csv_path = summary_dir / f"{base_experiment_name}_per_fold.csv"
    save_per_fold_csv(per_fold_csv_path, per_fold_results)

    summary_payload = {
        "base_experiment_name": base_experiment_name,
        "base_config_path": str(base_config_path),
        "num_folds": len(per_fold_results),
        "per_fold_results": per_fold_results,
        "aggregate_summary": aggregate_summary,
    }

    summary_yaml_path = summary_dir / f"{base_experiment_name}_summary.yaml"
    with open(summary_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(summary_payload), f, allow_unicode=True, sort_keys=False)

    return per_fold_csv_path, summary_yaml_path


def run_group_kfold(config_paths, fold_root, summary_dir):
    fold_dirs = list_fold_dirs(fold_root)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {fold_root}")

    for fold_dir in fold_dirs:
        for csv_name in ["train_pair.csv", "val_pair.csv", "test_pair.csv"]:
            csv_path = fold_dir / csv_name
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing required fold CSV: {csv_path}")

    print(f"Found {len(fold_dirs)} folds under {fold_root}")

    for config_path in config_paths:
        config_path = Path(config_path)
        base_cfg = load_config(config_path)
        base_experiment_name = base_cfg["experiment_name"]
        per_fold_results = []

        print(f"\n==============================")
        print(f"Running grouped k-fold experiment: {base_experiment_name}")
        print(f"Base config: {config_path}")
        print(f"==============================")

        for fold_dir in fold_dirs:
            print(f"\n----- Running {fold_dir.name} -----")
            fold_cfg = build_fold_config(base_cfg, fold_dir)
            result = run_experiment(fold_cfg, config_path=str(config_path))
            result["fold"] = fold_dir.name
            per_fold_results.append(result)

        aggregate_summary = aggregate_results(per_fold_results)
        per_fold_csv_path, summary_yaml_path = save_summary(
            summary_dir=summary_dir,
            base_experiment_name=base_experiment_name,
            base_config_path=config_path,
            per_fold_results=per_fold_results,
            aggregate_summary=aggregate_summary,
        )

        print(f"\nSummary for {base_experiment_name}:")
        for metric_name, stat in aggregate_summary.items():
            print(
                f"{metric_name}: "
                f"mean={stat['mean']:.6f}, std={stat['std']:.6f}, "
                f"min={stat['min']:.6f}, max={stat['max']:.6f}"
            )
        print(f"Saved per-fold CSV to {per_fold_csv_path}")
        print(f"Saved summary YAML to {summary_yaml_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-root",
        type=str,
        default="data_cv/group_kfold_pair",
        help="Directory containing fold_0, fold_1, ... subdirectories",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Base experiment configs to run across all folds",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="outputs/group_kfold_summary",
        help="Directory for aggregated fold summaries",
    )
    args = parser.parse_args()

    run_group_kfold(
        config_paths=args.configs,
        fold_root=Path(args.fold_root),
        summary_dir=Path(args.summary_dir),
    )


if __name__ == "__main__":
    main()
