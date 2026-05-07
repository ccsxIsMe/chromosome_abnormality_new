import argparse
import subprocess
import sys
from pathlib import Path

import yaml


BEST_SCORE_MODE = "chr_tail_zscore"
BEST_TAIL_QUANTILE = 0.9
DEFAULT_QUANTILES = "0.95,0.975,0.99"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the validated best posthoc setting for the straightened saved-eval branch. "
            "This fixes score_mode=chr_tail_zscore, tail_quantile=0.9, and selects the "
            "global val-best threshold as the formal operating point."
        )
    )
    parser.add_argument(
        "--saved_eval_dir",
        required=True,
        help="Directory containing train_predictions.csv / val_predictions.csv / test_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Defaults to a sibling directory next to saved_eval_dir.",
    )
    parser.add_argument(
        "--quantiles",
        default=DEFAULT_QUANTILES,
        help="Quantiles passed through to evaluate_p12_chr_conditioned_posthoc.py",
    )
    parser.add_argument(
        "--reference_results",
        default=None,
        help=(
            "Optional reference results.yaml for automatic comparison. "
            "Supports both posthoc results (global_eval.test_metrics_val_best) and "
            "plain saved-eval results (test_metrics_best)."
        ),
    )
    parser.add_argument(
        "--reference_name",
        default="reference",
        help="Display name used in the comparison summary when --reference_results is provided.",
    )
    return parser.parse_args()


def default_output_dir(saved_eval_dir):
    return saved_eval_dir.parent / f"{saved_eval_dir.name}__best_posthoc"


def ensure_prediction_files(saved_eval_dir):
    expected = [
        saved_eval_dir / "train_predictions.csv",
        saved_eval_dir / "val_predictions.csv",
        saved_eval_dir / "test_predictions.csv",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing saved-eval prediction files:\n" + "\n".join(f"- {path}" for path in missing)
        )
    return expected


def run_cmd(cmd):
    print("\n[Run]", " ".join(str(item) for item in cmd))
    subprocess.run(cmd, check=True)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def extract_reference_metrics(payload):
    if not isinstance(payload, dict):
        raise ValueError("Reference payload must be a mapping")

    global_eval = payload.get("global_eval")
    if isinstance(global_eval, dict) and "test_metrics_val_best" in global_eval:
        return global_eval["test_metrics_val_best"], "global_val_best"

    if "test_metrics_best" in payload:
        return payload["test_metrics_best"], "best_threshold"

    if "test_final_metrics" in payload:
        return payload["test_final_metrics"], "final"

    raise ValueError(
        "Unsupported reference results format. Expected one of: "
        "global_eval.test_metrics_val_best, test_metrics_best, or test_final_metrics."
    )


def metric_delta(current_value, reference_value):
    return float(current_value) - float(reference_value)


def build_selected_summary(results, output_dir, saved_eval_dir):
    global_eval = results["global_eval"]
    test_metrics = global_eval["test_metrics_val_best"]
    confusion = test_metrics["confusion_matrix"]

    summary = {
        "selection": {
            "score_mode": BEST_SCORE_MODE,
            "tail_quantile": BEST_TAIL_QUANTILE,
            "threshold_policy": "global_val_best",
            "selected_threshold": float(global_eval["val_best_threshold"]),
        },
        "test_metrics": {
            "f1": float(test_metrics["f1"]),
            "precision_abnormal": float(test_metrics["precision_abnormal"]),
            "recall_abnormal": float(test_metrics["recall_abnormal"]),
            "balanced_acc": float(test_metrics["balanced_acc"]),
            "auprc": float(test_metrics["auprc"]),
            "auroc": float(test_metrics["auroc"]),
            "confusion_matrix": {
                "tn": int(confusion["tn"]),
                "fp": int(confusion["fp"]),
                "fn": int(confusion["fn"]),
                "tp": int(confusion["tp"]),
            },
        },
        "artifacts": {
            "saved_eval_dir": str(saved_eval_dir),
            "output_dir": str(output_dir),
            "results_yaml": str(output_dir / "results.yaml"),
            "results_full_yaml": str(output_dir / "results_full.yaml"),
            "summary_table_md": str(output_dir / "summary_table.md"),
        },
    }
    return summary


def write_best_summary_md(path, selected_summary):
    metrics = selected_summary["test_metrics"]
    confusion = metrics["confusion_matrix"]
    selection = selected_summary["selection"]
    artifacts = selected_summary["artifacts"]

    lines = [
        "# Best Straightened Posthoc Summary",
        "",
        (
            f"- selected setting: `{selection['score_mode']}` + "
            f"`tail_quantile={selection['tail_quantile']}` + "
            f"`{selection['threshold_policy']}`"
        ),
        f"- selected threshold: `{selection['selected_threshold']:.6f}`",
        f"- saved_eval_dir: `{artifacts['saved_eval_dir']}`",
        f"- output_dir: `{artifacts['output_dir']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Test F1 | {metrics['f1']:.4f} |",
        f"| Test Precision_abnormal | {metrics['precision_abnormal']:.4f} |",
        f"| Test Recall_abnormal | {metrics['recall_abnormal']:.4f} |",
        f"| Test Balanced Acc | {metrics['balanced_acc']:.4f} |",
        f"| Test AUPRC | {metrics['auprc']:.4f} |",
        f"| Test AUROC | {metrics['auroc']:.4f} |",
        f"| TN | {confusion['tn']} |",
        f"| FP | {confusion['fp']} |",
        f"| FN | {confusion['fn']} |",
        f"| TP | {confusion['tp']} |",
        "",
        "## Artifacts",
        "",
        f"- results: `{artifacts['results_yaml']}`",
        f"- full results: `{artifacts['results_full_yaml']}`",
        f"- compact summary: `{artifacts['summary_table_md']}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_comparison_payload(current_summary, reference_metrics, reference_name, reference_selection):
    current_metrics = current_summary["test_metrics"]
    current_confusion = current_metrics["confusion_matrix"]
    reference_confusion = reference_metrics["confusion_matrix"]

    return {
        "current_selection": current_summary["selection"],
        "reference": {
            "name": reference_name,
            "selection": reference_selection,
        },
        "delta": {
            "f1": metric_delta(current_metrics["f1"], reference_metrics["f1"]),
            "precision_abnormal": metric_delta(
                current_metrics["precision_abnormal"],
                reference_metrics["precision_abnormal"],
            ),
            "recall_abnormal": metric_delta(
                current_metrics["recall_abnormal"],
                reference_metrics["recall_abnormal"],
            ),
            "balanced_acc": metric_delta(
                current_metrics["balanced_acc"],
                reference_metrics["balanced_acc"],
            ),
            "auprc": metric_delta(current_metrics["auprc"], reference_metrics["auprc"]),
            "auroc": metric_delta(current_metrics["auroc"], reference_metrics["auroc"]),
            "fp": int(current_confusion["fp"]) - int(reference_confusion["fp"]),
            "fn": int(current_confusion["fn"]) - int(reference_confusion["fn"]),
            "tp": int(current_confusion["tp"]) - int(reference_confusion["tp"]),
            "tn": int(current_confusion["tn"]) - int(reference_confusion["tn"]),
        },
        "current_metrics": current_metrics,
        "reference_metrics": {
            "f1": float(reference_metrics["f1"]),
            "precision_abnormal": float(reference_metrics["precision_abnormal"]),
            "recall_abnormal": float(reference_metrics["recall_abnormal"]),
            "balanced_acc": float(reference_metrics["balanced_acc"]),
            "auprc": float(reference_metrics["auprc"]),
            "auroc": float(reference_metrics["auroc"]),
            "confusion_matrix": {
                "tn": int(reference_confusion["tn"]),
                "fp": int(reference_confusion["fp"]),
                "fn": int(reference_confusion["fn"]),
                "tp": int(reference_confusion["tp"]),
            },
        },
    }


def write_comparison_md(path, comparison):
    current_metrics = comparison["current_metrics"]
    reference_metrics = comparison["reference_metrics"]
    delta = comparison["delta"]
    reference = comparison["reference"]

    lines = [
        "# Comparison With Reference",
        "",
        f"- current selection: `{comparison['current_selection']['score_mode']}` + "
        f"`tail_quantile={comparison['current_selection']['tail_quantile']}` + "
        f"`{comparison['current_selection']['threshold_policy']}`",
        f"- reference name: `{reference['name']}`",
        f"- reference selection: `{reference['selection']}`",
        "",
        "| Metric | Current | Reference | Delta |",
        "|---|---:|---:|---:|",
        f"| F1 | {current_metrics['f1']:.4f} | {reference_metrics['f1']:.4f} | {delta['f1']:+.4f} |",
        (
            "| Precision_abnormal | "
            f"{current_metrics['precision_abnormal']:.4f} | "
            f"{reference_metrics['precision_abnormal']:.4f} | "
            f"{delta['precision_abnormal']:+.4f} |"
        ),
        (
            "| Recall_abnormal | "
            f"{current_metrics['recall_abnormal']:.4f} | "
            f"{reference_metrics['recall_abnormal']:.4f} | "
            f"{delta['recall_abnormal']:+.4f} |"
        ),
        (
            "| Balanced Acc | "
            f"{current_metrics['balanced_acc']:.4f} | "
            f"{reference_metrics['balanced_acc']:.4f} | "
            f"{delta['balanced_acc']:+.4f} |"
        ),
        f"| AUPRC | {current_metrics['auprc']:.4f} | {reference_metrics['auprc']:.4f} | {delta['auprc']:+.4f} |",
        f"| AUROC | {current_metrics['auroc']:.4f} | {reference_metrics['auroc']:.4f} | {delta['auroc']:+.4f} |",
        (
            "| FP | "
            f"{current_metrics['confusion_matrix']['fp']} | "
            f"{reference_metrics['confusion_matrix']['fp']} | "
            f"{delta['fp']:+d} |"
        ),
        (
            "| FN | "
            f"{current_metrics['confusion_matrix']['fn']} | "
            f"{reference_metrics['confusion_matrix']['fn']} | "
            f"{delta['fn']:+d} |"
        ),
        (
            "| TP | "
            f"{current_metrics['confusion_matrix']['tp']} | "
            f"{reference_metrics['confusion_matrix']['tp']} | "
            f"{delta['tp']:+d} |"
        ),
        (
            "| TN | "
            f"{current_metrics['confusion_matrix']['tn']} | "
            f"{reference_metrics['confusion_matrix']['tn']} | "
            f"{delta['tn']:+d} |"
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    saved_eval_dir = Path(args.saved_eval_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(saved_eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_predictions, val_predictions, test_predictions = ensure_prediction_files(saved_eval_dir)

    cmd = [
        sys.executable,
        "scripts/evaluate_p12_chr_conditioned_posthoc.py",
        "--train_predictions",
        str(train_predictions),
        "--val_predictions",
        str(val_predictions),
        "--test_predictions",
        str(test_predictions),
        "--output_dir",
        str(output_dir),
        "--score_mode",
        BEST_SCORE_MODE,
        "--quantiles",
        str(args.quantiles),
        "--tail_quantile",
        str(BEST_TAIL_QUANTILE),
    ]
    run_cmd(cmd)

    results_path = output_dir / "results.yaml"
    if not results_path.exists():
        raise FileNotFoundError(f"Expected posthoc results not found: {results_path}")

    results = load_yaml(results_path)
    selected_summary = build_selected_summary(results, output_dir, saved_eval_dir)
    dump_yaml(output_dir / "best_selected_summary.yaml", selected_summary)
    write_best_summary_md(output_dir / "best_selected_summary.md", selected_summary)

    if args.reference_results:
        reference_path = Path(args.reference_results).resolve()
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference results file not found: {reference_path}")
        reference_payload = load_yaml(reference_path)
        reference_metrics, reference_selection = extract_reference_metrics(reference_payload)
        comparison = build_comparison_payload(
            current_summary=selected_summary,
            reference_metrics=reference_metrics,
            reference_name=str(args.reference_name),
            reference_selection=reference_selection,
        )
        dump_yaml(output_dir / "comparison_with_reference.yaml", comparison)
        write_comparison_md(output_dir / "comparison_with_reference.md", comparison)

    print(f"\nSaved best straightened posthoc outputs under {output_dir}")
    print(f"- results: {output_dir / 'results.yaml'}")
    print(f"- selected summary: {output_dir / 'best_selected_summary.md'}")
    if args.reference_results:
        print(f"- comparison: {output_dir / 'comparison_with_reference.md'}")


if __name__ == "__main__":
    main()
