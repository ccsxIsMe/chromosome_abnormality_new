import argparse
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bool_to_yes_no(value):
    return "Yes" if bool(value) else "No"


def format_float(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def detect_data_variant(payload):
    for key in ["train_csv", "val_csv", "test_csv"]:
        value = str(payload.get(key, ""))
        if "straight_out" in value or "straightened" in value:
            return "straightened"
    split_info = payload.get("fusion_split", {})
    for key in ["train_csv", "val_csv", "test_csv"]:
        value = str(split_info.get(key, ""))
        if "straight_out" in value or "straightened" in value:
            return "straightened"
    return "raw"


def summarize_kernel_catalog(catalog):
    if not catalog:
        return ""

    parts = []
    one_d = catalog.get("one_d_templates", [])
    if one_d:
        one_d_desc = ", ".join(f"{row['name']}={row['weights']}" for row in one_d)
        parts.append(f"1D: {one_d_desc}")

    two_d = catalog.get("two_d_templates", [])
    if two_d:
        width_sizes = catalog.get("two_d_width_sizes", [])
        two_d_desc = ", ".join(f"{row['name']}={row['weights']}" for row in two_d)
        parts.append(f"2D(widths={width_sizes}): {two_d_desc}")

    return " | ".join(parts)


def summarize_fusion_method(payload):
    if payload.get("method") != "p16_haar_boost_fusion":
        if payload.get("method") == "p16_haar_score_fusion":
            return "score-level fusion: weighted sum of normalized P16 score and Haar score"
        return "None"

    feature_settings = payload.get("feature_settings", {})
    haar_feature_set = feature_settings.get("haar_feature_set", feature_settings.get("feature_set", "1d"))
    use_embedding = bool(feature_settings.get("use_embedding", False))
    use_prototypes = bool(feature_settings.get("use_prototype_distances", False))
    use_scalars = bool(feature_settings.get("use_model_scalars", False))
    pca = feature_settings.get("embedding_pca")
    pca_desc = ""
    if isinstance(pca, dict) and pca.get("n_components") is not None:
        pca_desc = f" + embedding PCA({pca['n_components']})"

    parts = [f"feature-level fusion: Haar({haar_feature_set})"]
    nn_parts = []
    if use_embedding:
        nn_parts.append("P16 embedding")
    if use_prototypes:
        nn_parts.append("prototype distances")
    if use_scalars:
        nn_parts.append("model scalars")
    if nn_parts:
        parts.append(" + ".join(nn_parts) + pca_desc)
    parts.append("-> boosting classifier")
    return " ; ".join(parts)


def summarize_boosting(payload):
    model_settings = payload.get("model_settings", {})
    classifier = model_settings.get("classifier", "")
    if not classifier:
        return ""
    if classifier == "AdaBoostClassifier":
        weak = model_settings.get("weak_learner", "DecisionTreeClassifier")
        depth = model_settings.get("weak_learner_max_depth")
        return f"AdaBoost ({weak}, depth={depth})"
    if classifier == "gradient_boosting":
        depth = model_settings.get("max_depth")
        n_estimators = model_settings.get("n_estimators")
        lr = model_settings.get("learning_rate")
        return f"GradientBoosting (depth={depth}, n_estimators={n_estimators}, lr={lr})"
    return str(classifier)


def extract_test_metrics(payload):
    metrics = payload.get("test_metrics_best", {})
    return {
        "f1": metrics.get("f1"),
        "precision_abnormal": metrics.get("precision_abnormal"),
        "recall_abnormal": metrics.get("recall_abnormal"),
        "balanced_acc": metrics.get("balanced_acc"),
        "auprc": metrics.get("auprc"),
        "auroc": metrics.get("auroc"),
    }


def infer_experiment_name(payload, path):
    if payload.get("method") == "haar_like_adaboost_pair_baseline":
        feature_settings = payload.get("feature_settings", {})
        feature_set = feature_settings.get("feature_set", "1d")
        select_mode = feature_settings.get("feature_select_mode", "all")
        repr_version = feature_settings.get("representation_version", "")
        align = feature_settings.get("pair_orientation_align")
        return f"Haar-only ({feature_set}, {repr_version}, align={align}, select={select_mode})"

    if payload.get("method") == "p16_haar_boost_fusion":
        feature_settings = payload.get("feature_settings", {})
        feature_set = feature_settings.get("haar_feature_set", "1d")
        select_mode = feature_settings.get("haar_feature_select_mode", "all")
        return f"P16+Haar fusion ({feature_set}, select={select_mode})"

    if payload.get("method") == "p16_haar_score_fusion":
        return "P16+Haar score fusion"

    return path.parent.name


def build_row(results_path):
    payload = load_yaml(results_path)
    feature_settings = payload.get("feature_settings", {})
    kernel_catalog = feature_settings.get("haar_kernel_catalog")
    metrics = extract_test_metrics(payload)

    row = {
        "experiment_name": infer_experiment_name(payload, results_path),
        "results_path": str(results_path),
        "data_variant": detect_data_variant(payload),
        "model_family": str(payload.get("method", "")),
        "haar_feature_set": str(feature_settings.get("feature_set", feature_settings.get("haar_feature_set", ""))),
        "representation_version": str(feature_settings.get("representation_version", "")),
        "feature_select_mode": str(
            feature_settings.get("feature_select_mode", feature_settings.get("haar_feature_select_mode", ""))
        ),
        "selected_feature_count": str(
            feature_settings.get("num_feature_columns", feature_settings.get("selected_haar_feature_count", ""))
        ),
        "pair_orientation_align": bool_to_yes_no(feature_settings.get("pair_orientation_align", False)),
        "fusion_with_neural_network": summarize_fusion_method(payload),
        "boosting_type": summarize_boosting(payload),
        "haar_kernel_shape": summarize_kernel_catalog(kernel_catalog),
        "test_f1": format_float(metrics["f1"]),
        "test_precision_abnormal": format_float(metrics["precision_abnormal"]),
        "test_recall_abnormal": format_float(metrics["recall_abnormal"]),
        "test_balanced_acc": format_float(metrics["balanced_acc"]),
        "test_auprc": format_float(metrics["auprc"]),
        "test_auroc": format_float(metrics["auroc"]),
    }
    return row


def write_markdown_table(rows, output_path):
    headers = [
        "Experiment",
        "Data Variant",
        "Haar Features",
        "Representation",
        "Feature Select",
        "Pair Align",
        "Fusion Method",
        "Boosting",
        "Haar Kernels",
        "Test F1",
        "Prec_abn",
        "Rec_abn",
        "BalAcc",
        "AUPRC",
        "AUROC",
    ]
    lines = [
        "# Haar Experiment Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment_name"],
                    row["data_variant"],
                    row["haar_feature_set"],
                    row["representation_version"],
                    row["feature_select_mode"],
                    row["pair_orientation_align"],
                    row["fusion_with_neural_network"],
                    row["boosting_type"],
                    row["haar_kernel_shape"],
                    row["test_f1"],
                    row["test_precision_abnormal"],
                    row["test_recall_abnormal"],
                    row["test_balanced_acc"],
                    row["test_auprc"],
                    row["test_auroc"],
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows, output_path):
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = str(row.get(header, ""))
            value = value.replace('"', '""')
            if "," in value or '"' in value or "\n" in value:
                value = f'"{value}"'
            values.append(value)
        lines.append(",".join(values))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a set of Haar-related experiment results into markdown/csv tables. "
            "The output explicitly includes raw vs straightened, fusion method, boosting type, and Haar kernels."
        )
    )
    parser.add_argument("--results", nargs="+", required=True, help="One or more results.yaml files")
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    rows = [build_row(Path(path)) for path in args.results]
    write_markdown_table(rows, Path(args.output_md))
    write_csv(rows, Path(args.output_csv))
    print(f"Saved markdown summary to {args.output_md}")
    print(f"Saved csv summary to {args.output_csv}")


if __name__ == "__main__":
    main()
