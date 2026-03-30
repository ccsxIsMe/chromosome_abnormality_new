import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.metrics import compute_score_based_metrics, search_best_threshold


def to_serializable(value):
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def safe_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if std <= 1e-12:
        std = 1.0
    return mean, std


def safe_median_mad(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-12:
        mad = 1.0
    return median, mad


def load_prediction_csv(path):
    df = pd.read_csv(path)
    required_cols = {"label", "anomaly_score", "chromosome_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["label"] = df["label"].astype(int)
    df["anomaly_score"] = df["anomaly_score"].astype(float)
    df["chromosome_id"] = df["chromosome_id"].astype(str)

    if "subtype_status" in df.columns:
        df["subtype_status"] = df["subtype_status"].fillna("").astype(str)
    if "abnormal_subtype_id" in df.columns:
        df["abnormal_subtype_id"] = df["abnormal_subtype_id"].fillna("").astype(str)
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].fillna("").astype(str)

    return df


def build_chr_normal_stats(train_df):
    normal_df = train_df[train_df["label"] == 0].copy()
    if normal_df.empty:
        raise ValueError("Train predictions contain no normal samples.")

    global_mean, global_std = safe_mean_std(normal_df["anomaly_score"].to_numpy())
    global_median, global_mad = safe_median_mad(normal_df["anomaly_score"].to_numpy())
    global_stats = {
        "count": int(len(normal_df)),
        "mean": global_mean,
        "std": global_std,
        "median": global_median,
        "mad": global_mad,
    }

    chr_stats = {}
    for chromosome_id, group in normal_df.groupby("chromosome_id"):
        mean, std = safe_mean_std(group["anomaly_score"].to_numpy())
        median, mad = safe_median_mad(group["anomaly_score"].to_numpy())
        chr_stats[str(chromosome_id)] = {
            "count": int(len(group)),
            "mean": mean,
            "std": std,
            "median": median,
            "mad": mad,
        }

    return chr_stats, global_stats


def calibrate_score(raw_score, stats, score_mode):
    raw_score = float(raw_score)
    if score_mode == "raw":
        return raw_score
    if score_mode == "chr_zscore":
        return (raw_score - float(stats["mean"])) / max(float(stats["std"]), 1e-12)
    if score_mode == "chr_robust_zscore":
        return (raw_score - float(stats["median"])) / max(1.4826 * float(stats["mad"]), 1e-12)
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def apply_chr_calibration(df, chr_stats, global_stats, score_mode):
    calibrated_scores = []
    stat_sources = []
    for _, row in df.iterrows():
        chromosome_id = str(row["chromosome_id"])
        stats = chr_stats.get(chromosome_id, global_stats)
        calibrated_scores.append(calibrate_score(row["anomaly_score"], stats, score_mode))
        stat_sources.append("chromosome" if chromosome_id in chr_stats else "global")

    calibrated_df = df.copy()
    calibrated_df["raw_anomaly_score"] = calibrated_df["anomaly_score"].astype(float)
    calibrated_df["calibrated_score"] = np.asarray(calibrated_scores, dtype=np.float64)
    calibrated_df["calibration_source"] = stat_sources
    return calibrated_df


def build_global_quantile_thresholds(train_df, quantiles):
    normal_scores = train_df.loc[train_df["label"] == 0, "calibrated_score"].astype(float).to_numpy()
    return {float(q): float(np.quantile(normal_scores, q)) for q in quantiles}


def build_chr_quantile_thresholds(train_df, quantile):
    thresholds = {}
    normal_df = train_df[train_df["label"] == 0].copy()
    for chromosome_id, group in normal_df.groupby("chromosome_id"):
        thresholds[str(chromosome_id)] = float(np.quantile(group["calibrated_score"].astype(float).to_numpy(), quantile))
    return thresholds


def apply_constant_threshold(df, threshold):
    y_true = df["label"].astype(int).to_numpy()
    y_score = df["calibrated_score"].astype(float).to_numpy()
    y_pred = (y_score >= float(threshold)).astype(int)
    return y_true, y_score, y_pred


def apply_chr_thresholds(df, chr_thresholds, fallback_threshold):
    y_true = df["label"].astype(int).to_numpy()
    y_score = df["calibrated_score"].astype(float).to_numpy()
    y_pred = []
    used_thresholds = []

    for _, row in df.iterrows():
        chromosome_id = str(row["chromosome_id"])
        threshold = float(chr_thresholds.get(chromosome_id, fallback_threshold))
        used_thresholds.append(threshold)
        y_pred.append(1 if float(row["calibrated_score"]) >= threshold else 0)

    return y_true, y_score, np.asarray(y_pred, dtype=np.int64), np.asarray(used_thresholds, dtype=np.float64)


def compute_metrics_from_predictions(y_true, y_score, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision_abnormal = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_abnormal = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_abnormal = (
        2.0 * precision_abnormal * recall_abnormal / (precision_abnormal + recall_abnormal)
        if (precision_abnormal + recall_abnormal) > 0
        else 0.0
    )
    f1_normal = (
        2.0 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if (precision_normal + recall_normal) > 0
        else 0.0
    )

    metrics = compute_score_based_metrics(
        y_true=y_true.tolist(),
        y_score=y_score.tolist(),
        threshold=0.0,
        higher_score_more_positive=True,
    )
    metrics["f1"] = float(f1_abnormal)
    metrics["precision_abnormal"] = float(precision_abnormal)
    metrics["recall_abnormal"] = float(recall_abnormal)
    metrics["balanced_acc"] = float(0.5 * (recall_normal + recall_abnormal))
    metrics["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    metrics["normal"] = {
        "acc": float(recall_normal),
        "precision": float(precision_normal),
        "recall": float(recall_normal),
        "f1": float(f1_normal),
        "support": int((y_true == 0).sum()),
    }
    metrics["abnormal"] = {
        "acc": float(recall_abnormal),
        "precision": float(precision_abnormal),
        "recall": float(recall_abnormal),
        "f1": float(f1_abnormal),
        "support": int((y_true == 1).sum()),
    }
    return metrics


def summarize_subset_constant_threshold(df, threshold):
    if df.empty:
        return {"count": 0}
    y_true, y_score, y_pred = apply_constant_threshold(df, threshold)
    metrics = compute_metrics_from_predictions(y_true, y_score, y_pred)
    metrics["count"] = int(len(df))
    return metrics


def summarize_subset_chr_threshold(df, chr_thresholds, fallback_threshold):
    if df.empty:
        return {"count": 0}
    y_true, y_score, y_pred, _ = apply_chr_thresholds(df, chr_thresholds, fallback_threshold)
    metrics = compute_metrics_from_predictions(y_true, y_score, y_pred)
    metrics["count"] = int(len(df))
    return metrics


def summarize_by_subtype(df, pred_column, score_column):
    rows = []
    if "abnormal_subtype_id" not in df.columns:
        return rows

    abnormal_df = df[df["label"] == 1].copy()
    if abnormal_df.empty:
        return rows

    for subtype_id, group in abnormal_df.groupby("abnormal_subtype_id", dropna=False):
        scores = group[score_column].astype(float).to_numpy()
        preds = group[pred_column].astype(int).to_numpy()
        rows.append(
            {
                "abnormal_subtype_id": "" if pd.isna(subtype_id) else str(subtype_id),
                "chromosome_id": str(group["chromosome_id"].iloc[0]),
                "subtype_status": str(group["subtype_status"].iloc[0]) if "subtype_status" in group.columns else "",
                "count": int(len(group)),
                "recall_at_threshold": float(preds.mean()),
                "mean_score": float(scores.mean()),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
            }
        )
    return rows


def run_global_threshold_eval(train_df, val_df, test_df, quantiles):
    best_threshold, best_score, best_stats = search_best_threshold(
        y_true=val_df["label"].astype(int).tolist(),
        y_score=val_df["calibrated_score"].astype(float).tolist(),
        metric="f1",
        higher_score_more_positive=True,
    )

    test_metrics_best = compute_score_based_metrics(
        y_true=test_df["label"].astype(int).tolist(),
        y_score=test_df["calibrated_score"].astype(float).tolist(),
        threshold=best_threshold,
        higher_score_more_positive=True,
    )

    quantile_rows = []
    quantile_thresholds = build_global_quantile_thresholds(train_df, quantiles)
    for quantile, threshold in quantile_thresholds.items():
        val_metrics = compute_score_based_metrics(
            y_true=val_df["label"].astype(int).tolist(),
            y_score=val_df["calibrated_score"].astype(float).tolist(),
            threshold=threshold,
            higher_score_more_positive=True,
        )
        test_metrics = compute_score_based_metrics(
            y_true=test_df["label"].astype(int).tolist(),
            y_score=test_df["calibrated_score"].astype(float).tolist(),
            threshold=threshold,
            higher_score_more_positive=True,
        )
        quantile_rows.append(
            {
                "threshold_family": "global_quantile",
                "quantile": float(quantile),
                "threshold": float(threshold),
                "val_f1": float(val_metrics["f1"]),
                "val_recall_abnormal": float(val_metrics["recall_abnormal"]),
                "val_balanced_acc": float(val_metrics["balanced_acc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_precision_abnormal": float(test_metrics["precision_abnormal"]),
                "test_recall_abnormal": float(test_metrics["recall_abnormal"]),
                "test_balanced_acc": float(test_metrics["balanced_acc"]),
            }
        )

    return {
        "val_best_threshold": float(best_threshold),
        "val_best_score": float(best_score),
        "val_best_stats": best_stats,
        "test_metrics_val_best": test_metrics_best,
        "global_quantile_sweep": quantile_rows,
    }


def run_chr_threshold_eval(train_df, val_df, test_df, quantiles):
    global_fallbacks = build_global_quantile_thresholds(train_df, quantiles)
    rows = []

    for quantile in quantiles:
        chr_thresholds = build_chr_quantile_thresholds(train_df, quantile)
        fallback_threshold = float(global_fallbacks[float(quantile)])

        val_true, val_score, val_pred, _ = apply_chr_thresholds(val_df, chr_thresholds, fallback_threshold)
        test_true, test_score, test_pred, test_used_thresholds = apply_chr_thresholds(
            test_df,
            chr_thresholds,
            fallback_threshold,
        )

        val_metrics = compute_metrics_from_predictions(val_true, val_score, val_pred)
        test_metrics = compute_metrics_from_predictions(test_true, test_score, test_pred)

        rows.append(
            {
                "threshold_family": "chr_quantile",
                "quantile": float(quantile),
                "fallback_threshold": float(fallback_threshold),
                "val_f1": float(val_metrics["f1"]),
                "val_recall_abnormal": float(val_metrics["recall_abnormal"]),
                "val_balanced_acc": float(val_metrics["balanced_acc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_precision_abnormal": float(test_metrics["precision_abnormal"]),
                "test_recall_abnormal": float(test_metrics["recall_abnormal"]),
                "test_balanced_acc": float(test_metrics["balanced_acc"]),
                "test_tn": int(test_metrics["confusion_matrix"]["tn"]),
                "test_fp": int(test_metrics["confusion_matrix"]["fp"]),
                "test_fn": int(test_metrics["confusion_matrix"]["fn"]),
                "test_tp": int(test_metrics["confusion_matrix"]["tp"]),
                "thresholds": chr_thresholds,
                "test_used_threshold_mean": float(test_used_thresholds.mean()),
            }
        )

    rows = sorted(rows, key=lambda row: (-row["val_f1"], -row["val_recall_abnormal"], row["test_fp"]))
    best_row = rows[0]
    best_thresholds = best_row["thresholds"]
    fallback_threshold = float(best_row["fallback_threshold"])

    test_true, test_score, test_pred, test_used_thresholds = apply_chr_thresholds(
        test_df,
        best_thresholds,
        fallback_threshold,
    )
    best_test_metrics = compute_metrics_from_predictions(test_true, test_score, test_pred)

    val_true, val_score, val_pred, _ = apply_chr_thresholds(val_df, best_thresholds, fallback_threshold)
    best_val_metrics = compute_metrics_from_predictions(val_true, val_score, val_pred)

    calibrated_test_df = test_df.copy()
    calibrated_test_df["pred_label_chr_conditioned"] = test_pred.astype(int)
    calibrated_test_df["used_threshold_chr_conditioned"] = test_used_thresholds.astype(float)

    return {
        "best_quantile_from_val": float(best_row["quantile"]),
        "fallback_threshold": fallback_threshold,
        "best_chr_thresholds": best_thresholds,
        "val_metrics_best_chr": best_val_metrics,
        "test_metrics_best_chr": best_test_metrics,
        "chr_quantile_sweep": rows,
        "calibrated_test_df": calibrated_test_df,
    }


def write_summary_table(output_path, score_mode, global_eval, chr_eval):
    lines = [
        "# P12 Chromosome-Conditioned Posthoc Summary",
        "",
        f"- score_mode: `{score_mode}`",
        f"- global val-best threshold: `{global_eval['val_best_threshold']:.6f}`",
        f"- chr-conditioned best quantile from val: `{chr_eval['best_quantile_from_val']:.4f}`",
        "",
        "| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |",
        "|---|---:|---:|---:|---:|",
        (
            "| Global val-best | "
            f"{global_eval['test_metrics_val_best']['f1']:.4f} | "
            f"{global_eval['test_metrics_val_best']['precision_abnormal']:.4f} | "
            f"{global_eval['test_metrics_val_best']['recall_abnormal']:.4f} | "
            f"{global_eval['test_metrics_val_best']['balanced_acc']:.4f} |"
        ),
        (
            "| Chr-conditioned val-best | "
            f"{chr_eval['test_metrics_best_chr']['f1']:.4f} | "
            f"{chr_eval['test_metrics_best_chr']['precision_abnormal']:.4f} | "
            f"{chr_eval['test_metrics_best_chr']['recall_abnormal']:.4f} | "
            f"{chr_eval['test_metrics_best_chr']['balanced_acc']:.4f} |"
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_predictions", required=True)
    parser.add_argument("--val_predictions", required=True)
    parser.add_argument("--test_predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_mode", default="chr_zscore", choices=["raw", "chr_zscore", "chr_robust_zscore"])
    parser.add_argument("--quantiles", default="0.95,0.975,0.99")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantiles = parse_float_list(args.quantiles)
    for quantile in quantiles:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Invalid quantile: {quantile}")

    train_df = load_prediction_csv(args.train_predictions)
    val_df = load_prediction_csv(args.val_predictions)
    test_df = load_prediction_csv(args.test_predictions)

    chr_stats, global_stats = build_chr_normal_stats(train_df)

    train_df = apply_chr_calibration(train_df, chr_stats, global_stats, args.score_mode)
    val_df = apply_chr_calibration(val_df, chr_stats, global_stats, args.score_mode)
    test_df = apply_chr_calibration(test_df, chr_stats, global_stats, args.score_mode)

    global_eval = run_global_threshold_eval(train_df, val_df, test_df, quantiles)
    chr_eval = run_chr_threshold_eval(train_df, val_df, test_df, quantiles)
    calibrated_test_df = chr_eval.pop("calibrated_test_df")

    calibrated_test_df["pred_label_global_valbest"] = (
        calibrated_test_df["calibrated_score"].astype(float) >= float(global_eval["val_best_threshold"])
    ).astype(int)

    if "subtype_status" in calibrated_test_df.columns:
        test_seen_global = summarize_subset_constant_threshold(
            calibrated_test_df[calibrated_test_df["subtype_status"] == "seen"],
            global_eval["val_best_threshold"],
        )
        test_unseen_global = summarize_subset_constant_threshold(
            calibrated_test_df[calibrated_test_df["subtype_status"] == "unseen"],
            global_eval["val_best_threshold"],
        )
        test_seen_chr = summarize_subset_chr_threshold(
            calibrated_test_df[calibrated_test_df["subtype_status"] == "seen"],
            chr_eval["best_chr_thresholds"],
            chr_eval["fallback_threshold"],
        )
        test_unseen_chr = summarize_subset_chr_threshold(
            calibrated_test_df[calibrated_test_df["subtype_status"] == "unseen"],
            chr_eval["best_chr_thresholds"],
            chr_eval["fallback_threshold"],
        )
    else:
        test_seen_global = None
        test_unseen_global = None
        test_seen_chr = None
        test_unseen_chr = None

    results = {
        "method": "p12_chr_conditioned_posthoc",
        "score_mode": args.score_mode,
        "quantiles": quantiles,
        "global_normal_stats": global_stats,
        "chr_normal_stats": chr_stats,
        "global_eval": global_eval,
        "chr_eval": chr_eval,
        "test_seen_global_valbest": test_seen_global,
        "test_unseen_global_valbest": test_unseen_global,
        "test_seen_chr_conditioned": test_seen_chr,
        "test_unseen_chr_conditioned": test_unseen_chr,
        "test_by_subtype_global_valbest": summarize_by_subtype(
            calibrated_test_df,
            pred_column="pred_label_global_valbest",
            score_column="calibrated_score",
        ),
        "test_by_subtype_chr_conditioned": summarize_by_subtype(
            calibrated_test_df,
            pred_column="pred_label_chr_conditioned",
            score_column="calibrated_score",
        ),
    }

    train_df.to_csv(output_dir / "train_predictions_chr_calibrated.csv", index=False)
    val_df.to_csv(output_dir / "val_predictions_chr_calibrated.csv", index=False)
    calibrated_test_df.to_csv(output_dir / "test_predictions_chr_calibrated.csv", index=False)
    pd.DataFrame(global_eval["global_quantile_sweep"]).to_csv(output_dir / "global_quantile_sweep.csv", index=False)

    chr_sweep_rows = []
    for row in chr_eval["chr_quantile_sweep"]:
        copied = dict(row)
        copied["thresholds"] = yaml.safe_dump(
            to_serializable(copied["thresholds"]),
            allow_unicode=True,
            sort_keys=True,
        ).strip()
        chr_sweep_rows.append(copied)
    pd.DataFrame(chr_sweep_rows).to_csv(output_dir / "chr_quantile_sweep.csv", index=False)

    with open(output_dir / "results.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    write_summary_table(output_dir / "summary_table.md", args.score_mode, global_eval, chr_eval)

    print(f"Saved chromosome-conditioned posthoc results to {output_dir / 'results.yaml'}")
    print(f"Saved global sweep to {output_dir / 'global_quantile_sweep.csv'}")
    print(f"Saved chromosome-conditioned sweep to {output_dir / 'chr_quantile_sweep.csv'}")


if __name__ == "__main__":
    main()
