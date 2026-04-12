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


def load_prediction_csv(path):
    df = pd.read_csv(path).copy()
    return df


def choose_score_column(df, preferred_column=None, fallback_candidates=None):
    if preferred_column is not None:
        if preferred_column not in df.columns:
            raise ValueError(f"Requested score column '{preferred_column}' not found. Available columns: {list(df.columns)}")
        return preferred_column

    fallback_candidates = fallback_candidates or []
    for column in fallback_candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Could not infer score column. Available columns: {list(df.columns)}")


def build_merge_key(df):
    for keys in [
        ["case_id", "pair_key"],
        ["pair_key"],
        ["left_path", "right_path"],
    ]:
        if all(key in df.columns for key in keys):
            return keys
    raise ValueError(f"Could not find merge keys in columns: {list(df.columns)}")


def merge_predictions(p12_df, band_df, p12_score_col, band_score_col):
    merge_keys = build_merge_key(p12_df)
    if not all(key in band_df.columns for key in merge_keys):
        raise ValueError(f"Band prediction file missing merge keys {merge_keys}. Available columns: {list(band_df.columns)}")

    p12_keep_cols = merge_keys + [
        "label",
        "chromosome_id",
        "abnormal_subtype_id",
        "subtype_status",
        p12_score_col,
    ]
    band_keep_cols = merge_keys + [band_score_col]

    p12_subset = p12_df[[col for col in p12_keep_cols if col in p12_df.columns]].copy()
    band_subset = band_df[[col for col in band_keep_cols if col in band_df.columns]].copy()
    merged = p12_subset.merge(band_subset, on=merge_keys, how="inner", suffixes=("_p12", "_band"))

    if merged.empty:
        raise ValueError("Merged prediction dataframe is empty. Check that the two prediction files use the same split and key columns.")

    merged = merged.rename(
        columns={
            p12_score_col: "p12_score",
            band_score_col: "band_score",
        }
    )
    return merged, merge_keys


def fit_normalizer(values, method):
    values = np.asarray(values, dtype=np.float64)
    if method == "none":
        return {"method": method}
    if method == "zscore":
        mean = float(values.mean())
        std = float(values.std())
        if std <= 1e-12:
            std = 1.0
        return {"method": method, "mean": mean, "std": std}
    if method == "minmax":
        min_value = float(values.min())
        max_value = float(values.max())
        scale = max(max_value - min_value, 1e-12)
        return {"method": method, "min": min_value, "scale": scale}
    if method == "rank":
        sorted_values = np.sort(values.astype(np.float64))
        return {"method": method, "sorted_values": sorted_values.tolist()}
    raise ValueError(f"Unsupported normalize method: {method}")


def apply_normalizer(values, params):
    values = np.asarray(values, dtype=np.float64)
    method = params["method"]
    if method == "none":
        return values
    if method == "zscore":
        return (values - float(params["mean"])) / float(params["std"])
    if method == "minmax":
        return (values - float(params["min"])) / float(params["scale"])
    if method == "rank":
        sorted_values = np.asarray(params["sorted_values"], dtype=np.float64)
        if sorted_values.size <= 1:
            return np.full(values.shape, 0.5, dtype=np.float64)
        positions = np.searchsorted(sorted_values, values, side="right")
        return positions.astype(np.float64) / float(sorted_values.size)
    raise ValueError(f"Unsupported normalize method: {method}")


def fuse_scores(p12_scores, band_scores, alpha):
    alpha = float(alpha)
    return alpha * np.asarray(p12_scores, dtype=np.float64) + (1.0 - alpha) * np.asarray(band_scores, dtype=np.float64)


def summarize_subset(df, score_column, threshold):
    if df.empty:
        return {"count": 0}
    metrics = compute_score_based_metrics(
        y_true=df["label"].astype(int).tolist(),
        y_score=df[score_column].astype(float).tolist(),
        threshold=float(threshold),
        higher_score_more_positive=True,
    )
    metrics["count"] = int(len(df))
    return metrics


def summarize_by_subtype(df, score_column, threshold):
    rows = []
    if "abnormal_subtype_id" not in df.columns:
        return rows
    abnormal_df = df[df["label"].astype(int) == 1].copy()
    if abnormal_df.empty:
        return rows

    for subtype_id, group in abnormal_df.groupby("abnormal_subtype_id", dropna=False):
        scores = group[score_column].astype(float).to_numpy()
        pred = (scores >= float(threshold)).astype(int)
        rows.append(
            {
                "abnormal_subtype_id": "" if pd.isna(subtype_id) else str(subtype_id),
                "chromosome_id": str(group["chromosome_id"].iloc[0]) if "chromosome_id" in group.columns else "",
                "subtype_status": str(group["subtype_status"].iloc[0]) if "subtype_status" in group.columns else "",
                "count": int(len(group)),
                "recall_at_threshold": float(pred.mean()),
                "mean_score": float(scores.mean()),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p12_val_csv", required=True)
    parser.add_argument("--p12_test_csv", required=True)
    parser.add_argument("--band_val_csv", required=True)
    parser.add_argument("--band_test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p12_score_col", default=None)
    parser.add_argument("--band_score_col", default="score")
    parser.add_argument("--normalize", default="rank", choices=["none", "zscore", "minmax", "rank"])
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p12_val_df = load_prediction_csv(args.p12_val_csv)
    p12_test_df = load_prediction_csv(args.p12_test_csv)
    band_val_df = load_prediction_csv(args.band_val_csv)
    band_test_df = load_prediction_csv(args.band_test_csv)

    p12_score_col = choose_score_column(
        p12_val_df,
        preferred_column=args.p12_score_col,
        fallback_candidates=["calibrated_score", "casewise_score", "anomaly_score", "score"],
    )
    band_score_col = choose_score_column(
        band_val_df,
        preferred_column=args.band_score_col,
        fallback_candidates=["score", "casewise_score", "calibrated_score", "anomaly_score"],
    )

    val_df, merge_keys = merge_predictions(p12_val_df, band_val_df, p12_score_col, band_score_col)
    test_df, _ = merge_predictions(p12_test_df, band_test_df, p12_score_col, band_score_col)

    p12_norm = fit_normalizer(val_df["p12_score"].to_numpy(), args.normalize)
    band_norm = fit_normalizer(val_df["band_score"].to_numpy(), args.normalize)
    val_df["p12_score_norm"] = apply_normalizer(val_df["p12_score"].to_numpy(), p12_norm)
    val_df["band_score_norm"] = apply_normalizer(val_df["band_score"].to_numpy(), band_norm)
    test_df["p12_score_norm"] = apply_normalizer(test_df["p12_score"].to_numpy(), p12_norm)
    test_df["band_score_norm"] = apply_normalizer(test_df["band_score"].to_numpy(), band_norm)

    alphas = parse_float_list(args.alphas)
    sweep_rows = []
    best_alpha = None
    best_threshold = None
    best_score = -1.0
    best_stats = None

    for alpha in alphas:
        val_fused = fuse_scores(val_df["p12_score_norm"].to_numpy(), val_df["band_score_norm"].to_numpy(), alpha)
        val_threshold, val_score, val_stats = search_best_threshold(
            y_true=val_df["label"].astype(int).tolist(),
            y_score=val_fused.tolist(),
            metric="f1",
            higher_score_more_positive=True,
        )
        test_fused = fuse_scores(test_df["p12_score_norm"].to_numpy(), test_df["band_score_norm"].to_numpy(), alpha)
        test_metrics = compute_score_based_metrics(
            y_true=test_df["label"].astype(int).tolist(),
            y_score=test_fused.tolist(),
            threshold=val_threshold,
            higher_score_more_positive=True,
        )
        sweep_rows.append(
            {
                "alpha_p12": float(alpha),
                "alpha_band": float(1.0 - alpha),
                "val_best_threshold": float(val_threshold),
                "val_best_f1": float(val_score),
                "val_best_precision": float(val_stats["precision"]),
                "val_best_recall": float(val_stats["recall"]),
                "val_best_balanced_acc": float(val_stats["balanced_acc"]),
                "test_f1": float(test_metrics["f1"]),
                "test_precision_abnormal": float(test_metrics["precision_abnormal"]),
                "test_recall_abnormal": float(test_metrics["recall_abnormal"]),
                "test_balanced_acc": float(test_metrics["balanced_acc"]),
                "test_auprc": float(test_metrics["auprc"]) if test_metrics["auprc"] is not None else None,
                "test_auroc": float(test_metrics["auroc"]) if test_metrics["auroc"] is not None else None,
            }
        )
        if val_score > best_score:
            best_alpha = float(alpha)
            best_threshold = float(val_threshold)
            best_score = float(val_score)
            best_stats = val_stats

    val_df["fusion_score"] = fuse_scores(val_df["p12_score_norm"].to_numpy(), val_df["band_score_norm"].to_numpy(), best_alpha)
    test_df["fusion_score"] = fuse_scores(test_df["p12_score_norm"].to_numpy(), test_df["band_score_norm"].to_numpy(), best_alpha)

    val_metrics_best = compute_score_based_metrics(
        y_true=val_df["label"].astype(int).tolist(),
        y_score=val_df["fusion_score"].astype(float).tolist(),
        threshold=best_threshold,
        higher_score_more_positive=True,
    )
    test_metrics_best = compute_score_based_metrics(
        y_true=test_df["label"].astype(int).tolist(),
        y_score=test_df["fusion_score"].astype(float).tolist(),
        threshold=best_threshold,
        higher_score_more_positive=True,
    )
    p12_test_metrics = compute_score_based_metrics(
        y_true=test_df["label"].astype(int).tolist(),
        y_score=test_df["p12_score_norm"].astype(float).tolist(),
        threshold=search_best_threshold(
            y_true=val_df["label"].astype(int).tolist(),
            y_score=val_df["p12_score_norm"].astype(float).tolist(),
            metric="f1",
            higher_score_more_positive=True,
        )[0],
        higher_score_more_positive=True,
    )
    band_test_metrics = compute_score_based_metrics(
        y_true=test_df["label"].astype(int).tolist(),
        y_score=test_df["band_score_norm"].astype(float).tolist(),
        threshold=search_best_threshold(
            y_true=val_df["label"].astype(int).tolist(),
            y_score=val_df["band_score_norm"].astype(float).tolist(),
            metric="f1",
            higher_score_more_positive=True,
        )[0],
        higher_score_more_positive=True,
    )

    val_df["pred_label_fusion"] = (val_df["fusion_score"].astype(float) >= float(best_threshold)).astype(int)
    test_df["pred_label_fusion"] = (test_df["fusion_score"].astype(float) >= float(best_threshold)).astype(int)
    val_df.to_csv(output_dir / "val_fused_predictions.csv", index=False)
    test_df.to_csv(output_dir / "test_fused_predictions.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(output_dir / "fusion_sweep.csv", index=False)

    results = {
        "method": "p12_bandconv_late_fusion",
        "merge_keys": merge_keys,
        "p12_score_col": p12_score_col,
        "band_score_col": band_score_col,
        "normalize": args.normalize,
        "best_alpha_p12": float(best_alpha),
        "best_alpha_band": float(1.0 - best_alpha),
        "best_threshold": float(best_threshold),
        "best_val_f1": float(best_score),
        "best_val_stats": best_stats,
        "val_metrics_best": val_metrics_best,
        "test_metrics_best": test_metrics_best,
        "test_p12_only_metrics": p12_test_metrics,
        "test_band_only_metrics": band_test_metrics,
        "test_seen_best": summarize_subset(test_df[test_df["subtype_status"] == "seen"], "fusion_score", best_threshold)
        if "subtype_status" in test_df.columns
        else None,
        "test_unseen_best": summarize_subset(test_df[test_df["subtype_status"] == "unseen"], "fusion_score", best_threshold)
        if "subtype_status" in test_df.columns
        else None,
        "test_by_subtype_best": summarize_by_subtype(test_df, "fusion_score", best_threshold),
        "sweep": sweep_rows,
    }

    with open(output_dir / "results.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    summary_lines = [
        "# P12 + BandConv Late Fusion",
        "",
        "## Settings",
        f"- p12_score_col: `{p12_score_col}`",
        f"- band_score_col: `{band_score_col}`",
        f"- normalize: `{args.normalize}`",
        f"- best_alpha_p12: `{best_alpha:.2f}`",
        f"- best_alpha_band: `{1.0 - best_alpha:.2f}`",
        f"- best_threshold: `{best_threshold:.6f}`",
        "",
        "## Test",
        f"- fusion_f1: `{test_metrics_best['f1']:.4f}`",
        f"- fusion_precision_abnormal: `{test_metrics_best['precision_abnormal']:.4f}`",
        f"- fusion_recall_abnormal: `{test_metrics_best['recall_abnormal']:.4f}`",
        f"- fusion_balanced_acc: `{test_metrics_best['balanced_acc']:.4f}`",
        f"- fusion_auprc: `{test_metrics_best['auprc']:.4f}`",
        f"- fusion_auroc: `{test_metrics_best['auroc']:.4f}`",
        "",
        "## Baselines on same aligned subset",
        f"- p12_only_f1: `{p12_test_metrics['f1']:.4f}`",
        f"- band_only_f1: `{band_test_metrics['f1']:.4f}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved late fusion results to {output_dir / 'results.yaml'}")
    print(
        "Final fusion test metrics:",
        {
            "f1": round(float(test_metrics_best["f1"]), 4),
            "precision_abnormal": round(float(test_metrics_best["precision_abnormal"]), 4),
            "recall_abnormal": round(float(test_metrics_best["recall_abnormal"]), 4),
            "balanced_acc": round(float(test_metrics_best["balanced_acc"]), 4),
            "auprc": round(float(test_metrics_best["auprc"]), 4),
            "auroc": round(float(test_metrics_best["auroc"]), 4),
        },
    )


if __name__ == "__main__":
    main()
