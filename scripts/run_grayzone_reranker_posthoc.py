import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.main import (
    _forward_model,
    _safe_load_state_dict,
    build_eval_loader_for_csv,
    load_config,
    set_seed,
)
from src.models.build_model import build_model
from src.utils.chromosome_vocab import build_chr_vocab_from_csv
from src.utils.metrics import compute_score_based_metrics, search_best_threshold


def to_serializable(value):
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_model_from_config(cfg, chr_to_idx, device):
    return build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        use_chromosome_id=cfg["model"].get("use_chromosome_id", False),
        num_chromosome_types=len(chr_to_idx) if chr_to_idx is not None else None,
        chr_embed_dim=cfg["model"].get("chr_embed_dim", 16),
        use_pair_input=cfg["model"].get("use_pair_input", False),
        pair_model_type=cfg["model"].get("pair_model_type", "siamese"),
        use_pair_mixstyle=cfg["model"].get("use_pair_mixstyle", False),
        mixstyle_p=cfg["model"].get("mixstyle_p", 0.5),
        mixstyle_alpha=cfg["model"].get("mixstyle_alpha", 0.1),
        experiment_mode=cfg.get("experiment_mode", "classifier"),
        num_prototypes=cfg["model"].get("num_prototypes", 4),
        prototype_distance=cfg["model"].get("prototype_distance", "cosine"),
        normalize_prototype_embedding=cfg["model"].get("normalize_prototype_embedding", True),
        use_side_head=cfg["model"].get("use_side_head", False),
        num_side_classes=cfg["model"].get("num_side_classes", 2),
    ).to(device)


def default_save_dir(config_path, ckpt_path):
    config_stem = Path(config_path).stem
    ckpt_parent = Path(ckpt_path).resolve().parent.name
    project_root = Path(__file__).resolve().parents[1]
    return str(project_root / "outputs" / "posthoc_eval" / f"{ckpt_parent}__{config_stem}__grayzone_reranker")


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


def safe_tail_stats(values, tail_quantile):
    values = np.asarray(values, dtype=np.float64)
    tail_threshold = float(np.quantile(values, tail_quantile))
    tail_values = values[values >= tail_threshold]
    tail_excess = tail_values - tail_threshold
    tail_std_excess = float(tail_excess.std()) if len(tail_excess) > 0 else 0.0
    if tail_std_excess <= 1e-12:
        tail_std_excess = 1.0
    return {
        "tail_quantile": float(tail_quantile),
        "tail_threshold": tail_threshold,
        "tail_std_excess": tail_std_excess,
    }


def empirical_percentile(raw_score, sorted_scores):
    sorted_scores = np.asarray(sorted_scores, dtype=np.float64)
    if sorted_scores.size == 0:
        return 0.5
    rank = int(np.searchsorted(sorted_scores, float(raw_score), side="right"))
    return rank / float(sorted_scores.size)


def summarize_normal_distribution(values, tail_quantile):
    values = np.asarray(values, dtype=np.float64)
    mean, std = safe_mean_std(values)
    median, mad = safe_median_mad(values)
    tail_stats = safe_tail_stats(values, tail_quantile)
    return {
        "count": int(len(values)),
        "mean": mean,
        "std": std,
        "median": median,
        "mad": mad,
        **tail_stats,
    }


def build_chr_score_context(rows, tail_quantile):
    normal_rows = [row for row in rows if int(row["label"]) == 0]
    if not normal_rows:
        raise ValueError("Expected normal training rows to build chromosome score context")

    global_scores = np.asarray([row["raw_anomaly_score"] for row in normal_rows], dtype=np.float64)
    global_stats = summarize_normal_distribution(global_scores, tail_quantile)
    global_context = dict(global_stats)
    global_context["sorted_scores"] = np.sort(global_scores)

    chr_context = {}
    for chromosome_id in sorted({str(row["chromosome_id"]) for row in normal_rows}):
        scores = np.asarray(
            [row["raw_anomaly_score"] for row in normal_rows if str(row["chromosome_id"]) == chromosome_id],
            dtype=np.float64,
        )
        stats = summarize_normal_distribution(scores, tail_quantile)
        chr_context[chromosome_id] = {
            **stats,
            "sorted_scores": np.sort(scores),
        }

    return chr_context, global_context


def calibrate_score(raw_score, stats, score_mode):
    raw_score = float(raw_score)
    if score_mode == "raw":
        return raw_score
    if score_mode == "chr_zscore":
        return (raw_score - float(stats["mean"])) / max(float(stats["std"]), 1e-12)
    if score_mode == "chr_robust_zscore":
        return (raw_score - float(stats["median"])) / max(1.4826 * float(stats["mad"]), 1e-12)
    if score_mode == "chr_percentile":
        return empirical_percentile(raw_score, stats["sorted_scores"])
    if score_mode == "chr_tail_zscore":
        percentile = empirical_percentile(raw_score, stats["sorted_scores"])
        tail_threshold = float(stats["tail_threshold"])
        if raw_score <= tail_threshold:
            return percentile
        tail_std_excess = max(float(stats["tail_std_excess"]), 1e-12)
        tail_quantile = float(stats["tail_quantile"])
        return tail_quantile + (raw_score - tail_threshold) / tail_std_excess
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def _normalize_batch_meta(batch, metadata_keys, batch_size):
    normalized = {}
    for key in metadata_keys:
        if key not in batch:
            continue
        values = batch[key]
        if isinstance(values, torch.Tensor):
            normalized[key] = values.detach().cpu().tolist()
        elif isinstance(values, np.ndarray):
            normalized[key] = values.tolist()
        elif isinstance(values, (list, tuple)):
            normalized[key] = list(values)
        else:
            normalized[key] = [values] * batch_size

        if len(normalized[key]) != batch_size:
            normalized[key] = [normalized[key][0]] * batch_size
    return normalized


def _build_row_uid(row):
    return "||".join(
        [
            str(row.get("case_id", "")),
            str(row.get("pair_key", "")),
            str(row.get("left_path", "")),
            str(row.get("right_path", "")),
            str(row.get("chromosome_id", "")),
        ]
    )


def _tensor_to_numpy(outputs, key):
    value = outputs.get(key)
    if value is None or not isinstance(value, torch.Tensor):
        return None
    return value.detach().cpu().numpy()


@torch.no_grad()
def collect_rich_rows(model, loader, device, use_chromosome_id=False, use_pair_input=False):
    model.eval()
    rows = []
    metadata_keys = [
        "case_id",
        "pair_key",
        "chromosome_id",
        "abnormal_subtype_id",
        "subtype_status",
        "left_filename",
        "right_filename",
        "split",
        "left_path",
        "right_path",
    ]

    for batch in loader:
        labels = batch["label"].to(device)
        outputs = _forward_model(
            batch=batch,
            model=model,
            device=device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
            use_style_view=False,
        )
        if not isinstance(outputs, dict):
            raise ValueError("Expected dict model outputs for reranker feature extraction")
        if "embedding" not in outputs or "anomaly_score" not in outputs:
            raise ValueError("Model output must contain 'embedding' and 'anomaly_score'")

        embeddings = outputs["embedding"].detach().cpu().numpy()
        anomaly_scores = outputs["anomaly_score"].detach().cpu().numpy()
        prototype_distances = _tensor_to_numpy(outputs, "prototype_distances")
        pair_distance = _tensor_to_numpy(outputs, "pair_distance")
        pair_consistency_direct = _tensor_to_numpy(outputs, "pair_consistency_direct")
        pair_consistency_reverse = _tensor_to_numpy(outputs, "pair_consistency_reverse")
        reverse_gain = _tensor_to_numpy(outputs, "reverse_gain")

        batch_size = labels.size(0)
        normalized_meta = _normalize_batch_meta(batch, metadata_keys, batch_size)

        for idx in range(batch_size):
            proto = prototype_distances[idx] if prototype_distances is not None else None
            proto = np.asarray(proto, dtype=np.float64) if proto is not None else np.zeros(1, dtype=np.float64)
            proto_sorted = np.sort(proto)
            proto_min = float(proto_sorted[0])
            proto_second = float(proto_sorted[1]) if proto_sorted.size > 1 else proto_min

            row = {
                "label": int(labels[idx].item()),
                "raw_anomaly_score": float(anomaly_scores[idx]),
                "embedding_norm": float(np.linalg.norm(embeddings[idx])),
                "prototype_min_dist": proto_min,
                "prototype_second_dist": proto_second,
                "prototype_gap12": float(proto_second - proto_min),
                "prototype_mean_dist": float(proto.mean()),
                "prototype_std_dist": float(proto.std()),
                "prototype_max_dist": float(proto.max()),
                "pair_distance": float(pair_distance[idx]) if pair_distance is not None else 0.0,
                "pair_consistency_direct": float(pair_consistency_direct[idx]) if pair_consistency_direct is not None else 0.0,
                "pair_consistency_reverse": float(pair_consistency_reverse[idx]) if pair_consistency_reverse is not None else 0.0,
                "reverse_gain": float(reverse_gain[idx]) if reverse_gain is not None else 0.0,
            }
            row["pair_consistency_gap"] = row["pair_consistency_reverse"] - row["pair_consistency_direct"]
            row["pair_consistency_abs_gap"] = abs(row["pair_consistency_gap"])

            for key, values in normalized_meta.items():
                value = values[idx] if idx < len(values) else ""
                row[key] = "" if value is None else value

            row["row_uid"] = _build_row_uid(row)
            rows.append(row)

    return rows


def apply_score_calibrations(rows, chr_context, global_context, tail_quantile):
    enriched_rows = []
    for row in rows:
        chromosome_id = str(row.get("chromosome_id", ""))
        stats = chr_context.get(chromosome_id, global_context)
        raw = float(row["raw_anomaly_score"])
        enriched = dict(row)
        enriched["score_raw"] = raw
        enriched["score_chr_zscore"] = calibrate_score(raw, stats, "chr_zscore")
        enriched["score_chr_robust_zscore"] = calibrate_score(raw, stats, "chr_robust_zscore")
        enriched["score_chr_percentile"] = calibrate_score(raw, stats, "chr_percentile")
        enriched["score_chr_tail_zscore"] = calibrate_score(raw, stats, "chr_tail_zscore")
        enriched["tail_quantile"] = float(tail_quantile)
        enriched_rows.append(enriched)
    return enriched_rows


def build_feature_dataframe(rows):
    df = pd.DataFrame(rows).copy()
    if df.empty:
        raise ValueError("No rows collected for reranker")

    numeric_cols = [
        "label",
        "score_raw",
        "score_chr_zscore",
        "score_chr_robust_zscore",
        "score_chr_percentile",
        "score_chr_tail_zscore",
        "pair_distance",
        "pair_consistency_direct",
        "pair_consistency_reverse",
        "reverse_gain",
        "pair_consistency_gap",
        "pair_consistency_abs_gap",
        "prototype_min_dist",
        "prototype_second_dist",
        "prototype_gap12",
        "prototype_mean_dist",
        "prototype_std_dist",
        "prototype_max_dist",
        "embedding_norm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["label"] = df["label"].astype(int)
    df["chromosome_id"] = df["chromosome_id"].fillna("").astype(str)
    return df


def get_base_score_column(base_score_mode):
    mapping = {
        "raw": "score_raw",
        "chr_zscore": "score_chr_zscore",
        "chr_robust_zscore": "score_chr_robust_zscore",
        "chr_percentile": "score_chr_percentile",
        "chr_tail_zscore": "score_chr_tail_zscore",
    }
    return mapping[base_score_mode]


def assign_base_predictions(df, score_col, threshold):
    return (df[score_col].astype(float).to_numpy() >= float(threshold)).astype(int)


def compute_gray_zone_mask(df, score_col, threshold, ratio):
    distances = np.abs(df[score_col].astype(float).to_numpy() - float(threshold))
    margin = float(np.quantile(distances, float(ratio)))
    return distances <= margin, margin


def build_reranker_feature_matrix(df, feature_cols, use_chr_onehot=True):
    feature_df = df[feature_cols].copy()
    if use_chr_onehot:
        chr_onehot = pd.get_dummies(df["chromosome_id"].astype(str), prefix="chr")
        feature_df = pd.concat([feature_df, chr_onehot], axis=1)
    return feature_df.astype(float)


def metrics_from_binary_predictions(y_true, y_pred, score_reference):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    score_reference = np.asarray(score_reference, dtype=np.float64)

    metrics = compute_score_based_metrics(
        y_true=y_true.tolist(),
        y_score=score_reference.tolist(),
        threshold=0.5,
        higher_score_more_positive=True,
    )

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


def search_best_probability_threshold(base_pred, gray_mask, y_true, rerank_prob, score_reference):
    best_threshold = 0.5
    best_score = -1.0
    best_stats = {}
    candidates = np.unique(np.asarray(rerank_prob, dtype=np.float64))
    if candidates.size > 200:
        candidates = np.linspace(float(candidates.min()), float(candidates.max()), num=200)

    y_true = np.asarray(y_true, dtype=np.int64)
    base_pred = np.asarray(base_pred, dtype=np.int64)
    gray_mask = np.asarray(gray_mask, dtype=bool)
    rerank_prob = np.asarray(rerank_prob, dtype=np.float64)
    score_reference = np.asarray(score_reference, dtype=np.float64)

    for threshold in candidates:
        combined_pred = base_pred.copy()
        combined_pred[gray_mask] = (rerank_prob[gray_mask] >= float(threshold)).astype(np.int64)
        metrics = metrics_from_binary_predictions(y_true, combined_pred, score_reference)
        if float(metrics["f1"]) > best_score:
            best_score = float(metrics["f1"])
            best_threshold = float(threshold)
            best_stats = {
                "precision": float(metrics["precision_abnormal"]),
                "recall": float(metrics["recall_abnormal"]),
                "f1": float(metrics["f1"]),
                "balanced_acc": float(metrics["balanced_acc"]),
            }

    return best_threshold, best_score, best_stats


def summarize_by_subtype(df, pred_col):
    rows = []
    if "abnormal_subtype_id" not in df.columns:
        return rows
    abnormal_df = df[df["label"].astype(int) == 1].copy()
    if abnormal_df.empty:
        return rows
    for subtype, group in abnormal_df.groupby("abnormal_subtype_id", dropna=False):
        rows.append(
            {
                "abnormal_subtype_id": "" if pd.isna(subtype) else str(subtype),
                "chromosome_id": str(group["chromosome_id"].iloc[0]),
                "subtype_status": str(group["subtype_status"].iloc[0]) if "subtype_status" in group.columns else "",
                "count": int(len(group)),
                "recall_at_threshold": float(group[pred_col].astype(int).mean()),
                "gray_zone_count": int(group["is_gray_zone"].astype(int).sum()) if "is_gray_zone" in group.columns else 0,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument(
        "--base_score_mode",
        default="chr_tail_zscore",
        choices=["raw", "chr_zscore", "chr_robust_zscore", "chr_percentile", "chr_tail_zscore"],
    )
    parser.add_argument("--tail_quantile", type=float, default=0.95)
    parser.add_argument("--gray_zone_ratio", type=float, default=0.2)
    parser.add_argument("--reranker_c", type=float, default=1.0)
    parser.add_argument("--reranker_max_iter", type=int, default=2000)
    parser.add_argument("--disable_chr_onehot", action="store_true")
    args = parser.parse_args()

    if not 0.0 < float(args.tail_quantile) < 1.0:
        raise ValueError(f"Invalid tail_quantile: {args.tail_quantile}")
    if not 0.01 <= float(args.gray_zone_ratio) < 1.0:
        raise ValueError(f"Invalid gray_zone_ratio: {args.gray_zone_ratio}")

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    if cfg.get("experiment_mode") != "multi_prototype_metric":
        raise ValueError("This script only supports experiment_mode=multi_prototype_metric")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)
    if not use_chromosome_id or not use_pair_input:
        raise ValueError("This script expects chromosome-conditioned pair-input experiments")

    save_dir = args.save_dir or default_save_dir(args.config, args.ckpt)
    os.makedirs(save_dir, exist_ok=True)

    chr_to_idx, _ = build_chr_vocab_from_csv(cfg["data"]["train_csv"])
    model = build_model_from_config(cfg, chr_to_idx, device)
    _safe_load_state_dict(model, args.ckpt, device)

    train_loader = build_eval_loader_for_csv(cfg, cfg["data"]["train_csv"], chr_to_idx)
    val_loader = build_eval_loader_for_csv(cfg, cfg["data"]["val_csv"], chr_to_idx)
    test_loader = build_eval_loader_for_csv(cfg, cfg["data"]["test_csv"], chr_to_idx)

    train_rows = collect_rich_rows(model, train_loader, device, use_chromosome_id, use_pair_input)
    val_rows = collect_rich_rows(model, val_loader, device, use_chromosome_id, use_pair_input)
    test_rows = collect_rich_rows(model, test_loader, device, use_chromosome_id, use_pair_input)

    chr_context, global_context = build_chr_score_context(train_rows, args.tail_quantile)
    train_rows = apply_score_calibrations(train_rows, chr_context, global_context, args.tail_quantile)
    val_rows = apply_score_calibrations(val_rows, chr_context, global_context, args.tail_quantile)
    test_rows = apply_score_calibrations(test_rows, chr_context, global_context, args.tail_quantile)

    train_df = build_feature_dataframe(train_rows)
    val_df = build_feature_dataframe(val_rows)
    test_df = build_feature_dataframe(test_rows)

    base_score_col = get_base_score_column(args.base_score_mode)
    base_val_threshold, base_val_score, base_val_stats = search_best_threshold(
        val_df["label"].astype(int).tolist(),
        val_df[base_score_col].astype(float).tolist(),
        metric="f1",
        higher_score_more_positive=True,
    )

    val_df["base_pred"] = assign_base_predictions(val_df, base_score_col, base_val_threshold)
    test_df["base_pred"] = assign_base_predictions(test_df, base_score_col, base_val_threshold)

    val_gray_mask, gray_margin = compute_gray_zone_mask(val_df, base_score_col, base_val_threshold, args.gray_zone_ratio)
    test_gray_mask = np.abs(test_df[base_score_col].astype(float).to_numpy() - float(base_val_threshold)) <= float(gray_margin)
    val_df["is_gray_zone"] = val_gray_mask.astype(int)
    test_df["is_gray_zone"] = test_gray_mask.astype(int)

    feature_cols = [
        "score_raw",
        "score_chr_zscore",
        "score_chr_robust_zscore",
        "score_chr_percentile",
        "score_chr_tail_zscore",
        "pair_distance",
        "pair_consistency_direct",
        "pair_consistency_reverse",
        "reverse_gain",
        "pair_consistency_gap",
        "pair_consistency_abs_gap",
        "prototype_min_dist",
        "prototype_second_dist",
        "prototype_gap12",
        "prototype_mean_dist",
        "prototype_std_dist",
        "prototype_max_dist",
        "embedding_norm",
    ]

    fit_df = val_df[val_df["is_gray_zone"] == 1].copy()
    if fit_df["label"].nunique() < 2:
        raise ValueError("Gray-zone validation subset contains fewer than two classes. Increase --gray_zone_ratio.")

    X_fit_df = build_reranker_feature_matrix(fit_df, feature_cols, use_chr_onehot=not args.disable_chr_onehot)
    X_val_df = build_reranker_feature_matrix(val_df, feature_cols, use_chr_onehot=not args.disable_chr_onehot)
    X_test_df = build_reranker_feature_matrix(test_df, feature_cols, use_chr_onehot=not args.disable_chr_onehot)

    all_columns = sorted(set(X_fit_df.columns) | set(X_val_df.columns) | set(X_test_df.columns))
    X_fit_df = X_fit_df.reindex(columns=all_columns, fill_value=0.0)
    X_val_df = X_val_df.reindex(columns=all_columns, fill_value=0.0)
    X_test_df = X_test_df.reindex(columns=all_columns, fill_value=0.0)

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit_df.to_numpy(dtype=np.float64))
    X_val = scaler.transform(X_val_df.to_numpy(dtype=np.float64))
    X_test = scaler.transform(X_test_df.to_numpy(dtype=np.float64))

    reranker = LogisticRegression(
        C=float(args.reranker_c),
        max_iter=int(args.reranker_max_iter),
        class_weight="balanced",
        random_state=int(cfg.get("seed", 42)),
    )
    reranker.fit(X_fit, fit_df["label"].astype(int).to_numpy())

    val_rerank_prob = reranker.predict_proba(X_val)[:, 1]
    test_rerank_prob = reranker.predict_proba(X_test)[:, 1]
    val_df["rerank_prob"] = val_rerank_prob
    test_df["rerank_prob"] = test_rerank_prob
    fit_df = val_df[val_df["is_gray_zone"] == 1].copy()

    rerank_threshold, rerank_val_score, rerank_val_stats = search_best_probability_threshold(
        base_pred=val_df["base_pred"].astype(int).to_numpy(),
        gray_mask=val_gray_mask,
        y_true=val_df["label"].astype(int).to_numpy(),
        rerank_prob=val_rerank_prob,
        score_reference=val_df[base_score_col].astype(float).to_numpy(),
    )

    val_df["final_pred"] = val_df["base_pred"].astype(int)
    val_df.loc[val_df["is_gray_zone"] == 1, "final_pred"] = (
        val_df.loc[val_df["is_gray_zone"] == 1, "rerank_prob"].astype(float) >= float(rerank_threshold)
    ).astype(int)
    test_df["final_pred"] = test_df["base_pred"].astype(int)
    test_df.loc[test_df["is_gray_zone"] == 1, "final_pred"] = (
        test_df.loc[test_df["is_gray_zone"] == 1, "rerank_prob"].astype(float) >= float(rerank_threshold)
    ).astype(int)

    val_base_metrics = metrics_from_binary_predictions(
        val_df["label"].astype(int).to_numpy(),
        val_df["base_pred"].astype(int).to_numpy(),
        val_df[base_score_col].astype(float).to_numpy(),
    )
    test_base_metrics = metrics_from_binary_predictions(
        test_df["label"].astype(int).to_numpy(),
        test_df["base_pred"].astype(int).to_numpy(),
        test_df[base_score_col].astype(float).to_numpy(),
    )
    val_final_metrics = metrics_from_binary_predictions(
        val_df["label"].astype(int).to_numpy(),
        val_df["final_pred"].astype(int).to_numpy(),
        val_df[base_score_col].astype(float).to_numpy(),
    )
    test_final_metrics = metrics_from_binary_predictions(
        test_df["label"].astype(int).to_numpy(),
        test_df["final_pred"].astype(int).to_numpy(),
        test_df[base_score_col].astype(float).to_numpy(),
    )

    gray_val_metrics = metrics_from_binary_predictions(
        fit_df["label"].astype(int).to_numpy(),
        (fit_df["rerank_prob"].astype(float).to_numpy() >= float(rerank_threshold)).astype(int),
        fit_df["rerank_prob"].astype(float).to_numpy(),
    )
    gray_test_df = test_df[test_df["is_gray_zone"] == 1].copy()
    gray_test_metrics = (
        metrics_from_binary_predictions(
            gray_test_df["label"].astype(int).to_numpy(),
            (gray_test_df["rerank_prob"].astype(float).to_numpy() >= float(rerank_threshold)).astype(int),
            gray_test_df["rerank_prob"].astype(float).to_numpy(),
        )
        if not gray_test_df.empty
        else {"count": 0}
    )

    coef_pairs = sorted(
        zip(all_columns, reranker.coef_[0].tolist()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    results = {
        "method": "grayzone_reranker_posthoc",
        "config_path": args.config,
        "checkpoint_path": args.ckpt,
        "experiment_name": cfg.get("experiment_name"),
        "base_score_mode": args.base_score_mode,
        "tail_quantile": float(args.tail_quantile),
        "gray_zone_ratio": float(args.gray_zone_ratio),
        "gray_zone_margin": float(gray_margin),
        "base_val_threshold": float(base_val_threshold),
        "base_val_best_score": float(base_val_score),
        "base_val_best_stats": base_val_stats,
        "rerank_threshold_from_val": float(rerank_threshold),
        "rerank_val_best_score": float(rerank_val_score),
        "rerank_val_best_stats": rerank_val_stats,
        "fit_split": "val_gray_zone_only",
        "fit_counts": {
            "fit_total": int(len(fit_df)),
            "fit_normal": int((fit_df["label"] == 0).sum()),
            "fit_abnormal": int((fit_df["label"] == 1).sum()),
            "val_gray_zone_total": int(val_df["is_gray_zone"].sum()),
            "test_gray_zone_total": int(test_df["is_gray_zone"].sum()),
        },
        "feature_columns": all_columns,
        "top_coefficients": [{"feature": str(name), "coef": float(weight)} for name, weight in coef_pairs[:20]],
        "val_base_metrics": val_base_metrics,
        "test_base_metrics": test_base_metrics,
        "val_final_metrics": val_final_metrics,
        "test_final_metrics": test_final_metrics,
        "val_gray_reranker_metrics": gray_val_metrics,
        "test_gray_reranker_metrics": gray_test_metrics,
        "test_by_subtype_base": summarize_by_subtype(test_df, "base_pred"),
        "test_by_subtype_final": summarize_by_subtype(test_df, "final_pred"),
    }

    train_df.to_csv(os.path.join(save_dir, "train_rich_features.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val_rich_features.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test_rich_features.csv"), index=False)

    summary_lines = [
        "# Gray-Zone Reranker Summary",
        "",
        f"- base_score_mode: `{args.base_score_mode}`",
        f"- tail_quantile: `{args.tail_quantile}`",
        f"- gray_zone_ratio: `{args.gray_zone_ratio}`",
        f"- gray_zone_margin: `{gray_margin:.6f}`",
        f"- base val-best threshold: `{base_val_threshold:.6f}`",
        f"- rerank threshold from val: `{rerank_threshold:.6f}`",
        "",
        "| Setting | Test F1 | Precision_abn | Recall_abn | Balanced Acc | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            "| Base | "
            f"{test_base_metrics['f1']:.4f} | "
            f"{test_base_metrics['precision_abnormal']:.4f} | "
            f"{test_base_metrics['recall_abnormal']:.4f} | "
            f"{test_base_metrics['balanced_acc']:.4f} | "
            f"{test_base_metrics['confusion_matrix']['fp']} | "
            f"{test_base_metrics['confusion_matrix']['fn']} |"
        ),
        (
            "| Final reranked | "
            f"{test_final_metrics['f1']:.4f} | "
            f"{test_final_metrics['precision_abnormal']:.4f} | "
            f"{test_final_metrics['recall_abnormal']:.4f} | "
            f"{test_final_metrics['balanced_acc']:.4f} | "
            f"{test_final_metrics['confusion_matrix']['fp']} | "
            f"{test_final_metrics['confusion_matrix']['fn']} |"
        ),
    ]
    Path(save_dir, "summary_table.md").write_text("\n".join(summary_lines), encoding="utf-8")

    results_path = os.path.join(save_dir, "results.yaml")
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    print(f"Saved gray-zone reranker results to {results_path}")
    print(f"Saved train features to {os.path.join(save_dir, 'train_rich_features.csv')}")
    print(f"Saved val features to {os.path.join(save_dir, 'val_rich_features.csv')}")
    print(f"Saved test features to {os.path.join(save_dir, 'test_rich_features.csv')}")


if __name__ == "__main__":
    main()
