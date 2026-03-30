import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

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
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


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
    return str(project_root / "outputs" / "posthoc_eval" / f"{ckpt_parent}__{config_stem}__chr_mem_bank")


def _normalize_batch_meta(batch, metadata_keys, batch_size):
    normalized = {}
    for key in metadata_keys:
        if key not in batch:
            continue
        values = batch[key]
        if isinstance(values, list):
            normalized[key] = [str(v) for v in values]
        else:
            normalized[key] = [str(v) for v in values]

        if len(normalized[key]) != batch_size:
            normalized[key] = [normalized[key][0]] * batch_size
    return normalized


@torch.no_grad()
def collect_embeddings(model, loader, device, use_chromosome_id=False, use_pair_input=False):
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

        if not isinstance(outputs, dict) or "embedding" not in outputs:
            raise ValueError("Model output must be a dict containing 'embedding'")

        embeddings = outputs["embedding"].detach().cpu()
        anomaly_scores = None
        if "anomaly_score" in outputs:
            anomaly_scores = outputs["anomaly_score"].detach().cpu().numpy()

        batch_size = labels.size(0)
        normalized_meta = _normalize_batch_meta(batch, metadata_keys, batch_size)

        for idx in range(batch_size):
            row = {
                "label": int(labels[idx].item()),
                "embedding": embeddings[idx].numpy().astype(np.float32),
                "anomaly_score_model": float(anomaly_scores[idx]) if anomaly_scores is not None else None,
            }
            for key, values in normalized_meta.items():
                row[key] = values[idx]
            rows.append(row)

    return rows


def maybe_normalize_embeddings(embeddings, normalize):
    if not normalize:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def build_chr_memory_bank(rows, normalize_embeddings=True, max_per_chr=None, seed=42):
    rng = np.random.default_rng(seed)
    grouped = {}
    for row in rows:
        if int(row["label"]) != 0:
            continue
        chromosome_id = str(row["chromosome_id"])
        grouped.setdefault(chromosome_id, []).append(row["embedding"])

    memory_bank = {}
    for chromosome_id, emb_list in grouped.items():
        emb = np.stack(emb_list, axis=0).astype(np.float32)
        emb = maybe_normalize_embeddings(emb, normalize_embeddings)
        if max_per_chr is not None and emb.shape[0] > int(max_per_chr):
            indices = rng.choice(emb.shape[0], size=int(max_per_chr), replace=False)
            emb = emb[indices]
        memory_bank[chromosome_id] = emb

    if not memory_bank:
        raise ValueError("Memory bank is empty. Expected normal training samples.")
    return memory_bank


def compute_knn_distance(query_embedding, bank_embeddings, distance="cosine", knn_k=1):
    if distance == "cosine":
        sims = np.matmul(bank_embeddings, query_embedding)
        dists = 1.0 - sims
    elif distance == "euclidean":
        dists = np.linalg.norm(bank_embeddings - query_embedding[None, :], axis=1)
    else:
        raise ValueError(f"Unsupported distance: {distance}")

    actual_k = min(int(knn_k), int(dists.shape[0]))
    nearest = np.partition(dists, actual_k - 1)[:actual_k]
    return float(nearest.mean())


def score_rows_with_memory_bank(rows, memory_bank, normalize_embeddings=True, distance="cosine", knn_k=1):
    scored_rows = []

    all_bank_embeddings = np.concatenate(list(memory_bank.values()), axis=0)
    for row in rows:
        chromosome_id = str(row["chromosome_id"])
        bank = memory_bank.get(chromosome_id, all_bank_embeddings)

        query = np.asarray(row["embedding"], dtype=np.float32)
        if normalize_embeddings:
            denom = max(float(np.linalg.norm(query)), 1e-12)
            query = query / denom

        memory_score = compute_knn_distance(
            query_embedding=query,
            bank_embeddings=bank,
            distance=distance,
            knn_k=knn_k,
        )

        scored = dict(row)
        scored["memory_bank_score"] = float(memory_score)
        scored_rows.append(scored)

    return scored_rows


def records_to_dataframe(rows, score_column, threshold=None, pred_column="pred_label"):
    df = pd.DataFrame([{k: v for k, v in row.items() if k != "embedding"} for row in rows]).copy()
    if df.empty:
        return df
    if threshold is not None:
        df[pred_column] = (df[score_column].astype(float) >= float(threshold)).astype(int)
    return df


def summarize_score_binary(df, score_col, threshold):
    if df.empty:
        return {"count": 0}
    metrics = compute_score_based_metrics(
        y_true=df["label"].astype(int).tolist(),
        y_score=df[score_col].astype(float).tolist(),
        threshold=threshold,
        higher_score_more_positive=True,
    )
    metrics["count"] = int(len(df))
    return metrics


def summarize_by_subtype(df, score_col, threshold):
    rows = []
    if "abnormal_subtype_id" not in df.columns:
        return rows

    abnormal_df = df[df["label"].astype(int) == 1].copy()
    if abnormal_df.empty:
        return rows

    for subtype, group in abnormal_df.groupby("abnormal_subtype_id", dropna=False):
        scores = group[score_col].astype(float).to_numpy()
        pred = (scores >= threshold).astype(int)
        rows.append(
            {
                "abnormal_subtype_id": "" if pd.isna(subtype) else str(subtype),
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


def build_global_quantile_thresholds(train_df, quantiles):
    normal_scores = train_df.loc[train_df["label"].astype(int) == 0, "memory_bank_score"].astype(float).to_numpy()
    return {float(q): float(np.quantile(normal_scores, q)) for q in quantiles}


def compute_chr_quantile_thresholds(train_df, score_col, quantile):
    thresholds = {}
    normal_df = train_df[train_df["label"].astype(int) == 0].copy()
    for chromosome_id, group in normal_df.groupby("chromosome_id"):
        thresholds[str(chromosome_id)] = float(np.quantile(group[score_col].astype(float).to_numpy(), quantile))
    return thresholds


def apply_chr_thresholds(df, score_col, chr_thresholds, fallback_threshold):
    y_true = df["label"].astype(int).to_numpy()
    y_score = df[score_col].astype(float).to_numpy()
    y_pred = []
    used_thresholds = []

    for _, row in df.iterrows():
        chromosome_id = str(row["chromosome_id"])
        threshold = float(chr_thresholds.get(chromosome_id, fallback_threshold))
        y_pred.append(1 if float(row[score_col]) >= threshold else 0)
        used_thresholds.append(threshold)

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
    recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_abnormal = (
        2.0 * precision_abnormal * recall_abnormal / (precision_abnormal + recall_abnormal)
        if (precision_abnormal + recall_abnormal) > 0
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
    return metrics


def write_summary_table(output_path, global_eval, chr_eval, distance, knn_k):
    lines = [
        "# P12 Embedding Chromosome-Conditioned Memory Bank Summary",
        "",
        f"- distance: `{distance}`",
        f"- knn_k: `{knn_k}`",
        f"- global val-best threshold: `{global_eval['val_best_threshold']:.6f}`",
        f"- chr-conditioned best quantile from val: `{chr_eval['best_quantile_from_val']:.4f}`",
        "",
        "| Setting | Test F1 | Test Precision_abn | Test Recall_abn | Test Balanced Acc |",
        "|---|---:|---:|---:|---:|",
        (
            "| Memory bank global val-best | "
            f"{global_eval['test_metrics_val_best']['f1']:.4f} | "
            f"{global_eval['test_metrics_val_best']['precision_abnormal']:.4f} | "
            f"{global_eval['test_metrics_val_best']['recall_abnormal']:.4f} | "
            f"{global_eval['test_metrics_val_best']['balanced_acc']:.4f} |"
        ),
        (
            "| Memory bank chr-conditioned | "
            f"{chr_eval['test_metrics_best_chr']['f1']:.4f} | "
            f"{chr_eval['test_metrics_best_chr']['precision_abnormal']:.4f} | "
            f"{chr_eval['test_metrics_best_chr']['recall_abnormal']:.4f} | "
            f"{chr_eval['test_metrics_best_chr']['balanced_acc']:.4f} |"
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--distance", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--knn_k", type=int, default=1)
    parser.add_argument("--normalize_embeddings", action="store_true", default=True)
    parser.add_argument("--no_normalize_embeddings", action="store_false", dest="normalize_embeddings")
    parser.add_argument("--max_train_per_chr", type=int, default=0)
    parser.add_argument("--quantiles", default="0.95,0.975,0.99")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    if cfg.get("experiment_mode") != "multi_prototype_metric":
        raise ValueError("This script only supports experiment_mode=multi_prototype_metric")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)
    if not use_chromosome_id:
        raise ValueError("This script requires use_chromosome_id=True")
    if not use_pair_input:
        raise ValueError("This script currently expects pair-input experiments")

    quantiles = parse_float_list(args.quantiles)
    for quantile in quantiles:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Invalid quantile: {quantile}")
    if int(args.knn_k) <= 0:
        raise ValueError("--knn_k must be > 0")

    chr_to_idx, _ = build_chr_vocab_from_csv(cfg["data"]["train_csv"])
    model = build_model_from_config(cfg, chr_to_idx, device)
    _safe_load_state_dict(model, args.ckpt, device)
    model.eval()

    save_dir = args.save_dir or default_save_dir(args.config, args.ckpt)
    os.makedirs(save_dir, exist_ok=True)

    train_loader = build_eval_loader_for_csv(cfg, cfg["data"]["train_csv"], chr_to_idx)
    val_loader = build_eval_loader_for_csv(cfg, cfg["data"]["val_csv"], chr_to_idx)
    test_loader = build_eval_loader_for_csv(cfg, cfg["data"]["test_csv"], chr_to_idx)

    train_rows = collect_embeddings(model, train_loader, device, use_chromosome_id, use_pair_input)
    val_rows = collect_embeddings(model, val_loader, device, use_chromosome_id, use_pair_input)
    test_rows = collect_embeddings(model, test_loader, device, use_chromosome_id, use_pair_input)

    memory_bank = build_chr_memory_bank(
        train_rows,
        normalize_embeddings=args.normalize_embeddings,
        max_per_chr=(None if int(args.max_train_per_chr) <= 0 else int(args.max_train_per_chr)),
        seed=cfg.get("seed", 42),
    )

    train_rows = score_rows_with_memory_bank(
        train_rows,
        memory_bank,
        normalize_embeddings=args.normalize_embeddings,
        distance=args.distance,
        knn_k=args.knn_k,
    )
    val_rows = score_rows_with_memory_bank(
        val_rows,
        memory_bank,
        normalize_embeddings=args.normalize_embeddings,
        distance=args.distance,
        knn_k=args.knn_k,
    )
    test_rows = score_rows_with_memory_bank(
        test_rows,
        memory_bank,
        normalize_embeddings=args.normalize_embeddings,
        distance=args.distance,
        knn_k=args.knn_k,
    )

    train_df = records_to_dataframe(train_rows, score_column="memory_bank_score")
    val_df = records_to_dataframe(val_rows, score_column="memory_bank_score")
    test_df = records_to_dataframe(test_rows, score_column="memory_bank_score")

    best_threshold, best_score, best_stats = search_best_threshold(
        y_true=val_df["label"].astype(int).tolist(),
        y_score=val_df["memory_bank_score"].astype(float).tolist(),
        metric="f1",
        higher_score_more_positive=True,
    )
    test_metrics_val_best = compute_score_based_metrics(
        y_true=test_df["label"].astype(int).tolist(),
        y_score=test_df["memory_bank_score"].astype(float).tolist(),
        threshold=best_threshold,
        higher_score_more_positive=True,
    )

    global_quantile_thresholds = build_global_quantile_thresholds(train_df, quantiles)
    global_quantile_sweep = []
    for quantile, threshold in global_quantile_thresholds.items():
        val_metrics = compute_score_based_metrics(
            y_true=val_df["label"].astype(int).tolist(),
            y_score=val_df["memory_bank_score"].astype(float).tolist(),
            threshold=threshold,
            higher_score_more_positive=True,
        )
        test_metrics = compute_score_based_metrics(
            y_true=test_df["label"].astype(int).tolist(),
            y_score=test_df["memory_bank_score"].astype(float).tolist(),
            threshold=threshold,
            higher_score_more_positive=True,
        )
        global_quantile_sweep.append(
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

    chr_quantile_sweep = []
    for quantile in quantiles:
        chr_thresholds = compute_chr_quantile_thresholds(train_df, "memory_bank_score", quantile)
        fallback_threshold = float(global_quantile_thresholds[float(quantile)])
        val_true, val_score, val_pred, _ = apply_chr_thresholds(
            val_df,
            "memory_bank_score",
            chr_thresholds,
            fallback_threshold,
        )
        test_true, test_score, test_pred, test_used_thresholds = apply_chr_thresholds(
            test_df,
            "memory_bank_score",
            chr_thresholds,
            fallback_threshold,
        )
        val_metrics = compute_metrics_from_predictions(val_true, val_score, val_pred)
        test_metrics = compute_metrics_from_predictions(test_true, test_score, test_pred)
        chr_quantile_sweep.append(
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

    chr_quantile_sweep = sorted(
        chr_quantile_sweep,
        key=lambda row: (-row["val_f1"], -row["val_recall_abnormal"], row["test_fp"]),
    )
    best_chr_row = chr_quantile_sweep[0]
    best_chr_thresholds = best_chr_row["thresholds"]
    best_chr_fallback = float(best_chr_row["fallback_threshold"])
    test_true, test_score, test_pred, test_used_thresholds = apply_chr_thresholds(
        test_df,
        "memory_bank_score",
        best_chr_thresholds,
        best_chr_fallback,
    )
    test_metrics_best_chr = compute_metrics_from_predictions(test_true, test_score, test_pred)
    test_df["pred_label_global_valbest"] = (
        test_df["memory_bank_score"].astype(float) >= float(best_threshold)
    ).astype(int)
    test_df["pred_label_chr_conditioned"] = test_pred.astype(int)
    test_df["used_threshold_chr_conditioned"] = test_used_thresholds.astype(float)

    results = {
        "method": "p12_embedding_chr_memory_bank",
        "config_path": args.config,
        "checkpoint_path": args.ckpt,
        "distance": args.distance,
        "knn_k": int(args.knn_k),
        "normalize_embeddings": bool(args.normalize_embeddings),
        "quantiles": quantiles,
        "max_train_per_chr": None if int(args.max_train_per_chr) <= 0 else int(args.max_train_per_chr),
        "memory_bank_sizes": {chromosome_id: int(bank.shape[0]) for chromosome_id, bank in memory_bank.items()},
        "embedding_dim": int(next(iter(memory_bank.values())).shape[1]),
        "global_eval": {
            "val_best_threshold": float(best_threshold),
            "val_best_score": float(best_score),
            "val_best_stats": best_stats,
            "test_metrics_val_best": test_metrics_val_best,
            "global_quantile_sweep": global_quantile_sweep,
        },
        "chr_eval": {
            "best_quantile_from_val": float(best_chr_row["quantile"]),
            "fallback_threshold": best_chr_fallback,
            "best_chr_thresholds": best_chr_thresholds,
            "test_metrics_best_chr": test_metrics_best_chr,
            "chr_quantile_sweep": chr_quantile_sweep,
        },
        "test_seen_global_valbest": summarize_score_binary(
            test_df[test_df["subtype_status"] == "seen"],
            "memory_bank_score",
            best_threshold,
        ) if "subtype_status" in test_df.columns else None,
        "test_unseen_global_valbest": summarize_score_binary(
            test_df[test_df["subtype_status"] == "unseen"],
            "memory_bank_score",
            best_threshold,
        ) if "subtype_status" in test_df.columns else None,
        "test_seen_chr_conditioned": None,
        "test_unseen_chr_conditioned": None,
        "test_by_subtype_global_valbest": summarize_by_subtype(
            test_df,
            "memory_bank_score",
            best_threshold,
        ),
    }

    if "subtype_status" in test_df.columns:
        seen_df = test_df[test_df["subtype_status"] == "seen"].copy()
        unseen_df = test_df[test_df["subtype_status"] == "unseen"].copy()

        if not seen_df.empty:
            seen_true, seen_score, seen_pred, _ = apply_chr_thresholds(
                seen_df,
                "memory_bank_score",
                best_chr_thresholds,
                best_chr_fallback,
            )
            results["test_seen_chr_conditioned"] = compute_metrics_from_predictions(seen_true, seen_score, seen_pred)
            results["test_seen_chr_conditioned"]["count"] = int(len(seen_df))
        else:
            results["test_seen_chr_conditioned"] = {"count": 0}

        if not unseen_df.empty:
            unseen_true, unseen_score, unseen_pred, _ = apply_chr_thresholds(
                unseen_df,
                "memory_bank_score",
                best_chr_thresholds,
                best_chr_fallback,
            )
            results["test_unseen_chr_conditioned"] = compute_metrics_from_predictions(unseen_true, unseen_score, unseen_pred)
            results["test_unseen_chr_conditioned"]["count"] = int(len(unseen_df))
        else:
            results["test_unseen_chr_conditioned"] = {"count": 0}
    else:
        results["test_seen_chr_conditioned"] = None
        results["test_unseen_chr_conditioned"] = None

    test_by_subtype_chr = []
    if "abnormal_subtype_id" in test_df.columns:
        abnormal_df = test_df[test_df["label"].astype(int) == 1].copy()
        for subtype, group in abnormal_df.groupby("abnormal_subtype_id", dropna=False):
            _, _, y_pred_sub, used_sub = apply_chr_thresholds(
                group,
                "memory_bank_score",
                best_chr_thresholds,
                best_chr_fallback,
            )
            scores = group["memory_bank_score"].astype(float).to_numpy()
            test_by_subtype_chr.append(
                {
                    "abnormal_subtype_id": "" if pd.isna(subtype) else str(subtype),
                    "chromosome_id": str(group["chromosome_id"].iloc[0]),
                    "subtype_status": str(group["subtype_status"].iloc[0]) if "subtype_status" in group.columns else "",
                    "count": int(len(group)),
                    "recall_at_threshold": float(y_pred_sub.mean()),
                    "mean_score": float(scores.mean()),
                    "min_score": float(scores.min()),
                    "max_score": float(scores.max()),
                    "mean_threshold": float(used_sub.mean()),
                }
            )
    results["test_by_subtype_chr_conditioned"] = test_by_subtype_chr

    train_df.to_csv(os.path.join(save_dir, "train_predictions.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)
    pd.DataFrame(global_quantile_sweep).to_csv(os.path.join(save_dir, "global_quantile_sweep.csv"), index=False)

    chr_rows_for_csv = []
    for row in chr_quantile_sweep:
        copied = dict(row)
        copied["thresholds"] = yaml.safe_dump(to_serializable(copied["thresholds"]), allow_unicode=True, sort_keys=True).strip()
        chr_rows_for_csv.append(copied)
    pd.DataFrame(chr_rows_for_csv).to_csv(os.path.join(save_dir, "chr_quantile_sweep.csv"), index=False)

    results_path = os.path.join(save_dir, "results.yaml")
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    write_summary_table(
        Path(save_dir) / "summary_table.md",
        results["global_eval"],
        results["chr_eval"],
        args.distance,
        args.knn_k,
    )

    print(f"Saved embedding memory-bank evaluation to {results_path}")
    print(f"Saved train predictions to {os.path.join(save_dir, 'train_predictions.csv')}")
    print(f"Saved val predictions to {os.path.join(save_dir, 'val_predictions.csv')}")
    print(f"Saved test predictions to {os.path.join(save_dir, 'test_predictions.csv')}")
    print(f"Saved global sweep to {os.path.join(save_dir, 'global_quantile_sweep.csv')}")
    print(f"Saved chromosome-conditioned sweep to {os.path.join(save_dir, 'chr_quantile_sweep.csv')}")


if __name__ == "__main__":
    main()
