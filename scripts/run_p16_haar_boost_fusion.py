import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from src.main import _forward_model, _safe_load_state_dict, build_eval_loader_for_csv, load_config, set_seed
from src.models.build_model import build_model
from src.utils.casewise_calibration import summarize_case_isolation
from src.utils.chromosome_vocab import canonicalize_chromosome_id
from src.utils.haar_pair_features import (
    build_haar_kernel_catalog,
    extract_pair_features_from_paths,
    normalize_haar_feature_set,
)
from src.utils.metrics import compute_classification_metrics, search_best_threshold


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


def parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


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


def build_chr_vocab_from_splits(train_csv, val_csv, test_csv):
    ordered = []
    seen = set()
    for csv_path in [train_csv, val_csv, test_csv]:
        df = pd.read_csv(csv_path)
        if "chromosome_id" not in df.columns:
            raise ValueError(f"CSV missing chromosome_id: {csv_path}")
        for raw_id in df["chromosome_id"].tolist():
            canon = canonicalize_chromosome_id(raw_id)
            if canon not in seen:
                seen.add(canon)
                ordered.append(canon)

    final_order = []
    for idx in range(1, 23):
        value = str(idx)
        if value in seen:
            final_order.append(value)
    if "X" in seen:
        final_order.append("X")
    if "Y" in seen:
        final_order.append("Y")
    if "UNK" in seen:
        final_order.append("UNK")
    for value in ordered:
        if value not in final_order:
            final_order.append(value)

    chr_to_idx = {chromosome_id: idx for idx, chromosome_id in enumerate(final_order)}
    return chr_to_idx


def build_metadata_row(source_row, batch_size, idx):
    value = source_row
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value] * batch_size
    else:
        value = list(value)

    if idx >= len(value):
        return ""

    item = value[idx]
    if item is None:
        return ""
    return item


def collect_model_feature_dataframe(
    model,
    loader,
    device,
    use_chromosome_id,
    use_pair_input,
    include_embedding,
    include_prototype_distances,
    include_model_scalars,
):
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

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract-P16", leave=False):
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
                raise ValueError("Frozen P16 model must return a dict output.")

            embeddings = outputs.get("embedding")
            prototype_distances = outputs.get("prototype_distances")
            anomaly_scores = outputs.get("anomaly_score")
            pair_distance = outputs.get("pair_distance")
            direct_diag = outputs.get("direct_diag_similarity")
            reverse_diag = outputs.get("reverse_diag_similarity")
            reverse_gain = outputs.get("reverse_gain")
            nearest_prototype_idx = outputs.get("nearest_prototype_idx")

            batch_size = labels.size(0)
            labels_np = labels.detach().cpu().numpy()

            for idx in range(batch_size):
                row = {
                    "label": int(labels_np[idx]),
                }
                for key in metadata_keys:
                    if key in batch:
                        value = build_metadata_row(batch[key], batch_size, idx)
                        if key in {
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
                        }:
                            value = str(value)
                        row[key] = value

                if include_model_scalars:
                    if anomaly_scores is not None:
                        row["p16_anomaly_score"] = float(anomaly_scores[idx].detach().cpu().item())
                    if pair_distance is not None:
                        row["p16_pair_distance"] = float(pair_distance[idx].detach().cpu().item())
                    if direct_diag is not None:
                        row["p16_direct_diag_similarity"] = float(direct_diag[idx].detach().cpu().item())
                    if reverse_diag is not None:
                        row["p16_reverse_diag_similarity"] = float(reverse_diag[idx].detach().cpu().item())
                    if reverse_gain is not None:
                        row["p16_reverse_gain"] = float(reverse_gain[idx].detach().cpu().item())
                    if direct_diag is not None and reverse_diag is not None:
                        row["p16_direct_minus_reverse"] = float(
                            direct_diag[idx].detach().cpu().item() - reverse_diag[idx].detach().cpu().item()
                        )
                        row["p16_reverse_minus_direct"] = float(
                            reverse_diag[idx].detach().cpu().item() - direct_diag[idx].detach().cpu().item()
                        )
                    if nearest_prototype_idx is not None:
                        row["p16_nearest_prototype_idx"] = int(nearest_prototype_idx[idx].detach().cpu().item())

                if include_prototype_distances and prototype_distances is not None:
                    proto_row = prototype_distances[idx].detach().cpu().numpy().astype(np.float32)
                    for proto_idx, value in enumerate(proto_row.tolist()):
                        row[f"p16_proto_dist_{proto_idx:02d}"] = float(value)

                if include_embedding:
                    if embeddings is None:
                        raise ValueError("Requested embedding features but model output does not contain 'embedding'.")
                    emb_row = embeddings[idx].detach().cpu().numpy().astype(np.float32)
                    for emb_idx, value in enumerate(emb_row.tolist()):
                        row[f"p16_embedding_{emb_idx:03d}"] = float(value)

                rows.append(row)

    return pd.DataFrame(rows)


def build_haar_feature_dataframe(
    df,
    profile_length,
    band_width,
    kernel_sizes,
    split_name,
    representation_version,
    pair_orientation_align,
    feature_set,
):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract-Haar-{split_name}", leave=False):
        feature_row = extract_pair_features_from_paths(
            left_path=row["left_path"],
            right_path=row["right_path"],
            profile_length=profile_length,
            band_width=band_width,
            kernel_sizes=kernel_sizes,
            representation_version=representation_version,
            pair_orientation_align=pair_orientation_align,
            feature_set=feature_set,
        )
        feature_row["label"] = int(row["label"])
        feature_row["chromosome_id"] = str(row["chromosome_id"])
        feature_row["case_id"] = str(row["case_id"]) if "case_id" in df.columns else ""
        feature_row["pair_key"] = str(row["pair_key"]) if "pair_key" in df.columns else ""
        feature_row["split"] = str(row["split"]) if "split" in df.columns else split_name
        feature_row["left_path"] = row["left_path"]
        feature_row["right_path"] = row["right_path"]
        feature_row["abnormal_subtype_id"] = (
            str(row["abnormal_subtype_id"]) if "abnormal_subtype_id" in df.columns and not pd.isna(row["abnormal_subtype_id"]) else ""
        )
        feature_row["subtype_status"] = (
            str(row["subtype_status"]) if "subtype_status" in df.columns and not pd.isna(row["subtype_status"]) else ""
        )
        rows.append(feature_row)
    return pd.DataFrame(rows)


def select_haar_feature_columns(train_df, feature_columns, select_mode, top_k):
    select_mode = str(select_mode).strip().lower()
    if select_mode == "all":
        return feature_columns, []

    top_k = max(int(top_k), 1)
    labels = train_df["label"].astype(int).to_numpy()
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return feature_columns, []

    scored_rows = []
    for col in feature_columns:
        values = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
        pos_values = values[pos_mask]
        neg_values = values[neg_mask]
        pos_mean = float(pos_values.mean()) if pos_values.size > 0 else 0.0
        neg_mean = float(neg_values.mean()) if neg_values.size > 0 else 0.0
        pooled_std = float(values.std()) + 1e-6
        effect_size = abs(pos_mean - neg_mean) / pooled_std

        if select_mode == "topk_1d" and not (
            col.startswith("haar_")
            or col.startswith("profile_")
            or col.startswith("width_")
            or col.startswith("left_profile_")
            or col.startswith("right_profile_")
            or col.startswith("left_width_")
            or col.startswith("right_width_")
            or col.startswith("seg")
        ):
            continue
        if select_mode == "topk_2d" and not col.startswith("haar2d_"):
            continue
        if select_mode == "topk_shape" and (
            col.startswith("haar_")
            or col.startswith("haar2d_")
            or col.startswith("profile_")
            or col.startswith("seg")
        ):
            continue

        scored_rows.append(
            {
                "feature": col,
                "effect_size": float(effect_size),
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
            }
        )

    if not scored_rows:
        return feature_columns, []

    scored_rows = sorted(scored_rows, key=lambda item: item["effect_size"], reverse=True)
    selected = [row["feature"] for row in scored_rows[:top_k]]
    return selected, scored_rows


def _normalize_key_columns(df, keys):
    key_df = df[keys].copy()
    for key in keys:
        key_df[key] = key_df[key].fillna("").astype(str)
    return key_df


def _has_usable_unique_key(df, keys):
    if not all(key in df.columns for key in keys):
        return False

    key_df = _normalize_key_columns(df, keys)
    if len(keys) == 1 and key_df[keys[0]].eq("").all():
        return False
    if len(keys) > 1 and key_df.apply(lambda row: "".join(row.tolist()), axis=1).eq("").all():
        return False
    return not bool(key_df.duplicated().any())


def build_merge_key(model_df, haar_df):
    for keys in [
        ["case_id", "pair_key"],
        ["pair_key"],
        ["left_path", "right_path"],
    ]:
        if _has_usable_unique_key(model_df, keys) and _has_usable_unique_key(haar_df, keys):
            return keys
    raise ValueError(
        "Could not find a reliable unique merge key shared by model and Haar features. "
        f"model_cols={list(model_df.columns)}, haar_cols={list(haar_df.columns)}"
    )


def merge_feature_frames(model_df, haar_df):
    merge_keys = build_merge_key(model_df, haar_df)

    metadata_columns = {
        "label",
        "chromosome_id",
        "case_id",
        "pair_key",
        "split",
        "left_path",
        "right_path",
        "abnormal_subtype_id",
        "subtype_status",
        "left_filename",
        "right_filename",
    }
    haar_feature_columns = [col for col in haar_df.columns if col not in metadata_columns]
    model_key_df = _normalize_key_columns(model_df, merge_keys)
    haar_key_df = _normalize_key_columns(haar_df, merge_keys)
    model_df = model_df.copy()
    haar_df = haar_df.copy()
    for key in merge_keys:
        model_df[key] = model_key_df[key]
        haar_df[key] = haar_key_df[key]
    merged = model_df.merge(
        haar_df[merge_keys + haar_feature_columns],
        on=merge_keys,
        how="inner",
        suffixes=("", "_haar"),
    )
    if merged.empty:
        raise ValueError("Merged feature frame is empty. Check that the CSV split and pair keys match.")
    if len(merged) != len(model_df) or len(merged) != len(haar_df):
        raise ValueError(
            "Merged feature frame row count does not match source frames. "
            f"model_rows={len(model_df)}, haar_rows={len(haar_df)}, merged_rows={len(merged)}, merge_keys={merge_keys}"
        )
    return merged


def maybe_apply_embedding_pca(train_df, val_df, test_df, embedding_pca_dim, seed):
    embedding_columns = [col for col in train_df.columns if col.startswith("p16_embedding_")]
    if int(embedding_pca_dim) <= 0 or len(embedding_columns) == 0:
        return train_df, val_df, test_df, None

    actual_dim = min(int(embedding_pca_dim), len(embedding_columns), len(train_df))
    if actual_dim <= 0:
        return train_df, val_df, test_df, None

    pca = PCA(n_components=actual_dim, random_state=seed)
    train_emb = train_df[embedding_columns].astype(np.float32).to_numpy()
    val_emb = val_df[embedding_columns].astype(np.float32).to_numpy()
    test_emb = test_df[embedding_columns].astype(np.float32).to_numpy()

    train_pca = pca.fit_transform(train_emb)
    val_pca = pca.transform(val_emb)
    test_pca = pca.transform(test_emb)
    pca_columns = [f"p16_embedding_pca_{idx:03d}" for idx in range(train_pca.shape[1])]

    for df, values in [
        (train_df, train_pca),
        (val_df, val_pca),
        (test_df, test_pca),
    ]:
        for col_idx, col_name in enumerate(pca_columns):
            df[col_name] = values[:, col_idx].astype(np.float32)
        df.drop(columns=embedding_columns, inplace=True)

    pca_info = {
        "n_components": int(actual_dim),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_.tolist()],
    }
    return train_df, val_df, test_df, pca_info


def sanitize_feature_frames(train_df, val_df, test_df, feature_columns):
    fill_values = {}
    replacement_summary = {}

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        numeric = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        replacement_summary[f"{split_name}_nan_like_before_fill"] = int(numeric.isna().sum().sum())
        if split_name == "train":
            fill_series = numeric.median(axis=0).fillna(0.0)
            fill_values = {col: float(fill_series[col]) for col in feature_columns}
        numeric = numeric.fillna(fill_values)
        remaining_nan = int(numeric.isna().sum().sum())
        if remaining_nan > 0:
            numeric = numeric.fillna(0.0)
        replacement_summary[f"{split_name}_nan_like_after_fill"] = int(numeric.isna().sum().sum())
        df.loc[:, feature_columns] = numeric.astype(np.float32)

    return train_df, val_df, test_df, replacement_summary


def get_feature_columns(df):
    exclude_columns = {
        "label",
        "chromosome_id",
        "case_id",
        "pair_key",
        "split",
        "left_path",
        "right_path",
        "abnormal_subtype_id",
        "subtype_status",
        "left_filename",
        "right_filename",
        "score",
        "pred_label_05",
        "pred_label_best",
    }
    return [col for col in df.columns if col not in exclude_columns]


def compute_balanced_sample_weights(y):
    y = np.asarray(y, dtype=np.int64)
    unique, counts = np.unique(y, return_counts=True)
    if unique.size < 2:
        raise ValueError(
            "Supervised boosting fusion requires at least two classes in train split. "
            "Current split appears to be single-class."
        )
    total = float(y.size)
    class_weight = {
        int(label): total / float(unique.size * count)
        for label, count in zip(unique.tolist(), counts.tolist())
    }
    return np.asarray([class_weight[int(label)] for label in y.tolist()], dtype=np.float64)


def build_classifier(classifier_name, n_estimators, learning_rate, max_depth, subsample, seed):
    if classifier_name == "adaboost":
        weak_learner = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        try:
            return AdaBoostClassifier(
                estimator=weak_learner,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                algorithm="SAMME",
                random_state=seed,
            )
        except TypeError:
            return AdaBoostClassifier(
                base_estimator=weak_learner,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                algorithm="SAMME",
                random_state=seed,
            )

    if classifier_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=float(subsample),
            random_state=seed,
        )

    raise ValueError(f"Unsupported classifier: {classifier_name}")


def predict_scores(model, features):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:
            if scores.max() > scores.min():
                return (scores - scores.min()) / (scores.max() - scores.min())
            return np.zeros_like(scores, dtype=np.float64)
    return model.predict(features).astype(np.float64)


def maybe_save_feature_importance(model, feature_columns, output_dir):
    if not hasattr(model, "feature_importances_"):
        return []
    importance = np.asarray(model.feature_importances_, dtype=np.float64)
    rows = [
        {
            "feature": feature_name,
            "importance": float(score),
        }
        for feature_name, score in zip(feature_columns, importance.tolist())
    ]
    rows = sorted(rows, key=lambda item: item["importance"], reverse=True)
    pd.DataFrame(rows).to_csv(Path(output_dir) / "feature_importance.csv", index=False)
    return rows


def summarize_feature_source_importance(feature_rows):
    totals = {
        "haar_branch": 0.0,
        "p16_embedding_branch": 0.0,
        "p16_prototype_branch": 0.0,
        "p16_scalar_branch": 0.0,
        "other": 0.0,
    }
    for row in feature_rows:
        feature = str(row["feature"])
        importance = float(row["importance"])
        if feature.startswith("p16_embedding_") or feature.startswith("p16_embedding_pca_"):
            totals["p16_embedding_branch"] += importance
        elif feature.startswith("p16_proto_dist_"):
            totals["p16_prototype_branch"] += importance
        elif feature.startswith("p16_"):
            totals["p16_scalar_branch"] += importance
        else:
            totals["haar_branch"] += importance
    return [{"feature_source": key, "importance": float(value)} for key, value in totals.items()]


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
                "chromosome_id": str(group["chromosome_id"].iloc[0]),
                "subtype_status": str(group["subtype_status"].iloc[0]) if "subtype_status" in group.columns else "",
                "count": int(len(group)),
                "recall_at_threshold": float(pred.mean()) if pred.size > 0 else 0.0,
                "mean_score": float(scores.mean()),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profile_length", type=int, default=128)
    parser.add_argument("--band_width", type=int, default=32)
    parser.add_argument("--kernel_sizes", default="4,8,16,32,64")
    parser.add_argument("--representation_version", default="v2", choices=["v1", "v2", "v3"])
    parser.add_argument("--pair_orientation_align", action="store_true")
    parser.add_argument("--feature_set", default="1d", choices=["1d", "2d", "1d2d", "1d+2d", "both", "all"])
    parser.add_argument(
        "--haar_feature_select_mode",
        default="all",
        choices=["all", "topk", "topk_1d", "topk_2d", "topk_shape"],
    )
    parser.add_argument("--haar_feature_topk", type=int, default=128)
    parser.add_argument("--classifier", default="gradient_boosting", choices=["adaboost", "gradient_boosting"])
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--embedding_pca_dim", type=int, default=0)
    parser.add_argument("--disable_haar", action="store_true")
    parser.add_argument("--disable_embedding", action="store_true")
    parser.add_argument("--disable_prototype_distances", action="store_true")
    parser.add_argument("--disable_model_scalars", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_set = normalize_haar_feature_set(args.feature_set)

    cfg = load_config(args.config)
    set_seed(int(args.seed))

    if cfg.get("experiment_mode") != "multi_prototype_metric":
        raise ValueError("This script currently expects experiment_mode=multi_prototype_metric.")
    if not cfg["model"].get("use_pair_input", False):
        raise ValueError("This script currently expects pair-input models.")
    if not cfg["model"].get("use_chromosome_id", False):
        raise ValueError("This script requires chromosome-conditioned models.")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    chr_to_idx = build_chr_vocab_from_splits(args.train_csv, args.val_csv, args.test_csv)
    model = build_model_from_config(cfg, chr_to_idx, device)
    _safe_load_state_dict(model, args.ckpt, device)
    model.eval()

    split_summary = summarize_case_isolation(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )
    train_csv_df = pd.read_csv(args.train_csv)
    val_csv_df = pd.read_csv(args.val_csv)
    test_csv_df = pd.read_csv(args.test_csv)

    train_loader = build_eval_loader_for_csv(cfg, args.train_csv, chr_to_idx)
    val_loader = build_eval_loader_for_csv(cfg, args.val_csv, chr_to_idx)
    test_loader = build_eval_loader_for_csv(cfg, args.test_csv, chr_to_idx)

    train_model_df = collect_model_feature_dataframe(
        model=model,
        loader=train_loader,
        device=device,
        use_chromosome_id=True,
        use_pair_input=True,
        include_embedding=not args.disable_embedding,
        include_prototype_distances=not args.disable_prototype_distances,
        include_model_scalars=not args.disable_model_scalars,
    )
    val_model_df = collect_model_feature_dataframe(
        model=model,
        loader=val_loader,
        device=device,
        use_chromosome_id=True,
        use_pair_input=True,
        include_embedding=not args.disable_embedding,
        include_prototype_distances=not args.disable_prototype_distances,
        include_model_scalars=not args.disable_model_scalars,
    )
    test_model_df = collect_model_feature_dataframe(
        model=model,
        loader=test_loader,
        device=device,
        use_chromosome_id=True,
        use_pair_input=True,
        include_embedding=not args.disable_embedding,
        include_prototype_distances=not args.disable_prototype_distances,
        include_model_scalars=not args.disable_model_scalars,
    )

    if args.disable_haar:
        train_df = train_model_df.copy()
        val_df = val_model_df.copy()
        test_df = test_model_df.copy()
        selected_haar_columns = []
        haar_selection_rows = []
    else:
        kernel_sizes = parse_int_list(args.kernel_sizes)
        train_haar_df = build_haar_feature_dataframe(
            train_csv_df,
            profile_length=args.profile_length,
            band_width=args.band_width,
            kernel_sizes=kernel_sizes,
            split_name="train",
            representation_version=args.representation_version,
            pair_orientation_align=args.pair_orientation_align,
            feature_set=feature_set,
        )
        val_haar_df = build_haar_feature_dataframe(
            val_csv_df,
            profile_length=args.profile_length,
            band_width=args.band_width,
            kernel_sizes=kernel_sizes,
            split_name="val",
            representation_version=args.representation_version,
            pair_orientation_align=args.pair_orientation_align,
            feature_set=feature_set,
        )
        test_haar_df = build_haar_feature_dataframe(
            test_csv_df,
            profile_length=args.profile_length,
            band_width=args.band_width,
            kernel_sizes=kernel_sizes,
            split_name="test",
            representation_version=args.representation_version,
            pair_orientation_align=args.pair_orientation_align,
            feature_set=feature_set,
        )
        metadata_columns = {
            "label",
            "chromosome_id",
            "case_id",
            "pair_key",
            "split",
            "left_path",
            "right_path",
            "abnormal_subtype_id",
            "subtype_status",
        }
        haar_feature_columns = [col for col in train_haar_df.columns if col not in metadata_columns]
        selected_haar_columns, haar_selection_rows = select_haar_feature_columns(
            train_df=train_haar_df,
            feature_columns=haar_feature_columns,
            select_mode=args.haar_feature_select_mode,
            top_k=args.haar_feature_topk,
        )
        keep_columns = [col for col in train_haar_df.columns if col in metadata_columns or col in selected_haar_columns]
        train_haar_df = train_haar_df[keep_columns].copy()
        val_haar_df = val_haar_df[keep_columns].copy()
        test_haar_df = test_haar_df[keep_columns].copy()
        train_df = merge_feature_frames(train_model_df, train_haar_df)
        val_df = merge_feature_frames(val_model_df, val_haar_df)
        test_df = merge_feature_frames(test_model_df, test_haar_df)

    train_df, val_df, test_df, pca_info = maybe_apply_embedding_pca(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        embedding_pca_dim=int(args.embedding_pca_dim),
        seed=int(args.seed),
    )

    feature_columns = get_feature_columns(train_df)
    if len(feature_columns) == 0:
        raise ValueError("No fusion features available. Enable at least one branch.")
    train_df, val_df, test_df, feature_sanitize_info = sanitize_feature_frames(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=feature_columns,
    )

    x_train = train_df[feature_columns].astype(np.float32).to_numpy()
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = val_df[feature_columns].astype(np.float32).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    x_test = test_df[feature_columns].astype(np.float32).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()
    sample_weight = compute_balanced_sample_weights(y_train)

    classifier = build_classifier(
        classifier_name=args.classifier,
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        seed=int(args.seed),
    )
    classifier.fit(x_train, y_train, sample_weight=sample_weight)

    val_scores = predict_scores(classifier, x_val)
    test_scores = predict_scores(classifier, x_test)

    val_metrics_05 = compute_classification_metrics(y_true=y_val, y_prob=val_scores, threshold=0.5)
    best_threshold, best_score, best_stats = search_best_threshold(y_val, val_scores, metric="f1")
    test_metrics_05 = compute_classification_metrics(y_true=y_test, y_prob=test_scores, threshold=0.5)
    test_metrics_best = compute_classification_metrics(y_true=y_test, y_prob=test_scores, threshold=best_threshold)

    val_df["score"] = val_scores.astype(np.float64)
    val_df["pred_label_05"] = (val_df["score"].astype(float) >= 0.5).astype(int)
    val_df["pred_label_best"] = (val_df["score"].astype(float) >= float(best_threshold)).astype(int)
    test_df["score"] = test_scores.astype(np.float64)
    test_df["pred_label_05"] = (test_df["score"].astype(float) >= 0.5).astype(int)
    test_df["pred_label_best"] = (test_df["score"].astype(float) >= float(best_threshold)).astype(int)

    train_df.to_csv(output_dir / "train_features.csv", index=False)
    val_df.to_csv(output_dir / "val_features.csv", index=False)
    test_df.to_csv(output_dir / "test_features.csv", index=False)

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": classifier,
                "feature_columns": feature_columns,
                "config": vars(args),
                "pca_info": pca_info,
            },
            f,
        )

    feature_importance_rows = maybe_save_feature_importance(classifier, feature_columns, output_dir)
    source_importance_rows = summarize_feature_source_importance(feature_importance_rows)
    pd.DataFrame(source_importance_rows).to_csv(output_dir / "feature_source_importance.csv", index=False)
    if haar_selection_rows:
        pd.DataFrame(haar_selection_rows).to_csv(output_dir / "haar_feature_selection_scores.csv", index=False)

    summary_lines = [
        "# P16 + Haar Frozen-Feature Boost Fusion",
        "",
        "## Settings",
        f"- frozen_model_config: `{args.config}`",
        f"- frozen_model_ckpt: `{args.ckpt}`",
        f"- classifier: `{args.classifier}`",
        f"- n_estimators: `{args.n_estimators}`",
        f"- learning_rate: `{args.learning_rate}`",
        f"- max_depth: `{args.max_depth}`",
        f"- subsample: `{args.subsample}`",
        f"- use_haar: `{not args.disable_haar}`",
        f"- use_embedding: `{not args.disable_embedding}`",
        f"- use_prototype_distances: `{not args.disable_prototype_distances}`",
        f"- use_model_scalars: `{not args.disable_model_scalars}`",
        f"- haar_feature_set: `{feature_set}`",
        f"- haar_feature_select_mode: `{args.haar_feature_select_mode}`",
        f"- selected_haar_feature_count: `{len(selected_haar_columns)}`",
        f"- embedding_pca_dim: `{args.embedding_pca_dim}`",
        f"- num_feature_columns: `{len(feature_columns)}`",
        f"- feature_sanitize_info: `{feature_sanitize_info}`",
        "",
        "## Validation",
        f"- best_threshold: `{best_threshold:.6f}`",
        f"- best_val_f1: `{best_score:.4f}`",
        "",
        "## Test",
        f"- test_f1_at_best_threshold: `{test_metrics_best['f1']:.4f}`",
        f"- test_precision_abnormal: `{test_metrics_best['precision_abnormal']:.4f}`",
        f"- test_recall_abnormal: `{test_metrics_best['recall_abnormal']:.4f}`",
        f"- test_balanced_acc: `{test_metrics_best['balanced_acc']:.4f}`",
        f"- test_auprc: `{test_metrics_best['auprc']:.4f}`" if test_metrics_best["auprc"] is not None else "- test_auprc: `None`",
        f"- test_auroc: `{test_metrics_best['auroc']:.4f}`" if test_metrics_best["auroc"] is not None else "- test_auroc: `None`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    results = {
        "method": "p16_haar_boost_fusion",
        "frozen_model": {
            "config": args.config,
            "checkpoint": args.ckpt,
        },
        "fusion_split": {
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
        },
        "split_summary": split_summary,
        "feature_settings": {
            "profile_length": int(args.profile_length),
            "band_width": int(args.band_width),
            "kernel_sizes": parse_int_list(args.kernel_sizes),
            "representation_version": args.representation_version,
            "pair_orientation_align": bool(args.pair_orientation_align),
            "haar_feature_set": feature_set,
            "haar_feature_select_mode": str(args.haar_feature_select_mode),
            "haar_feature_topk": int(args.haar_feature_topk),
            "selected_haar_feature_count": int(len(selected_haar_columns)),
            "selected_haar_feature_columns": selected_haar_columns,
            "haar_kernel_catalog": build_haar_kernel_catalog(
                kernel_sizes=parse_int_list(args.kernel_sizes),
                feature_set=feature_set,
                band_width=int(args.band_width),
            ) if not args.disable_haar else None,
            "use_haar": bool(not args.disable_haar),
            "use_embedding": bool(not args.disable_embedding),
            "use_prototype_distances": bool(not args.disable_prototype_distances),
            "use_model_scalars": bool(not args.disable_model_scalars),
            "embedding_pca": pca_info,
            "num_feature_columns": int(len(feature_columns)),
            "feature_sanitize_info": feature_sanitize_info,
        },
        "model_settings": {
            "classifier": args.classifier,
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "seed": int(args.seed),
        },
        "val_metrics_05": val_metrics_05,
        "best_threshold": float(best_threshold),
        "best_threshold_score": float(best_score),
        "best_threshold_stats": best_stats,
        "test_metrics_05": test_metrics_05,
        "test_metrics_best": test_metrics_best,
        "top_feature_importance": feature_importance_rows[:50],
        "feature_source_importance": source_importance_rows,
        "test_by_subtype_best_threshold": summarize_by_subtype(
            test_df,
            score_column="score",
            threshold=best_threshold,
        ),
    }

    with open(output_dir / "results.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    print(f"Saved fusion results to {output_dir / 'results.yaml'}")
    print(f"Saved merged feature tables to {output_dir}")
    print(
        "Test @ best threshold:",
        {
            "f1": round(float(test_metrics_best["f1"]), 4),
            "precision_abnormal": round(float(test_metrics_best["precision_abnormal"]), 4),
            "recall_abnormal": round(float(test_metrics_best["recall_abnormal"]), 4),
            "balanced_acc": round(float(test_metrics_best["balanced_acc"]), 4),
        },
    )


if __name__ == "__main__":
    main()
