import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from src.utils.casewise_calibration import summarize_case_isolation
from src.utils.haar_pair_features import extract_pair_features_from_paths
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


def build_adaboost_classifier(n_estimators, learning_rate, max_depth, seed):
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


def build_feature_dataframe(
    df,
    profile_length,
    band_width,
    kernel_sizes,
    split_name,
    representation_version,
    pair_orientation_align,
):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract-{split_name}", leave=False):
        feature_row = extract_pair_features_from_paths(
            left_path=row["left_path"],
            right_path=row["right_path"],
            profile_length=profile_length,
            band_width=band_width,
            kernel_sizes=kernel_sizes,
            representation_version=representation_version,
            pair_orientation_align=pair_orientation_align,
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
    }
    return [col for col in df.columns if col not in exclude_columns]


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


def summarize_by_chromosome(df):
    rows = []
    for chromosome_id, group in df.groupby("chromosome_id"):
        rows.append(
            {
                "chromosome_id": str(chromosome_id),
                "count": int(len(group)),
                "normal_count": int((group["label"].astype(int) == 0).sum()),
                "abnormal_count": int((group["label"].astype(int) == 1).sum()),
            }
        )
    return rows


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
    return rows[:50]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profile_length", type=int, default=128)
    parser.add_argument("--band_width", type=int, default=32)
    parser.add_argument("--kernel_sizes", default="4,8,16,32,64")
    parser.add_argument("--representation_version", default="v1", choices=["v1", "v2", "v3"])
    parser.add_argument("--pair_orientation_align", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--max_depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_sizes = parse_int_list(args.kernel_sizes)
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    split_summary = summarize_case_isolation(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )

    print("Extracting pair Haar-like features...")
    train_features_df = build_feature_dataframe(
        train_df,
        profile_length=args.profile_length,
        band_width=args.band_width,
        kernel_sizes=kernel_sizes,
        split_name="train",
        representation_version=args.representation_version,
        pair_orientation_align=args.pair_orientation_align,
    )
    val_features_df = build_feature_dataframe(
        val_df,
        profile_length=args.profile_length,
        band_width=args.band_width,
        kernel_sizes=kernel_sizes,
        split_name="val",
        representation_version=args.representation_version,
        pair_orientation_align=args.pair_orientation_align,
    )
    test_features_df = build_feature_dataframe(
        test_df,
        profile_length=args.profile_length,
        band_width=args.band_width,
        kernel_sizes=kernel_sizes,
        split_name="test",
        representation_version=args.representation_version,
        pair_orientation_align=args.pair_orientation_align,
    )

    feature_columns = get_feature_columns(train_features_df)
    x_train = train_features_df[feature_columns].astype(np.float32).to_numpy()
    y_train = train_features_df["label"].astype(int).to_numpy()
    x_val = val_features_df[feature_columns].astype(np.float32).to_numpy()
    y_val = val_features_df["label"].astype(int).to_numpy()
    x_test = test_features_df[feature_columns].astype(np.float32).to_numpy()
    y_test = test_features_df["label"].astype(int).to_numpy()

    print(
        "Training AdaBoost baseline:",
        {
            "num_features": int(len(feature_columns)),
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
        },
    )
    model = build_adaboost_classifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    model.fit(x_train, y_train)

    val_scores = predict_scores(model, x_val)
    test_scores = predict_scores(model, x_test)

    val_metrics_05 = compute_classification_metrics(y_true=y_val, y_prob=val_scores, threshold=0.5)
    best_threshold, best_score, best_stats = search_best_threshold(y_val, val_scores, metric="f1")
    test_metrics_05 = compute_classification_metrics(y_true=y_test, y_prob=test_scores, threshold=0.5)
    test_metrics_best = compute_classification_metrics(y_true=y_test, y_prob=test_scores, threshold=best_threshold)

    val_features_df["score"] = val_scores.astype(np.float64)
    val_features_df["pred_label_05"] = (val_features_df["score"].astype(float) >= 0.5).astype(int)
    val_features_df["pred_label_best"] = (val_features_df["score"].astype(float) >= float(best_threshold)).astype(int)
    test_features_df["score"] = test_scores.astype(np.float64)
    test_features_df["pred_label_05"] = (test_features_df["score"].astype(float) >= 0.5).astype(int)
    test_features_df["pred_label_best"] = (test_features_df["score"].astype(float) >= float(best_threshold)).astype(int)

    train_features_df.to_csv(output_dir / "train_features.csv", index=False)
    val_features_df.to_csv(output_dir / "val_features.csv", index=False)
    test_features_df.to_csv(output_dir / "test_features.csv", index=False)

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "config": vars(args),
            },
            f,
        )

    top_features = maybe_save_feature_importance(model, feature_columns, output_dir)

    summary_lines = [
        "# Haar-like + AdaBoost Pair Baseline",
        "",
        "## Settings",
        f"- profile_length: `{args.profile_length}`",
        f"- band_width: `{args.band_width}`",
        f"- kernel_sizes: `{kernel_sizes}`",
        f"- representation_version: `{args.representation_version}`",
        f"- pair_orientation_align: `{args.pair_orientation_align}`",
        f"- n_estimators: `{args.n_estimators}`",
        f"- learning_rate: `{args.learning_rate}`",
        f"- weak_learner_max_depth: `{args.max_depth}`",
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
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    results = {
        "method": "haar_like_adaboost_pair_baseline",
        "task": "supervised_pair_abnormality",
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "split_summary": split_summary,
        "feature_settings": {
            "profile_length": int(args.profile_length),
            "band_width": int(args.band_width),
            "kernel_sizes": kernel_sizes,
            "representation_version": args.representation_version,
            "pair_orientation_align": bool(args.pair_orientation_align),
            "num_feature_columns": int(len(feature_columns)),
        },
        "model_settings": {
            "classifier": "AdaBoostClassifier",
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "weak_learner": "DecisionTreeClassifier",
            "weak_learner_max_depth": int(args.max_depth),
            "seed": int(args.seed),
        },
        "val_metrics_05": val_metrics_05,
        "best_threshold": float(best_threshold),
        "best_threshold_score": float(best_score),
        "best_threshold_stats": best_stats,
        "test_metrics_05": test_metrics_05,
        "test_metrics_best": test_metrics_best,
        "top_feature_importance": top_features,
        "train_distribution_by_chromosome": summarize_by_chromosome(train_features_df),
        "val_distribution_by_chromosome": summarize_by_chromosome(val_features_df),
        "test_distribution_by_chromosome": summarize_by_chromosome(test_features_df),
        "test_by_subtype_best_threshold": summarize_by_subtype(
            test_features_df,
            score_column="score",
            threshold=best_threshold,
        ),
    }

    with open(output_dir / "results.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    print(f"Saved Haar-like AdaBoost baseline results to {output_dir / 'results.yaml'}")
    print(f"Saved extracted features to {output_dir}")
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
