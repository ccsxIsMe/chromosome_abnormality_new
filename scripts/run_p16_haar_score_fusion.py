import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_scores(model_bundle, feature_csv):
    payload = load_pickle(model_bundle)
    model = payload["model"]
    feature_columns = payload["feature_columns"]

    df = pd.read_csv(feature_csv)
    x = df[feature_columns].astype(np.float32).to_numpy()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        scores = probs[:, 1]
    elif hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=np.float64)
        if raw.max() > raw.min():
            scores = (raw - raw.min()) / (raw.max() - raw.min())
        else:
            scores = np.zeros_like(raw, dtype=np.float64)
    else:
        scores = model.predict(x).astype(np.float64)

    out = df.copy()
    out["score"] = scores.astype(np.float64)
    return out


def minmax_by_train(train_scores, values):
    train_scores = np.asarray(train_scores, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    lo = float(train_scores.min())
    hi = float(train_scores.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def evaluate_weighted_fusion(val_df, test_df, weight):
    weight = float(weight)
    val_fused = weight * val_df["p16_score_norm"].astype(float).to_numpy() + (1.0 - weight) * val_df["haar_score_norm"].astype(float).to_numpy()
    test_fused = weight * test_df["p16_score_norm"].astype(float).to_numpy() + (1.0 - weight) * test_df["haar_score_norm"].astype(float).to_numpy()

    best_threshold, best_score, best_stats = search_best_threshold(
        y_true=val_df["label"].astype(int).to_numpy(),
        y_score=val_fused,
        metric="f1",
    )
    test_metrics = compute_classification_metrics(
        y_true=test_df["label"].astype(int).to_numpy(),
        y_prob=test_fused,
        threshold=best_threshold,
    )
    return {
        "weight_p16": weight,
        "weight_haar": 1.0 - weight,
        "val_best_threshold": float(best_threshold),
        "val_best_score": float(best_score),
        "val_best_stats": best_stats,
        "test_metrics_best": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run score-level fusion between a P16-based score model and a Haar-based score model. "
            "This is distinct from feature-level boosting fusion."
        )
    )
    parser.add_argument("--p16_model_pkl", required=True)
    parser.add_argument("--p16_train_features_csv", required=True)
    parser.add_argument("--p16_val_features_csv", required=True)
    parser.add_argument("--p16_test_features_csv", required=True)
    parser.add_argument("--haar_model_pkl", required=True)
    parser.add_argument("--haar_train_features_csv", required=True)
    parser.add_argument("--haar_val_features_csv", required=True)
    parser.add_argument("--haar_test_features_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p16_train = load_model_scores(args.p16_model_pkl, args.p16_train_features_csv)
    p16_val = load_model_scores(args.p16_model_pkl, args.p16_val_features_csv)
    p16_test = load_model_scores(args.p16_model_pkl, args.p16_test_features_csv)

    haar_train = load_model_scores(args.haar_model_pkl, args.haar_train_features_csv)
    haar_val = load_model_scores(args.haar_model_pkl, args.haar_val_features_csv)
    haar_test = load_model_scores(args.haar_model_pkl, args.haar_test_features_csv)

    p16_val["haar_score"] = haar_val["score"].astype(float).to_numpy()
    p16_test["haar_score"] = haar_test["score"].astype(float).to_numpy()
    p16_val["p16_score"] = p16_val["score"].astype(float).to_numpy()
    p16_test["p16_score"] = p16_test["score"].astype(float).to_numpy()

    p16_val["p16_score_norm"] = minmax_by_train(p16_train["score"].astype(float).to_numpy(), p16_val["p16_score"].astype(float).to_numpy())
    p16_test["p16_score_norm"] = minmax_by_train(p16_train["score"].astype(float).to_numpy(), p16_test["p16_score"].astype(float).to_numpy())
    p16_val["haar_score_norm"] = minmax_by_train(haar_train["score"].astype(float).to_numpy(), p16_val["haar_score"].astype(float).to_numpy())
    p16_test["haar_score_norm"] = minmax_by_train(haar_train["score"].astype(float).to_numpy(), p16_test["haar_score"].astype(float).to_numpy())

    weights = [float(x.strip()) for x in str(args.weights).split(",") if str(x).strip()]
    rows = [evaluate_weighted_fusion(p16_val, p16_test, weight) for weight in weights]
    rows = sorted(rows, key=lambda row: row["val_best_score"], reverse=True)
    best_row = rows[0]

    sweep_rows = []
    for row in rows:
        metrics = row["test_metrics_best"]
        sweep_rows.append(
            {
                "weight_p16": float(row["weight_p16"]),
                "weight_haar": float(row["weight_haar"]),
                "val_best_f1": float(row["val_best_score"]),
                "test_f1": float(metrics["f1"]),
                "test_precision_abnormal": float(metrics["precision_abnormal"]),
                "test_recall_abnormal": float(metrics["recall_abnormal"]),
                "test_balanced_acc": float(metrics["balanced_acc"]),
                "test_auprc": float(metrics["auprc"]) if metrics["auprc"] is not None else None,
                "test_auroc": float(metrics["auroc"]) if metrics["auroc"] is not None else None,
            }
        )

    pd.DataFrame(sweep_rows).to_csv(output_dir / "weight_sweep.csv", index=False)

    summary_lines = [
        "# P16 + Haar Score-Level Fusion",
        "",
        "## Best Validation Weight",
        f"- weight_p16: `{best_row['weight_p16']:.2f}`",
        f"- weight_haar: `{best_row['weight_haar']:.2f}`",
        f"- best_val_f1: `{best_row['val_best_score']:.4f}`",
        "",
        "## Test",
        f"- test_f1_at_best_weight: `{best_row['test_metrics_best']['f1']:.4f}`",
        f"- test_precision_abnormal: `{best_row['test_metrics_best']['precision_abnormal']:.4f}`",
        f"- test_recall_abnormal: `{best_row['test_metrics_best']['recall_abnormal']:.4f}`",
        f"- test_balanced_acc: `{best_row['test_metrics_best']['balanced_acc']:.4f}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    results = {
        "method": "p16_haar_score_fusion",
        "fusion_type": "score_level_weighted_sum",
        "p16_model_pkl": args.p16_model_pkl,
        "haar_model_pkl": args.haar_model_pkl,
        "weights": weights,
        "best_by_val": best_row,
        "weight_sweep": sweep_rows,
    }
    with open(output_dir / "results.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    print(f"Saved score fusion results to {output_dir / 'results.yaml'}")


if __name__ == "__main__":
    main()
