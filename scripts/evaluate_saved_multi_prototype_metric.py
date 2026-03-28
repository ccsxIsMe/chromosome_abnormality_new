import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.losses.loss_factory import build_loss
from src.main import (
    _safe_load_state_dict,
    build_eval_loader_for_csv,
    evaluate_multi_prototype_metric,
    estimate_normal_threshold,
    export_prediction_records,
    load_config,
    set_seed,
    to_serializable,
)
from src.models.build_model import build_model
from src.utils.casewise_calibration import (
    calibrate_scores_casewise,
    evaluate_casewise_calibration,
    search_best_threshold_casewise,
    summarize_case_isolation,
)
from src.utils.chromosome_vocab import build_chr_vocab_from_csv
from src.utils.metrics import compute_score_based_metrics, search_best_threshold


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
                "case_ids": ",".join(sorted(group["case_id"].astype(str).unique().tolist()))
                if "case_id" in group.columns
                else "",
            }
        )
    return rows


def evaluate_threshold_strategy(
    train_y_true,
    train_scores,
    test_y_true,
    test_scores,
    method,
    quantile,
    mean_std_k,
):
    threshold = estimate_normal_threshold(
        y_true=train_y_true,
        y_score=train_scores,
        method=method,
        quantile=quantile,
        mean_std_k=mean_std_k,
    )
    metrics = compute_score_based_metrics(
        y_true=test_y_true,
        y_score=test_scores,
        threshold=threshold,
        higher_score_more_positive=True,
    )
    return float(threshold), metrics


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


def records_to_dataframe(records, casewise_scores=None, raw_threshold=None, casewise_threshold=None):
    df = pd.DataFrame(records).copy()
    if df.empty:
        return df

    if raw_threshold is not None:
        df["pred_label_raw"] = (df["anomaly_score"].astype(float) >= float(raw_threshold)).astype(int)

    if casewise_scores is not None:
        df["casewise_score"] = np.asarray(casewise_scores, dtype=np.float64)
        if casewise_threshold is not None:
            df["pred_label_casewise"] = (df["casewise_score"] >= float(casewise_threshold)).astype(int)

    return df


def default_save_dir(config_path, ckpt_path):
    config_stem = Path(config_path).stem
    ckpt_parent = Path(ckpt_path).resolve().parent.name
    project_root = Path(__file__).resolve().parents[1]
    return str(project_root / "outputs" / "posthoc_eval" / f"{ckpt_parent}__{config_stem}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--sweep_quantiles", default="0.95,0.975,0.99")
    parser.add_argument("--sweep_mean_std_k", default="2.0,2.5,3.0")
    parser.add_argument("--disable_quantile_sweep", action="store_true")
    parser.add_argument("--disable_mean_std_sweep", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    if cfg.get("experiment_mode") != "multi_prototype_metric":
        raise ValueError("This script only supports experiment_mode=multi_prototype_metric")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)
    if not use_chromosome_id:
        raise ValueError("multi_prototype_metric evaluation requires use_chromosome_id=True")
    if not use_pair_input:
        raise ValueError("This script currently expects pair-input experiments")

    chr_to_idx, _ = build_chr_vocab_from_csv(cfg["data"]["train_csv"])
    model = build_model_from_config(cfg, chr_to_idx, device)
    criterion = build_loss(
        cfg["loss"],
        device,
        experiment_mode=cfg.get("experiment_mode", "multi_prototype_metric"),
        model=model,
    ).to(device)
    _safe_load_state_dict(model, args.ckpt, device)

    save_dir = args.save_dir or default_save_dir(args.config, args.ckpt)
    os.makedirs(save_dir, exist_ok=True)

    sweep_quantiles = [
        float(x.strip()) for x in str(args.sweep_quantiles).split(",") if str(x).strip()
    ]
    sweep_mean_std_k = [
        float(x.strip()) for x in str(args.sweep_mean_std_k).split(",") if str(x).strip()
    ]

    split_summary = summarize_case_isolation(
        train_csv=cfg["data"]["train_csv"],
        val_csv=cfg["data"]["val_csv"],
        test_csv=cfg["data"]["test_csv"],
    )

    train_loader = build_eval_loader_for_csv(cfg, cfg["data"]["train_csv"], chr_to_idx)
    val_loader = build_eval_loader_for_csv(cfg, cfg["data"]["val_csv"], chr_to_idx)
    test_loader = build_eval_loader_for_csv(cfg, cfg["data"]["test_csv"], chr_to_idx)

    train_metrics_05, train_y_true, train_y_score, train_case_ids, train_records = evaluate_multi_prototype_metric(
        model,
        train_loader,
        criterion,
        device,
        threshold=0.5,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input,
    )
    val_metrics_05, val_y_true, val_y_score, val_case_ids, val_records = evaluate_multi_prototype_metric(
        model,
        val_loader,
        criterion,
        device,
        threshold=0.5,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input,
    )
    test_metrics_05, test_y_true, test_y_score, test_case_ids, test_records = evaluate_multi_prototype_metric(
        model,
        test_loader,
        criterion,
        device,
        threshold=0.5,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input,
    )

    best_th, best_score, best_stats = search_best_threshold(
        val_y_true,
        val_y_score,
        metric="f1",
        higher_score_more_positive=True,
    )
    test_metrics_best = compute_score_based_metrics(
        y_true=test_y_true,
        y_score=test_y_score,
        threshold=best_th,
        higher_score_more_positive=True,
    )
    test_metrics_best["loss"] = test_metrics_05["loss"]

    calibration_cfg = cfg.get("calibration", {})
    calibration_enabled = calibration_cfg.get("enabled", True)
    calibration_method = calibration_cfg.get("method", "zscore")

    val_casewise_metrics_05 = None
    val_casewise_best = None
    test_casewise_metrics_05 = None
    test_casewise_best = None
    casewise_best_th = None
    casewise_best_score = None
    casewise_best_stats = None
    train_casewise_scores = None
    val_casewise_scores = None
    test_casewise_scores = None

    if calibration_enabled and len(val_case_ids) == len(val_y_score) and len(test_case_ids) == len(test_y_score):
        val_casewise_metrics_05, val_casewise_scores = evaluate_casewise_calibration(
            y_true=val_y_true,
            raw_scores=val_y_score,
            case_ids=val_case_ids,
            threshold=0.5,
            method=calibration_method,
        )
        casewise_best_th, casewise_best_score, casewise_best_stats, val_casewise_scores = search_best_threshold_casewise(
            y_true=val_y_true,
            raw_scores=val_y_score,
            case_ids=val_case_ids,
            metric="f1",
            method=calibration_method,
        )
        val_casewise_best = compute_score_based_metrics(
            y_true=val_y_true,
            y_score=val_casewise_scores,
            threshold=casewise_best_th,
            higher_score_more_positive=True,
        )
        val_casewise_best["loss"] = val_metrics_05["loss"]

        test_casewise_metrics_05, test_casewise_scores = evaluate_casewise_calibration(
            y_true=test_y_true,
            raw_scores=test_y_score,
            case_ids=test_case_ids,
            threshold=0.5,
            method=calibration_method,
        )
        test_casewise_metrics_05["loss"] = test_metrics_05["loss"]

        test_casewise_scores = calibrate_scores_casewise(
            scores=test_y_score,
            case_ids=test_case_ids,
            method=calibration_method,
        )
        test_casewise_best = compute_score_based_metrics(
            y_true=test_y_true,
            y_score=test_casewise_scores,
            threshold=casewise_best_th,
            higher_score_more_positive=True,
        )
        test_casewise_best["loss"] = test_metrics_05["loss"]

        if len(train_case_ids) == len(train_y_score):
            train_casewise_scores = calibrate_scores_casewise(
                scores=train_y_score,
                case_ids=train_case_ids,
                method=calibration_method,
            )

    anomaly_threshold_cfg = cfg.get("anomaly_threshold", {})
    anomaly_threshold_enabled = anomaly_threshold_cfg.get("enabled", False)
    anomaly_threshold_method = anomaly_threshold_cfg.get("method", "quantile")
    anomaly_threshold_quantile = anomaly_threshold_cfg.get("quantile", 0.99)
    anomaly_threshold_mean_std_k = anomaly_threshold_cfg.get("mean_std_k", 3.0)
    anomaly_threshold_use_casewise = anomaly_threshold_cfg.get("use_casewise_scores", False)

    if anomaly_threshold_enabled and anomaly_threshold_use_casewise and not calibration_enabled:
        raise ValueError("anomaly_threshold.use_casewise_scores=True requires calibration.enabled=True")

    anomaly_threshold = None
    anomaly_threshold_stats = None
    test_metrics_anomaly_threshold = None
    threshold_sweep = []

    if anomaly_threshold_enabled:
        threshold_scores = train_y_score
        threshold_source = "train_normal_raw"
        if anomaly_threshold_use_casewise:
            if train_casewise_scores is None:
                raise ValueError("anomaly_threshold.use_casewise_scores=True requires train casewise scores")
            threshold_scores = train_casewise_scores
            threshold_source = f"train_normal_casewise_{calibration_method}"

        anomaly_threshold = estimate_normal_threshold(
            y_true=train_y_true,
            y_score=threshold_scores,
            method=anomaly_threshold_method,
            quantile=anomaly_threshold_quantile,
            mean_std_k=anomaly_threshold_mean_std_k,
        )

        test_scores_for_anomaly_threshold = test_casewise_scores if anomaly_threshold_use_casewise else test_y_score
        if test_scores_for_anomaly_threshold is None:
            raise ValueError("Failed to prepare test scores for anomaly threshold evaluation")

        test_metrics_anomaly_threshold = compute_score_based_metrics(
            y_true=test_y_true,
            y_score=test_scores_for_anomaly_threshold,
            threshold=anomaly_threshold,
            higher_score_more_positive=True,
        )
        test_metrics_anomaly_threshold["loss"] = test_metrics_05["loss"]

        anomaly_threshold_stats = {
            "source": threshold_source,
            "method": anomaly_threshold_method,
            "quantile": anomaly_threshold_quantile,
            "mean_std_k": anomaly_threshold_mean_std_k,
            "use_casewise_scores": anomaly_threshold_use_casewise,
            "threshold": anomaly_threshold,
        }

    sweep_sources = [
        {
            "score_name": "raw",
            "train_scores": train_y_score,
            "test_scores": test_y_score,
        }
    ]
    if train_casewise_scores is not None and test_casewise_scores is not None:
        sweep_sources.append(
            {
                "score_name": f"casewise_{calibration_method}",
                "train_scores": train_casewise_scores,
                "test_scores": test_casewise_scores,
            }
        )

    for source in sweep_sources:
        if not args.disable_quantile_sweep:
            for quantile in sweep_quantiles:
                threshold, metrics = evaluate_threshold_strategy(
                    train_y_true=train_y_true,
                    train_scores=source["train_scores"],
                    test_y_true=test_y_true,
                    test_scores=source["test_scores"],
                    method="quantile",
                    quantile=quantile,
                    mean_std_k=3.0,
                )
                metrics["loss"] = test_metrics_05["loss"]
                threshold_sweep.append(
                    {
                        "score_name": source["score_name"],
                        "threshold_family": "quantile",
                        "quantile": quantile,
                        "mean_std_k": None,
                        "threshold": threshold,
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "f1": metrics["f1"],
                        "precision_abnormal": metrics["precision_abnormal"],
                        "recall_abnormal": metrics["recall_abnormal"],
                        "balanced_acc": metrics["balanced_acc"],
                        "tn": metrics["confusion_matrix"]["tn"],
                        "fp": metrics["confusion_matrix"]["fp"],
                        "fn": metrics["confusion_matrix"]["fn"],
                        "tp": metrics["confusion_matrix"]["tp"],
                    }
                )

        if not args.disable_mean_std_sweep:
            for mean_std_k in sweep_mean_std_k:
                threshold, metrics = evaluate_threshold_strategy(
                    train_y_true=train_y_true,
                    train_scores=source["train_scores"],
                    test_y_true=test_y_true,
                    test_scores=source["test_scores"],
                    method="mean_std",
                    quantile=0.99,
                    mean_std_k=mean_std_k,
                )
                metrics["loss"] = test_metrics_05["loss"]
                threshold_sweep.append(
                    {
                        "score_name": source["score_name"],
                        "threshold_family": "mean_std",
                        "quantile": None,
                        "mean_std_k": mean_std_k,
                        "threshold": threshold,
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "f1": metrics["f1"],
                        "precision_abnormal": metrics["precision_abnormal"],
                        "recall_abnormal": metrics["recall_abnormal"],
                        "balanced_acc": metrics["balanced_acc"],
                        "tn": metrics["confusion_matrix"]["tn"],
                        "fp": metrics["confusion_matrix"]["fp"],
                        "fn": metrics["confusion_matrix"]["fn"],
                        "tp": metrics["confusion_matrix"]["tp"],
                    }
                )

    threshold_sweep = sorted(
        threshold_sweep,
        key=lambda row: (
            -float(row["f1"]),
            -float(row["recall_abnormal"]),
            float(row["fp"]),
        ),
    )

    export_prediction_records(
        val_records,
        os.path.join(save_dir, "val_predictions.csv"),
        raw_threshold=best_th,
        casewise_scores=val_casewise_scores,
        casewise_threshold=casewise_best_th,
    )
    export_prediction_records(
        test_records,
        os.path.join(save_dir, "test_predictions.csv"),
        raw_threshold=best_th,
        casewise_scores=test_casewise_scores,
        casewise_threshold=casewise_best_th,
    )
    export_prediction_records(
        train_records,
        os.path.join(save_dir, "train_predictions.csv"),
        raw_threshold=anomaly_threshold if anomaly_threshold is not None else best_th,
        casewise_scores=train_casewise_scores,
        casewise_threshold=anomaly_threshold if anomaly_threshold_use_casewise else None,
    )

    train_df = records_to_dataframe(
        train_records,
        casewise_scores=train_casewise_scores,
        raw_threshold=anomaly_threshold if anomaly_threshold is not None else best_th,
        casewise_threshold=anomaly_threshold if anomaly_threshold_use_casewise else None,
    )
    val_df = records_to_dataframe(
        val_records,
        casewise_scores=val_casewise_scores,
        raw_threshold=best_th,
        casewise_threshold=casewise_best_th,
    )
    test_df = records_to_dataframe(
        test_records,
        casewise_scores=test_casewise_scores,
        raw_threshold=best_th,
        casewise_threshold=casewise_best_th,
    )

    summary = {
        "config_path": args.config,
        "checkpoint_path": args.ckpt,
        "save_dir": save_dir,
        "split_summary": split_summary,
        "best_threshold": best_th,
        "best_threshold_score": best_score,
        "best_threshold_stats": best_stats,
        "train_metrics_05": train_metrics_05,
        "val_metrics_05": val_metrics_05,
        "test_metrics_05": test_metrics_05,
        "test_metrics_best": test_metrics_best,
        "val_casewise_metrics_05": val_casewise_metrics_05,
        "val_casewise_best": val_casewise_best,
        "test_casewise_metrics_05": test_casewise_metrics_05,
        "test_casewise_best": test_casewise_best,
        "casewise_best_threshold": casewise_best_th,
        "casewise_best_threshold_score": casewise_best_score,
        "casewise_best_threshold_stats": casewise_best_stats,
        "anomaly_threshold": anomaly_threshold,
        "anomaly_threshold_stats": anomaly_threshold_stats,
        "test_metrics_anomaly_threshold": test_metrics_anomaly_threshold,
        "test_seen_best_raw": summarize_score_binary(
            test_df[test_df["subtype_status"] == "seen"], "anomaly_score", best_th
        ),
        "test_unseen_best_raw": summarize_score_binary(
            test_df[test_df["subtype_status"] == "unseen"], "anomaly_score", best_th
        ),
        "test_by_subtype_best_raw": summarize_by_subtype(test_df, "anomaly_score", best_th),
    }

    if "casewise_score" in test_df.columns and casewise_best_th is not None:
        summary["test_seen_best_casewise"] = summarize_score_binary(
            test_df[test_df["subtype_status"] == "seen"], "casewise_score", casewise_best_th
        )
        summary["test_unseen_best_casewise"] = summarize_score_binary(
            test_df[test_df["subtype_status"] == "unseen"], "casewise_score", casewise_best_th
        )
        summary["test_by_subtype_best_casewise"] = summarize_by_subtype(
            test_df, "casewise_score", casewise_best_th
        )

    if "casewise_score" in train_df.columns and anomaly_threshold is not None and anomaly_threshold_use_casewise:
        summary["test_seen_anomaly_threshold"] = summarize_score_binary(
            test_df[test_df["subtype_status"] == "seen"], "casewise_score", anomaly_threshold
        )
        summary["test_unseen_anomaly_threshold"] = summarize_score_binary(
            test_df[test_df["subtype_status"] == "unseen"], "casewise_score", anomaly_threshold
        )

    summary["threshold_sweep"] = threshold_sweep

    results_path = os.path.join(save_dir, "results.yaml")
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(summary), f, allow_unicode=True, sort_keys=False)

    if len(threshold_sweep) > 0:
        pd.DataFrame(threshold_sweep).to_csv(os.path.join(save_dir, "threshold_sweep.csv"), index=False)

    print(f"Saved posthoc evaluation to {results_path}")
    print(f"Saved train predictions to {os.path.join(save_dir, 'train_predictions.csv')}")
    print(f"Saved val predictions to {os.path.join(save_dir, 'val_predictions.csv')}")
    print(f"Saved test predictions to {os.path.join(save_dir, 'test_predictions.csv')}")
    if len(threshold_sweep) > 0:
        print(f"Saved threshold sweep to {os.path.join(save_dir, 'threshold_sweep.csv')}")


if __name__ == "__main__":
    main()
