import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.chromosome_dataset import ChromosomeDataset
from src.losses.loss_factory import (
    build_loss,
    extract_anomaly_scores,
    extract_embeddings,
    extract_logits,
)
from src.models.build_model import build_model
from src.transforms import (
    build_train_transform,
    build_style_transform,
    build_val_transform,
)
from src.utils.casewise_calibration import (
    calibrate_scores_casewise,
    evaluate_casewise_calibration,
    search_best_threshold_casewise,
    summarize_case_isolation,
)
from src.utils.chromosome_vocab import build_chr_vocab_from_csv
from src.utils.metrics import (
    compute_classification_metrics,
    compute_multiclass_metrics,
    compute_score_based_metrics,
    search_best_threshold,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_load_state_dict(model, path, device):
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)


def _forward_model(batch, model, device, use_chromosome_id=False, use_pair_input=False, use_style_view=False):
    if use_pair_input:
        left_key = "left_image_style" if use_style_view else "left_image"
        right_key = "right_image_style" if use_style_view else "right_image"

        left_images = batch[left_key].to(device)
        right_images = batch[right_key].to(device)

        if use_chromosome_id:
            chr_idx = batch["chr_idx"].to(device)
            return model(left_images, right_images, chr_idx)

        return model(left_images, right_images)

    image_key = "image_style" if use_style_view else "image"
    images = batch[image_key].to(device)

    if use_chromosome_id:
        chr_idx = batch["chr_idx"].to(device)
        return model(images, chr_idx)

    return model(images)


def compute_style_consistency_loss(
    clean_output,
    style_output,
    teacher_detach=True,
    embedding_weight=1.0,
    score_weight=1.0,
    embedding_mode="cosine",
    score_mode="mse",
):
    total = None
    logs = {
        "embedding_consistency": 0.0,
        "score_consistency": 0.0,
    }

    clean_emb = extract_embeddings(clean_output)
    style_emb = extract_embeddings(style_output)

    clean_score = extract_anomaly_scores(clean_output)
    style_score = extract_anomaly_scores(style_output)

    if clean_emb is not None and style_emb is not None and embedding_weight > 0:
        teacher_emb = clean_emb.detach() if teacher_detach else clean_emb

        if embedding_mode == "cosine":
            emb_loss = 1.0 - F.cosine_similarity(
                F.normalize(style_emb, dim=1),
                F.normalize(teacher_emb, dim=1),
                dim=1,
            ).mean()
        elif embedding_mode == "mse":
            emb_loss = F.mse_loss(style_emb, teacher_emb)
        else:
            raise ValueError(f"Unsupported embedding consistency mode: {embedding_mode}")

        total = embedding_weight * emb_loss if total is None else total + embedding_weight * emb_loss
        logs["embedding_consistency"] = float(emb_loss.item())

    if clean_score is not None and style_score is not None and score_weight > 0:
        teacher_score = clean_score.detach() if teacher_detach else clean_score

        if score_mode == "mse":
            score_loss = F.mse_loss(style_score, teacher_score)
        elif score_mode == "l1":
            score_loss = F.l1_loss(style_score, teacher_score)
        elif score_mode == "smooth_l1":
            score_loss = F.smooth_l1_loss(style_score, teacher_score)
        else:
            raise ValueError(f"Unsupported score consistency mode: {score_mode}")

        total = score_weight * score_loss if total is None else total + score_weight * score_loss
        logs["score_consistency"] = float(score_loss.item())

    if total is None:
        total = clean_output["embedding"].new_tensor(0.0)

    return total, logs


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_chromosome_id=False,
    use_pair_input=False,
    style_consistency_cfg=None,
):
    model.train()
    running_loss = 0.0
    running_main_loss = 0.0
    running_style_task_loss = 0.0
    running_embedding_consistency = 0.0
    running_score_consistency = 0.0

    style_enabled = bool(style_consistency_cfg and style_consistency_cfg.get("enabled", False))

    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        clean_output = _forward_model(
            batch=batch,
            model=model,
            device=device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
            use_style_view=False,
        )
        main_loss = criterion(clean_output, labels)
        total_loss = main_loss

        style_task_loss_value = 0.0
        emb_cons_value = 0.0
        score_cons_value = 0.0

        if style_enabled and "left_image_style" in batch and "right_image_style" in batch:
            style_output = _forward_model(
                batch=batch,
                model=model,
                device=device,
                use_chromosome_id=use_chromosome_id,
                use_pair_input=use_pair_input,
                use_style_view=True,
            )

            style_task_weight = style_consistency_cfg.get("style_task_weight", 0.5)
            embedding_weight = style_consistency_cfg.get("embedding_weight", 0.5)
            score_weight = style_consistency_cfg.get("score_weight", 0.5)
            teacher_detach = style_consistency_cfg.get("teacher_detach", True)
            embedding_mode = style_consistency_cfg.get("embedding_mode", "cosine")
            score_mode = style_consistency_cfg.get("score_mode", "mse")

            style_task_loss = criterion(style_output, labels)
            consistency_loss, consistency_logs = compute_style_consistency_loss(
                clean_output=clean_output,
                style_output=style_output,
                teacher_detach=teacher_detach,
                embedding_weight=embedding_weight,
                score_weight=score_weight,
                embedding_mode=embedding_mode,
                score_mode=score_mode,
            )

            total_loss = total_loss + style_task_weight * style_task_loss + consistency_loss

            style_task_loss_value = float(style_task_loss.item())
            emb_cons_value = consistency_logs["embedding_consistency"]
            score_cons_value = consistency_logs["score_consistency"]

        total_loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += total_loss.item() * batch_size
        running_main_loss += main_loss.item() * batch_size
        running_style_task_loss += style_task_loss_value * batch_size
        running_embedding_consistency += emb_cons_value * batch_size
        running_score_consistency += score_cons_value * batch_size

    logs = {
        "total_loss": running_loss / len(loader.dataset),
        "main_loss": running_main_loss / len(loader.dataset),
        "style_task_loss": running_style_task_loss / len(loader.dataset),
        "embedding_consistency": running_embedding_consistency / len(loader.dataset),
        "score_consistency": running_score_consistency / len(loader.dataset),
    }
    return logs


@torch.no_grad()
def evaluate_classifier(model, loader, criterion, device, threshold=0.5, use_chromosome_id=False, use_pair_input=False):
    model.eval()
    running_loss = 0.0
    y_true, y_prob = [], []
    num_classes = None

    for batch in tqdm(loader, desc="EvalClassifier", leave=False):
        labels = batch["label"].to(device)

        model_output = _forward_model(
            batch=batch,
            model=model,
            device=device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
            use_style_view=False,
        )

        logits = extract_logits(model_output)
        loss = criterion(model_output, labels)
        probs = torch.softmax(logits, dim=1)
        if num_classes is None:
            num_classes = probs.shape[1]

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        y_true.extend(labels.cpu().numpy().tolist())
        if probs.shape[1] == 2:
            y_prob.extend(probs[:, 1].cpu().numpy().tolist())
        else:
            y_prob.extend(probs.cpu().numpy().tolist())

    if num_classes == 2:
        metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)
    else:
        metrics = compute_multiclass_metrics(y_true, y_prob)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics, y_true, y_prob


@torch.no_grad()
def evaluate_multi_prototype_metric(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
    use_chromosome_id=False,
    use_pair_input=False,
):
    model.eval()
    running_loss = 0.0
    y_true, y_score, case_ids = [], [], []

    for batch in tqdm(loader, desc="EvalMultiProto", leave=False):
        labels = batch["label"].to(device)

        model_output = _forward_model(
            batch=batch,
            model=model,
            device=device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
            use_style_view=False,
        )

        scores = extract_anomaly_scores(model_output)
        if scores is None:
            raise ValueError("multi_prototype_metric mode requires model output to contain 'anomaly_score'")

        loss = criterion(model_output, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        y_true.extend(labels.cpu().numpy().tolist())
        y_score.extend(scores.detach().cpu().numpy().tolist())

        if "case_id" in batch:
            batch_case_ids = batch["case_id"]
            if isinstance(batch_case_ids, list):
                case_ids.extend([str(x) for x in batch_case_ids])
            else:
                case_ids.extend([str(x) for x in batch_case_ids])

    metrics = compute_score_based_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
        higher_score_more_positive=True,
    )
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics, y_true, y_score, case_ids

def build_datasets(cfg, chr_to_idx):
    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)

    style_cfg = cfg.get("style_consistency", {})
    style_enabled = style_cfg.get("enabled", False)

    if use_pair_input:
        from src.datasets.chromosome_pair_dataset import ChromosomePairDataset
        dataset_cls = ChromosomePairDataset
    else:
        dataset_cls = ChromosomeDataset

    train_kwargs = {
        "csv_path": cfg["data"]["train_csv"],
        "transform": build_train_transform(cfg["data"]["image_size"]),
        "chr_to_idx": chr_to_idx,
        "use_chromosome_id": use_chromosome_id,
    }

    if use_pair_input:
        train_kwargs["return_style_view"] = style_enabled
        train_kwargs["style_transform"] = build_style_transform(cfg["data"]["image_size"]) if style_enabled else None

    train_dataset = dataset_cls(**train_kwargs)

    val_dataset = dataset_cls(
        cfg["data"]["val_csv"],
        transform=build_val_transform(cfg["data"]["image_size"]),
        chr_to_idx=chr_to_idx,
        use_chromosome_id=use_chromosome_id,
    )
    test_dataset = dataset_cls(
        cfg["data"]["test_csv"],
        transform=build_val_transform(cfg["data"]["image_size"]),
        chr_to_idx=chr_to_idx,
        use_chromosome_id=use_chromosome_id,
    )
    return train_dataset, val_dataset, test_dataset


def build_loaders(cfg, train_dataset, val_dataset, test_dataset):
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def build_training_context(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    experiment_mode = cfg.get("experiment_mode", "classifier")

    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)
    pair_model_type = cfg["model"].get("pair_model_type", "siamese")

    chr_to_idx = None
    if use_chromosome_id:
        chr_to_idx, _ = build_chr_vocab_from_csv(cfg["data"]["train_csv"])

    print("Chromosome vocab:", chr_to_idx)

    train_dataset, val_dataset, test_dataset = build_datasets(cfg, chr_to_idx)
    train_loader, val_loader, test_loader = build_loaders(cfg, train_dataset, val_dataset, test_dataset)

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        use_chromosome_id=use_chromosome_id,
        num_chromosome_types=len(chr_to_idx) if chr_to_idx is not None else None,
        chr_embed_dim=cfg["model"].get("chr_embed_dim", 16),
        use_pair_input=use_pair_input,
        pair_model_type=pair_model_type,
        use_pair_mixstyle=cfg["model"].get("use_pair_mixstyle", False),
        mixstyle_p=cfg["model"].get("mixstyle_p", 0.5),
        mixstyle_alpha=cfg["model"].get("mixstyle_alpha", 0.1),
        experiment_mode=experiment_mode,
        num_prototypes=cfg["model"].get("num_prototypes", 4),
        prototype_distance=cfg["model"].get("prototype_distance", "cosine"),
        normalize_prototype_embedding=cfg["model"].get("normalize_prototype_embedding", True),
    ).to(device)

    criterion = build_loss(cfg["loss"], device, experiment_mode=experiment_mode)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    return {
        "device": device,
        "experiment_mode": experiment_mode,
        "use_chromosome_id": use_chromosome_id,
        "use_pair_input": use_pair_input,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


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


def _print_final_metrics(title, metrics, threshold=None):
    print("\n" + "=" * 80)
    if threshold is None:
        print(title)
    else:
        print(f"{title} @ threshold={threshold:.4f}")
    print("=" * 80)

    print(f"AUROC: {metrics['auroc']}")
    print(f"AUPRC: {metrics['auprc']}")
    print(f"F1 (abnormal as positive): {metrics['f1']}")
    print(f"Balanced Accuracy: {metrics['balanced_acc']}")
    print(f"Abnormal Recall: {metrics['abnormal']['recall']}")
    print(f"Loss: {metrics['loss']}")

    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix:")
    print(f"TN: {cm['tn']}, FP: {cm['fp']}")
    print(f"FN: {cm['fn']}, TP: {cm['tp']}")

    print("\nNormal Class Metrics:")
    print(metrics["normal"])
    print("\nAbnormal Class Metrics:")
    print(metrics["abnormal"])


def _print_final_multiclass_metrics(title, metrics):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Top-1 Accuracy: {metrics['top1_acc']}")
    if "top3_acc" in metrics:
        print(f"Top-3 Accuracy: {metrics['top3_acc']}")
    print(f"Macro-F1: {metrics['macro_f1']}")
    print(f"Balanced Accuracy: {metrics['balanced_acc']}")
    print(f"Loss: {metrics['loss']}")


def run_classifier_experiment(cfg, config_path=None):
    cfg = deepcopy(cfg)
    context = build_training_context(cfg)

    device = context["device"]
    use_chromosome_id = context["use_chromosome_id"]
    use_pair_input = context["use_pair_input"]
    train_loader = context["train_loader"]
    val_loader = context["val_loader"]
    test_loader = context["test_loader"]
    model = context["model"]
    criterion = context["criterion"]
    optimizer = context["optimizer"]
    scheduler = context["scheduler"]
    num_classes = cfg["model"]["num_classes"]

    save_dir = os.path.join(cfg["output"]["save_dir"], cfg["experiment_name"])
    os.makedirs(save_dir, exist_ok=True)

    best_metric_name = cfg.get("best_model_metric", "auprc")
    best_metric = -1.0
    best_path = os.path.join(save_dir, "best_model.pth")

    for epoch in range(cfg["train"]["epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['train']['epochs']}")

        train_logs = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
            style_consistency_cfg=None,
        )

        val_metrics, y_true, y_prob = evaluate_classifier(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )
        scheduler.step()

        print(f"Train Logs: {train_logs}")
        print(f"Val Metrics: {val_metrics}")
        if num_classes == 2:
            best_th, best_score, best_stats = search_best_threshold(
                y_true, y_prob, metric="f1", higher_score_more_positive=True
            )
            print(f"Best Val Threshold: {best_th:.4f}, Best F1: {best_score:.4f}, Stats: {best_stats}")

        current_metric = val_metrics.get(best_metric_name)
        if current_metric is None:
            current_metric = -1.0

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print("Loading best model for final test...")
    _safe_load_state_dict(model, best_path, device)

    if num_classes == 2:
        val_metrics_05, val_y_true, val_y_prob = evaluate_classifier(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )
        best_th, best_score, best_stats = search_best_threshold(
            val_y_true, val_y_prob, metric="f1", higher_score_more_positive=True
        )

        test_metrics_05, _, _ = evaluate_classifier(
            model,
            test_loader,
            criterion,
            device,
            threshold=0.5,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )
        test_metrics_best, _, _ = evaluate_classifier(
            model,
            test_loader,
            criterion,
            device,
            threshold=best_th,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )

        _print_final_metrics("Final Test Metrics", test_metrics_05, threshold=0.5)
        _print_final_metrics("Final Test Metrics", test_metrics_best, threshold=best_th)

        results = {
            "experiment_mode": "classifier",
            "config_path": config_path,
            "experiment_name": cfg["experiment_name"],
            "save_dir": save_dir,
            "best_model_path": best_path,
            "best_model_metric": best_metric_name,
            "best_model_metric_value": best_metric,
            "best_threshold": best_th,
            "best_threshold_score": best_score,
            "best_threshold_stats": best_stats,
            "val_metrics_05": val_metrics_05,
            "test_metrics_05": test_metrics_05,
            "test_metrics_best": test_metrics_best,
        }
    else:
        val_metrics, _, _ = evaluate_classifier(
            model,
            val_loader,
            criterion,
            device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )
        test_metrics, _, _ = evaluate_classifier(
            model,
            test_loader,
            criterion,
            device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )

        _print_final_multiclass_metrics("Final Val Metrics", val_metrics)
        _print_final_multiclass_metrics("Final Test Metrics", test_metrics)

        results = {
            "experiment_mode": "classifier",
            "config_path": config_path,
            "experiment_name": cfg["experiment_name"],
            "save_dir": save_dir,
            "best_model_path": best_path,
            "best_model_metric": best_metric_name,
            "best_model_metric_value": best_metric,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    results_path = os.path.join(save_dir, "results.yaml")
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)
    print(f"Saved results to {results_path}")

    return results


def run_multi_prototype_metric_experiment(cfg, config_path=None):
    cfg = deepcopy(cfg)
    context = build_training_context(cfg)

    device = context["device"]
    use_chromosome_id = context["use_chromosome_id"]
    use_pair_input = context["use_pair_input"]
    train_loader = context["train_loader"]
    val_loader = context["val_loader"]
    test_loader = context["test_loader"]
    model = context["model"]
    criterion = context["criterion"]
    optimizer = context["optimizer"]
    scheduler = context["scheduler"]

    if not use_chromosome_id:
        raise ValueError("multi_prototype_metric mode requires use_chromosome_id=True")

    save_dir = os.path.join(cfg["output"]["save_dir"], cfg["experiment_name"])
    os.makedirs(save_dir, exist_ok=True)

    best_metric_name = cfg.get("best_model_metric", "auprc")
    best_metric = -1.0
    best_path = os.path.join(save_dir, "best_model.pth")

    style_consistency_cfg = cfg.get("style_consistency", {})
    calibration_cfg = cfg.get("calibration", {})
    calibration_enabled = calibration_cfg.get("enabled", True)
    calibration_method = calibration_cfg.get("method", "zscore")

    # split isolation summary
    split_summary = summarize_case_isolation(
        train_csv=cfg["data"]["train_csv"],
        val_csv=cfg["data"]["val_csv"],
        test_csv=cfg["data"]["test_csv"],
    )
    print("\nSplit isolation summary:")
    print(split_summary)

    for epoch in range(cfg["train"]["epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['train']['epochs']}")

        train_logs = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
            style_consistency_cfg=style_consistency_cfg,
        )

        val_metrics, y_true, y_score, val_case_ids = evaluate_multi_prototype_metric(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )

        best_th, best_score, best_stats = search_best_threshold(
            y_true,
            y_score,
            metric="f1",
            higher_score_more_positive=True,
        )

        if calibration_enabled and len(val_case_ids) == len(y_score) and len(val_case_ids) > 0:
            cal_best_th, cal_best_score, cal_best_stats, _ = search_best_threshold_casewise(
                y_true=y_true,
                raw_scores=y_score,
                case_ids=val_case_ids,
                metric="f1",
                method=calibration_method,
            )
        else:
            cal_best_th, cal_best_score, cal_best_stats = None, None, None

        scheduler.step()

        print(f"Train Logs: {train_logs}")
        print(f"Val Metrics (multi-prototype anomaly score @0.5): {val_metrics}")
        print(f"Best Val Threshold: {best_th:.4f}, Best F1: {best_score:.4f}, Stats: {best_stats}")

        if cal_best_th is not None:
            print(
                f"Best Val Threshold (case-wise {calibration_method}): "
                f"{cal_best_th:.4f}, Best F1: {cal_best_score:.4f}, Stats: {cal_best_stats}"
            )

        current_metric = val_metrics.get(best_metric_name)
        if current_metric is None:
            current_metric = -1.0

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print("Loading best model for final test...")
    _safe_load_state_dict(model, best_path, device)

    # ---------- raw val ----------
    val_metrics_05, val_y_true, val_y_score, val_case_ids = evaluate_multi_prototype_metric(
        model,
        val_loader,
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

    # ---------- raw test ----------
    test_metrics_05, test_y_true, test_y_score, test_case_ids = evaluate_multi_prototype_metric(
        model,
        test_loader,
        criterion,
        device,
        threshold=0.5,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input,
    )
    test_metrics_best, _, _, _ = evaluate_multi_prototype_metric(
        model,
        test_loader,
        criterion,
        device,
        threshold=best_th,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input,
    )

    _print_final_metrics("Final Test Metrics", test_metrics_05, threshold=0.5)
    _print_final_metrics("Final Test Metrics", test_metrics_best, threshold=best_th)

    # ---------- case-wise calibrated evaluation ----------
    val_casewise_metrics_05 = None
    val_casewise_best = None
    test_casewise_metrics_05 = None
    test_casewise_best = None
    casewise_best_th = None
    casewise_best_score = None
    casewise_best_stats = None

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

        print("\n" + "=" * 80)
        print(f"Case-wise calibration enabled: method={calibration_method}")
        print("=" * 80)
        _print_final_metrics(
            f"Case-wise calibrated Test Metrics ({calibration_method})",
            test_casewise_metrics_05,
            threshold=0.5,
        )
        _print_final_metrics(
            f"Case-wise calibrated Test Metrics ({calibration_method})",
            test_casewise_best,
            threshold=casewise_best_th,
        )
    else:
        print("\nCase-wise calibration skipped: missing usable case_id in val/test.")

    results = {
        "experiment_mode": "multi_prototype_metric",
        "config_path": config_path,
        "experiment_name": cfg["experiment_name"],
        "save_dir": save_dir,
        "best_model_path": best_path,
        "best_model_metric": best_metric_name,
        "best_model_metric_value": best_metric,
        "best_threshold": best_th,
        "best_threshold_score": best_score,
        "best_threshold_stats": best_stats,
        "split_summary": split_summary,
        "model_settings": {
            "num_prototypes": cfg["model"].get("num_prototypes", 4),
            "prototype_distance": cfg["model"].get("prototype_distance", "cosine"),
            "normalize_prototype_embedding": cfg["model"].get("normalize_prototype_embedding", True),
        },
        "loss_settings": cfg.get("loss", {}),
        "style_consistency": style_consistency_cfg,
        "calibration": calibration_cfg,
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
    }

    results_path = os.path.join(save_dir, "results.yaml")
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)
    print(f"Saved results to {results_path}")

    return results


def run_experiment(cfg, config_path=None):
    experiment_mode = cfg.get("experiment_mode", "classifier")

    if experiment_mode == "classifier":
        return run_classifier_experiment(cfg, config_path=config_path)

    if experiment_mode == "multi_prototype_metric":
        return run_multi_prototype_metric_experiment(cfg, config_path=config_path)

    raise ValueError(f"Unsupported experiment_mode: {experiment_mode}")


def main(config_path):
    cfg = load_config(config_path)
    return run_experiment(cfg, config_path=config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
