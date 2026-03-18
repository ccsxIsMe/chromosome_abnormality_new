import os
from copy import deepcopy

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.chromosome_dataset import ChromosomeDataset
from src.losses.loss_factory import build_loss, extract_logits
from src.models.build_model import build_model
from src.transforms import build_train_transform, build_val_transform
from src.utils.chromosome_vocab import build_chr_vocab_from_csv
from src.utils.metrics import compute_classification_metrics, search_best_threshold


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, use_chromosome_id=False, use_pair_input=False):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_pair_input:
            left_images = batch["left_image"].to(device)
            right_images = batch["right_image"].to(device)

            if use_chromosome_id:
                chr_idx = batch["chr_idx"].to(device)
                model_output = model(left_images, right_images, chr_idx)
            else:
                model_output = model(left_images, right_images)

        elif use_chromosome_id:
            images = batch["image"].to(device)
            chr_idx = batch["chr_idx"].to(device)
            model_output = model(images, chr_idx)

        else:
            images = batch["image"].to(device)
            model_output = model(images)

        loss = criterion(model_output, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, use_chromosome_id=False, use_pair_input=False):
    model.eval()
    running_loss = 0.0
    y_true, y_prob = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        labels = batch["label"].to(device)

        if use_pair_input:
            left_images = batch["left_image"].to(device)
            right_images = batch["right_image"].to(device)

            if use_chromosome_id:
                chr_idx = batch["chr_idx"].to(device)
                model_output = model(left_images, right_images, chr_idx)
            else:
                model_output = model(left_images, right_images)

        elif use_chromosome_id:
            images = batch["image"].to(device)
            chr_idx = batch["chr_idx"].to(device)
            model_output = model(images, chr_idx)

        else:
            images = batch["image"].to(device)
            model_output = model(images)

        logits = extract_logits(model_output)
        loss = criterion(model_output, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics, y_true, y_prob


def build_datasets(cfg, chr_to_idx):
    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)

    if use_pair_input:
        from src.datasets.chromosome_pair_dataset import ChromosomePairDataset

        dataset_cls = ChromosomePairDataset
    else:
        dataset_cls = ChromosomeDataset

    train_dataset = dataset_cls(
        cfg["data"]["train_csv"],
        transform=build_train_transform(cfg["data"]["image_size"]),
        chr_to_idx=chr_to_idx,
        use_chromosome_id=use_chromosome_id,
    )
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
    ).to(device)

    criterion = build_loss(cfg["loss"], device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    return {
        "device": device,
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


def run_experiment(cfg, config_path=None):
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

    save_dir = os.path.join(cfg["output"]["save_dir"], cfg["experiment_name"])
    os.makedirs(save_dir, exist_ok=True)

    best_metric_name = cfg.get("best_model_metric", "auprc")
    best_metric = -1.0
    best_path = os.path.join(save_dir, "best_model.pth")

    for epoch in range(cfg["train"]["epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['train']['epochs']}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )

        val_metrics, y_true, y_prob = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
        )

        best_th, best_score, best_stats = search_best_threshold(y_true, y_prob, metric="f1")
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Metrics: {val_metrics}")
        print(f"Best Val Threshold: {best_th:.2f}, Best F1: {best_score:.4f}, Stats: {best_stats}")

        current_metric = val_metrics[best_metric_name]
        if current_metric is None:
            current_metric = -1.0

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print("Loading best model for final test...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    val_metrics_05, val_y_true, val_y_prob = evaluate(
        model,
        val_loader,
        criterion,
        device,
        threshold=0.5,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input,
    )
    best_th, best_score, best_stats = search_best_threshold(val_y_true, val_y_prob, metric="f1")

    print(f"\nBest threshold selected from val: {best_th:.2f}")
    print(f"Best val stats: {best_stats}")

    test_metrics_05, _, _ = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=0.5,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input,
    )

    print("\nFinal Test Metrics @ threshold=0.5:")
    print(f"AUROC: {test_metrics_05['auroc']}")
    print(f"AUPRC: {test_metrics_05['auprc']}")
    print(f"F1 (abnormal as positive): {test_metrics_05['f1']}")
    print(f"Balanced Accuracy: {test_metrics_05['balanced_acc']}")
    print(f"Loss: {test_metrics_05['loss']}")

    cm_05 = test_metrics_05["confusion_matrix"]
    print("\nConfusion Matrix @0.5:")
    print(f"TN: {cm_05['tn']}, FP: {cm_05['fp']}")
    print(f"FN: {cm_05['fn']}, TP: {cm_05['tp']}")
    print("\nNormal Class Metrics @0.5:")
    print(test_metrics_05["normal"])
    print("\nAbnormal Class Metrics @0.5:")
    print(test_metrics_05["abnormal"])

    test_metrics_best, _, _ = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=best_th,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input,
    )

    print(f"\nFinal Test Metrics @ threshold={best_th:.2f}:")
    print(f"AUROC: {test_metrics_best['auroc']}")
    print(f"AUPRC: {test_metrics_best['auprc']}")
    print(f"F1 (abnormal as positive): {test_metrics_best['f1']}")
    print(f"Balanced Accuracy: {test_metrics_best['balanced_acc']}")
    print(f"Loss: {test_metrics_best['loss']}")

    cm_best = test_metrics_best["confusion_matrix"]
    print(f"\nConfusion Matrix @{best_th:.2f}:")
    print(f"TN: {cm_best['tn']}, FP: {cm_best['fp']}")
    print(f"FN: {cm_best['fn']}, TP: {cm_best['tp']}")
    print(f"\nNormal Class Metrics @{best_th:.2f}:")
    print(test_metrics_best["normal"])
    print(f"\nAbnormal Class Metrics @{best_th:.2f}:")
    print(test_metrics_best["abnormal"])

    results = {
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

    results_path = os.path.join(save_dir, "results.yaml")
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)
    print(f"Saved results to {results_path}")

    return results


def main(config_path):
    cfg = load_config(config_path)
    return run_experiment(cfg, config_path=config_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
