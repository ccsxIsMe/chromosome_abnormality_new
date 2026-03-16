# export HF_ENDPOINT=https://hf-mirror.com

import os
from xml.parsers.expat import model
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets.chromosome_dataset import ChromosomeDataset
from src.transforms import build_train_transform, build_val_transform
from src.models.build_model import build_model
from src.losses.loss_factory import build_loss
from src.utils.metrics import compute_classification_metrics, search_best_threshold
from src.utils.chromosome_vocab import build_chr_vocab_from_csv


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_chromosome_id=False, use_pair_input=False):
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
                logits = model(left_images, right_images, chr_idx)
            else:
                logits = model(left_images, right_images)

        elif use_chromosome_id:
            images = batch["image"].to(device)
            chr_idx = batch["chr_idx"].to(device)
            logits = model(images, chr_idx)

        else:
            images = batch["image"].to(device)
            logits = model(images)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5,
             use_chromosome_id=False, use_pair_input=False):
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
                logits = model(left_images, right_images, chr_idx)
            else:
                logits = model(left_images, right_images)

        elif use_chromosome_id:
            images = batch["image"].to(device)
            chr_idx = batch["chr_idx"].to(device)
            logits = model(images, chr_idx)

        else:
            images = batch["image"].to(device)
            logits = model(images)

        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics, y_true, y_prob

def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)
    pair_model_type = cfg["model"].get("pair_model_type", "siamese")

    chr_to_idx = None
    idx_to_chr = None
    if use_chromosome_id:
        chr_to_idx, idx_to_chr = build_chr_vocab_from_csv(cfg["data"]["train_csv"])

    print("Chromosome vocab:", chr_to_idx)

    if use_pair_input:
        from src.datasets.chromosome_pair_dataset import ChromosomePairDataset

        train_dataset = ChromosomePairDataset(
            cfg["data"]["train_csv"],
            transform=build_train_transform(cfg["data"]["image_size"]),
            chr_to_idx=chr_to_idx,
            use_chromosome_id=use_chromosome_id,
        )
        val_dataset = ChromosomePairDataset(
            cfg["data"]["val_csv"],
            transform=build_val_transform(cfg["data"]["image_size"]),
            chr_to_idx=chr_to_idx,
            use_chromosome_id=use_chromosome_id,
        )
        test_dataset = ChromosomePairDataset(
            cfg["data"]["test_csv"],
            transform=build_val_transform(cfg["data"]["image_size"]),
            chr_to_idx=chr_to_idx,
            use_chromosome_id=use_chromosome_id,
        )
    else:
        train_dataset = ChromosomeDataset(
            cfg["data"]["train_csv"],
            transform=build_train_transform(cfg["data"]["image_size"]),
            chr_to_idx=chr_to_idx,
            use_chromosome_id=use_chromosome_id,
        )
        val_dataset = ChromosomeDataset(
            cfg["data"]["val_csv"],
            transform=build_val_transform(cfg["data"]["image_size"]),
            chr_to_idx=chr_to_idx,
            use_chromosome_id=use_chromosome_id,
        )
        test_dataset = ChromosomeDataset(
            cfg["data"]["test_csv"],
            transform=build_val_transform(cfg["data"]["image_size"]),
            chr_to_idx=chr_to_idx,
            use_chromosome_id=use_chromosome_id,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"]
    )

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        use_chromosome_id=use_chromosome_id,
        num_chromosome_types=len(chr_to_idx) if chr_to_idx is not None else None,
        chr_embed_dim=cfg["model"].get("chr_embed_dim", 16),
        use_pair_input=use_pair_input,
        pair_model_type=pair_model_type,
    ).to(device)

    criterion = build_loss(cfg["loss"], device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    save_dir = os.path.join(cfg["output"]["save_dir"], cfg["experiment_name"])
    os.makedirs(save_dir, exist_ok=True)

    best_metric_name = cfg.get("best_model_metric", "auprc")
    best_metric = -1.0
    best_path = os.path.join(save_dir, "best_model.pth")

    for epoch in range(cfg["train"]["epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['train']['epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input
        )

        val_metrics, y_true, y_prob = evaluate(
            model, val_loader, criterion, device,
            threshold=0.5,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input
        )

        best_th, best_score, best_stats = search_best_threshold(y_true, y_prob, metric="f1")
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Metrics: {val_metrics}")
        print(f"Best Val Threshold: {best_th:.2f}, Best F1: {best_score:.4f}, Stats: {best_stats}")

        current_metric = val_metrics[best_metric_name]

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print("Loading best model for final test...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    # 在 val 上重新找 best threshold
    val_metrics, val_y_true, val_y_prob = evaluate(
        model, val_loader, criterion, device,
        threshold=0.5,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input
    )
    best_th, best_score, best_stats = search_best_threshold(val_y_true, val_y_prob, metric="f1")

    print(f"\nBest threshold selected from val: {best_th:.2f}")
    print(f"Best val stats: {best_stats}")

    # test @ 0.5
    test_metrics_05, _, _ = evaluate(
        model, test_loader, criterion, device,
        threshold=0.5,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input
    )

    print("\nFinal Test Metrics @ threshold=0.5:")
    print(f"AUROC: {test_metrics_05['auroc']}")
    print(f"AUPRC: {test_metrics_05['auprc']}")
    print(f"F1 (abnormal as positive): {test_metrics_05['f1']}")
    print(f"Balanced Accuracy: {test_metrics_05['balanced_acc']}")
    print(f"Loss: {test_metrics_05['loss']}")

    cm = test_metrics_05["confusion_matrix"]
    print("\nConfusion Matrix @0.5:")
    print(f"TN: {cm['tn']}, FP: {cm['fp']}")
    print(f"FN: {cm['fn']}, TP: {cm['tp']}")

    print("\nNormal Class Metrics @0.5:")
    print(test_metrics_05["normal"])

    print("\nAbnormal Class Metrics @0.5:")
    print(test_metrics_05["abnormal"])

    # test @ best threshold
    test_metrics_best, _, _ = evaluate(
        model, test_loader, criterion, device,
        threshold=best_th,
        use_chromosome_id=use_chromosome_id,
        use_pair_input=use_pair_input
    )

    print(f"\nFinal Test Metrics @ threshold={best_th:.2f}:")
    print(f"AUROC: {test_metrics_best['auroc']}")
    print(f"AUPRC: {test_metrics_best['auprc']}")
    print(f"F1 (abnormal as positive): {test_metrics_best['f1']}")
    print(f"Balanced Accuracy: {test_metrics_best['balanced_acc']}")
    print(f"Loss: {test_metrics_best['loss']}")

    cm = test_metrics_best["confusion_matrix"]
    print(f"\nConfusion Matrix @{best_th:.2f}:")
    print(f"TN: {cm['tn']}, FP: {cm['fp']}")
    print(f"FN: {cm['fn']}, TP: {cm['tp']}")

    print(f"\nNormal Class Metrics @{best_th:.2f}:")
    print(test_metrics_best["normal"])

    print(f"\nAbnormal Class Metrics @{best_th:.2f}:")
    print(test_metrics_best["abnormal"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)