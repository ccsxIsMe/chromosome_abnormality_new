import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.datasets.pair_composite_dataset import PairCompositeDataset
from src.utils.metrics import compute_score_based_metrics, search_best_threshold


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


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


class TensorResizeNormalize:
    def __init__(self, image_size):
        self.image_size = int(image_size)

    def __call__(self, tensor):
        tensor = tensor.float()
        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W], got {tuple(tensor.shape)}")
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor


class ResNetPatchExtractor(torch.nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, layers=("layer2", "layer3")):
        super().__init__()
        from torchvision import models

        self.layers = tuple(layers)
        self.features = {}

        if backbone == "resnet18":
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                self.model = models.resnet18(weights=weights)
            except AttributeError:
                self.model = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet50":
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                self.model = models.resnet50(weights=weights)
            except AttributeError:
                self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        for name in self.layers:
            if not hasattr(self.model, name):
                raise ValueError(f"Backbone {backbone} does not have layer: {name}")
            module = getattr(self.model, name)
            module.register_forward_hook(self._make_hook(name))

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _make_hook(self, name):
        def hook(_module, _inp, output):
            self.features[name] = output

        return hook

    @torch.no_grad()
    def forward(self, images):
        self.features = {}
        _ = self.model(images)

        feats = [self.features[name] for name in self.layers]
        target_h = max(feat.shape[-2] for feat in feats)
        target_w = max(feat.shape[-1] for feat in feats)

        resized = []
        for feat in feats:
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            resized.append(feat)

        patch_map = torch.cat(resized, dim=1)
        patch_map = F.normalize(patch_map, dim=1)
        return patch_map


def collate_records(batch):
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
        first_value = values[0]

        if torch.is_tensor(first_value):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(first_value, (int, np.integer)):
            collated[key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(first_value, (float, np.floating)):
            collated[key] = torch.tensor(values, dtype=torch.float32)
        else:
            collated[key] = values

    return collated


def flatten_patch_map(patch_map):
    batch_size, channels, height, width = patch_map.shape
    return patch_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)


def sample_patches(patch_embeddings, max_patches_per_image, generator):
    num_patches = patch_embeddings.size(0)
    if num_patches <= max_patches_per_image:
        return patch_embeddings
    indices = torch.randperm(num_patches, generator=generator)[:max_patches_per_image]
    indices = indices.to(device=patch_embeddings.device)
    return patch_embeddings[indices]


def build_memory_bank(model, loader, device, max_patches_per_image=64, max_memory_patches=20000, seed=42):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    memory_chunks = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        patch_map = model(images)
        patch_embeddings = flatten_patch_map(patch_map)

        for idx in range(images.size(0)):
            if int(labels[idx]) != 0:
                continue
            sampled = sample_patches(
                patch_embeddings[idx],
                max_patches_per_image=max_patches_per_image,
                generator=generator,
            )
            memory_chunks.append(sampled.detach().cpu())

    if len(memory_chunks) == 0:
        raise ValueError("Memory bank is empty. Expected at least one normal training sample.")

    memory_bank = torch.cat(memory_chunks, dim=0)
    if memory_bank.size(0) > max_memory_patches:
        indices = torch.randperm(memory_bank.size(0), generator=generator)[:max_memory_patches]
        memory_bank = memory_bank[indices]

    memory_bank = F.normalize(memory_bank, dim=1)
    return memory_bank


def compute_patch_min_distances(patches, memory_bank, distance="cosine", chunk_size=1024):
    min_dists = []
    for start in range(0, patches.size(0), chunk_size):
        chunk = patches[start : start + chunk_size]
        if distance == "cosine":
            sim = torch.matmul(chunk, memory_bank.t())
            dist = 1.0 - sim
        elif distance == "euclidean":
            dist = torch.cdist(chunk, memory_bank)
        else:
            raise ValueError(f"Unsupported distance: {distance}")
        min_dists.append(dist.min(dim=1).values)
    return torch.cat(min_dists, dim=0)


def aggregate_patch_scores(min_dists, topk_ratio=0.1):
    k = max(1, int(math.ceil(min_dists.numel() * topk_ratio)))
    topk_vals = torch.topk(min_dists, k=k, largest=True).values
    return float(topk_vals.mean().item())


@torch.no_grad()
def score_loader(
    model,
    loader,
    memory_bank,
    device,
    distance="cosine",
    topk_ratio=0.1,
    nn_chunk_size=1024,
):
    rows = []
    memory_bank = memory_bank.to(device)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        patch_map = model(images)
        patch_embeddings = flatten_patch_map(patch_map)

        batch_size = images.size(0)
        for idx in range(batch_size):
            image_patches = F.normalize(patch_embeddings[idx], dim=1)
            min_dists = compute_patch_min_distances(
                image_patches,
                memory_bank=memory_bank,
                distance=distance,
                chunk_size=nn_chunk_size,
            )
            score = aggregate_patch_scores(min_dists, topk_ratio=topk_ratio)

            row = {
                "label": int(batch["label"][idx]),
                "anomaly_score": score,
            }
            for key, values in batch.items():
                if key in {"image", "label"}:
                    continue
                value = values[idx]
                row[key] = value if isinstance(value, str) else str(value)
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_subtype(df, threshold):
    rows = []
    abnormal_df = df[df["label"].astype(int) == 1].copy()
    if abnormal_df.empty:
        return rows

    for subtype, group in abnormal_df.groupby("abnormal_subtype_id", dropna=False):
        scores = group["anomaly_score"].astype(float).to_numpy()
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


def summarize_binary_subset(df, threshold):
    if df.empty:
        return {"count": 0}
    metrics = compute_score_based_metrics(
        y_true=df["label"].astype(int).tolist(),
        y_score=df["anomaly_score"].astype(float).tolist(),
        threshold=threshold,
        higher_score_more_positive=True,
    )
    metrics["count"] = int(len(df))
    return metrics


def evaluate_threshold_sweep(train_df, test_df, quantiles=(0.95, 0.975, 0.99), mean_std_ks=(2.0, 2.5, 3.0)):
    normal_scores = train_df.loc[train_df["label"].astype(int) == 0, "anomaly_score"].astype(float).to_numpy()
    rows = []

    for quantile in quantiles:
        threshold = float(np.quantile(normal_scores, quantile))
        metrics = compute_score_based_metrics(
            y_true=test_df["label"].astype(int).tolist(),
            y_score=test_df["anomaly_score"].astype(float).tolist(),
            threshold=threshold,
            higher_score_more_positive=True,
        )
        rows.append(
            {
                "threshold_family": "quantile",
                "quantile": quantile,
                "mean_std_k": None,
                "threshold": threshold,
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

    mean_score = float(normal_scores.mean())
    std_score = float(normal_scores.std())
    for mean_std_k in mean_std_ks:
        threshold = mean_score + mean_std_k * std_score
        metrics = compute_score_based_metrics(
            y_true=test_df["label"].astype(int).tolist(),
            y_score=test_df["anomaly_score"].astype(float).tolist(),
            threshold=threshold,
            higher_score_more_positive=True,
        )
        rows.append(
            {
                "threshold_family": "mean_std",
                "quantile": None,
                "mean_std_k": mean_std_k,
                "threshold": float(threshold),
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

    rows = sorted(rows, key=lambda row: (-float(row["f1"]), -float(row["recall_abnormal"]), float(row["fp"])))
    return rows


def build_loader(csv_path, image_size, batch_size, num_workers):
    dataset = PairCompositeDataset(
        csv_path=csv_path,
        transform=TensorResizeNormalize(image_size=image_size),
        return_metadata=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_records,
        pin_memory=torch.cuda.is_available(),
    )


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [float(item) for item in items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--layers", default="layer2,layer3")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--distance", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--topk_ratio", type=float, default=0.1)
    parser.add_argument("--max_patches_per_image", type=int, default=64)
    parser.add_argument("--max_memory_patches", type=int, default=20000)
    parser.add_argument("--nn_chunk_size", type=int, default=1024)
    parser.add_argument("--threshold_quantiles", default="0.95,0.975,0.99")
    parser.add_argument("--threshold_mean_std_ks", default="2.0,2.5,3.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    threshold_quantiles = parse_float_list(args.threshold_quantiles)
    threshold_mean_std_ks = parse_float_list(args.threshold_mean_std_ks)
    layers = tuple(item.strip() for item in args.layers.split(",") if item.strip())
    if not layers:
        raise ValueError("At least one feature layer must be provided in --layers")
    if not 0.0 < args.topk_ratio <= 1.0:
        raise ValueError("--topk_ratio must be in (0, 1]")
    if args.max_patches_per_image <= 0:
        raise ValueError("--max_patches_per_image must be > 0")
    if args.max_memory_patches <= 0:
        raise ValueError("--max_memory_patches must be > 0")
    if args.nn_chunk_size <= 0:
        raise ValueError("--nn_chunk_size must be > 0")
    for quantile in threshold_quantiles:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Invalid threshold quantile: {quantile}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetPatchExtractor(
        backbone=args.backbone,
        pretrained=args.pretrained,
        layers=layers,
    ).to(device)
    model.eval()

    train_loader = build_loader(args.train_csv, args.image_size, args.batch_size, args.num_workers)
    val_loader = build_loader(args.val_csv, args.image_size, args.batch_size, args.num_workers)
    test_loader = build_loader(args.test_csv, args.image_size, args.batch_size, args.num_workers)

    memory_bank = build_memory_bank(
        model=model,
        loader=train_loader,
        device=device,
        max_patches_per_image=args.max_patches_per_image,
        max_memory_patches=args.max_memory_patches,
        seed=args.seed,
    )

    train_df = score_loader(
        model,
        train_loader,
        memory_bank,
        device,
        distance=args.distance,
        topk_ratio=args.topk_ratio,
        nn_chunk_size=args.nn_chunk_size,
    )
    val_df = score_loader(
        model,
        val_loader,
        memory_bank,
        device,
        distance=args.distance,
        topk_ratio=args.topk_ratio,
        nn_chunk_size=args.nn_chunk_size,
    )
    test_df = score_loader(
        model,
        test_loader,
        memory_bank,
        device,
        distance=args.distance,
        topk_ratio=args.topk_ratio,
        nn_chunk_size=args.nn_chunk_size,
    )

    best_threshold, best_score, best_stats = search_best_threshold(
        val_df["label"].astype(int).tolist(),
        val_df["anomaly_score"].astype(float).tolist(),
        metric="f1",
        higher_score_more_positive=True,
    )

    results = {
        "method": "pair_patch_memory_bank_ad",
        "backbone": args.backbone,
        "distance": args.distance,
        "pretrained": bool(args.pretrained),
        "layers": list(layers),
        "topk_ratio": args.topk_ratio,
        "max_patches_per_image": args.max_patches_per_image,
        "max_memory_patches": int(memory_bank.size(0)),
        "memory_dim": int(memory_bank.size(1)),
        "nn_chunk_size": int(args.nn_chunk_size),
        "best_threshold_from_val": float(best_threshold),
        "best_threshold_score_from_val": float(best_score),
        "best_threshold_stats_from_val": best_stats,
        "val_metrics_05": compute_score_based_metrics(
            y_true=val_df["label"].astype(int).tolist(),
            y_score=val_df["anomaly_score"].astype(float).tolist(),
            threshold=0.5,
            higher_score_more_positive=True,
        ),
        "val_metrics_best": compute_score_based_metrics(
            y_true=val_df["label"].astype(int).tolist(),
            y_score=val_df["anomaly_score"].astype(float).tolist(),
            threshold=best_threshold,
            higher_score_more_positive=True,
        ),
        "test_metrics_05": compute_score_based_metrics(
            y_true=test_df["label"].astype(int).tolist(),
            y_score=test_df["anomaly_score"].astype(float).tolist(),
            threshold=0.5,
            higher_score_more_positive=True,
        ),
        "test_metrics_best": compute_score_based_metrics(
            y_true=test_df["label"].astype(int).tolist(),
            y_score=test_df["anomaly_score"].astype(float).tolist(),
            threshold=best_threshold,
            higher_score_more_positive=True,
        ),
        "test_seen_best": summarize_binary_subset(test_df[test_df["subtype_status"] == "seen"], best_threshold)
        if "subtype_status" in test_df.columns
        else None,
        "test_unseen_best": summarize_binary_subset(test_df[test_df["subtype_status"] == "unseen"], best_threshold)
        if "subtype_status" in test_df.columns
        else None,
        "test_by_subtype_best": summarize_by_subtype(test_df, best_threshold)
        if "abnormal_subtype_id" in test_df.columns
        else None,
    }

    threshold_sweep = evaluate_threshold_sweep(
        train_df,
        test_df,
        quantiles=tuple(threshold_quantiles),
        mean_std_ks=tuple(threshold_mean_std_ks),
    )
    results["threshold_sweep"] = threshold_sweep

    train_df.to_csv(output_dir / "train_predictions.csv", index=False)
    val_df.to_csv(output_dir / "val_predictions.csv", index=False)
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(threshold_sweep).to_csv(output_dir / "threshold_sweep.csv", index=False)

    with open(output_dir / "results.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    print(f"Saved results to {output_dir / 'results.yaml'}")
    print(f"Saved threshold sweep to {output_dir / 'threshold_sweep.csv'}")


if __name__ == "__main__":
    main()
