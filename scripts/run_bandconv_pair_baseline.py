import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.models.bandconv_pair_model import BandConvPairClassifier
from src.utils.casewise_calibration import summarize_case_isolation
from src.utils.haar_pair_features import extract_pair_band_representations_from_paths
from src.utils.metrics import compute_classification_metrics, search_best_threshold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class PrecomputedBandPairDataset(Dataset):
    def __init__(self, arrays):
        self.left_band = torch.from_numpy(arrays["left_band"]).float()
        self.right_band = torch.from_numpy(arrays["right_band"]).float()
        self.left_profile = torch.from_numpy(arrays["left_profile"]).float()
        self.right_profile = torch.from_numpy(arrays["right_profile"]).float()
        self.left_width = torch.from_numpy(arrays["left_width"]).float()
        self.right_width = torch.from_numpy(arrays["right_width"]).float()
        self.labels = torch.from_numpy(arrays["labels"]).long()
        self.case_ids = arrays["case_ids"].astype(str)
        self.chromosome_ids = arrays["chromosome_ids"].astype(str)
        self.pair_keys = arrays["pair_keys"].astype(str)
        self.left_paths = arrays["left_paths"].astype(str)
        self.right_paths = arrays["right_paths"].astype(str)
        self.abnormal_subtype_ids = arrays["abnormal_subtype_ids"].astype(str)
        self.subtype_status = arrays["subtype_status"].astype(str)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "left_band": self.left_band[idx],
            "right_band": self.right_band[idx],
            "left_profile": self.left_profile[idx],
            "right_profile": self.right_profile[idx],
            "left_width": self.left_width[idx],
            "right_width": self.right_width[idx],
            "label": self.labels[idx],
            "case_id": self.case_ids[idx],
            "chromosome_id": self.chromosome_ids[idx],
            "pair_key": self.pair_keys[idx],
            "left_path": self.left_paths[idx],
            "right_path": self.right_paths[idx],
            "abnormal_subtype_id": self.abnormal_subtype_ids[idx],
            "subtype_status": self.subtype_status[idx],
        }


def build_or_load_precomputed_arrays(
    csv_path,
    cache_path,
    profile_length,
    band_width,
    representation_version,
    rebuild_cache=False,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild_cache:
        loaded = np.load(cache_path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}

    df = pd.read_csv(csv_path)
    left_band = []
    right_band = []
    left_profile = []
    right_profile = []
    left_width = []
    right_width = []
    labels = []
    case_ids = []
    chromosome_ids = []
    pair_keys = []
    left_paths = []
    right_paths = []
    abnormal_subtype_ids = []
    subtype_status = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Precompute-{Path(csv_path).stem}", leave=False):
        left_repr, right_repr = extract_pair_band_representations_from_paths(
            left_path=row["left_path"],
            right_path=row["right_path"],
            profile_length=profile_length,
            band_width=band_width,
            representation_version=representation_version,
        )
        left_band.append(left_repr.band_image[None, :, :].astype(np.float32))
        right_band.append(right_repr.band_image[None, :, :].astype(np.float32))
        left_profile.append(left_repr.profile[None, :].astype(np.float32))
        right_profile.append(right_repr.profile[None, :].astype(np.float32))
        left_width.append(left_repr.width_profile[None, :].astype(np.float32))
        right_width.append(right_repr.width_profile[None, :].astype(np.float32))
        labels.append(int(row["label"]))
        case_ids.append(str(row["case_id"]) if "case_id" in df.columns else "")
        chromosome_ids.append(str(row["chromosome_id"]))
        pair_keys.append(str(row["pair_key"]) if "pair_key" in df.columns else "")
        left_paths.append(str(row["left_path"]))
        right_paths.append(str(row["right_path"]))
        abnormal_subtype_ids.append(
            str(row["abnormal_subtype_id"]) if "abnormal_subtype_id" in df.columns and not pd.isna(row["abnormal_subtype_id"]) else ""
        )
        subtype_status.append(
            str(row["subtype_status"]) if "subtype_status" in df.columns and not pd.isna(row["subtype_status"]) else ""
        )

    arrays = {
        "left_band": np.stack(left_band, axis=0).astype(np.float32),
        "right_band": np.stack(right_band, axis=0).astype(np.float32),
        "left_profile": np.stack(left_profile, axis=0).astype(np.float32),
        "right_profile": np.stack(right_profile, axis=0).astype(np.float32),
        "left_width": np.stack(left_width, axis=0).astype(np.float32),
        "right_width": np.stack(right_width, axis=0).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "case_ids": np.asarray(case_ids, dtype=object),
        "chromosome_ids": np.asarray(chromosome_ids, dtype=object),
        "pair_keys": np.asarray(pair_keys, dtype=object),
        "left_paths": np.asarray(left_paths, dtype=object),
        "right_paths": np.asarray(right_paths, dtype=object),
        "abnormal_subtype_ids": np.asarray(abnormal_subtype_ids, dtype=object),
        "subtype_status": np.asarray(subtype_status, dtype=object),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def build_loader(arrays, batch_size, num_workers, shuffle=False, weighted_sampler=False, seed=42):
    dataset = PrecomputedBandPairDataset(arrays)
    sampler = None
    if weighted_sampler:
        labels = arrays["labels"].astype(np.int64)
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_weights = {int(label): 1.0 / max(int(count), 1) for label, count in zip(unique_labels, counts)}
        sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float64)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(len(sample_weights)),
            replacement=True,
            generator=generator,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
    )
    return dataset, loader


def forward_model(model, batch, device):
    return model(
        left_band=batch["left_band"].to(device),
        right_band=batch["right_band"].to(device),
        left_profile=batch["left_profile"].to(device),
        right_profile=batch["right_profile"].to(device),
        left_width=batch["left_width"].to(device),
        right_width=batch["right_width"].to(device),
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true = []
    y_prob = []

    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        output = forward_model(model, batch, device)
        logits = output["logits"]
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_prob = []
    records = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        labels = batch["label"].to(device)
        output = forward_model(model, batch, device)
        logits = output["logits"]
        loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        prob_np = probs.detach().cpu().numpy()
        label_np = labels.detach().cpu().numpy()
        y_true.extend(label_np.tolist())
        y_prob.extend(prob_np.tolist())

        for idx in range(batch_size):
            records.append(
                {
                    "label": int(label_np[idx]),
                    "score": float(prob_np[idx]),
                    "case_id": str(batch["case_id"][idx]),
                    "chromosome_id": str(batch["chromosome_id"][idx]),
                    "pair_key": str(batch["pair_key"][idx]),
                    "left_path": str(batch["left_path"][idx]),
                    "right_path": str(batch["right_path"][idx]),
                    "abnormal_subtype_id": str(batch["abnormal_subtype_id"][idx]),
                    "subtype_status": str(batch["subtype_status"][idx]),
                }
            )

    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics, y_true, y_prob, records


def summarize_by_subtype(records, threshold):
    df = pd.DataFrame(records)
    if df.empty or "abnormal_subtype_id" not in df.columns:
        return []
    abnormal_df = df[df["label"].astype(int) == 1].copy()
    if abnormal_df.empty:
        return []

    rows = []
    for subtype_id, group in abnormal_df.groupby("abnormal_subtype_id", dropna=False):
        scores = group["score"].astype(float).to_numpy()
        pred = (scores >= float(threshold)).astype(int)
        rows.append(
            {
                "abnormal_subtype_id": "" if pd.isna(subtype_id) else str(subtype_id),
                "chromosome_id": str(group["chromosome_id"].iloc[0]),
                "subtype_status": str(group["subtype_status"].iloc[0]) if "subtype_status" in group.columns else "",
                "count": int(len(group)),
                "recall_at_threshold": float(pred.mean()),
                "mean_score": float(scores.mean()),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
            }
        )
    return rows


def export_prediction_records(records, output_path, threshold_05=0.5, threshold_best=None):
    df = pd.DataFrame(records).copy()
    if df.empty:
        df.to_csv(output_path, index=False)
        return
    df["pred_label_05"] = (df["score"].astype(float) >= float(threshold_05)).astype(int)
    if threshold_best is not None:
        df["pred_label_best"] = (df["score"].astype(float) >= float(threshold_best)).astype(int)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profile_length", type=int, default=128)
    parser.add_argument("--band_width", type=int, default=32)
    parser.add_argument("--representation_version", default="v2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--weighted_sampler", action="store_true")
    parser.add_argument("--best_model_metric", default="auprc", choices=["auprc", "best_f1"])
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "precomputed_cache"
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    split_summary = summarize_case_isolation(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
    )
    print("Split isolation summary:", split_summary)

    train_arrays = build_or_load_precomputed_arrays(
        csv_path=args.train_csv,
        cache_path=cache_dir / "train.npz",
        profile_length=args.profile_length,
        band_width=args.band_width,
        representation_version=args.representation_version,
        rebuild_cache=args.rebuild_cache,
    )
    val_arrays = build_or_load_precomputed_arrays(
        csv_path=args.val_csv,
        cache_path=cache_dir / "val.npz",
        profile_length=args.profile_length,
        band_width=args.band_width,
        representation_version=args.representation_version,
        rebuild_cache=args.rebuild_cache,
    )
    test_arrays = build_or_load_precomputed_arrays(
        csv_path=args.test_csv,
        cache_path=cache_dir / "test.npz",
        profile_length=args.profile_length,
        band_width=args.band_width,
        representation_version=args.representation_version,
        rebuild_cache=args.rebuild_cache,
    )

    _, train_loader = build_loader(
        train_arrays,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        weighted_sampler=args.weighted_sampler,
        seed=args.seed,
    )
    _, val_loader = build_loader(
        val_arrays,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        weighted_sampler=False,
    )
    _, test_loader = build_loader(
        test_arrays,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        weighted_sampler=False,
    )

    model = BandConvPairClassifier(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))

    best_metric = -1.0
    best_path = output_dir / "best_model.pth"

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics_05, val_y_true, val_y_prob, _ = evaluate(model, val_loader, device, threshold=0.5)
        best_threshold, best_score, best_stats = search_best_threshold(val_y_true, val_y_prob, metric="f1")
        scheduler.step()

        print(f"Train summary: loss={train_metrics['loss']:.4f}, auprc={train_metrics['auprc']:.4f}, f1@0.5={train_metrics['f1']:.4f}")
        print(f"Val summary @0.5: loss={val_metrics_05['loss']:.4f}, auprc={val_metrics_05['auprc']:.4f}, auroc={val_metrics_05['auroc']:.4f}")
        print(f"Val best threshold: {best_threshold:.4f}, best_f1={best_score:.4f}, stats={best_stats}")

        current_metric = float(val_metrics_05["auprc"]) if args.best_model_metric == "auprc" else float(best_score)
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print("Loading best model for final evaluation...")
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)

    val_metrics_05, val_y_true, val_y_prob, val_records = evaluate(model, val_loader, device, threshold=0.5)
    best_threshold, best_score, best_stats = search_best_threshold(val_y_true, val_y_prob, metric="f1")
    val_metrics_best, _, _, _ = evaluate(model, val_loader, device, threshold=best_threshold)
    test_metrics_05, test_y_true, test_y_prob, test_records = evaluate(model, test_loader, device, threshold=0.5)
    test_metrics_best, _, _, _ = evaluate(model, test_loader, device, threshold=best_threshold)

    export_prediction_records(val_records, output_dir / "val_predictions.csv", threshold_05=0.5, threshold_best=best_threshold)
    export_prediction_records(test_records, output_dir / "test_predictions.csv", threshold_05=0.5, threshold_best=best_threshold)

    results = {
        "method": "bandconv_pair_baseline",
        "task": "supervised_pair_abnormality",
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "split_summary": split_summary,
        "representation_settings": {
            "profile_length": int(args.profile_length),
            "band_width": int(args.band_width),
            "representation_version": args.representation_version,
        },
        "training_settings": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "hidden_dim": int(args.hidden_dim),
            "weighted_sampler": bool(args.weighted_sampler),
            "best_model_metric": args.best_model_metric,
            "seed": int(args.seed),
        },
        "val_metrics_05": val_metrics_05,
        "val_metrics_best": val_metrics_best,
        "best_threshold": float(best_threshold),
        "best_threshold_score": float(best_score),
        "best_threshold_stats": best_stats,
        "test_metrics_05": test_metrics_05,
        "test_metrics_best": test_metrics_best,
        "test_by_subtype_best_threshold": summarize_by_subtype(test_records, threshold=best_threshold),
    }

    with open(output_dir / "results.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(results), f, allow_unicode=True, sort_keys=False)

    summary_lines = [
        "# BandConv Pair Baseline",
        "",
        "## Settings",
        f"- representation_version: `{args.representation_version}`",
        f"- profile_length: `{args.profile_length}`",
        f"- band_width: `{args.band_width}`",
        f"- epochs: `{args.epochs}`",
        f"- batch_size: `{args.batch_size}`",
        f"- weighted_sampler: `{args.weighted_sampler}`",
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
        f"- test_auprc: `{test_metrics_best['auprc']:.4f}`",
        f"- test_auroc: `{test_metrics_best['auroc']:.4f}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved BandConv baseline results to {output_dir / 'results.yaml'}")
    print(
        "Final test @ best threshold:",
        {
            "f1": round(float(test_metrics_best["f1"]), 4),
            "precision_abnormal": round(float(test_metrics_best["precision_abnormal"]), 4),
            "recall_abnormal": round(float(test_metrics_best["recall_abnormal"]), 4),
            "balanced_acc": round(float(test_metrics_best["balanced_acc"]), 4),
            "auprc": round(float(test_metrics_best["auprc"]), 4),
            "auroc": round(float(test_metrics_best["auroc"]), 4),
        },
    )


if __name__ == "__main__":
    main()
