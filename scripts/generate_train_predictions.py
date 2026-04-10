"""
For experiments where train_predictions.csv was not generated during training
(because anomaly_threshold.enabled was not set), this script loads a saved
checkpoint and runs inference on the train split to produce the missing file.

Usage:
    python scripts/generate_train_predictions.py \
        --config configs/p12_pair_normal_only_v1_correspondence_interval_multi_prototype_metric_k8.yaml
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.losses.loss_factory import build_loss, extract_anomaly_scores
from src.main import (
    _forward_model,
    build_eval_loader_for_csv,
    export_prediction_records,
    load_config,
    set_seed,
)
from src.models.build_model import build_model
from src.utils.chromosome_vocab import build_chr_vocab_from_csv


def run(config_path):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)

    save_dir = os.path.join(cfg["output"]["save_dir"], cfg["experiment_name"])
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    out_path = os.path.join(save_dir, "train_predictions.csv")

    if os.path.exists(out_path):
        print(f"train_predictions.csv already exists at {out_path}, skipping.")
        return

    print(f"Loading checkpoint: {ckpt_path}")

    chr_to_idx = None
    if use_chromosome_id:
        chr_to_idx, _ = build_chr_vocab_from_csv(cfg["data"]["train_csv"])
    print("Chromosome vocab:", chr_to_idx)

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"].get("num_classes", 2),
        pretrained=False,
        use_chromosome_id=use_chromosome_id,
        num_chromosome_types=len(chr_to_idx) if chr_to_idx is not None else None,
        chr_embed_dim=cfg["model"].get("chr_embed_dim", 16),
        use_pair_input=use_pair_input,
        pair_model_type=cfg["model"].get("pair_model_type", "siamese"),
        experiment_mode=cfg.get("experiment_mode", "classifier"),
        num_prototypes=cfg["model"].get("num_prototypes", 4),
        prototype_distance=cfg["model"].get("prototype_distance", "cosine"),
        normalize_prototype_embedding=cfg["model"].get("normalize_prototype_embedding", True),
    ).to(device)

    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    criterion = build_loss(cfg["loss"], device)
    train_loader = build_eval_loader_for_csv(cfg, cfg["data"]["train_csv"], chr_to_idx)

    records = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Inferring train set"):
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
            batch_size = labels.size(0)
            batch_labels = labels.cpu().numpy().tolist()
            batch_scores = scores.detach().cpu().numpy().tolist()

            metadata_keys = [
                "case_id", "pair_key", "chromosome_id",
                "abnormal_subtype_id", "subtype_status",
                "left_filename", "right_filename", "split",
            ]
            normalized_meta = {}
            for key in metadata_keys:
                if key not in batch:
                    continue
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    normalized_meta[key] = value.detach().cpu().tolist()
                elif isinstance(value, np.ndarray):
                    normalized_meta[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    normalized_meta[key] = list(value)
                else:
                    normalized_meta[key] = [value] * batch_size

            for idx in range(batch_size):
                record = {
                    "label": int(batch_labels[idx]),
                    "anomaly_score": float(batch_scores[idx]),
                }
                for key, values in normalized_meta.items():
                    record[key] = values[idx] if idx < len(values) else ""
                records.append(record)

    export_prediction_records(records, out_path, raw_threshold=None)
    print(f"Saved {len(records)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run(args.config)
