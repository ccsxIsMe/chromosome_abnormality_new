import argparse
import os

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.chromosome_pair_dataset import ChromosomePairDataset
from src.losses.loss_factory import build_loss, extract_logits
from src.main import _forward_model, load_config, search_best_threshold, set_seed
from src.models.build_model import build_model
from src.transforms import build_val_transform
from src.utils.chromosome_vocab import build_chr_vocab_from_csv
from src.utils.metrics import compute_classification_metrics


@torch.no_grad()
def run_loader(model, loader, device, criterion, use_chromosome_id, use_pair_input):
    model.eval()

    rows = []
    for batch in loader:
        labels = batch["label"].to(device)
        output = _forward_model(
            batch=batch,
            model=model,
            device=device,
            use_chromosome_id=use_chromosome_id,
            use_pair_input=use_pair_input,
            use_style_view=False,
        )
        logits = extract_logits(output)
        _ = criterion(output, labels, batch=batch)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()

        batch_size = labels.size(0)
        for i in range(batch_size):
            rows.append(
                {
                    "label": int(batch["label"][i]),
                    "prob_abnormal": float(probs[i]),
                    "case_id": str(batch["case_id"][i]) if "case_id" in batch else "",
                    "pair_key": str(batch["pair_key"][i]) if "pair_key" in batch else "",
                    "chromosome_id": str(batch["chromosome_id"][i]),
                    "abnormal_subtype_id": str(batch["abnormal_subtype_id"][i]) if "abnormal_subtype_id" in batch else "",
                    "subtype_status": str(batch["subtype_status"][i]) if "subtype_status" in batch else "",
                }
            )
    return pd.DataFrame(rows)


def summarize_binary(df, threshold):
    if df.empty:
        return {
            "count": 0,
        }
    metrics = compute_classification_metrics(
        y_true=df["label"].tolist(),
        y_prob=df["prob_abnormal"].tolist(),
        threshold=threshold,
    )
    metrics["count"] = int(len(df))
    return metrics


def summarize_by_subtype(df, threshold):
    rows = []
    abnormal_df = df[df["label"] == 1].copy()
    if abnormal_df.empty:
        return rows

    for subtype, g in abnormal_df.groupby("abnormal_subtype_id"):
        probs = g["prob_abnormal"].to_numpy()
        pred = (probs >= threshold).astype(int)
        rows.append(
            {
                "abnormal_subtype_id": subtype,
                "chromosome_id": str(g["chromosome_id"].iloc[0]),
                "subtype_status": str(g["subtype_status"].iloc[0]),
                "count": int(len(g)),
                "recall_at_threshold": float(pred.mean()),
                "mean_prob": float(probs.mean()),
                "min_prob": float(probs.min()),
                "max_prob": float(probs.max()),
                "case_ids": ",".join(sorted(g["case_id"].astype(str).unique().tolist())),
            }
        )
    return rows


def build_loader(csv_path, cfg, chr_to_idx):
    dataset = ChromosomePairDataset(
        csv_path=csv_path,
        transform=build_val_transform(cfg["data"]["image_size"]),
        chr_to_idx=chr_to_idx,
        use_chromosome_id=cfg["model"].get("use_chromosome_id", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
    )
    return dataset, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)
    if not use_pair_input:
        raise ValueError("This script is for pair-input experiments only.")

    chr_to_idx = None
    if use_chromosome_id:
        chr_to_idx, _ = build_chr_vocab_from_csv(cfg["data"]["train_csv"])

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        use_chromosome_id=use_chromosome_id,
        num_chromosome_types=len(chr_to_idx) if chr_to_idx is not None else None,
        chr_embed_dim=cfg["model"].get("chr_embed_dim", 16),
        use_pair_input=use_pair_input,
        pair_model_type=cfg["model"].get("pair_model_type", "siamese"),
        use_pair_mixstyle=cfg["model"].get("use_pair_mixstyle", False),
        mixstyle_p=cfg["model"].get("mixstyle_p", 0.5),
        mixstyle_alpha=cfg["model"].get("mixstyle_alpha", 0.1),
        experiment_mode=cfg.get("experiment_mode", "classifier"),
        use_side_head=cfg["model"].get("use_side_head", False),
        num_side_classes=cfg["model"].get("num_side_classes", 2),
    ).to(device)

    criterion = build_loss(
        cfg["loss"],
        device,
        experiment_mode=cfg.get("experiment_mode", "classifier"),
        model=model,
    ).to(device)

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg["output"]["save_dir"], cfg["experiment_name"], "best_model.pth")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    _, val_loader = build_loader(cfg["data"]["val_csv"], cfg, chr_to_idx)
    _, test_loader = build_loader(cfg["data"]["test_csv"], cfg, chr_to_idx)

    val_df = run_loader(model, val_loader, device, criterion, use_chromosome_id, use_pair_input)
    test_df = run_loader(model, test_loader, device, criterion, use_chromosome_id, use_pair_input)

    best_threshold, best_score, best_stats = search_best_threshold(
        val_df["label"].tolist(),
        val_df["prob_abnormal"].tolist(),
        metric="f1",
        higher_score_more_positive=True,
    )

    summary = {
        "config_path": args.config,
        "checkpoint_path": ckpt_path,
        "best_threshold_from_val": float(best_threshold),
        "best_threshold_score_from_val": float(best_score),
        "best_threshold_stats_from_val": best_stats,
        "val_overall_05": summarize_binary(val_df, threshold=0.5),
        "val_overall_best": summarize_binary(val_df, threshold=best_threshold),
        "test_overall_05": summarize_binary(test_df, threshold=0.5),
        "test_overall_best": summarize_binary(test_df, threshold=best_threshold),
        "val_seen_05": summarize_binary(val_df[val_df["subtype_status"] == "seen"], threshold=0.5),
        "val_unseen_05": summarize_binary(val_df[val_df["subtype_status"] == "unseen"], threshold=0.5),
        "val_seen_best": summarize_binary(val_df[val_df["subtype_status"] == "seen"], threshold=best_threshold),
        "val_unseen_best": summarize_binary(val_df[val_df["subtype_status"] == "unseen"], threshold=best_threshold),
        "test_seen_05": summarize_binary(test_df[test_df["subtype_status"] == "seen"], threshold=0.5),
        "test_unseen_05": summarize_binary(test_df[test_df["subtype_status"] == "unseen"], threshold=0.5),
        "test_seen_best": summarize_binary(test_df[test_df["subtype_status"] == "seen"], threshold=best_threshold),
        "test_unseen_best": summarize_binary(test_df[test_df["subtype_status"] == "unseen"], threshold=best_threshold),
        "test_by_subtype_best": summarize_by_subtype(test_df, threshold=best_threshold),
    }

    save_dir = os.path.join(cfg["output"]["save_dir"], cfg["experiment_name"])
    os.makedirs(save_dir, exist_ok=True)

    val_df.to_csv(os.path.join(save_dir, "val_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(save_dir, "seen_unseen_breakdown.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)

    print(f"Saved breakdown to {os.path.join(save_dir, 'seen_unseen_breakdown.yaml')}")
    print(f"Saved val predictions to {os.path.join(save_dir, 'val_predictions.csv')}")
    print(f"Saved test predictions to {os.path.join(save_dir, 'test_predictions.csv')}")


if __name__ == "__main__":
    main()
