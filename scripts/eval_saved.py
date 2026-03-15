import os
import argparse
import torch

from src.main import load_config, set_seed, evaluate
from src.models.build_model import build_model
from src.losses.loss_factory import build_loss
from src.datasets.chromosome_pair_dataset import ChromosomePairDataset
from src.transforms import build_val_transform
from torch.utils.data import DataLoader


def main(config_path, ckpt_path=None):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)

    # determine checkpoint path
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg["output"]["save_dir"], cfg["experiment_name"], "best_model.pth")

    print("Using checkpoint:", ckpt_path)

    # build dataset / loader
    if use_pair_input:
        test_dataset = ChromosomePairDataset(
            cfg["data"]["test_csv"],
            transform=build_val_transform(cfg["data"]["image_size"]),
            use_chromosome_id=use_chromosome_id,
        )
    else:
        raise RuntimeError("This eval script expects pair input datasets for this experiment.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
    )

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"].get("num_classes", 2),
        pretrained=cfg["model"].get("pretrained", True),
        use_chromosome_id=use_chromosome_id,
        num_chromosome_types=None,
        chr_embed_dim=cfg["model"].get("chr_embed_dim", 16),
        use_pair_input=use_pair_input,
        pair_model_type=cfg["model"].get("pair_model_type", "siamese"),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded model state dict.")

    criterion = build_loss(cfg["loss"], device)

    metrics, y_true, y_prob = evaluate(
        model, test_loader, criterion, device,
        threshold=0.5, use_chromosome_id=use_chromosome_id, use_pair_input=use_pair_input
    )

    print("\nTest metrics @0.5:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    main(args.config, args.ckpt)
