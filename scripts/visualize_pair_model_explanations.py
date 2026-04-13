import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets.chromosome_pair_dataset import ChromosomePairDataset
from src.main import _safe_load_state_dict, load_config, set_seed
from src.models.build_model import build_model
from src.transforms import build_val_transform
from src.utils.chromosome_vocab import build_chr_vocab_from_csv


IMAGENET_MEAN = np.asarray([0.485, 0.485, 0.485], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.229, 0.229], dtype=np.float32)


class ForwardGradRecorder:
    def __init__(self, module):
        self.records = []
        self.handle = module.register_forward_hook(self._forward_hook)

    def clear(self):
        self.records.clear()

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _forward_hook(self, module, inputs, output):
        record = {
            "activation": output.detach(),
            "grad": None,
        }
        self.records.append(record)
        index = len(self.records) - 1

        if output.requires_grad:
            output.register_hook(lambda grad, idx=index: self._save_grad(idx, grad))

    def _save_grad(self, index, grad):
        self.records[index]["grad"] = grad.detach()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--pred_column",
        default="auto",
        help="Prediction column in predictions_csv. Use auto to prefer pred_label_global_valbest > pred_label_chr_conditioned > pred_label_casewise > pred_label_raw.",
    )
    parser.add_argument(
        "--score_column",
        default="auto",
        help="Score column in predictions_csv. Use auto to prefer calibrated_score > casewise_score > anomaly_score.",
    )
    parser.add_argument(
        "--groups",
        default="tp,fp,fn,tn",
        help="Comma-separated confusion groups to visualize. Supported: tp,fp,fn,tn.",
    )
    parser.add_argument("--num_per_group", type=int, default=4)
    parser.add_argument(
        "--target_mode",
        default="anomaly_score",
        choices=["anomaly_score", "pair_distance", "reverse_gain", "direct_mismatch"],
        help="Backprop target for attribution.",
    )
    parser.add_argument(
        "--cam_mode",
        default="band",
        choices=["band", "gradcam2d", "both"],
        help="band: width-collapsed saliency masked to chromosome foreground; gradcam2d: classic 2D Grad-CAM; both: save both views.",
    )
    parser.add_argument(
        "--sort_mode",
        default="confidence",
        choices=["confidence", "score", "random"],
        help="How to rank examples within each confusion group.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pick_prediction_column(df, name):
    if name != "auto":
        if name not in df.columns:
            raise ValueError(f"Prediction column not found: {name}")
        return name

    candidates = [
        "pred_label_global_valbest",
        "pred_label_chr_conditioned",
        "pred_label_casewise",
        "pred_label_raw",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No supported prediction column found. Available columns: {list(df.columns)}")


def pick_score_column(df, name):
    if name != "auto":
        if name not in df.columns:
            raise ValueError(f"Score column not found: {name}")
        return name

    candidates = [
        "calibrated_score",
        "casewise_score",
        "anomaly_score",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No supported score column found. Available columns: {list(df.columns)}")


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


def find_target_module(model):
    base_model = getattr(model, "base_model", model)
    if hasattr(base_model, "feature_proj"):
        return base_model.feature_proj
    if hasattr(base_model, "encoder"):
        return base_model.encoder
    raise ValueError("Could not find a usable target module for Grad-CAM.")


def load_prediction_dataframe(path, pred_column, score_column):
    df = pd.read_csv(path).copy()
    required_columns = {
        "label",
        "left_path",
        "right_path",
        "chromosome_id",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["label"] = df["label"].astype(int)
    df[pred_column] = df[pred_column].astype(int)
    df[score_column] = df[score_column].astype(float)
    df["chromosome_id"] = df["chromosome_id"].astype(str)
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].fillna("").astype(str)
    if "abnormal_subtype_id" in df.columns:
        df["abnormal_subtype_id"] = df["abnormal_subtype_id"].fillna("").astype(str)
    if "subtype_status" in df.columns:
        df["subtype_status"] = df["subtype_status"].fillna("").astype(str)
    return df


def assign_confusion_group(df, pred_column):
    conditions = {
        "tp": (df["label"] == 1) & (df[pred_column] == 1),
        "fp": (df["label"] == 0) & (df[pred_column] == 1),
        "fn": (df["label"] == 1) & (df[pred_column] == 0),
        "tn": (df["label"] == 0) & (df[pred_column] == 0),
    }
    group = np.full(len(df), "", dtype=object)
    for name, mask in conditions.items():
        group[mask.to_numpy()] = name
    out = df.copy()
    out["confusion_group"] = group
    return out


def rank_group_rows(group_df, score_column, sort_mode, seed):
    if group_df.empty:
        return group_df

    if sort_mode == "random":
        return group_df.sample(frac=1.0, random_state=seed)

    ascending = bool(group_df["confusion_group"].iloc[0] in {"tn", "fn"})
    ranked = group_df.copy()
    ranked["abs_score"] = ranked[score_column].abs()

    if sort_mode == "score":
        return ranked.sort_values(score_column, ascending=ascending)

    return ranked.sort_values("abs_score", ascending=False)


def select_examples(df, groups, num_per_group, score_column, sort_mode, seed):
    selected_rows = []
    for group_name in groups:
        group_df = df[df["confusion_group"] == group_name].copy()
        group_df = rank_group_rows(group_df, score_column, sort_mode, seed)
        if not group_df.empty:
            selected_rows.append(group_df.head(num_per_group))

    if not selected_rows:
        raise ValueError("No rows selected for visualization. Check groups/prediction columns.")

    return pd.concat(selected_rows, axis=0).copy()


def tensor_to_rgb_image(tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return image


def normalize_map(array):
    array = np.asarray(array, dtype=np.float32)
    array = array - float(array.min())
    max_value = float(array.max())
    if max_value > 1e-12:
        array = array / max_value
    return array


def estimate_foreground_mask(image_rgb):
    h, w, _ = image_rgb.shape
    patch_h = max(4, h // 12)
    patch_w = max(4, w // 12)
    corner_patches = [
        image_rgb[:patch_h, :patch_w],
        image_rgb[:patch_h, -patch_w:],
        image_rgb[-patch_h:, :patch_w],
        image_rgb[-patch_h:, -patch_w:],
    ]
    bg_pixels = np.concatenate([patch.reshape(-1, 3) for patch in corner_patches], axis=0)
    bg_mean = bg_pixels.mean(axis=0, keepdims=True)
    bg_dist = np.linalg.norm(image_rgb - bg_mean, axis=2)
    bg_corner_dist = np.linalg.norm(bg_pixels - bg_mean, axis=1)
    threshold = max(float(np.quantile(bg_corner_dist, 0.995)) + 0.02, 0.04)
    mask = bg_dist > threshold

    mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    mask_tensor = F.avg_pool2d(mask_tensor, kernel_size=7, stride=1, padding=3)
    mask_tensor = (mask_tensor > 0.08).float()
    mask_tensor = F.avg_pool2d(mask_tensor, kernel_size=9, stride=1, padding=4)
    mask_tensor = (mask_tensor > 0.04).float()

    return mask_tensor[0, 0].numpy().astype(np.float32)


def build_overlay(image_rgb, heatmap, alpha=0.42):
    cmap = cm.get_cmap("jet")
    heat_rgb = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    overlay = (1.0 - alpha) * image_rgb + alpha * heat_rgb
    return np.clip(overlay, 0.0, 1.0)


def build_masked_band_overlay(image_rgb, band_profile, foreground_mask, alpha=0.42):
    h, w, _ = image_rgb.shape
    band_profile = np.asarray(band_profile, dtype=np.float32)
    band_profile = normalize_map(band_profile)
    band_map = np.repeat(band_profile[:, None], w, axis=1)
    band_map = band_map * foreground_mask
    return build_overlay(image_rgb, band_map, alpha=alpha), band_map


def cosine_similarity_1d(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def pearson_correlation_1d(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 1e-12 or b_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_profile_alignment_stats(left_profile, right_profile):
    left_profile = normalize_map(left_profile)
    right_profile = normalize_map(right_profile)
    right_profile_rev = right_profile[::-1].copy()

    stats = {
        "left_profile": left_profile,
        "right_profile": right_profile,
        "right_profile_reversed": right_profile_rev,
        "pearson_direct": pearson_correlation_1d(left_profile, right_profile),
        "pearson_reverse": pearson_correlation_1d(left_profile, right_profile_rev),
        "cosine_direct": cosine_similarity_1d(left_profile, right_profile),
        "cosine_reverse": cosine_similarity_1d(left_profile, right_profile_rev),
    }
    stats["reverse_minus_direct_pearson"] = (
        stats["pearson_reverse"] - stats["pearson_direct"]
    )
    stats["reverse_minus_direct_cosine"] = (
        stats["cosine_reverse"] - stats["cosine_direct"]
    )
    return stats


def compute_cam_from_record(record, out_hw):
    activation = record["activation"]
    grad = record["grad"]
    if grad is None:
        raise ValueError("Gradient was not captured for Grad-CAM target module.")

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    return normalize_map(cam)


def compute_band_profile_from_record(record, out_h):
    activation = record["activation"]
    grad = record["grad"]
    if grad is None:
        raise ValueError("Gradient was not captured for band saliency target module.")

    weighted = F.relu((grad * activation).sum(dim=1, keepdim=False))
    profile = weighted.mean(dim=2)
    profile = F.interpolate(profile.unsqueeze(1), size=out_h, mode="linear", align_corners=False)
    profile = profile[0, 0].detach().cpu().numpy()
    return normalize_map(profile)


def to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def build_metadata_text(row, score_column, pred_column, model_output):
    profile_stats = row.get("_profile_stats", {})
    lines = [
        f"confusion_group: {row.get('confusion_group', '')}",
        f"label: {int(row['label'])}",
        f"pred({pred_column}): {int(row[pred_column])}",
        f"score({score_column}): {float(row[score_column]):.6f}",
        f"chromosome_id: {row.get('chromosome_id', '')}",
        f"case_id: {row.get('case_id', '')}",
        f"pair_key: {row.get('pair_key', '')}",
        f"subtype_status: {row.get('subtype_status', '')}",
        f"abnormal_subtype_id: {row.get('abnormal_subtype_id', '')}",
        f"raw_anomaly_score: {float(row['anomaly_score']):.6f}" if "anomaly_score" in row else "",
        f"nearest_prototype_idx: {int(to_python_scalar(model_output['nearest_prototype_idx'][0]))}"
        if "nearest_prototype_idx" in model_output
        else "",
        f"reverse_gain: {float(to_python_scalar(model_output['reverse_gain'][0])):.6f}"
        if "reverse_gain" in model_output
        else "",
        f"direct_diag_similarity: {float(to_python_scalar(model_output['direct_diag_similarity'][0])):.6f}"
        if "direct_diag_similarity" in model_output
        else "",
        f"reverse_diag_similarity: {float(to_python_scalar(model_output['reverse_diag_similarity'][0])):.6f}"
        if "reverse_diag_similarity" in model_output
        else "",
        f"profile_pearson_direct: {float(profile_stats.get('pearson_direct', 0.0)):.4f}",
        f"profile_pearson_reverse: {float(profile_stats.get('pearson_reverse', 0.0)):.4f}",
        f"profile_cosine_direct: {float(profile_stats.get('cosine_direct', 0.0)):.4f}",
        f"profile_cosine_reverse: {float(profile_stats.get('cosine_reverse', 0.0)):.4f}",
        f"reverse_minus_direct_pearson: {float(profile_stats.get('reverse_minus_direct_pearson', 0.0)):.4f}",
        f"reverse_minus_direct_cosine: {float(profile_stats.get('reverse_minus_direct_cosine', 0.0)):.4f}",
        f"left_file: {row.get('left_filename', os.path.basename(str(row.get('left_path', ''))))}",
        f"right_file: {row.get('right_filename', os.path.basename(str(row.get('right_path', ''))))}",
    ]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def save_explanation_panel(
    output_path,
    left_image,
    right_image,
    left_band_overlay,
    right_band_overlay,
    left_foreground_mask,
    right_foreground_mask,
    interval_attention,
    direct_corr,
    reverse_corr,
    corr_delta,
    metadata_text,
    title,
    profile_stats,
    left_gradcam2d=None,
    right_gradcam2d=None,
):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle(title, fontsize=13)

    axes[0, 0].imshow(left_image)
    axes[0, 0].set_title("Left chromosome")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(left_band_overlay)
    axes[0, 1].set_title("Left band saliency")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(interval_attention, cmap="magma", aspect="auto")
    axes[0, 2].set_title("Interval attention")
    axes[0, 2].set_xlabel("Right token index")
    axes[0, 2].set_ylabel("Left token index")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[0, 3].imshow(direct_corr, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    axes[0, 3].set_title("Direct correlation")
    axes[0, 3].set_xlabel("Right token index")
    axes[0, 3].set_ylabel("Left token index")

    axes[0, 4].imshow(reverse_corr, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    axes[0, 4].set_title("Reverse correlation")
    axes[0, 4].set_xlabel("Right token index")
    axes[0, 4].set_ylabel("Left token index")

    axes[1, 0].imshow(right_image)
    axes[1, 0].set_title("Right chromosome")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(right_band_overlay)
    axes[1, 1].set_title("Right band saliency")
    axes[1, 1].axis("off")

    y_axis = np.arange(len(profile_stats["left_profile"]))
    axes[1, 2].plot(profile_stats["left_profile"], y_axis, label="left", linewidth=2.0)
    axes[1, 2].plot(profile_stats["right_profile"], y_axis, label="right", linewidth=1.6)
    axes[1, 2].plot(profile_stats["right_profile_reversed"], y_axis, label="flip(right)", linewidth=1.6)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_title("Band profile alignment")
    axes[1, 2].set_xlabel("Normalized saliency")
    axes[1, 2].set_ylabel("Long-axis position")
    axes[1, 2].legend(loc="lower right", fontsize=8)

    axes[1, 3].imshow(corr_delta, cmap="bwr", aspect="auto", vmin=-1.0, vmax=1.0)
    axes[1, 3].set_title("Corr delta (reverse - direct)")
    axes[1, 3].set_xlabel("Right token index")
    axes[1, 3].set_ylabel("Left token index")

    axes[1, 4].axis("off")
    axes[1, 4].text(
        0.0,
        1.0,
        metadata_text,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    if left_gradcam2d is not None and right_gradcam2d is not None:
        extra_path = Path(output_path).with_name(Path(output_path).stem + "__gradcam2d" + Path(output_path).suffix)
        extra_fig, extra_axes = plt.subplots(1, 2, figsize=(10, 4))
        extra_axes[0].imshow(left_gradcam2d)
        extra_axes[0].set_title("Left classic Grad-CAM")
        extra_axes[0].axis("off")
        extra_axes[1].imshow(right_gradcam2d)
        extra_axes[1].set_title("Right classic Grad-CAM")
        extra_axes[1].axis("off")
        plt.tight_layout()
        extra_fig.savefig(extra_path, dpi=220, bbox_inches="tight")
        plt.close(extra_fig)

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_attribution_target(model_output, target_mode):
    if target_mode == "anomaly_score":
        return model_output["anomaly_score"][0]
    if target_mode == "pair_distance":
        return model_output["pair_distance"][0]
    if target_mode == "reverse_gain":
        return model_output["reverse_gain"][0]
    if target_mode == "direct_mismatch":
        return 1.0 - model_output["direct_diag_similarity"][0]
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def visualize_one_sample(
    model,
    recorder,
    sample,
    row,
    device,
    pred_column,
    score_column,
    save_path,
    args,
):
    recorder.clear()
    model.zero_grad(set_to_none=True)

    left_image = sample["left_image"].unsqueeze(0).to(device)
    right_image = sample["right_image"].unsqueeze(0).to(device)
    chr_idx = torch.tensor([int(sample["chr_idx"])], device=device) if "chr_idx" in sample else None

    with torch.enable_grad():
        if chr_idx is None:
            model_output = model(left_image, right_image)
        else:
            model_output = model(left_image, right_image, chr_idx)

        target = select_attribution_target(model_output, args.target_mode)
        target.backward()

    if len(recorder.records) < 2:
        raise ValueError(
            f"Expected at least two feature_proj activations for pair input, got {len(recorder.records)}"
        )

    left_rgb = tensor_to_rgb_image(sample["left_image"])
    right_rgb = tensor_to_rgb_image(sample["right_image"])
    out_hw = left_rgb.shape[:2]
    left_mask = estimate_foreground_mask(left_rgb)
    right_mask = estimate_foreground_mask(right_rgb)

    left_gradcam2d = None
    right_gradcam2d = None
    if args.cam_mode in {"gradcam2d", "both"}:
        left_cam_2d = compute_cam_from_record(recorder.records[0], out_hw)
        right_cam_2d = compute_cam_from_record(recorder.records[1], out_hw)
        left_gradcam2d = build_overlay(left_rgb, left_cam_2d * left_mask)
        right_gradcam2d = build_overlay(right_rgb, right_cam_2d * right_mask)

    left_band_profile = compute_band_profile_from_record(recorder.records[0], out_hw[0])
    right_band_profile = compute_band_profile_from_record(recorder.records[1], out_hw[0])
    profile_stats = compute_profile_alignment_stats(left_band_profile, right_band_profile)
    left_band_overlay, _ = build_masked_band_overlay(left_rgb, left_band_profile, left_mask)
    right_band_overlay, _ = build_masked_band_overlay(right_rgb, right_band_profile, right_mask)

    interval_attention = normalize_map(
        model_output.get("interval_attention", torch.zeros(1, 1, 8, 8, device=device))[0, 0].detach().cpu().numpy()
    )
    direct_corr = normalize_map(
        model_output.get("direct_corr", torch.zeros(1, 8, 8, device=device))[0].detach().cpu().numpy()
    )
    reverse_corr = normalize_map(
        model_output.get("reverse_corr", torch.zeros(1, 8, 8, device=device))[0].detach().cpu().numpy()
    )
    corr_delta = model_output.get("corr_delta", torch.zeros(1, 8, 8, device=device))[0].detach().cpu().numpy()
    max_abs = max(float(np.abs(corr_delta).max()), 1e-6)
    corr_delta = np.clip(corr_delta / max_abs, -1.0, 1.0)

    title = (
        f"{row.get('confusion_group', '')} | chr {row.get('chromosome_id', '')} | "
        f"label={int(row['label'])} pred={int(row[pred_column])}"
    )
    row = row.copy()
    row["_profile_stats"] = profile_stats
    metadata_text = build_metadata_text(row, score_column, pred_column, model_output)
    save_explanation_panel(
        output_path=save_path,
        left_image=left_rgb,
        right_image=right_rgb,
        left_band_overlay=left_band_overlay,
        right_band_overlay=right_band_overlay,
        left_foreground_mask=left_mask,
        right_foreground_mask=right_mask,
        interval_attention=interval_attention,
        direct_corr=direct_corr,
        reverse_corr=reverse_corr,
        corr_delta=corr_delta,
        metadata_text=metadata_text,
        title=title,
        profile_stats=profile_stats,
        left_gradcam2d=left_gradcam2d,
        right_gradcam2d=right_gradcam2d,
    )
    return profile_stats


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    predictions_df = pd.read_csv(args.predictions_csv)
    pred_column = pick_prediction_column(predictions_df, args.pred_column)
    score_column = pick_score_column(predictions_df, args.score_column)
    predictions_df = load_prediction_dataframe(args.predictions_csv, pred_column, score_column)
    predictions_df = assign_confusion_group(predictions_df, pred_column)

    groups = [item.strip().lower() for item in str(args.groups).split(",") if item.strip()]
    supported_groups = {"tp", "fp", "fn", "tn"}
    unknown_groups = [group for group in groups if group not in supported_groups]
    if unknown_groups:
        raise ValueError(f"Unsupported groups: {unknown_groups}")

    selected_df = select_examples(
        predictions_df,
        groups=groups,
        num_per_group=args.num_per_group,
        score_column=score_column,
        sort_mode=args.sort_mode,
        seed=args.seed,
    )

    use_chromosome_id = cfg["model"].get("use_chromosome_id", False)
    use_pair_input = cfg["model"].get("use_pair_input", False)
    if not use_pair_input:
        raise ValueError("This visualization script currently supports pair-input models only.")

    chr_to_idx = None
    if use_chromosome_id:
        chr_to_idx, _ = build_chr_vocab_from_csv(cfg["data"]["train_csv"])

    dataset = ChromosomePairDataset(
        csv_path=args.predictions_csv,
        transform=build_val_transform(cfg["data"]["image_size"]),
        chr_to_idx=chr_to_idx,
        use_chromosome_id=use_chromosome_id,
    )

    selected_indices = selected_df.index.to_list()
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, chr_to_idx, device)
    _safe_load_state_dict(model, args.ckpt, device)
    model.eval()

    target_module = find_target_module(model)
    recorder = ForwardGradRecorder(target_module)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    try:
        for batch_index, sample in enumerate(loader):
            original_index = selected_indices[batch_index]
            row = selected_df.loc[original_index]

            simple_sample = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    simple_sample[key] = value[0]
                elif isinstance(value, list):
                    simple_sample[key] = value[0]
                else:
                    simple_sample[key] = value

            case_id = str(row.get("case_id", ""))
            pair_key = str(row.get("pair_key", ""))
            group = str(row.get("confusion_group", ""))
            file_stem = f"{batch_index:02d}_{group}_case-{case_id}_pair-{pair_key}".replace("/", "_").replace("\\", "_")
            save_path = save_dir / f"{file_stem}.png"

            profile_stats = visualize_one_sample(
                model=model,
                recorder=recorder,
                sample=simple_sample,
                row=row,
                device=device,
                pred_column=pred_column,
                score_column=score_column,
                save_path=save_path,
                args=args,
            )

            manifest_rows.append(
                {
                    "image_path": str(save_path),
                    "row_index": int(original_index),
                    "confusion_group": group,
                    "label": int(row["label"]),
                    "pred": int(row[pred_column]),
                    "score": float(row[score_column]),
                    "case_id": case_id,
                    "pair_key": pair_key,
                    "chromosome_id": str(row.get("chromosome_id", "")),
                    "abnormal_subtype_id": str(row.get("abnormal_subtype_id", "")),
                    "subtype_status": str(row.get("subtype_status", "")),
                    "profile_pearson_direct": float(profile_stats["pearson_direct"]),
                    "profile_pearson_reverse": float(profile_stats["pearson_reverse"]),
                    "profile_cosine_direct": float(profile_stats["cosine_direct"]),
                    "profile_cosine_reverse": float(profile_stats["cosine_reverse"]),
                    "reverse_minus_direct_pearson": float(profile_stats["reverse_minus_direct_pearson"]),
                    "reverse_minus_direct_cosine": float(profile_stats["reverse_minus_direct_cosine"]),
                    "left_path": str(row.get("left_path", "")),
                    "right_path": str(row.get("right_path", "")),
                }
            )
    finally:
        recorder.close()

    pd.DataFrame(manifest_rows).to_csv(save_dir / "manifest.csv", index=False)
    summary = [
        "# Pair Explanation Visualization",
        "",
        f"- config: `{args.config}`",
        f"- ckpt: `{args.ckpt}`",
        f"- predictions_csv: `{args.predictions_csv}`",
        f"- pred_column: `{pred_column}`",
        f"- score_column: `{score_column}`",
        f"- groups: `{','.join(groups)}`",
        f"- num_per_group: `{args.num_per_group}`",
        f"- target_mode: `{args.target_mode}`",
        f"- cam_mode: `{args.cam_mode}`",
        "",
        "Saved files:",
        f"- manifest: `{save_dir / 'manifest.csv'}`",
        f"- example panels: `{save_dir}`",
    ]
    (save_dir / "README.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"Saved explanation panels to {save_dir}")
    print(f"Saved manifest to {save_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
