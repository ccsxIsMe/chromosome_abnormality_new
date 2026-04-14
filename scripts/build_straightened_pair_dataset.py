import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.haar_pair_features import extract_straightened_chromosome_image_from_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_csv_dir", required=True)
    parser.add_argument("--output_image_dir", required=True)
    parser.add_argument("--output_report_dir", required=True)
    parser.add_argument("--output_height", type=int, default=300)
    parser.add_argument(
        "--output_width",
        type=int,
        default=0,
        help="Used only when resize_mode=fixed, or as an optional width cap when resize_mode=height_only.",
    )
    parser.add_argument(
        "--canvas_size",
        type=int,
        default=0,
        help="If > 0, paste the straightened image onto a white square canvas. Default 0 keeps raw aspect ratio output.",
    )
    parser.add_argument("--smooth_kernel_size", type=int, default=5)
    parser.add_argument(
        "--straightening_method",
        default="projection_split_v1",
        choices=["centerline_unfold", "projection_split_v1"],
        help="Straightening strategy. projection_split_v1 follows the bend-point split-and-rotate idea.",
    )
    parser.add_argument("--global_angle_step", type=int, default=5)
    parser.add_argument("--local_angle_step", type=int, default=5)
    parser.add_argument("--seam_trim", type=int, default=3)
    parser.add_argument(
        "--mask_mode",
        default="white_bg_exact",
        choices=["auto", "basic", "cv2", "white_bg_exact"],
        help="Foreground extraction mode. white_bg_exact keeps every non-pure-white pixel as chromosome.",
    )
    parser.add_argument(
        "--white_threshold",
        type=float,
        default=254.5 / 255.0,
        help="Pixels darker than this value are treated as foreground when mask_mode=white_bg_exact.",
    )
    parser.add_argument(
        "--repair_expand_ratio",
        type=float,
        default=0.0,
        help="Extra horizontal expansion applied during mask repair. Keep 0.0 to avoid artificial widening.",
    )
    parser.add_argument(
        "--resize_mode",
        default="height_only",
        choices=["fixed", "height_only"],
        help="height_only preserves the post-straightening aspect ratio and avoids forced widening.",
    )
    parser.add_argument(
        "--save_mask_preview",
        action="store_true",
        help="Also save a foreground-mask preview for each unfolded image.",
    )
    return parser.parse_args()


def safe_rel_image_path(original_path: str) -> Path:
    src = Path(original_path)
    parent_name = src.parent.name if src.parent.name else "root"
    stem = src.stem
    suffix = src.suffix if src.suffix else ".png"
    path_hash = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return Path(parent_name) / f"{stem}__{path_hash}{suffix}"


def save_grayscale_image(image_array: np.ndarray, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(np.asarray(image_array, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(save_path)


def paste_on_square_canvas(image_array: np.ndarray, canvas_size: int, fill_value: float) -> np.ndarray:
    image_array = np.asarray(image_array, dtype=np.float32)
    if int(canvas_size) <= 0:
        return image_array.astype(np.float32)

    h, w = image_array.shape
    if h > int(canvas_size) or w > int(canvas_size):
        raise ValueError(
            f"Image shape {image_array.shape} is larger than canvas_size={canvas_size}. "
            "Increase canvas_size or reduce output_height/output_width."
        )

    canvas = np.full((int(canvas_size), int(canvas_size)), float(fill_value), dtype=np.float32)
    top = (int(canvas_size) - h) // 2
    left = (int(canvas_size) - w) // 2
    canvas[top : top + h, left : left + w] = image_array
    return canvas.astype(np.float32)


def build_unique_path_table(train_df, val_df, test_df):
    all_paths = []
    for df in (train_df, val_df, test_df):
        all_paths.extend(df["left_path"].astype(str).tolist())
        all_paths.extend(df["right_path"].astype(str).tolist())
    unique_paths = sorted(set(all_paths))
    return pd.DataFrame({"source_path": unique_paths})


def process_unique_images(
    unique_path_df,
    output_image_dir: Path,
    output_height,
    output_width,
    canvas_size,
    smooth_kernel_size,
    save_mask_preview,
    straightening_method,
    global_angle_step,
    local_angle_step,
    seam_trim,
    mask_mode,
    white_threshold,
    repair_expand_ratio,
    resize_mode,
):
    rows = []
    path_map = {}

    for _, record in tqdm(unique_path_df.iterrows(), total=len(unique_path_df), desc="StraightenImages"):
        source_path = str(record["source_path"])
        straightened = extract_straightened_chromosome_image_from_path(
            image_path=source_path,
            output_height=output_height,
            output_width=output_width,
            smooth_kernel_size=smooth_kernel_size,
            method=straightening_method,
            global_angle_step=global_angle_step,
            local_angle_step=local_angle_step,
            seam_trim=seam_trim,
            mask_mode=mask_mode,
            white_threshold=white_threshold,
            repair_expand_ratio=repair_expand_ratio,
            resize_mode=resize_mode,
        )
        straightened_image = paste_on_square_canvas(
            straightened.image,
            canvas_size=canvas_size,
            fill_value=1.0,
        )
        straightened_mask = paste_on_square_canvas(
            straightened.mask,
            canvas_size=canvas_size,
            fill_value=0.0,
        )

        rel_path = safe_rel_image_path(source_path)
        save_path = output_image_dir / rel_path
        save_grayscale_image(straightened_image, save_path)
        if save_mask_preview:
            mask_path = save_path.with_name(save_path.stem + "__mask" + save_path.suffix)
            save_grayscale_image(straightened_mask, mask_path)

        path_map[source_path] = str(save_path)
        rows.append(
            {
                "source_path": source_path,
                "straightened_path": str(save_path),
                "major_axis_angle_deg": float(straightened.major_axis_angle_deg),
                "foreground_area": int(straightened.foreground_area),
                "bbox_height": int(straightened.bbox_height),
                "bbox_width": int(straightened.bbox_width),
                "valid_profile_fraction": float(straightened.valid_profile_fraction),
                "output_height": int(output_height),
                "output_width": int(output_width),
                "canvas_size": int(canvas_size),
                "straightening_method": str(straightening_method),
                "mask_mode": str(mask_mode),
                "white_threshold": float(white_threshold),
                "repair_expand_ratio": float(repair_expand_ratio),
                "resize_mode": str(resize_mode),
            }
        )

    return pd.DataFrame(rows), path_map


def rewrite_pair_csv(source_df, path_map):
    df = source_df.copy()
    df["left_path"] = df["left_path"].astype(str).map(path_map)
    df["right_path"] = df["right_path"].astype(str).map(path_map)
    if df["left_path"].isna().any() or df["right_path"].isna().any():
        missing_left = int(df["left_path"].isna().sum())
        missing_right = int(df["right_path"].isna().sum())
        raise ValueError(f"Missing rewritten paths: left={missing_left}, right={missing_right}")
    return df


def summarize_split(df, split_name):
    labels = df["label"].astype(int)
    return {
        "split": split_name,
        "pairs": int(len(df)),
        "normal_pairs": int((labels == 0).sum()),
        "abnormal_pairs": int((labels == 1).sum()),
        "cases": int(df["case_id"].astype(str).nunique()) if "case_id" in df.columns else 0,
        "chromosomes": int(df["chromosome_id"].astype(str).nunique()) if "chromosome_id" in df.columns else 0,
    }


def write_report(report_dir: Path, args, split_rows, unique_image_count):
    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Straightened Pair Dataset",
        "",
        "Definition",
        "- each chromosome image is transformed into a straightened grayscale image using the selected straightening method",
        "- pair CSV structure is preserved; only `left_path` / `right_path` are rewritten",
        "- this is an additive preprocessing protocol and does not overwrite the source protocol",
        "",
        "Settings",
        f"- output_height: `{args.output_height}`",
        f"- output_width: `{args.output_width}`",
        f"- canvas_size: `{args.canvas_size}`",
        f"- smooth_kernel_size: `{args.smooth_kernel_size}`",
        f"- straightening_method: `{args.straightening_method}`",
        f"- mask_mode: `{args.mask_mode}`",
        f"- white_threshold: `{args.white_threshold}`",
        f"- repair_expand_ratio: `{args.repair_expand_ratio}`",
        f"- resize_mode: `{args.resize_mode}`",
        f"- global_angle_step: `{args.global_angle_step}`",
        f"- local_angle_step: `{args.local_angle_step}`",
        f"- seam_trim: `{args.seam_trim}`",
        f"- unique_source_images: `{unique_image_count}`",
        "",
        "Split summary",
    ]
    for row in split_rows:
        report_lines.append(
            f"- {row['split']}: pairs={row['pairs']}, normal={row['normal_pairs']}, abnormal={row['abnormal_pairs']}, cases={row['cases']}, chromosomes={row['chromosomes']}"
        )

    (report_dir / "protocol_notes.md").write_text("\n".join(report_lines), encoding="utf-8")
    pd.DataFrame(split_rows).to_csv(report_dir / "split_summary.csv", index=False)


def main():
    args = parse_args()

    output_csv_dir = Path(args.output_csv_dir)
    output_image_dir = Path(args.output_image_dir)
    output_report_dir = Path(args.output_report_dir)
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    unique_path_df = build_unique_path_table(train_df, val_df, test_df)
    image_manifest_df, path_map = process_unique_images(
        unique_path_df=unique_path_df,
        output_image_dir=output_image_dir,
        output_height=args.output_height,
        output_width=args.output_width,
        canvas_size=args.canvas_size,
        smooth_kernel_size=args.smooth_kernel_size,
        save_mask_preview=args.save_mask_preview,
        straightening_method=args.straightening_method,
        global_angle_step=args.global_angle_step,
        local_angle_step=args.local_angle_step,
        seam_trim=args.seam_trim,
        mask_mode=args.mask_mode,
        white_threshold=args.white_threshold,
        repair_expand_ratio=args.repair_expand_ratio,
        resize_mode=args.resize_mode,
    )

    train_out = rewrite_pair_csv(train_df, path_map)
    val_out = rewrite_pair_csv(val_df, path_map)
    test_out = rewrite_pair_csv(test_df, path_map)

    train_out.to_csv(output_csv_dir / "train.csv", index=False)
    val_out.to_csv(output_csv_dir / "val.csv", index=False)
    test_out.to_csv(output_csv_dir / "test.csv", index=False)
    image_manifest_df.to_csv(output_report_dir / "image_manifest.csv", index=False)

    split_rows = [
        summarize_split(train_out, "train"),
        summarize_split(val_out, "val"),
        summarize_split(test_out, "test"),
    ]
    write_report(
        report_dir=output_report_dir,
        args=args,
        split_rows=split_rows,
        unique_image_count=len(unique_path_df),
    )

    print(f"Saved straightened pair CSVs to {output_csv_dir}")
    print(f"Saved straightened images to {output_image_dir}")
    print(f"Saved report to {output_report_dir}")


if __name__ == "__main__":
    main()
