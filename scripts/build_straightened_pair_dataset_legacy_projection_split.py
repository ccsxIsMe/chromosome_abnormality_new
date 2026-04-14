import argparse
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependent
    raise ImportError("This script requires OpenCV (`cv2`).") from exc

try:
    from scipy.interpolate import interp1d
    from scipy.signal import argrelextrema
except ImportError:  # pragma: no cover - runtime dependent
    interp1d = None
    argrelextrema = None


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
        help="If > 0, paste onto a white square canvas. Default 0 keeps raw aspect ratio.",
    )
    parser.add_argument("--global_angle_step", type=int, default=5)
    parser.add_argument("--local_angle_step", type=int, default=5)
    parser.add_argument("--seam_trim", type=int, default=3)
    parser.add_argument(
        "--white_threshold",
        type=float,
        default=254.5 / 255.0,
        help="Pixels darker than this value are treated as chromosome foreground.",
    )
    parser.add_argument(
        "--resize_mode",
        default="height_only",
        choices=["fixed", "height_only"],
    )
    parser.add_argument("--save_mask_preview", action="store_true")
    return parser.parse_args()


def load_grayscale_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_grayscale_image(image_array: np.ndarray, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(np.asarray(image_array, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(save_path)


def safe_rel_image_path(original_path: str) -> Path:
    src = Path(original_path)
    parent_name = src.parent.name if src.parent.name else "root"
    stem = src.stem
    suffix = src.suffix if src.suffix else ".png"
    path_hash = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return Path(parent_name) / f"{stem}__{path_hash}{suffix}"


def paste_on_square_canvas(image_array: np.ndarray, canvas_size: int, fill_value: float) -> np.ndarray:
    image_array = np.asarray(image_array, dtype=np.float32)
    if int(canvas_size) <= 0:
        return image_array.astype(np.float32)

    h, w = image_array.shape
    if h > int(canvas_size) or w > int(canvas_size):
        raise ValueError(
            f"Image shape {image_array.shape} is larger than canvas_size={canvas_size}. "
            "Increase canvas_size or reduce output size."
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
        "# Legacy Projection-Split Straightened Pair Dataset",
        "",
        "Definition",
        "- this script follows the user-provided bend-point split-and-rotate straightening idea directly",
        "- pair CSV structure is preserved; only `left_path` / `right_path` are rewritten",
        "- this is additive and does not overwrite the source protocol",
        "",
        "Settings",
        f"- output_height: `{args.output_height}`",
        f"- output_width: `{args.output_width}`",
        f"- canvas_size: `{args.canvas_size}`",
        f"- global_angle_step: `{args.global_angle_step}`",
        f"- local_angle_step: `{args.local_angle_step}`",
        f"- seam_trim: `{args.seam_trim}`",
        f"- white_threshold: `{args.white_threshold}`",
        f"- resize_mode: `{args.resize_mode}`",
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


def _binary_mask_uint8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255


def build_white_background_mask(gray_image: np.ndarray, white_threshold: float) -> np.ndarray:
    gray_image = np.asarray(np.clip(gray_image, 0.0, 1.0), dtype=np.float32)
    mask = gray_image < float(white_threshold)
    if not mask.any():
        mask = gray_image < 1.0
    if not mask.any():
        return mask.astype(bool)

    mask_uint8 = _binary_mask_uint8(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_uint8 > 0


def add_border(gray_image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray_image.shape
    border = max(h // 2, w // 2)
    padded_gray = np.pad(gray_image, ((border, border), (border, border)), constant_values=1.0)
    padded_mask = np.pad(mask.astype(np.uint8), ((border, border), (border, border)), constant_values=0)
    return padded_gray.astype(np.float32), padded_mask.astype(bool)


def crop_to_nonzero(gray_image: np.ndarray, mask_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_uint8 > 0)
    if ys.size == 0 or xs.size == 0:
        return gray_image.astype(np.float32), mask_uint8.astype(bool)
    y0 = max(int(ys.min()), 0)
    y1 = min(int(ys.max()) + 1, gray_image.shape[0])
    x0 = max(int(xs.min()), 0)
    x1 = min(int(xs.max()) + 1, gray_image.shape[1])
    return gray_image[y0:y1, x0:x1].astype(np.float32), (mask_uint8[y0:y1, x0:x1] > 0)


def rotate_gray_and_mask(gray_image: np.ndarray, mask_uint8: np.ndarray, degree: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray_image.shape
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(degree), 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated_gray = cv2.warpAffine(
        np.clip(gray_image * 255.0, 0.0, 255.0).astype(np.uint8),
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    rotated_mask = cv2.warpAffine(
        mask_uint8,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_gray.astype(np.float32) / 255.0, rotated_mask.astype(np.uint8)


def smooth_projection(y_projection: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_projection = np.asarray(y_projection, dtype=np.float32)
    src_idx = np.arange(y_projection.size, dtype=np.float32)
    if y_projection.size < 4:
        return src_idx, y_projection

    target_size = max(int(0.5 * y_projection.size), 8)
    dst_idx = np.linspace(0.0, float(y_projection.size - 1), target_size, dtype=np.float32)
    if interp1d is not None and y_projection.size >= 6:
        cubic = interp1d(src_idx, y_projection, kind="cubic")
        smooth_values = cubic(dst_idx).astype(np.float32)
    else:
        smooth_values = np.interp(dst_idx, src_idx, y_projection).astype(np.float32)
    return dst_idx, smooth_values


def find_local_extrema(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 3:
        return np.asarray([], dtype=np.int32)

    if argrelextrema is not None:
        comparator = np.greater if mode == "max" else np.less
        extrema = argrelextrema(values, comparator)[0]
        return extrema.astype(np.int32)

    left = values[1:-1] - values[:-2]
    right = values[1:-1] - values[2:]
    if mode == "max":
        mask = (left >= 0) & (right >= 0) & ((left > 0) | (right > 0))
    elif mode == "min":
        mask = (left <= 0) & (right <= 0) & ((left < 0) | (right < 0))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return np.where(mask)[0] + 1


def calculate_s_score(mask_uint8: np.ndarray) -> Tuple[float, Optional[int]]:
    y_projection = np.sum(mask_uint8 > 0, axis=1).astype(np.float32)
    interp_idx, interp_values = smooth_projection(y_projection)
    sag_indices = find_local_extrema(interp_values, mode="min")
    if sag_indices.size == 0:
        return float("inf"), None

    sag_idx = int(sag_indices[np.argmin(interp_values[sag_indices])])
    sag_value = float(interp_values[sag_idx])

    left_values = interp_values[:sag_idx]
    right_values = interp_values[sag_idx + 1 :]
    left_peaks = find_local_extrema(left_values, mode="max")
    right_peaks = find_local_extrema(right_values, mode="max")
    if left_peaks.size == 0 or right_peaks.size == 0:
        return float("inf"), None

    peak_1 = float(left_values[left_peaks].max())
    peak_2 = float(right_values[right_peaks].max())
    denom = max(peak_1 + peak_2, 1e-6)
    r1 = abs(peak_1 - peak_2) / denom
    r2 = sag_value / denom
    score = 0.5 * r1 + 0.5 * r2
    bend_row = int(round(float(interp_idx[sag_idx])))
    return float(score), bend_row


def find_best_global_rotation(gray_image: np.ndarray, mask: np.ndarray, angle_step: int) -> Tuple[np.ndarray, np.ndarray, float, Optional[int], float]:
    padded_gray, padded_mask = add_border(gray_image, mask)
    mask_uint8 = _binary_mask_uint8(padded_mask)

    best = None
    for degree in range(0, 180, int(angle_step)):
        rotated_gray, rotated_mask = rotate_gray_and_mask(padded_gray, mask_uint8, degree)
        cropped_gray, cropped_mask = crop_to_nonzero(rotated_gray, rotated_mask)
        score, bend_row = calculate_s_score(_binary_mask_uint8(cropped_mask))
        if bend_row is None:
            continue
        if best is None or score < best["score"]:
            best = {
                "gray": cropped_gray,
                "mask": cropped_mask,
                "degree": float(degree),
                "bend_row": int(bend_row),
                "score": float(score),
            }

    if best is None:
        return gray_image.astype(np.float32), mask.astype(bool), 0.0, None, float("inf")
    return best["gray"], best["mask"], best["degree"], best["bend_row"], best["score"]


def find_min_width_rotation(gray_image: np.ndarray, mask: np.ndarray, angle_step: int) -> Tuple[np.ndarray, np.ndarray, float]:
    if gray_image.size == 0:
        return gray_image.astype(np.float32), mask.astype(bool), 0.0

    padded_gray, padded_mask = add_border(gray_image, mask)
    mask_uint8 = _binary_mask_uint8(padded_mask)
    best = None
    for degree in range(-90, 90, int(angle_step)):
        rotated_gray, rotated_mask = rotate_gray_and_mask(padded_gray, mask_uint8, degree)
        cropped_gray, cropped_mask = crop_to_nonzero(rotated_gray, rotated_mask)
        x_projection = np.sum(cropped_mask, axis=0).astype(np.float32)
        nonzero_cols = np.where(x_projection > 0)[0]
        if nonzero_cols.size == 0:
            continue
        width = int(nonzero_cols[-1] - nonzero_cols[0] + 1)
        if best is None or width < best["width"]:
            best = {
                "gray": cropped_gray,
                "mask": cropped_mask,
                "degree": float(degree),
                "width": int(width),
            }

    if best is None:
        return gray_image.astype(np.float32), mask.astype(bool), 0.0
    return best["gray"], best["mask"], best["degree"]


def pad_right_to_width(image: np.ndarray, target_width: int, fill_value: float) -> np.ndarray:
    if image.shape[1] >= target_width:
        return image.astype(np.float32)
    pad = target_width - image.shape[1]
    return np.pad(image, ((0, 0), (0, pad)), constant_values=float(fill_value)).astype(np.float32)


def resize_gray_image(image: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    resized = cv2.resize(
        np.asarray(image, dtype=np.float32),
        (int(out_width), int(out_height)),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def resize_image_and_mask(image: np.ndarray, mask: np.ndarray, output_height: int, output_width: int, resize_mode: str):
    src_h, src_w = image.shape
    target_height = max(int(output_height), 1)
    if resize_mode == "fixed":
        target_width = max(int(output_width), 1)
    elif resize_mode == "height_only":
        scale = float(target_height) / max(float(src_h), 1.0)
        target_width = max(int(round(float(src_w) * scale)), 1)
        if int(output_width) > 0:
            target_width = min(target_width, int(output_width))
    else:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    resized_image = resize_gray_image(image, target_width, target_height)
    resized_mask = resize_gray_image(mask.astype(np.float32), target_width, target_height)
    return resized_image.astype(np.float32), resized_mask.astype(np.float32)


def straighten_legacy_projection_split(
    gray_image: np.ndarray,
    white_threshold: float,
    output_height: int,
    output_width: int,
    resize_mode: str,
    global_angle_step: int,
    local_angle_step: int,
    seam_trim: int,
) -> Dict[str, np.ndarray]:
    mask = build_white_background_mask(gray_image, white_threshold=white_threshold)
    aligned_gray, aligned_mask, degree, bend_row, score = find_best_global_rotation(
        gray_image=gray_image,
        mask=mask,
        angle_step=global_angle_step,
    )

    if bend_row is None or bend_row <= 2 or bend_row >= aligned_gray.shape[0] - 2:
        bend_row = aligned_gray.shape[0] // 2

    upper_gray = aligned_gray[:bend_row, :]
    lower_gray = aligned_gray[bend_row:, :]
    upper_mask = aligned_mask[:bend_row, :]
    lower_mask = aligned_mask[bend_row:, :]

    upper_rot_gray, upper_rot_mask, upper_degree = find_min_width_rotation(
        upper_gray,
        upper_mask,
        angle_step=local_angle_step,
    )
    lower_rot_gray, lower_rot_mask, lower_degree = find_min_width_rotation(
        lower_gray,
        lower_mask,
        angle_step=local_angle_step,
    )

    target_width = max(upper_rot_gray.shape[1], lower_rot_gray.shape[1])
    upper_rot_gray = pad_right_to_width(upper_rot_gray, target_width, fill_value=1.0)
    lower_rot_gray = pad_right_to_width(lower_rot_gray, target_width, fill_value=1.0)
    upper_rot_mask = pad_right_to_width(upper_rot_mask.astype(np.float32), target_width, fill_value=0.0) > 0.5
    lower_rot_mask = pad_right_to_width(lower_rot_mask.astype(np.float32), target_width, fill_value=0.0) > 0.5

    trim = max(int(seam_trim), 0)
    upper_keep = upper_rot_gray[:-trim, :] if trim > 0 and upper_rot_gray.shape[0] > trim else upper_rot_gray
    lower_keep = lower_rot_gray[trim:, :] if trim > 0 and lower_rot_gray.shape[0] > trim else lower_rot_gray
    upper_mask_keep = upper_rot_mask[:-trim, :] if trim > 0 and upper_rot_mask.shape[0] > trim else upper_rot_mask
    lower_mask_keep = lower_rot_mask[trim:, :] if trim > 0 and lower_rot_mask.shape[0] > trim else lower_rot_mask

    sewn_gray = np.concatenate([upper_keep, lower_keep], axis=0).astype(np.float32)
    sewn_mask = np.concatenate([upper_mask_keep, lower_mask_keep], axis=0).astype(np.float32)

    straightened_image, straightened_mask = resize_image_and_mask(
        sewn_gray,
        sewn_mask,
        output_height=output_height,
        output_width=output_width,
        resize_mode=resize_mode,
    )
    straightened_mask = (straightened_mask > 0.10).astype(np.float32)
    straightened_image = np.where(straightened_mask > 0.5, straightened_image, 1.0).astype(np.float32)

    return {
        "image": straightened_image,
        "mask": straightened_mask,
        "global_degree": float(degree),
        "upper_degree": float(upper_degree),
        "lower_degree": float(lower_degree),
        "bend_row": int(bend_row),
        "s_score": float(score),
        "bbox_height": int(sewn_gray.shape[0]),
        "bbox_width": int(sewn_gray.shape[1]),
        "foreground_area": int((sewn_mask > 0.5).sum()),
        "valid_profile_fraction": float(straightened_mask.mean()),
    }


def process_unique_images(unique_path_df, output_image_dir: Path, args):
    rows = []
    path_map = {}

    for _, record in tqdm(unique_path_df.iterrows(), total=len(unique_path_df), desc="LegacyStraighten"):
        source_path = str(record["source_path"])
        gray_image = load_grayscale_image(source_path)
        result = straighten_legacy_projection_split(
            gray_image=gray_image,
            white_threshold=args.white_threshold,
            output_height=args.output_height,
            output_width=args.output_width,
            resize_mode=args.resize_mode,
            global_angle_step=args.global_angle_step,
            local_angle_step=args.local_angle_step,
            seam_trim=args.seam_trim,
        )

        straightened_image = paste_on_square_canvas(result["image"], args.canvas_size, fill_value=1.0)
        straightened_mask = paste_on_square_canvas(result["mask"], args.canvas_size, fill_value=0.0)

        rel_path = safe_rel_image_path(source_path)
        save_path = output_image_dir / rel_path
        save_grayscale_image(straightened_image, save_path)
        if args.save_mask_preview:
            mask_path = save_path.with_name(save_path.stem + "__mask" + save_path.suffix)
            save_grayscale_image(straightened_mask, mask_path)

        path_map[source_path] = str(save_path)
        rows.append(
            {
                "source_path": source_path,
                "straightened_path": str(save_path),
                "foreground_area": int(result["foreground_area"]),
                "bbox_height": int(result["bbox_height"]),
                "bbox_width": int(result["bbox_width"]),
                "valid_profile_fraction": float(result["valid_profile_fraction"]),
                "global_degree": float(result["global_degree"]),
                "upper_degree": float(result["upper_degree"]),
                "lower_degree": float(result["lower_degree"]),
                "bend_row": int(result["bend_row"]),
                "s_score": float(result["s_score"]),
                "output_height": int(args.output_height),
                "output_width": int(args.output_width),
                "canvas_size": int(args.canvas_size),
                "resize_mode": str(args.resize_mode),
                "white_threshold": float(args.white_threshold),
            }
        )

    return pd.DataFrame(rows), path_map


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
    image_manifest_df, path_map = process_unique_images(unique_path_df, output_image_dir, args)

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
    write_report(output_report_dir, args, split_rows, len(unique_path_df))

    print(f"Saved legacy straightened pair CSVs to {output_csv_dir}")
    print(f"Saved legacy straightened images to {output_image_dir}")
    print(f"Saved report to {output_report_dir}")


if __name__ == "__main__":
    main()
