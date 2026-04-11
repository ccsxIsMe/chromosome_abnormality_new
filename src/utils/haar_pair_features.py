from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class ChromosomeBandRepresentation:
    profile: np.ndarray
    width_profile: np.ndarray
    band_image: np.ndarray
    major_axis_angle_deg: float
    foreground_area: int
    bbox_height: int
    bbox_width: int
    valid_profile_fraction: float


def load_grayscale_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5

    values = np.clip(values, 0.0, 1.0)
    hist, bin_edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1e-12)
    mean_total = float((hist * centers).sum() / total)
    mean_fg = (mean_total * total - np.cumsum(hist * centers)) / np.maximum(weight_fg, 1e-12)
    inter_class_var = weight_bg[:-1] * weight_fg[:-1] * (mean_bg[:-1] - mean_fg[:-1]) ** 2
    best_idx = int(np.argmax(inter_class_var))
    return float(centers[best_idx])


def largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(binary_mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_component: List[Tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component: List[Tuple[int, int]] = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))

                y0 = max(cy - 1, 0)
                y1 = min(cy + 2, height)
                x0 = max(cx - 1, 0)
                x1 = min(cx + 2, width)
                for ny in range(y0, y1):
                    for nx in range(x0, x1):
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if len(component) > len(best_component):
                best_component = component

    largest = np.zeros_like(mask, dtype=bool)
    for y, x in best_component:
        largest[y, x] = True
    return largest


def estimate_foreground_mask(gray_image: np.ndarray, min_area_ratio: float = 0.002) -> np.ndarray:
    gray_image = np.asarray(gray_image, dtype=np.float32)
    foreground_score = 1.0 - np.clip(gray_image, 0.0, 1.0)
    threshold = otsu_threshold(foreground_score)
    mask = foreground_score >= threshold

    if mask.mean() < min_area_ratio:
        relaxed_threshold = max(float(np.quantile(foreground_score, 0.85)), threshold * 0.7)
        mask = foreground_score >= relaxed_threshold

    mask = largest_connected_component(mask)
    if mask.sum() == 0:
        mask = foreground_score >= float(np.quantile(foreground_score, 0.90))
        mask = largest_connected_component(mask)
    return mask


def mask_bounding_box(mask: np.ndarray, margin: int = 4) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        height, width = mask.shape
        return 0, height, 0, width

    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, mask.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, mask.shape[1])
    return y0, y1, x0, x1


def estimate_major_axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size < 2:
        return 0.0

    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    coords = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return angle_deg


def rotate_image_and_mask(gray_image: np.ndarray, mask: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    pil_image = Image.fromarray(np.clip(gray_image * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

    rotate_by = 90.0 - angle_deg
    rotated_image = pil_image.rotate(rotate_by, resample=Image.BILINEAR, expand=True, fillcolor=255)
    rotated_mask = pil_mask.rotate(rotate_by, resample=Image.NEAREST, expand=True, fillcolor=0)

    image_np = np.asarray(rotated_image, dtype=np.float32) / 255.0
    mask_np = np.asarray(rotated_mask, dtype=np.uint8) > 0
    return image_np, mask_np


def interpolate_missing_1d(values: np.ndarray, fill_value: float = 1.0) -> Tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float32)
    valid_mask = np.isfinite(values)
    valid_fraction = float(valid_mask.mean()) if values.size > 0 else 0.0

    if values.size == 0:
        return values.astype(np.float32), 0.0

    if valid_mask.sum() == 0:
        return np.full_like(values, fill_value, dtype=np.float32), 0.0

    indices = np.arange(values.size, dtype=np.float32)
    filled = values.copy()
    filled[~valid_mask] = np.interp(
        indices[~valid_mask],
        indices[valid_mask],
        values[valid_mask],
    )
    return filled.astype(np.float32), valid_fraction


def resample_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == target_length:
        return values.astype(np.float32)
    if values.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if values.size == 1:
        return np.full(target_length, float(values[0]), dtype=np.float32)

    src_x = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(dst_x, src_x, values).astype(np.float32)


def extract_band_profile(
    gray_image: np.ndarray,
    mask: np.ndarray,
    profile_length: int = 128,
    band_width: int = 32,
) -> ChromosomeBandRepresentation:
    if mask.sum() == 0:
        mask = np.ones_like(gray_image, dtype=bool)

    y0, y1, x0, x1 = mask_bounding_box(mask)
    cropped_image = gray_image[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]

    angle_deg = estimate_major_axis_angle(cropped_mask)
    rotated_image, rotated_mask = rotate_image_and_mask(cropped_image, cropped_mask, angle_deg)
    if rotated_mask.sum() == 0:
        rotated_mask = np.ones_like(rotated_image, dtype=bool)

    y0, y1, x0, x1 = mask_bounding_box(rotated_mask)
    aligned_image = rotated_image[y0:y1, x0:x1]
    aligned_mask = rotated_mask[y0:y1, x0:x1]
    if aligned_image.size == 0:
        aligned_image = gray_image.copy()
        aligned_mask = np.ones_like(gray_image, dtype=bool)

    row_values = np.full(aligned_image.shape[0], np.nan, dtype=np.float32)
    width_values = np.full(aligned_image.shape[0], np.nan, dtype=np.float32)
    strip_rows: List[np.ndarray] = []
    default_fill = float(np.nanmean(aligned_image)) if aligned_image.size > 0 else 1.0
    if not np.isfinite(default_fill):
        default_fill = 1.0

    for row_idx in range(aligned_image.shape[0]):
        row_mask = aligned_mask[row_idx]
        fg_cols = np.where(row_mask)[0]
        if fg_cols.size < 2:
            strip_rows.append(np.full(band_width, np.nan, dtype=np.float32))
            continue

        left = int(fg_cols.min())
        right = int(fg_cols.max())
        if right <= left:
            strip_rows.append(np.full(band_width, np.nan, dtype=np.float32))
            continue

        fg_values = aligned_image[row_idx, fg_cols]
        row_values[row_idx] = float(np.median(fg_values))
        width_values[row_idx] = float(fg_cols.size)

        row_segment = aligned_image[row_idx, left : right + 1]
        src_x = np.linspace(0.0, 1.0, num=row_segment.size, dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, num=band_width, dtype=np.float32)
        strip_rows.append(np.interp(dst_x, src_x, row_segment).astype(np.float32))

    filled_profile, valid_fraction = interpolate_missing_1d(row_values, fill_value=default_fill)
    filled_width, _ = interpolate_missing_1d(width_values, fill_value=0.0)

    band_rows = np.stack(strip_rows, axis=0) if strip_rows else np.zeros((1, band_width), dtype=np.float32)
    for col_idx in range(band_rows.shape[1]):
        band_rows[:, col_idx], _ = interpolate_missing_1d(
            band_rows[:, col_idx],
            fill_value=default_fill,
        )

    profile = resample_1d(filled_profile, profile_length)
    width_profile = resample_1d(filled_width, profile_length)

    band_image = np.stack(
        [resample_1d(band_rows[:, col_idx], profile_length) for col_idx in range(band_rows.shape[1])],
        axis=1,
    ).astype(np.float32)

    profile = (profile - profile.mean()) / max(float(profile.std()), 1e-6)
    if width_profile.std() > 1e-6:
        width_profile = (width_profile - width_profile.mean()) / max(float(width_profile.std()), 1e-6)

    return ChromosomeBandRepresentation(
        profile=profile.astype(np.float32),
        width_profile=width_profile.astype(np.float32),
        band_image=band_image.astype(np.float32),
        major_axis_angle_deg=float(angle_deg),
        foreground_area=int(aligned_mask.sum()),
        bbox_height=int(aligned_mask.shape[0]),
        bbox_width=int(aligned_mask.shape[1]),
        valid_profile_fraction=float(valid_fraction),
    )


def build_haar_kernel(size: int, weights: Sequence[float]) -> np.ndarray:
    size = int(size)
    weights = [float(w) for w in weights]
    if size < len(weights):
        raise ValueError(f"Kernel size {size} is smaller than number of segments {len(weights)}")

    segments = len(weights)
    base = size // segments
    remainder = size % segments
    kernel = np.zeros(size, dtype=np.float32)

    cursor = 0
    for idx, weight in enumerate(weights):
        seg_len = base + (1 if idx < remainder else 0)
        kernel[cursor : cursor + seg_len] = weight
        cursor += seg_len

    kernel -= kernel.mean()
    norm = float(np.linalg.norm(kernel))
    if norm > 1e-6:
        kernel /= norm
    return kernel


def default_haar_specifications() -> List[Tuple[str, Tuple[float, ...]]]:
    return [
        ("step", (1.0, -1.0)),
        ("reverse_step", (-1.0, 1.0)),
        ("peak", (1.0, -2.0, 1.0)),
        ("valley", (-1.0, 2.0, -1.0)),
        ("edge_triplet", (1.0, 0.0, -1.0)),
    ]


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def summarize_response(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_mean_abs": 0.0,
            f"{prefix}_max_abs": 0.0,
        }

    abs_values = np.abs(values)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_mean_abs": float(abs_values.mean()),
        f"{prefix}_max_abs": float(abs_values.max()),
    }


def extract_profile_summary_features(profile: np.ndarray, prefix: str) -> Dict[str, float]:
    profile = np.asarray(profile, dtype=np.float32)
    diffs = np.diff(profile)
    return {
        f"{prefix}_mean": float(profile.mean()),
        f"{prefix}_std": float(profile.std()),
        f"{prefix}_min": float(profile.min()),
        f"{prefix}_max": float(profile.max()),
        f"{prefix}_range": float(profile.max() - profile.min()),
        f"{prefix}_diff_mean_abs": float(np.abs(diffs).mean()) if diffs.size > 0 else 0.0,
        f"{prefix}_diff_max_abs": float(np.abs(diffs).max()) if diffs.size > 0 else 0.0,
    }


def extract_segment_difference_features(
    left_profile: np.ndarray,
    right_profile: np.ndarray,
    num_segments: Iterable[int] = (4, 8),
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    left_profile = np.asarray(left_profile, dtype=np.float32)
    right_profile = np.asarray(right_profile, dtype=np.float32)
    reversed_right = right_profile[::-1]

    for num_segments in num_segments:
        edges = np.linspace(0, left_profile.size, num_segments + 1, dtype=np.int32)
        for seg_idx in range(num_segments):
            start = int(edges[seg_idx])
            end = int(edges[seg_idx + 1])
            if end <= start:
                continue
            left_seg = left_profile[start:end]
            right_seg = right_profile[start:end]
            reverse_seg = reversed_right[start:end]
            features[f"seg{num_segments}_{seg_idx}_direct_l1"] = float(np.mean(np.abs(left_seg - right_seg)))
            features[f"seg{num_segments}_{seg_idx}_reverse_l1"] = float(np.mean(np.abs(left_seg - reverse_seg)))
            features[f"seg{num_segments}_{seg_idx}_reverse_gain"] = (
                features[f"seg{num_segments}_{seg_idx}_direct_l1"]
                - features[f"seg{num_segments}_{seg_idx}_reverse_l1"]
            )
    return features


def extract_pair_haar_features(
    left_repr: ChromosomeBandRepresentation,
    right_repr: ChromosomeBandRepresentation,
    kernel_sizes: Sequence[int] = (4, 8, 16, 32, 64),
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    left_profile = np.asarray(left_repr.profile, dtype=np.float32)
    right_profile = np.asarray(right_repr.profile, dtype=np.float32)
    right_profile_reversed = right_profile[::-1]

    features.update(extract_profile_summary_features(left_profile, "left_profile"))
    features.update(extract_profile_summary_features(right_profile, "right_profile"))
    features.update(extract_profile_summary_features(left_repr.width_profile, "left_width"))
    features.update(extract_profile_summary_features(right_repr.width_profile, "right_width"))

    features["profile_direct_corr"] = safe_corrcoef(left_profile, right_profile)
    features["profile_reverse_corr"] = safe_corrcoef(left_profile, right_profile_reversed)
    features["profile_reverse_gain"] = features["profile_reverse_corr"] - features["profile_direct_corr"]
    features["profile_direct_l1"] = float(np.mean(np.abs(left_profile - right_profile)))
    features["profile_reverse_l1"] = float(np.mean(np.abs(left_profile - right_profile_reversed)))
    features["profile_reverse_l1_gain"] = features["profile_direct_l1"] - features["profile_reverse_l1"]

    features["width_direct_l1"] = float(np.mean(np.abs(left_repr.width_profile - right_repr.width_profile)))
    features["width_reverse_l1"] = float(
        np.mean(np.abs(left_repr.width_profile - right_repr.width_profile[::-1]))
    )
    features["width_reverse_l1_gain"] = features["width_direct_l1"] - features["width_reverse_l1"]

    features["left_major_axis_angle_deg"] = float(left_repr.major_axis_angle_deg)
    features["right_major_axis_angle_deg"] = float(right_repr.major_axis_angle_deg)
    features["abs_major_axis_angle_diff"] = abs(
        float(left_repr.major_axis_angle_deg) - float(right_repr.major_axis_angle_deg)
    )
    features["left_foreground_area"] = float(left_repr.foreground_area)
    features["right_foreground_area"] = float(right_repr.foreground_area)
    features["foreground_area_ratio"] = float(
        left_repr.foreground_area / max(float(right_repr.foreground_area), 1.0)
    )
    features["abs_foreground_area_diff"] = abs(
        float(left_repr.foreground_area) - float(right_repr.foreground_area)
    )
    features["left_valid_profile_fraction"] = float(left_repr.valid_profile_fraction)
    features["right_valid_profile_fraction"] = float(right_repr.valid_profile_fraction)

    features.update(extract_segment_difference_features(left_profile, right_profile))

    for size in kernel_sizes:
        for kernel_name, kernel_weights in default_haar_specifications():
            if size > left_profile.size:
                continue

            kernel = build_haar_kernel(size=size, weights=kernel_weights)
            left_response = np.convolve(left_profile, kernel, mode="valid")
            right_response = np.convolve(right_profile, kernel, mode="valid")
            reverse_response = np.convolve(right_profile_reversed, kernel, mode="valid")

            prefix = f"haar_{kernel_name}_k{size}"
            features.update(summarize_response(left_response, f"{prefix}_left"))
            features.update(summarize_response(right_response, f"{prefix}_right"))
            features.update(summarize_response(left_response - right_response, f"{prefix}_direct_diff"))
            features.update(summarize_response(left_response - reverse_response, f"{prefix}_reverse_diff"))
            features[f"{prefix}_direct_corr"] = safe_corrcoef(left_response, right_response)
            features[f"{prefix}_reverse_corr"] = safe_corrcoef(left_response, reverse_response)
            features[f"{prefix}_reverse_gain"] = (
                features[f"{prefix}_reverse_corr"] - features[f"{prefix}_direct_corr"]
            )

    return features


def extract_pair_features_from_paths(
    left_path: str,
    right_path: str,
    profile_length: int = 128,
    band_width: int = 32,
    kernel_sizes: Sequence[int] = (4, 8, 16, 32, 64),
) -> Dict[str, float]:
    left_gray = load_grayscale_image(left_path)
    right_gray = load_grayscale_image(right_path)

    left_mask = estimate_foreground_mask(left_gray)
    right_mask = estimate_foreground_mask(right_gray)

    left_repr = extract_band_profile(
        gray_image=left_gray,
        mask=left_mask,
        profile_length=profile_length,
        band_width=band_width,
    )
    right_repr = extract_band_profile(
        gray_image=right_gray,
        mask=right_mask,
        profile_length=profile_length,
        band_width=band_width,
    )

    return extract_pair_haar_features(
        left_repr=left_repr,
        right_repr=right_repr,
        kernel_sizes=kernel_sizes,
    )
