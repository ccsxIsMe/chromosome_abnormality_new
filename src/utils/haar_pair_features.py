from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local runtime
    cv2 = None


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


@dataclass
class StraightenedChromosomeImage:
    image: np.ndarray
    mask: np.ndarray
    major_axis_angle_deg: float
    foreground_area: int
    bbox_height: int
    bbox_width: int
    valid_profile_fraction: float


def moving_average_1d(values: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or kernel_size <= 1:
        return values.astype(np.float32)
    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


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


def remove_small_connected_components(binary_mask: np.ndarray, min_pixels: int = 8) -> np.ndarray:
    mask = np.asarray(binary_mask, dtype=bool)
    min_pixels = max(int(min_pixels), 1)
    if mask.sum() == 0:
        return mask

    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        kept = np.zeros_like(mask, dtype=bool)
        for label_idx in range(1, int(num_labels)):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area >= min_pixels:
                kept |= labels == label_idx
        return kept if kept.any() else mask

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=bool)

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

            if len(component) >= min_pixels:
                for cy, cx in component:
                    kept[cy, cx] = True

    return kept if kept.any() else mask


def _estimate_foreground_mask_basic(gray_image: np.ndarray, min_area_ratio: float = 0.002) -> np.ndarray:
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


def _estimate_foreground_mask_cv2(gray_image: np.ndarray, min_area_ratio: float = 0.002) -> np.ndarray:
    _require_cv2("_estimate_foreground_mask_cv2")

    gray = np.asarray(np.clip(gray_image, 0.0, 1.0) * 255.0, dtype=np.uint8)
    h, w = gray.shape
    min_side = max(min(h, w), 1)

    # Blur first so bright/dark bands collapse into a single chromosome body before thresholding.
    sigma = max(float(min_side) / 32.0, 2.0)
    kernel = max(int(round(sigma * 4)) * 2 + 1, 9)
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    inverted = 255 - blurred

    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_size = max((min_side // 24) | 1, 7)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    dilate_size = max((min_side // 40) | 1, 3)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    binary = cv2.dilate(binary, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _estimate_foreground_mask_basic(gray_image, min_area_ratio=min_area_ratio)

    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, [largest], contourIdx=-1, color=255, thickness=-1)

    # One more close step to remove band-induced notches.
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, close_kernel)
    mask = filled > 0

    if mask.mean() < min_area_ratio:
        return _estimate_foreground_mask_basic(gray_image, min_area_ratio=min_area_ratio)

    return largest_connected_component(mask)


def _estimate_foreground_mask_white_bg(
    gray_image: np.ndarray,
    white_threshold: float = 254.5 / 255.0,
    min_component_pixels: int = 8,
) -> np.ndarray:
    gray_image = np.asarray(np.clip(gray_image, 0.0, 1.0), dtype=np.float32)
    mask = gray_image < float(white_threshold)
    if not mask.any():
        mask = gray_image < 1.0
    if not mask.any():
        return mask.astype(bool)

    if cv2 is not None:
        mask_uint8 = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = mask_uint8 > 0

    mask = remove_small_connected_components(mask, min_pixels=min_component_pixels)
    return mask.astype(bool)


def estimate_foreground_mask(
    gray_image: np.ndarray,
    min_area_ratio: float = 0.002,
    mode: str = "auto",
    white_threshold: float = 254.5 / 255.0,
) -> np.ndarray:
    if mode == "white_bg_exact":
        return _estimate_foreground_mask_white_bg(
            gray_image=gray_image,
            white_threshold=white_threshold,
        )
    if mode == "basic":
        return _estimate_foreground_mask_basic(gray_image, min_area_ratio=min_area_ratio)
    if mode == "cv2":
        if cv2 is None:
            return _estimate_foreground_mask_basic(gray_image, min_area_ratio=min_area_ratio)
        return _estimate_foreground_mask_cv2(gray_image, min_area_ratio=min_area_ratio)
    if mode != "auto":
        raise ValueError(f"Unsupported foreground mask mode: {mode}")

    if cv2 is not None:
        try:
            return _estimate_foreground_mask_cv2(gray_image, min_area_ratio=min_area_ratio)
        except Exception:
            pass
    return _estimate_foreground_mask_basic(gray_image, min_area_ratio=min_area_ratio)


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


def bilinear_sample(image: np.ndarray, y_coords: np.ndarray, x_coords: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    y_coords = np.asarray(y_coords, dtype=np.float32)
    x_coords = np.asarray(x_coords, dtype=np.float32)

    h, w = image.shape
    y0 = np.floor(y_coords).astype(np.int32)
    x0 = np.floor(x_coords).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    valid = (y0 >= 0) & (x0 >= 0) & (y1 < h) & (x1 < w)
    sampled = np.full(y_coords.shape, float(fill_value), dtype=np.float32)
    if not np.any(valid):
        return sampled

    y0v = y0[valid]
    x0v = x0[valid]
    y1v = y1[valid]
    x1v = x1[valid]
    wy = y_coords[valid] - y0v.astype(np.float32)
    wx = x_coords[valid] - x0v.astype(np.float32)

    top_left = image[y0v, x0v]
    top_right = image[y0v, x1v]
    bottom_left = image[y1v, x0v]
    bottom_right = image[y1v, x1v]

    sampled_valid = (
        (1.0 - wy) * (1.0 - wx) * top_left
        + (1.0 - wy) * wx * top_right
        + wy * (1.0 - wx) * bottom_left
        + wy * wx * bottom_right
    )
    sampled[valid] = sampled_valid.astype(np.float32)
    return sampled


def estimate_centerline_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    center_x = np.full(mask.shape[0], np.nan, dtype=np.float32)
    width_values = np.full(mask.shape[0], np.nan, dtype=np.float32)

    for row_idx in range(mask.shape[0]):
        fg_cols = np.where(mask[row_idx])[0]
        if fg_cols.size == 0:
            continue
        center_x[row_idx] = float(fg_cols.mean())
        width_values[row_idx] = float(fg_cols.size)

    center_x, valid_fraction = interpolate_missing_1d(center_x, fill_value=float(mask.shape[1] / 2.0))
    width_values, _ = interpolate_missing_1d(width_values, fill_value=0.0)
    center_x = moving_average_1d(center_x, kernel_size=7)
    width_values = moving_average_1d(width_values, kernel_size=7)
    return center_x.astype(np.float32), width_values.astype(np.float32)


def estimate_centerline_edges_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mask = np.asarray(mask, dtype=bool)
    h, _ = mask.shape
    left_edges = np.full(h, np.nan, dtype=np.float32)
    right_edges = np.full(h, np.nan, dtype=np.float32)

    for row_idx in range(h):
        fg_cols = np.where(mask[row_idx])[0]
        if fg_cols.size == 0:
            continue
        left_edges[row_idx] = float(fg_cols.min())
        right_edges[row_idx] = float(fg_cols.max())

    valid_rows = np.isfinite(left_edges) & np.isfinite(right_edges)
    valid_fraction = float(valid_rows.mean()) if h > 0 else 0.0
    if valid_rows.sum() == 0:
        center_x = np.full(h, np.nan, dtype=np.float32)
        widths = np.zeros(h, dtype=np.float32)
        return center_x, left_edges, right_edges, valid_fraction

    row_axis = np.arange(h, dtype=np.float32)
    left_interp = np.interp(row_axis, row_axis[valid_rows], left_edges[valid_rows]).astype(np.float32)
    right_interp = np.interp(row_axis, row_axis[valid_rows], right_edges[valid_rows]).astype(np.float32)

    smooth_kernel = 9 if h >= 9 else max((h // 2) * 2 + 1, 3)
    left_interp = moving_average_1d(left_interp, kernel_size=smooth_kernel)
    right_interp = moving_average_1d(right_interp, kernel_size=smooth_kernel)
    center_x = 0.5 * (left_interp + right_interp)
    widths = np.maximum(right_interp - left_interp + 1.0, 1.0).astype(np.float32)
    return center_x.astype(np.float32), left_interp.astype(np.float32), right_interp.astype(np.float32), valid_fraction


def skeletonize_mask_cv2(mask: np.ndarray) -> np.ndarray:
    _require_cv2("skeletonize_mask_cv2")
    mask_uint8 = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
    if cv2.countNonZero(mask_uint8) == 0:
        return np.zeros_like(mask, dtype=bool)

    skeleton = np.zeros_like(mask_uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    working = mask_uint8.copy()

    while True:
        opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(working, opened)
        eroded = cv2.erode(working, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        working = eroded
        if cv2.countNonZero(working) == 0:
            break

    return (skeleton > 0)


def _skeleton_neighbors(coords_set: set, y: int, x: int) -> List[Tuple[int, int]]:
    neighbors: List[Tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = (y + dy, x + dx)
            if neighbor in coords_set:
                neighbors.append(neighbor)
    return neighbors


def extract_skeleton_main_path(mask: np.ndarray) -> Optional[np.ndarray]:
    skeleton = skeletonize_mask_cv2(mask)
    skeleton = largest_connected_component(skeleton)
    coords = np.argwhere(skeleton)
    if coords.shape[0] < 2:
        return None

    coords_list = [tuple(int(v) for v in coord) for coord in coords]
    coords_set = set(coords_list)
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = {}
    for y, x in coords_list:
        edges: List[Tuple[Tuple[int, int], float]] = []
        for ny, nx in _skeleton_neighbors(coords_set, y, x):
            weight = float(np.hypot(float(ny - y), float(nx - x)))
            edges.append(((ny, nx), weight))
        adjacency[(y, x)] = edges

    def farthest_from(start: Tuple[int, int]):
        dist: Dict[Tuple[int, int], float] = {start: 0.0}
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
        heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        while heap:
            cur_dist, node = heapq.heappop(heap)
            if cur_dist > dist[node] + 1e-6:
                continue
            for nxt, weight in adjacency[node]:
                cand = cur_dist + weight
                if cand < dist.get(nxt, float("inf")):
                    dist[nxt] = cand
                    prev[nxt] = node
                    heapq.heappush(heap, (cand, nxt))
        end_node = max(dist.items(), key=lambda item: item[1])[0]
        return end_node, dist, prev

    start_seed = min(coords_list, key=lambda item: (item[0], item[1]))
    path_start, _, _ = farthest_from(start_seed)
    path_end, _, prev = farthest_from(path_start)

    path_nodes: List[Tuple[int, int]] = [path_end]
    current = path_end
    while current != path_start:
        parent = prev.get(current)
        if parent is None:
            break
        path_nodes.append(parent)
        current = parent

    if len(path_nodes) < 2:
        return None

    path_nodes.reverse()
    path_array = np.asarray([[float(x), float(y)] for y, x in path_nodes], dtype=np.float32)
    return path_array


def resample_polyline(points_xy: np.ndarray, num_points: int) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"Expected polyline with shape [N, 2], got {points_xy.shape}")
    if points_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points_xy.shape[0] == 1 or num_points <= 1:
        return np.repeat(points_xy[:1], max(int(num_points), 1), axis=0).astype(np.float32)

    deltas = np.diff(points_xy, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    arc_length = np.zeros(points_xy.shape[0], dtype=np.float32)
    arc_length[1:] = np.cumsum(seg_lengths.astype(np.float32))
    total_length = float(arc_length[-1])
    if total_length <= 1e-6:
        return np.repeat(points_xy[:1], max(int(num_points), 1), axis=0).astype(np.float32)

    sample_s = np.linspace(0.0, total_length, num=max(int(num_points), 2), dtype=np.float32)
    sample_x = np.interp(sample_s, arc_length, points_xy[:, 0]).astype(np.float32)
    sample_y = np.interp(sample_s, arc_length, points_xy[:, 1]).astype(np.float32)
    return np.stack([sample_x, sample_y], axis=1).astype(np.float32)


def build_skeleton_path_straightened_image(
    gray_image: np.ndarray,
    mask: np.ndarray,
    target_width: Optional[int] = None,
    width_scale: float = 1.35,
    min_margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    gray_image = np.asarray(gray_image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        mask = np.ones_like(gray_image, dtype=bool)

    path_xy = extract_skeleton_main_path(mask)
    if path_xy is None or path_xy.shape[0] < 2:
        return build_centerline_shift_straightened_image(
            gray_image=gray_image,
            mask=mask,
            target_width=target_width,
            width_scale=width_scale,
            min_margin=min_margin,
        )

    smooth_kernel = 9 if path_xy.shape[0] >= 9 else 5
    path_x = moving_average_1d(path_xy[:, 0], kernel_size=smooth_kernel)
    path_y = moving_average_1d(path_xy[:, 1], kernel_size=smooth_kernel)
    smooth_path_xy = np.stack([path_x, path_y], axis=1).astype(np.float32)

    deltas = np.diff(smooth_path_xy, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    total_length = float(seg_lengths.sum())
    target_height = max(int(round(total_length)) + 1, int(path_xy.shape[0]))
    sample_path_xy = resample_polyline(smooth_path_xy, num_points=target_height)

    tangent = np.gradient(sample_path_xy, axis=0)
    tangent_norm = np.sqrt((tangent ** 2).sum(axis=1, keepdims=True))
    tangent_norm = np.maximum(tangent_norm, 1e-6)
    tangent = tangent / tangent_norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1).astype(np.float32)

    distance_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    sample_radius = bilinear_sample(
        distance_map,
        sample_path_xy[:, 1],
        sample_path_xy[:, 0],
        fill_value=0.0,
    )
    sample_radius = np.maximum(sample_radius.astype(np.float32), 1.5)
    robust_radius = float(np.quantile(sample_radius, 0.95)) if sample_radius.size > 0 else 4.0
    raw_width = int(np.ceil(max(2.0 * robust_radius * float(width_scale) + 2.0 * float(min_margin), 8.0)))
    if target_width is not None and int(target_width) > 0:
        raw_width = max(raw_width, int(target_width))

    offsets = np.arange(raw_width, dtype=np.float32) - (float(raw_width - 1) / 2.0)
    sampled_rows = []
    sampled_masks = []
    mask_float = mask.astype(np.float32)
    valid_ratios = []

    for idx in range(sample_path_xy.shape[0]):
        half_extent = max(float(sample_radius[idx]) * float(width_scale), float(raw_width) / 2.5)
        scaled_offsets = offsets * (half_extent / max(float(raw_width) / 2.0, 1e-6))
        sample_x = sample_path_xy[idx, 0] + scaled_offsets * normal[idx, 0]
        sample_y = sample_path_xy[idx, 1] + scaled_offsets * normal[idx, 1]
        sampled_row = bilinear_sample(gray_image, sample_y, sample_x, fill_value=1.0)
        sampled_mask = bilinear_sample(mask_float, sample_y, sample_x, fill_value=0.0)
        sampled_rows.append(sampled_row.astype(np.float32))
        sampled_masks.append(sampled_mask.astype(np.float32))
        valid_ratios.append(float((sampled_mask > 0.5).mean()))

    straightened_image = np.stack(sampled_rows, axis=0).astype(np.float32)
    straightened_mask = np.stack(sampled_masks, axis=0).astype(np.float32)
    straightened_mask = (straightened_mask > 0.10).astype(np.float32)
    straightened_image = np.where(straightened_mask > 0.5, straightened_image, 1.0).astype(np.float32)
    straightened_image = np.clip(straightened_image, 0.0, 1.0).astype(np.float32)
    valid_fraction = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    return straightened_image, straightened_mask, valid_fraction


def _safe_odd_kernel_size(length: int, desired: int, minimum: int = 3) -> int:
    length = int(length)
    if length <= 1:
        return 1
    upper = length if length % 2 == 1 else length - 1
    upper = max(upper, 1)
    desired = min(int(desired), upper)
    if desired % 2 == 0:
        desired = max(desired - 1, 1)
    minimum = min(int(minimum), upper)
    if minimum % 2 == 0:
        minimum = max(minimum - 1, 1)
    return max(desired, minimum)


def nonlinear_band_enhance_1d(values: np.ndarray, fine_kernel: int = 5, coarse_kernel: int = 21) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.float32)

    fine_kernel = _safe_odd_kernel_size(values.size, fine_kernel, minimum=1)
    coarse_kernel = _safe_odd_kernel_size(values.size, coarse_kernel, minimum=max(fine_kernel, 3))
    coarse = moving_average_1d(values, kernel_size=coarse_kernel)
    detail = values - coarse
    if fine_kernel > 1:
        detail = moving_average_1d(detail, kernel_size=fine_kernel)

    local_var = moving_average_1d(detail ** 2, kernel_size=coarse_kernel)
    scale = np.sqrt(np.maximum(local_var, 1e-6))
    enhanced = detail / scale
    enhanced = (enhanced - enhanced.mean()) / max(float(enhanced.std()), 1e-6)
    return enhanced.astype(np.float32)


def nonlinear_band_enhance_2d(band_image: np.ndarray, fine_kernel: int = 5, coarse_kernel: int = 21) -> np.ndarray:
    band_image = np.asarray(band_image, dtype=np.float32)
    if band_image.ndim != 2:
        raise ValueError(f"Expected 2D band image, got shape={band_image.shape}")
    if band_image.size == 0:
        return band_image.astype(np.float32)

    enhanced_cols = [
        nonlinear_band_enhance_1d(band_image[:, col_idx], fine_kernel=fine_kernel, coarse_kernel=coarse_kernel)
        for col_idx in range(band_image.shape[1])
    ]
    enhanced = np.stack(enhanced_cols, axis=1).astype(np.float32)
    enhanced = enhanced - enhanced.mean(axis=1, keepdims=True)
    enhanced = (enhanced - enhanced.mean()) / max(float(enhanced.std()), 1e-6)
    return enhanced.astype(np.float32)


def build_skeleton_density_band_image(
    gray_image: np.ndarray,
    mask: np.ndarray,
    band_width: int,
    width_scale: float = 1.20,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
    gray_image = np.asarray(gray_image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        mask = np.ones_like(gray_image, dtype=bool)

    if cv2 is None:
        return None, None, None, 0.0

    path_xy = extract_skeleton_main_path(mask)
    if path_xy is None or path_xy.shape[0] < 2:
        return None, None, None, 0.0

    smooth_kernel = _safe_odd_kernel_size(path_xy.shape[0], desired=9, minimum=3)
    path_x = moving_average_1d(path_xy[:, 0], kernel_size=smooth_kernel)
    path_y = moving_average_1d(path_xy[:, 1], kernel_size=smooth_kernel)
    smooth_path_xy = np.stack([path_x, path_y], axis=1).astype(np.float32)

    deltas = np.diff(smooth_path_xy, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    total_length = float(seg_lengths.sum())
    target_height = max(int(round(total_length)) + 1, int(path_xy.shape[0]), 8)
    sample_path_xy = resample_polyline(smooth_path_xy, num_points=target_height)

    tangent = np.gradient(sample_path_xy, axis=0)
    tangent_norm = np.sqrt((tangent ** 2).sum(axis=1, keepdims=True))
    tangent_norm = np.maximum(tangent_norm, 1e-6)
    tangent = tangent / tangent_norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1).astype(np.float32)

    distance_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    sample_radius = bilinear_sample(
        distance_map,
        sample_path_xy[:, 1],
        sample_path_xy[:, 0],
        fill_value=0.0,
    )
    sample_radius = np.maximum(sample_radius.astype(np.float32), 1.5)

    offsets = np.linspace(-1.0, 1.0, num=int(band_width), dtype=np.float32)
    mask_float = mask.astype(np.float32)
    band_rows = []
    density_profile = []
    width_values = []
    valid_ratios = []

    for idx in range(sample_path_xy.shape[0]):
        half_extent = max(float(sample_radius[idx]) * float(width_scale), 2.0)
        sample_x = sample_path_xy[idx, 0] + offsets * half_extent * normal[idx, 0]
        sample_y = sample_path_xy[idx, 1] + offsets * half_extent * normal[idx, 1]
        sampled_pixels = bilinear_sample(gray_image, sample_y, sample_x, fill_value=1.0)
        sampled_mask = bilinear_sample(mask_float, sample_y, sample_x, fill_value=0.0)
        valid_mask = sampled_mask > 0.5

        density_strip = 1.0 - sampled_pixels
        density_strip = np.where(valid_mask, density_strip, 0.0).astype(np.float32)
        if valid_mask.any():
            density_profile.append(float(np.median(density_strip[valid_mask])))
            valid_ratios.append(float(valid_mask.mean()))
        else:
            density_profile.append(0.0)
            valid_ratios.append(0.0)

        band_rows.append(density_strip.astype(np.float32))
        width_values.append(float(sample_radius[idx] * 2.0))

    band_image = np.stack(band_rows, axis=0).astype(np.float32)
    density_profile = np.asarray(density_profile, dtype=np.float32)
    width_values = np.asarray(width_values, dtype=np.float32)
    valid_fraction = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    return band_image, density_profile, width_values, valid_fraction


def build_centerline_band_image(
    gray_image: np.ndarray,
    mask: np.ndarray,
    band_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    center_x, width_values = estimate_centerline_from_mask(mask)
    row_indices = np.arange(mask.shape[0], dtype=np.float32)
    tangent = np.gradient(center_x.astype(np.float32))
    normal_x = np.ones_like(tangent, dtype=np.float32)
    normal_y = -tangent.astype(np.float32)
    norm = np.sqrt(normal_x ** 2 + normal_y ** 2)
    normal_x = normal_x / np.maximum(norm, 1e-6)
    normal_y = normal_y / np.maximum(norm, 1e-6)

    half_extent = np.maximum(width_values * 0.6, 3.0)
    offsets = np.linspace(-1.0, 1.0, num=band_width, dtype=np.float32)

    band_rows = []
    row_values = []
    valid_ratios = []
    mask_float = mask.astype(np.float32)
    default_fill = float(np.nanmean(gray_image)) if gray_image.size > 0 else 1.0
    if not np.isfinite(default_fill):
        default_fill = 1.0

    for idx in range(mask.shape[0]):
        sample_x = center_x[idx] + offsets * half_extent[idx] * normal_x[idx]
        sample_y = row_indices[idx] + offsets * half_extent[idx] * normal_y[idx]
        sampled_pixels = bilinear_sample(gray_image, sample_y, sample_x, fill_value=default_fill)
        sampled_mask = bilinear_sample(mask_float, sample_y, sample_x, fill_value=0.0)
        valid_mask = sampled_mask > 0.5

        if valid_mask.any():
            valid_pixels = sampled_pixels[valid_mask]
            row_values.append(float(np.median(valid_pixels)))
            valid_ratios.append(float(valid_mask.mean()))
        else:
            valid_pixels = sampled_pixels
            row_values.append(float(np.median(valid_pixels)))
            valid_ratios.append(0.0)

        band_rows.append(sampled_pixels.astype(np.float32))

    band_image = np.stack(band_rows, axis=0).astype(np.float32)
    row_values = np.asarray(row_values, dtype=np.float32)
    valid_ratios = np.asarray(valid_ratios, dtype=np.float32)
    valid_fraction = float(valid_ratios.mean()) if valid_ratios.size > 0 else 0.0
    return band_image, row_values, width_values, valid_fraction


def build_centerline_shift_straightened_image(
    gray_image: np.ndarray,
    mask: np.ndarray,
    target_width: Optional[int] = None,
    width_scale: float = 1.10,
    min_margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    gray_image = np.asarray(gray_image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        mask = np.ones_like(gray_image, dtype=bool)

    center_x, left_edges, right_edges, valid_fraction = estimate_centerline_edges_from_mask(mask)
    valid_rows = np.isfinite(center_x)
    if valid_rows.sum() == 0:
        return gray_image.astype(np.float32), mask.astype(np.float32), 0.0

    widths = np.maximum(right_edges - left_edges + 1.0, 1.0)
    robust_width = float(np.quantile(widths[valid_rows], 0.95)) if valid_rows.any() else float(gray_image.shape[1])
    raw_width = int(
        np.ceil(
            max(
                robust_width * float(width_scale),
                float(np.nanmax(widths[valid_rows])) + 2.0 * float(min_margin),
            )
        )
    )
    if target_width is not None and int(target_width) > 0:
        raw_width = max(raw_width, int(target_width))
    raw_width = max(raw_width, 8)

    row_indices = np.arange(mask.shape[0], dtype=np.float32)
    centerline_x = moving_average_1d(center_x.astype(np.float32), kernel_size=11 if mask.shape[0] >= 11 else 5)
    centerline_y = row_indices.astype(np.float32)
    widths_smooth = moving_average_1d(widths.astype(np.float32), kernel_size=11 if mask.shape[0] >= 11 else 5)

    diff_x = np.diff(centerline_x)
    diff_y = np.diff(centerline_y)
    step_lengths = np.sqrt(diff_x ** 2 + diff_y ** 2)
    arc_length = np.zeros(mask.shape[0], dtype=np.float32)
    if step_lengths.size > 0:
        arc_length[1:] = np.cumsum(step_lengths.astype(np.float32))

    total_length = float(arc_length[-1]) if arc_length.size > 0 else 0.0
    target_height = max(int(round(total_length)) + 1, int(mask.shape[0]))
    sample_s = np.linspace(0.0, total_length, num=target_height, dtype=np.float32) if total_length > 0 else row_indices

    sample_center_x = np.interp(sample_s, arc_length, centerline_x).astype(np.float32)
    sample_center_y = np.interp(sample_s, arc_length, centerline_y).astype(np.float32)
    sample_widths = np.interp(sample_s, arc_length, widths_smooth).astype(np.float32)

    tangent_x = np.gradient(sample_center_x.astype(np.float32))
    tangent_y = np.gradient(sample_center_y.astype(np.float32))
    tangent_norm = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
    tangent_norm = np.maximum(tangent_norm, 1e-6)
    tangent_x = tangent_x / tangent_norm
    tangent_y = tangent_y / tangent_norm

    normal_x = -tangent_y
    normal_y = tangent_x

    offsets = np.arange(raw_width, dtype=np.float32) - (float(raw_width - 1) / 2.0)

    sampled_rows = []
    sampled_masks = []
    mask_float = mask.astype(np.float32)
    for sample_idx in range(sample_center_x.size):
        half_extent = max(float(sample_widths[sample_idx]) * 0.55, float(raw_width) / 2.0)
        scaled_offsets = offsets * (half_extent / max(float(raw_width) / 2.0, 1e-6))
        sample_x = sample_center_x[sample_idx] + scaled_offsets * normal_x[sample_idx]
        sample_y = sample_center_y[sample_idx] + scaled_offsets * normal_y[sample_idx]
        sampled_row = bilinear_sample(gray_image, sample_y, sample_x, fill_value=1.0)
        sampled_mask = bilinear_sample(mask_float, sample_y, sample_x, fill_value=0.0)
        sampled_rows.append(sampled_row.astype(np.float32))
        sampled_masks.append(sampled_mask.astype(np.float32))

    straightened_image = np.stack(sampled_rows, axis=0).astype(np.float32)
    straightened_mask = np.stack(sampled_masks, axis=0).astype(np.float32)
    straightened_mask = (straightened_mask > 0.10).astype(np.float32)
    straightened_image = np.where(straightened_mask > 0.5, straightened_image, 1.0).astype(np.float32)
    straightened_image = np.clip(straightened_image, 0.0, 1.0).astype(np.float32)
    return straightened_image, straightened_mask, float(valid_fraction)


def _require_cv2(method_name: str):
    if cv2 is None:
        raise ImportError(
            f"{method_name} requires OpenCV (`cv2`) but it is not installed in the current environment."
        )


def resample_2d_height_first(image: np.ndarray, target_height: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={image.shape}")

    if image.shape[0] == target_height:
        return image.astype(np.float32)

    columns = [
        resample_1d(image[:, col_idx], target_height)
        for col_idx in range(image.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def _prepare_aligned_crop(gray_image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
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

    return aligned_image.astype(np.float32), aligned_mask.astype(bool), float(angle_deg)


def repair_vertical_chromosome_mask(
    mask: np.ndarray,
    smooth_kernel_size: int = 11,
    expand_ratio: float = 0.0,
    min_expand_pixels: float = 0.0,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return mask

    h, w = mask.shape
    left_edges = np.full(h, np.nan, dtype=np.float32)
    right_edges = np.full(h, np.nan, dtype=np.float32)

    for row_idx in range(h):
        cols = np.where(mask[row_idx])[0]
        if cols.size == 0:
            continue
        left_edges[row_idx] = float(cols.min())
        right_edges[row_idx] = float(cols.max())

    valid_rows = np.where(np.isfinite(left_edges) & np.isfinite(right_edges))[0]
    if valid_rows.size < 3:
        return mask

    row_axis = np.arange(h, dtype=np.float32)
    left_interp = np.interp(row_axis, valid_rows.astype(np.float32), left_edges[valid_rows]).astype(np.float32)
    right_interp = np.interp(row_axis, valid_rows.astype(np.float32), right_edges[valid_rows]).astype(np.float32)

    kernel = max(int(smooth_kernel_size) | 1, 3)
    if kernel < h:
        left_interp = moving_average_1d(left_interp, kernel_size=kernel)
        right_interp = moving_average_1d(right_interp, kernel_size=kernel)

    widths = np.maximum(right_interp - left_interp + 1.0, 1.0)
    median_width = float(np.median(widths[valid_rows])) if valid_rows.size > 0 else 1.0
    expand = max(float(min_expand_pixels), float(expand_ratio) * median_width)

    repaired = np.zeros_like(mask, dtype=bool)
    start_row = int(valid_rows.min())
    end_row = int(valid_rows.max())
    for row_idx in range(start_row, end_row + 1):
        left = int(np.floor(max(left_interp[row_idx] - expand, 0.0)))
        right = int(np.ceil(min(right_interp[row_idx] + expand, w - 1.0)))
        if right >= left:
            repaired[row_idx, left : right + 1] = True

    repaired |= mask
    repaired = remove_small_connected_components(repaired, min_pixels=8)
    return largest_connected_component(repaired)


def refine_mask_preserve_geometry(mask: np.ndarray, close_kernel_size: int = 3) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return mask

    refined = remove_small_connected_components(mask, min_pixels=8)
    refined = largest_connected_component(refined)

    if cv2 is not None and int(close_kernel_size) > 1:
        kernel_size = max(int(close_kernel_size) | 1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        refined_uint8 = cv2.morphologyEx(
            refined.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )
        refined = refined_uint8 > 0
        refined = largest_connected_component(refined)

    return refined.astype(bool)


def _resize_gray_image(image: np.ndarray, out_width: int, out_height: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if cv2 is not None:
        resized = cv2.resize(
            image,
            (int(out_width), int(out_height)),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

    pil_image = Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    resized = pil_image.resize((int(out_width), int(out_height)), resample=Image.BILINEAR)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32)


def _resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    output_height: int,
    output_width: int,
    resize_mode: str = "fixed",
) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

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

    resized_image = _resize_gray_image(image, out_width=target_width, out_height=target_height)
    resized_mask = _resize_gray_image(mask.astype(np.float32), out_width=target_width, out_height=target_height)
    return resized_image.astype(np.float32), resized_mask.astype(np.float32)


def _add_border_gray(gray_image: np.ndarray, value: float = 1.0) -> np.ndarray:
    h, w = gray_image.shape
    border = max(h // 2, w // 2)
    return np.pad(
        gray_image,
        ((border, border), (border, border)),
        mode="constant",
        constant_values=float(value),
    ).astype(np.float32)


def _binary_mask_uint8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255


def _crop_to_nonzero(image: np.ndarray, mask_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_uint8 > 0)
    if ys.size == 0 or xs.size == 0:
        return image.astype(np.float32), (mask_uint8 > 0)

    y0 = max(int(ys.min()), 0)
    y1 = min(int(ys.max()) + 1, image.shape[0])
    x0 = max(int(xs.min()), 0)
    x1 = min(int(xs.max()) + 1, image.shape[1])
    return image[y0:y1, x0:x1].astype(np.float32), (mask_uint8[y0:y1, x0:x1] > 0)


def _rotate_gray_and_mask_cv2(gray_image: np.ndarray, mask_uint8: np.ndarray, degree: float) -> Tuple[np.ndarray, np.ndarray]:
    _require_cv2("_rotate_gray_and_mask_cv2")

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


def _find_local_extrema_1d(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size < 3:
        return np.asarray([], dtype=np.int32)

    left = values[1:-1] - values[:-2]
    right = values[1:-1] - values[2:]
    if mode == "max":
        mask = (left >= 0) & (right >= 0) & ((left > 0) | (right > 0))
    elif mode == "min":
        mask = (left <= 0) & (right <= 0) & ((left < 0) | (right < 0))
    else:
        raise ValueError(f"Unsupported extrema mode: {mode}")
    return np.where(mask)[0] + 1


def _smooth_projection_for_bend(y_projection: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_projection = np.asarray(y_projection, dtype=np.float32)
    if y_projection.size < 4:
        idx = np.arange(y_projection.size, dtype=np.float32)
        return idx, y_projection

    src_idx = np.arange(y_projection.size, dtype=np.float32)
    dst_size = max(int(np.ceil(y_projection.size * 0.5)), 8)
    dst_idx = np.linspace(0.0, float(y_projection.size - 1), dst_size, dtype=np.float32)
    inter = np.interp(dst_idx, src_idx, y_projection).astype(np.float32)
    inter = moving_average_1d(inter, kernel_size=7 if inter.size >= 7 else 3)
    return dst_idx, inter


def _s_score_from_projection(mask_uint8: np.ndarray) -> Tuple[float, Optional[int], Dict[str, np.ndarray]]:
    y_projection = np.sum(mask_uint8 > 0, axis=1).astype(np.float32)
    smooth_idx, smooth_projection = _smooth_projection_for_bend(y_projection)

    sag_indices = _find_local_extrema_1d(smooth_projection, mode="min")
    if sag_indices.size == 0:
        return float("inf"), None, {}

    sag_idx = int(sag_indices[np.argmin(smooth_projection[sag_indices])])
    sag_value = float(smooth_projection[sag_idx])

    left_projection = smooth_projection[:sag_idx]
    right_projection = smooth_projection[sag_idx + 1 :]
    left_peaks = _find_local_extrema_1d(left_projection, mode="max")
    right_peaks = _find_local_extrema_1d(right_projection, mode="max")
    if left_peaks.size == 0 or right_peaks.size == 0:
        return float("inf"), None, {}

    left_peak = float(left_projection[left_peaks].max())
    right_peak = float(right_projection[right_peaks].max())
    denom = max(left_peak + right_peak, 1e-6)
    r1 = abs(left_peak - right_peak) / denom
    r2 = sag_value / denom
    score = 0.5 * r1 + 0.5 * r2
    bend_row = int(round(float(smooth_idx[sag_idx])))
    debug = {
        "y_projection": y_projection,
        "smooth_idx": smooth_idx,
        "smooth_projection": smooth_projection,
    }
    return float(score), bend_row, debug


def _find_best_global_bend_rotation(gray_image: np.ndarray, mask: np.ndarray, angle_step: int = 5) -> Tuple[np.ndarray, np.ndarray, float, Optional[int], float]:
    _require_cv2("_find_best_global_bend_rotation")

    bordered_gray = _add_border_gray(gray_image, value=1.0)
    bordered_mask = _add_border_gray(mask.astype(np.float32), value=0.0) > 0.5
    mask_uint8 = _binary_mask_uint8(bordered_mask)

    best = None
    for degree in range(0, 180, int(angle_step)):
        rotated_gray, rotated_mask = _rotate_gray_and_mask_cv2(bordered_gray, mask_uint8, float(degree))
        cropped_gray, cropped_mask = _crop_to_nonzero(rotated_gray, rotated_mask)
        cropped_mask_uint8 = _binary_mask_uint8(cropped_mask)
        score, bend_row, _ = _s_score_from_projection(cropped_mask_uint8)
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
        aligned_gray, aligned_mask, angle_deg = _prepare_aligned_crop(gray_image, mask)
        return aligned_gray, aligned_mask, float(angle_deg), None, float("inf")

    return best["gray"], best["mask"], best["degree"], best["bend_row"], best["score"]


def _find_min_width_rotation(gray_image: np.ndarray, mask: np.ndarray, angle_step: int = 5) -> Tuple[np.ndarray, np.ndarray, float]:
    _require_cv2("_find_min_width_rotation")

    if gray_image.size == 0:
        return gray_image.astype(np.float32), mask.astype(bool), 0.0

    bordered_gray = _add_border_gray(gray_image, value=1.0)
    bordered_mask = _add_border_gray(mask.astype(np.float32), value=0.0) > 0.5
    mask_uint8 = _binary_mask_uint8(bordered_mask)

    best = None
    for degree in range(-90, 90, int(angle_step)):
        rotated_gray, rotated_mask = _rotate_gray_and_mask_cv2(bordered_gray, mask_uint8, float(degree))
        cropped_gray, cropped_mask = _crop_to_nonzero(rotated_gray, rotated_mask)
        projection = np.sum(cropped_mask, axis=0).astype(np.float32)
        nonzero_cols = np.where(projection > 0)[0]
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


def _pad_right_to_width(image: np.ndarray, target_width: int, fill_value: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.shape[1] >= target_width:
        return image
    pad_width = int(target_width - image.shape[1])
    return np.pad(image, ((0, 0), (0, pad_width)), mode="constant", constant_values=float(fill_value)).astype(np.float32)


def extract_projection_split_straightened_image(
    gray_image: np.ndarray,
    mask: np.ndarray,
    output_height: int = 300,
    output_width: int = 96,
    global_angle_step: int = 5,
    local_angle_step: int = 5,
    seam_trim: int = 3,
    repair_expand_ratio: float = 0.0,
    resize_mode: str = "fixed",
) -> StraightenedChromosomeImage:
    aligned_gray, aligned_mask, global_degree, bend_row, _ = _find_best_global_bend_rotation(
        gray_image=gray_image,
        mask=mask,
        angle_step=global_angle_step,
    )
    aligned_mask = repair_vertical_chromosome_mask(
        aligned_mask,
        expand_ratio=repair_expand_ratio,
    )

    if bend_row is None or bend_row <= 2 or bend_row >= aligned_gray.shape[0] - 2:
        return extract_straightened_chromosome_image(
            gray_image=gray_image,
            mask=mask,
            output_height=output_height,
            output_width=output_width,
            smooth_kernel_size=5,
            method="centerline_unfold",
            repair_expand_ratio=repair_expand_ratio,
            resize_mode=resize_mode,
        )

    upper_gray = aligned_gray[:bend_row, :]
    lower_gray = aligned_gray[bend_row:, :]
    upper_mask = aligned_mask[:bend_row, :]
    lower_mask = aligned_mask[bend_row:, :]
    upper_mask = repair_vertical_chromosome_mask(
        upper_mask,
        expand_ratio=repair_expand_ratio,
    )
    lower_mask = repair_vertical_chromosome_mask(
        lower_mask,
        expand_ratio=repair_expand_ratio,
    )

    upper_rot_gray, upper_rot_mask, _ = _find_min_width_rotation(
        gray_image=upper_gray,
        mask=upper_mask,
        angle_step=local_angle_step,
    )
    lower_rot_gray, lower_rot_mask, _ = _find_min_width_rotation(
        gray_image=lower_gray,
        mask=lower_mask,
        angle_step=local_angle_step,
    )

    target_width = max(upper_rot_gray.shape[1], lower_rot_gray.shape[1])
    upper_rot_gray = _pad_right_to_width(upper_rot_gray, target_width, fill_value=1.0)
    lower_rot_gray = _pad_right_to_width(lower_rot_gray, target_width, fill_value=1.0)
    upper_rot_mask = _pad_right_to_width(upper_rot_mask.astype(np.float32), target_width, fill_value=0.0) > 0.5
    lower_rot_mask = _pad_right_to_width(lower_rot_mask.astype(np.float32), target_width, fill_value=0.0) > 0.5

    trim = max(int(seam_trim), 0)
    upper_keep = upper_rot_gray[:-trim, :] if trim > 0 and upper_rot_gray.shape[0] > trim else upper_rot_gray
    lower_keep = lower_rot_gray[trim:, :] if trim > 0 and lower_rot_gray.shape[0] > trim else lower_rot_gray
    upper_mask_keep = upper_rot_mask[:-trim, :] if trim > 0 and upper_rot_mask.shape[0] > trim else upper_rot_mask
    lower_mask_keep = lower_rot_mask[trim:, :] if trim > 0 and lower_rot_mask.shape[0] > trim else lower_rot_mask

    sewn_gray = np.concatenate([upper_keep, lower_keep], axis=0).astype(np.float32)
    sewn_mask = np.concatenate([upper_mask_keep, lower_mask_keep], axis=0).astype(np.float32)
    if sewn_gray.size == 0:
        sewn_gray = aligned_gray.astype(np.float32)
        sewn_mask = aligned_mask.astype(np.float32)

    straightened_image, straightened_mask = _resize_image_and_mask(
        sewn_gray,
        sewn_mask.astype(np.float32),
        output_height=output_height,
        output_width=output_width,
        resize_mode=resize_mode,
    )
    straightened_mask = (straightened_mask > 0.10).astype(np.float32)
    straightened_image = np.where(straightened_mask > 0.5, straightened_image, 1.0).astype(np.float32)
    straightened_image = np.clip(straightened_image, 0.0, 1.0).astype(np.float32)

    valid_profile_fraction = float(straightened_mask.mean())
    return StraightenedChromosomeImage(
        image=straightened_image,
        mask=straightened_mask,
        major_axis_angle_deg=float(global_degree),
        foreground_area=int(np.asarray(sewn_mask > 0.5, dtype=np.uint8).sum()),
        bbox_height=int(sewn_gray.shape[0]),
        bbox_width=int(sewn_gray.shape[1]),
        valid_profile_fraction=float(valid_profile_fraction),
    )


def extract_straightened_chromosome_image(
    gray_image: np.ndarray,
    mask: np.ndarray,
    output_height: int = 300,
    output_width: int = 96,
    smooth_kernel_size: int = 5,
    method: str = "centerline_unfold",
    global_angle_step: int = 5,
    local_angle_step: int = 5,
    seam_trim: int = 3,
    repair_expand_ratio: float = 0.0,
    resize_mode: str = "fixed",
) -> StraightenedChromosomeImage:
    if method == "projection_split_v1":
        return extract_projection_split_straightened_image(
            gray_image=gray_image,
            mask=mask,
            output_height=output_height,
            output_width=output_width,
            global_angle_step=global_angle_step,
            local_angle_step=local_angle_step,
            seam_trim=seam_trim,
            repair_expand_ratio=repair_expand_ratio,
            resize_mode=resize_mode,
        )
    if method == "skeleton_path_v1":
        aligned_image, aligned_mask, angle_deg = _prepare_aligned_crop(gray_image, mask)
        aligned_mask = refine_mask_preserve_geometry(aligned_mask, close_kernel_size=3)

        band_width = int(output_width) if int(output_width) > 0 and resize_mode == "fixed" else None
        straightened_image, straightened_mask, valid_fraction = build_skeleton_path_straightened_image(
            gray_image=aligned_image,
            mask=aligned_mask,
            target_width=band_width,
        )

        if output_height > 0 and straightened_image.shape[0] != int(output_height):
            resized_width = int(output_width) if resize_mode == "fixed" else straightened_image.shape[1]
            straightened_image, straightened_mask = _resize_image_and_mask(
                straightened_image,
                straightened_mask,
                output_height=output_height,
                output_width=resized_width,
                resize_mode="fixed" if resize_mode == "fixed" else "height_only",
            )

        effective_smooth_kernel = min(int(smooth_kernel_size), 3)
        if effective_smooth_kernel > 1 and straightened_image.shape[0] >= effective_smooth_kernel:
            smoothed_cols = [
                moving_average_1d(straightened_image[:, col_idx], kernel_size=effective_smooth_kernel)
                for col_idx in range(straightened_image.shape[1])
            ]
            straightened_image = np.stack(smoothed_cols, axis=1).astype(np.float32)

        straightened_mask = np.clip(straightened_mask, 0.0, 1.0).astype(np.float32)
        straightened_mask = (straightened_mask > 0.10).astype(np.float32)
        straightened_image = np.where(straightened_mask > 0.5, straightened_image, 1.0).astype(np.float32)
        straightened_image = np.clip(straightened_image, 0.0, 1.0).astype(np.float32)

        return StraightenedChromosomeImage(
            image=straightened_image,
            mask=straightened_mask,
            major_axis_angle_deg=float(angle_deg),
            foreground_area=int(aligned_mask.sum()),
            bbox_height=int(aligned_mask.shape[0]),
            bbox_width=int(aligned_mask.shape[1]),
            valid_profile_fraction=float(valid_fraction),
        )
    if method == "centerline_shift_v1":
        aligned_image, aligned_mask, angle_deg = _prepare_aligned_crop(gray_image, mask)
        aligned_mask = repair_vertical_chromosome_mask(
            aligned_mask,
            expand_ratio=repair_expand_ratio,
        )

        band_width = int(output_width) if int(output_width) > 0 and resize_mode == "fixed" else None
        straightened_image, straightened_mask, valid_fraction = build_centerline_shift_straightened_image(
            gray_image=aligned_image,
            mask=aligned_mask,
            target_width=band_width,
        )

        if output_height > 0 and straightened_image.shape[0] != int(output_height):
            resized_width = int(output_width) if resize_mode == "fixed" else straightened_image.shape[1]
            straightened_image, straightened_mask = _resize_image_and_mask(
                straightened_image,
                straightened_mask,
                output_height=output_height,
                output_width=resized_width,
                resize_mode="fixed" if resize_mode == "fixed" else "height_only",
            )

        effective_smooth_kernel = min(int(smooth_kernel_size), 3)
        if effective_smooth_kernel > 1 and straightened_image.shape[0] >= effective_smooth_kernel:
            smoothed_cols = [
                moving_average_1d(straightened_image[:, col_idx], kernel_size=effective_smooth_kernel)
                for col_idx in range(straightened_image.shape[1])
            ]
            straightened_image = np.stack(smoothed_cols, axis=1).astype(np.float32)

        straightened_mask = np.clip(straightened_mask, 0.0, 1.0).astype(np.float32)
        straightened_mask = (straightened_mask > 0.10).astype(np.float32)
        straightened_image = np.where(straightened_mask > 0.5, straightened_image, 1.0).astype(np.float32)
        straightened_image = np.clip(straightened_image, 0.0, 1.0).astype(np.float32)

        return StraightenedChromosomeImage(
            image=straightened_image,
            mask=straightened_mask,
            major_axis_angle_deg=float(angle_deg),
            foreground_area=int(aligned_mask.sum()),
            bbox_height=int(aligned_mask.shape[0]),
            bbox_width=int(aligned_mask.shape[1]),
            valid_profile_fraction=float(valid_fraction),
        )

    if method != "centerline_unfold":
        raise ValueError(f"Unsupported straightening method: {method}")

    aligned_image, aligned_mask, angle_deg = _prepare_aligned_crop(gray_image, mask)
    aligned_mask = repair_vertical_chromosome_mask(
        aligned_mask,
        expand_ratio=repair_expand_ratio,
    )

    band_width = int(output_width) if int(output_width) > 0 else max(int(aligned_mask.shape[1]), 32)

    band_image_raw, _, _, valid_fraction = build_centerline_band_image(
        gray_image=aligned_image,
        mask=aligned_mask,
        band_width=band_width,
    )
    mask_image_raw, _, _, _ = build_centerline_band_image(
        gray_image=aligned_mask.astype(np.float32),
        mask=aligned_mask,
        band_width=band_width,
    )

    straightened_image = resample_2d_height_first(band_image_raw, output_height)
    straightened_mask = resample_2d_height_first(mask_image_raw, output_height)

    if resize_mode == "fixed":
        straightened_image = _resize_gray_image(straightened_image, out_width=output_width, out_height=output_height)
        straightened_mask = _resize_gray_image(straightened_mask, out_width=output_width, out_height=output_height)
    elif resize_mode != "height_only":
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    if smooth_kernel_size > 1 and straightened_image.shape[0] >= smooth_kernel_size:
        smoothed_cols = [
            moving_average_1d(straightened_image[:, col_idx], kernel_size=smooth_kernel_size)
            for col_idx in range(straightened_image.shape[1])
        ]
        straightened_image = np.stack(smoothed_cols, axis=1).astype(np.float32)

    straightened_mask = np.clip(straightened_mask, 0.0, 1.0).astype(np.float32)
    straightened_mask = (straightened_mask > 0.10).astype(np.float32)

    # Keep the background white after unfolding to avoid introducing artificial dark bands.
    straightened_image = np.where(straightened_mask > 0.5, straightened_image, 1.0).astype(np.float32)
    straightened_image = np.clip(straightened_image, 0.0, 1.0).astype(np.float32)

    return StraightenedChromosomeImage(
        image=straightened_image,
        mask=straightened_mask,
        major_axis_angle_deg=float(angle_deg),
        foreground_area=int(aligned_mask.sum()),
        bbox_height=int(aligned_mask.shape[0]),
        bbox_width=int(aligned_mask.shape[1]),
        valid_profile_fraction=float(valid_fraction),
    )


def _extract_band_profile_v1(
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


def _extract_band_profile_v2(
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

    band_image_raw, row_values, width_values, valid_fraction = build_centerline_band_image(
        gray_image=aligned_image,
        mask=aligned_mask,
        band_width=band_width,
    )

    profile = resample_1d(row_values, profile_length)
    width_profile = resample_1d(width_values, profile_length)
    band_image = np.stack(
        [resample_1d(band_image_raw[:, col_idx], profile_length) for col_idx in range(band_image_raw.shape[1])],
        axis=1,
    ).astype(np.float32)

    profile = moving_average_1d(profile, kernel_size=5)
    width_profile = moving_average_1d(width_profile, kernel_size=5)
    if band_image.shape[0] >= 5:
        smoothed = [moving_average_1d(band_image[:, col_idx], kernel_size=5) for col_idx in range(band_image.shape[1])]
        band_image = np.stack(smoothed, axis=1).astype(np.float32)

    profile = (profile - profile.mean()) / max(float(profile.std()), 1e-6)
    if width_profile.std() > 1e-6:
        width_profile = (width_profile - width_profile.mean()) / max(float(width_profile.std()), 1e-6)
    band_image = (band_image - band_image.mean()) / max(float(band_image.std()), 1e-6)

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


def _extract_band_profile_v3(
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

    aligned_mask = refine_mask_preserve_geometry(aligned_mask, close_kernel_size=3)
    band_image_raw, density_profile_raw, width_values_raw, valid_fraction = build_skeleton_density_band_image(
        gray_image=aligned_image,
        mask=aligned_mask,
        band_width=band_width,
        width_scale=1.20,
    )
    if band_image_raw is None or density_profile_raw is None or width_values_raw is None:
        return _extract_band_profile_v2(
            gray_image=gray_image,
            mask=mask,
            profile_length=profile_length,
            band_width=band_width,
        )

    profile = resample_1d(density_profile_raw, profile_length)
    width_profile = resample_1d(width_values_raw, profile_length)
    band_image = np.stack(
        [resample_1d(band_image_raw[:, col_idx], profile_length) for col_idx in range(band_image_raw.shape[1])],
        axis=1,
    ).astype(np.float32)

    profile = nonlinear_band_enhance_1d(profile, fine_kernel=5, coarse_kernel=21)
    band_image = nonlinear_band_enhance_2d(band_image, fine_kernel=5, coarse_kernel=21)
    width_profile = moving_average_1d(
        width_profile,
        kernel_size=_safe_odd_kernel_size(width_profile.size, desired=7, minimum=3),
    )
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


def extract_band_profile(
    gray_image: np.ndarray,
    mask: np.ndarray,
    profile_length: int = 128,
    band_width: int = 32,
    version: str = "v1",
) -> ChromosomeBandRepresentation:
    if version == "v1":
        return _extract_band_profile_v1(
            gray_image=gray_image,
            mask=mask,
            profile_length=profile_length,
            band_width=band_width,
        )
    if version == "v2":
        return _extract_band_profile_v2(
            gray_image=gray_image,
            mask=mask,
            profile_length=profile_length,
            band_width=band_width,
        )
    if version == "v3":
        return _extract_band_profile_v3(
            gray_image=gray_image,
            mask=mask,
            profile_length=profile_length,
            band_width=band_width,
        )
    raise ValueError(f"Unsupported band representation version: {version}")


def flip_band_representation(repr_obj: ChromosomeBandRepresentation) -> ChromosomeBandRepresentation:
    return ChromosomeBandRepresentation(
        profile=repr_obj.profile[::-1].copy(),
        width_profile=repr_obj.width_profile[::-1].copy(),
        band_image=repr_obj.band_image[::-1].copy(),
        major_axis_angle_deg=float(repr_obj.major_axis_angle_deg),
        foreground_area=int(repr_obj.foreground_area),
        bbox_height=int(repr_obj.bbox_height),
        bbox_width=int(repr_obj.bbox_width),
        valid_profile_fraction=float(repr_obj.valid_profile_fraction),
    )


def align_pair_orientation(
    left_repr: ChromosomeBandRepresentation,
    right_repr: ChromosomeBandRepresentation,
) -> ChromosomeBandRepresentation:
    direct_corr = safe_corrcoef(left_repr.width_profile, right_repr.width_profile)
    flipped_right = flip_band_representation(right_repr)
    reverse_corr = safe_corrcoef(left_repr.width_profile, flipped_right.width_profile)
    if reverse_corr > direct_corr:
        return flipped_right
    return right_repr


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
    representation_version: str = "v1",
    pair_orientation_align: bool = False,
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
        version=representation_version,
    )
    right_repr = extract_band_profile(
        gray_image=right_gray,
        mask=right_mask,
        profile_length=profile_length,
        band_width=band_width,
        version=representation_version,
    )
    if pair_orientation_align:
        right_repr = align_pair_orientation(left_repr, right_repr)

    return extract_pair_haar_features(
        left_repr=left_repr,
        right_repr=right_repr,
        kernel_sizes=kernel_sizes,
    )


def extract_single_band_representation_from_path(
    image_path: str,
    profile_length: int = 128,
    band_width: int = 32,
    representation_version: str = "v1",
) -> ChromosomeBandRepresentation:
    gray_image = load_grayscale_image(image_path)
    mask = estimate_foreground_mask(gray_image)
    return extract_band_profile(
        gray_image=gray_image,
        mask=mask,
        profile_length=profile_length,
        band_width=band_width,
        version=representation_version,
    )


def extract_straightened_chromosome_image_from_path(
    image_path: str,
    output_height: int = 300,
    output_width: int = 96,
    smooth_kernel_size: int = 5,
    method: str = "centerline_unfold",
    global_angle_step: int = 5,
    local_angle_step: int = 5,
    seam_trim: int = 3,
    mask_mode: str = "auto",
    white_threshold: float = 254.5 / 255.0,
    repair_expand_ratio: float = 0.0,
    resize_mode: str = "fixed",
) -> StraightenedChromosomeImage:
    gray_image = load_grayscale_image(image_path)
    mask = estimate_foreground_mask(
        gray_image,
        mode=mask_mode,
        white_threshold=white_threshold,
    )
    return extract_straightened_chromosome_image(
        gray_image=gray_image,
        mask=mask,
        output_height=output_height,
        output_width=output_width,
        smooth_kernel_size=smooth_kernel_size,
        method=method,
        global_angle_step=global_angle_step,
        local_angle_step=local_angle_step,
        seam_trim=seam_trim,
        repair_expand_ratio=repair_expand_ratio,
        resize_mode=resize_mode,
    )


def extract_pair_band_representations_from_paths(
    left_path: str,
    right_path: str,
    profile_length: int = 128,
    band_width: int = 32,
    representation_version: str = "v1",
) -> Tuple[ChromosomeBandRepresentation, ChromosomeBandRepresentation]:
    left_repr = extract_single_band_representation_from_path(
        image_path=left_path,
        profile_length=profile_length,
        band_width=band_width,
        representation_version=representation_version,
    )
    right_repr = extract_single_band_representation_from_path(
        image_path=right_path,
        profile_length=profile_length,
        band_width=band_width,
        representation_version=representation_version,
    )
    return left_repr, right_repr
