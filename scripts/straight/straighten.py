
"""
Chromosome Straightening Algorithm (v5-debug-shoulders)
======================================
Pipeline:
  1. Load RGB png -> grayscale.
  2. Binarize: "not pure white" + hole-fill + largest CC -> full silhouette mask.
  3. Centerline:
       - contour_pairing_v2 (new): detect end-caps, pair only side walls,
         then extend centerline to the end-cap by tangent intersection.
       - contour_pairing (legacy): split contour by two tip points and pair sides.
       - pca_slices / medial_axis_pca as before.
  4. For non-contour methods, optionally extend endpoints along tangent until
     they hit the mask boundary.
  5. Fit smooth B-spline; arc-length resample.
  6. Orient so sample[0] = top of chromosome in the original image.
  7. Sample grayscale along perpendicular normals (bilinear).
  8. Paste onto 300x300 white canvas (no rescaling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev
from scipy.ndimage import (
    map_coordinates, binary_fill_holes, binary_closing, distance_transform_edt,
    gaussian_filter1d,
)
from skimage.morphology import remove_small_objects, disk, medial_axis
from skimage.measure import label, regionprops, find_contours
from skimage.filters import gaussian


@dataclass
class StraightenConfig:
    # Binarization
    bg_threshold: int = 250
    presmooth_sigma: float = 0.8
    closing_radius: int = 2
    min_object_size: int = 80

    # Centerline method:
    #   "auto"               -> try contour_pairing_v2, then contour_pairing,
    #                           then medial_axis_pca fallback.
    #   "contour_pairing_v2" -> detect end-cap segments + pair only side walls.
    #   "contour_pairing"    -> legacy tip-based contour pairing.
    #   "pca_slices"         -> PCA principal-axis slicing.
    #   "medial_axis_pca"    -> medial axis ordered by PCA.
    centerline_method: str = "auto"

    # Shared contour pairing params
    contour_n_samples: int = 200

    # Legacy contour_pairing auto threshold (kept for compatibility)
    auto_bend_ratio: float = 0.18

    # contour_pairing_v2 parameters
    contour_smooth_sigma: float = 2.0
    contour_end_band_frac: float = 0.08
    contour_cross_align_thresh: float = 0.55
    endpoint_tangent_pts: int = 10
    min_cap_points: int = 8

    # PCA slicing
    pca_n_slices: int = 120

    # B-spline
    spline_smoothing: Optional[float] = None
    spline_smoothing_scale: float = 3.0
    spline_k: int = 3

    # Endpoint extension (non contour methods only)
    endpoint_extend: int = 12

    # Normal-scan output
    n_centerline_samples: Optional[int] = None
    normal_half_width: int = 40
    normal_step: float = 1.0
    interp_order: int = 1
    fill_value: float = 255.0

    # Orientation
    orient_top_to_bottom: bool = True

    # Output canvas
    output_canvas_size: Optional[Tuple[int, int]] = (300, 300)


def load_and_binarize(image_path, cfg):
    img = Image.open(image_path).convert("L")
    gray = np.asarray(img, dtype=np.float32)
    gray_s = gaussian(gray, sigma=cfg.presmooth_sigma, preserve_range=True) \
             if cfg.presmooth_sigma > 0 else gray
    mask = gray_s < cfg.bg_threshold
    if cfg.closing_radius > 0:
        mask = binary_closing(mask, structure=disk(cfg.closing_radius))
    mask = binary_fill_holes(mask)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mask = remove_small_objects(mask, min_size=cfg.min_object_size)

    lbl = label(mask)
    if lbl.max() == 0:
        raise ValueError(f"No foreground found in {image_path}")
    largest = max(regionprops(lbl), key=lambda r: r.area)
    mask = lbl == largest.label
    mask = binary_fill_holes(mask)
    return gray, mask


def centerline_pca_slices(mask, cfg):
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        raise ValueError("Mask too small for PCA centerline")
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    v1, v2 = vt[0], vt[1]
    s = centered @ v1
    p = centered @ v2
    dt = distance_transform_edt(mask)
    weights = dt[ys, xs] + 1e-3
    s_min, s_max = s.min(), s.max()
    n = max(15, min(cfg.pca_n_slices, int(round(s_max - s_min))))
    edges = np.linspace(s_min, s_max, n + 1)
    centers = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        sel = (s >= lo) & (s <= hi)
        if sel.sum() < 2:
            continue
        p_mean = np.average(p[sel], weights=weights[sel])
        s_mid = 0.5 * (lo + hi)
        centers.append(centroid + s_mid * v1 + p_mean * v2)
    if len(centers) < 5:
        raise ValueError("PCA slicing produced too few centerline points")
    path_yx = np.asarray(centers, dtype=np.float64)
    H, W = mask.shape
    skel_like = np.zeros((H, W), dtype=bool)
    ii = np.clip(np.round(path_yx[:, 0]).astype(int), 0, H - 1)
    jj = np.clip(np.round(path_yx[:, 1]).astype(int), 0, W - 1)
    skel_like[ii, jj] = True
    return skel_like, path_yx


def centerline_contour_pairing(mask, cfg):
    """
    Legacy tip-based contour pairing.
    """
    H, W = mask.shape
    contours = find_contours(mask.astype(np.float32), level=0.5)
    if not contours:
        raise ValueError("No contour found")
    contour = max(contours, key=len)

    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    N = len(contour)
    if N < 20:
        raise ValueError("Contour too short")

    ys, xs = np.nonzero(mask)
    pts_mask = np.stack([ys, xs], axis=1).astype(np.float64)
    centroid = pts_mask.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_mask - centroid, full_matrices=False)
    v1 = vt[0]

    contour_centered = contour - centroid
    s_contour = contour_centered @ v1
    p_contour = contour_centered @ vt[1]

    s_min_val, s_max_val = s_contour.min(), s_contour.max()
    s_range = s_max_val - s_min_val
    band = max(2.0, 0.03 * s_range)

    def _tip(is_max):
        target = s_max_val if is_max else s_min_val
        cand = np.where(np.abs(s_contour - target) <= band)[0]
        if len(cand) == 0:
            return int(np.argmax(s_contour) if is_max else np.argmin(s_contour))
        best_local = int(np.argmin(np.abs(p_contour[cand])))
        return int(cand[best_local])

    tip_a = _tip(is_max=False)
    tip_b = _tip(is_max=True)

    def _arc(start, end):
        if end >= start:
            idx = np.arange(start, end + 1)
        else:
            idx = np.concatenate([np.arange(start, N), np.arange(0, end + 1)])
        return contour[idx]

    arc1 = _arc(tip_a, tip_b)
    arc2 = _arc(tip_b, tip_a)[::-1]

    def _resample(arc, n):
        diffs = np.diff(arc, axis=0)
        seg_len = np.sqrt((diffs ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        if cum[-1] < 1e-6:
            raise ValueError("Degenerate arc")
        target = np.linspace(0, cum[-1], n)
        y_interp = np.interp(target, cum, arc[:, 0])
        x_interp = np.interp(target, cum, arc[:, 1])
        return np.stack([y_interp, x_interp], axis=1)

    n = max(30, cfg.contour_n_samples)
    side1 = _resample(arc1, n)
    side2 = _resample(arc2, n)

    path_yx = 0.5 * (side1 + side2)

    skel_like = np.zeros((H, W), dtype=bool)
    ii = np.clip(np.round(path_yx[:, 0]).astype(int), 0, H - 1)
    jj = np.clip(np.round(path_yx[:, 1]).astype(int), 0, W - 1)
    skel_like[ii, jj] = True
    return (
        skel_like,
        np.asarray(path_yx, dtype=np.float64),
        side1,
        side2,
        None,  # start_cap
        None,  # end_cap
        None,  # start_shoulders
        None,  # end_shoulders
        None,  # start_pseudo_mid
        None,  # end_pseudo_mid
    )


# ---------------------------------------------------------------------------
# contour_pairing_v2 helpers (all in y,x order)
# ---------------------------------------------------------------------------
def _smooth_closed_curve_yx(contour_yx: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return contour_yx.astype(np.float64)
    y = gaussian_filter1d(contour_yx[:, 0], sigma=sigma, mode="wrap")
    x = gaussian_filter1d(contour_yx[:, 1], sigma=sigma, mode="wrap")
    return np.stack([y, x], axis=1)


def _iter_indices_circular(start: int, end: int, n: int) -> np.ndarray:
    if start <= end:
        return np.arange(start, end + 1, dtype=np.int64)
    return np.concatenate([
        np.arange(start, n, dtype=np.int64),
        np.arange(0, end + 1, dtype=np.int64),
    ])


def _run_length_circular(run: Tuple[int, int], n: int) -> int:
    a, b = run
    if a <= b:
        return int(b - a + 1)
    return int((n - a) + b + 1)


def _contiguous_runs_circular(mask_1d: np.ndarray):
    idx = np.flatnonzero(mask_1d)
    if len(idx) == 0:
        return []
    runs = []
    start = int(idx[0])
    prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if cur == prev + 1:
            prev = cur
            continue
        runs.append((start, prev))
        start = cur
        prev = cur
    runs.append((start, prev))

    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == len(mask_1d) - 1:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return runs


def _distance_to_run_circular(run: Tuple[int, int], idx: int, n: int) -> int:
    a, b = run
    if a <= b and a <= idx <= b:
        return 0
    if a > b and (idx >= a or idx <= b):
        return 0
    return int(min((idx - b) % n, (a - idx) % n))


def _contour_tangents_yx(contour_yx: np.ndarray) -> np.ndarray:
    prev_pts = np.roll(contour_yx, 1, axis=0)
    next_pts = np.roll(contour_yx, -1, axis=0)
    tangents = next_pts - prev_pts
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
    return tangents


def _sample_run_circular(points: np.ndarray, run: Tuple[int, int]) -> np.ndarray:
    idx = _iter_indices_circular(run[0], run[1], len(points))
    return points[idx]


def _resample_polyline_yx(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("empty polyline")
    if len(points) == 1:
        return np.repeat(points[:1], n, axis=0)

    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-8:
        return np.repeat(points[:1], n, axis=0)

    target = np.linspace(0.0, float(s[-1]), n)
    y = np.interp(target, s, points[:, 0])
    x = np.interp(target, s, points[:, 1])
    return np.stack([y, x], axis=1)


def _line_segment_intersection_yx(
    p_yx: np.ndarray,
    d_yx: np.ndarray,
    a_yx: np.ndarray,
    b_yx: np.ndarray,
):
    seg = b_yx - a_yx
    mat = np.array([[d_yx[0], -seg[0]], [d_yx[1], -seg[1]]], dtype=np.float64)
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-8:
        return None
    rhs = a_yx - p_yx
    t, u = np.linalg.solve(mat, rhs)
    if 0.0 <= u <= 1.0:
        return p_yx + t * d_yx, float(t), float(u)
    return None


def _intersect_line_with_polyline_yx(
    p_yx: np.ndarray,
    d_yx: np.ndarray,
    polyline_yx: np.ndarray,
    prefer_positive_t: bool = True,
):
    hits = []
    for i in range(len(polyline_yx) - 1):
        hit = _line_segment_intersection_yx(
            p_yx, d_yx, polyline_yx[i], polyline_yx[i + 1]
        )
        if hit is None:
            continue
        pt, t, _u = hit
        if prefer_positive_t:
            if t >= -1e-6:
                hits.append((pt, t))
        else:
            if t <= 1e-6:
                hits.append((pt, t))
    if not hits:
        return None
    if prefer_positive_t:
        hits.sort(key=lambda z: z[1])
    else:
        hits.sort(key=lambda z: -z[1])
    return hits[0][0]

def _fit_line_direction_pca_yx(points_yx: np.ndarray) -> np.ndarray:
    """
    Fit a local line direction from a short centerline window using PCA.
    Returns a unit direction vector in (y, x).
    """
    pts = np.asarray(points_yx, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to fit local direction")
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    d = d / (np.linalg.norm(d) + 1e-8)
    return d


def _sample_mask_value(mask: np.ndarray, q_yx: np.ndarray) -> float:
    """
    Bilinear sample on mask at subpixel y,x.
    """
    y = float(q_yx[0])
    x = float(q_yx[1])
    v = map_coordinates(
        mask.astype(np.float32),
        [[y], [x]],
        order=1,
        mode="constant",
        cval=0.0,
    )[0]
    return float(v)


def _ray_hit_mask_boundary_yx(
    start_yx: np.ndarray,
    dir_yx: np.ndarray,
    mask: np.ndarray,
    step: float = 0.5,
    max_dist: Optional[float] = None,
) -> np.ndarray:
    """
    Shoot a ray from an interior point along dir_yx until it exits the mask.
    Return the last in-mask point (refined by binary search), in y,x.

    This defines the endpoint by 'trunk direction hitting the true mask boundary',
    instead of by the end-cap contour geometry itself.
    """
    H, W = mask.shape
    d = np.asarray(dir_yx, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-8)

    p0 = np.asarray(start_yx, dtype=np.float64)
    if max_dist is None:
        max_dist = float(np.hypot(H, W) + 10.0)

    # If the start point is already outside (should not happen), return it.
    if _sample_mask_value(mask, p0) < 0.5:
        return p0

    last_inside = p0.copy()
    first_outside = None

    n_steps = int(np.ceil(max_dist / step))
    for i in range(1, n_steps + 1):
        q = p0 + d * (i * step)
        y, x = float(q[0]), float(q[1])

        # Out of image = outside mask
        if y < -1 or y > H or x < -1 or x > W:
            first_outside = q
            break

        inside = _sample_mask_value(mask, q) >= 0.5
        if inside:
            last_inside = q
        else:
            first_outside = q
            break

    if first_outside is None:
        return last_inside

    # Binary search between last_inside and first_outside
    lo = last_inside.copy()
    hi = first_outside.copy()
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        inside = _sample_mask_value(mask, mid) >= 0.5
        if inside:
            lo = mid
        else:
            hi = mid

    return lo

def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v.copy()
    return v / n


def _smooth_open_centerline_yx(points_yx: np.ndarray, smoothing_scale: float) -> np.ndarray:
    if len(points_yx) < 4:
        return points_yx
    d = np.linalg.norm(np.diff(points_yx, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < 1e-8:
        return points_yx
    try:
        tck, _u = splprep(
            [points_yx[:, 1], points_yx[:, 0]],
            u=s / total,
            s=max(0.0, smoothing_scale) * len(points_yx),
            k=min(3, len(points_yx) - 1),
        )
        uu = np.linspace(0.0, 1.0, len(points_yx))
        x_new, y_new = splev(uu, tck)
        return np.stack([y_new, x_new], axis=1)
    except Exception:
        return points_yx


def centerline_contour_pairing_v2(mask, cfg):
    """
    Improved contour pairing in y,x coordinates.

    Key idea:
      1) detect whole end-cap segments at both ends;
      2) pair only the two side-wall arcs;
      3) extend the centerline to each end-cap by tangent intersection.

    This is much more stable on slanted/oblique tips than choosing a single
    tip pixel inside an extreme projection band.
    """
    H, W = mask.shape
    contours = find_contours(mask.astype(np.float32), level=0.5)
    if not contours:
        raise ValueError("No contour found")
    contour = max(contours, key=len)
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    if len(contour) < 20:
        raise ValueError("Contour too short")

    contour_s = _smooth_closed_curve_yx(np.asarray(contour, dtype=np.float64), cfg.contour_smooth_sigma)

    ys, xs = np.nonzero(mask)
    pts_mask = np.stack([ys, xs], axis=1).astype(np.float64)
    if len(pts_mask) < 20:
        raise ValueError("Mask too small")
    centroid = pts_mask.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_mask - centroid, full_matrices=False)
    v1 = vt[0]  # principal axis in (y, x)

    contour_centered = contour_s - centroid
    s = contour_centered @ v1
    tangents = _contour_tangents_yx(contour_s)
    align = np.abs(tangents @ v1)  # side wall ~= 1, end cap ~= 0

    s_min = float(s.min())
    s_max = float(s.max())
    band = max(2.0, cfg.contour_end_band_frac * (s_max - s_min))
    n_contour = len(contour_s)

    def _select_cap(which: str):
        if which == "start":
            end_band_mask = (s <= s_min + band)
            cap_mask = end_band_mask & (align <= cfg.contour_cross_align_thresh)
            extreme_idx = int(np.argmin(s))
        else:
            end_band_mask = (s >= s_max - band)
            cap_mask = end_band_mask & (align <= cfg.contour_cross_align_thresh)
            extreme_idx = int(np.argmax(s))

        def _pick(mask_1d, min_points):
            runs = _contiguous_runs_circular(mask_1d)
            runs = [r for r in runs if _run_length_circular(r, n_contour) >= min_points]
            if not runs:
                return None
            return min(
                runs,
                key=lambda r: (
                    _distance_to_run_circular(r, extreme_idx, n_contour),
                    -_run_length_circular(r, n_contour),
                ),
            )

        # First pass: require tangent to be sufficiently cross-axis.
        run = _pick(cap_mask, cfg.min_cap_points)
        if run is not None:
            return run

        # Relaxation 1: accept shorter runs inside the same filtered mask.
        run = _pick(cap_mask, max(3, cfg.min_cap_points // 2))
        if run is not None:
            return run

        # Relaxation 2: if the end is too oblique to satisfy the tangent test,
        # fall back to the full end band and pick the run around the extreme.
        run = _pick(end_band_mask, max(3, cfg.min_cap_points // 2))
        if run is not None:
            return run

        return None

    start_cap_run = _select_cap("start")
    end_cap_run = _select_cap("end")
    if start_cap_run is None or end_cap_run is None:
        raise ValueError("Failed to detect end-cap segments")

    # Avoid accidental overlap or near overlap.
    start_cap_idx = set(_iter_indices_circular(start_cap_run[0], start_cap_run[1], n_contour).tolist())
    end_cap_idx = set(_iter_indices_circular(end_cap_run[0], end_cap_run[1], n_contour).tolist())
    if len(start_cap_idx & end_cap_idx) > 0:
        raise ValueError("Start/end end-cap segments overlap")

    start_a, start_b = start_cap_run
    end_a, end_b = end_cap_run
    arc1_idx = _iter_indices_circular(start_b, end_a, n_contour)
    arc2_idx = _iter_indices_circular(end_b, start_a, n_contour)
    arc1 = contour_s[arc1_idx]
    arc2 = contour_s[arc2_idx]

    if len(arc1) < 5 or len(arc2) < 5:
        raise ValueError("Side arc too short")

    n = max(30, cfg.contour_n_samples)
    side1 = _resample_polyline_yx(arc1, n)
    side2 = _resample_polyline_yx(arc2, n)

    # Make both sides run from low-projection end -> high-projection end.
    if (side1 @ v1)[0] > (side1 @ v1)[-1]:
        side1 = side1[::-1]
    if (side2 @ v1)[0] > (side2 @ v1)[-1]:
        side2 = side2[::-1]

        # ------------------------------------------------------------------
    # NEW endpoint strategy:
    #   1) trim a small fraction of paired side-wall samples near both ends
    #      so the centerline core is not contaminated by trapezoid / flat-end geometry;
    #   2) fit local trunk direction from the INNER centerline windows;
    #   3) extend that direction outward until it hits the true mask boundary.
    #
    # This fixes the case where a flat trapezoid top edge is geometrically
    # extreme, but the morphologically correct chromosome end lies on the
    # oblique side wall.
    # ------------------------------------------------------------------

    # Keep end-cap points only for debug / validation display
    start_cap_pts = _sample_run_circular(contour_s, start_cap_run)
    end_cap_pts = _sample_run_circular(contour_s, end_cap_run)
    if float(((start_cap_pts - centroid) @ v1).mean()) > float(((end_cap_pts - centroid) @ v1).mean()):
        start_cap_pts, end_cap_pts = end_cap_pts, start_cap_pts

    # Trim off endpoint-contaminated side-wall samples before building the core centerline
    trim = max(3, int(round(0.10 * n)))
    if len(side1) - 2 * trim >= 12 and len(side2) - 2 * trim >= 12:
        side1_core = side1[trim:-trim]
        side2_core = side2[trim:-trim]
    else:
        side1_core = side1
        side2_core = side2

    mid_core = 0.5 * (side1_core + side2_core)
    mid_core = _smooth_open_centerline_yx(mid_core, cfg.spline_smoothing_scale)

    if len(mid_core) < 8:
        raise ValueError("Centerline core too short after trimming")

    # Use INNER windows to estimate local trunk direction, not the first few
    # points which are most affected by cap / trapezoid geometry.
    fit_skip = max(2, int(round(0.05 * len(mid_core))))
    fit_len = max(6, min(cfg.endpoint_tangent_pts, len(mid_core) // 3))

    # Start-side fitting window
    a0 = min(fit_skip, len(mid_core) - 2)
    b0 = min(a0 + fit_len, len(mid_core))
    if b0 - a0 < 2:
        a0, b0 = 0, min(max(2, fit_len), len(mid_core))
    start_win = mid_core[a0:b0]

    # End-side fitting window
    b1 = max(len(mid_core) - fit_skip, 2)
    a1 = max(b1 - fit_len, 0)
    if b1 - a1 < 2:
        a1, b1 = max(len(mid_core) - max(2, fit_len), 0), len(mid_core)
    end_win = mid_core[a1:b1]

    # PCA line direction on the local windows
    u0 = _fit_line_direction_pca_yx(start_win)
    u1 = _fit_line_direction_pca_yx(end_win)

    # Fix the sign so u0/u1 point INWARD along the centerline
    inward0 = start_win[-1] - start_win[0]
    if float(np.dot(u0, inward0)) < 0:
        u0 = -u0

    inward1 = end_win[-1] - end_win[0]
    if float(np.dot(u1, inward1)) < 0:
        u1 = -u1

    # Debug shoulder points: for v2, the two ends of each detected end-cap
    # are the current approximation of the left/right bending points ("shoulders").
    # These are stored in StraightenResult so visualize.py can mark them.
    start_shoulders = None
    end_shoulders = None
    start_pseudo_mid = None
    end_pseudo_mid = None

    if start_cap_pts is not None and len(start_cap_pts) >= 2:
        start_shoulders = np.vstack([start_cap_pts[0], start_cap_pts[-1]]).astype(np.float64)
        start_pseudo_mid = start_shoulders.mean(axis=0)

    if end_cap_pts is not None and len(end_cap_pts) >= 2:
        end_shoulders = np.vstack([end_cap_pts[0], end_cap_pts[-1]]).astype(np.float64)
        end_pseudo_mid = end_shoulders.mean(axis=0)

    # Outward rays:
    #   start endpoint = shoot from core start along -u0
    #   end endpoint   = shoot from core end   along +u1
    p0 = _ray_hit_mask_boundary_yx(mid_core[0], -u0, mask)
    p1 = _ray_hit_mask_boundary_yx(mid_core[-1],  u1, mask)

    path_yx = np.vstack([p0[None, :], mid_core, p1[None, :]])

    skel_like = np.zeros((H, W), dtype=bool)
    ii = np.clip(np.round(path_yx[:, 0]).astype(int), 0, H - 1)
    jj = np.clip(np.round(path_yx[:, 1]).astype(int), 0, W - 1)
    skel_like[ii, jj] = True
    return (
        skel_like,
        np.asarray(path_yx, dtype=np.float64),
        np.asarray(side1, dtype=np.float64),
        np.asarray(side2, dtype=np.float64),
        np.asarray(start_cap_pts, dtype=np.float64),
        np.asarray(end_cap_pts, dtype=np.float64),
        start_shoulders,
        end_shoulders,
        start_pseudo_mid,
        end_pseudo_mid,
    )


def centerline_medial_axis_pca_ordered(mask, cfg):
    skel = medial_axis(mask)
    ys, xs = np.nonzero(skel)
    if len(ys) < 5:
        raise ValueError("Medial axis too short")

    mys, mxs = np.nonzero(mask)
    mpts = np.stack([mys, mxs], axis=1).astype(np.float64)
    centroid = mpts.mean(axis=0)
    _, _, vt = np.linalg.svd(mpts - centroid, full_matrices=False)
    v1 = vt[0]
    v2 = vt[1]

    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    centered = pts - centroid
    s_coord = centered @ v1
    p_coord = centered @ v2

    order = np.argsort(s_coord)
    s_sorted = s_coord[order]
    p_sorted = p_coord[order]

    s_min, s_max = s_sorted.min(), s_sorted.max()
    n_bins = max(15, min(cfg.pca_n_slices, int(round(s_max - s_min))))
    edges = np.linspace(s_min, s_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(s_sorted, edges) - 1, 0, n_bins - 1)

    path = []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() == 0:
            continue
        p_med = np.median(p_sorted[sel])
        s_mid = 0.5 * (edges[b] + edges[b + 1])
        yx = centroid + s_mid * v1 + p_med * v2
        path.append(yx)

    if len(path) < 5:
        raise ValueError("Medial-axis PCA ordering produced too few points")
    path_yx = np.asarray(path, dtype=np.float64)
    return skel, path_yx


def _contour_pairing_is_valid(path_yx, mask, side1, side2):
    H, W = mask.shape
    ii = np.clip(np.round(path_yx[:, 0]).astype(int), 0, H - 1)
    jj = np.clip(np.round(path_yx[:, 1]).astype(int), 0, W - 1)
    inside_frac = float(mask[ii, jj].mean())
    if inside_frac < 0.90:
        return False, f"only {inside_frac*100:.0f}% of midpoints inside mask"

    if len(side1) >= 3:
        tan = np.diff(path_yx, axis=0, append=path_yx[-1:])
        v1 = side1 - path_yx[:len(side1)]
        v2 = side2 - path_yx[:len(side2)]
        tan = tan[:min(len(v1), len(v2))]
        v1 = v1[:len(tan)]
        v2 = v2[:len(tan)]
        cross1 = tan[:, 1] * v1[:, 0] - tan[:, 0] * v1[:, 1]
        cross2 = tan[:, 1] * v2[:, 0] - tan[:, 0] * v2[:, 1]
        same_side_frac = float(np.mean(np.sign(cross1) == np.sign(cross2)))
        if same_side_frac > 0.15:
            return False, f"{same_side_frac*100:.0f}% of pairings folded over"

    return True, "ok"


def _contour_pairing_v2_is_valid(path_yx, mask, side1, side2, start_cap, end_cap):
    ok, reason = _contour_pairing_is_valid(path_yx, mask, side1, side2)
    if not ok:
        return ok, reason

    if start_cap is None or end_cap is None or len(start_cap) < 3 or len(end_cap) < 3:
        return False, "missing or too-short end-cap segment"

    # Ensure the new endpoint extension actually has some geometric span.
    if len(path_yx) >= 4:
        start_span = float(np.linalg.norm(path_yx[1] - path_yx[0]))
        end_span = float(np.linalg.norm(path_yx[-1] - path_yx[-2]))
        if start_span < 0.3 and end_span < 0.3:
            return False, "both end-cap intersections collapsed to interior points"

    return True, "ok"


def get_centerline(mask, cfg):
    """
    Returns:
        skel_like, path_yx, side1, side2,
        start_cap, end_cap, start_shoulders, end_shoulders,
        start_pseudo_mid, end_pseudo_mid, method_used
    """
    method = cfg.centerline_method

    empty_dbg = (None, None, None, None)

    if method == "auto":
        # 1) Try contour_pairing_v2
        try:
            skel, path, s1, s2, c0, c1, sh0, sh1, pm0, pm1 = centerline_contour_pairing_v2(mask, cfg)
            ok, _reason = _contour_pairing_v2_is_valid(path, mask, s1, s2, c0, c1)
            if ok:
                return skel, path, s1, s2, c0, c1, sh0, sh1, pm0, pm1, "contour_pairing_v2"
        except Exception:
            pass

        # 2) Try legacy contour_pairing
        try:
            skel, path, s1, s2, c0, c1, sh0, sh1, pm0, pm1 = centerline_contour_pairing(mask, cfg)
            ok, _reason = _contour_pairing_is_valid(path, mask, s1, s2)
            if ok:
                return skel, path, s1, s2, c0, c1, sh0, sh1, pm0, pm1, "contour_pairing"
        except Exception:
            pass

        # 3) Fall back
        skel, path = centerline_medial_axis_pca_ordered(mask, cfg)
        return skel, path, None, None, None, None, None, None, None, None, "medial_axis_pca"

    if method == "contour_pairing_v2":
        skel, path, s1, s2, c0, c1, sh0, sh1, pm0, pm1 = centerline_contour_pairing_v2(mask, cfg)
        return skel, path, s1, s2, c0, c1, sh0, sh1, pm0, pm1, "contour_pairing_v2"
    elif method == "contour_pairing":
        skel, path, s1, s2, c0, c1, sh0, sh1, pm0, pm1 = centerline_contour_pairing(mask, cfg)
        return skel, path, s1, s2, c0, c1, sh0, sh1, pm0, pm1, "contour_pairing"
    elif method == "pca_slices":
        skel, path = centerline_pca_slices(mask, cfg)
        return skel, path, None, None, None, None, None, None, None, None, "pca_slices"
    elif method == "medial_axis_pca":
        skel, path = centerline_medial_axis_pca_ordered(mask, cfg)
        return skel, path, None, None, None, None, None, None, None, None, "medial_axis_pca"
    else:
        raise ValueError(f"Unknown centerline_method: {method}")


def extend_path_to_boundary(path_yx, mask, extend_px):
    if extend_px <= 0 or len(path_yx) < 5:
        return path_yx
    H, W = mask.shape

    def _extend(points, at_start):
        k = min(8, len(points) - 1)
        if at_start:
            p0 = points[0]
            tangent = points[0] - points[k]
        else:
            p0 = points[-1]
            tangent = points[-1] - points[-1 - k]
        n = np.linalg.norm(tangent)
        if n < 1e-6:
            return []
        tangent = tangent / n
        added = []
        for step in range(1, extend_px + 1):
            q = p0 + tangent * step
            qi = (int(round(q[0])), int(round(q[1])))
            if not (0 <= qi[0] < H and 0 <= qi[1] < W):
                break
            if not mask[qi]:
                break
            added.append(q)
        return added

    head = _extend(path_yx, at_start=True)
    tail = _extend(path_yx, at_start=False)
    head_arr = np.array(head[::-1]) if head else np.empty((0, 2))
    tail_arr = np.array(tail) if tail else np.empty((0, 2))
    return np.concatenate([head_arr, path_yx, tail_arr], axis=0)


def fit_and_resample_centerline(path_yx, cfg):
    diffs = np.diff(path_yx, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    keep = np.concatenate([[True], seg_len > 1e-6])
    path_yx = path_yx[keep]

    ys = path_yx[:, 0]
    xs = path_yx[:, 1]
    s = cfg.spline_smoothing
    if s is None:
        s = max(len(path_yx) * cfg.spline_smoothing_scale, 10.0)
    k = min(cfg.spline_k, len(path_yx) - 1)
    if k < 1:
        raise ValueError("Centerline path too short to fit a spline")
    tck, _u = splprep([xs, ys], s=s, k=k)

    u_dense = np.linspace(0, 1, 2000)
    x_dense, y_dense = splev(u_dense, tck)
    seg_len = np.sqrt(np.diff(x_dense) ** 2 + np.diff(y_dense) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1]
    M = int(round(total_len)) if cfg.n_centerline_samples is None \
        else cfg.n_centerline_samples
    M = max(M, 10)
    target_arcs = np.linspace(0, total_len, M)
    u_equal = np.interp(target_arcs, arc, u_dense)
    x_pts, y_pts = splev(u_equal, tck)
    dx_pts, dy_pts = splev(u_equal, tck, der=1)

    pts = np.stack([y_pts, x_pts], axis=1)
    tan = np.stack([dy_pts, dx_pts], axis=1)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    norm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    if cfg.orient_top_to_bottom and pts[0, 0] > pts[-1, 0]:
        pts = pts[::-1]
        tan = -tan[::-1]
        norm = -norm[::-1]

    return pts, tan, norm


def straighten_along_normals(gray, pts, norms, cfg):
    offsets = np.arange(-cfg.normal_half_width,
                        cfg.normal_half_width + 1, cfg.normal_step)
    sample_ys = pts[:, 0:1] + offsets[None, :] * norms[:, 0:1]
    sample_xs = pts[:, 1:2] + offsets[None, :] * norms[:, 1:2]
    coords = np.stack([sample_ys, sample_xs], axis=0)
    straightened = map_coordinates(
        gray, coords, order=cfg.interp_order,
        mode="constant", cval=cfg.fill_value,
    )
    return straightened, sample_ys, sample_xs


def paste_on_white_canvas(img, canvas_size, fill=255.0):
    Hc, Wc = canvas_size
    out = np.full((Hc, Wc), fill, dtype=np.float32)
    Hi, Wi = img.shape
    src_y0 = max((Hi - Hc) // 2, 0)
    src_x0 = max((Wi - Wc) // 2, 0)
    src_h = min(Hi, Hc)
    src_w = min(Wi, Wc)
    src = img[src_y0:src_y0 + src_h, src_x0:src_x0 + src_w]
    dst_y0 = (Hc - src_h) // 2
    dst_x0 = (Wc - src_w) // 2
    out[dst_y0:dst_y0 + src_h, dst_x0:dst_x0 + src_w] = src
    return out


@dataclass
class StraightenResult:
    gray: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    raw_path_yx: np.ndarray
    extended_path_yx: np.ndarray
    centerline_pts: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    sample_ys: np.ndarray
    sample_xs: np.ndarray
    straightened_raw: np.ndarray
    straightened: np.ndarray
    contour_side1: Optional[np.ndarray] = None
    contour_side2: Optional[np.ndarray] = None
    contour_start_cap: Optional[np.ndarray] = None
    contour_end_cap: Optional[np.ndarray] = None

    # NEW debug geometry for contour_pairing_v2.
    # start/end_shoulders are the two endpoints of the detected cap segment,
    # i.e. the current candidate bending/turning points you want to inspect.
    start_shoulders: Optional[np.ndarray] = None   # shape (2, 2), yx
    end_shoulders: Optional[np.ndarray] = None     # shape (2, 2), yx
    start_pseudo_mid: Optional[np.ndarray] = None  # shape (2,), yx
    end_pseudo_mid: Optional[np.ndarray] = None    # shape (2,), yx

    method_used: str = ""


def straighten_chromosome(image_path, cfg=None):
    if cfg is None:
        cfg = StraightenConfig()
    gray, mask = load_and_binarize(image_path, cfg)

    (
        skel, path_yx, side1, side2,
        start_cap, end_cap,
        start_shoulders, end_shoulders,
        start_pseudo_mid, end_pseudo_mid,
        method_used,
    ) = get_centerline(mask, cfg)

    if method_used in ("contour_pairing", "contour_pairing_v2"):
        extended = path_yx
    else:
        extended = extend_path_to_boundary(path_yx, mask, cfg.endpoint_extend)

    pts, tan, norm = fit_and_resample_centerline(extended, cfg)
    straight_raw, sy, sx = straighten_along_normals(gray, pts, norm, cfg)
    if cfg.output_canvas_size is not None:
        final = paste_on_white_canvas(
            straight_raw, cfg.output_canvas_size, fill=cfg.fill_value
        )
    else:
        final = straight_raw

    return StraightenResult(
        gray=gray, mask=mask, skeleton=skel,
        raw_path_yx=path_yx, extended_path_yx=extended,
        centerline_pts=pts, tangents=tan, normals=norm,
        sample_ys=sy, sample_xs=sx,
        straightened_raw=straight_raw, straightened=final,
        contour_side1=side1, contour_side2=side2,
        contour_start_cap=start_cap, contour_end_cap=end_cap,
        start_shoulders=start_shoulders, end_shoulders=end_shoulders,
        start_pseudo_mid=start_pseudo_mid, end_pseudo_mid=end_pseudo_mid,
        method_used=method_used,
    )
