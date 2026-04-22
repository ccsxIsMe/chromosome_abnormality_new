"""
Chromosome Straightening Algorithm
==================================
Pipeline:
  1. Load RGB png -> grayscale -> binary mask (background is white)
  2. Keep largest connected component (remove small noise)
  3. Skeletonize -> prune to longest path (the true medial axis)
  4. Fit smooth B-spline through skeleton points
  5. Arc-length resample along spline -> get centerline points + tangents
  6. For each centerline point, sample along the perpendicular normal
     using bilinear interpolation -> one row of the straightened image
  7. Stack rows -> straightened chromosome

Author: (you)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class StraightenConfig:
    # Binarization: pixels with grayscale < threshold are considered foreground.
    # Background is near-white (255), chromosome is darker (stained).
    binary_threshold: int = 240

    # Remove connected components smaller than this (pixels)
    min_object_size: int = 50

    # B-spline smoothing factor. Higher = smoother but less faithful.
    # scipy's splprep s-parameter. Set to None to auto-pick.
    spline_smoothing: Optional[float] = None

    # Spline degree (3 = cubic, recommended)
    spline_k: int = 3

    # Number of points sampled along the centerline (= output height)
    # None = use arc-length in pixels (1 sample per pixel along the curve)
    n_centerline_samples: Optional[int] = None

    # Half-width of the normal scan line, in pixels.
    # Total output width = 2 * normal_half_width + 1
    normal_half_width: int = 40

    # Sampling step along the normal (pixels). 1.0 = one sample per pixel.
    normal_step: float = 1.0

    # Interpolation order for map_coordinates: 1 = bilinear, 3 = cubic
    interp_order: int = 1

    # Fill value for samples that fall outside the image
    fill_value: float = 255.0  # white background


# ---------------------------------------------------------------------------
# Step 1 & 2: Load + binarize + largest component
# ---------------------------------------------------------------------------
def load_and_binarize(
    image_path: str, cfg: StraightenConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (grayscale_image_float, binary_mask_bool)."""
    img = Image.open(image_path).convert("L")  # grayscale
    gray = np.asarray(img, dtype=np.float32)
    mask = gray < cfg.binary_threshold
    # skimage 0.26+ is in a transitional state where min_size is deprecated
    # but still supported; silence the warning for cleaner logs.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mask = remove_small_objects(mask, min_size=cfg.min_object_size)

    # Keep largest connected component only
    lbl = label(mask)
    if lbl.max() == 0:
        raise ValueError(f"No foreground found in {image_path}")
    regions = regionprops(lbl)
    largest = max(regions, key=lambda r: r.area)
    mask = lbl == largest.label
    return gray, mask


# ---------------------------------------------------------------------------
# Step 3: Skeleton + prune to longest path
# ---------------------------------------------------------------------------
def _skeleton_neighbors(skel: np.ndarray):
    """For each skeleton pixel, list its 8-connected skeleton neighbors."""
    ys, xs = np.nonzero(skel)
    coords = list(zip(ys.tolist(), xs.tolist()))
    coord_set = set(coords)
    neighbors = {c: [] for c in coords}
    for y, x in coords:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in coord_set:
                    neighbors[(y, x)].append(nb)
    return coords, neighbors


def _bfs_farthest(start, neighbors):
    """BFS on skeleton graph. Returns (farthest_node, parent_map)."""
    from collections import deque

    dist = {start: 0}
    parent = {start: None}
    q = deque([start])
    farthest = start
    while q:
        u = q.popleft()
        if dist[u] > dist[farthest]:
            farthest = u
        for v in neighbors[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    return farthest, parent


def longest_path_in_skeleton(skel: np.ndarray) -> np.ndarray:
    """Return the longest path in the skeleton as an ordered (N, 2) array of (y, x)."""
    coords, neighbors = _skeleton_neighbors(skel)
    if not coords:
        raise ValueError("Empty skeleton")

    # Double BFS: a classic trick for finding the diameter of a tree.
    # Works well because pruned skeletons are tree-like.
    start = coords[0]
    far1, _ = _bfs_farthest(start, neighbors)
    far2, parent = _bfs_farthest(far1, neighbors)

    # Reconstruct path far1 -> far2
    path = []
    node = far2
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return np.asarray(path, dtype=np.float64)  # (N, 2) in (y, x)


def get_centerline(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (skeleton_image, ordered_path_yx)."""
    skel = skeletonize(mask)
    path_yx = longest_path_in_skeleton(skel)
    return skel, path_yx


# ---------------------------------------------------------------------------
# Step 4 & 5: Smooth spline + arc-length resample
# ---------------------------------------------------------------------------
def fit_and_resample_centerline(
    path_yx: np.ndarray, cfg: StraightenConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a B-spline through the ordered skeleton path, then resample points
    at equal arc length.

    Returns:
        pts   : (M, 2) array of (y, x) points along the smooth centerline
        tans  : (M, 2) array of unit tangents (dy, dx)
        norms : (M, 2) array of unit normals (dy, dx) = rotate(tan, 90deg)
    """
    ys = path_yx[:, 0]
    xs = path_yx[:, 1]

    # splprep wants [x, y] list; we pass both but keep track of order.
    s = cfg.spline_smoothing
    if s is None:
        # Mild smoothing proportional to path length
        s = max(len(path_yx) * 0.5, 5.0)
    k = min(cfg.spline_k, len(path_yx) - 1)
    if k < 1:
        raise ValueError("Skeleton path too short to fit a spline")

    tck, _u = splprep([xs, ys], s=s, k=k)

    # Dense evaluation to compute arc length
    u_dense = np.linspace(0, 1, 2000)
    x_dense, y_dense = splev(u_dense, tck)
    dx = np.diff(x_dense)
    dy = np.diff(y_dense)
    seg_len = np.sqrt(dx * dx + dy * dy)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1]

    # Choose number of output samples
    if cfg.n_centerline_samples is None:
        M = int(round(total_len))
    else:
        M = cfg.n_centerline_samples
    M = max(M, 10)

    # Arc-length equispaced points: invert arc(u) to get u at each target length
    target_arcs = np.linspace(0, total_len, M)
    u_equal = np.interp(target_arcs, arc, u_dense)

    x_pts, y_pts = splev(u_equal, tck)
    dx_pts, dy_pts = splev(u_equal, tck, der=1)

    tan = np.stack([dy_pts, dx_pts], axis=1)  # (M, 2) in (y, x) order
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12

    # Normal = rotate tangent by +90 degrees in (y, x) space
    # If tangent is (ty, tx), normal is (-tx, ty)   (right-hand perpendicular)
    norm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    pts = np.stack([y_pts, x_pts], axis=1)  # (M, 2)
    return pts, tan, norm


# ---------------------------------------------------------------------------
# Step 6 & 7: Sample along normals -> straightened image
# ---------------------------------------------------------------------------
def straighten_along_normals(
    gray: np.ndarray,
    pts: np.ndarray,
    norms: np.ndarray,
    cfg: StraightenConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        straightened : (M, W) float image
        sample_ys    : (M, W) y-coords in original image of each sampled pixel
        sample_xs    : (M, W) x-coords in original image of each sampled pixel
    """
    W_half = cfg.normal_half_width
    step = cfg.normal_step
    offsets = np.arange(-W_half, W_half + 1, step)  # (W,)
    W = len(offsets)
    M = len(pts)

    # Build sampling grid: for each centerline point i and offset j,
    # sample at pts[i] + offsets[j] * norms[i]
    sample_ys = pts[:, 0:1] + offsets[None, :] * norms[:, 0:1]  # (M, W)
    sample_xs = pts[:, 1:2] + offsets[None, :] * norms[:, 1:2]  # (M, W)

    # map_coordinates wants shape (2, ...) with (row, col) order
    coords = np.stack([sample_ys, sample_xs], axis=0)  # (2, M, W)
    straightened = map_coordinates(
        gray,
        coords,
        order=cfg.interp_order,
        mode="constant",
        cval=cfg.fill_value,
    )
    return straightened, sample_ys, sample_xs


# ---------------------------------------------------------------------------
# Full pipeline bundled result
# ---------------------------------------------------------------------------
@dataclass
class StraightenResult:
    gray: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray
    raw_path_yx: np.ndarray     # skeleton longest-path, integer pixels
    centerline_pts: np.ndarray  # (M, 2) smooth spline points
    tangents: np.ndarray        # (M, 2)
    normals: np.ndarray         # (M, 2)
    sample_ys: np.ndarray       # (M, W)
    sample_xs: np.ndarray       # (M, W)
    straightened: np.ndarray    # (M, W)


def straighten_chromosome(
    image_path: str, cfg: Optional[StraightenConfig] = None
) -> StraightenResult:
    if cfg is None:
        cfg = StraightenConfig()
    gray, mask = load_and_binarize(image_path, cfg)
    skel, path_yx = get_centerline(mask)
    pts, tan, norm = fit_and_resample_centerline(path_yx, cfg)
    straight, sy, sx = straighten_along_normals(gray, pts, norm, cfg)
    return StraightenResult(
        gray=gray,
        mask=mask,
        skeleton=skel,
        raw_path_yx=path_yx,
        centerline_pts=pts,
        tangents=tan,
        normals=norm,
        sample_ys=sy,
        sample_xs=sx,
        straightened=straight,
    )
