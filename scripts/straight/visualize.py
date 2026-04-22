"""
Visualization / debug plots for the straightening pipeline.
Saves a multi-panel figure showing every intermediate step.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from straighten import StraightenResult


def save_debug_figure(
    res: StraightenResult,
    out_path: str,
    title: Optional[str] = None,
    normal_stride: int = 10,
) -> None:
    """
    Saves a 2x3 figure:
        [original]           [binary mask]        [skeleton + longest path]
        [smooth centerline]  [normals overlay]    [straightened result]
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # (0, 0) original grayscale
    ax = axes[0, 0]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title("1. Original (grayscale)")
    ax.axis("off")

    # (0, 1) binary mask
    ax = axes[0, 1]
    ax.imshow(res.mask, cmap="gray")
    ax.set_title("2. Binary mask (largest CC)")
    ax.axis("off")

    # (0, 2) skeleton + raw longest path
    ax = axes[0, 2]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    # overlay skeleton in red
    ys, xs = np.nonzero(res.skeleton)
    ax.scatter(xs, ys, s=1, c="red", alpha=0.4, label="skeleton")
    # overlay longest path in yellow
    ax.plot(
        res.raw_path_yx[:, 1], res.raw_path_yx[:, 0],
        color="yellow", linewidth=1.2, label="longest path"
    )
    ax.set_title("3. Skeleton + longest path")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")

    # (1, 0) smooth spline centerline
    ax = axes[1, 0]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    ax.plot(
        res.centerline_pts[:, 1], res.centerline_pts[:, 0],
        color="lime", linewidth=1.5, label="smooth centerline"
    )
    ax.scatter(
        res.centerline_pts[0, 1], res.centerline_pts[0, 0],
        c="cyan", s=30, label="start"
    )
    ax.scatter(
        res.centerline_pts[-1, 1], res.centerline_pts[-1, 0],
        c="magenta", s=30, label="end"
    )
    ax.set_title("4. Smooth B-spline centerline")
    ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")

    # (1, 1) normals overlay (sparse, otherwise unreadable)
    ax = axes[1, 1]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    ax.plot(
        res.centerline_pts[:, 1], res.centerline_pts[:, 0],
        color="lime", linewidth=1.0, alpha=0.8,
    )
    # draw every Nth normal as a line segment
    pts = res.centerline_pts
    norms = res.normals
    # infer half-width from sampled grid
    W = res.sample_xs.shape[1]
    half = (W - 1) / 2.0
    for i in range(0, len(pts), max(1, normal_stride)):
        p = pts[i]
        n = norms[i]
        p0 = p - half * n
        p1 = p + half * n
        ax.plot([p0[1], p1[1]], [p0[0], p1[0]],
                color="orange", linewidth=0.8, alpha=0.7)
    ax.set_title(f"5. Normals (every {normal_stride}th)")
    ax.axis("off")

    # (1, 2) straightened result
    ax = axes[1, 2]
    ax.imshow(res.straightened, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.set_title(f"6. Straightened ({res.straightened.shape[0]}x{res.straightened.shape[1]})")
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_straightened_png(straightened: np.ndarray, out_path: str) -> None:
    """Save just the straightened image as an 8-bit PNG."""
    from PIL import Image
    arr = np.clip(straightened, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(arr).save(out_path)
