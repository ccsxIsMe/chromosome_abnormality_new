"""
Visualization / debug plots for the straightening pipeline (v4).
"""
from __future__ import annotations
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from straighten import StraightenResult

# Version banner so we can tell at a glance whether new code is running.
VIS_VERSION = "v5-debug-shoulders"


def save_debug_figure(
    res: StraightenResult,
    out_path: str,
    title: Optional[str] = None,
    normal_stride: int = 3,
) -> None:
    """
    2x3 grid:
      [1 original]        [2 filled mask]          [3 PCA centerline points]
      [4 smooth spline]   [5 normals overlay]      [6 final straightened]
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. original grayscale
    ax = axes[0, 0]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title("1. Original (grayscale)")
    ax.axis("off")

    # 2. mask
    ax = axes[0, 1]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    ax.contour(res.mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)
    ax.imshow(
        np.ma.masked_where(~res.mask, res.mask.astype(float)),
        cmap="autumn", alpha=0.25,
    )
    ax.set_title("2. Mask (holes filled)")
    ax.axis("off")

    # 3. Centerline extraction detail
    ax = axes[0, 2]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)

    if res.contour_side1 is not None and res.contour_side2 is not None:
        # Contour-pairing visualization: show both sides + pairing lines
        s1 = res.contour_side1
        s2 = res.contour_side2
        ax.plot(s1[:, 1], s1[:, 0], color="blue", linewidth=1.2,
                label=f"contour side A ({len(s1)})")
        ax.plot(s2[:, 1], s2[:, 0], color="orange", linewidth=1.2,
                label=f"contour side B ({len(s2)})")
        # Draw a few pairing lines to show opposite-side matching
        stride = max(1, len(s1) // 15)
        for i in range(0, len(s1), stride):
            ax.plot([s1[i, 1], s2[i, 1]], [s1[i, 0], s2[i, 0]],
                    color="gray", linewidth=0.4, alpha=0.6)
        ax.scatter(
            res.raw_path_yx[:, 1], res.raw_path_yx[:, 0],
            s=6, c="red", zorder=5,
            label=f"midpoints = centerline ({len(res.raw_path_yx)})",
        )

        # v2 debug geometry: cap segments + shoulder/corner candidates.
        # Use getattr so this visualize.py remains compatible with older StraightenResult.
        start_cap = getattr(res, "contour_start_cap", None)
        end_cap = getattr(res, "contour_end_cap", None)
        start_shoulders = getattr(res, "start_shoulders", None)
        end_shoulders = getattr(res, "end_shoulders", None)
        start_pseudo_mid = getattr(res, "start_pseudo_mid", None)
        end_pseudo_mid = getattr(res, "end_pseudo_mid", None)

        if start_cap is not None and len(start_cap) > 0:
            ax.plot(
                start_cap[:, 1], start_cap[:, 0],
                color="purple", linewidth=2.0, alpha=0.95,
                label=f"start cap ({len(start_cap)})",
            )
        if end_cap is not None and len(end_cap) > 0:
            ax.plot(
                end_cap[:, 1], end_cap[:, 0],
                color="brown", linewidth=2.0, alpha=0.95,
                label=f"end cap ({len(end_cap)})",
            )
        if start_shoulders is not None and len(start_shoulders) == 2:
            ax.scatter(
                start_shoulders[:, 1], start_shoulders[:, 0],
                s=85, c="yellow", edgecolor="black", linewidths=1.0,
                zorder=8, label="start shoulders/corners",
            )
            ax.plot(
                start_shoulders[:, 1], start_shoulders[:, 0],
                color="yellow", linewidth=1.5, alpha=0.9, zorder=7,
            )
        if end_shoulders is not None and len(end_shoulders) == 2:
            ax.scatter(
                end_shoulders[:, 1], end_shoulders[:, 0],
                s=85, c="lime", edgecolor="black", linewidths=1.0,
                zorder=8, label="end shoulders/corners",
            )
            ax.plot(
                end_shoulders[:, 1], end_shoulders[:, 0],
                color="lime", linewidth=1.5, alpha=0.9, zorder=7,
            )
        if start_pseudo_mid is not None:
            ax.scatter(
                start_pseudo_mid[1], start_pseudo_mid[0],
                s=120, c="cyan", marker="x", linewidths=2.5,
                zorder=9, label="start pseudo-mid",
            )
        if end_pseudo_mid is not None:
            ax.scatter(
                end_pseudo_mid[1], end_pseudo_mid[0],
                s=120, c="magenta", marker="x", linewidths=2.5,
                zorder=9, label="end pseudo-mid",
            )

        ax.set_title("3. Contour pairing + cap/shoulder debug")
    else:
        ax.scatter(
            res.raw_path_yx[:, 1], res.raw_path_yx[:, 0],
            s=10, c="red", label=f"centerline points ({len(res.raw_path_yx)})",
        )
        if len(res.extended_path_yx) > len(res.raw_path_yx):
            raw_set = {tuple(p.round(4)) for p in res.raw_path_yx}
            ext_only = np.array([p for p in res.extended_path_yx
                                 if tuple(p.round(4)) not in raw_set])
            if len(ext_only):
                ax.scatter(ext_only[:, 1], ext_only[:, 0],
                           s=14, c="deepskyblue",
                           label=f"endpoint extension ({len(ext_only)})")
        ax.set_title(f"3. Centerline points ({res.method_used})")
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")

    # 4. smooth spline + TOP indicator
    ax = axes[1, 0]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    ax.plot(
        res.centerline_pts[:, 1], res.centerline_pts[:, 0],
        color="lime", linewidth=1.8, label=f"smooth centerline ({len(res.centerline_pts)} pts)",
    )
    # First point (index 0) should be TOP; last point BOTTOM
    ax.scatter(res.centerline_pts[0, 1], res.centerline_pts[0, 0],
               c="cyan", s=60, edgecolor="black", zorder=5,
               label="index 0 = TOP of straightened image")
    ax.scatter(res.centerline_pts[-1, 1], res.centerline_pts[-1, 0],
               c="magenta", s=60, edgecolor="black", zorder=5,
               label="last = BOTTOM")
    ax.set_title("4. Smooth B-spline centerline")
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")

    # 5. normals (dense by default)
    ax = axes[1, 1]
    ax.imshow(res.gray, cmap="gray", vmin=0, vmax=255)
    ax.plot(
        res.centerline_pts[:, 1], res.centerline_pts[:, 0],
        color="lime", linewidth=1.0, alpha=0.9,
    )
    pts = res.centerline_pts
    norms = res.normals
    W = res.sample_xs.shape[1]
    half = (W - 1) / 2.0
    total_pts = len(pts)
    drawn = 0
    for i in range(0, total_pts, max(1, normal_stride)):
        p = pts[i]; n = norms[i]
        p0 = p - half * n; p1 = p + half * n
        ax.plot([p0[1], p1[1]], [p0[0], p1[0]],
                color="orange", linewidth=0.5, alpha=0.6)
        drawn += 1
    ax.set_title(
        f"5. Normals: {drawn} drawn / {total_pts} total "
        f"(stride={normal_stride})"
    )
    ax.axis("off")

    # 6. final output
    ax = axes[1, 2]
    ax.imshow(res.straightened, cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        f"6. Straightened (canvas {res.straightened.shape[0]}x{res.straightened.shape[1]}; "
        f"raw {res.straightened_raw.shape[0]}x{res.straightened_raw.shape[1]})"
    )
    ax.axis("off")

    # Build a clear suptitle that ALWAYS includes the method and version,
    # so the user can tell at a glance whether new code is running.
    suptitle_bits = [f"[{VIS_VERSION}] method={res.method_used}"]
    if title:
        suptitle_bits.insert(0, title)
    fig.suptitle(" | ".join(suptitle_bits), fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_straightened_png(straightened: np.ndarray, out_path: str) -> None:
    from PIL import Image
    arr = np.clip(straightened, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(arr).save(out_path)