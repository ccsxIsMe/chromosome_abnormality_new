
"""
Batch driver for chromosome straightening.

Directory layout assumed (example):
    /data5/chensx/MyProject/UAE/data/splits-case/
        train/
            <case_id>/
                normal/   *.png
                abnormal/ *.png
        val/
            ...
        test/
            ...

Output layout (mirrors the input):
    <out_root>/
        straightened/
            train/<case_id>/<normal|abnormal>/<filename>.png
        debug/
            train/<case_id>/<normal|abnormal>/<filename>.png
        failures.log

Usage:
    python run_batch.py --input-root /path/to/splits-case \
                        --output-root /path/to/out \
                        --limit 20 \
                        --splits test \
                        --centerline-method contour_pairing_v2 \
                        --save-debug \
                        --workers 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from straighten import StraightenConfig, straighten_chromosome
from visualize import save_debug_figure, save_straightened_png


def find_images(input_root: str, splits: List[str]) -> List[Tuple[str, str]]:
    results = []
    for split in splits:
        split_dir = os.path.join(input_root, split)
        if not os.path.isdir(split_dir):
            print(f"[warn] split folder missing: {split_dir}", file=sys.stderr)
            continue
        pattern = os.path.join(split_dir, "*", "*", "*.png")
        for p in sorted(glob.glob(pattern)):
            rel = os.path.relpath(p, input_root)
            results.append((p, rel))
    return results


def process_one(
    input_path: str,
    rel_path: str,
    output_root: str,
    save_debug: bool,
    cfg_dict: dict,
) -> Tuple[str, bool, str, str]:
    try:
        cfg = StraightenConfig(**cfg_dict)
        res = straighten_chromosome(input_path, cfg)
        straight_path = os.path.join(output_root, "straightened", rel_path)
        save_straightened_png(res.straightened, straight_path)
        if save_debug:
            debug_path = os.path.join(output_root, "debug", rel_path)
            save_debug_figure(res, debug_path, title=rel_path)
        return (rel_path, True, "", res.method_used)
    except Exception as e:
        return (
            rel_path,
            False,
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True,
                    help="Root folder containing train/val/test subfolders")
    ap.add_argument("--output-root", required=True,
                    help="Where to write straightened images and debug figures")
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"],
                    help="Which splits to process")
    ap.add_argument("--limit", type=int, default=20,
                    help="Max images to process. 0 or negative = all.")
    ap.add_argument("--save-debug", action="store_true",
                    help="Save multi-panel debug figures (slower).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes. 1 = serial.")

    ap.add_argument("--bg-threshold", type=int, default=250,
                    help="Background threshold: pixels >= this are treated as white background.")
    ap.add_argument("--min-object-size", type=int, default=80)
    ap.add_argument("--centerline-method", default="auto",
                    choices=[
                        "auto",
                        "contour_pairing_v2",
                        "contour_pairing",
                        "pca_slices",
                        "medial_axis_pca",
                    ],
                    help="Centerline extraction method")
    ap.add_argument("--auto-bend-ratio", type=float, default=0.18,
                    help="Kept for backward compatibility")
    ap.add_argument("--contour-n-samples", type=int, default=200)
    ap.add_argument("--contour-smooth-sigma", type=float, default=2.0)
    ap.add_argument("--contour-end-band-frac", type=float, default=0.08)
    ap.add_argument("--contour-cross-align-thresh", type=float, default=0.55)
    ap.add_argument("--endpoint-tangent-pts", type=int, default=10)
    ap.add_argument("--min-cap-points", type=int, default=8)
    ap.add_argument("--normal-half-width", type=int, default=40)
    ap.add_argument("--endpoint-extend", type=int, default=12)
    ap.add_argument("--spline-smoothing", type=float, default=-1.0,
                    help="Negative = auto")
    ap.add_argument("--spline-smoothing-scale", type=float, default=3.0,
                    help="Higher -> straighter centerline (auto smoothing only).")
    ap.add_argument("--canvas-size", type=int, default=300,
                    help="Output canvas size (square). 0 = no canvas, output raw.")
    args = ap.parse_args()

    canvas = None if args.canvas_size <= 0 else (args.canvas_size, args.canvas_size)

    cfg_dict = dict(
        bg_threshold=args.bg_threshold,
        min_object_size=args.min_object_size,
        centerline_method=args.centerline_method,
        auto_bend_ratio=args.auto_bend_ratio,
        contour_n_samples=args.contour_n_samples,
        contour_smooth_sigma=args.contour_smooth_sigma,
        contour_end_band_frac=args.contour_end_band_frac,
        contour_cross_align_thresh=args.contour_cross_align_thresh,
        endpoint_tangent_pts=args.endpoint_tangent_pts,
        min_cap_points=args.min_cap_points,
        normal_half_width=args.normal_half_width,
        endpoint_extend=args.endpoint_extend,
        spline_smoothing=(None if args.spline_smoothing < 0 else args.spline_smoothing),
        spline_smoothing_scale=args.spline_smoothing_scale,
        output_canvas_size=canvas,
    )

    from visualize import VIS_VERSION
    print(f"[info] visualize module version: {VIS_VERSION}")
    print(f"[info] centerline_method={args.centerline_method}")
    print(f"[info] scanning {args.input_root} for splits={args.splits}")
    items = find_images(args.input_root, args.splits)
    print(f"[info] found {len(items)} images total")

    if args.limit and args.limit > 0:
        items = items[: args.limit]
        print(f"[info] limiting to first {len(items)} images (testing mode)")

    os.makedirs(args.output_root, exist_ok=True)
    fail_log_path = os.path.join(args.output_root, "failures.log")
    failures = []

    if args.workers <= 1:
        for i, (inp, rel) in enumerate(items, 1):
            rel_out, ok, err, method = process_one(
                inp, rel, args.output_root, args.save_debug, cfg_dict
            )
            status = "OK  " if ok else "FAIL"
            tag = f" [{method}]" if method else ""
            print(f"[{i:>5}/{len(items)}] {status}{tag} {rel_out}")
            if not ok:
                failures.append((rel_out, err))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    process_one, inp, rel, args.output_root, args.save_debug, cfg_dict
                ): rel
                for inp, rel in items
            }
            for i, fut in enumerate(as_completed(futs), 1):
                rel_out, ok, err, method = fut.result()
                status = "OK  " if ok else "FAIL"
                tag = f" [{method}]" if method else ""
                print(f"[{i:>5}/{len(items)}] {status}{tag} {rel_out}")
                if not ok:
                    failures.append((rel_out, err))

    if failures:
        with open(fail_log_path, "w") as f:
            for rel, err in failures:
                f.write(f"=== {rel} ===\n{err}\n\n")
        print(f"[warn] {len(failures)} failures; details in {fail_log_path}")
    else:
        print("[info] all images processed successfully")


if __name__ == "__main__":
    main()
