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
            train/<case_id>/<normal|abnormal>/<filename>.png   (multi-panel debug figure)
        failures.log   (list of images that failed, with error messages)

Usage:
    python run_batch.py --input-root /path/to/splits-case \
                        --output-root /path/to/out \
                        --limit 20            # only process first 20 images (testing)
                        --limit 0             # 0 or negative = process all
                        --splits test         # or: train val test (space separated)
                        --save-debug          # save multi-panel debug figures
                        --workers 1           # parallel workers
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


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def find_images(input_root: str, splits: List[str]) -> List[Tuple[str, str]]:
    """
    Returns a list of (abs_input_path, relative_path_from_input_root).
    We only look for .png files under <split>/<case>/<normal|abnormal>/.
    """
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


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def process_one(
    input_path: str,
    rel_path: str,
    output_root: str,
    save_debug: bool,
    cfg_dict: dict,
) -> Tuple[str, bool, str]:
    """
    Process a single image. Returns (rel_path, success, error_message).
    cfg_dict is passed instead of a dataclass for pickle-friendliness.
    """
    try:
        cfg = StraightenConfig(**cfg_dict)
        res = straighten_chromosome(input_path, cfg)

        straight_path = os.path.join(output_root, "straightened", rel_path)
        save_straightened_png(res.straightened, straight_path)

        if save_debug:
            debug_path = os.path.join(output_root, "debug", rel_path)
            save_debug_figure(res, debug_path, title=rel_path)

        return (rel_path, True, "")
    except Exception as e:
        return (rel_path, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    # Algorithm parameters (expose the common ones)
    ap.add_argument("--binary-threshold", type=int, default=240)
    ap.add_argument("--min-object-size", type=int, default=50)
    ap.add_argument("--normal-half-width", type=int, default=40)
    ap.add_argument("--spline-smoothing", type=float, default=-1.0,
                    help="Negative = auto")
    args = ap.parse_args()

    cfg_dict = dict(
        binary_threshold=args.binary_threshold,
        min_object_size=args.min_object_size,
        normal_half_width=args.normal_half_width,
        spline_smoothing=(None if args.spline_smoothing < 0 else args.spline_smoothing),
    )

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
        # Serial: easier to debug
        for i, (inp, rel) in enumerate(items, 1):
            rel_out, ok, err = process_one(
                inp, rel, args.output_root, args.save_debug, cfg_dict
            )
            status = "OK  " if ok else "FAIL"
            print(f"[{i:>5}/{len(items)}] {status} {rel_out}")
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
                rel_out, ok, err = fut.result()
                status = "OK  " if ok else "FAIL"
                print(f"[{i:>5}/{len(items)}] {status} {rel_out}")
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
