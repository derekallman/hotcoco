"""Benchmark faster-coco-eval vs hotcoco on Objects365 val.

Generates synthetic detections (capped at MAX_DET_PER_IMAGE per image)
from the ground truth annotations and writes them to a temp file for reuse.

pycocotools is excluded (DNF at O365 scale — needs a beefier machine).

Usage (from crates/hotcoco-pyo3/):
    uv run python data/bench_objects365.py [--gt PATH] [--max-det N]

Defaults:
    --gt      <workspace>/data/annotations/zhiyuan_objv2_val.json
    --max-det 100
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

_WORKSPACE = Path(__file__).resolve().parents[3]
_DATA = _WORKSPACE / "data"
RUST_BIN = str(_WORKSPACE / "target/release/coco-eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gt",
        default=str(_DATA / "annotations/zhiyuan_objv2_val.json"),
        help="Path to Objects365 val annotation JSON",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Max synthesized detections per image (default: 100)",
    )
    p.add_argument(
        "--dt",
        default=None,
        help="Pre-generated detections JSON (skip generation if provided)",
    )
    return p.parse_args()


def generate_detections(gt_path, max_det_per_image, out_path):
    """Generate noisy synthetic detections from GT, capped per image."""
    print(f"Loading GT from {gt_path} ...")
    with open(gt_path) as f:
        gt = json.load(f)

    rng = random.Random(42)

    # Group annotations by image_id
    by_image: dict[int, list] = {}
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        by_image.setdefault(ann["image_id"], []).append(ann)

    results = []
    for img_id, anns in by_image.items():
        # Shuffle so the cap is random, not biased toward first categories
        rng.shuffle(anns)
        for ann in anns[:max_det_per_image]:
            x, y, w, h = ann["bbox"]
            noisy = [
                x + rng.gauss(0, 2),
                y + rng.gauss(0, 2),
                max(1.0, w + rng.gauss(0, 3)),
                max(1.0, h + rng.gauss(0, 3)),
            ]
            results.append(
                {
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [round(v, 2) for v in noisy],
                    "score": round(min(1.0, max(0.01, rng.gauss(0.7, 0.2))), 4),
                }
            )

    print(
        f"Generated {len(results):,} detections across {len(by_image):,} images "
        f"(cap={max_det_per_image}/image)"
    )
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved detections to {out_path}")
    return out_path


def peak_rss_thread(proc, result):
    """Poll a subprocess's RSS in a background thread; store peak in result[0]."""
    peak = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                mem = p.memory_info().rss
                if mem > peak:
                    peak = mem
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        pass
    result[0] = peak


_FASTER_RUNNER = """
import sys, time
from faster_coco_eval import COCO, COCOeval_faster
gt_file, dt_file = sys.argv[1], sys.argv[2]
gt = COCO(gt_file)
dt = gt.loadRes(dt_file)
ev = COCOeval_faster(gt, dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
"""


def bench_faster_coco_eval(gt_file, dt_file):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_FASTER_RUNNER)
        runner = f.name

    try:
        proc = subprocess.Popen(
            [sys.executable, runner, gt_file, dt_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        peak_result = [0]
        monitor = threading.Thread(target=peak_rss_thread, args=(proc, peak_result))
        monitor.start()

        t0 = time.perf_counter()
        stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - t0
        monitor.join()
    finally:
        os.unlink(runner)

    if proc.returncode != 0:
        print(f"  faster-coco-eval failed:\n{stderr}", file=sys.stderr)
        return None, None

    return elapsed, peak_result[0]


def bench_hotcoco(gt_file, dt_file):
    if not os.path.exists(RUST_BIN):
        print(f"  hotcoco binary not found at {RUST_BIN} — skipping")
        print("  Run: cargo build -p hotcoco-cli --release")
        return None, None

    proc = subprocess.Popen(
        [RUST_BIN, "--gt", gt_file, "--dt", dt_file, "--iou-type", "bbox"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    peak_result = [0]
    monitor = threading.Thread(target=peak_rss_thread, args=(proc, peak_result))
    monitor.start()

    t0 = time.perf_counter()
    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - t0
    monitor.join()

    if proc.returncode != 0:
        print(f"  hotcoco failed:\n{stderr}", file=sys.stderr)
        return None, None

    return elapsed, peak_result[0]


def fmt_time(t):
    if t is None:
        return "     N/A"
    return f"{t:>7.2f}s"


def fmt_mem(b):
    if b is None or b <= 0:
        return "     N/A"
    return f"{b / 1e9:>6.2f} GB"


def fmt_speedup(base, t):
    if t is None or base is None:
        return "   N/A"
    return f"{base / t:>5.1f}x"


def main():
    args = parse_args()

    if not os.path.exists(args.gt):
        print(f"ERROR: GT file not found: {args.gt}", file=sys.stderr)
        sys.exit(1)

    # Generate or use existing detections
    if args.dt:
        dt_file = args.dt
        print(f"Using pre-generated detections: {dt_file}")
    else:
        dt_path = str(_DATA / f"objects365_val_synth_det_{args.max_det}per.json")
        if os.path.exists(dt_path):
            print(f"Using cached detections: {dt_path}")
            dt_file = dt_path
        else:
            dt_file = generate_detections(args.gt, args.max_det, dt_path)

    print()
    print("=" * 65)
    print("Objects365 val — bbox evaluation")
    print("=" * 65)

    fc_time, fc_mem = None, None
    hc_time, hc_mem = None, None

    print("Running hotcoco ...", flush=True)
    hc_time, hc_mem = bench_hotcoco(args.gt, dt_file)
    if hc_time is not None:
        print(f"  done in {hc_time:.2f}s, peak RSS {fmt_mem(hc_mem)}")

    print("Running faster-coco-eval ...", flush=True)
    try:
        fc_time, fc_mem = bench_faster_coco_eval(args.gt, dt_file)
        print(f"  done in {fc_time:.2f}s, peak RSS {fmt_mem(fc_mem)}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print()
    print("=" * 65)
    print(f"{'Library':<20} {'Time':>8}  {'Peak RAM':>8}  {'vs faster-coco-eval':>19}")
    print("-" * 65)
    print(f"{'faster-coco-eval':<20} {fmt_time(fc_time)}  {fmt_mem(fc_mem)}  {'(baseline)':>19}")
    print(f"{'hotcoco':<20} {fmt_time(hc_time)}  {fmt_mem(hc_mem)}  {fmt_speedup(fc_time, hc_time):>19}")
    print("=" * 65)
    print("(pycocotools: DNF at O365 scale)")


if __name__ == "__main__":
    main()
