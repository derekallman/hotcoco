"""Benchmark: pycocotools vs faster-coco-eval vs hotcoco on COCO val2017.

Wall clock time for all three evaluation types. Optional --scale flag
multiplies detections to simulate higher load (replaces bench_10x.py).

Usage (from crates/hotcoco-pyo3/):
    uv run python data/bench.py
    uv run python data/bench.py --scale 10
    uv run python data/bench.py --types bbox segm
"""

import argparse
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval
from faster_coco_eval import COCO as FcCOCO, COCOeval_faster
from hotcoco import COCO, COCOeval

WORKSPACE = Path(__file__).resolve().parents[3]
DATA = WORKSPACE / "data"

BENCHMARKS = [
    {
        "name": "bbox",
        "gt": DATA / "annotations/instances_val2017.json",
        "dt": DATA / "bbox_val2017_results.json",
        "iou_type": "bbox",
    },
    {
        "name": "segm",
        "gt": DATA / "annotations/instances_val2017.json",
        "dt": DATA / "segm_val2017_results.json",
        "iou_type": "segm",
    },
    {
        "name": "keypoints",
        "gt": DATA / "annotations/person_keypoints_val2017.json",
        "dt": DATA / "kpt_val2017_results.json",
        "iou_type": "keypoints",
    },
]


@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout at the file-descriptor level (catches Rust println! too)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    old_sys = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        os.dup2(old_fd, 1)
        os.close(old_fd)
        os.close(devnull_fd)
        sys.stdout = old_sys


def scale_detections(dets, scale):
    """Duplicate detections with small score/bbox jitter to simulate higher load."""
    if scale == 1:
        return dets
    rng = random.Random(42)
    result = list(dets)
    for i in range(1, scale):
        for det in dets:
            d = dict(det)
            d["score"] = max(0.001, det.get("score", 0.5) - i * 0.01)
            if "bbox" in det:
                b = det["bbox"]
                d["bbox"] = [b[0] + i * 0.5, b[1] + i * 0.5, b[2], b[3]]
            result.append(d)
    return result


def bench_pycocotools(gt_file, dt_file, iou_type):
    t0 = time.perf_counter()
    with suppress_stdout():
        gt = PyCOCO(str(gt_file))
        dt = gt.loadRes(str(dt_file))
        ev = PyCOCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return time.perf_counter() - t0


def bench_faster_coco_eval(gt_file, dt_file, iou_type):
    t0 = time.perf_counter()
    with suppress_stdout():
        gt = FcCOCO(str(gt_file))
        dt = gt.loadRes(str(dt_file))
        ev = COCOeval_faster(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return time.perf_counter() - t0


def bench_hotcoco(gt_file, dt_file, iou_type):
    t0 = time.perf_counter()
    with suppress_stdout():
        gt = COCO(str(gt_file))
        dt = gt.load_res(str(dt_file))
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return time.perf_counter() - t0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", type=int, default=1, metavar="N",
                   help="Multiply detections N times to simulate higher load (default: 1)")
    p.add_argument("--types", nargs="+",
                   choices=["bbox", "segm", "keypoints"],
                   default=["bbox", "segm", "keypoints"],
                   help="Eval types to run (default: all three)")
    return p.parse_args()


def main():
    args = parse_args()
    active = [b for b in BENCHMARKS if b["name"] in args.types]

    scale_note = f" ({args.scale}x detections)" if args.scale > 1 else ""
    print(f"\nCOCO val2017{scale_note}")
    print("=" * 68)
    print(f"  {'Eval Type':<12} {'pycocotools':>13} {'faster-coco-eval':>17} {'hotcoco':>9}")
    print("=" * 68)

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for bench in active:
            with open(bench["dt"]) as f:
                dets = json.load(f)
            dets = scale_detections(dets, args.scale)

            dt_path = os.path.join(tmpdir, f"{bench['name']}_dt.json")
            with open(dt_path, "w") as f:
                json.dump(dets, f)

            py_t = bench_pycocotools(bench["gt"], dt_path, bench["iou_type"])
            fc_t = bench_faster_coco_eval(bench["gt"], dt_path, bench["iou_type"])
            hc_t = bench_hotcoco(bench["gt"], dt_path, bench["iou_type"])
            results.append((bench["name"], py_t, fc_t, hc_t))

            fc_x = py_t / fc_t
            hc_x = py_t / hc_t
            print(f"  {bench['name']:<12} {py_t:>11.2f}s"
                  f"  {fc_t:>6.2f}s ({fc_x:.1f}x)"
                  f"  {hc_t:>6.2f}s ({hc_x:.1f}x)")

    print("=" * 68)
    print("  Speedups are relative to pycocotools.")


if __name__ == "__main__":
    main()
