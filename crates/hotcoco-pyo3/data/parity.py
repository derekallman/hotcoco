"""Parity check: hotcoco vs pycocotools on COCO val2017.

Verifies all three eval types (bbox, segm, keypoints) against pycocotools.
Tolerances match documented verified parity:
  bbox:      <= 1e-4
  segm:      <= 2e-4
  keypoints: exact (0)

Usage (from crates/hotcoco-pyo3/):
    uv run python data/parity.py
"""

import contextlib
import io
import os
import sys
from pathlib import Path

from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

from hotcoco import COCO, COCOeval

WORKSPACE = Path(__file__).resolve().parents[3]
DATA = WORKSPACE / "data"

BENCHMARKS = [
    {
        "name": "bbox",
        "gt": DATA / "annotations/instances_val2017.json",
        "dt": DATA / "bbox_val2017_results.json",
        "iou_type": "bbox",
        "tol": 1e-4,
        "metric_names": ["AP", "AP50", "AP75", "APs", "APm", "APl",
                         "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
    },
    {
        "name": "segm",
        "gt": DATA / "annotations/instances_val2017.json",
        "dt": DATA / "segm_val2017_results.json",
        "iou_type": "segm",
        "tol": 2e-4,
        "metric_names": ["AP", "AP50", "AP75", "APs", "APm", "APl",
                         "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
    },
    {
        "name": "keypoints",
        "gt": DATA / "annotations/person_keypoints_val2017.json",
        "dt": DATA / "kpt_val2017_results.json",
        "iou_type": "keypoints",
        "tol": 1e-10,
        "metric_names": ["AP", "AP50", "AP75", "APm", "APl",
                         "AR1", "AR10", "AR100", "ARm", "ARl"],
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


def run_pycocotools(gt_file, dt_file, iou_type):
    with suppress_stdout():
        gt = PyCOCO(str(gt_file))
        dt = gt.loadRes(str(dt_file))
        ev = PyCOCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats.tolist()


def run_hotcoco(gt_file, dt_file, iou_type):
    with suppress_stdout():
        gt = COCO(str(gt_file))
        dt = gt.load_res(str(dt_file))
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats


all_pass = True

for bench in BENCHMARKS:
    tol = bench["tol"]
    tol_label = f"<= {tol:.0e}" if tol > 0 else "exact"
    print(f"\n{'=' * 68}")
    print(f"  {bench['name']}  ({tol_label})")
    print(f"{'=' * 68}")
    print(f"  {'Metric':<8} {'pycocotools':>14} {'hotcoco':>14} {'diff':>12}  status")
    print(f"  {'-' * 58}")

    py = run_pycocotools(bench["gt"], bench["dt"], bench["iou_type"])
    rs = run_hotcoco(bench["gt"], bench["dt"], bench["iou_type"])

    type_pass = True
    for i, name in enumerate(bench["metric_names"]):
        diff = abs(py[i] - rs[i])
        ok = diff <= tol
        if not ok:
            type_pass = False
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<8} {py[i]:>14.8f} {rs[i]:>14.8f} {diff:>12.2e}  {status}")

    result_label = "ALL PASS" if type_pass else "SOME METRICS FAILED"
    print(f"\n  {result_label}")

print(f"\n{'=' * 68}")
if all_pass:
    print("ALL METRICS PASS")
else:
    print("PARITY FAILURES DETECTED")
    sys.exit(1)
