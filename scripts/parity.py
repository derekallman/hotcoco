"""Parity check: hotcoco vs pycocotools on COCO val2017.

Uses third-party published fake result files from ppwwyyxx/cocoapi.
See docs/getting-started/installation.md for download instructions.

Tolerances match documented verified parity:
  bbox:      <= 1e-4
  segm:      <= 2e-4
  keypoints: exact (0)

Usage:
    uv run python scripts/parity.py
    just parity
"""

import sys

from helpers import DATA_DIR, suppress_stdout
from hotcoco import COCO, COCOeval
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

BENCHMARKS = [
    {
        "name": "bbox",
        "gt": DATA_DIR / "annotations/instances_val2017.json",
        "dt": DATA_DIR / "bbox_val2017_results.json",
        "iou_type": "bbox",
        "tol": 1e-4,
    },
    {
        "name": "segm",
        "gt": DATA_DIR / "annotations/instances_val2017.json",
        "dt": DATA_DIR / "segm_val2017_results.json",
        "iou_type": "segm",
        "tol": 2e-4,
    },
    {
        "name": "keypoints",
        "gt": DATA_DIR / "annotations/person_keypoints_val2017.json",
        "dt": DATA_DIR / "kpt_val2017_results.json",
        "iou_type": "keypoints",
        "tol": 1e-10,
    },
]


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
    return ev.stats, ev.metric_keys()


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
    rs, metric_names = run_hotcoco(bench["gt"], bench["dt"], bench["iou_type"])

    type_pass = True
    for i, name in enumerate(metric_names):
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
