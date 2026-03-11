"""Benchmark: pycocotools vs faster-coco-eval vs hotcoco on COCO val2017.

Generates deterministic synthetic detections from GT annotations (seed=42).
Only the val2017 annotation files are needed — no result files required.
See docs/getting-started/installation.md for download instructions.

Wall clock time for all three evaluation types. Optional --scale flag
multiplies detections to simulate higher load.

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

import pycocotools.mask as mask_utils
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
        "iou_type": "bbox",
    },
    {
        "name": "segm",
        "gt": DATA / "annotations/instances_val2017.json",
        "iou_type": "segm",
    },
    {
        "name": "keypoints",
        "gt": DATA / "annotations/person_keypoints_val2017.json",
        "iou_type": "keypoints",
    },
]

# ~1 detection per GT annotation in val2017 instances (~36,781 annotations).
BASE_DETS = 36_781


def generate_detections(gt_path, iou_type, n_dets, seed=42):
    """Generate deterministic synthetic detections for timing benchmarks.

    AP scores are meaningless, but detection count and format are representative.
    Fixed seed ensures identical output across runs.
    """
    rng = random.Random(seed)
    with open(gt_path) as f:
        gt = json.load(f)

    images = {img["id"]: img for img in gt["images"]}
    cat_ids = [c["id"] for c in gt["categories"]]
    img_ids = list(images.keys())

    dets = []
    for _ in range(n_dets):
        img_id = rng.choice(img_ids)
        img = images[img_id]
        w, h = img["width"], img["height"]
        cat_id = rng.choice(cat_ids)
        score = rng.uniform(0.01, 0.99)

        bw = rng.uniform(10, max(11, w * 0.5))
        bh = rng.uniform(10, max(11, h * 0.5))
        x = rng.uniform(0, max(0, w - bw))
        y = rng.uniform(0, max(0, h - bh))

        det = {"image_id": img_id, "category_id": cat_id, "score": score}

        if iou_type == "bbox":
            det["bbox"] = [x, y, bw, bh]
        elif iou_type == "segm":
            poly = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
            rle = mask_utils.frPyObjects(poly, int(h), int(w))[0]
            rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
            det["segmentation"] = rle
        elif iou_type == "keypoints":
            kpts = []
            for _ in range(17):
                kx = rng.uniform(x, x + bw)
                ky = rng.uniform(y, y + bh)
                kpts.extend([kx, ky, 2])
            det["keypoints"] = kpts
            det["bbox"] = [x, y, bw, bh]
            det["category_id"] = cat_ids[0]  # person

        dets.append(det)

    return dets


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
                   help=f"Multiply baseline detection count ({BASE_DETS:,}) by N (default: 1)")
    p.add_argument("--types", nargs="+",
                   choices=["bbox", "segm", "keypoints"],
                   default=["bbox", "segm", "keypoints"],
                   help="Eval types to run (default: all three)")
    return p.parse_args()


def main():
    args = parse_args()
    active = [b for b in BENCHMARKS if b["name"] in args.types]
    n_dets = BASE_DETS * args.scale

    scale_note = f" ({args.scale}× detections)" if args.scale > 1 else ""
    print(f"\nCOCO val2017{scale_note} — {n_dets:,} synthetic detections (seed=42)")
    print("=" * 68)
    print(f"  {'Eval Type':<12} {'pycocotools':>13} {'faster-coco-eval':>17} {'hotcoco':>9}")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as tmpdir:
        for bench in active:
            dets = generate_detections(bench["gt"], bench["iou_type"], n_dets)
            dt_path = os.path.join(tmpdir, f"{bench['name']}_dt.json")
            with open(dt_path, "w") as f:
                json.dump(dets, f)

            py_t = bench_pycocotools(bench["gt"], dt_path, bench["iou_type"])
            fc_t = bench_faster_coco_eval(bench["gt"], dt_path, bench["iou_type"])
            hc_t = bench_hotcoco(bench["gt"], dt_path, bench["iou_type"])

            fc_x = py_t / fc_t
            hc_x = py_t / hc_t
            print(f"  {bench['name']:<12} {py_t:>11.2f}s"
                  f"  {fc_t:>6.2f}s ({fc_x:.1f}×)"
                  f"  {hc_t:>6.2f}s ({hc_x:.1f}×)")

    print("=" * 68)
    print("  Speedups are relative to pycocotools.")


if __name__ == "__main__":
    main()
