"""Benchmark: pycocotools, faster-coco-eval, ultrafast-pycocotools, vernier, and
hotcoco on COCO val2017.

Generates deterministic synthetic detections from GT annotations (seed=42).
Only the val2017 annotation files are needed — no result files required.
See docs/getting-started/installation.md for download instructions.

Every (library, eval type) cell runs in a fresh subprocess, so each reports its
own wall-clock time and peak resident memory with nothing left over from the
library before it. Cells run `--reps` times and the table shows the per-cell
median. ultrafast-pycocotools and vernier are optional: a missing one leaves its
column blank.

Usage:
    uv run python scripts/bench.py
    uv run python scripts/bench.py --scale 10
    uv run python scripts/bench.py --types bbox segm
    uv run python scripts/bench.py --impls pycocotools hotcoco --reps 1
    just bench
"""

import argparse
import importlib.util
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import pycocotools.mask as mask_utils
from helpers import VAL2017, suppress_output

# Detections are synthesized from the GT, so only the annotation files in
# `helpers.VAL2017` are used — the result files are not read here.

# ~1 detection per GT annotation in val2017 instances (~36,781 annotations).
BASE_DETS = 36_781

# Column order. hotcoco last so the eye lands on it; the three others are the
# baselines it is measured against.
IMPLS = ["pycocotools", "faster-coco-eval", "ultrafast", "vernier", "hotcoco"]
IMPORT_NAME = {
    "pycocotools": "pycocotools",
    "faster-coco-eval": "faster_coco_eval",
    "ultrafast": "ultrafast_pycocotools",
    "vernier": "vernier",
    "hotcoco": "hotcoco",
}


def generate_detections(gt_path, iou_type, n_dets, seed=42):
    """Generate deterministic synthetic detections for timing benchmarks.

    AP scores are meaningless, but detection count and format are representative.
    Fixed seed ensures identical output across runs. Every detection carries a
    `bbox`, as detector output does: pycocotools derives it from the mask for
    segm results anyway, and vernier's native reader requires it.
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

        det = {"image_id": img_id, "category_id": cat_id, "score": score, "bbox": [x, y, bw, bh]}

        if iou_type == "segm":
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
            det["category_id"] = cat_ids[0]  # person

        dets.append(det)

    return dets


def strip_segmentation(gt_path, out_path):
    """Write a copy of a GT file with every `segmentation` field removed.

    The official instances files carry a polygon on every annotation — about
    two-thirds of the file bytes — which bbox evaluation never reads. This
    variant represents datasets that never had masks (custom bbox datasets,
    YOLO conversions, Objects365).
    """
    with open(gt_path) as f:
        gt = json.load(f)
    for ann in gt["annotations"]:
        ann.pop("segmentation", None)
    with open(out_path, "w") as f:
        json.dump(gt, f)


# ---------------------------------------------------------------------------
# Child process: one (library, eval type) cell
# ---------------------------------------------------------------------------


def _peak_rss_mb():
    """Peak resident set of this process, in MB (macOS reports bytes, Linux KB)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def _run_cell(impl, iou_type, gt_path, dt_path):
    """Time load (ctor + loadRes) and eval (evaluate + accumulate + summarize).

    Returns `(load_seconds, eval_seconds, ap)`. Each library is driven the way
    its own README benchmarks it; vernier's native `Evaluator` is used rather
    than its pycocotools shim, which would time pycocotools' loader instead.
    """
    if impl == "pycocotools":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        t0 = time.perf_counter()
        gt = COCO(gt_path)
        dt = gt.loadRes(dt_path)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return t_load, time.perf_counter() - t0, float(ev.stats[0])

    if impl == "faster-coco-eval":
        from faster_coco_eval import COCO, COCOeval_faster

        t0 = time.perf_counter()
        gt = COCO(gt_path)
        dt = gt.loadRes(dt_path)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = COCOeval_faster(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return t_load, time.perf_counter() - t0, float(ev.stats[0])

    if impl == "ultrafast":
        from ultrafast_pycocotools import COCO, COCOeval

        t0 = time.perf_counter()
        gt = COCO(gt_path)
        dt = gt.loadRes(dt_path)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return t_load, time.perf_counter() - t0, float(ev.stats[0])

    if impl == "vernier":
        from vernier.instance import Bbox, CocoDataset, Evaluator, Keypoints, Segm

        kind = {"bbox": Bbox, "segm": Segm, "keypoints": Keypoints}[iou_type]()
        t0 = time.perf_counter()
        with open(gt_path, "rb") as f:
            gt = CocoDataset.from_json(f.read())
        with open(dt_path, "rb") as f:
            dt = f.read()
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        summary = Evaluator(iou=kind, parity_mode="strict").evaluate(gt, dt)
        return t_load, time.perf_counter() - t0, float(summary.stats[0])

    if impl == "hotcoco":
        from hotcoco import COCO, COCOeval

        t0 = time.perf_counter()
        gt = COCO(gt_path)
        dt = gt.load_res(dt_path)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return t_load, time.perf_counter() - t0, float(ev.stats[0])

    raise SystemExit(f"unknown impl {impl!r}")


def _child_main(impl, iou_type, gt_path, dt_path):
    with suppress_output(stderr=False):
        t_load, t_eval, ap = _run_cell(impl, iou_type, gt_path, dt_path)
    print(json.dumps({"load": t_load, "eval": t_eval, "rss_mb": _peak_rss_mb(), "ap": ap}))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _installed(impl):
    return importlib.util.find_spec(IMPORT_NAME[impl]) is not None


def run_cell(impl, iou_type, gt_path, dt_path, reps):
    """Median load/eval/peak-RSS over `reps` fresh processes; `None` if the
    library is not installed."""
    if not _installed(impl):
        return None
    runs = []
    for _ in range(reps):
        proc = subprocess.run(
            [sys.executable, __file__, "--_child", impl, iou_type, str(gt_path), str(dt_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(f"  {impl} {iou_type}: failed\n{proc.stderr[-600:]}", file=sys.stderr)
            return None
        runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    med = {k: statistics.median(r[k] for r in runs) for k in ("load", "eval", "rss_mb")}
    med["ap"] = runs[0]["ap"]
    return med


def _fmt_time(result, baseline, phase=None):
    if result is None:
        return "-"
    t = result[phase] if phase else result["load"] + result["eval"]
    if baseline is None:
        return f"{t:.2f}s"
    b = baseline[phase] if phase else baseline["load"] + baseline["eval"]
    return f"{t:.2f}s ({b / t:.1f}×)"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scale",
        type=int,
        default=1,
        metavar="N",
        help=f"Multiply baseline detection count ({BASE_DETS:,}) by N (default: 1)",
    )
    p.add_argument(
        "--types",
        nargs="+",
        choices=["bbox", "segm", "keypoints"],
        default=["bbox", "segm", "keypoints"],
        help="Eval types to run (default: all three)",
    )
    p.add_argument(
        "--impls",
        nargs="+",
        choices=IMPLS,
        default=IMPLS,
        help="Libraries to run (default: all five; a library that is not installed is skipped)",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=3,
        metavar="N",
        help="Fresh-process runs per cell; the table shows the median (default: 3)",
    )
    p.add_argument(
        "--phases",
        action="store_true",
        help="Split each result into load (ctor + loadRes) and eval "
        "(evaluate + accumulate + summarize), and add a bbox-only-GT variant "
        "row (segmentation fields stripped) when bbox is among the types",
    )
    p.add_argument("--_child", nargs=4, metavar=("IMPL", "IOU_TYPE", "GT", "DT"), help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args._child:
        _child_main(*args._child)
        return

    n_dets = BASE_DETS * args.scale
    impls = [i for i in args.impls if _installed(i)]
    missing = [i for i in args.impls if not _installed(i)]

    scale_note = f" ({args.scale}× detections)" if args.scale > 1 else ""
    print(f"\nCOCO val2017{scale_note} — {n_dets:,} synthetic detections (seed=42)")
    print(f"Per-cell median of {args.reps} fresh-process run(s); speedups relative to pycocotools.")
    if missing:
        print(f"Not installed, skipped: {', '.join(missing)}")
    width = 22 if args.phases else 12
    header = f"  {'Eval Type':<{width}}" + ("  Phase " if args.phases else "") + "".join(f"{i:>18}" for i in impls)
    rule = "=" * len(header)
    print(rule)
    print(header)
    print(rule)

    results = {}  # (row, impl) -> median dict
    with tempfile.TemporaryDirectory() as tmpdir:
        # (row label, GT file, iou_type). The label and the iou_type coincide for
        # the three real types and diverge only for the stripped-GT variant below.
        rows = [(iou_type, files["gt"], iou_type) for iou_type, files in VAL2017.items() if iou_type in args.types]
        if args.phases and "bbox" in args.types:
            # Same detections, GT without its (unread) polygon masks.
            stripped = os.path.join(tmpdir, "gt_bbox_only.json")
            strip_segmentation(VAL2017["bbox"]["gt"], stripped)
            rows.append(("bbox (bbox-only GT)", stripped, "bbox"))

        for name, gt_path, iou_type in rows:
            # Detections are generated from the *official* GT so the
            # bbox-only-GT variant times the same workload on a lighter file.
            src_gt = VAL2017["bbox"]["gt"] if name.startswith("bbox") else gt_path
            dets = generate_detections(src_gt, iou_type, n_dets)
            dt_path = os.path.join(tmpdir, "dt.json")
            with open(dt_path, "w") as f:
                json.dump(dets, f)

            for impl in impls:
                results[(name, impl)] = run_cell(impl, iou_type, gt_path, dt_path, args.reps)
            baseline = results.get((name, "pycocotools"))

            if args.phases:
                for phase in ("load", "eval"):
                    label = name if phase == "load" else ""
                    cells = "".join(f"{_fmt_time(results[(name, i)], baseline, phase):>18}" for i in impls)
                    print(f"  {label:<{width}}  {phase:<6}{cells}")
            else:
                cells = "".join(f"{_fmt_time(results[(name, i)], baseline):>18}" for i in impls)
                print(f"  {name:<{width}}{cells}")

            aps = {i: r["ap"] for i, r in ((i, results[(name, i)]) for i in impls) if r is not None}
            spread = max(aps.values()) - min(aps.values()) if aps else 0.0
            if spread > 1e-6:
                print(f"  {'':<{width}}  AP disagrees across libraries by {spread:.2e}: {aps}")

    print(rule)
    print("\n  Peak resident memory (MB), same runs:")
    print(rule)
    print(header if not args.phases else f"  {'Eval Type':<{width}}" + "".join(f"{i:>18}" for i in impls))
    print(rule)
    for name, _, _ in rows:
        cells = "".join(
            f"{results[(name, i)]['rss_mb']:>17.0f} " if results[(name, i)] is not None else f"{'-':>18}" for i in impls
        )
        print(f"  {name:<{width}}{cells}")
    print(rule)


if __name__ == "__main__":
    main()
