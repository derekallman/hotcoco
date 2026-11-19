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

`--workload arrays` times the other way results reach an evaluator: columns
already in memory, as a training loop holds them. It is the shape RF-DETR's
`OnePassCocoMeanAveragePrecision` hands its backend at `compute()` — val2017's
boxes as ground truth, 300 detections per image (1.5M), `maxDets` [1, 10, 500],
TorchMetrics' float32 threshold grids — with each library called the way that
adapter calls it. No JSON is read, so "load" there is the cost of turning
columns into the library's own dataset.

Usage:
    uv run python scripts/bench.py
    uv run python scripts/bench.py --scale 10
    uv run python scripts/bench.py --types bbox segm
    uv run python scripts/bench.py --impls pycocotools hotcoco --reps 1
    uv run python scripts/bench.py --workload arrays --impls vernier hotcoco
    uv run python scripts/bench.py --workload arrays --threads 1
    just bench
    just bench-arrays
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
from concurrent.futures import ProcessPoolExecutor

import numpy as np
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

# The arrays workload: who is timed, and its fixed shape.
ARRAY_IMPLS = ["pycocotools", "vernier", "hotcoco"]
ARRAY_PER_IMAGE = 300
ARRAY_MAX_DETS = [1, 10, 500]
# TorchMetrics builds its grids in float32; the values below are those, widened.
ARRAY_IOU_THRS = np.linspace(0.5, 0.95, 10, dtype=np.float32).astype(np.float64)
ARRAY_REC_THRS = np.linspace(0.0, 1.0, 101, dtype=np.float32).astype(np.float64)


def _threads():
    """Thread budget for a cell: `--threads`, else every core."""
    return int(os.environ.get("BENCH_THREADS") or os.cpu_count())


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


def write_detections(src_gt, iou_type, n_dets, out_path):
    """`generate_detections` written to `out_path` as JSON; returns the path.

    Top-level so a process pool can run one per eval type. `json.dumps` runs
    the C encoder and `json.dump` the pure-Python one: same bytes, and the
    write is 1.8× faster (10× keypoints, 310 MB, takes 6 s instead of 10).
    """
    dets = generate_detections(src_gt, iou_type, n_dets)
    with open(out_path, "w") as f:
        f.write(json.dumps(dets))
    return out_path


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
        f.write(json.dumps(gt))


def generate_array_state(gt_path, out_path, seed=0):
    """Write the arrays workload's columns to an `.npz`.

    Ground truth is val2017's boxes with images and categories renumbered from
    0, the way a metric that never saw the JSON holds them. Each image gets
    `ARRAY_PER_IMAGE` detections: a jittered copy of about 70% of its ground
    truth under the true label, then low-scoring random boxes, so matching does
    real work and AP is neither 0 nor 1. Boxes are float32 `xywh`, as stored
    metric state is.
    """
    with open(gt_path) as f:
        gt = json.load(f)
    cat_index = {c: i for i, c in enumerate(sorted(c["id"] for c in gt["categories"]))}
    images = sorted(gt["images"], key=lambda im: im["id"])
    img_index = {im["id"]: i for i, im in enumerate(images)}
    per_img = [[] for _ in images]
    for a in gt["annotations"]:
        per_img[img_index[a["image_id"]]].append(a)

    rng = np.random.default_rng(seed)
    gt_boxes, gt_labels, gt_crowd, gt_area, gt_counts = [], [], [], [], []
    det_boxes, det_scores, det_labels = [], [], []
    for im, anns in zip(images, per_img):
        w, h = im["width"], im["height"]
        for a in anns:
            gt_boxes.append(a["bbox"])
            gt_labels.append(cat_index[a["category_id"]])
            gt_crowd.append(a["iscrowd"])
            gt_area.append(a["area"])
        gt_counts.append(len(anns))

        boxes, scores, labels = [], [], []
        for a in anns:
            if rng.random() < 0.7 and len(boxes) < ARRAY_PER_IMAGE:
                x, y, bw, bh = a["bbox"]
                j = rng.normal(0, 0.1, 4) * [bw, bh, bw, bh]
                boxes.append([x + j[0], y + j[1], max(1.0, bw + j[2]), max(1.0, bh + j[3])])
                scores.append(rng.uniform(0.3, 1.0))
                labels.append(cat_index[a["category_id"]])
        k = ARRAY_PER_IMAGE - len(boxes)
        bw = rng.uniform(4, w / 2, k)
        bh = rng.uniform(4, h / 2, k)
        bx = rng.uniform(0, w - bw)
        by = rng.uniform(0, h - bh)
        det_boxes += boxes + np.stack([bx, by, bw, bh], axis=1).tolist()
        det_scores += scores + rng.uniform(0.0, 0.4, k).tolist()
        det_labels += labels + rng.integers(0, len(cat_index), k).tolist()

    np.savez(
        out_path,
        det_boxes=np.asarray(det_boxes, np.float32),
        det_scores=np.asarray(det_scores, np.float32),
        det_labels=np.asarray(det_labels, np.int64),
        det_counts=np.full(len(images), ARRAY_PER_IMAGE, np.int64),
        gt_boxes=np.asarray(gt_boxes, np.float32),
        gt_labels=np.asarray(gt_labels, np.int64),
        gt_crowd=np.asarray(gt_crowd, np.int64),
        gt_area=np.asarray(gt_area, np.float32),
        gt_counts=np.asarray(gt_counts, np.int64),
        n_classes=np.int64(len(cat_index)),
    )


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
        # vernier evaluates on one thread unless told otherwise, while every
        # other library here uses all cores by default. Give it the same budget
        # (RF-DETR passes torch.get_num_threads() for the same reason); without
        # this, tables up to 2026-10-07 compared one-thread vernier to 8-thread
        # rivals.
        summary = Evaluator(iou=kind, parity_mode="strict").evaluate(gt, dt, num_threads=_threads())
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


def _detection_rows(s):
    """Detections as the `(N, 7)` float64 rows `loadRes` takes:
    image, x, y, w, h, score, category."""
    image = np.repeat(np.arange(len(s["det_counts"])), s["det_counts"])
    return np.concatenate(
        [
            image.astype(np.float64)[:, None],
            s["det_boxes"].astype(np.float64),
            s["det_scores"].astype(np.float64)[:, None],
            s["det_labels"].astype(np.float64)[:, None],
        ],
        axis=1,
    )


def _run_array_cell(impl, state_path):
    """Time the arrays workload: load (columns to datasets) and eval.

    Returns `(load_seconds, eval_seconds, ap)`. Reading the `.npz` and importing
    the library sit outside both spans: a training loop has done both already.
    hotcoco and vernier are called as RF-DETR 1.12's adapter calls them;
    pycocotools as TorchMetrics does, a dict dataset plus `createIndex()`.
    """
    with np.load(state_path) as z:
        s = {k: z[k] for k in z.files}
    n_images = len(s["det_counts"])
    classes = list(range(int(s["n_classes"])))
    gt_boxes = s["gt_boxes"].astype(np.float64)
    gt_area = np.where(s["gt_area"] > 0, s["gt_area"].astype(np.float64), gt_boxes[:, 2] * gt_boxes[:, 3])
    gt_image = np.repeat(np.arange(n_images), s["gt_counts"])

    if impl == "pycocotools":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        t0 = time.perf_counter()
        gt = COCO()
        gt.dataset = {
            "images": [{"id": i} for i in range(n_images)],
            "categories": [{"id": c, "name": str(c)} for c in classes],
            "annotations": [
                {"id": k + 1, "image_id": i, "category_id": c, "bbox": b, "area": a, "iscrowd": crowd}
                for k, (i, c, b, a, crowd) in enumerate(
                    zip(
                        gt_image.tolist(),
                        s["gt_labels"].tolist(),
                        gt_boxes.tolist(),
                        gt_area.tolist(),
                        s["gt_crowd"].tolist(),
                    )
                )
            ],
        }
        gt.createIndex()
        dt = gt.loadRes(_detection_rows(s))
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.iouThrs = ARRAY_IOU_THRS
        ev.params.recThrs = ARRAY_REC_THRS
        ev.params.maxDets = ARRAY_MAX_DETS
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        t_eval = time.perf_counter() - t0
        # This pycocotools reads AP at maxDets 100, which is not in the list, so
        # stats[0] is -1. Take it from the precision array at the last cap, all
        # areas, as the other two report it.
        precision = ev.eval["precision"][:, :, :, 0, -1]
        return t_load, t_eval, float(precision[precision > -1].mean())

    if impl == "vernier":
        import vernier

        t0 = time.perf_counter()
        det = {"boxes": s["det_boxes"], "scores": s["det_scores"], "labels": s["det_labels"], "counts": s["det_counts"]}
        tgt = {
            "boxes": s["gt_boxes"],
            "labels": s["gt_labels"],
            "iscrowd": s["gt_crowd"],
            "area": s["gt_area"],
            "counts": s["gt_counts"],
        }
        gt, dt = vernier.adapters.coco_inputs_from_columns(det, tgt, box_format="xywh", categories=classes, area="auto")
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        grid = vernier.instance.evaluate_bbox_grid(
            gt,
            dt,
            parity_mode="corrected",
            max_dets_per_image=ARRAY_MAX_DETS[-1],
            use_cats=True,
            iou_thresholds=ARRAY_IOU_THRS.tolist(),
            recall_thresholds=ARRAY_REC_THRS.tolist(),
            num_threads=_threads(),
            dt_area="bbox",
        )
        stats = grid.accumulate(ARRAY_MAX_DETS).summarize().stats
        return t_load, time.perf_counter() - t0, float(stats[0])

    if impl == "hotcoco":
        from hotcoco import COCO, COCOeval

        t0 = time.perf_counter()
        gt = COCO.from_arrays(
            [{"id": i} for i in range(n_images)],
            [{"id": c, "name": str(c)} for c in classes],
            gt_image,
            s["gt_labels"],
            gt_boxes,
            area=gt_area,
            iscrowd=s["gt_crowd"],
        )
        dt = gt.load_res(_detection_rows(s))
        # RF-DETR's adapter reads this; it once built every record (see
        # tests/test_live_dataset.py), so it stays inside the timed span.
        len(dt.imgs)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = COCOeval(gt, dt, "bbox")
        p = ev.params
        p.iouThrs = ARRAY_IOU_THRS
        p.recThrs = ARRAY_REC_THRS
        p.maxDets = ARRAY_MAX_DETS
        p.catIds = classes
        ev.params = p
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return t_load, time.perf_counter() - t0, float(ev.stats[0])

    raise SystemExit(f"{impl!r} has no arrays workload")


def _child_main(impl, iou_type, gt_path, dt_path):
    with suppress_output(stderr=False):
        if iou_type == "arrays":
            t_load, t_eval, ap = _run_array_cell(impl, gt_path)
        else:
            t_load, t_eval, ap = _run_cell(impl, iou_type, gt_path, dt_path)
    print(json.dumps({"load": t_load, "eval": t_eval, "rss_mb": _peak_rss_mb(), "ap": ap}))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _installed(impl):
    return importlib.util.find_spec(IMPORT_NAME[impl]) is not None


# Wall time spent per library across the whole run, startup and imports
# included, and time spent generating inputs. Printed at the end.
WALL = {}
SETUP = [0.0]


def run_cells(impls, iou_type, gt_path, dt_path, reps):
    """Median load/eval/peak-RSS per library over `reps` fresh processes each;
    `None` for a library that failed.

    Reps are interleaved — every library once, then every library again — so a
    slow spell on the machine lands on every column, not on one library's reps.
    """
    runs = {impl: [] for impl in impls}
    for _ in range(reps):
        for impl in impls:
            if runs[impl] is None:
                continue
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, __file__, "--_child", impl, iou_type, str(gt_path), str(dt_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            WALL[impl] = WALL.get(impl, 0.0) + time.perf_counter() - t0
            if proc.returncode != 0:
                print(f"  {impl} {iou_type}: failed\n{proc.stderr[-600:]}", file=sys.stderr)
                runs[impl] = None
                continue
            runs[impl].append(json.loads(proc.stdout.strip().splitlines()[-1]))
    out = {}
    for impl, rs in runs.items():
        if rs is None:
            out[impl] = None
            continue
        out[impl] = {k: statistics.median(r[k] for r in rs) for k in ("load", "eval", "rss_mb")}
        out[impl]["ap"] = rs[0]["ap"]
    return out


def _print_wall():
    """Where the run's own time went: slow setup shows up here."""
    parts = [f"generating inputs {SETUP[0]:.0f}s"] + [f"{i} {t:.0f}s" for i, t in WALL.items()]
    print(f"\n  Wall time: {', '.join(parts)}")


def _fmt_time(result, baseline, phase=None, digits=2):
    if result is None:
        return "-"
    t = result[phase] if phase else result["load"] + result["eval"]
    if baseline is None:
        return f"{t:.{digits}f}s"
    b = baseline[phase] if phase else baseline["load"] + baseline["eval"]
    return f"{t:.{digits}f}s ({b / t:.1f}×)"


def _print_ap_spread(results, name, impls, indent):
    """Say so when the libraries disagree on AP for a row."""
    aps = {i: results[(name, i)]["ap"] for i in impls if results[(name, i)] is not None}
    spread = max(aps.values()) - min(aps.values()) if aps else 0.0
    if spread > 1e-6:
        print(f"  {'':<{indent}}  AP disagrees across libraries by {spread:.2e}: {aps}")


def main_arrays(args):
    """The `--workload arrays` table: load, eval, and total per library."""
    name = "bbox (arrays)"
    impls = [i for i in args.impls if i in ARRAY_IMPLS and _installed(i)]
    skipped = [i for i in args.impls if i not in impls]
    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        state = os.path.join(tmpdir, "state.npz")
        t0 = time.perf_counter()
        generate_array_state(VAL2017["bbox"]["gt"], state)
        SETUP[0] += time.perf_counter() - t0
        with np.load(state) as z:
            n_dets, n_gt, n_images = len(z["det_scores"]), len(z["gt_labels"]), len(z["det_counts"])
        print(f"\nCOCO val2017 boxes as arrays — {n_images:,} images, {n_gt:,} GT, {n_dets:,} detections (seed=0)")
        print(f"maxDets {ARRAY_MAX_DETS}, float32 threshold grids, {_threads()} thread(s).")
        print(f"Per-cell median of {args.reps} fresh-process run(s); speedups relative to pycocotools when it runs.")
        if skipped:
            print(f"Not run (not installed, or no arrays workload): {', '.join(skipped)}")
        for impl, r in run_cells(impls, "arrays", state, "-", args.reps).items():
            results[(name, impl)] = r

    header = f"  {'Phase':<8}" + "".join(f"{i:>18}" for i in impls)
    rule = "=" * len(header)
    baseline = results.get((name, "pycocotools"))
    print(rule)
    print(header)
    print(rule)
    for label, phase in (("load", "load"), ("eval", "eval"), ("total", None)):
        print(f"  {label:<8}" + "".join(f"{_fmt_time(results[(name, i)], baseline, phase, 3):>18}" for i in impls))
    _print_ap_spread(results, name, impls, 8)
    print(rule)
    cells = "".join(
        f"{results[(name, i)]['rss_mb']:>17.0f} " if results[(name, i)] is not None else f"{'-':>18}" for i in impls
    )
    print(f"  {'peak MB':<8}{cells}")
    print(rule)
    _print_wall()


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
    p.add_argument(
        "--workload",
        choices=["files", "arrays"],
        default="files",
        help="files: JSON ground truth and results (default). arrays: the RF-DETR "
        "compute() shape, boxes passed as in-memory columns; --scale, --types, and "
        "--phases do not apply",
    )
    p.add_argument(
        "--threads", type=int, metavar="N", help="Thread budget for every library that takes one (default: all cores)"
    )
    p.add_argument("--_child", nargs=4, metavar=("IMPL", "IOU_TYPE", "GT", "DT"), help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args._child:
        _child_main(*args._child)
        return
    if args.threads:
        # Children inherit both: rayon reads the first (hotcoco, ultrafast),
        # `_threads` the second (vernier takes its budget as an argument).
        os.environ["RAYON_NUM_THREADS"] = os.environ["BENCH_THREADS"] = str(args.threads)
    if args.workload == "arrays":
        main_arrays(args)
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

        # Detections are generated from the *official* GT so the bbox-only-GT
        # variant times the same workload on a lighter file. Every row's file
        # is written up front in worker processes, before any cell runs, so
        # generation never shares the machine with a timed run and the driver
        # holds no detections while cells run. At 1× the eval types generate
        # in parallel (about 100 MB each). Above 1× they take turns: at 10× they
        # peak at 0.4, 1.0, and 1.1 GB, which together is 2.5 GB on an 8 GB
        # machine, enough pressure to slow the timed cells that follow.
        jobs = {}
        for name, gt_path, iou_type in rows:
            src_gt = VAL2017["bbox"]["gt"] if name.startswith("bbox") else gt_path
            jobs.setdefault((src_gt, iou_type), os.path.join(tmpdir, f"dt_{iou_type}.json"))
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=len(jobs) if args.scale == 1 else 1) as pool:
            futures = [pool.submit(write_detections, src, t, n_dets, path) for (src, t), path in jobs.items()]
            for future in futures:
                future.result()
        SETUP[0] += time.perf_counter() - t0

        for name, gt_path, iou_type in rows:
            src_gt = VAL2017["bbox"]["gt"] if name.startswith("bbox") else gt_path
            dt_path = jobs[(src_gt, iou_type)]
            for impl, r in run_cells(impls, iou_type, gt_path, dt_path, args.reps).items():
                results[(name, impl)] = r
            baseline = results.get((name, "pycocotools"))

            if args.phases:
                for phase in ("load", "eval"):
                    label = name if phase == "load" else ""
                    cells = "".join(f"{_fmt_time(results[(name, i)], baseline, phase):>18}" for i in impls)
                    print(f"  {label:<{width}}  {phase:<6}{cells}")
            else:
                cells = "".join(f"{_fmt_time(results[(name, i)], baseline):>18}" for i in impls)
                print(f"  {name:<{width}}{cells}")

            _print_ap_spread(results, name, impls, width)

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
    _print_wall()


if __name__ == "__main__":
    main()
