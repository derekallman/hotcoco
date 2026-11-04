"""Shared helpers for the dev scripts here and the tests in `tests/`.

`pythonpath` in `pyproject.toml` puts this directory on the path for pytest, so
both import `helpers` by the same name and there is one copy of these.
"""

import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import tempfile
import types
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "data"
# Tracked test oracles: the adversarial corpus, `val2017_expected.json`,
# `oid_tf_expected.json`. The `gen_*` scripts write here; the tests read here.
FIXTURES_DIR = WORKSPACE / "tests" / "fixtures"

# The val2017 ground-truth / result pairs every real-data script runs against,
# keyed by iou_type. Absolute paths derived from DATA_DIR: a script that spells
# these as relative strings works from the repo root and nowhere else, which is
# the failure DATA_DIR exists to prevent.
#
# `data/` is gitignored, so these files may not exist — callers that can survive
# a missing dataset should check `.exists()` rather than assume.
VAL2017 = {
    "bbox": {"gt": DATA_DIR / "annotations/instances_val2017.json", "dt": DATA_DIR / "bbox_val2017_results.json"},
    "segm": {"gt": DATA_DIR / "annotations/instances_val2017.json", "dt": DATA_DIR / "segm_val2017_results.json"},
    "keypoints": {
        "gt": DATA_DIR / "annotations/person_keypoints_val2017.json",
        "dt": DATA_DIR / "kpt_val2017_results.json",
    },
}

# COCO panoptic val2017: the JSON, its PNG folder (panopticapi's convention is
# the JSON path minus `.json`), and the perturbed predictions
# `scripts/download_panoptic.py` writes beside them. Same gitignored `data/`.
PANOPTIC_VAL2017 = {
    "gt": DATA_DIR / "annotations/panoptic_val2017.json",
    "gt_folder": DATA_DIR / "annotations/panoptic_val2017",
    "dt": DATA_DIR / "panoptic_val2017_results.json",
    "dt_folder": DATA_DIR / "panoptic_val2017_results",
}

# ---------------------------------------------------------------------------
# COCO keypoint constants
# ---------------------------------------------------------------------------

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

# ---------------------------------------------------------------------------
# stdout suppression
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def suppress_output(*, stderr: bool = True):
    """Suppress stdout — and stderr unless ``stderr=False`` — at the fd level.

    Redirecting the file descriptors, not just `sys.stdout`, is what catches Rust
    `println!`. fd 1 alone is enough for pycocotools' chatter; it is not enough for
    hotcoco, whose `summarize()` writes comparability warnings to fd 2, deliberately
    bypassing `sys.stderr` so they survive redirection. A script that evaluates in a
    loop — `tests/test_parity_oid.py` runs 70 cases — otherwise buries its own result under one
    warning per case.
    """
    fds = (1, 2) if stderr else (1,)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(fd) for fd in fds]
    for fd in fds:
        os.dup2(devnull_fd, fd)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    if stderr:
        sys.stderr = io.StringIO()
    try:
        yield
    finally:
        for fd, saved_fd in zip(fds, saved):
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(devnull_fd)
        sys.stdout, sys.stderr = old_out, old_err


# ---------------------------------------------------------------------------
# Temp-JSON round-trip
# ---------------------------------------------------------------------------


def torchmetrics_dataset(predictions, *, seed=7, canvas=32, origin=20, box=(4, 12), half_mask=False, bytes_counts=True):
    """A dataset shaped the way torchmetrics' ``_get_coco_format`` builds one for
    a bbox+segm run: ``area_bbox`` and ``area_segm`` on every annotation, an RLE
    ``segmentation``, and ``score`` on predictions. Six ``canvas``-square images
    with four boxes each, origins below ``origin``, sides in ``box``, categories
    1-3. ``bytes_counts`` keeps ``counts`` as the bytes ``mask.encode`` returns
    with ``size`` a tuple, as torchmetrics does; ``half_mask`` fills the left half
    of each box only, so the two areas land in different size ranges.
    ``predictions`` jitters the boxes so matches land at several IoUs."""
    from hotcoco import mask

    rng = np.random.default_rng(seed)
    images, anns = [], []
    for image_id in range(6):
        images.append({"id": image_id, "width": canvas, "height": canvas})
        for _ in range(4):
            x, y = (int(v) for v in rng.integers(0, origin, size=2))
            w, h = (int(v) for v in rng.integers(*box, size=2))
            if predictions:
                x, y = x + int(rng.integers(0, 3)), y + int(rng.integers(0, 3))
            m = np.zeros((canvas, canvas), np.uint8)
            m[y : y + h, x : x + (w // 2 if half_mask else w)] = 1
            rle = mask.encode(np.asfortranarray(m))
            assert isinstance(rle["counts"], bytes)
            bbox = [float(x), float(y), float(w), float(h)]
            ann = {
                "id": len(anns) + 1,
                "image_id": image_id,
                "area": int(mask.area(rle)),
                "category_id": 1 + len(anns) % 3,
                "iscrowd": 0,
                "area_bbox": bbox[2] * bbox[3],
                "area_segm": int(mask.area(rle)),
                "bbox": bbox,
                "segmentation": {"size": tuple(rle["size"]), "counts": rle["counts"]}
                if bytes_counts
                else {"size": list(rle["size"]), "counts": rle["counts"].decode()},
            }
            if predictions:
                ann["score"] = float(rng.random())
            anns.append(ann)
    categories = [{"id": c, "name": str(c)} for c in (1, 2, 3)]
    return {"images": images, "annotations": anns, "categories": categories}


def rect_polygon(x, y, w, h):
    """The `[x, y, w, h]` box as a one-ring COCO polygon."""
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


@contextlib.contextmanager
def written_json(*objects, quiet: bool = False):
    """Write each object to its own temp ``.json`` file and yield their paths.

    Always yields a tuple, one path per object, so the two-file GT/DT case reads
    ``as (gt_path, dt_path)`` and the single-file case ``as (path,)``. Files are
    removed on the way out even if the body raises.

    ``quiet=True`` also suppresses stdout for the duration of the body, which is
    what a differential run wants: `COCO(path)` chatters on both sides.
    """
    paths = []
    try:
        for obj in objects:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(obj, f)
                paths.append(f.name)
        if quiet:
            with suppress_output(stderr=False):
                yield tuple(paths)
        else:
            yield tuple(paths)
    finally:
        for path in paths:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Differential parity scaffold
# ---------------------------------------------------------------------------

# Floating-point noise only. Individual callers pass a looser `tolerance` when
# they are comparing against a reference that genuinely diverges (see
# tests/test_parity_lvis.py and adversarial_harness.py) — that per-script sizing is
# deliberate and lives at the call site.
TOLERANCE = 1e-10


class MetricMismatch(NamedTuple):
    """One metric where the two implementations disagree beyond tolerance."""

    index: int
    name: str
    py: float
    rs: float
    diff: float

    def line(self) -> str:
        return f"  [{self.index}] {self.name}: py={self.py:.15f} rs={self.rs:.15f} diff={self.diff:.2e}"


def metric_names_for(iou_type):
    """Canonical metric names from the Rust evaluator for a given iou_type.

    Asked of an empty evaluator, so it works before (or without) a run, and so
    the names come from the implementation rather than a list transcribed here
    that a new metric would silently outgrow.
    """
    from hotcoco import COCO, COCOeval

    return COCOeval(COCO(), COCO(), iou_type).metric_keys()


def compare_metrics(py_stats, rs_stats, metric_names, *, tolerance=TOLERANCE):
    """Compare two stats vectors. Returns the list of :class:`MetricMismatch`.

    Metrics that are the -1.0 "not computed for this configuration" sentinel on
    *both* sides are skipped: agreeing that a metric is undefined is a real
    assertion (0.5 against -1.0 is a mismatch here), but it exercises no
    arithmetic. Cases that care whether a run reached the numeric paths pin the
    sentinels themselves — see ``test_empty_gt`` and ``test_all_crowd`` — which is
    the stronger check, since it names *which* metrics must be undefined.
    """
    expected_len = len(metric_names)
    if len(py_stats) != expected_len:
        raise ValueError(f"reference returned {len(py_stats)} metrics, expected {expected_len}")
    if len(rs_stats) != expected_len:
        raise ValueError(f"hotcoco returned {len(rs_stats)} metrics, expected {expected_len}")

    mismatches = []
    for i in range(expected_len):
        py_val, rs_val = py_stats[i], rs_stats[i]
        if py_val == -1.0 and rs_val == -1.0:
            continue
        diff = abs(py_val - rs_val)
        if diff > tolerance:
            mismatches.append(MetricMismatch(i, metric_names[i], py_val, rs_val, diff))
    return mismatches


def assert_metrics_match(py_stats, rs_stats, iou_type, *, tolerance=TOLERANCE, on_mismatch=None):
    """Assert every metric matches within tolerance.

    ``on_mismatch`` is called with the list of :class:`MetricMismatch` before the
    assertion fires — the fuzzer uses it to save a reproducer to disk.
    """
    mismatches = compare_metrics(py_stats, rs_stats, metric_names_for(iou_type), tolerance=tolerance)
    if mismatches:
        if on_mismatch is not None:
            on_mismatch(mismatches)
        raise AssertionError(
            f"\n{iou_type} metric mismatch (tol={tolerance}):\n" + "\n".join(m.line() for m in mismatches)
        )


def reference_stats(gt_file, dt_file, iou_type):
    """pycocotools' 12 (or 10) summary numbers for a GT/DT file pair, quietly.

    The one definition of "what the reference says" for val2017: `parity.py`
    compares against it live and `gen_val2017_baseline.py` pins it.
    """
    from pycocotools.coco import COCO as PyCOCO
    from pycocotools.cocoeval import COCOeval as PyCOCOeval

    with suppress_output(stderr=False):
        gt = PyCOCO(str(gt_file))
        dt = gt.loadRes(str(dt_file))
        ev = PyCOCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [float(v) for v in ev.stats]


def float32_grids():
    """The default IoU and recall grids rounded through float32, as torchmetrics'
    ``torch.linspace(...).tolist()`` returns them: up to 4e-8 from the default."""
    iou = np.linspace(0.5, 0.95, 10, dtype=np.float32).astype(np.float64).tolist()
    rec = np.linspace(0.0, 1.0, 101, dtype=np.float32).astype(np.float64).tolist()
    return iou, rec


def grid_sensitive_records(n_gt=20):
    """One category, ``n_gt`` ground truths on as many images, and a detection
    per image with the true positives at triangular-number ranks. Recall climbs
    in exact steps of ``1 / n_gt``, which land on recall-grid points, while
    precision falls, so a grid point one ulp off picks another precision. A flat
    precision curve would hide that. Returns ``(images, annotations,
    detections)`` as plain dicts."""
    images = [{"id": i, "width": 200, "height": 200, "file_name": f"{i}.jpg"} for i in range(1, n_gt + 1)]
    anns = [
        {"id": i, "image_id": i, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0}
        for i in range(1, n_gt + 1)
    ]
    tp_ranks = {k * (k + 1) // 2 for k in range(1, n_gt) if k * (k + 1) // 2 <= n_gt}
    dets = [
        {
            "image_id": rank,
            "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 20.0] if rank in tp_ranks else [150.0, 150.0, 20.0, 20.0],
            "score": 1.0 - 0.001 * rank,
        }
        for rank in range(1, n_gt + 1)
    ]
    return images, anns, dets


def run_both(gt_dataset, dt_results, iou_type):
    """Evaluate one (GT dict, DT list) pair through both implementations.

    Returns ``(py_stats, rs_stats, rs_ev)``. The evaluator comes back because the
    fuzzer asserts hotcoco-only invariants on it; callers that only diff metrics
    unpack it as ``_``. Both tools read from files, so the pair is round-tripped
    through temp JSON — which is also what the real entry points do.
    """
    from hotcoco import COCO as RsCOCO  # noqa: PLC0415
    from hotcoco import COCOeval as RsCOCOeval  # noqa: PLC0415
    from pycocotools.coco import COCO as PyCOCO  # noqa: PLC0415
    from pycocotools.cocoeval import COCOeval as PyCOCOeval  # noqa: PLC0415

    with written_json(gt_dataset, dt_results, quiet=True) as (gt_path, dt_path):
        py_gt = PyCOCO(gt_path)
        py_dt = py_gt.loadRes(dt_path)
        py_ev = PyCOCOeval(py_gt, py_dt, iou_type)
        py_ev.evaluate()
        py_ev.accumulate()
        py_ev.summarize()
        py_stats = py_ev.stats.tolist()

        rs_gt = RsCOCO(gt_path)
        rs_dt = rs_gt.load_res(dt_path)
        rs_ev = RsCOCOeval(rs_gt, rs_dt, iou_type)
        rs_ev.evaluate()
        rs_ev.accumulate()
        rs_ev.summarize()
        rs_stats = rs_ev.stats

    return py_stats, rs_stats, rs_ev


def hotcoco_eval_obb(obb_gt, obb_dt):
    """Run hotcoco OBB evaluation and return the 12-metric stats vector.

    A minimal dataset: one GT and one DT on a 4096x4096 image. The detection
    scores 1.0 — the only score a single-detection case can meaningfully carry.
    `tests/test_obb_eval.py` and `tests/fuzz_obb.py` share it.
    """
    from hotcoco import COCO, COCOeval

    gt_area = obb_gt[2] * obb_gt[3]

    gt_data = {
        "images": [{"id": 1, "width": 4096, "height": 4096, "file_name": "test.png"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "obb": list(obb_gt),
                "area": gt_area,
                "bbox": [0, 0, 100, 100],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }

    dt_data = [{"image_id": 1, "category_id": 1, "obb": list(obb_dt), "score": 1.0}]

    with written_json(gt_data, dt_data, quiet=True) as (gt_path, dt_path):
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.load_res(dt_path)
        ev = COCOeval(coco_gt, coco_dt, "obb")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return ev.stats


# ---------------------------------------------------------------------------
# Panoptic: the reference run, the comparison, and the perturbation recipe
# ---------------------------------------------------------------------------
#
# Shared by `scripts/parity_panoptic.py` (val2017) and
# `tests/test_parity_panoptic.py` (synthetic, in CI) so the two are one check
# on two inputs rather than two checks. panopticapi is imported inside the
# functions: it is a dev-only dependency and this module is imported by every
# test.


def panopticapi_reference(gt_json, pred_json, gt_folder, pred_folder, *, multi_core: bool):
    """panopticapi's `pq_compute`, split open to keep the per-category counts.

    Returns ``(averages, counts)``: ``averages`` maps ``All``/``Things``/``Stuff``
    to the dict ``pq_average`` returns, or ``None`` where the reference divides
    by zero (nothing to average); ``counts`` maps category id to its
    ``tp``/``fp``/``fn``/``iou``. Does the function's own glue — pairing by
    image id, raising on a missing prediction — and calls the worker it calls:
    the multiprocess one for a real dataset, the single-core one for tests.
    """
    from panopticapi.evaluation import pq_compute_multi_core, pq_compute_single_core

    gt = json.loads(Path(gt_json).read_text())
    pred = json.loads(Path(pred_json).read_text())
    categories = {el["id"]: el for el in gt["categories"]}
    pred_by_image = {el["image_id"]: el for el in pred["annotations"]}
    matched = []
    for gt_ann in gt["annotations"]:
        if gt_ann["image_id"] not in pred_by_image:
            raise Exception(f"no prediction for the image with id: {gt_ann['image_id']}")
        matched.append((gt_ann, pred_by_image[gt_ann["image_id"]]))
    # The reference prints its core count and per-core progress.
    with suppress_output(stderr=False):
        if multi_core:
            pq_stat = pq_compute_multi_core(matched, str(gt_folder), str(pred_folder), categories)
        else:
            pq_stat = pq_compute_single_core(0, matched, str(gt_folder), str(pred_folder), categories)
    averages = {}
    for name, isthing in (("All", None), ("Things", True), ("Stuff", False)):
        try:
            averages[name] = pq_stat.pq_average(categories, isthing)[0]
        except ZeroDivisionError:
            averages[name] = None
    counts = {
        cid: {"tp": s.tp, "fp": s.fp, "fn": s.fn, "iou": s.iou}
        for cid, s in pq_stat.pq_per_cat.items()
        if cid in categories
    }
    return averages, counts


def pq_disagreements(ref_averages, ref_counts, got, category_ids, *, tol=1e-9):
    """Where hotcoco's `results()` departs from the reference, as strings.

    Counts must match exactly; summed IoU and the three averages within
    ``tol``. A category the reference saw nothing for must read as the
    ``-1.0`` sentinel, not a score. ``category_ids`` are the categories to
    check, so a category missing from either side is a finding.
    """
    out = []
    for split in ("All", "Things", "Stuff"):
        ref, g = ref_averages[split], got[split]
        if ref is None:
            if g["n"] != 0 or g["pq"] != -1.0:
                out.append(f"{split}: reference has nothing to average, hotcoco reports {g}")
            continue
        if ref["n"] != g["n"]:
            out.append(f"{split}.n: ref {ref['n']} got {g['n']}")
        for key in ("pq", "sq", "rq"):
            if abs(ref[key] - g[key]) > tol:
                out.append(f"{split}.{key}: ref {ref[key]:.12f} got {g[key]:.12f} diff {abs(ref[key] - g[key]):.3e}")
    for cid in category_ids:
        ref = ref_counts.get(cid, {"tp": 0, "fp": 0, "fn": 0, "iou": 0.0})
        g = got["per_class"].get(cid)
        if g is None:
            out.append(f"category {cid}: missing from hotcoco")
            continue
        for key in ("tp", "fp", "fn"):
            if ref[key] != g[key]:
                out.append(f"category {cid}.{key}: ref {ref[key]} got {g[key]}")
        if abs(ref["iou"] - g["iou"]) > tol:
            out.append(f"category {cid}.iou: ref {ref['iou']:.12f} got {g['iou']:.12f}")
        if ref["tp"] + ref["fp"] + ref["fn"] == 0 and (g["pq"], g["sq"], g["rq"]) != (-1.0, -1.0, -1.0):
            out.append(f"category {cid}: nothing to score, hotcoco reports pq={g['pq']} sq={g['sq']} rq={g['rq']}")
    return out


def erode(mask, px: int):
    """Shrink a boolean mask by `px` pixels on every side (no scipy)."""
    out = mask.copy()
    for _ in range(px):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        out = shrunk
    return out


def perturb_label_map(gt_map, gt_segs, category_ids, rng, *, shift: int, erode_px, first_id: int, spurious: int):
    """A prediction derived from a ground-truth label map, every rule in play.

    Each segment is dropped, relabeled to a random category, eroded, shifted,
    or merged into a same-category neighbor, with the given odds; up to
    ``spurious`` rectangles are added on top. Later segments paint over
    earlier ones, so the result stays a partition. Returns
    ``(pred_map, segments_info)`` with ids from ``first_id``.
    """
    h, w = gt_map.shape
    pred = np.zeros_like(gt_map)
    segs = []
    next_id = first_id
    for seg in gt_segs:
        m = gt_map == seg["id"]
        if not m.any():
            continue
        roll = rng.random()
        cat = seg["category_id"]
        if roll < 0.1:
            continue  # dropped: a miss
        if roll < 0.2:
            cat = rng.choice(category_ids)  # relabeled: a miss and a false positive
        elif roll < 0.3:
            m = erode(m, rng.choice(erode_px))  # IoU drifts toward 0.5
        elif roll < 0.4:
            dy, dx = rng.randint(-shift, shift), rng.randint(-shift, shift)
            m = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        elif roll < 0.5:
            others = [s for s in gt_segs if s["id"] != seg["id"] and s["category_id"] == cat]
            if others:
                m = m | (gt_map == rng.choice(others)["id"])
        if not m.any():
            continue
        pred[m] = next_id
        segs.append({"id": next_id, "category_id": cat})
        next_id += 1
    # Spurious segments on void or on top of anything: false positives, some
    # majority-void so the ignore rule fires.
    for _ in range(rng.randint(0, spurious)):
        sh, sw = rng.randint(2, max(3, h // 3)), rng.randint(2, max(3, w // 3))
        y0, x0 = rng.randint(0, max(0, h - sh)), rng.randint(0, max(0, w - sw))
        pred[y0 : y0 + sh, x0 : x0 + sw] = next_id
        segs.append({"id": next_id, "category_id": rng.choice(category_ids)})
        next_id += 1
    present = set(np.unique(pred).tolist())
    return pred, [s for s in segs if s["id"] in present]


# ---------------------------------------------------------------------------
# Panoptic: synthetic cases
# ---------------------------------------------------------------------------
#
# Ground truth and prediction for a few images, as label maps plus segments_info,
# written out the way the COCO panoptic release lays them out.
# `tests/test_parity_panoptic.py` runs them through panopticapi and `PanopticEval`;
# `tests/test_composed_metrics.py` through `PanopticEval` and the primitives.

PANOPTIC_CATEGORIES = [
    {"id": 1, "name": "person", "isthing": 1},
    {"id": 2, "name": "car", "isthing": 1},
    {"id": 7, "name": "dog", "isthing": 1},
    {"id": 100, "name": "sky", "isthing": 0},
    {"id": 101, "name": "road", "isthing": 0},
]
PANOPTIC_CATEGORY_IDS = [c["id"] for c in PANOPTIC_CATEGORIES]


def id2rgb(label_map: np.ndarray) -> np.ndarray:
    """A label map as COCO panoptic PNG colors: id = R + 256 G + 256² B, as panopticapi's `id2rgb`."""
    ids = label_map.astype(np.uint32)
    return np.stack([ids % 256, ids // 256 % 256, ids // 256**2 % 256], axis=-1).astype(np.uint8)


class PanopticCase:
    """Ground truth and prediction for a few images, as label maps + segments."""

    def __init__(self, name: str, *, png_only: bool = False):
        self.name = name
        # True when the case leans on something only the PNG format can say —
        # a PNG id with no segments_info entry, or a stated area that differs
        # from the pixels — so the RLE path is not expected to reproduce it.
        self.png_only = png_only
        self.images: list[dict] = []
        # image_id -> (label map, segments_info)
        self.gt: dict[int, tuple[np.ndarray, list[dict]]] = {}
        self.pred: dict[int, tuple[np.ndarray, list[dict]]] = {}
        self.categories = [dict(c) for c in PANOPTIC_CATEGORIES]

    def add(self, image_id: int, gt_map, gt_segs, pred_map, pred_segs):
        h, w = gt_map.shape
        self.images.append({"id": image_id, "height": h, "width": w, "file_name": f"{image_id}.jpg"})
        self.gt[image_id] = (gt_map, gt_segs)
        self.pred[image_id] = (pred_map, pred_segs)
        return self

    def write(self, root: Path) -> dict:
        """Write PNG files + JSON the way the COCO panoptic release lays them out."""
        from PIL import Image

        paths = {}
        for side, data in (("gt", self.gt), ("pred", self.pred)):
            folder = root / f"panoptic_{side}"
            folder.mkdir(parents=True, exist_ok=True)
            annotations = []
            for image_id, (label_map, segs) in data.items():
                Image.fromarray(id2rgb(label_map)).save(folder / f"{image_id}.png")
                annotations.append({"image_id": image_id, "file_name": f"{image_id}.png", "segments_info": segs})
            payload = {"images": self.images, "annotations": annotations, "categories": self.categories}
            json_path = root / f"panoptic_{side}.json"
            json_path.write_text(json.dumps(payload))
            paths[side] = (json_path, folder)
        return paths

    def _numbered(self, side: str):
        """`(annotation id, image id, label map, segment)` for every segment of
        `side`, numbered 1, 2, ... across the images in order, as `as_coco` numbers
        its annotations."""
        data = self.gt if side == "gt" else self.pred
        segments = ((image_id, label_map, seg) for image_id, (label_map, segs) in data.items() for seg in segs)
        for ann_id, (image_id, label_map, seg) in enumerate(segments, 1):
            yield ann_id, image_id, label_map, seg

    def as_coco(self, side: str):
        """The same maps as a detection-style dataset with RLE masks."""
        from hotcoco import COCO, mask

        annotations = []
        for ann_id, image_id, label_map, seg in self._numbered(side):
            binary = np.asfortranarray((label_map == seg["id"]).astype(np.uint8))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": seg["category_id"],
                    "iscrowd": seg.get("iscrowd", 0),
                    "segmentation": mask.encode(binary),
                    "area": int(binary.sum()),
                }
            )
        return COCO({"images": self.images, "annotations": annotations, "categories": self.categories})

    def segment_ids(self, side: str) -> dict[tuple[int, int], int]:
        """The segment id each annotation of `as_coco(side)` stands for, keyed by
        `(image id, annotation id)`: segment ids restart in every image, so an id
        read without its image could name another image's segment."""
        return {(image_id, ann_id): seg["id"] for ann_id, image_id, _, seg in self._numbered(side)}


def segments_from_map(label_map: np.ndarray, category_of, rng: random.Random, *, crowd_rate=0.0, area_noise=0.0):
    """segments_info for every non-zero id in `label_map`."""
    segs = []
    for sid in sorted(int(v) for v in np.unique(label_map) if v != 0):
        area = int((label_map == sid).sum())
        if area_noise and rng.random() < area_noise:
            # panopticapi takes the JSON's word for a ground-truth area.
            area += rng.randint(-area // 4, area // 2)
        segs.append(
            {"id": sid, "category_id": category_of(sid), "area": area, "iscrowd": 1 if rng.random() < crowd_rate else 0}
        )
    return segs


def random_partition(rng: random.Random, h: int, w: int, n: int, first_id: int = 1) -> np.ndarray:
    """`n` rectangles painted in order, later over earlier; 0 stays void."""
    m = np.zeros((h, w), dtype=np.uint32)
    sid = first_id
    for _ in range(n):
        rh, rw = rng.randint(2, max(3, h // 2)), rng.randint(2, max(3, w // 2))
        y, x = rng.randint(0, h - rh), rng.randint(0, w - rw)
        m[y : y + rh, x : x + rw] = sid
        sid += 1
    return m


def random_panoptic_case(seed: int, area_noise: float = 0.0) -> PanopticCase:
    rng = random.Random(seed)
    case = PanopticCase(f"random-{seed}" + ("-area-noise" if area_noise else ""), png_only=bool(area_noise))
    for image_id in (1, 2):
        h, w = rng.randint(12, 40), rng.randint(12, 40)
        gt_map = random_partition(rng, h, w, rng.randint(2, 7))
        cats = {}

        def category_of(sid, cats=cats, rng=rng):
            return cats.setdefault(sid, rng.choice(PANOPTIC_CATEGORY_IDS))

        gt_segs = segments_from_map(gt_map, category_of, rng, crowd_rate=0.15, area_noise=area_noise)
        pred_map, pred_segs = perturb_label_map(
            gt_map, gt_segs, PANOPTIC_CATEGORY_IDS, rng, shift=3, erode_px=(1,), first_id=1000, spurious=3
        )
        case.add(image_id, gt_map, gt_segs, pred_map, pred_segs)
    return case


def random_panoptic_cases() -> list[PanopticCase]:
    """Forty random cases, then ten whose stated ground-truth areas are off."""
    return [random_panoptic_case(seed) for seed in range(40)] + [
        random_panoptic_case(seed, area_noise=0.5) for seed in range(10)
    ]


def hand_panoptic_cases() -> list[PanopticCase]:
    """One case per rule, with the geometry written out."""
    cases = []

    # Crowd absorption: a prediction mostly on a crowd region of its own
    # category is ignored; the same prediction of another category is a FP.
    gt = np.zeros((8, 8), dtype=np.uint32)
    gt[:, :4] = 1  # crowd person
    gt[0:2, 6:8] = 2  # person
    pred = np.zeros_like(gt)
    pred[0:3, :4] = 11  # on the crowd, cat person -> ignored
    pred[4:7, :4] = 12  # on the crowd, cat car -> FP
    pred[0:4, 6:8] = 13  # covers segment 2 plus void -> matched, IoU 1.0
    cases.append(
        PanopticCase("crowd").add(
            1,
            gt,
            [
                {"id": 1, "category_id": 1, "area": 32, "iscrowd": 1},
                {"id": 2, "category_id": 1, "area": 4, "iscrowd": 0},
            ],
            pred,
            [{"id": 11, "category_id": 1}, {"id": 12, "category_id": 2}, {"id": 13, "category_id": 1}],
        )
    )

    # Exactly 0.5 is not a match.
    gt = np.zeros((4, 4), dtype=np.uint32)
    gt[:, 0:3] = 1
    gt[:, 3] = 2
    pred = np.zeros_like(gt)
    pred[:, 0:2] = 11
    pred[:, 2:4] = 12
    # gt1 (12 px) vs pred 11 (8 px): inter 8, union 12 -> 0.667 match.
    # gt2 (4 px) vs pred 12 (8 px): inter 4, union 8 -> 0.5 no match.
    cases.append(
        PanopticCase("half").add(
            1,
            gt,
            [
                {"id": 1, "category_id": 100, "area": 12, "iscrowd": 0},
                {"id": 2, "category_id": 101, "area": 4, "iscrowd": 0},
            ],
            pred,
            [{"id": 11, "category_id": 100}, {"id": 12, "category_id": 101}],
        )
    )

    # A ground-truth PNG id with no segments_info entry: not void, so it
    # neither shrinks the union nor helps a prediction get ignored.
    gt = np.zeros((4, 8), dtype=np.uint32)
    gt[:, 0:4] = 9  # unknown to the JSON
    gt[:, 4:8] = 1
    pred = np.zeros_like(gt)
    pred[:, 2:8] = 11  # 8 on gt1, 8 on the unknown id: iou 16/(24+16-16) = 2/3
    cases.append(
        PanopticCase("unknown-gt-label", png_only=True).add(
            1, gt, [{"id": 1, "category_id": 2, "area": 16, "iscrowd": 0}], pred, [{"id": 11, "category_id": 2}]
        )
    )

    # The JSON's ground-truth area, not the PNG's, enters the union.
    gt = np.zeros((4, 4), dtype=np.uint32)
    gt[:, :] = 1
    pred = np.zeros_like(gt)
    pred[:, :] = 11
    cases.append(
        PanopticCase("json-area", png_only=True)
        .add(1, gt, [{"id": 1, "category_id": 7, "area": 40, "iscrowd": 0}], pred, [{"id": 11, "category_id": 7}])
        .add(2, gt, [{"id": 1, "category_id": 7, "area": 16, "iscrowd": 0}], pred, [{"id": 11, "category_id": 7}])
    )

    # An image where nothing was predicted, and one with nothing to find.
    gt = np.zeros((6, 6), dtype=np.uint32)
    gt[1:5, 1:5] = 1
    empty = np.zeros_like(gt)
    pred = np.zeros_like(gt)
    pred[2:4, 2:4] = 11
    cases.append(
        PanopticCase("empty-sides")
        .add(1, gt, [{"id": 1, "category_id": 1, "area": 16, "iscrowd": 0}], empty, [])
        .add(2, empty, [], pred, [{"id": 11, "category_id": 100}])
    )

    # An id listed twice: panopticapi's dict keeps the last entry, at the
    # first one's position. Here that turns segment 1 from a car into a
    # person, which pred 11 then matches.
    gt = np.zeros((4, 4), dtype=np.uint32)
    gt[:, 0:2] = 1
    gt[:, 2:4] = 2
    pred = gt + 10 * (gt > 0)
    cases.append(
        PanopticCase("repeated-id", png_only=True).add(
            1,
            gt,
            [
                {"id": 1, "category_id": 2, "area": 8, "iscrowd": 0},
                {"id": 2, "category_id": 100, "area": 8, "iscrowd": 0},
                {"id": 1, "category_id": 1, "area": 8, "iscrowd": 0},
            ],
            pred,
            [{"id": 11, "category_id": 1}, {"id": 12, "category_id": 100}, {"id": 11, "category_id": 1}],
        )
    )

    # Ids past 32 bits, in no PNG: each its own missed segment, a repeated
    # one once, as panopticapi's dict holds them.
    gt = np.zeros((4, 4), dtype=np.uint32)
    gt[:, 0:2] = 1
    pred = gt * 11
    cases.append(
        PanopticCase("wide-ids", png_only=True).add(
            1,
            gt,
            [
                {"id": 1, "category_id": 1, "area": 8, "iscrowd": 0},
                {"id": 2**33, "category_id": 1, "area": 4, "iscrowd": 0},
                {"id": 2**34, "category_id": 2, "area": 4, "iscrowd": 0},
                {"id": 2**33, "category_id": 2, "area": 4, "iscrowd": 0},
            ],
            pred,
            [{"id": 11, "category_id": 1}],
        )
    )

    # Id 0 listed as a segment: panopticapi scores the void pixels as one, and
    # subtracts them from its union as void too. Ground-truth 0 misses pred 0
    # (IoU 2 / (2 + 8 - 2 - 2)); pred 0, all on void, is ignored.
    gt = np.zeros((2, 2), dtype=np.uint32)
    gt[1] = 1
    cases.append(
        PanopticCase("void-id", png_only=True).add(
            1,
            gt,
            [
                {"id": 0, "category_id": 1, "area": 8, "iscrowd": 0},
                {"id": 1, "category_id": 1, "area": 2, "iscrowd": 0},
            ],
            gt.copy(),
            [{"id": 0, "category_id": 1}, {"id": 1, "category_id": 1}],
        )
    )

    # Two ground truths stated at 1 pixel each, 3 found: pred 11 covers both
    # at IoU 3 / (6 + 1 - 3) = 0.75 and matches twice, as panopticapi counts it.
    gt = np.zeros((2, 3), dtype=np.uint32)
    gt[0] = 1
    gt[1] = 2
    cases.append(
        PanopticCase("double-match", png_only=True).add(
            1,
            gt,
            [
                {"id": 1, "category_id": 1, "area": 1, "iscrowd": 0},
                {"id": 2, "category_id": 1, "area": 1, "iscrowd": 0},
            ],
            np.full_like(gt, 11),
            [{"id": 11, "category_id": 1}],
        )
    )

    # A stated ground-truth area below the pixels found is taken at its word:
    # 8 pixels stated as 4 give pred 11 an IoU of 8 / (8 + 4 - 8) = 2.
    gt = np.zeros((4, 4), dtype=np.uint32)
    gt[:, 0:2] = 1
    pred = gt * 11
    cases.append(
        PanopticCase("understated-area", png_only=True).add(
            1, gt, [{"id": 1, "category_id": 7, "area": 4, "iscrowd": 0}], pred, [{"id": 11, "category_id": 7}]
        )
    )

    return cases


# ── a detection corpus built to tie ───────────────────────────────────────────
#
# `tests/test_composed_metrics.py` matches it with `greedy_match` against
# `COCOeval`'s `evalImgs`; `tests/test_build_your_own_metric.py` runs the
# custom-metrics guide's code on it.


def detection_corpus(seed: int) -> tuple[dict, list[dict]]:
    """A ground-truth dataset and a results list built to tie: duplicate boxes
    (equal IoU to two ground truths), coarse scores (equal scores), crowd regions
    that several detections fall inside, and boxes of every size for the area
    ranges. The ground-truth ids are shuffled out of dataset order, as val2017's
    are, so code that reads annotations in id order breaks ties differently from
    `COCOeval`, which reads them in dataset order."""
    rng = random.Random(seed)
    images = [{"id": i, "width": 100, "height": 100} for i in range(1, 16)]
    categories = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}]
    gts, dts = [], []

    def box():
        w, h = rng.choice([6, 12, 20, 40, 60]), rng.choice([6, 12, 20, 40, 60])
        return [rng.randint(0, 100 - w), rng.randint(0, 100 - h), w, h]

    def jitter(b, px):
        return [b[0] + rng.randint(-px, px), b[1] + rng.randint(-px, px), b[2], b[3]]

    for image in images:
        for category in categories:
            cell = {"image_id": image["id"], "category_id": category["id"]}
            boxes = [box() for _ in range(rng.randint(0, 4))]
            if boxes and rng.random() < 0.3:
                boxes.append(list(boxes[0]))  # a duplicate ground truth
            for b in boxes:
                crowd = int(rng.random() < 0.15)
                gts.append({**cell, "bbox": b, "iscrowd": crowd, "area": float(b[2] * b[3])})
                for _ in range(rng.randint(0, 3 if crowd else 2)):
                    dts.append({**cell, "bbox": jitter(b, 3), "score": rng.choice([0.3, 0.5, 0.5, 0.9])})
            for _ in range(rng.randint(0, 2)):
                dts.append({**cell, "bbox": box(), "score": rng.random()})
    # One box on each area-range bound (32² and 96²), matched by a detection of
    # the same size, so a reading of the bounds as exclusive fails the gates.
    for side in (32, 96):
        b = [1, 1, side, side]
        gts.append({"image_id": 1, "category_id": 1, "bbox": b, "iscrowd": 0, "area": float(side * side)})
        dts.append({"image_id": 1, "category_id": 1, "bbox": list(b), "score": 0.7})
    ids = list(range(1, len(gts) + 1))
    rng.shuffle(ids)
    for gt, gt_id in zip(gts, ids):
        gt["id"] = gt_id
    return {"images": images, "annotations": gts, "categories": categories}, dts


# ── COCOeval's counts, read from evalImgs ─────────────────────────────────────
#
# One reading of what `accumulate()` counts. `tests/test_composed_metrics.py`
# holds the primitives' matches to it, and `tests/test_build_your_own_metric.py`
# holds the guide page's counts to it, so the two cannot disagree about COCOeval.


def outside_area_range(anns, area_range) -> np.ndarray:
    """True for each annotation whose `area` lies outside `area_range`, both ends
    inclusive: COCOeval ignores a ground truth there, and an unmatched detection."""
    lo, hi = area_range
    return np.array([not lo <= a["area"] <= hi for a in anns], dtype=bool)


def coco_counts(matched, dt_ignore, gt_ignore) -> tuple[int, int, int]:
    """`(tp, fp, fn)` the way `accumulate()` counts one cell at one IoU threshold.

    A detection is a true positive when matched and not ignored, and a false
    positive when neither; a ground truth not ignored that no true positive took
    is a miss. `dt_ignore` and `gt_ignore` are `evalImgs`' `dtIgnore` and
    `gtIgnore`: for `greedy_match`, the `dt_ignore` it returns, and
    `gt_ignore | iscrowd`.
    """
    matched, dt_ignore, gt_ignore = (np.asarray(a, dtype=bool) for a in (matched, dt_ignore, gt_ignore))
    tp = int(np.sum(matched & ~dt_ignore))
    return tp, int(np.sum(~matched & ~dt_ignore)), int(np.sum(~gt_ignore)) - tp


def eval_img_counts(e: dict, t: int = 0, score_thr: float = float("-inf")) -> tuple[int, int, int]:
    """`coco_counts` of one `evalImgs` entry at `iou_thrs[t]`, over its detections
    scoring `score_thr` or more. `dtMatches` holds 0 for "no match", and `dtIgnore`
    already ignores an unmatched detection outside the entry's area range."""
    confident = np.array(e["dtScores"], dtype=float) >= score_thr
    matched = np.array(e["dtMatches"][t], dtype=float) > 0
    dt_ignore = np.array(e["dtIgnore"][t], dtype=bool)
    return coco_counts(matched[confident], dt_ignore[confident], e["gtIgnore"])


def counts_from_eval_imgs(
    eval_imgs: list, area_range, score_thr: float = float("-inf"), t: int = 0
) -> dict[int, tuple[int, int, int]]:
    """`(tp, fp, fn)` per category id at `iou_thrs[t]`, summed over the `evalImgs`
    entries for `area_range`."""
    counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for e in eval_imgs:
        if e is not None and e["aRng"] == list(area_range):
            counts[e["category_id"]] += eval_img_counts(e, t, score_thr)
    return {category_id: tuple(n.tolist()) for category_id, n in counts.items()}


# ── running a reference from its pinned source ────────────────────────────────
#
# Some references are not pip-installable, or drag in a framework the metric code
# never calls. A fixture generator downloads the file it needs at a pinned commit,
# stands in for the imports that would pull the framework, and loads the module;
# `scripts/gen_semantic_fixtures.py` is the pattern.


def fetch_source(url: str, dest: Path) -> None:
    """Download one source file to `dest`."""
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
        dest.write_bytes(r.read())


def stub_module(name: str, **attrs) -> types.ModuleType:
    """Register a bare module under `name` carrying `attrs`, for an import the
    code under test makes and never uses."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def load_module(path: Path, name: str):
    """Import the file at `path` as `name`; a dotted name makes relative imports
    resolve against the stubbed parent package."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
