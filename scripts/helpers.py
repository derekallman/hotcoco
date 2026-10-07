"""Shared helpers for the dev scripts here and the tests in `tests/`.

`pythonpath` in `pyproject.toml` puts this directory on the path for pytest, so
both import `helpers` by the same name and there is one copy of these.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

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
    import numpy as np  # noqa: PLC0415

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
    import numpy as np

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
