"""Hypothesis-based parity fuzzer: hotcoco vs pycocotools.

A bug-hunting tool, not a CI gate. Generates thousands of diverse COCO datasets
using hypothesis, evaluates through both pycocotools and hotcoco, and compares all
metrics to within 1e-10 tolerance. When discrepancies are found, hypothesis
auto-minimizes to the smallest failing case.

Workflow: use this fuzzer to *find* bugs, then prove fixes with Rust integration
tests. Do not add this to CI — it takes several minutes to run.

Usage:
    uv run pytest tests/fuzz_parity.py -v -x --tb=short
    just fuzz
"""

import json
import os
import time

import hypothesis.strategies as st
from helpers import COCO_KEYPOINT_NAMES, COCO_SKELETON, FIXTURES_DIR, assert_metrics_match, metric_names_for, run_both
from hypothesis import HealthCheck, given, settings
from hypothesis.database import DirectoryBasedExampleDatabase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_DIR = str(FIXTURES_DIR / "parity_failures")


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def coco_images(draw):
    """Generate 1-20 COCO image dicts."""
    n = draw(st.integers(min_value=1, max_value=20))
    images = []
    for i in range(n):
        images.append(
            {
                "id": i + 1,
                "width": draw(st.integers(min_value=32, max_value=2048)),
                "height": draw(st.integers(min_value=32, max_value=2048)),
                "file_name": f"img_{i + 1}.jpg",
            }
        )
    return images


@st.composite
def coco_categories(draw, iou_type="bbox"):
    """Generate 1-5 categories (1 for keypoints)."""
    if iou_type == "keypoints":
        return [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": COCO_KEYPOINT_NAMES,
                "skeleton": COCO_SKELETON,
            }
        ]
    n = draw(st.integers(min_value=1, max_value=5))
    return [{"id": i + 1, "name": f"cat_{i + 1}", "supercategory": "none"} for i in range(n)]


@st.composite
def coco_bbox(draw, w, h):
    """Generate a [x, y, w, h] bbox within image bounds.

    Includes edge cases: zero-area, full-image, 1x1, area-boundary boxes.
    """
    kind = draw(
        st.sampled_from(
            [
                "normal",
                "normal",
                "normal",
                "normal",
                "zero_w",
                "zero_h",
                "full",
                "tiny",
                "boundary_1024",
                "boundary_9216",
            ]
        )
    )
    if kind == "full":
        return [0.0, 0.0, round(float(w), 2), round(float(h), 2)]
    if kind == "tiny":
        x = round(draw(st.floats(min_value=0, max_value=max(0, w - 1))), 2)
        y = round(draw(st.floats(min_value=0, max_value=max(0, h - 1))), 2)
        return [x, y, 1.0, 1.0]
    if kind == "zero_w":
        x = round(draw(st.floats(min_value=0, max_value=float(w))), 2)
        y = round(draw(st.floats(min_value=0, max_value=float(h))), 2)
        bh = round(draw(st.floats(min_value=0, max_value=float(h) - y)), 2)
        return [x, y, 0.0, bh]
    if kind == "zero_h":
        x = round(draw(st.floats(min_value=0, max_value=float(w))), 2)
        y = round(draw(st.floats(min_value=0, max_value=float(h))), 2)
        bw = round(draw(st.floats(min_value=0, max_value=float(w) - x)), 2)
        return [x, y, bw, 0.0]
    if kind.startswith("boundary_"):
        target_area = float(kind.split("_")[1])
        side = target_area**0.5
        if side <= min(w, h):
            x = round(draw(st.floats(min_value=0, max_value=max(0, float(w) - side))), 2)
            y = round(draw(st.floats(min_value=0, max_value=max(0, float(h) - side))), 2)
            return [x, y, round(side, 2), round(side, 2)]
        # fallthrough to normal
    # normal
    x = round(draw(st.floats(min_value=0, max_value=float(w) - 1)), 2)
    y = round(draw(st.floats(min_value=0, max_value=float(h) - 1)), 2)
    bw = round(draw(st.floats(min_value=0.01, max_value=float(w) - x)), 2)
    bh = round(draw(st.floats(min_value=0.01, max_value=float(h) - y)), 2)
    return [x, y, bw, bh]


@st.composite
def coco_polygon(draw, w, h):
    """Generate a polygon segmentation [[x1,y1,x2,y2,...]] with 3-20 vertices."""
    n_verts = draw(st.integers(min_value=3, max_value=20))
    coords = []
    for _ in range(n_verts):
        coords.append(round(draw(st.floats(min_value=0, max_value=float(w))), 2))
        coords.append(round(draw(st.floats(min_value=0, max_value=float(h))), 2))
    return [coords]


@st.composite
def coco_keypoints(draw, w, h):
    """Generate (keypoints_flat, num_visible) for 17 COCO keypoints."""
    kpts = []
    num_vis = 0
    for _ in range(17):
        v = draw(st.sampled_from([0, 0, 1, 2, 2]))  # bias toward visible
        if v == 0:
            kpts.extend([0, 0, 0])
        else:
            kpts.append(round(draw(st.floats(min_value=0, max_value=float(w))), 2))
            kpts.append(round(draw(st.floats(min_value=0, max_value=float(h))), 2))
            kpts.append(v)
            num_vis += 1
    return kpts, num_vis


def _score_strategy():
    """Score with edge-case bias toward 0.0 and 1.0."""
    return st.one_of(st.just(0.0), st.just(1.0), st.floats(min_value=0.0, max_value=1.0).map(lambda x: round(x, 4)))


@st.composite
def gt_annotation(draw, ann_id, images, categories, iou_type):
    """Generate a single ground-truth annotation."""
    img = draw(st.sampled_from(images))
    cat = draw(st.sampled_from(categories))
    w, h = img["width"], img["height"]
    bbox = draw(coco_bbox(w, h))
    area = round(bbox[2] * bbox[3], 2)
    iscrowd = draw(st.sampled_from([0, 0, 0, 0, 1]))  # ~20% crowd

    ann = {
        "id": ann_id,
        "image_id": img["id"],
        "category_id": cat["id"],
        "bbox": bbox,
        "area": area,
        "iscrowd": iscrowd,
    }

    if iou_type == "segm":
        if iscrowd:
            # Crowd annotations use RLE; for simplicity use bbox-derived polygon
            ann["segmentation"] = [
                [
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3],
                    bbox[0],
                    bbox[1] + bbox[3],
                ]
            ]
        else:
            ann["segmentation"] = draw(coco_polygon(w, h))
    elif iou_type == "keypoints":
        kpts, num_vis = draw(coco_keypoints(w, h))
        ann["keypoints"] = kpts
        ann["num_keypoints"] = num_vis

    return ann


@st.composite
def detection(draw, images, categories, iou_type):
    """Generate a single detection result."""
    img = draw(st.sampled_from(images))
    cat = draw(st.sampled_from(categories))
    w, h = img["width"], img["height"]
    bbox = draw(coco_bbox(w, h))
    score = draw(_score_strategy())

    det = {"image_id": img["id"], "category_id": cat["id"], "bbox": bbox, "score": score}

    if iou_type == "keypoints":
        kpts, num_vis = draw(coco_keypoints(w, h))
        det["keypoints"] = kpts

    # For segm: loadRes creates polygon from bbox automatically, so bbox-only is fine.
    return det


@st.composite
def coco_eval_data(draw, iou_type):
    """Generate a complete (gt_dataset, dt_results) pair."""
    images = draw(coco_images())
    categories = draw(coco_categories(iou_type=iou_type))

    # 0-50 GTs (some images may have 0 annotations)
    n_gt = draw(st.integers(min_value=0, max_value=50))
    annotations = []
    for i in range(n_gt):
        ann = draw(gt_annotation(i + 1, images, categories, iou_type))
        annotations.append(ann)

    gt_dataset = {"images": images, "annotations": annotations, "categories": categories}

    # 1-50 detections
    n_dt = draw(st.integers(min_value=1, max_value=50))
    dt_results = []
    for _ in range(n_dt):
        det = draw(detection(images, categories, iou_type))
        dt_results.append(det)

    return gt_dataset, dt_results


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------


def assert_hotcoco_invariants(rs_ev, iou_type):
    """Properties that must hold regardless of what pycocotools says.

    The fuzzer is otherwise purely differential, which means its ~10,000 generated
    datasets only ever check the surfaces pycocotools also computes. Everything
    hotcoco adds — Open Images, oriented boxes, LVIS frequency groups, TIDE,
    calibration, the confusion matrix — gets no fuzz coverage at all, and those are
    exactly the surfaces `report()` marks `Provenance::Extension` *because* no
    reference exists. Invariants are the only check available there.

    Cheap to run on every case, and they generalize: the same assertions hold for
    a family that has no reference at all.
    """
    stats = rs_ev.stats
    names = metric_names_for(iou_type)
    assert len(stats) == len(names), f"{len(stats)} metrics for {len(names)} names"

    for name, v in zip(names, stats):
        # -1.0 is "not computed for this configuration" and is not a low score.
        # Compare against it exactly: a `>= 0.0` guard lets a genuine sign bug
        # hide behind the sentinel.
        assert v == -1.0 or 0.0 <= v <= 1.0, f"{name} = {v} is neither the -1.0 sentinel nor in [0, 1]"

    by_name = dict(zip(names, stats))

    # Larger maxDets keeps a superset of each image's detections while num_gt is
    # unchanged, so recall cannot fall.
    ar_ladder = [k for k in ("AR1", "AR10", "AR100") if k in by_name]
    live = [(k, by_name[k]) for k in ar_ladder if by_name[k] != -1.0]
    for (lo_name, lo), (hi_name, hi) in zip(live, live[1:]):
        assert hi >= lo - 1e-12, f"{hi_name}={hi} below {lo_name}={lo}: raising maxDets lost recall"

    # AP50 is one slice of the IoU sweep; AP averages over all ten, so the max
    # cannot be below the mean.
    if by_name.get("AP", -1.0) != -1.0 and by_name.get("AP50", -1.0) != -1.0:
        assert by_name["AP50"] >= by_name["AP"] - 1e-12, f"AP50={by_name['AP50']} below AP={by_name['AP']}"

    # Recall never exceeds 1.0. Not hypothetical: Open Images returned 4.0, by
    # crediting group-of detections against a denominator that excluded them.
    acc = rs_ev.eval
    if acc is not None:
        recall = acc.get("recall") if isinstance(acc, dict) else None
        if recall is not None:
            flat = recall.ravel().tolist() if hasattr(recall, "ravel") else list(recall)
            for v in flat:
                assert v == -1.0 or 0.0 <= v <= 1.0, f"recall {v} outside [0, 1]"


def assert_metrics_match_or_save(py_stats, rs_stats, iou_type, gt_dataset, dt_results):
    """Shared differential assertion, plus a saved reproducer for any mismatch.

    Hypothesis has already minimized the failing case by the time this fires, so
    the JSON pair it writes is the smallest input that diverges — the reason the
    fuzzer wants a side effect on failure and the CI suite does not. The dataset
    is required, not optional: a caller that omitted it would get the plain
    assertion with the reproducer silently not written, which is the one thing
    this wrapper exists to do.
    """

    def _save(_mismatches):
        save_failure(gt_dataset, dt_results, iou_type, py_stats, rs_stats)

    assert_metrics_match(py_stats, rs_stats, iou_type, on_mismatch=_save)


def save_failure(gt_dataset, dt_results, iou_type, py_stats, rs_stats):
    """Save failing case to disk for debugging."""
    os.makedirs(FAILURE_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    prefix = os.path.join(FAILURE_DIR, f"{iou_type}_{ts}")

    with open(f"{prefix}_gt.json", "w") as f:
        json.dump(gt_dataset, f, indent=2)
    with open(f"{prefix}_dt.json", "w") as f:
        json.dump(dt_results, f, indent=2)

    metric_names = metric_names_for(iou_type)

    with open(f"{prefix}_stats.txt", "w") as f:
        f.write(f"iou_type: {iou_type}\n")
        f.write(f"{'Metric':<8} {'pycocotools':>20} {'hotcoco':>20} {'diff':>12}\n")
        f.write("-" * 65 + "\n")
        for i in range(len(py_stats)):
            diff = abs(py_stats[i] - rs_stats[i])
            f.write(f"{metric_names[i]:<8} {py_stats[i]:>20.15f} {rs_stats[i]:>20.15f} {diff:>12.2e}\n")


# ---------------------------------------------------------------------------
# Property-based tests (~3,334 examples each = 10,000 total)
# ---------------------------------------------------------------------------

HYPOTHESIS_SETTINGS = dict(
    max_examples=3334,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    database=DirectoryBasedExampleDatabase(str(FIXTURES_DIR / ".hypothesis")),
)


@given(data=st.data())
@settings(**HYPOTHESIS_SETTINGS)
def test_bbox_parity(data):
    gt_dataset, dt_results = data.draw(coco_eval_data("bbox"))
    py_stats, rs_stats, rs_ev = run_both(gt_dataset, dt_results, "bbox")
    assert_hotcoco_invariants(rs_ev, "bbox")
    assert_metrics_match_or_save(py_stats, rs_stats, "bbox", gt_dataset, dt_results)


@given(data=st.data())
@settings(**HYPOTHESIS_SETTINGS)
def test_segm_parity(data):
    gt_dataset, dt_results = data.draw(coco_eval_data("segm"))
    py_stats, rs_stats, rs_ev = run_both(gt_dataset, dt_results, "segm")
    assert_hotcoco_invariants(rs_ev, "segm")
    assert_metrics_match_or_save(py_stats, rs_stats, "segm", gt_dataset, dt_results)


@given(data=st.data())
@settings(**HYPOTHESIS_SETTINGS)
def test_kpt_parity(data):
    gt_dataset, dt_results = data.draw(coco_eval_data("keypoints"))
    py_stats, rs_stats, rs_ev = run_both(gt_dataset, dt_results, "keypoints")
    assert_hotcoco_invariants(rs_ev, "keypoints")
    assert_metrics_match_or_save(py_stats, rs_stats, "keypoints", gt_dataset, dt_results)
