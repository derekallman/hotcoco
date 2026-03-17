"""Property-based parity testing: hotcoco vs pycocotools.

Generates thousands of diverse COCO datasets using hypothesis, evaluates through
both pycocotools and hotcoco, and compares all metrics to within 1e-10 tolerance.
When discrepancies are found, hypothesis auto-minimizes to the smallest failing case.

Usage:
    uv run pytest scripts/test_parity.py -v -x --tb=short
    just test
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hotcoco import COCO as RsCOCO
from hotcoco import COCOeval as RsCOCOeval
from hypothesis import HealthCheck, given, settings
from hypothesis.database import DirectoryBasedExampleDatabase
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOLERANCE = 1e-10
_DATA_DIR = Path(__file__).resolve().parent
FAILURE_DIR = str(_DATA_DIR / "fixtures" / "parity_failures")

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

COCO_KPT_OKS_SIGMAS = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]

# ---------------------------------------------------------------------------
# stdout suppression (Rust println! bypasses Python sys.stdout)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def suppress_stdout():
    """Suppress stdout at the file-descriptor level."""
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


def run_both(gt_dataset, dt_results, iou_type):
    """Run evaluation through both pycocotools and hotcoco, return stats."""
    gt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    dt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(gt_dataset, gt_file)
        gt_file.close()
        json.dump(dt_results, dt_file)
        dt_file.close()

        with suppress_stdout():
            # pycocotools
            py_gt = PyCOCO(gt_file.name)
            py_dt = py_gt.loadRes(dt_file.name)
            py_ev = PyCOCOeval(py_gt, py_dt, iou_type)
            py_ev.evaluate()
            py_ev.accumulate()
            py_ev.summarize()
            py_stats = py_ev.stats.tolist()

            # hotcoco
            rs_gt = RsCOCO(gt_file.name)
            rs_dt = rs_gt.load_res(dt_file.name)
            rs_ev = RsCOCOeval(rs_gt, rs_dt, iou_type)
            rs_ev.evaluate()
            rs_ev.accumulate()
            rs_ev.summarize()
            rs_stats = rs_ev.stats

        return py_stats, rs_stats
    finally:
        os.unlink(gt_file.name)
        os.unlink(dt_file.name)


def assert_metrics_match(py_stats, rs_stats, iou_type, gt_dataset=None, dt_results=None):
    """Assert all metrics match within tolerance."""
    expected_len = 10 if iou_type == "keypoints" else 12
    assert len(py_stats) == expected_len, f"pycocotools returned {len(py_stats)} metrics, expected {expected_len}"
    assert len(rs_stats) == expected_len, f"hotcoco returned {len(rs_stats)} metrics, expected {expected_len}"

    metric_names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    if iou_type == "keypoints":
        metric_names = ["AP", "AP50", "AP75", "APm", "APl", "AR1", "AR10", "AR100", "ARm", "ARl"]

    mismatches = []
    for i in range(expected_len):
        py_val, rs_val = py_stats[i], rs_stats[i]
        # Both -1.0 means "not computed" — skip
        if py_val == -1.0 and rs_val == -1.0:
            continue
        diff = abs(py_val - rs_val)
        if diff > TOLERANCE:
            mismatches.append(f"  [{i}] {metric_names[i]}: py={py_val:.15f} rs={rs_val:.15f} diff={diff:.2e}")

    if mismatches:
        if gt_dataset is not None and dt_results is not None:
            save_failure(gt_dataset, dt_results, iou_type, py_stats, rs_stats)
        msg = f"\n{iou_type} metric mismatch (tol={TOLERANCE}):\n" + "\n".join(mismatches)
        raise AssertionError(msg)


def save_failure(gt_dataset, dt_results, iou_type, py_stats, rs_stats):
    """Save failing case to disk for debugging."""
    os.makedirs(FAILURE_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    prefix = os.path.join(FAILURE_DIR, f"{iou_type}_{ts}")

    with open(f"{prefix}_gt.json", "w") as f:
        json.dump(gt_dataset, f, indent=2)
    with open(f"{prefix}_dt.json", "w") as f:
        json.dump(dt_results, f, indent=2)

    metric_names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    if iou_type == "keypoints":
        metric_names = ["AP", "AP50", "AP75", "APm", "APl", "AR1", "AR10", "AR100", "ARm", "ARl"]

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
    database=DirectoryBasedExampleDatabase(str(_DATA_DIR / "fixtures" / ".hypothesis")),
)


@given(data=st.data())
@settings(**HYPOTHESIS_SETTINGS)
def test_bbox_parity(data):
    gt_dataset, dt_results = data.draw(coco_eval_data("bbox"))
    py_stats, rs_stats = run_both(gt_dataset, dt_results, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox", gt_dataset, dt_results)


@given(data=st.data())
@settings(**HYPOTHESIS_SETTINGS)
def test_segm_parity(data):
    gt_dataset, dt_results = data.draw(coco_eval_data("segm"))
    py_stats, rs_stats = run_both(gt_dataset, dt_results, "segm")
    assert_metrics_match(py_stats, rs_stats, "segm", gt_dataset, dt_results)


@given(data=st.data())
@settings(**HYPOTHESIS_SETTINGS)
def test_kpt_parity(data):
    gt_dataset, dt_results = data.draw(coco_eval_data("keypoints"))
    py_stats, rs_stats = run_both(gt_dataset, dt_results, "keypoints")
    assert_metrics_match(py_stats, rs_stats, "keypoints", gt_dataset, dt_results)


# ---------------------------------------------------------------------------
# Dedicated edge-case tests
# ---------------------------------------------------------------------------


def _make_minimal_gt(iou_type, images=None, categories=None, annotations=None):
    """Build a minimal GT dataset dict."""
    if images is None:
        images = [{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg"}]
    if categories is None:
        if iou_type == "keypoints":
            categories = [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": COCO_KEYPOINT_NAMES,
                    "skeleton": COCO_SKELETON,
                }
            ]
        else:
            categories = [{"id": 1, "name": "cat", "supercategory": "none"}]
    if annotations is None:
        annotations = []
    return {"images": images, "annotations": annotations, "categories": categories}


def _make_bbox_ann(ann_id, img_id=1, cat_id=1, bbox=None, iscrowd=0, **extra):
    """Shortcut to create a bbox GT annotation."""
    if bbox is None:
        bbox = [10.0, 10.0, 100.0, 100.0]
    ann = {
        "id": ann_id,
        "image_id": img_id,
        "category_id": cat_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": iscrowd,
    }
    ann.update(extra)
    return ann


def _make_bbox_det(img_id=1, cat_id=1, bbox=None, score=0.9):
    """Shortcut to create a bbox detection."""
    if bbox is None:
        bbox = [10.0, 10.0, 100.0, 100.0]
    return {"image_id": img_id, "category_id": cat_id, "bbox": bbox, "score": score}


def test_empty_gt():
    """No GT annotations, some detections → all metrics -1.0."""
    for iou_type in ["bbox", "segm"]:
        gt = _make_minimal_gt(iou_type)
        dts = [_make_bbox_det(score=0.5)]
        py_stats, rs_stats = run_both(gt, dts, iou_type)
        assert_metrics_match(py_stats, rs_stats, iou_type)
        # All AP metrics should be -1 when no GT exists
        assert all(s == -1.0 for s in py_stats[:6]), f"Expected -1.0 AP metrics, got {py_stats[:6]}"


def test_all_crowd():
    """Every GT is iscrowd=1."""
    anns = [
        _make_bbox_ann(1, bbox=[10, 10, 100, 100], iscrowd=1),
        _make_bbox_ann(2, bbox=[200, 200, 50, 50], iscrowd=1),
    ]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[10, 10, 100, 100], score=0.9), _make_bbox_det(bbox=[200, 200, 50, 50], score=0.5)]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_identical_boxes():
    """All GTs and DTs have the exact same bbox."""
    bbox = [50.0, 50.0, 200.0, 150.0]
    anns = [_make_bbox_ann(i + 1, bbox=bbox) for i in range(3)]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=bbox, score=round(0.3 + i * 0.3, 1)) for i in range(3)]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_area_at_boundaries():
    """Annotations with area exactly at 1024 and 9216 (small/medium/large boundaries)."""
    # area = 1024 → sqrt(1024) ≈ 32
    anns = [
        _make_bbox_ann(1, bbox=[10.0, 10.0, 32.0, 32.0]),  # area = 1024
        _make_bbox_ann(2, bbox=[100.0, 100.0, 96.0, 96.0]),  # area = 9216
    ]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [
        _make_bbox_det(bbox=[10.0, 10.0, 32.0, 32.0], score=0.8),
        _make_bbox_det(bbox=[100.0, 100.0, 96.0, 96.0], score=0.7),
    ]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_single_image_single_cat():
    """Minimal dataset: one image, one category, one GT, one DT."""
    anns = [_make_bbox_ann(1, bbox=[100.0, 100.0, 50.0, 50.0])]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[100.0, 100.0, 50.0, 50.0], score=1.0)]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_zero_area_boxes():
    """Boxes with zero width or height."""
    anns = [
        _make_bbox_ann(1, bbox=[10.0, 10.0, 0.0, 50.0]),  # zero width
        _make_bbox_ann(2, bbox=[50.0, 50.0, 50.0, 0.0]),  # zero height
        _make_bbox_ann(3, bbox=[100.0, 100.0, 80.0, 80.0]),  # normal
    ]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [
        _make_bbox_det(bbox=[10.0, 10.0, 0.0, 50.0], score=0.9),
        _make_bbox_det(bbox=[50.0, 50.0, 50.0, 0.0], score=0.8),
        _make_bbox_det(bbox=[100.0, 100.0, 80.0, 80.0], score=0.7),
    ]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_many_detections_few_gt():
    """Many detections for a single GT — tests maxDet handling."""
    anns = [_make_bbox_ann(1, bbox=[100.0, 100.0, 50.0, 50.0])]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[100.0 + i * 2, 100.0, 50.0, 50.0], score=round(1.0 - i * 0.005, 4)) for i in range(200)]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_kpt_no_visible():
    """Keypoint GT with num_keypoints=0 should be ignored."""
    kpts_zero = [0, 0, 0] * 17
    kpts_some = []
    for i in range(17):
        kpts_some.extend([float(100 + i * 10), float(100 + i * 5), 2])

    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [50.0, 50.0, 200.0, 200.0],
            "area": 40000.0,
            "iscrowd": 0,
            "keypoints": kpts_zero,
            "num_keypoints": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [50.0, 50.0, 200.0, 200.0],
            "area": 40000.0,
            "iscrowd": 0,
            "keypoints": kpts_some,
            "num_keypoints": 17,
        },
    ]
    gt = _make_minimal_gt("keypoints", annotations=anns)
    dts = [{"image_id": 1, "category_id": 1, "bbox": [50.0, 50.0, 200.0, 200.0], "score": 0.9, "keypoints": kpts_some}]
    py_stats, rs_stats = run_both(gt, dts, "keypoints")
    assert_metrics_match(py_stats, rs_stats, "keypoints")


def test_segm_polygon_rasterization():
    """Segmentation with polygon GTs and bbox DTs — tests the full segm pipeline."""
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 100.0, 100.0],
            "area": 10000.0,
            "iscrowd": 0,
            "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [200.0, 200.0, 50.0, 50.0],
            "area": 2500.0,
            "iscrowd": 0,
            "segmentation": [[200, 200, 250, 200, 250, 250, 200, 250]],
        },
    ]
    gt = _make_minimal_gt("segm", annotations=anns)
    dts = [
        _make_bbox_det(bbox=[10.0, 10.0, 100.0, 100.0], score=0.95),
        _make_bbox_det(bbox=[200.0, 200.0, 50.0, 50.0], score=0.8),
    ]
    py_stats, rs_stats = run_both(gt, dts, "segm")
    assert_metrics_match(py_stats, rs_stats, "segm")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x", "--tb=short"]))
