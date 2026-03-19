"""Fast parity regression tests for CI.

Hand-crafted edge-case tests (bbox/segm/keypoints) and OID evaluation tests.
All tests run in under 30 seconds and are safe to run on every commit.

To hunt for new parity bugs, use the hypothesis fuzzer instead:
    just fuzz

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

import pytest
from hotcoco import COCO, COCOeval, Hierarchy
from hotcoco import COCO as RsCOCO
from hotcoco import COCOeval as RsCOCOeval
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

# ---------------------------------------------------------------------------
# Constants
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

TOLERANCE = 1e-10

# ---------------------------------------------------------------------------
# Helpers
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
            py_gt = PyCOCO(gt_file.name)
            py_dt = py_gt.loadRes(dt_file.name)
            py_ev = PyCOCOeval(py_gt, py_dt, iou_type)
            py_ev.evaluate()
            py_ev.accumulate()
            py_ev.summarize()
            py_stats = py_ev.stats.tolist()

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


def assert_metrics_match(py_stats, rs_stats, iou_type):
    """Assert all metrics match within tolerance."""
    expected_len = 10 if iou_type == "keypoints" else 12
    assert len(py_stats) == expected_len
    assert len(rs_stats) == expected_len

    metric_names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    if iou_type == "keypoints":
        metric_names = ["AP", "AP50", "AP75", "APm", "APl", "AR1", "AR10", "AR100", "ARm", "ARl"]

    mismatches = []
    for i in range(expected_len):
        py_val, rs_val = py_stats[i], rs_stats[i]
        if py_val == -1.0 and rs_val == -1.0:
            continue
        diff = abs(py_val - rs_val)
        if diff > TOLERANCE:
            mismatches.append(f"  [{i}] {metric_names[i]}: py={py_val:.15f} rs={rs_val:.15f} diff={diff:.2e}")

    if mismatches:
        raise AssertionError(f"\n{iou_type} metric mismatch (tol={TOLERANCE}):\n" + "\n".join(mismatches))


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


# ---------------------------------------------------------------------------
# OID helpers
# ---------------------------------------------------------------------------


def _make_coco(dataset: dict) -> COCO:
    """Write dataset to a temp file and load as COCO."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dataset, f)
        path = f.name
    try:
        return COCO(path)
    finally:
        os.unlink(path)


def _img(id: int = 1) -> dict:
    return {"id": id, "file_name": f"img{id}.jpg", "height": 640, "width": 640}


def _ann(
    id: int,
    image_id: int,
    category_id: int,
    bbox: list,
    area: float,
    score: float | None = None,
    is_group_of: bool | None = None,
) -> dict:
    ann = {
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
    }
    if score is not None:
        ann["score"] = score
    if is_group_of is not None:
        ann["is_group_of"] = is_group_of
    return ann


def _cat(id: int, name: str, supercategory: str | None = None) -> dict:
    cat = {"id": id, "name": name}
    if supercategory is not None:
        cat["supercategory"] = supercategory
    return cat


# ---------------------------------------------------------------------------
# Edge-case tests: bbox / segm / keypoints
# ---------------------------------------------------------------------------


def test_empty_gt():
    """No GT annotations, some detections → all metrics -1.0."""
    for iou_type in ["bbox", "segm"]:
        gt = _make_minimal_gt(iou_type)
        dts = [_make_bbox_det(score=0.5)]
        py_stats, rs_stats = run_both(gt, dts, iou_type)
        assert_metrics_match(py_stats, rs_stats, iou_type)
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
# OID evaluation tests
# ---------------------------------------------------------------------------


def test_basic_hierarchy():
    """Dog detection on Poodle GT: correct at Dog level, wrong at Poodle."""
    hierarchy = Hierarchy.from_parent_map({1: 2, 2: 3})

    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000)],  # Poodle
            "categories": [
                _cat(1, "poodle", "dog"),
                _cat(2, "dog", "animal"),
                _cat(3, "animal"),
            ],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 2, [10, 10, 100, 100], 10000, score=0.9)],  # Dog
            "categories": [
                _cat(1, "poodle", "dog"),
                _cat(2, "dog", "animal"),
                _cat(3, "animal"),
            ],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev.run()
    results = ev.results(per_class=True)

    per_class = results["per_class"]
    assert "dog" in per_class, f"Expected 'dog' in per_class, got {list(per_class.keys())}"
    assert abs(per_class["dog"] - 1.0) < 1e-6, f"Dog AP should be 1.0, got {per_class['dog']}"


def test_group_of_multi_match():
    """3 detections on 1 group-of GT: all should be TPs."""
    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [300, 300, 100, 100], 10000),  # Normal GT
                _ann(2, 1, 1, [0, 0, 200, 200], 40000, is_group_of=True),  # Group-of
            ],
            "categories": [_cat(1, "person")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [300, 300, 100, 100], 10000, score=0.9),  # Matches normal GT
                _ann(2, 1, 1, [10, 10, 80, 80], 6400, score=0.8),  # Matches group-of
                _ann(3, 1, 1, [50, 50, 80, 80], 6400, score=0.7),  # Also matches group-of
            ],
            "categories": [_cat(1, "person")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    assert ev.stats[0] > 0.99, f"AP should be ~1.0 with group-of multi-match, got {ev.stats[0]:.4f}"


def test_group_of_no_fn():
    """Unmatched group-of GT should not count as FN."""
    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [0, 0, 100, 100], 10000),  # Normal GT
                _ann(2, 1, 1, [400, 400, 100, 100], 10000, is_group_of=True),  # Group-of (far away)
            ],
            "categories": [_cat(1, "person")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [0, 0, 100, 100], 10000, score=0.9),  # Matches normal GT only
            ],
            "categories": [_cat(1, "person")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    assert ev.stats[0] > 0.99, f"AP should be ~1.0 (unmatched group-of not FN), got {ev.stats[0]:.4f}"


def test_pre_expanded_idempotent():
    """Pre-expanded GTs should produce same results as unexpanded."""
    hierarchy = Hierarchy.from_parent_map({1: 2})

    gt_unexpanded = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    gt_expanded = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [10, 10, 100, 100], 10000),
                _ann(2, 1, 2, [10, 10, 100, 100], 10000),  # Expanded animal
            ],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )

    ev1 = COCOeval(gt_unexpanded, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev1.run()

    dt2 = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    ev2 = COCOeval(gt_expanded, dt2, "bbox", oid_style=True, hierarchy=hierarchy)
    ev2.run()

    assert abs(ev1.stats[0] - ev2.stats[0]) < 1e-10, (
        f"Pre-expanded should match unexpanded: {ev1.stats[0]:.6f} vs {ev2.stats[0]:.6f}"
    )


def test_dt_expansion():
    """expand_dt=True: Dog prediction gets credit at Animal level."""
    hierarchy = Hierarchy.from_parent_map({1: 2})

    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 2, [10, 10, 100, 100], 10000)],  # Animal GT
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],  # Dog
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )

    ev1 = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev1.run()
    ap_no_expand = ev1.stats[0]

    gt2 = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 2, [10, 10, 100, 100], 10000)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    dt2 = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )

    ev2 = COCOeval(gt2, dt2, "bbox", oid_style=True, hierarchy=hierarchy)
    p = ev2.params
    p.expand_dt = True
    ev2.params = p
    ev2.run()
    ap_expand = ev2.stats[0]

    assert ap_expand > ap_no_expand, f"DT expansion should improve AP: {ap_expand:.4f} vs {ap_no_expand:.4f}"


def test_virtual_nodes():
    """Supercategory not in categories list: should still work via virtual node."""
    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000)],
            "categories": [_cat(1, "chair", "furniture")],  # "furniture" → virtual node
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],
            "categories": [_cat(1, "chair", "furniture")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    # chair matches (AP=1.0), virtual_furniture has GT but no DT (AP=0.0) → mean=0.5
    assert abs(ev.stats[0] - 0.5) < 1e-6, f"AP should be 0.5 (chair + virtual), got {ev.stats[0]:.4f}"
    results = ev.results(per_class=True)
    assert results["per_class"]["chair"] > 0.99, (
        f"Chair per-class AP should be 1.0, got {results['per_class']['chair']:.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x", "--tb=short"]))
