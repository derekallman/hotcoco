"""Fast parity regression tests for CI.

Hand-crafted edge-case tests (bbox/segm/keypoints) and OID evaluation tests.
All tests run in under 30 seconds and are safe to run on every commit.

To hunt for new parity bugs, use the hypothesis fuzzer instead:
    just fuzz

Usage:
    uv run pytest tests/test_parity.py -v -x --tb=short
    just test
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from helpers import COCO_KEYPOINT_NAMES, COCO_SKELETON, assert_metrics_match, run_both, suppress_output, written_json
from hotcoco import COCO, COCOeval, Hierarchy, mask

# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# OID helpers
# ---------------------------------------------------------------------------


def _img(id: int = 1) -> dict:
    return {"id": id, "file_name": f"img{id}.jpg", "height": 640, "width": 640}


def _cat(id: int, name: str, supercategory: str | None = None) -> dict:
    cat = {"id": id, "name": name}
    if supercategory is not None:
        cat["supercategory"] = supercategory
    return cat


# ---------------------------------------------------------------------------
# Edge-case tests: bbox / segm / keypoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_empty_gt(iou_type):
    """No GT annotations, some detections → all metrics -1.0.

    Every metric here is the sentinel on both sides, so this asserts agreement
    about undefinedness and nothing numeric. That is the whole point of the case,
    and the expectation is pinned on *hotcoco*: checking ``py_stats`` alone would
    test pycocotools against itself.
    """
    gt = _make_minimal_gt(iou_type)
    dts = [_make_bbox_det(score=0.5)]
    py_stats, rs_stats, _ = run_both(gt, dts, iou_type)
    assert_metrics_match(py_stats, rs_stats, iou_type)
    assert all(s == -1.0 for s in rs_stats[:6]), f"hotcoco: expected -1.0 AP metrics, got {rs_stats[:6]}"
    assert all(s == -1.0 for s in py_stats[:6]), f"pycocotools: expected -1.0 AP metrics, got {py_stats[:6]}"


def test_all_crowd():
    """Every GT is iscrowd=1."""
    anns = [
        _make_bbox_ann(1, bbox=[10, 10, 100, 100], iscrowd=1),
        _make_bbox_ann(2, bbox=[200, 200, 50, 50], iscrowd=1),
    ]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[10, 10, 100, 100], score=0.9), _make_bbox_det(bbox=[200, 200, 50, 50], score=0.5)]
    py_stats, rs_stats, _ = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")
    # Crowd GTs are ignored *and* rematchable: the detections they absorb are
    # neither TPs nor FPs, and no non-crowd GT remains to measure recall against.
    # Every metric is therefore undefined rather than zero — pin that on hotcoco,
    # since agreement alone would also hold if both sides were wrong the same way.
    assert all(s == -1.0 for s in rs_stats), f"hotcoco: all-crowd GT should give all -1.0, got {rs_stats}"


def test_identical_boxes():
    """All GTs and DTs have the exact same bbox."""
    bbox = [50.0, 50.0, 200.0, 150.0]
    anns = [_make_bbox_ann(i + 1, bbox=bbox) for i in range(3)]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=bbox, score=round(0.3 + i * 0.3, 1)) for i in range(3)]
    py_stats, rs_stats, _ = run_both(gt, dts, "bbox")
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
    py_stats, rs_stats, _ = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_single_image_single_cat():
    """Minimal dataset: one image, one category, one GT, one DT."""
    anns = [_make_bbox_ann(1, bbox=[100.0, 100.0, 50.0, 50.0])]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[100.0, 100.0, 50.0, 50.0], score=1.0)]
    py_stats, rs_stats, _ = run_both(gt, dts, "bbox")
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
    py_stats, rs_stats, _ = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_many_detections_few_gt():
    """Many detections for a single GT — tests maxDet handling."""
    anns = [_make_bbox_ann(1, bbox=[100.0, 100.0, 50.0, 50.0])]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[100.0 + i * 2, 100.0, 50.0, 50.0], score=round(1.0 - i * 0.005, 4)) for i in range(200)]
    py_stats, rs_stats, _ = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


# ---------------------------------------------------------------------------
# Accumulated arrays, bit for bit
# ---------------------------------------------------------------------------


class _Lcg:
    """Deterministic 64-bit LCG so the score ties below never depend on a seed
    anyone can change."""

    def __init__(self, state: int) -> None:
        self.state = state

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.state >> 33

    def score(self) -> float:
        """Two-decimal scores: ~100 distinct values over ~500 detections."""
        return (self.next() % 100) / 100.0


def _tie_heavy_dataset():
    """12 images x 3 categories, 14 detections per cell, scores quantized to two
    decimals so equal scores span images, plus one hand-built tie: a TP in image 1
    and an FP in image 2 both scoring exactly 0.70. Category 3 has no ground truth.
    Same data, generator, and seed as ``tie_heavy_datasets`` in
    ``crates/hotcoco/tests/integration_test.rs``; change both together.

    Every cell holds more detections than the middle ``maxDets`` cap, so per-image
    truncation fires, and the cross-image ties make the stable tie-break inside
    ``accumulate()`` observable in the precision curve.
    """
    sizes = [20.0, 50.0, 120.0]  # small, medium, large by COCO area
    dets_per_cell = 14
    rng = _Lcg(0x9E3779B97F4A7C15)
    images = [{"id": i, "width": 640, "height": 480, "file_name": f"{i}.jpg"} for i in range(1, 13)]
    no_gt_cat = 3
    categories = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": no_gt_cat, "name": "no-gt"}]
    anns, dts = [], []
    for img_id in range(1, 13):
        for cat_id in range(1, 4):
            n_gt = 0 if cat_id == no_gt_cat else (img_id + cat_id) % 4
            cell = []
            for j in range(n_gt):
                size = sizes[(j + cat_id) % 3]
                gt_box = [20.0 + 130.0 * j, 20.0 + 150.0 * cat_id, size, size]
                anns.append(_make_bbox_ann(len(anns) + 1, img_id, cat_id, bbox=gt_box))
                # Two pixels off the box: IoU well above 0.5 at every size.
                cell.append(([gt_box[0] + 2.0, gt_box[1] + 2.0, size, size], rng.score()))
            while len(cell) < dets_per_cell:
                size = sizes[rng.next() % 3]
                # Far below every GT row, so never a match.
                cell.append(([float(rng.next() % 500), 480.0, size, size], rng.score()))
            dts.extend(_make_bbox_det(img_id, cat_id, bbox=box, score=s) for box, s in cell)
    # Image 1 / category 1 has three GT boxes; the first is [20, 170, 50, 50].
    assert any(a["image_id"] == 1 and a["category_id"] == 1 and a["bbox"] == [20.0, 170.0, 50.0, 50.0] for a in anns)
    dts.append(_make_bbox_det(1, 1, bbox=[22.0, 172.0, 50.0, 50.0], score=0.70))
    dts.append(_make_bbox_det(2, 1, bbox=[300.0, 480.0, 50.0, 50.0], score=0.70))
    return _make_minimal_gt("bbox", images=images, categories=categories, annotations=anns), dts


def _accumulated_arrays(gt, dts, max_dets, acc_max_dets=None):
    """Run evaluate + accumulate through both tools and return the two ``eval``
    dicts. ``acc_max_dets`` re-assigns the caps between the two calls — the
    pycocotools ``accumulate(p)`` idiom, which leaves cells holding more
    detections than the current cap."""
    from pycocotools.coco import COCO as PyCOCO  # noqa: PLC0415
    from pycocotools.cocoeval import COCOeval as PyCOCOeval  # noqa: PLC0415

    with written_json(gt, dts) as (gt_path, dt_path):
        with suppress_output():
            py_gt = PyCOCO(gt_path)
            py_ev = PyCOCOeval(py_gt, py_gt.loadRes(dt_path), "bbox")
            py_ev.params.maxDets = list(max_dets)
            py_ev.evaluate()
            if acc_max_dets is not None:
                py_ev.params.maxDets = list(acc_max_dets)
            py_ev.accumulate()

        rs_gt = COCO(gt_path)
        rs_ev = COCOeval(rs_gt, rs_gt.load_res(dt_path), "bbox")
        rs_ev.params.max_dets = list(max_dets)
        rs_ev.evaluate()
        if acc_max_dets is not None:
            rs_ev.params.max_dets = list(acc_max_dets)
        rs_ev.accumulate()
    return py_ev.eval, rs_ev.eval


def _assert_arrays_bit_equal(py_eval, rs_eval, what):
    for key in ("precision", "recall", "scores"):
        py_arr, rs_arr = np.asarray(py_eval[key]), np.asarray(rs_eval[key])
        assert py_arr.shape == rs_arr.shape, f"{what}: {key} shape {py_arr.shape} vs {rs_arr.shape}"
        mismatch = np.argwhere(py_arr.view(np.uint64) != rs_arr.view(np.uint64))
        assert mismatch.size == 0, (
            f"{what}: {key} differs at {len(mismatch)} positions, first (t, r, k, a, m)={mismatch[0].tolist()}: "
            f"pycocotools {py_arr[tuple(mismatch[0])]!r} vs hotcoco {rs_arr[tuple(mismatch[0])]!r}"
        )


def _rank_filler_dataset():
    """Two images, one category: image 1 has a small GT with a TP at 0.5 and an
    FP at 0.4; image 2 has no GT and one large detection at 0.9."""
    images = [{"id": i, "width": 200, "height": 200, "file_name": f"{i}.jpg"} for i in (1, 2)]
    categories = [{"id": 1, "name": "object"}]
    anns = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0], "area": 100.0, "iscrowd": 0}]
    dts = [
        _make_bbox_det(1, 1, bbox=[0.0, 0.0, 10.0, 10.0], score=0.5),
        _make_bbox_det(1, 1, bbox=[50.0, 50.0, 10.0, 10.0], score=0.4),
        _make_bbox_det(2, 1, bbox=[0.0, 0.0, 100.0, 100.0], score=0.9),
    ]
    return _make_minimal_gt("bbox", images=images, categories=categories, annotations=anns), dts


def test_rank_filler_and_epsilon_guard_match_pycocotools_bit_for_bit():
    """Under ``small``, image 2's cell has detections but no GT and every
    detection is area-ignored; pycocotools keeps the cell (see ``gather_pair``
    in matching.rs), so the 0.9 fills rank 0 of ``scores``. Image 1's lone
    leading TP then shows the ``np.spacing(1)`` guard in ``precision``."""
    gt, dts = _rank_filler_dataset()
    py_eval, rs_eval = _accumulated_arrays(gt, dts, [1, 10, 100])
    small = 1  # area-range index: all, small, medium, large
    # (t=IoU 0.5, r=recall 0, k=cat 1, a=small, m=maxDets 100)
    assert np.asarray(py_eval["scores"])[0, 0, 0, small, 2] == 0.9, "fixture must put the ignored 0.9 at rank 0"
    # Under ``small`` the 0.9 is ignored, so the 0.5 TP is a lone leading TP.
    assert np.asarray(py_eval["precision"])[0, 0, 0, small, 2] == 1.0 - 2.0**-52, (
        "fixture must expose the epsilon guard"
    )
    _assert_arrays_bit_equal(py_eval, rs_eval, "rank filler + epsilon guard")


def test_accumulated_arrays_match_pycocotools_bit_for_bit():
    """``precision``, ``recall`` and ``scores`` equal pycocotools' exactly, not
    within a tolerance, on a dataset built to expose ranking order.

    Each headline AP averages 1,010 precision points, so one tie broken the wrong
    way moves it by about 1e-4 — inside the tolerance the val2017 parity run
    accepts — and the fast suite has no other dataset where scores collide across
    images while ``maxDets`` truncates. The arrays compare every point and name
    the first ``(t, r, k, a, m)`` that differs. Two cases: the default caps, and
    caps lowered between ``evaluate()`` and ``accumulate()``.
    """
    gt, dts = _tie_heavy_dataset()
    scores = [d["score"] for d in dts]
    assert len(set(scores)) < len(scores) // 4, "scores must collide across cells for the tie-break to matter"

    py_eval, rs_eval = _accumulated_arrays(gt, dts, [1, 10, 100])
    precision = np.asarray(py_eval["precision"])
    assert not np.array_equal(precision[..., 0], precision[..., 1]), "cap 1 and cap 10 must yield different curves"
    assert ((precision > 0) & (precision < 1)).any(), "curves must be non-trivial"
    _assert_arrays_bit_equal(py_eval, rs_eval, "maxDets=[1, 10, 100]")

    lowered = [10, 1]
    py_eval, rs_eval = _accumulated_arrays(gt, dts, [1, 10, 100], acc_max_dets=lowered)
    assert np.asarray(rs_eval["precision"]).shape[-1] == len(lowered)
    _assert_arrays_bit_equal(py_eval, rs_eval, "evaluate maxDets=[1, 10, 100], accumulate maxDets=[10, 1]")


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
    py_stats, rs_stats, _ = run_both(gt, dts, "keypoints")
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
    py_stats, rs_stats, _ = run_both(gt, dts, "segm")
    assert_metrics_match(py_stats, rs_stats, "segm")


def _bytes_rle():
    """An RLE with ``counts`` as bytes, exactly what ``mask.encode`` returns."""
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:5, 2:5] = 1
    rle = mask.encode(np.asfortranarray(m))
    assert isinstance(rle["counts"], bytes)
    return rle


def _dataset_with_segmentation(segmentation):
    return {
        "images": [{"id": 1, "height": 10, "width": 10, "file_name": "test.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2.0, 2.0, 3.0, 3.0],
                "area": 9.0,
                "iscrowd": 0,
                "segmentation": segmentation,
            }
        ],
        "categories": [{"id": 1, "name": "cat", "supercategory": "none"}],
    }


def test_bytes_rle_round_trips_through_coco():
    """encode() -> COCO() -> load_anns() -> decode() keeps the mask.

    `mask.encode` returns `counts` as bytes (pycocotools' format). A `bytes`
    object is a Python sequence of ints, so extracting it as a list of counts
    succeeds and silently produces a garbage uncompressed RLE instead of
    raising — the failure surfaced as segmentation AP of exactly 0.000.
    """
    rle = _bytes_rle()
    coco = COCO(_dataset_with_segmentation(rle))
    loaded = coco.load_anns([1])[0]["segmentation"]
    assert mask.decode(loaded).sum() == 9


def test_bytes_rle_round_trips_through_load_res():
    """The same path for detections, which is how mask consumers evaluate."""
    rle = _bytes_rle()
    coco = COCO(_dataset_with_segmentation(rle))
    # No bbox, so `load_res` treats the results as segm and derives area from
    # the mask — a wrong parse shows up as a wrong area, not only a wrong mask.
    dt = coco.load_res([{"image_id": 1, "category_id": 1, "score": 0.9, "segmentation": rle}])
    ann = dt.load_anns(dt.get_ann_ids())[0]
    assert mask.decode(ann["segmentation"]).sum() == 9
    assert ann["area"] == 9.0


def test_every_counts_form_agrees():
    """bytes, str, and an uncompressed list of ints must all load the same mask."""
    rle = _bytes_rle()
    as_str = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
    # Column-major runs for a 3x3 block at rows 2-4, cols 2-4 of a 10x10 mask.
    as_list = {"size": [10, 10], "counts": [22, 3, 7, 3, 7, 3, 55]}
    decoded = [
        mask.decode(COCO(_dataset_with_segmentation(seg)).load_anns([1])[0]["segmentation"])
        for seg in (rle, as_str, as_list)
    ]
    assert decoded[0].sum() == 9
    assert (decoded[0] == decoded[1]).all()
    assert (decoded[0] == decoded[2]).all()


def test_unsupported_counts_type_raises():
    """Anything that is not str, bytes, or a list of ints is an error, not an empty mask."""
    with pytest.raises(TypeError, match="counts"):
        COCO(_dataset_with_segmentation({"size": [10, 10], "counts": 3.5}))


# ---------------------------------------------------------------------------
# OID evaluation tests
# ---------------------------------------------------------------------------


def test_basic_hierarchy():
    """Dog detection on Poodle GT: correct at Dog level, wrong at Poodle."""
    hierarchy = Hierarchy.from_parent_map({1: 2, 2: 3})

    gt = COCO(
        {
            "images": [_img()],
            "annotations": [_make_bbox_ann(1, 1, 1, [10, 10, 100, 100])],  # Poodle
            "categories": [_cat(1, "poodle", "dog"), _cat(2, "dog", "animal"), _cat(3, "animal")],
        }
    )
    dt = COCO(
        {
            "images": [_img()],
            "annotations": [_make_bbox_ann(1, 1, 2, [10, 10, 100, 100], score=0.9)],  # Dog
            "categories": [_cat(1, "poodle", "dog"), _cat(2, "dog", "animal"), _cat(3, "animal")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev.run()
    results = ev.results(per_class=True)

    per_class = results["per_class"]
    assert "dog" in per_class, f"Expected 'dog' in per_class, got {list(per_class.keys())}"
    assert abs(per_class["dog"] - 1.0) < 1e-6, f"Dog AP should be 1.0, got {per_class['dog']}"


def test_group_of_scores_one_tp_and_absorbs_the_rest():
    """A group-of box is worth one TP; surplus detections inside it are ignored.

    Open Images protocol: "If at least one detection is inside group-of box a
    single True Positive is scored. ... Multiple correct detections inside the
    same group-of box is still count as a single True Positive."

    The old version asserted "all should be TPs" and used detections at IoU 0.16
    against the group box, so the group-of path never executed -- it passed on VOC
    interpolation. Both halves are fixed here.
    """
    gt = COCO(
        {
            "images": [_img()],
            "annotations": [
                _make_bbox_ann(1, 1, 1, [300, 300, 100, 100]),  # Normal GT
                _make_bbox_ann(2, 1, 1, [0, 0, 200, 200], is_group_of=True),  # Group-of
            ],
            "categories": [_cat(1, "person")],
        }
    )
    dt = COCO(
        {
            "images": [_img()],
            "annotations": [
                _make_bbox_ann(1, 1, 1, [300, 300, 100, 100], score=0.9),  # ordinary TP
                _make_bbox_ann(2, 1, 1, [10, 10, 80, 80], score=0.8),  # scores the group box
                _make_bbox_ann(3, 1, 1, [50, 50, 80, 80], score=0.7),  # absorbed, ignored
            ],
            "categories": [_cat(1, "person")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    assert ev.stats[0] > 0.99, f"both GTs found, surplus ignored -> AP ~1.0, got {ev.stats[0]:.4f}"

    e = ev.eval_imgs[0]
    ignored = [i for i, ig in zip(e["dtIds"], e["dtIgnore"][0]) if ig]
    assert ignored == [3], f"only the surplus detection should be ignored, got {ignored}"
    assert sum(e["gtInDenominator"]) == 2, "ordinary GT + group-of box = 2 in the denominator"


def test_group_of_matches_on_ioa_not_iou():
    """A detection smaller than the group-of box must still be absorbed.

    Group-of matching uses IoA (intersection / detection area), so a detection
    wholly inside the box scores 1.0 however small it is. Under plain IoU this
    80x80 detection scores 0.16 against the 200x200 box and leaks out as a false
    positive -- the defect this guards.

    The inside-the-box detection deliberately outranks the ordinary true positive
    so a regression lands its false positive *before* full recall. Reverse the
    scores and AP reads 1.0 either way, proving nothing.
    """
    gt = COCO(
        {
            "images": [_img()],
            "annotations": [
                _make_bbox_ann(1, 1, 1, [300, 300, 100, 100]),
                _make_bbox_ann(2, 1, 1, [0, 0, 200, 200], is_group_of=True),
            ],
            "categories": [_cat(1, "person")],
        }
    )
    dt = COCO(
        {
            "images": [_img()],
            "annotations": [
                _make_bbox_ann(1, 1, 1, [10, 10, 80, 80], score=0.9),
                _make_bbox_ann(2, 1, 1, [300, 300, 100, 100], score=0.8),
            ],
            "categories": [_cat(1, "person")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    # Matched on IoU instead, AP collapses to ~0.25.
    assert ev.stats[0] > 0.99, f"detection inside a group-of box must be absorbed, got {ev.stats[0]:.4f}"


def test_undetected_group_of_is_a_miss():
    """An undetected group-of box is a single false negative.

    Protocol: "Otherwise, the group-of box is counted as a single False Negative."
    This previously asserted the opposite, which is the Open Images *V2* metric
    (TF `group_of_weight=0.0`), not the Challenge metric hotcoco targets.
    """
    gt = COCO(
        {
            "images": [_img()],
            "annotations": [
                _make_bbox_ann(1, 1, 1, [0, 0, 100, 100]),  # Normal GT
                _make_bbox_ann(2, 1, 1, [400, 400, 100, 100], is_group_of=True),  # Group-of (far away)
            ],
            "categories": [_cat(1, "person")],
        }
    )
    dt = COCO(
        {
            "images": [_img()],
            "annotations": [
                _make_bbox_ann(1, 1, 1, [0, 0, 100, 100], score=0.9)  # Matches normal GT only
            ],
            "categories": [_cat(1, "person")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    assert abs(ev.stats[0] - 0.5) < 0.02, f"one of two GTs found -> AP ~0.5, got {ev.stats[0]:.4f}"


def test_pre_expanded_idempotent():
    """Pre-expanded GTs should produce same results as unexpanded."""
    hierarchy = Hierarchy.from_parent_map({1: 2})

    gt_unexpanded = COCO(
        {
            "images": [_img()],
            "annotations": [_make_bbox_ann(1, 1, 1, [10, 10, 100, 100])],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    gt_expanded = COCO(
        {
            "images": [_img()],
            "annotations": [
                _make_bbox_ann(1, 1, 1, [10, 10, 100, 100]),
                _make_bbox_ann(2, 1, 2, [10, 10, 100, 100]),  # Expanded animal
            ],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    dt = COCO(
        {
            "images": [_img()],
            "annotations": [_make_bbox_ann(1, 1, 1, [10, 10, 100, 100], score=0.9)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )

    ev1 = COCOeval(gt_unexpanded, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev1.run()

    ev2 = COCOeval(gt_expanded, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev2.run()

    assert abs(ev1.stats[0] - ev2.stats[0]) < 1e-10, (
        f"Pre-expanded should match unexpanded: {ev1.stats[0]:.6f} vs {ev2.stats[0]:.6f}"
    )


def test_dt_expansion():
    """expand_dt=True: Dog prediction gets credit at Animal level."""
    hierarchy = Hierarchy.from_parent_map({1: 2})

    gt = COCO(
        {
            "images": [_img()],
            "annotations": [_make_bbox_ann(1, 1, 2, [10, 10, 100, 100])],  # Animal GT
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    dt = COCO(
        {
            "images": [_img()],
            "annotations": [_make_bbox_ann(1, 1, 1, [10, 10, 100, 100], score=0.9)],  # Dog
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )

    ev1 = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev1.run()
    ap_no_expand = ev1.stats[0]

    ev2 = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    p = ev2.params
    p.expand_dt = True
    ev2.params = p
    ev2.run()
    ap_expand = ev2.stats[0]

    assert ap_expand > ap_no_expand, f"DT expansion should improve AP: {ap_expand:.4f} vs {ap_no_expand:.4f}"


def test_virtual_nodes():
    """Supercategory not in categories list: should still work via virtual node."""
    gt = COCO(
        {
            "images": [_img()],
            "annotations": [_make_bbox_ann(1, 1, 1, [10, 10, 100, 100])],
            "categories": [_cat(1, "chair", "furniture")],  # "furniture" → virtual node
        }
    )
    dt = COCO(
        {
            "images": [_img()],
            "annotations": [_make_bbox_ann(1, 1, 1, [10, 10, 100, 100], score=0.9)],
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
# EvalReport
# ---------------------------------------------------------------------------


def _eval_for_report(iou_type="bbox"):
    gt = _make_minimal_gt(
        iou_type, annotations=[_make_bbox_ann(1, bbox=[10, 10, 50, 50]), _make_bbox_ann(2, bbox=[100, 100, 40, 40])]
    )
    dt = [_make_bbox_det(bbox=[10, 10, 50, 50], score=0.9), _make_bbox_det(bbox=[100, 100, 40, 40], score=0.8)]
    coco_gt = COCO(gt)
    coco_dt = coco_gt.load_res(dt)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    with suppress_output(stderr=False):
        ev.run()
    return ev


def test_report_dict_shape():
    report = _eval_for_report().report()

    assert set(report) == {"task", "provenance", "metrics", "per_class", "per_group", "curves", "params"}
    assert report["task"] == "detection"
    assert report["metrics"]["AP"] == pytest.approx(1.0)
    # Nested per-class, not flattened "AP/name".
    assert all(isinstance(v, dict) for v in report["per_class"].values())


def test_report_provenance_distinguishes_extension_from_parity():
    """OBB has no reference implementation, so it must not read as leaderboard-comparable."""
    assert _eval_for_report("bbox").report()["provenance"] == "parity_verified"
    assert _eval_for_report("obb").report()["provenance"] == "extension"


def test_provenance_survives_into_the_render_layer():
    """The marker has to reach whatever draws the numbers, not stop at the dict.

    A PDF or dashboard is the artifact that gets circulated to someone who never
    ran the eval, so an unmarked one is the failure `Provenance` exists to
    prevent. `report()` computed it correctly for a whole release while every
    Python renderer ignored it.
    """
    from hotcoco.plot.data import PlotData

    verified = PlotData.from_coco_eval(_eval_for_report())
    assert verified.provenance == "parity_verified"
    assert verified.is_benchmark_standard
    assert verified.deviations == []

    # The sharp case. Still `iou_type="bbox"` in plain COCO mode, so a renderer
    # deriving the marker from eval_mode or geometry — the obvious shortcut —
    # would call this leaderboard-comparable.
    ev = _eval_for_report()
    ev.params.recThrs = [i / 10 for i in range(11)]
    with suppress_output(stderr=False), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.run()

    extension = PlotData.from_coco_eval(ev)
    assert extension.eval_mode == "coco" and extension.iou_type == "bbox"
    assert extension.provenance == "extension"
    assert not extension.is_benchmark_standard
    assert any("rec_thrs" in d for d in extension.deviations), extension.deviations


def test_provenance_accessor_sees_unevaluated_param_changes():
    """`ev.params.x = ...` mutates a Python-side copy that only syncs on entry.

    Reading the evaluator directly reported `parity_verified` for a run that was
    about to be an extension — comparability answered from stale state, which is
    worse than not answering. Checked here because the accessor's whole point is
    working *before* a long evaluation.
    """
    ev = _eval_for_report()
    assert ev.provenance() == "parity_verified"  # control

    off = _eval_for_report()
    off.params.iouThrs = [0.5]
    assert off.provenance() == "extension"
    assert any("iou_thrs" in d for d in off.reference_deviations())


def test_report_curves_are_plottable():
    report = _eval_for_report().report()
    curves = report["curves"]

    rec_thrs = curves["rec_thrs"]
    assert len(rec_thrs) == 101
    # One curve per IoU threshold, each sharing the rec_thrs x-axis.
    pr_curves = {k: v for k, v in curves.items() if k.startswith("pr@")}
    assert len(pr_curves) == 10
    for name, curve in pr_curves.items():
        assert len(curve) == len(rec_thrs), name


# ---------------------------------------------------------------------------
# Drop-in behaviors that are not about metric values
# ---------------------------------------------------------------------------


def test_params_in_place_mutation_takes_effect():
    """`ev.params.imgIds = [...]` must configure the run, as it does in pycocotools.

    This is the canonical idiom — it appears in pycocotools' own demo — and it
    used to be a silent no-op here: the `params` getter cloned into a fresh object
    each access, so the assignment mutated a temporary and evaluation proceeded
    over the whole dataset.

    Asserted through a *result*, not just the attribute, because reading the
    attribute back would also pass if params were a persistent object the
    evaluator never consulted.
    """
    anns = [_make_bbox_ann(1, img_id=1, bbox=[10, 10, 50, 50]), _make_bbox_ann(2, img_id=2, bbox=[10, 10, 50, 50])]
    images = [
        {"id": 1, "width": 640, "height": 480, "file_name": "a.jpg"},
        {"id": 2, "width": 640, "height": 480, "file_name": "b.jpg"},
    ]
    gt = _make_minimal_gt("bbox", images=images, annotations=anns)
    dts = [
        _make_bbox_det(img_id=1, bbox=[10, 10, 50, 50], score=0.9),
        _make_bbox_det(img_id=2, bbox=[500, 400, 20, 20], score=0.9),
    ]

    with written_json(gt, dts, quiet=True) as (gt_path, dt_path):
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.load_res(dt_path)

        both = COCOeval(coco_gt, coco_dt, "bbox")
        both.evaluate()
        both.accumulate()
        both.summarize()

        only_good = COCOeval(coco_gt, coco_dt, "bbox")
        only_good.params.imgIds = [1]
        assert list(only_good.params.imgIds) == [1], "params did not retain the assignment"
        only_good.evaluate()
        only_good.accumulate()
        only_good.summarize()

    # Image 1 is a perfect detection, image 2 is a total miss. Restricting to
    # image 1 must therefore score strictly higher.
    assert only_good.stats[0] > both.stats[0], (
        f"restricting to imgIds=[1] changed nothing: {only_good.stats[0]} vs {both.stats[0]} "
        "- params mutation is not reaching the evaluator"
    )


def test_non_reference_params_warn_and_downgrade_provenance():
    """Off-reference configuration is visible from Python, both ways.

    The Rust `summarize()` writes its warnings with `eprintln!`, straight to
    file descriptor 2 — which bypasses `sys.stderr`, so they would be invisible
    in a notebook, invisible to `capsys`, and uncatchable by
    `warnings.catch_warnings`. The binding emits real Python warnings instead.
    """
    gt = _make_minimal_gt("bbox", annotations=[_make_bbox_ann(1, bbox=[10, 10, 50, 50])])
    dts = [_make_bbox_det(bbox=[10, 10, 50, 50], score=0.9)]

    with written_json(gt, dts, quiet=True) as (gt_path, dt_path):
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.load_res(dt_path)

        ref = COCOeval(coco_gt, coco_dt, "bbox")
        ref.evaluate()
        ref.accumulate()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ref.summarize()
        assert not caught, f"default params should not warn, got {[str(w.message) for w in caught]}"
        assert ref.results()["provenance"] == "parity_verified"

        off = COCOeval(coco_gt, coco_dt, "bbox")
        off.params.recThrs = [i / 10 for i in range(11)]
        off.evaluate()
        off.accumulate()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            off.summarize()

    messages = [str(w.message) for w in caught]
    assert any("rec_thrs" in m for m in messages), f"expected a rec_thrs warning catchable from Python, got {messages}"
    # Provenance has to survive into the archived artifact, not just the report.
    assert off.results()["provenance"] == "extension"


# ---------------------------------------------------------------------------
# mask.encode dtypes
# ---------------------------------------------------------------------------


def test_encode_accepts_bool_masks():
    """A `bool` mask encodes to the same RLE as its `uint8` twin.

    pycocotools takes `uint8` only, but every torch-side consumer stores masks
    as `bool` -- TorchMetrics does -- so `bool` reaching the drop-in path is the
    common case, not a mistake.
    """
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 2:5] = True

    for arr in (m, np.asfortranarray(m)):
        assert mask.encode(arr) == mask.encode(arr.astype(np.uint8))


def test_encode_accepts_bool_mask_stacks():
    """The reported repro: a 3-D `(H, W, N)` Fortran-order `bool` stack."""
    m = np.zeros((10, 10, 1), dtype=bool)
    m[2:5, 2:5, 0] = True
    stack = np.asfortranarray(m)

    assert mask.encode(stack) == mask.encode(stack.astype(np.uint8))


def test_encode_rejects_wide_dtypes_with_a_readable_message():
    """A dtype that is not one byte wide is an error naming the dtype and the fix.

    The message is the point: the old failure was `'ndarray' object is not an
    instance of 'ndarray'`, which names neither the dtype nor the argument.
    """
    with pytest.raises(TypeError, match=r"float32.*astype"):
        mask.encode(np.zeros((10, 10), dtype=np.float32))

    with pytest.raises(TypeError, match=r"mask must be a numpy array"):
        mask.encode([[0, 1], [1, 0]])


# ---------------------------------------------------------------------------
# update_anns: edit loaded annotations without a rebuild
# ---------------------------------------------------------------------------


def _size_bucket_fixture():
    """One GT box of 400 px² (small) with a detection on top of it."""
    gt = _make_minimal_gt("bbox", annotations=[_make_bbox_ann(1, bbox=[10.0, 10.0, 20.0, 20.0])])
    coco_gt = COCO(gt)
    coco_dt = coco_gt.load_res([_make_bbox_det(bbox=[10.0, 10.0, 20.0, 20.0], score=0.9)])
    return coco_gt, coco_dt


def _bbox_stats(coco_gt, coco_dt):
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    with suppress_output(stderr=False):
        ev.run()
    return ev.stats


def test_update_anns_area_moves_the_size_bucket_metrics():
    # The multi-IoU-type case: an annotation's active `area` has to follow the
    # box for bbox and the mask for segm. Reading `coco.dataset`, editing the
    # copy, and evaluating changes nothing — the mutator is what makes it land.
    coco_gt, coco_dt = _size_bucket_fixture()
    before = _bbox_stats(coco_gt, coco_dt)
    assert before[3] == pytest.approx(1.0), "400 px² starts in the small bucket"
    assert before[4] == -1.0, "and nothing is medium yet"

    # In-place mutation of the copy stays a no-op, as documented.
    coco_gt.dataset["annotations"][0]["area"] = 5000.0
    unchanged = _bbox_stats(coco_gt, coco_dt)
    assert unchanged[3] == pytest.approx(1.0)
    assert unchanged[4] == -1.0

    coco_gt.update_anns([{"id": 1, "area": 5000.0}])
    after = _bbox_stats(coco_gt, coco_dt)
    assert after[3] == -1.0, "no small ground truth is left"
    assert after[4] == pytest.approx(1.0), "5000 px² is a medium annotation"


def test_an_evaluator_built_before_an_edit_keeps_its_snapshot():
    # The evaluator shares the dataset rather than copying it; an edit after
    # construction lands on a private copy, so the evaluator's numbers do not
    # move underneath it and the COCO object sees the edit.
    coco_gt, coco_dt = _size_bucket_fixture()
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    coco_gt.update_anns([{"id": 1, "area": 5000.0}])
    with suppress_output(stderr=False):
        ev.run()
    assert ev.stats[3] == pytest.approx(1.0), "the evaluator still sees a small GT"
    assert ev.coco_gt.dataset["annotations"][0]["area"] == 400.0
    assert coco_gt.dataset["annotations"][0]["area"] == 5000.0


def test_update_anns_merges_and_keeps_the_other_fields():
    gt = _make_minimal_gt("bbox", annotations=[_make_bbox_ann(1, iscrowd=1, note="keep me")])
    coco = COCO(gt)

    coco.update_anns([{"id": 1, "area": 7.0}])

    ann = coco.dataset["annotations"][0]
    assert ann["area"] == 7.0
    assert ann["bbox"] == [10.0, 10.0, 100.0, 100.0]
    assert ann["iscrowd"] == 1
    assert ann["note"] == "keep me", "custom keys survive a partial edit"


def test_update_anns_follows_a_moved_annotation():
    gt = _make_minimal_gt(
        "bbox",
        images=[{"id": 1, "width": 640, "height": 480}, {"id": 2, "width": 640, "height": 480}],
        annotations=[_make_bbox_ann(1), _make_bbox_ann(2)],
    )
    coco = COCO(gt)

    coco.update_anns([{"id": 2, "image_id": 2}])

    assert coco.get_ann_ids(img_ids=[1]) == [1]
    assert coco.get_ann_ids(img_ids=[2]) == [2], "the index followed the moved annotation"
    assert coco.get_img_ids(cat_ids=[1]) == [1, 2]


def test_update_anns_accepts_whole_dicts_from_dataset():
    coco = COCO(_make_minimal_gt("bbox", annotations=[_make_bbox_ann(1), _make_bbox_ann(2)]))

    anns = coco.dataset["annotations"]
    for ann in anns:
        ann["area"] = ann["bbox"][2] * ann["bbox"][3] / 2
    coco.update_anns(anns)

    assert [a["area"] for a in coco.dataset["annotations"]] == [5000.0, 5000.0]


def test_update_anns_rejects_unknown_annotation_ids():
    coco = COCO(_make_minimal_gt("bbox", annotations=[_make_bbox_ann(1)]))

    with pytest.raises(KeyError, match=r"\[99, 100\]"):
        coco.update_anns([{"id": 1, "area": 1.0}, {"id": 99}, {"id": 100}])

    # Nothing was written, the valid edit included.
    assert coco.dataset["annotations"][0]["area"] == 10000.0


def test_update_anns_sets_scalar_shaped_and_custom_keys_alike():
    coco = COCO(_make_minimal_gt("bbox", annotations=[_make_bbox_ann(1)]))

    coco.update_anns([{"id": 1, "iscrowd": True}])
    assert coco.dataset["annotations"][0]["iscrowd"] == 1
    coco.update_anns([{"id": 1, "iscrowd": 0}])
    assert coco.dataset["annotations"][0]["iscrowd"] == 0

    coco.update_anns([{"id": 1, "bbox": [1.0, 2.0, 3.0, 4.0]}])
    assert coco.dataset["annotations"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]

    coco.update_anns([{"id": 1, "provenance": "hand-drawn"}])
    assert coco.dataset["annotations"][0]["provenance"] == "hand-drawn"
    coco.update_anns([{"id": 1, "provenance": "traced"}])
    assert coco.dataset["annotations"][0]["provenance"] == "traced"


def test_update_anns_rejects_a_value_that_does_not_fit():
    coco = COCO(_make_minimal_gt("bbox", annotations=[_make_bbox_ann(1)]))

    with pytest.raises(TypeError):
        coco.update_anns([{"id": 1, "area": "big"}])
    with pytest.raises(TypeError):
        coco.update_anns([{"id": -1, "area": 5.0}])
    assert coco.dataset["annotations"][0]["area"] == 10000.0


def test_update_anns_input_shape():
    coco = COCO(_make_minimal_gt("bbox", annotations=[_make_bbox_ann(0)]))

    with pytest.raises(TypeError):
        coco.update_anns(["not a dict"])
    coco.update_anns([])

    # Annotation id 0 is a real id, told apart from an absent "id" key.
    coco.update_anns([{"id": 0, "area": 5.0}])
    assert coco.dataset["annotations"][0]["area"] == 5.0


def test_update_anns_treats_a_misspelled_field_as_a_custom_key():
    # As pycocotools does: `ann["Area"] = x` adds a key, and the data shows it.
    coco = COCO(_make_minimal_gt("bbox", annotations=[_make_bbox_ann(1)]))

    coco.update_anns([{"id": 1, "Area": 5000.0}])

    ann = coco.dataset["annotations"][0]
    assert ann["area"] == 10000.0
    assert ann["Area"] == 5000.0


def test_update_anns_reports_the_missing_id_first():
    # The dict is missing `image_id` too; the absent `id` is the more useful
    # diagnostic, so it must not be overtaken by the conversion.
    coco = COCO(_make_minimal_gt("bbox", annotations=[_make_bbox_ann(1)]))

    with pytest.raises(KeyError, match="id"):
        coco.update_anns([{"area": 1.0}])
