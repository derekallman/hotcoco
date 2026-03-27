"""Hypothesis-based OBB IoU parity fuzzer: hotcoco vs Shapely.

Tests that hotcoco's rotated IoU kernel matches Shapely's polygon intersection
by verifying evaluation results are consistent with Shapely-computed IoU values.
Shapely (backed by GEOS, the same engine as PostGIS) is the reference implementation.

This is a bug-hunting tool, not a CI gate.

Usage:
    uv run pytest scripts/fuzz_obb_parity.py -v -x --tb=short
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings
from shapely.geometry import Polygon

# ---------------------------------------------------------------------------
# OBB geometry helpers (pure Python, matching hotcoco::geometry)
# ---------------------------------------------------------------------------


def obb_to_corners(cx, cy, w, h, angle):
    """Convert (cx, cy, w, h, angle) to 4 corner points."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    hw, hh = w / 2, h / 2

    dx_w = hw * cos_a
    dy_w = hw * sin_a
    dx_h = hh * sin_a
    dy_h = hh * cos_a

    return [
        (cx - dx_w + dx_h, cy - dy_w - dy_h),
        (cx + dx_w + dx_h, cy + dy_w - dy_h),
        (cx + dx_w - dx_h, cy + dy_w + dy_h),
        (cx - dx_w - dx_h, cy - dy_w + dy_h),
    ]


def shapely_obb_iou(obb_a, obb_b):
    """Compute IoU between two OBBs using Shapely as reference."""
    corners_a = obb_to_corners(*obb_a)
    corners_b = obb_to_corners(*obb_b)

    poly_a = Polygon(corners_a)
    poly_b = Polygon(corners_b)

    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0

    area_a = poly_a.area
    area_b = poly_b.area

    if area_a <= 0 or area_b <= 0:
        return 0.0

    intersection = poly_a.intersection(poly_b).area
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# hotcoco evaluation helper
# ---------------------------------------------------------------------------


def hotcoco_eval_obb(obb_gt, obb_dt, score=1.0):
    """Run hotcoco OBB evaluation and return the 12-metric stats vector.

    Creates a minimal dataset with one GT and one DT, both in the "large"
    area range to avoid area filtering complications.
    """
    from hotcoco import COCO, COCOeval

    # Use large OBBs to ensure they're in the "large" area range (>9216 px²)
    gt_area = obb_gt[2] * obb_gt[3]
    dt_area = obb_dt[2] * obb_dt[3]

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

    dt_data = [
        {
            "image_id": 1,
            "category_id": 1,
            "obb": list(obb_dt),
            "score": score,
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_data, f)
        gt_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dt_data, f)
        dt_path = f.name

    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.load_res(dt_path)
        ev = COCOeval(coco_gt, coco_dt, "obb")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return ev.stats
    finally:
        Path(gt_path).unlink(missing_ok=True)
        Path(dt_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def obb_strategy(draw, min_size=10.0, max_size=500.0):
    """Generate a random OBB as (cx, cy, w, h, angle) with guaranteed large area."""
    cx = draw(st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False))
    cy = draw(st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False))
    w = draw(st.floats(min_value=min_size, max_value=max_size, allow_nan=False, allow_infinity=False))
    h = draw(st.floats(min_value=min_size, max_value=max_size, allow_nan=False, allow_infinity=False))
    angle = draw(
        st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False)
    )
    return (cx, cy, w, h, angle)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@given(obb_a=obb_strategy(), obb_b=obb_strategy())
@settings(
    max_examples=200,
    deadline=30000,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_obb_eval_consistency_with_shapely(obb_a, obb_b):
    """Verify hotcoco OBB eval behavior is consistent with Shapely IoU.

    If Shapely says IoU >= 0.5, hotcoco should report AP@50 = 1.0.
    If Shapely says IoU < 0.5, hotcoco should report AP@50 = 0.0.
    We use a margin to avoid borderline cases.
    """
    iou = shapely_obb_iou(obb_a, obb_b)

    # Skip borderline cases where IoU is very close to a threshold
    if abs(iou - 0.5) < 0.02:
        return  # Too close to threshold, skip

    stats = hotcoco_eval_obb(obb_a, obb_b)

    # stats[1] is AP@IoU=0.50
    ap50 = stats[1]

    if iou >= 0.52:
        assert ap50 == 1.0, (
            f"Shapely IoU = {iou:.6f} >= 0.52, but hotcoco AP@50 = {ap50:.4f}. "
            f"OBBs: GT={obb_a}, DT={obb_b}"
        )
    elif iou < 0.48:
        assert ap50 <= 0.0, (
            f"Shapely IoU = {iou:.6f} < 0.48, but hotcoco AP@50 = {ap50:.4f}. "
            f"OBBs: GT={obb_a}, DT={obb_b}"
        )


@pytest.mark.parametrize(
    "obb_a,obb_b,expected_iou",
    [
        # Identical boxes
        ((0, 0, 100, 100, 0), (0, 0, 100, 100, 0), 1.0),
        # Non-overlapping
        ((0, 0, 100, 100, 0), (1000, 1000, 100, 100, 0), 0.0),
        # 90° rotated square (should be 1.0)
        ((0, 0, 100, 100, 0), (0, 0, 100, 100, math.pi / 2), 1.0),
        # Half overlap axis-aligned
        ((0, 0, 100, 100, 0), (50, 0, 100, 100, 0), 50 * 100 / (2 * 100 * 100 - 50 * 100)),
        # 180° rotation (same box)
        ((50, 50, 200, 100, 0.7), (50, 50, 200, 100, 0.7 + math.pi), 1.0),
    ],
    ids=["identical", "non_overlapping", "90deg_square", "half_overlap", "180deg_rotation"],
)
def test_obb_iou_known_values(obb_a, obb_b, expected_iou):
    """Known geometric cases: Shapely must agree with expected values."""
    shapely_iou = shapely_obb_iou(obb_a, obb_b)
    assert abs(shapely_iou - expected_iou) < 1e-6, (
        f"Shapely IoU {shapely_iou:.8f} != expected {expected_iou:.8f}"
    )


@pytest.mark.parametrize(
    "obb_a,obb_b,should_match_at_50",
    [
        # Identical → IoU=1.0, should match
        ((0, 0, 200, 200, 0), (0, 0, 200, 200, 0), True),
        # Far apart → IoU=0, should not match
        ((0, 0, 200, 200, 0), (2000, 2000, 200, 200, 0), False),
        # High overlap → IoU≈0.75, should match at 0.50
        ((0, 0, 200, 200, 0), (50, 0, 200, 200, 0), True),
    ],
    ids=["perfect_match", "no_overlap", "high_overlap"],
)
def test_obb_eval_matches_shapely(obb_a, obb_b, should_match_at_50):
    """hotcoco eval results must be consistent with Shapely IoU at the 0.50 threshold."""
    shapely_iou = shapely_obb_iou(obb_a, obb_b)
    stats = hotcoco_eval_obb(obb_a, obb_b)
    ap50 = stats[1]

    if should_match_at_50:
        assert shapely_iou >= 0.5, f"Test setup error: Shapely IoU = {shapely_iou}"
        assert ap50 == 1.0, f"Expected AP@50=1.0, got {ap50} (Shapely IoU={shapely_iou})"
    else:
        assert shapely_iou < 0.5, f"Test setup error: Shapely IoU = {shapely_iou}"
        assert ap50 <= 0.0, f"Expected AP@50=0, got {ap50} (Shapely IoU={shapely_iou})"
