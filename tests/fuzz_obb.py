"""Hypothesis-based OBB IoU parity fuzzer: hotcoco vs Shapely.

Tests that hotcoco's rotated IoU kernel matches Shapely's polygon intersection
by verifying evaluation results are consistent with Shapely-computed IoU values.
Shapely (backed by GEOS, the same engine as PostGIS) is the reference implementation.

This is a bug-hunting tool, not a CI gate.

Usage:
    uv run pytest tests/fuzz_obb.py -v -x --tb=short
"""

import math

import hypothesis.strategies as st
import pytest
from gen_obb_fixtures import shapely_iou as shapely_obb_iou
from helpers import written_json
from hypothesis import HealthCheck, given, settings

# The Shapely oracle — corner math and IoU — is owned by `gen_obb_fixtures.py`,
# which freezes its values into the Rust test fixture. This file had a second
# copy of both functions. Two oracles cannot disagree usefully: whichever one
# drifts, the Rust fixture and this fuzzer would then be checking hotcoco against
# different definitions of the same number.
#
# `shapely_iou` returns None for a degenerate polygon Shapely calls invalid;
# `obb_strategy` cannot generate one (MIN_SIZE below), but callers below skip on
# None rather than coercing it to 0.0 and asserting on a fabricated value.

# Side-length bounds for every generated box. Module constants because the
# docstring quotes them and the module comment above reasons about them — as
# strategy defaults, nothing kept the three in step, and no caller ever
# overrode them.
MIN_SIZE = 10.0
MAX_SIZE = 500.0

# ---------------------------------------------------------------------------
# hotcoco evaluation helper
# ---------------------------------------------------------------------------


def hotcoco_eval_obb(obb_gt, obb_dt):
    """Run hotcoco OBB evaluation and return the 12-metric stats vector.

    A minimal dataset: one GT and one DT on a 4096x4096 image. The detection
    scores 1.0 — the only score a single-detection case can meaningfully carry,
    and no caller ever passed another.
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
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def obb_strategy(draw):
    """Generate a random OBB as (cx, cy, w, h, angle).

    Centers land within +/-500 of the origin and both side lengths in
    [MIN_SIZE, MAX_SIZE], so the two boxes overlap often enough for the 0.5
    threshold to be the interesting question.
    """
    cx = draw(st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False))
    cy = draw(st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False))
    w = draw(st.floats(min_value=MIN_SIZE, max_value=MAX_SIZE, allow_nan=False, allow_infinity=False))
    h = draw(st.floats(min_value=MIN_SIZE, max_value=MAX_SIZE, allow_nan=False, allow_infinity=False))
    angle = draw(st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False))
    return (cx, cy, w, h, angle)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@given(obb_a=obb_strategy(), obb_b=obb_strategy())
@settings(max_examples=200, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
def test_obb_eval_consistency_with_shapely(obb_a, obb_b):
    """Verify hotcoco's OBB *evaluation* agrees with Shapely about the threshold.

    This checks the eval pipeline, not the IoU kernel. The kernel is compared to
    Shapely by value, at 1e-9, in `primitives::sim::tests::obb_iou_matches_shapely`
    against a frozen fixture — which is where a systematic offset would show up.
    This test exists to catch the pipeline *around* it: area ranges, matching,
    accumulation.

    The dead band around 0.5 is 1e-6: measured kernel agreement with Shapely is
    1.2e-14, so the band only needs to absorb the boundary itself, and a wider
    one would let a systematic IoU error pass.
    """
    iou = shapely_obb_iou(obb_a, obb_b)

    # Only genuinely-on-the-boundary cases are ambiguous now. `None` means
    # Shapely refused the polygon, so there is no reference value to assert on.
    if iou is None or abs(iou - 0.5) < 1e-6:
        return

    stats = hotcoco_eval_obb(obb_a, obb_b)

    # stats[1] is AP@IoU=0.50
    ap50 = stats[1]

    if iou > 0.5 + 1e-6:
        assert ap50 == 1.0, (
            f"Shapely IoU = {iou:.9f} > 0.5, but hotcoco AP@50 = {ap50:.4f}. OBBs: GT={obb_a}, DT={obb_b}"
        )
    elif iou < 0.5 - 1e-6:
        assert ap50 <= 0.0, (
            f"Shapely IoU = {iou:.9f} < 0.5, but hotcoco AP@50 = {ap50:.4f}. OBBs: GT={obb_a}, DT={obb_b}"
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
    ref_iou = shapely_obb_iou(obb_a, obb_b)
    assert ref_iou is not None, f"Test setup error: Shapely rejected {obb_a} / {obb_b}"
    stats = hotcoco_eval_obb(obb_a, obb_b)
    ap50 = stats[1]

    if should_match_at_50:
        assert ref_iou >= 0.5, f"Test setup error: Shapely IoU = {ref_iou}"
        assert ap50 == 1.0, f"Expected AP@50=1.0, got {ap50} (Shapely IoU={ref_iou})"
    else:
        assert ref_iou < 0.5, f"Test setup error: Shapely IoU = {ref_iou}"
        assert ap50 <= 0.0, f"Expected AP@50=0, got {ap50} (Shapely IoU={ref_iou})"
