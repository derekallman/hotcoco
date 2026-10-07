"""Oriented-box evaluation at the 0.50 threshold, on boxes whose IoU is worked
out by hand.

`fuzz_obb.py` checks the same against Shapely over random boxes, but pytest
does not collect the fuzzers, and an assertion there went wrong unseen for
weeks. These cases need neither Shapely nor hypothesis, so `just test` and CI
run them.
"""

import math

import pytest
from helpers import hotcoco_eval_obb

GT = (0, 0, 200, 200, 0)


@pytest.mark.parametrize(
    ("dt", "ap50"),
    [
        # The ground truth itself: IoU 1.
        (GT, 1.0),
        # The same square turned a quarter turn: the same polygon, IoU 1.
        ((0, 0, 200, 200, math.pi / 2), 1.0),
        # Shifted a quarter of its width: 150 x 200 overlap, IoU 30,000 / 50,000 = 0.6.
        ((50, 0, 200, 200, 0), 1.0),
        # Shifted 120: 80 x 200 overlap, IoU 16,000 / 64,000 = 0.25.
        ((120, 0, 200, 200, 0), 0.0),
        # Far apart: IoU 0.
        ((2000, 2000, 200, 200, 0), 0.0),
    ],
    ids=["identical", "quarter_turn", "iou_0.6", "iou_0.25", "disjoint"],
)
def test_ap50_follows_the_iou(dt, ap50):
    # A lone true positive has precision 1 - 2**-52, not 1.0: pycocotools adds
    # `np.spacing(1)` to the denominator, and the arrays match it.
    assert hotcoco_eval_obb(GT, dt)[1] == pytest.approx(ap50, abs=1e-9)
