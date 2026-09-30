"""gen_obb_fixtures.py — freeze Shapely's rotated-box IoU as a Rust test oracle.

Oriented boxes have no reference *evaluator*, which is why `report()` marks them
`Provenance::Extension`. But the geometry underneath is not novel: rotated-rectangle
intersection is a solved problem, and Shapely/GEOS is an independent implementation
of it. That makes it a real oracle for `primitives::sim::obb_iou`, even though
nothing can validate the AP built on top.

The existing check did not use it as one. `tests/fuzz_obb.py` only asserted which
side of 0.5 hotcoco's AP@50 landed on, skipping a +/-0.02 dead band — so a
systematic IoU error of 0.02 passed 200 examples — and its `test_obb_iou_known_values`
compared Shapely against hand-derived constants *without calling hotcoco at all*,
validating the oracle rather than the subject. It also carried its own copy of the
corner math and the IoU below; that copy is gone, and the fuzzer now imports
`corners`/`shapely_iou` from here, so the frozen fixture and the fuzzer cannot end
up checking hotcoco against two drifting definitions of the same number.

Freezing the values instead means the Rust test compares numbers, at 1e-9, with no
Python in the loop and no reimplementation to agree with it.

Cases are weighted toward the configurations where a polygon-clipping bug actually
hides: near-tangent boxes, tiny rotation offsets, one box contained in another,
degenerate zero-area boxes, and identical boxes at angles that differ by a multiple
of pi/2 (where the true IoU is exactly 1.0).

    uv run python scripts/gen_obb_fixtures.py
"""

import json
import math
import random
from pathlib import Path

from shapely.geometry import Polygon

OUT = Path(__file__).parent.parent / "crates/hotcoco/src/primitives/testdata/obb_iou_shapely.json"

rng = random.Random(0x0BB100)


def corners(obb):
    """(cx, cy, w, h, angle) -> four corners, counter-clockwise.

    Deliberately written from the OBB definition rather than ported from
    `hotcoco::geometry`: an oracle that shares its subject's corner math checks
    nothing about that math.
    """
    cx, cy, w, h, a = obb
    cos_a, sin_a = math.cos(a), math.sin(a)
    hw, hh = w / 2.0, h / 2.0
    return [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]


def shapely_iou(a, b):
    pa, pb = Polygon(corners(a)), Polygon(corners(b))
    if not pa.is_valid or not pb.is_valid:
        return None
    union = pa.union(pb).area
    if union <= 0.0:
        return 0.0
    return pa.intersection(pb).area / union


def r(lo, hi, nd=4):
    return round(rng.uniform(lo, hi), nd)


def gen_case(kind):
    if kind == "identical":
        a = (r(-50, 50), r(-50, 50), r(1, 100), r(1, 100), r(-math.pi, math.pi))
        return a, a
    if kind == "quarter_turn":
        # A square rotated by pi/2 is the same square: true IoU is exactly 1.0.
        cx, cy, s = r(-50, 50), r(-50, 50), r(1, 100)
        ang = r(-math.pi, math.pi)
        k = rng.choice([1, 2, 3])
        return (cx, cy, s, s, ang), (cx, cy, s, s, ang + k * math.pi / 2)
    if kind == "tiny_rotation":
        # Nearly-aligned boxes: where clipping error accumulates fastest.
        a = (r(-20, 20), r(-20, 20), r(5, 60), r(5, 60), r(-math.pi, math.pi))
        return a, (a[0] + r(-1, 1), a[1] + r(-1, 1), a[2], a[3], a[4] + r(-0.02, 0.02, 6))
    if kind == "contained":
        cx, cy = r(-30, 30), r(-30, 30)
        big = (cx, cy, r(50, 100), r(50, 100), r(-math.pi, math.pi))
        return big, (cx, cy, big[2] / rng.uniform(2, 6), big[3] / rng.uniform(2, 6), big[4])
    if kind == "tangent":
        # Edge-to-edge contact: intersection area is exactly zero, and a clipper
        # that mishandles collinear edges returns a sliver instead.
        w = r(10, 40)
        cx, cy = r(-20, 20), r(-20, 20)
        return (cx, cy, w, w, 0.0), (cx + w, cy, w, w, 0.0)
    if kind == "degenerate":
        a = (r(-20, 20), r(-20, 20), rng.choice([0.0, r(0.001, 0.01, 6)]), r(1, 40), r(-math.pi, math.pi))
        return a, (r(-20, 20), r(-20, 20), r(1, 40), r(1, 40), r(-math.pi, math.pi))
    if kind == "disjoint":
        return (
            (r(-500, -400), r(-500, -400), r(1, 50), r(1, 50), r(-math.pi, math.pi)),
            (r(400, 500), r(400, 500), r(1, 50), r(1, 50), r(-math.pi, math.pi)),
        )
    # generic overlap
    return (
        (r(-40, 40), r(-40, 40), r(1, 80), r(1, 80), r(-math.pi, math.pi)),
        (r(-40, 40), r(-40, 40), r(1, 80), r(1, 80), r(-math.pi, math.pi)),
    )


KINDS = [
    ("generic", 260),
    ("tiny_rotation", 120),
    ("contained", 60),
    ("identical", 40),
    ("quarter_turn", 40),
    ("tangent", 30),
    ("degenerate", 30),
    ("disjoint", 20),
]


def main():
    cases = []
    for kind, n in KINDS:
        for _ in range(n):
            a, b = gen_case(kind)
            iou = shapely_iou(a, b)
            if iou is None:
                continue
            cases.append({"kind": kind, "a": list(a), "b": list(b), "iou": iou})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1) + "\n")
    nonzero = sum(1 for c in cases if c["iou"] > 0)
    print(f"wrote {len(cases)} cases to {OUT} ({nonzero} with non-zero IoU)")
    by = {}
    for c in cases:
        by[c["kind"]] = by.get(c["kind"], 0) + 1
    for k, n in sorted(by.items()):
        print(f"  {k:<14} {n}")


if __name__ == "__main__":
    main()
