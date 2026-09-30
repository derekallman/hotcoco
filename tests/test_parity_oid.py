"""Open Images parity: hotcoco `oid_style=True` vs the TensorFlow reference.

Compares hotcoco against frozen output from the TF Object Detection API's
`OpenImagesDetectionEvaluator(group_of_weight=1.0)` — the Open Images Challenge
detection metric.

The fixture is checked in (`fixtures/oid_tf_expected.json`), so this needs no
network and no TensorFlow. Regenerate it with `just gen-oid-fixtures`.

Both mAP and per-class AP are compared. Per-class matters on its own: two
categories whose APs are swapped leave the mean identical, which is the same
argument `test_adversarial.py` makes for diffing per-detection decisions rather
than only metrics.

What this does and does not cover
---------------------------------
Covered: group-of absorption (one TP per group-of box, surplus detections ignored,
undetected group-of box is a miss), IoA as the containment measure, and VOC 2010
all-points AP.

Not covered: the non-exhaustive image-level-label rule, the Challenge's third
mechanism, which hotcoco does not implement — the oracle is generated with the
evaluator that omits it, so this comparison says nothing about it. Also not covered:
hierarchy expansion, which hotcoco applies before evaluation rather than inside it.

A mismatch is not automatically a hotcoco bug. It can mean the reference moved;
check `reference` in the fixture file before assuming.

    uv run pytest tests/test_parity_oid.py -v
"""

from __future__ import annotations

import json

import pytest
from gen_oid_fixtures import MIN_DISCRIMINATING
from helpers import FIXTURES_DIR, suppress_output
from hotcoco import COCO, COCOeval

FIXTURES = FIXTURES_DIR / "oid_tf_expected.json"
TOL = 1e-9

_DATA = json.loads(FIXTURES.read_text())
CASES = _DATA["cases"]


def hotcoco_aps(case) -> tuple[float, dict[str, float]]:
    """Return (mAP, {category_id: AP}) for one case."""
    gt = COCO(
        {
            "images": [{"id": 1, "width": 800, "height": 800, "file_name": "a.jpg"}],
            "annotations": case["annotations"],
            "categories": case["categories"],
        }
    )
    dt = gt.load_res(case["detections"])
    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    # Every OID run prints its `Provenance::Extension` deviation to fd 2. That is
    # correct — and is exactly the gap this test narrows — but one copy per case
    # would bury the result seventy times over.
    with suppress_output():
        ev.run()

    # `per_class` is keyed by category *name*; the fixture is keyed by id, since
    # that is what the TF result keys carry. Map through the case's own category
    # list rather than assuming the `c{i}` convention a second time.
    by_name = ev.results(per_class=True).get("per_class", {})
    per_class = {str(c["id"]): by_name.get(c["name"]) for c in case["categories"]}
    return float(ev.stats[0]), per_class


def per_class_mismatches(expected: dict, got: dict) -> list[str]:
    """Per-category disagreements, as human-readable strings.

    `None` on the reference side means TF returned NaN — a category with no ground
    truth, which it excludes from the mean. hotcoco reports its `-1.0` "not
    computed" sentinel for the same case, so the two agree by *both* declining to
    score it; anything else is a real disagreement about whether a category is
    evaluable.
    """
    out = []
    for cid, exp in expected.items():
        g = got.get(cid)
        if exp is None:
            if g is not None and g >= 0.0:
                out.append(f"c{cid}: reference has no AP, hotcoco reports {g:.6f}")
        elif g is None or g < 0.0:
            out.append(f"c{cid}: reference {exp:.6f}, hotcoco reports nothing")
        elif abs(g - exp) > TOL:
            out.append(f"c{cid}: ref={exp:.6f} got={g:.6f} diff={abs(g - exp):.3e}")
    return out


def test_fixture_set_discriminates():
    """A corpus that is all 0.0/1.0 agrees with a stub implementation.

    The floor is `gen_oid_fixtures.MIN_DISCRIMINATING`, imported rather than
    restated: asserting it here as well means a regenerated corpus cannot quietly
    weaken the consumer, and there is one number to change if the bar moves.
    """
    discriminating = sum(1 for c in CASES if 0.0 < c["expected"]["mAP"] < 1.0)
    assert discriminating >= len(CASES) * MIN_DISCRIMINATING, (
        f"{discriminating}/{len(CASES)} cases score strictly between 0 and 1; the set is mostly saturated"
    )


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_map_and_per_class_ap_match_tf(case):
    expected = case["expected"]
    got_map, got_per_class = hotcoco_aps(case)

    why = []
    diff = abs(got_map - expected["mAP"])
    if diff > TOL:
        why.append(f"mAP ref={expected['mAP']:.6f} got={got_map:.6f} diff={diff:.3e}")
    why += per_class_mismatches(expected["per_class"], got_per_class)
    assert not why, f"{case['name']} diverges from {_DATA['reference']['evaluator']}:\n  " + "\n  ".join(why)
