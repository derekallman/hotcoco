"""gen_oid_fixtures.py — freeze Open Images AP from the TensorFlow reference.

Open Images is the one detection family hotcoco ships with no parity check. Its
reference — the TF Object Detection API, which the official protocol page points
at — is not pip-installable: `object_detection` lives in the `research/` tree and
normally needs protoc-compiled protos. That made a comparison look impossible, and
a group-of defect survived the whole life of the feature partly because of it.

It is not impossible. The evaluation path is pure numpy: `per_image_evaluation.py`
imports nothing but numpy and sibling `np_box_*` modules, and
`object_detection_evaluation.py` touches TensorFlow only inside `add_eval_dict` /
`get_estimator_eval_metric_ops`, neither of which evaluation calls. So this script
downloads those modules, stubs the two imports that would drag in TF and protobuf,
runs the reference, and freezes its output.

Same pattern as `primitives/testdata/lsap_scipy.json` (scipy) and
`obb_iou_shapely.json` (Shapely): the *oracle* runs here, the *fixtures* are
checked in, and `tests/test_parity_oid.py` needs no network and no TensorFlow.

    uv run python scripts/gen_oid_fixtures.py

Which evaluator, and why it is not the one named "Challenge"
------------------------------------------------------------
`OpenImagesDetectionChallengeEvaluator` is `OpenImagesDetectionEvaluator` with
`group_of_weight=1.0` plus one extra mechanism: it drops detections whose class is
not verified on that image (the non-exhaustive-label rule). hotcoco implements the
first and not the second, and the second needs image-level label data that COCO
JSON cannot carry.

So we drive `OpenImagesDetectionEvaluator(group_of_weight=1.0)`, which is the
Challenge detection metric with that filter absent. That is exactly the surface
hotcoco claims, and mixing in an unimplemented mechanism would produce a red
comparison that says nothing about the code under test.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import random
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

import numpy as np
from helpers import FIXTURES_DIR

OUT = FIXTURES_DIR / "oid_tf_expected.json"

# Minimum share of cases that must score strictly between 0 and 1. A corpus of
# saturated cases agrees with any implementation that always returns 0 or 1.
# `tests/test_parity_oid.py` imports this and asserts the same floor on the checked-in
# corpus, so regenerating with a degenerate scenario set fails loudly at both
# ends instead of quietly weakening the consumer.
MIN_DISCRIMINATING = 0.3

# Pinned, not `master`. Every other generator in this repo records an exact
# reference version (`md.version("pycocotools")`, installed scipy/Shapely). An
# oracle tracking a moving branch is the same defect as a pinned semver baseline
# running zero checks: green, and you cannot say what it checked.
REF_COMMIT = "4d7bdd8c170ee90850f2f9ccef0f6d19b817de35"
RAW = f"https://raw.githubusercontent.com/tensorflow/models/{REF_COMMIT}/research/object_detection"
MODULES = (
    "core/standard_fields.py",
    "utils/object_detection_evaluation.py",
    "utils/per_image_evaluation.py",
    "utils/metrics.py",
    "utils/np_box_ops.py",
    "utils/np_box_list.py",
    "utils/np_box_list_ops.py",
    "utils/np_box_mask_list.py",
    "utils/np_box_mask_list_ops.py",
    "utils/np_mask_ops.py",
)


def _install_reference(root: Path):
    """Materialize a minimal `object_detection` package and import the evaluator.

    Two imports are stubbed rather than downloaded. `tensorflow.compat.v1` is
    imported at module scope but only *used* by the estimator helpers, so a bare
    module object satisfies the import and nothing calls into it.
    `label_map_util` pulls in compiled protobufs and is only used by
    label-map-based constructors we do not touch.
    """
    pkg = root / "object_detection"
    for sub in ("", "core", "utils"):
        d = pkg / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("")

    for rel in MODULES:
        dest = pkg / rel
        with urllib.request.urlopen(f"{RAW}/{rel}", timeout=60) as r:  # noqa: S310
            dest.write_bytes(r.read())

    tf = types.ModuleType("tensorflow")
    compat = types.ModuleType("tensorflow.compat")
    v1 = types.ModuleType("tensorflow.compat.v1")
    compat.v1 = v1
    tf.compat = compat
    sys.modules.update({"tensorflow": tf, "tensorflow.compat": compat, "tensorflow.compat.v1": v1})

    # `label_map_util` pulls in compiled protobufs, but the evaluator only calls
    # `create_category_index`, which is a one-line dict build. Reimplementing it
    # is cheaper and more auditable than vendoring the proto toolchain.
    lmu = types.ModuleType("object_detection.utils.label_map_util")
    lmu.create_category_index = lambda categories: {c["id"]: c for c in categories}
    sys.modules["object_detection.utils.label_map_util"] = lmu

    sys.path.insert(0, str(root))
    from object_detection.utils import object_detection_evaluation as ode  # noqa: PLC0415

    return ode


def _xywh_to_yxyx(b):
    """COCO `[x, y, w, h]` -> the reference's `[ymin, xmin, ymax, xmax]`."""
    x, y, w, h = b
    return [y, x, y + h, x + w]


# --- scenarios -------------------------------------------------------------
# Curated cases first, each aimed at one rule of the protocol, then randomized
# ones so the comparison is not limited to situations we thought to write down.


def _case(name, anns, dets, n_cat=1):
    # `categories` is what the fixture carries; `n_cat` is only a local shorthand
    # for building it. The `c{i}` naming is part of the oracle's contract
    # (`reference_ap` matches TF result keys with `endswith(f"/c{i}")`), so the
    # fixture should carry the list rather than have two files independently agree
    # to spell it the same way — and carrying the count *as well* is a second
    # source of truth for the same fact.
    return {
        "name": name,
        "categories": [{"id": i, "name": f"c{i}"} for i in range(1, n_cat + 1)],
        "annotations": anns,
        "detections": dets,
    }


def _a(i, box, cat=1, group_of=False):
    w, h = box[2], box[3]
    d = {"id": i, "image_id": 1, "category_id": cat, "bbox": list(box), "area": w * h, "iscrowd": 0}
    if group_of:
        d["is_group_of"] = True
    return d


def _d(i, box, score, cat=1):
    w, h = box[2], box[3]
    return {"id": i, "image_id": 1, "category_id": cat, "bbox": list(box), "area": w * h, "iscrowd": 0, "score": score}


def curated_cases():
    G = (0.0, 0.0, 200.0, 200.0)  # a group-of box
    OBJ = (300.0, 300.0, 100.0, 100.0)  # an ordinary object
    return [
        _case("no_group_of", [_a(1, OBJ)], [_d(1, OBJ, 0.9)]),
        _case("group_of_one_detection_inside", [_a(1, G, group_of=True)], [_d(1, (10, 10, 80, 80), 0.9)]),
        _case(
            "group_of_many_detections_inside",
            [_a(1, OBJ), _a(2, G, group_of=True)],
            [_d(1, OBJ, 0.9), _d(2, (10, 10, 80, 80), 0.8), _d(3, (50, 50, 60, 60), 0.7)],
        ),
        _case("group_of_undetected", [_a(1, OBJ), _a(2, G, group_of=True)], [_d(1, OBJ, 0.9)]),
        # The IoA case: small box inside, scoring above the ordinary TP so any
        # regression lands its error before full recall.
        _case(
            "small_detection_inside_group_of",
            [_a(1, OBJ), _a(2, G, group_of=True)],
            [_d(1, (10, 10, 80, 80), 0.9), _d(2, OBJ, 0.8)],
        ),
        # Straddles the boundary: half the detection lies outside, IoA 0.5 exactly.
        _case("detection_half_outside_group_of", [_a(1, G, group_of=True)], [_d(1, (150.0, 0.0, 100.0, 100.0), 0.9)]),
        _case("detection_mostly_outside_group_of", [_a(1, G, group_of=True)], [_d(1, (180.0, 0.0, 100.0, 100.0), 0.9)]),
        _case("group_of_only_no_detections", [_a(1, G, group_of=True)], []),
        _case(
            "two_group_of_boxes",
            [_a(1, G, group_of=True), _a(2, (400.0, 400.0, 200.0, 200.0), group_of=True)],
            [_d(1, (10, 10, 50, 50), 0.9), _d(2, (410, 410, 50, 50), 0.8)],
        ),
        _case(
            "two_categories",
            [_a(1, OBJ, cat=1), _a(2, G, cat=2, group_of=True)],
            [_d(1, OBJ, 0.9, cat=1), _d(2, (10, 10, 80, 80), 0.8, cat=2)],
            n_cat=2,
        ),
    ]


def random_cases(n, seed=20260729):
    """Randomized cases whose detections are *derived from* the ground truth.

    Independently random boxes almost never overlap: the first version of this
    generator produced 56 vacuous cases out of 60, every one scoring mAP 0.0 and
    therefore agreeing with any implementation that returns zero. Fixtures like
    that inflate the case count and check nothing.

    So each detection is a jittered copy of some ground-truth box, with the jitter
    straddling the decision boundary — near-perfect copies, boxes shrunk to sit
    wholly inside (IoA 1.0 but low IoU, the case group-of handling turns on), boxes
    pushed mostly outside, wrong-class copies, and pure background. `main` asserts
    the resulting spread.
    """
    rng = random.Random(seed)
    out = []
    for c in range(n):
        n_cat = rng.choice([1, 2, 3])
        anns, dets = [], []
        aid = did = 1
        for _ in range(rng.randint(1, 5)):
            w, h = rng.uniform(40, 200), rng.uniform(40, 200)
            anns.append(
                _a(
                    aid,
                    (rng.uniform(0, 400), rng.uniform(0, 400), w, h),
                    cat=rng.randint(1, n_cat),
                    group_of=rng.random() < 0.4,
                )
            )
            aid += 1

        for a in anns:
            if rng.random() < 0.2:
                continue  # leave this one undetected
            x, y, w, h = a["bbox"]
            kind = rng.choice(["tight", "jitter", "inside", "mostly_out", "wrong_class"])
            box = (x, y, w, h)  # "tight", and the base for "wrong_class"
            if kind == "jitter":
                f = rng.uniform(0.7, 1.3)
                box = (x + rng.uniform(-0.3, 0.3) * w, y + rng.uniform(-0.3, 0.3) * h, w * f, h * f)
            elif kind == "inside":
                # Wholly contained: IoA 1.0, IoU as low as 0.09. Absorbed under the
                # protocol, a false positive under plain IoU.
                frac = rng.uniform(0.3, 0.7)
                box = (x + w * (1 - frac) / 2, y + h * (1 - frac) / 2, w * frac, h * frac)
            elif kind == "mostly_out":
                box = (x + w * rng.uniform(0.6, 0.9), y, w, h)
            # `wrong_class` keeps the tight box and moves the label instead. With
            # one category it degenerates to `tight`, which is why multi-category
            # cases carry the weight of that arm.
            cat = a["category_id"]
            if kind == "wrong_class" and n_cat > 1:
                cat = 1 + (cat % n_cat)
            dets.append(_d(did, box, round(rng.uniform(0.05, 0.99), 4), cat=cat))
            did += 1

        for _ in range(rng.randint(0, 3)):  # background detections
            w, h = rng.uniform(20, 120), rng.uniform(20, 120)
            dets.append(
                _d(
                    did,
                    (rng.uniform(0, 500), rng.uniform(0, 500), w, h),
                    round(rng.uniform(0.05, 0.99), 4),
                    cat=rng.randint(1, n_cat),
                )
            )
            did += 1

        out.append(_case(f"random_{c:03d}", anns, dets, n_cat=n_cat))
    return out


def reference_ap(ode, case):
    """Run the TF evaluator on one case and return per-class AP plus mAP."""
    ev = ode.OpenImagesDetectionEvaluator(case["categories"], group_of_weight=1.0)

    fields = ode.standard_fields.InputDataFields
    dfields = ode.standard_fields.DetectionResultFields

    anns, dets = case["annotations"], case["detections"]
    ev.add_single_ground_truth_image_info(
        1,
        {
            fields.groundtruth_boxes: np.array([_xywh_to_yxyx(a["bbox"]) for a in anns], dtype=float).reshape(-1, 4),
            fields.groundtruth_classes: np.array([a["category_id"] for a in anns], dtype=int),
            fields.groundtruth_group_of: np.array([bool(a.get("is_group_of", False)) for a in anns], dtype=bool),
        },
    )
    ev.add_single_detected_image_info(
        1,
        {
            dfields.detection_boxes: np.array([_xywh_to_yxyx(d["bbox"]) for d in dets], dtype=float).reshape(-1, 4),
            dfields.detection_scores: np.array([d["score"] for d in dets], dtype=float),
            dfields.detection_classes: np.array([d["category_id"] for d in dets], dtype=int),
        },
    )

    result = ev.evaluate()
    per_class = {}
    for cat in case["categories"]:
        for key, val in result.items():
            if key.endswith(f"/{cat['name']}"):
                per_class[str(cat["id"])] = None if np.isnan(val) else float(val)
    mean_key = next(k for k in result if "mAP" in k)
    return per_class, float(result[mean_key])


def main() -> int:
    cases = curated_cases() + random_cases(60)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"downloading {len(MODULES)} reference modules from tensorflow/models ...")
        try:
            ode = _install_reference(Path(tmp))
        except Exception as exc:  # noqa: BLE001
            print(f"could not set up the reference: {exc}")
            print("This script needs network access. Fixtures are checked in; you only")
            print("need to run it when regenerating them.")
            return 1

        out_cases = []
        for case in cases:
            try:
                per_class, mean_ap = reference_ap(ode, case)
            except Exception as exc:  # noqa: BLE001
                print(f"  {case['name']:38s} SKIPPED ({type(exc).__name__}: {exc})")
                continue
            out_cases.append({**case, "expected": {"per_class": per_class, "mAP": mean_ap}})
            print(f"  {case['name']:38s} mAP={mean_ap:.6f}")

    if len(out_cases) < len(cases):
        print(f"\n{len(cases) - len(out_cases)} case(s) failed to evaluate.")
        print("Refusing to write a partial fixture set — it would silently stop checking them.")
        return 1

    # A case scoring exactly 0.0 or 1.0 agrees with any implementation that always
    # returns 0 or always returns 1, so it discriminates nothing. Saturated cases
    # are still worth keeping (0.0 and 1.0 are real answers and the curated set
    # deliberately pins some), but a fixture file that is mostly saturated is a
    # corpus that looks large and checks little -- the first draft of this
    # generator produced exactly that, 56 zeros out of 60.
    interesting = [c for c in out_cases if 0.0 < c["expected"]["mAP"] < 1.0]
    frac = len(interesting) / len(out_cases)
    print(f"\n{len(interesting)}/{len(out_cases)} cases score strictly between 0 and 1 ({frac:.0%})")
    if frac < MIN_DISCRIMINATING:
        print("Refusing to write: too few discriminating cases. Widen the generator.")
        return 1

    payload = {
        "_comment": (
            "Open Images detection AP produced by the TensorFlow Object Detection API "
            "(OpenImagesDetectionEvaluator, group_of_weight=1.0), NOT by hotcoco. This is "
            "the Challenge detection metric without the non-exhaustive image-level-label "
            "filter, which hotcoco does not implement. Consumed by tests/test_parity_oid.py. "
            "Regenerate with scripts/gen_oid_fixtures.py (needs network access)."
        ),
        "reference": {
            # The commit, not the branch. This string is what a reader consults
            # when a comparison goes red, so naming `master` while the download
            # pins a SHA sends them to a tree the numbers never came from.
            "source": f"github.com/tensorflow/models @ {REF_COMMIT}, research/object_detection",
            "evaluator": "OpenImagesDetectionEvaluator(group_of_weight=1.0)",
            "numpy": md.version("numpy"),
        },
        "cases": out_cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT} ({len(out_cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
