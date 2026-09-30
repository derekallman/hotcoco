"""LVIS parity: lvis-api LVISEval vs hotcoco LVISeval.

Runs both implementations on synthetic LVIS datasets covering:
  - neg_category_ids  (DT on confirmed-absent category → FP)
  - not_exhaustive_category_ids  (unmatched DT → ignored)
  - frequency field  (APr / APc / APf grouping)
  - Standard COCO-style matching (TP, FP)
  - Mixed scenarios: some images have GT, some don't

Compares all 13 LVIS metrics between the two implementations.
Tolerance: 1e-4 (same as the COCO parity tests).

    uv run pytest tests/test_parity_lvis.py -v
"""

import numpy as np
import pytest
from helpers import compare_metrics, written_json

# Compatibility shim: lvis-api 0.5.3 uses np.float, removed in numpy 1.24. Only
# patch what lvis actually needs; do NOT patch np.bool (breaks numpy.ma).
np.float = float  # type: ignore[attr-defined]

from hotcoco import COCO, LVISeval  # noqa: E402
from lvis import LVIS, LVISEval  # noqa: E402
from lvis import LVISResults as LVISResultsRef  # noqa: E402

TOL = 1e-4

# Every scenario below builds bbox data, so this is the only iou_type these
# comparisons ever exercise. LVIS segm parity is *not* covered here.
IOU_TYPE = "bbox"

# The 13 LVIS metric names (12 COCO-style + AR@300, APr/APc/APf), from the
# evaluator rather than restated, so a new metric cannot silently go uncompared.
METRIC_NAMES = LVISeval(COCO(), COCO(), IOU_TYPE).metric_keys()


# ── helpers ──────────────────────────────────────────────────────────────────


def bbox(x, y, w, h):
    return [float(x), float(y), float(w), float(h)]


def make_lvis_gt(images, annotations, categories):
    return {
        "info": {"description": "synthetic LVIS parity test"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": [],
    }


def run_lvis_ref(gt_path, dt_list):
    """Run lvis-api LVISEval and return results dict."""
    lvis_gt = LVIS(gt_path)
    lvis_dt = LVISResultsRef(lvis_gt, dt_list)
    ev = LVISEval(lvis_gt, lvis_dt, IOU_TYPE)
    ev.run()
    return ev.results


def run_hotcoco(gt_path, dt_list):
    """Run hotcoco LVISeval and return get_results() dict."""
    gt = COCO(gt_path)
    dt = gt.load_res(dt_list)
    ev = LVISeval(gt, dt, IOU_TYPE)
    ev.run()
    return ev.get_results()


# ── dataset builders ─────────────────────────────────────────────────────────
# Each returns (gt_dataset, detections).


def _build_basic():
    """
    4 images, 3 categories:
      cat 1 freq="r", cat 2 freq="c", cat 3 freq="f"

    img 1: GT for cat 1 (area 400) and cat 2 (area 2500)
           neg_category_ids=[3]  → DT on cat 3 is FP
    img 2: GT for cat 1 (area 400)
           not_exhaustive_category_ids=[2]  → unmatched DT on cat 2 is ignored
    img 3: GT for cat 3 (area 9000)  (large area)
           neg_category_ids=[], not_exhaustive=[]
    img 4: no GT; neg_category_ids=[1]  → DT on cat 1 is FP
           not_exhaustive_category_ids=[2]  → DT on cat 2 is ignored
    """
    images = [
        {
            "id": 1,
            "file_name": "img1.jpg",
            "width": 100,
            "height": 100,
            "neg_category_ids": [3],
            "not_exhaustive_category_ids": [],
        },
        {
            "id": 2,
            "file_name": "img2.jpg",
            "width": 100,
            "height": 100,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [2],
        },
        {
            "id": 3,
            "file_name": "img3.jpg",
            "width": 200,
            "height": 200,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        },
        {
            "id": 4,
            "file_name": "img4.jpg",
            "width": 100,
            "height": 100,
            "neg_category_ids": [1],
            "not_exhaustive_category_ids": [2],
        },
    ]
    categories = [
        {"id": 1, "name": "cat_r", "frequency": "r", "supercategory": "animal"},
        {"id": 2, "name": "cat_c", "frequency": "c", "supercategory": "animal"},
        {"id": 3, "name": "cat_f", "frequency": "f", "supercategory": "animal"},
    ]
    # GT annotations
    gt_anns = [
        # img 1
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": bbox(5, 5, 20, 20),
            "area": 400.0,
            "segmentation": [],
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 2,
            "bbox": bbox(30, 30, 50, 50),
            "area": 2500.0,
            "segmentation": [],
            "iscrowd": 0,
        },
        # img 2
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "bbox": bbox(10, 10, 20, 20),
            "area": 400.0,
            "segmentation": [],
            "iscrowd": 0,
        },
        # img 3
        {
            "id": 4,
            "image_id": 3,
            "category_id": 3,
            "bbox": bbox(10, 10, 95, 95),
            "area": 9025.0,
            "segmentation": [],
            "iscrowd": 0,
        },
    ]
    # Detection results
    dts = [
        # img 1, cat 1: perfect match (TP)
        {"image_id": 1, "category_id": 1, "bbox": bbox(5, 5, 20, 20), "score": 0.95, "segmentation": []},
        # img 1, cat 2: good match (TP)
        {"image_id": 1, "category_id": 2, "bbox": bbox(30, 30, 50, 50), "score": 0.90, "segmentation": []},
        # img 1, cat 3: neg category → FP
        {"image_id": 1, "category_id": 3, "bbox": bbox(5, 5, 20, 20), "score": 0.85, "segmentation": []},
        # img 2, cat 1: perfect match (TP)
        {"image_id": 2, "category_id": 1, "bbox": bbox(10, 10, 20, 20), "score": 0.88, "segmentation": []},
        # img 2, cat 2: no GT, not_exhaustive → ignored
        {"image_id": 2, "category_id": 2, "bbox": bbox(20, 20, 30, 30), "score": 0.70, "segmentation": []},
        # img 3, cat 3: perfect match (TP)
        {"image_id": 3, "category_id": 3, "bbox": bbox(10, 10, 95, 95), "score": 0.92, "segmentation": []},
        # img 4, cat 1: neg category → FP
        {"image_id": 4, "category_id": 1, "bbox": bbox(5, 5, 20, 20), "score": 0.75, "segmentation": []},
        # img 4, cat 2: no GT, not_exhaustive → ignored
        {"image_id": 4, "category_id": 2, "bbox": bbox(10, 10, 30, 30), "score": 0.60, "segmentation": []},
    ]
    return make_lvis_gt(images, gt_anns, categories), dts


def _build_three_freq():
    """
    10 categories: 3 rare, 4 common, 3 frequent. Bounding boxes.

    5 images with various GT/DT patterns; tests that APr/APc/APf are computed
    correctly across groups.

    Every annotation here carries an empty `segmentation`, so this does not
    cover LVIS *segmentation* parity. Adding it means giving these annotations
    real polygons and running with `IOU_TYPE = "segm"`.
    """
    categories = (
        [{"id": i, "name": f"rare_{i}", "frequency": "r", "supercategory": "x"} for i in range(1, 4)]
        + [{"id": i, "name": f"common_{i}", "frequency": "c", "supercategory": "x"} for i in range(4, 8)]
        + [{"id": i, "name": f"freq_{i}", "frequency": "f", "supercategory": "x"} for i in range(8, 11)]
    )
    cat_ids = [c["id"] for c in categories]

    images = []
    gt_anns = []
    dts = []
    ann_id = 1

    rng = np.random.default_rng(0)

    for img_id in range(1, 6):
        # Randomly assign neg / not_exhaustive (avoid GT cats)
        neg_cats_for_img = []
        nexh_cats_for_img = []
        images.append(
            {
                "id": img_id,
                "file_name": f"img{img_id}.jpg",
                "width": 200,
                "height": 200,
                "neg_category_ids": neg_cats_for_img,
                "not_exhaustive_category_ids": nexh_cats_for_img,
            }
        )

        # Each image has GT for 4 randomly chosen categories
        chosen = rng.choice(cat_ids, size=4, replace=False).tolist()
        for cat_id in chosen:
            x, y = rng.integers(5, 50, size=2).tolist()
            w, h = rng.integers(20, 80, size=2).tolist()
            area = float(w * h)
            gt_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cat_id),
                    "bbox": bbox(x, y, w, h),
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            # Matching DT with high score (TP)
            dts.append(
                {
                    "image_id": img_id,
                    "category_id": int(cat_id),
                    "bbox": bbox(
                        x + rng.integers(-2, 3),
                        y + rng.integers(-2, 3),
                        w + rng.integers(-2, 3),
                        h + rng.integers(-2, 3),
                    ),
                    "score": float(rng.uniform(0.7, 0.99)),
                    "segmentation": [],
                }
            )
            ann_id += 1

            # Occasionally add a false positive (FP) DT on same cat, different box
            if rng.random() < 0.3:
                dts.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cat_id),
                        "bbox": bbox(rng.integers(5, 50), rng.integers(5, 50), 15, 15),
                        "score": float(rng.uniform(0.3, 0.6)),
                        "segmentation": [],
                    }
                )

        # For all non-GT categories, decide neg/nexh/drop
        non_gt_cats = [c for c in cat_ids if c not in chosen]
        for cat_id in non_gt_cats:
            roll = rng.random()
            if roll < 0.3:
                neg_cats_for_img.append(int(cat_id))
                # Add DT → should be FP
                dts.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cat_id),
                        "bbox": bbox(5, 5, 10, 10),
                        "score": float(rng.uniform(0.4, 0.8)),
                        "segmentation": [],
                    }
                )
            elif roll < 0.5:
                nexh_cats_for_img.append(int(cat_id))
                # Add DT → should be ignored
                dts.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cat_id),
                        "bbox": bbox(5, 5, 10, 10),
                        "score": float(rng.uniform(0.4, 0.8)),
                        "segmentation": [],
                    }
                )
            # else: truly absent, no DT (dropped by federated filter)

    return make_lvis_gt(images, gt_anns, categories), dts


def _build_edge_cases():
    """
    Edge cases:
      - Category with no GT anywhere (APl=-1 expected)
      - Image with only not_exhaustive DTs, no GT (all ignored)
      - Multiple unmatched DTs on not_exhaustive category (none should be FP)
      - DT matching crowd GT (still TP)
      - Category present in both neg and not_exhaustive for different images
    """
    categories = [
        {"id": 1, "name": "ghost", "frequency": "r", "supercategory": "x"},  # no GT anywhere
        {"id": 2, "name": "crowd_cat", "frequency": "c", "supercategory": "x"},
        {"id": 3, "name": "nexh_cat", "frequency": "f", "supercategory": "x"},
    ]
    images = [
        # img1: cat 2 has crowd GT; cat 1 in neg (no GT → DT is FP for cat 1)
        {
            "id": 1,
            "file_name": "img1.jpg",
            "width": 100,
            "height": 100,
            "neg_category_ids": [1],
            "not_exhaustive_category_ids": [],
        },
        # img2: cat 3 is not_exhaustive, 3 DTs → all ignored
        {
            "id": 2,
            "file_name": "img2.jpg",
            "width": 100,
            "height": 100,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [3],
        },
        # img3: standard GT for cat 2 and cat 3
        {
            "id": 3,
            "file_name": "img3.jpg",
            "width": 100,
            "height": 100,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        },
    ]
    gt_anns = [
        # img1: crowd GT for cat 2
        {
            "id": 1,
            "image_id": 1,
            "category_id": 2,
            "bbox": bbox(0, 0, 80, 80),
            "area": 6400.0,
            "segmentation": [],
            "iscrowd": 1,
        },
        # img3: regular GT for cat 2 and cat 3
        {
            "id": 2,
            "image_id": 3,
            "category_id": 2,
            "bbox": bbox(10, 10, 40, 40),
            "area": 1600.0,
            "segmentation": [],
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 3,
            "category_id": 3,
            "bbox": bbox(5, 5, 30, 30),
            "area": 900.0,
            "segmentation": [],
            "iscrowd": 0,
        },
    ]
    dts = [
        # img1, cat 1: neg category → FP
        {"image_id": 1, "category_id": 1, "bbox": bbox(5, 5, 10, 10), "score": 0.9, "segmentation": []},
        # img1, cat 2: overlaps crowd GT (TP via crowd matching, ignored as DT)
        {"image_id": 1, "category_id": 2, "bbox": bbox(0, 0, 80, 80), "score": 0.95, "segmentation": []},
        {"image_id": 1, "category_id": 2, "bbox": bbox(5, 5, 70, 70), "score": 0.80, "segmentation": []},
        # img2, cat 3: not_exhaustive, 3 DTs → all ignored
        {"image_id": 2, "category_id": 3, "bbox": bbox(0, 0, 20, 20), "score": 0.88, "segmentation": []},
        {"image_id": 2, "category_id": 3, "bbox": bbox(30, 30, 20, 20), "score": 0.75, "segmentation": []},
        {"image_id": 2, "category_id": 3, "bbox": bbox(60, 60, 15, 15), "score": 0.55, "segmentation": []},
        # img3, cat 2: matches GT (TP)
        {"image_id": 3, "category_id": 2, "bbox": bbox(10, 10, 40, 40), "score": 0.92, "segmentation": []},
        # img3, cat 3: matches GT (TP)
        {"image_id": 3, "category_id": 3, "bbox": bbox(5, 5, 30, 30), "score": 0.85, "segmentation": []},
    ]
    return make_lvis_gt(images, gt_anns, categories), dts


SCENARIOS = [("basic", _build_basic), ("three_freq", _build_three_freq), ("edge_cases", _build_edge_cases)]


@pytest.mark.parametrize(("name", "build"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_all_13_metrics_match_lvis_api(name, build):
    gt_data, dts = build()
    with written_json(gt_data, quiet=True) as (gt_path,):
        ref = run_lvis_ref(gt_path, dts)
        got = run_hotcoco(gt_path, dts)

    # lvis-api omits keys whose value is -1 (undefined); hotcoco reports -1.0.
    # Both are read against the evaluator's own key list, so a key missing from
    # both sides is a -1/-1 pair the comparison sees, not a metric it never meets.
    unknown = (set(ref) | set(got)) - set(METRIC_NAMES)
    assert not unknown, f"{name}: keys outside the LVIS metric list: {sorted(unknown)}"
    ref_v = [ref.get(k, -1.0) for k in METRIC_NAMES]
    got_v = [got.get(k, -1.0) for k in METRIC_NAMES]
    mismatches = compare_metrics(ref_v, got_v, METRIC_NAMES, tolerance=TOL)
    assert not mismatches, f"{name}: {len(mismatches)} metric(s) differ from lvis-api\n" + "\n".join(
        m.line() for m in mismatches
    )
