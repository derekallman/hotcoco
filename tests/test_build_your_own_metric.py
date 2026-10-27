"""The code on the "Build your own metric" guide page runs, and its numbers are the families'.

Every ``python`` block on ``docs/guide/custom-metrics.md`` runs, in page order and
one namespace, in a directory holding the files the page names. The page is the
source — nothing here copies its code — so it cannot drift from what is tested.
Each metric it defines is then held against the family that owns the number:

- ``detection_counts``: the true positives, false positives, and misses per
  category that ``COCOeval`` records in ``eval_imgs``, at two IoU thresholds, two
  score thresholds, and two area ranges.
- ``confusion``: ``COCOeval.confusion_matrix()``, at two IoU thresholds and two
  detection caps.
- ``panoptic_counts`` and ``pq_average``: ``PanopticEval``'s per-category counts
  exactly, IoU sums included, and its PQ, SQ, and RQ to 1e-12, on every case
  ``test_composed_metrics.py`` checks. The page averages with ``numpy.mean``;
  that file holds the averages to ``==`` with a running sum.

The detection corpus lists its ground truths out of id order, as val2017 does,
and :func:`add_edge_cases` adds what the random corpus never reaches, so a page
that reads annotations in id order, forgets a rule, or drops an input
``COCOeval`` keeps fails here.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict

import numpy as np
import pytest
from helpers import WORKSPACE, detection_corpus, hand_panoptic_cases, random_panoptic_case, random_panoptic_cases
from hotcoco import COCO, COCOeval, panoptic

PAGE = WORKSPACE / "docs" / "guide" / "custom-metrics.md"
CASES = hand_panoptic_cases() + random_panoptic_cases()
EVERY_AREA, SMALL = (0, 1e10), (0, 32**2)
NO_THRESHOLD = float("-inf")


def page_blocks(section=None):
    """The page's ``python`` blocks, or those under one ``##`` heading."""
    text = PAGE.read_text(encoding="utf-8")
    if section is not None:
        text = text.split(f"\n## {section}\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^```python\n(.*?)^```", text, re.DOTALL | re.MULTILINE)


def run_blocks(blocks, root):
    """Execute ``blocks`` in order, in one fresh namespace, from ``root``."""
    namespace: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        for i, block in enumerate(blocks, 1):
            exec(compile(block, f"{PAGE.name}, python block {i}", "exec"), namespace)
    return namespace


def add_edge_cases(dataset, results, seed):
    """What the random corpus never produces, where a page that gets a rule wrong
    gets a different number.

    - Image 1, category 1: 120 more detections at three scores, past COCO's cap
      of 100, with ties at the cut.
    - Image 16, one category: the first detection overlaps two ground truths
      equally and takes the later one, which leaves the first for the second
      detection — two true positives. In the other order it is one and a false
      positive. The ground truth listed first has the larger id, so reading in id
      order is the other order.
    - Image 17, across categories: two equal-score detections of classes 2 and 1
      on one class-1 ground truth, listed class 2 first, and two identical ground
      truths of classes 2 and 1 under a class-3 detection, listed class 2 first.
      COCOeval's confusion pass sorts both sides by category first.
    - Image 18: a detection scored -2.0, which COCOeval matches like any other,
      and one of category 99, which the ground truth does not list and
      ``load_res`` keeps.
    """
    dataset["images"] += [{"id": i, "width": 100, "height": 100} for i in (16, 17, 18)]
    n = max(a["id"] for a in dataset["annotations"])
    for ann_id, image_id, category_id, bbox in [
        (n + 2, 16, 2, [10, 0, 10, 10]),
        (n + 1, 16, 2, [12, 0, 10, 10]),
        (n + 3, 17, 1, [0, 0, 10, 10]),
        (n + 4, 17, 2, [50, 50, 10, 10]),
        (n + 5, 17, 1, [50, 50, 10, 10]),
        (n + 6, 18, 1, [0, 0, 10, 10]),
    ]:
        dataset["annotations"].append(
            {"id": ann_id, "image_id": image_id, "category_id": category_id, "bbox": bbox, "iscrowd": 0, "area": 100.0}
        )
    for image_id, category_id, bbox, score in [
        (16, 2, [11, 0, 10, 10], 0.9),  # IoU 0.82 with both ground truths
        (16, 2, [8, 0, 10, 10], 0.8),  # IoU 0.67 with the first, 0.43 with the second
        (17, 2, [1, 0, 10, 10], 0.5),
        (17, 1, [1, 0, 10, 10], 0.5),
        (17, 3, [51, 50, 10, 10], 0.7),
        (18, 1, [0, 0, 10, 10], -2.0),
        (18, 99, [0, 0, 10, 10], 0.9),
    ]:
        results.append({"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score})
    rng = random.Random(seed)
    for _ in range(120):
        box = [rng.randint(0, 80), rng.randint(0, 80), rng.choice([10, 20]), 20]
        results.append({"image_id": 1, "category_id": 1, "bbox": box, "score": rng.choice([0.2, 0.4, 0.6])})


def write_detection_files(root, seed):
    """``detection_corpus`` and :func:`add_edge_cases` as the page's two files."""
    dataset, results = detection_corpus(seed)
    add_edge_cases(dataset, results, seed)
    (root / "instances_val2017.json").write_text(json.dumps(dataset))
    (root / "detections.json").write_text(json.dumps(results))
    gt = COCO(str(root / "instances_val2017.json"))
    return gt, gt.load_res(str(root / "detections.json"))


def write_panoptic_files(root, case):
    """A panoptic case under the names the page uses."""
    paths = case.write(root)
    for side, name in (("gt", "panoptic_val2017"), ("pred", "panoptic_predictions")):
        json_path, folder = paths[side]
        json_path.rename(root / f"{name}.json")
        folder.rename(root / name)


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The page's namespace after every block has run."""
    root = tmp_path_factory.mktemp("page")
    write_detection_files(root, seed=0)
    write_panoptic_files(root, random_panoptic_case(0))
    namespace = run_blocks(page_blocks(), root)
    namespace["root"] = root
    return namespace


def as_tuples(counts):
    return {c: tuple(np.asarray(n).tolist()) for c, n in counts.items()}


def counts_from_eval_imgs(ev, area_range, score_thr, t=0):
    """``[tp, fp, fn]`` per category from ``COCOeval``'s records at ``iou_thrs[t]``."""
    counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for e in ev.eval_imgs:
        if e is None or e["aRng"] != list(area_range):
            continue
        confident = np.array(e["dtScores"], dtype=float) >= score_thr
        matched = np.array(e["dtMatches"][t], dtype=float) > 0  # 0 is "no match"
        ignored = np.array(e["dtIgnore"][t], dtype=bool)
        tp = np.sum(matched & ~ignored & confident)
        fp = np.sum(~matched & ~ignored & confident)
        counts[e["category_id"]] += (tp, fp, np.sum(~np.array(e["gtIgnore"], dtype=bool)) - tp)
    return as_tuples(counts)


def test_the_pages_own_numbers_are_the_families(page):
    """What the page's example lines compute, on the files they read."""
    ev = COCOeval(page["gt"], page["dt"], "bbox")
    ev.evaluate()
    want = counts_from_eval_imgs(ev, EVERY_AREA, 0.5)
    assert as_tuples(page["counts"]) == want
    tp, fp, fn = np.sum(list(want.values()), axis=0)
    assert (page["precision"], page["recall"], page["f1"]) == (
        tp / (tp + fp),
        tp / (tp + fn),
        2 * tp / (2 * tp + fp + fn),
    )
    assert as_tuples(page["small"]) == counts_from_eval_imgs(ev, SMALL, NO_THRESHOLD)
    np.testing.assert_array_equal(page["cm"], ev.confusion_matrix()["matrix"])

    root = page["root"]
    pe = panoptic.PanopticEval(
        str(root / "panoptic_val2017.json"),
        str(root / "panoptic_predictions.json"),
        gt_folder=str(root / "panoptic_val2017"),
        pred_folder=str(root / "panoptic_predictions"),
    )
    pe.evaluate()
    want_pq = pe.results()
    got = {"All": (page["pq"], page["sq"], page["rq"]), "Things": page["pq_things"], "Stuff": page["pq_stuff"]}
    for split, scores in got.items():
        w = want_pq[split]
        assert scores == pytest.approx((w["pq"], w["sq"], w["rq"]), rel=1e-12), split


def test_the_panoptic_section_runs_on_its_own(page):
    """Two pages link straight to it, so a reader can start there."""
    section = run_blocks(page_blocks("Panoptic quality from the segment matcher"), page["root"])
    assert (section["pq"], section["sq"], section["rq"]) == (page["pq"], page["sq"], page["rq"])


def test_precision_recall_f1_marks_what_it_cannot_divide(page):
    prf = page["precision_recall_f1"]
    assert prf(0, 0, 0) == (-1.0, -1.0, -1.0)
    assert prf(0, 2, 0) == (0.0, -1.0, 0.0)  # detections of a class no ground truth has
    assert prf(0, 0, 3) == (-1.0, 0.0, 0.0)  # ground truths no detection found


@pytest.mark.parametrize("seed", [0, 1])
def test_detection_counts_are_cocoevals(page, tmp_path, seed):
    gt, dt = write_detection_files(tmp_path, seed)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    for t in (0, 5):  # IoU 0.50 and 0.75
        for area_range in (EVERY_AREA, SMALL):
            for score_thr in (NO_THRESHOLD, 0.5):
                got = page["detection_counts"](
                    gt, dt, ev.params.iou_thrs[t], score_thr=score_thr, area_range=area_range
                )
                want = counts_from_eval_imgs(ev, area_range, score_thr, t)
                assert as_tuples(got) == want, (t, area_range, score_thr)


@pytest.mark.parametrize("seed", [0, 1])
def test_confusion_is_cocoevals(page, tmp_path, seed):
    gt, dt = write_detection_files(tmp_path, seed)
    ev = COCOeval(gt, dt, "bbox")
    for iou_thr in (0.5, 0.75):
        for max_det in (100, 50):
            np.testing.assert_array_equal(
                page["confusion"](gt, dt, iou_thr, max_det), ev.confusion_matrix(iou_thr, max_det)["matrix"]
            )


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_panoptic_quality_is_panoptic_evals(page, tmp_path, case):
    paths = case.write(tmp_path)
    (gt_json, gt_folder), (pred_json, pred_folder) = paths["gt"], paths["pred"]
    categories, counts = page["panoptic_counts"](gt_json, pred_json, gt_folder, pred_folder)
    ev = panoptic.PanopticEval(str(gt_json), str(pred_json), gt_folder=str(gt_folder), pred_folder=str(pred_folder))
    ev.evaluate()
    want = ev.results()

    for category_id, (iou_sum, tp, fp, fn) in counts.items():
        got = want["per_class"][category_id]
        assert (iou_sum, tp, fp, fn) == (got["iou"], got["tp"], got["fp"], got["fn"]), category_id
    things = [c for c in categories if c["isthing"]]
    stuff = [c for c in categories if not c["isthing"]]
    for split, members in (("All", categories), ("Things", things), ("Stuff", stuff)):
        w = want[split]
        assert page["pq_average"](members, counts) == pytest.approx((w["pq"], w["sq"], w["rq"]), rel=1e-12), split
