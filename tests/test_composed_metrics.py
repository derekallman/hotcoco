"""Metrics written in Python from ``hotcoco.primitives`` equal the families' own.

This is what makes the primitives a public API rather than merely reachable: a
metric built from them in a few lines of Python reproduces the number the family
driver reports, exactly — ``==``, not a tolerance.

- **Panoptic quality.** ``segment_overlaps`` and ``match_segments`` per image,
  counts summed per category, ``metrics.panoptic_quality`` per category, then
  the mean: every PQ, SQ, and RQ that ``PanopticEval`` reports, every count, and
  every per-image match record.
- **Detection matching.** ``bbox_iou`` and ``greedy_match`` per (image,
  category): every match and ``dtIgnore`` flag ``COCOeval`` records in
  ``evalImgs``, at all ten thresholds and all four area ranges, and the true
  positives, false positives, and misses ``accumulate()`` counts from them,
  which ``helpers.eval_img_counts`` reads.
- **Open Images matching.** The same two calls with ``group_of`` in place of
  ``iscrowd``, on the cases ``test_parity_oid.py`` checks ``COCOeval`` against
  the TensorFlow Object Detection API on: every match and ``dtIgnore`` flag of
  an ``oid_style`` run, group-of boxes credited once, surplus detections
  absorbed, and undetected boxes missed.

The panoptic cases are the ones ``test_parity_panoptic.py`` checks
``PanopticEval`` against panopticapi on; this file checks the primitives
against ``PanopticEval``. ``test_build_your_own_metric.py`` checks the guide
page's metrics, true- and false-positive counts included, the same way, and
reads ``COCOeval``'s counts through the same helper.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pytest
from helpers import (
    FIXTURES_DIR,
    coco_counts,
    detection_corpus,
    eval_img_counts,
    hand_panoptic_cases,
    outside_area_range,
    random_panoptic_cases,
    suppress_output,
)
from hotcoco import COCO, COCOeval, metrics, panoptic, primitives

CASES = hand_panoptic_cases() + random_panoptic_cases()

# ── panoptic quality from segment_overlaps + match_segments ──────────────────


def pq_from_primitives(images, categories):
    """PQ, SQ, and RQ the way ``PanopticEval`` computes them, from the primitives.

    ``images`` is a list of ``(gt_map, gt_segments, pred_map, pred_segments)``;
    the segment lists are COCO panoptic ``segments_info``. Returns the per-image
    match records, the per-category ``[iou_sum, tp, fp, fn]`` counts, and the
    ``All``/``Things``/``Stuff`` averages.

    The counting is the guide page's ``image_counts`` and ``panoptic_counts``
    again, which ``test_build_your_own_metric.py`` holds to ``PanopticEval`` on
    the same cases. This copy stays for what that test does not hold: the
    averages to ``==`` rather than 1e-12, which takes ``PanopticEval``'s running
    sum; each category's PQ, SQ, and RQ to ``==``; the per-image match records
    against ``PanopticEval.eval_imgs``; and label maps taken from memory rather
    than read back from PNG files.
    """
    totals = {c["id"]: [0.0, 0, 0, 0] for c in categories}
    records = []
    for gt_map, gt_segments, pred_map, pred_segments in images:
        overlaps = primitives.segment_overlaps(gt_map, pred_map)
        m = primitives.match_segments(gt_segments, pred_segments, overlaps)
        records.append(m)

        gt_category = {s["id"]: s["category_id"] for s in gt_segments}
        pred_category = {s["id"]: s["category_id"] for s in pred_segments}
        # One image's counts, then into the totals: summing the IoUs in this
        # order is what makes them equal to the last bit.
        image = defaultdict(lambda: [0.0, 0, 0, 0])
        for gt_id, _, iou in m["matched"]:
            image[gt_category[gt_id]][0] += iou
            image[gt_category[gt_id]][1] += 1
        for pred_id in m["false_positives"]:
            image[pred_category[pred_id]][2] += 1
        for gt_id in m["missed"]:
            image[gt_category[gt_id]][3] += 1
        for category_id, counts in image.items():
            for k in range(4):
                totals[category_id][k] += counts[k]

    averages = {}
    for split, isthing in (("All", None), ("Things", 1), ("Stuff", 0)):
        # A category with nothing on either side is -1.0 and left out of the mean.
        pq = sq = rq = 0.0
        n = 0
        for c in categories:
            if isthing is not None and c["isthing"] != isthing:
                continue
            scores = metrics.panoptic_quality(*totals[c["id"]])
            if metrics.is_computed(scores[0]):
                # A running sum, not sum(): since Python 3.12 sum() compensates
                # for rounding, and the last bit then differs.
                pq, sq, rq, n = pq + scores[0], sq + scores[1], rq + scores[2], n + 1
        averages[split] = {
            "pq": pq / n if n else -1.0,
            "sq": sq / n if n else -1.0,
            "rq": rq / n if n else -1.0,
            "n": n,
        }
    return records, totals, averages


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_pq_from_the_primitives_is_panoptic_eval(tmp_path, case):
    paths = case.write(tmp_path)
    ev = panoptic.PanopticEval(paths["gt"][0], paths["pred"][0], gt_folder=paths["gt"][1], pred_folder=paths["pred"][1])
    assert ev.eval_imgs == []
    ev.evaluate()
    want = ev.results()

    images = [(*case.gt[image_id], *case.pred[image_id]) for image_id in case.gt]
    records, totals, averages = pq_from_primitives(images, case.categories)

    for split in ("All", "Things", "Stuff"):
        assert averages[split] == want[split], split
    for category_id, (iou_sum, tp, fp, fn) in totals.items():
        got = want["per_class"][category_id]
        assert (iou_sum, tp, fp, fn) == (got["iou"], got["tp"], got["fp"], got["fn"]), category_id
        assert metrics.panoptic_quality(iou_sum, tp, fp, fn) == (got["pq"], got["sq"], got["rq"])
    # The records PanopticEval keeps are the primitive's answers, image by image.
    assert ev.eval_imgs == [{"image_id": image_id, **m} for image_id, m in zip(case.gt, records)]


def test_match_segments_takes_panopticapis_own_histogram():
    """``overlaps`` may come from panopticapi's ``np.unique`` idiom instead."""
    gt_map, gt_segments = CASES[0].gt[1]
    pred_map, pred_segments = CASES[0].pred[1]
    offset = 256 * 256 * 256
    labels, counts = np.unique(gt_map.astype(np.uint64) * offset + pred_map, return_counts=True)
    gt_pred_map = {(label // offset, label % offset): n for label, n in zip(labels, counts)}
    overlaps = primitives.segment_overlaps(gt_map, pred_map)
    assert gt_pred_map == overlaps
    assert primitives.match_segments(gt_segments, pred_segments, gt_pred_map) == primitives.match_segments(
        gt_segments, pred_segments, overlaps
    )


# ── detection matches from bbox_iou + greedy_match ───────────────────────────

EVERY_AREA = [0.0, 1e10]


def eval_img_cells(ev):
    """``ev``'s ``evalImgs`` entries after ``evaluate()``, each beside the ground
    truths and detections of its (image, category), in the order the datasets hold
    them."""

    def by_cell(coco):
        out = defaultdict(list)
        for ann in coco.dataset["annotations"]:
            out[ann["image_id"], ann["category_id"]].append(ann)
        return out

    gt_cells, dt_cells = by_cell(ev.coco_gt), by_cell(ev.coco_dt)
    for e in ev.eval_imgs:
        if e is not None:
            key = e["image_id"], e["category_id"]
            yield e, gt_cells[key], dt_cells[key]


def cells(seed):
    """The corpus's cells under COCO evaluation."""
    dataset, results = detection_corpus(seed)
    gt = COCO(dataset)
    ev = COCOeval(gt, gt.load_res(results), "bbox")
    ev.evaluate()
    return eval_img_cells(ev)


def assert_same_matches(m, e, gts, dts):
    """``greedy_match``'s result ``m`` for a cell, by index, is the ``evalImgs`` entry
    ``e`` for it, by annotation id, at every threshold."""
    dt_row = {d["id"]: i for i, d in enumerate(dts)}
    gt_col = {g["id"]: i for i, g in enumerate(gts)}
    dt_matches, gt_matches, dt_ignore = (np.atleast_2d(m[k]) for k in ("dt_matches", "gt_matches", "dt_ignore"))
    # evalImgs lists detections best first; its dtIgnore is greedy_match's, verbatim.
    ranked = [dt_row[d] for d in e["dtIds"]]
    want_ignore = np.array(e["dtIgnore"], dtype=bool).reshape(len(e["dtMatches"]), len(ranked))
    np.testing.assert_array_equal(dt_ignore[:, ranked], want_ignore, err_msg=str(e["image_id"]))
    for t in range(len(e["dtMatches"])):
        # evalImgs uses 0 for "no match".
        for k, d in enumerate(e["dtIds"]):
            got = dt_matches[t, dt_row[d]]
            assert e["dtMatches"][t][k] == (gts[got]["id"] if got >= 0 else 0), (e["image_id"], t, d)
        for k, g in enumerate(e["gtIds"]):
            got = gt_matches[t, gt_col[g]]
            assert e["gtMatches"][t][k] == (dts[got]["id"] if got >= 0 else 0), (e["image_id"], t, g)


def match_cell(gts, dts, area_range, iou_thrs):
    """One (image, category, area range) cell matched the way COCO evaluation does,
    and ``COCOeval``'s ``gtIgnore`` for it."""
    iscrowd = [g["iscrowd"] for g in gts]
    gt_outside = outside_area_range(gts, area_range)
    sim = primitives.bbox_iou([d["bbox"] for d in dts], [g["bbox"] for g in gts], iscrowd)
    m = primitives.greedy_match(
        sim,
        [d["score"] for d in dts],
        iou_thrs,
        gt_ignore=gt_outside.tolist(),
        iscrowd=iscrowd,
        dt_ignore=outside_area_range(dts, area_range),
    )
    return m, np.array(iscrowd, dtype=bool) | gt_outside


@pytest.mark.parametrize("seed", [0, 1])
def test_greedy_match_makes_every_match_cocoeval_makes(seed):
    crowd_absorbed_twice = ignored_ground_truth = False
    for e, gts, dts in cells(seed):
        # No thresholds given: greedy_match's default is COCOeval's ten.
        m, gt_ignore = match_cell(gts, dts, e["aRng"], None)
        assert_same_matches(m, e, gts, dts)
        for t in range(len(e["dtMatches"])):
            # What accumulate() counts from the cell, the ground truths it ignores included.
            counts = coco_counts(m["dt_matches"][t] >= 0, m["dt_ignore"][t], gt_ignore)
            assert counts == eval_img_counts(e, t), (e["image_id"], e["category_id"], e["aRng"], t)
        hits = m["dt_matches"][0][m["dt_matches"][0] >= 0]
        crowd_absorbed_twice |= len(set(hits.tolist())) < len(hits)
        ignored_ground_truth |= bool(outside_area_range(gts, e["aRng"]).any())

    # The corpus reaches the rules it was built for.
    assert crowd_absorbed_twice and ignored_ground_truth


def test_the_comparison_sees_tie_order():
    """The check above can fail: the same matches with the ground truths in reverse
    order break an equal-IoU tie the other way somewhere in the corpus."""
    differs = False
    for e, gts, dts in cells(seed=0):
        if e["aRng"] != EVERY_AREA:
            continue
        m, _ = match_cell(gts, dts, EVERY_AREA, 0.5)
        r, _ = match_cell(gts[::-1], dts, EVERY_AREA, 0.5)
        flipped = np.where(r["dt_matches"] >= 0, len(gts) - 1 - r["dt_matches"], -1)
        differs |= bool(np.any(flipped != m["dt_matches"]))
    assert differs


# ── Open Images matches from bbox_iou + greedy_match(group_of=) ──────────────

OID_CASES = json.loads((FIXTURES_DIR / "oid_tf_expected.json").read_text())["cases"]


# Open Images evaluates one range, ``all``; the second run splits the fixture's
# boxes, so some group-of boxes fall outside the range being scored.
OID_AREA_RANGES = [[[0, 1e10]], [[0, 1e10], [0, 20_000], [20_000, 1e10]]]


def oid_cells():
    """Every cell of every Open Images parity case under ``oid_style`` evaluation,
    at its one threshold, 0.5, under the default area range and then a split."""
    for case in OID_CASES:
        gt = COCO(
            {
                "images": [{"id": 1, "width": 800, "height": 800, "file_name": "a.jpg"}],
                "annotations": case["annotations"],
                "categories": case["categories"],
            }
        )
        for ranges in OID_AREA_RANGES:
            ev = COCOeval(gt, gt.load_res(case["detections"]), "bbox", oid_style=True)
            ev.params.areaRng = ranges
            ev.params.areaRngLbl = ["all", "small", "large"][: len(ranges)]
            with suppress_output():  # the Extension provenance notice, once per run
                ev.evaluate()
            yield from eval_img_cells(ev)


def match_oid_cell(gts, dts, area_range, flag):
    """One cell matched the Open Images way, the group-of flags passed as ``flag``.
    Open Images reads ``is_group_of`` where COCO reads ``iscrowd``, for the IoA
    columns of ``sim`` and for the matcher alike; the area range flags both sides
    as under COCO."""
    group_of = [bool(g.get("is_group_of", False)) for g in gts]
    sim = primitives.bbox_iou([d["bbox"] for d in dts], [g["bbox"] for g in gts], group_of)
    return primitives.greedy_match(
        sim,
        [d["score"] for d in dts],
        0.5,
        gt_ignore=outside_area_range(gts, area_range),
        dt_ignore=outside_area_range(dts, area_range),
        **{flag: group_of},
    )


def test_greedy_match_makes_every_match_open_images_makes():
    credited = absorbed = missed = outside = False
    for e, gts, dts in oid_cells():
        m = match_oid_cell(gts, dts, e["aRng"], "group_of")
        assert_same_matches(m, e, gts, dts)
        # The cell's true positives, false positives, and misses: a group-of box
        # counts as one ground truth, found or not, so only the area range ignores.
        matched = m["dt_matches"] >= 0
        gt_outside = outside_area_range(gts, e["aRng"])
        counts = coco_counts(matched, m["dt_ignore"], gt_outside)
        assert counts == eval_img_counts(e), (e["image_id"], e["category_id"], e["aRng"])

        group_of = np.array([bool(g.get("is_group_of", False)) for g in gts], dtype=bool)
        on_group = matched & np.isin(m["dt_matches"], np.flatnonzero(group_of))
        on_outside = matched & np.isin(m["dt_matches"], np.flatnonzero(group_of & gt_outside))
        credited |= bool(np.any(on_group & ~m["dt_ignore"]))
        absorbed |= bool(np.any(on_group & m["dt_ignore"] & ~on_outside))
        missed |= bool(np.any(group_of & ~gt_outside & (m["gt_matches"] < 0)))
        # A detection inside a group-of box the range leaves out is ignored, never
        # a true positive against a count that has no box for it.
        assert not np.any(on_outside & ~m["dt_ignore"]), (e["image_id"], e["aRng"])
        outside |= bool(np.any(on_outside))
    # The fixture reaches every clause of the rule.
    assert credited and absorbed and missed and outside


def test_iscrowd_is_not_group_of():
    """The trap the keyword exists for: the two flags share the IoA measure, so
    ``iscrowd=group_of`` looks right and is not. A crowd region absorbs every
    detection as ignored; a group-of box credits the first as a true positive, so the
    composed count comes out low with no error. The check above tells them apart."""
    differs = False
    for e, gts, dts in oid_cells():
        if not any(g.get("is_group_of") for g in gts):
            continue
        m = match_oid_cell(gts, dts, e["aRng"], "iscrowd")
        try:
            assert_same_matches(m, e, gts, dts)
        except AssertionError:
            differs = True
    assert differs
