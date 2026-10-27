"""Metrics written in Python from ``hotcoco.primitives`` equal the families' own.

This is what makes the primitives a public API rather than merely reachable: a
metric built from them in a few lines of Python reproduces the number the family
driver reports, exactly — ``==``, not a tolerance.

- **Panoptic quality.** ``segment_overlaps`` and ``match_segments`` per image,
  counts summed per category, ``metrics.panoptic_quality`` per category, then
  the mean: every PQ, SQ, and RQ that ``PanopticEval`` reports, every count, and
  every per-image match record.
- **Detection matching.** ``bbox_iou`` and ``greedy_match`` per (image,
  category): every match ``COCOeval`` records in ``evalImgs``, at all ten
  thresholds and all four area ranges.

The panoptic cases are the ones ``test_parity_panoptic.py`` checks
``PanopticEval`` against panopticapi on; this file checks the primitives
against ``PanopticEval``. ``test_build_your_own_metric.py`` checks the guide
page's metrics, true- and false-positive counts included, the same way.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
from helpers import detection_corpus, hand_panoptic_cases, random_panoptic_cases
from hotcoco import COCO, COCOeval, metrics, panoptic, primitives

CASES = hand_panoptic_cases() + random_panoptic_cases()

# ── panoptic quality from segment_overlaps + match_segments ──────────────────


def pq_from_primitives(images, categories):
    """PQ, SQ, and RQ the way ``PanopticEval`` computes them, from the primitives.

    ``images`` is a list of ``(gt_map, gt_segments, pred_map, pred_segments)``;
    the segment lists are COCO panoptic ``segments_info``. Returns the per-image
    match records, the per-category ``[iou_sum, tp, fp, fn]`` counts, and the
    ``All``/``Things``/``Stuff`` averages.
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


def cells(seed):
    """``COCOeval``'s ``evalImgs`` entries for a corpus, each beside the ground truths
    and detections of its (image, category), in the order the datasets hold them."""
    dataset, results = detection_corpus(seed)
    gt = COCO(dataset)
    dt = gt.load_res(results)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()

    def by_cell(coco):
        out = defaultdict(list)
        for ann in coco.dataset["annotations"]:
            out[ann["image_id"], ann["category_id"]].append(ann)
        return out

    gt_cells, dt_cells = by_cell(gt), by_cell(dt)
    for e in ev.eval_imgs:
        if e is not None:
            key = e["image_id"], e["category_id"]
            yield e, gt_cells[key], dt_cells[key]


def match_cell(gts, dts, area_range, iou_thrs):
    """One (image, category, area range) cell matched the way COCO evaluation does."""
    lo, hi = area_range
    iscrowd = [g["iscrowd"] for g in gts]
    sim = primitives.bbox_iou([d["bbox"] for d in dts], [g["bbox"] for g in gts], iscrowd)
    m = primitives.greedy_match(
        sim, [d["score"] for d in dts], iou_thrs, gt_ignore=[not lo <= g["area"] <= hi for g in gts], iscrowd=iscrowd
    )
    # COCO also ignores an unmatched detection outside the area range.
    outside = np.array([not lo <= d["area"] <= hi for d in dts], dtype=bool)
    ignore = m["dt_ignore"] | ((m["dt_matches"] < 0) & outside)
    return m, ignore


@pytest.mark.parametrize("seed", [0, 1])
def test_greedy_match_makes_every_match_cocoeval_makes(seed):
    crowd_absorbed_twice = ignored_ground_truth = False
    for e, gts, dts in cells(seed):
        # No thresholds given: greedy_match's default is COCOeval's ten.
        m, ignore = match_cell(gts, dts, e["aRng"], None)
        dt_row = {d["id"]: i for i, d in enumerate(dts)}
        gt_col = {g["id"]: i for i, g in enumerate(gts)}
        for t in range(len(e["dtMatches"])):
            # evalImgs lists detections best first and uses 0 for "no match".
            for k, d in enumerate(e["dtIds"]):
                got = m["dt_matches"][t, dt_row[d]]
                assert e["dtMatches"][t][k] == (gts[got]["id"] if got >= 0 else 0), (e["image_id"], t, d)
                assert e["dtIgnore"][t][k] == ignore[t, dt_row[d]], (e["image_id"], t, d)
            for k, g in enumerate(e["gtIds"]):
                got = m["gt_matches"][t, gt_col[g]]
                assert e["gtMatches"][t][k] == (dts[got]["id"] if got >= 0 else 0), (e["image_id"], t, g)
        hits = m["dt_matches"][0][m["dt_matches"][0] >= 0]
        crowd_absorbed_twice |= len(set(hits.tolist())) < len(hits)
        ignored_ground_truth |= any(not e["aRng"][0] <= g["area"] <= e["aRng"][1] for g in gts)

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
