"""Panoptic parity: hotcoco vs panopticapi on synthetic label maps.

Every case is written to disk as COCO panoptic PNG files plus JSON, run through
panopticapi's own worker (`pq_compute_single_core`, the function `pq_compute`
fans out to), and through `hotcoco.panoptic.PanopticEval` on the same files.
Compared below the averages, where a mistake cannot cancel:

- per-category TP, FP, FN exactly and summed IoU to 1e-9;
- PQ, SQ, RQ and N for All, Things, Stuff to 1e-9.

The same label maps also go through the path with no PNG files — detection-
style datasets carrying RLEs — which must reproduce the PNG path exactly: the
same numbers, and the same per-image match records once each annotation id is
read back as the segment id it was painted from.

What the random cases exercise, by construction: crowd segments absorbing
predictions of their category, predictions mostly on void being ignored,
category mismatches, IoU on both sides of 0.5, segments dropped, merged and
shifted, images with empty predictions, and ground-truth `area` fields that
disagree with the PNG (panopticapi believes the JSON). The reference here is
the pinned panopticapi commit the dev extra installs; `scripts/parity_panoptic.py`
runs the same comparison on COCO val2017.

    uv run pytest tests/test_parity_panoptic.py -v
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import PANOPTIC_CATEGORIES as CATEGORIES
from helpers import PanopticCase as Case
from helpers import hand_panoptic_cases, panopticapi_reference, pq_disagreements, random_panoptic_cases, suppress_output
from hotcoco import panoptic
from panopticapi.evaluation import pq_compute

TOL = 1e-9

# Share of random cases that must land strictly inside (0, 1): a corpus of
# perfect or empty matches agrees with any implementation that returns 0 or 1.
MIN_DISCRIMINATING = 0.5

# ── running both ────────────────────────────────────────────────────────────


def run_reference(paths: dict) -> tuple[dict, dict]:
    """panopticapi on the written files: (averages, per-category counts)."""
    return panopticapi_reference(paths["gt"][0], paths["pred"][0], paths["gt"][1], paths["pred"][1], multi_core=False)


def evaluate_png(paths: dict) -> panoptic.PanopticEval:
    gt_json, gt_folder = paths["gt"]
    pred_json, pred_folder = paths["pred"]
    ev = panoptic.PanopticEval(gt_json, pred_json, gt_folder=gt_folder, pred_folder=pred_folder)
    ev.evaluate()
    return ev


def run_hotcoco_png(paths: dict) -> dict:
    return evaluate_png(paths).results()


def evaluate_rle(case: Case) -> panoptic.PanopticEval:
    ev = panoptic.PanopticEval(case.as_coco("gt"), case.as_coco("pred"))
    ev.evaluate()
    return ev


def as_segment_ids(case: Case, records: list[dict]) -> list[dict]:
    """The RLE run's ``eval_imgs`` with every annotation id read back as the segment
    id ``as_coco`` painted it from, which is the id the PNG run reports."""
    ids = {side: case.segment_ids(side) for side in ("gt", "pred")}

    def segment(side, image, ann_id):
        # An annotation of another image reads as no segment, never as one of this image's.
        return ids[side].get((image, ann_id), f"{side} annotation {ann_id}, not in image {image}")

    out = []
    for r in records:
        i = r["image_id"]
        out.append(
            {
                "image_id": i,
                "matched": [(segment("gt", i, g), segment("pred", i, p), iou) for g, p, iou in r["matched"]],
                "missed": [segment("gt", i, g) for g in r["missed"]],
                "false_positives": [segment("pred", i, p) for p in r["false_positives"]],
                "ignored": [segment("pred", i, p) for p in r["ignored"]],
            }
        )
    return out


def disagreements(ref_avg: dict, ref_counts: dict, got: dict) -> list[str]:
    return pq_disagreements(ref_avg, ref_counts, got, [c["id"] for c in CATEGORIES], tol=TOL)


RANDOM_CASES = random_panoptic_cases()
CASES = hand_panoptic_cases() + RANDOM_CASES


@pytest.fixture(params=CASES, ids=[c.name for c in CASES])
def written_case(request, tmp_path):
    case = request.param
    return case, case.write(tmp_path)


def test_random_cases_discriminate():
    """Most random cases must score strictly between 0 and 1."""
    inside = 0
    for case in RANDOM_CASES:
        with tempfile.TemporaryDirectory() as d:
            pq = run_hotcoco_png(case.write(Path(d)))["All"]["pq"]
        inside += 0.0 < pq < 1.0
    assert inside >= len(RANDOM_CASES) * MIN_DISCRIMINATING, f"{inside}/{len(RANDOM_CASES)} cases inside (0, 1)"


def test_png_path_matches_panopticapi(written_case):
    case, paths = written_case
    ref_avg, ref_counts = run_reference(paths)
    got = run_hotcoco_png(paths)
    why = disagreements(ref_avg, ref_counts, got)
    assert not why, f"{case.name} diverges from panopticapi:\n  " + "\n  ".join(why)


def test_rle_path_reproduces_png_path(written_case):
    """No PNG files, same pixels, same numbers — exactly — and the same matches.

    The numbers alone cannot see a record that names the wrong segment: PQ never
    reads an id.
    """
    case, paths = written_case
    if case.png_only:
        pytest.skip("case depends on PNG-only semantics")
    png_ev, rle_ev = evaluate_png(paths), evaluate_rle(case)
    from_png, from_rle = png_ev.results(), rle_ev.results()
    for key in ("All", "Things", "Stuff", "per_class"):
        assert from_rle[key] == from_png[key], f"{case.name}: {key} differs between the RLE and PNG paths"
    assert as_segment_ids(case, rle_ev.eval_imgs) == png_ev.eval_imgs, f"{case.name}: eval_imgs differ"


def test_pq_compute_is_a_drop_in(tmp_path):
    """Same signature, same keys, same values as panopticapi's `pq_compute`."""
    case = RANDOM_CASES[3]
    paths = case.write(tmp_path)
    with suppress_output(stderr=False):
        ref = pq_compute(str(paths["gt"][0]), str(paths["pred"][0]), str(paths["gt"][1]), str(paths["pred"][1]))
        got = panoptic.pq_compute(paths["gt"][0], paths["pred"][0], paths["gt"][1], paths["pred"][1])
    assert set(ref) <= set(got), f"panopticapi keys missing: {set(ref) - set(got)}"
    for split in ("All", "Things", "Stuff"):
        for key in ("pq", "sq", "rq", "n"):
            assert abs(ref[split][key] - got[split][key]) <= TOL, (split, key)
    for cid, ref_scores in ref["per_class"].items():
        g = got["per_class"][cid]
        if g["tp"] + g["fp"] + g["fn"] == 0:
            # panopticapi prints 0.0 for a category it then skips; hotcoco
            # says "not computed". The counts beside it are the tie-breaker.
            assert ref_scores == {"pq": 0.0, "sq": 0.0, "rq": 0.0}
            assert (g["pq"], g["sq"], g["rq"]) == (-1.0, -1.0, -1.0)
        else:
            for key in ("pq", "sq", "rq"):
                assert abs(ref_scores[key] - g[key]) <= TOL, (cid, key)

    # The default folders are the JSON paths without `.json` on both sides.
    ref_default = ref
    with suppress_output(stderr=False):
        got_default = panoptic.pq_compute(paths["gt"][0], paths["pred"][0])
    assert got_default["All"] == got["All"]
    assert abs(ref_default["All"]["pq"] - got_default["All"]["pq"]) <= TOL


def test_comparator_catches_a_wrong_number(tmp_path):
    """The check above must be able to fail: perturb one count and one score."""
    case = RANDOM_CASES[0]
    paths = case.write(tmp_path)
    ref_avg, ref_counts = run_reference(paths)
    got = run_hotcoco_png(paths)
    assert not disagreements(ref_avg, ref_counts, got)

    wrong = copy.deepcopy(got)
    wrong["All"]["pq"] += 1e-6
    assert any(p.startswith("All.pq") for p in disagreements(ref_avg, ref_counts, wrong))

    wrong = copy.deepcopy(got)
    cid = next(c for c, v in got["per_class"].items() if v["tp"] + v["fp"] + v["fn"] > 0)
    wrong["per_class"][cid]["fp"] += 1
    assert any(p.startswith(f"category {cid}.fp") for p in disagreements(ref_avg, ref_counts, wrong))


@pytest.mark.parametrize(
    "break_it, message",
    [
        (lambda pred: pred["annotations"][0]["segments_info"].pop(), "is in the label map but not in segments_info"),
        (
            lambda pred: pred["annotations"][0]["segments_info"].append({"id": 424242, "category_id": 1}),
            "is in segments_info but not in the label map",
        ),
        (
            lambda pred: pred["annotations"][0]["segments_info"][0].__setitem__("category_id", 999),
            "unknown category_id 999",
        ),
        (lambda pred: pred["annotations"].pop(), "no prediction for the image with id"),
        # Both at once: panopticapi checks a category in its loop over the PNG's
        # ids, before it looks for ids the JSON lists and the PNG lacks.
        (
            lambda pred: (
                pred["annotations"][0]["segments_info"][0].__setitem__("category_id", 999),
                pred["annotations"][0]["segments_info"].append({"id": 424242, "category_id": 1}),
            ),
            "unknown category_id 999",
        ),
    ],
    ids=["png-id-not-in-json", "json-id-not-in-png", "unknown-category", "missing-image", "category-first"],
)
def test_reference_errors_raise_here_too(tmp_path, break_it, message):
    """Where panopticapi raises, hotcoco raises, naming the same problem."""
    case = RANDOM_CASES[1]
    paths = case.write(tmp_path)
    pred_json = json.loads(paths["pred"][0].read_text())
    break_it(pred_json)
    paths["pred"][0].write_text(json.dumps(pred_json))

    with pytest.raises(Exception):
        with suppress_output(stderr=False):
            pq_compute(str(paths["gt"][0]), str(paths["pred"][0]), str(paths["gt"][1]), str(paths["pred"][1]))
    with pytest.raises(ValueError, match=message):
        run_hotcoco_png(paths)


def test_missing_isthing_is_an_extension(tmp_path):
    case = Case("no-isthing")
    gt = np.ones((4, 4), dtype=np.uint32)
    case.add(1, gt, [{"id": 1, "category_id": 1, "area": 16, "iscrowd": 0}], gt, [{"id": 1, "category_id": 1}])
    for c in case.categories:
        c.pop("isthing")
    paths = case.write(tmp_path)
    ev = panoptic.PanopticEval(paths["gt"][0], paths["pred"][0])
    assert ev.provenance() == "extension"
    assert "isthing" in ev.reference_deviations()[0]
    with pytest.warns(UserWarning, match="isthing"):
        ev.run()
    res = ev.results()
    assert res["All"]["pq"] == 1.0
    assert res["Things"]["n"] == 0 and res["Things"]["pq"] == -1.0
    report = ev.report()
    assert report["provenance"] == "extension"
    assert "things" not in report["per_group"]
    assert report["metrics"]["PQ"] == 1.0


def semantic_from_panoptic(case: Case) -> dict:
    """mIoU the way a panoptic→semantic conversion computes it: every segment
    painted with its category (crowd included), ground-truth void ignored, a
    predicted void counted as a miss for the ground-truth class and a false
    positive for nothing. Per-class IoU and mIoU by the usual formulas."""
    cat_ids = sorted({c["id"] for c in case.categories})
    n = len(cat_ids)
    conf = np.zeros((n, n + 1), dtype=np.int64)  # column n: predicted void
    for image_id, (gt_map, gt_segs) in case.gt.items():
        pred_map, pred_segs = case.pred[image_id]
        gt_cat = {s["id"]: cat_ids.index(s["category_id"]) for s in gt_segs}
        pred_cat = {s["id"]: cat_ids.index(s["category_id"]) for s in pred_segs}
        for g, p in zip(gt_map.ravel().tolist(), pred_map.ravel().tolist()):
            if g not in gt_cat:  # void, or a PNG id with no segments_info entry
                continue
            conf[gt_cat[g], pred_cat[p] if p != 0 else n] += 1
    tp = np.diag(conf[:, :n]).astype(float)
    gt_area = conf.sum(1).astype(float)
    pred_area = conf[:, :n].sum(0).astype(float)
    union = gt_area + pred_area - tp
    iou = np.where(union > 0, tp / np.where(union > 0, union, 1), np.nan)
    return {"IoU": iou, "mIoU": float(np.nanmean(iou)), "aAcc": tp.sum() / gt_area.sum(), "cat_ids": cat_ids}


@pytest.mark.parametrize("case", RANDOM_CASES[:12], ids=lambda c: c.name)
def test_panoptic_report_carries_semantic_miou(tmp_path, case):
    """`report()["metrics"]["mIoU"]` is the panoptic prediction scored as a semantic
    one — PQ and mIoU from one pass, as panoptic papers report them."""
    paths = case.write(tmp_path)
    ev = panoptic.PanopticEval(paths["gt"][0], paths["pred"][0], gt_folder=paths["gt"][1], pred_folder=paths["pred"][1])
    ev.evaluate()
    report = ev.report()
    want = semantic_from_panoptic(case)
    assert abs(report["metrics"]["mIoU"] - want["mIoU"]) < TOL
    assert abs(report["metrics"]["aAcc"] - want["aAcc"]) < TOL
    names = {c["id"]: c["name"] for c in case.categories}
    for cat_id, iou in zip(want["cat_ids"], want["IoU"]):
        got = report["per_class"].get(names[cat_id], {}).get("IoU")
        if np.isnan(iou):
            assert got is None, names[cat_id]
        else:
            assert got is not None and abs(got - iou) < TOL, names[cat_id]


def test_report_has_the_family_shape(tmp_path):
    case = RANDOM_CASES[2]
    paths = case.write(tmp_path)
    ev = panoptic.PanopticEval(paths["gt"][0], paths["pred"][0], gt_folder=paths["gt"][1], pred_folder=paths["pred"][1])
    ev.evaluate()
    report = ev.report()
    assert report["task"] == "panoptic"
    assert report["provenance"] == "parity_verified"
    assert set(report["metrics"]) == set(panoptic.METRIC_NAMES) | {"mIoU", "fwIoU", "mAcc", "aAcc"}
    assert ev.stats == [report["metrics"][k] for k in panoptic.METRIC_NAMES]
    assert set(report["per_group"]) <= {"all", "things", "stuff"}
    assert set(report["per_group"]["all"]) == {"PQ", "SQ", "RQ", "n"}
    for name, scores in report["per_class"].items():
        assert set(scores) <= {"PQ", "SQ", "RQ", "IoU", "Acc"}, name
        assert all(0.0 <= v <= 1.0 for v in scores.values()), name
    assert report["curves"] == {}
    assert report["params"]["n_images"] == 2
