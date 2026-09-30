"""Regression tests for `StreamingEval` (candidate H, PyO3 layer).

`StreamingEval.add_image()`/`finalize()` must reproduce the batch
`COCO(dict)` + `COCOeval` pipeline exactly, one image at a time, and must
raise rather than silently misbehave once `finalize()` has consumed it. None
of these need the gitignored data/ directory.
"""

import warnings

import hotcoco
import pytest
from hotcoco import COCO, COCOeval, StreamingEval


def categories():
    return [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}, {"id": 3, "name": "no-gt"}]


def images():
    return [
        {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
        {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
    ]


def gt_annotations():
    return [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
        {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
    ]


def dt_annotations():
    return [
        {"id": 101, "image_id": 1, "category_id": 1, "bbox": [11, 11, 30, 30], "score": 0.9},
        # Unmatched — falls outside gt 2's box entirely.
        {"id": 102, "image_id": 2, "category_id": 2, "bbox": [5, 5, 10, 10], "score": 0.6},
    ]


def group_by_image(anns):
    by_image = {}
    for ann in anns:
        by_image.setdefault(ann["image_id"], []).append(ann)
    return by_image


def all_area_cell(eval_imgs):
    """The `aRng == [0.0, 1e10]` ("all") cell among an evaluator's `eval_imgs`."""
    return next(c for c in eval_imgs if c is not None and c["aRng"] == [0.0, 1e10])


# ---------------------------------------------------------------------------
# StreamingEval reproduces the batch pipeline
# ---------------------------------------------------------------------------


class TestStreamingMatchesBatch:
    def test_get_results_equal_to_batch(self):
        cats, imgs, gts, dts = categories(), images(), gt_annotations(), dt_annotations()

        gt_ds = {"images": imgs, "annotations": gts, "categories": cats}
        dt_ds = {"images": imgs, "annotations": dts, "categories": cats}
        batch = COCOeval(COCO(gt_ds), COCO(dt_ds).load_res(dts), "bbox")
        batch.evaluate()
        batch.accumulate()
        batch.summarize()

        se = StreamingEval(cats, iou_type="bbox")
        gt_by_image, dt_by_image = group_by_image(gts), group_by_image(dts)
        for image in imgs:
            se.add_image(image, gt_by_image.get(image["id"], []), dt_by_image.get(image["id"], []))
        streamed = se.finalize()
        streamed.accumulate()
        streamed.summarize()

        # Category 3 ("no-gt") has zero annotations on either side and must
        # still show up as a -1.0 slot in both, not vanish from the K axis.
        assert batch.get_results(per_class=True) == streamed.get_results(per_class=True)
        assert batch.stats.tolist() == streamed.stats.tolist()

    def test_empty_image_is_a_no_op(self):
        se = StreamingEval(categories(), iou_type="bbox")
        se.add_image({"id": 99, "width": 10, "height": 10}, [], [])
        ev = se.finalize()
        # No cells were ever populated, but `finalize()` still ran — the same
        # as a batch `COCOeval.evaluate()` on a zero-annotation dataset, which
        # does not warn on `accumulate()` either: `evaluated()` tracks whether
        # evaluation ran, not whether it found anything.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ev.accumulate()
        ev.summarize()
        assert all(v == -1.0 for v in ev.stats.tolist())


class TestStreamingLvisFederatedCategories:
    def test_neg_category_scores_unmatched_dt_as_fp(self):
        cats = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        image = {"id": 1, "width": 100, "height": 100, "neg_category_ids": [2]}
        se = StreamingEval(cats, iou_type="bbox", lvis_style=True)
        se.add_image(image, [], [{"id": 101, "image_id": 1, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9}])
        ev = se.finalize()
        # A confirmed-negative category is not dropped: the cell exists, and
        # the unmatched detection is not ignored (dtIgnore False) — it scores
        # as a false positive at every IoU threshold.
        cell = all_area_cell(ev.eval_imgs)
        assert cell["dtIds"] == [101]
        assert cell["dtIgnore"] == [[False]] * len(cell["dtIgnore"])

    def test_not_exhaustive_category_ignores_unmatched_dt(self):
        cats = [{"id": 1, "name": "a"}]
        image = {"id": 1, "width": 100, "height": 100, "not_exhaustive_category_ids": [1]}
        se = StreamingEval(cats, iou_type="bbox", lvis_style=True)
        se.add_image(
            image,
            [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}],
            # Unmatched, but not_exhaustive: ignored, not a false positive.
            [{"id": 101, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.9}],
        )
        ev = se.finalize()
        # dtIgnore is True at every threshold for the unmatched detection —
        # without not_exhaustive it would be False (a scored false positive).
        cell = all_area_cell(ev.eval_imgs)
        assert cell["dtIds"] == [101]
        assert cell["dtIgnore"] == [[True]] * len(cell["dtIgnore"])


# ---------------------------------------------------------------------------
# A spent StreamingEval raises instead of silently misbehaving
# ---------------------------------------------------------------------------


class TestSpentStreamingEvalRaises:
    def test_add_image_after_finalize_raises(self):
        se = StreamingEval(categories(), iou_type="bbox")
        se.add_image(images()[0], gt_annotations()[:1], dt_annotations()[:1])
        se.finalize()
        with pytest.raises(RuntimeError, match="spent"):
            se.add_image(images()[0], [], [])

    def test_finalize_twice_raises(self):
        se = StreamingEval(categories(), iou_type="bbox")
        se.add_image(images()[0], gt_annotations()[:1], dt_annotations()[:1])
        se.finalize()
        with pytest.raises(RuntimeError, match="spent"):
            se.finalize()


def test_streaming_eval_is_exported():
    assert hotcoco.StreamingEval is StreamingEval
