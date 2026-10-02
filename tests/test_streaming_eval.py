"""`StreamingEval` reproduces the batch `COCO(dict)` + `COCOeval` pipeline.

Images go in one at a time, in any order; the finalized evaluator must give
the same stats and per-class results as a batch run over the same
annotations, in COCO and LVIS mode, and must raise once `finalize()` has
consumed it. No `data/` needed.
"""

import hotcoco
import pytest
from hotcoco import COCO, COCOeval, StreamingEval


def categories():
    return [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}, {"id": 3, "name": "no-gt"}]


def images():
    return [
        {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
        {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
        {"id": 3, "width": 100, "height": 100, "file_name": "c.jpg"},
    ]


def gt_annotations():
    return [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
        {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [0, 0, 40, 40], "area": 1600, "iscrowd": 0},
    ]


def dt_annotations():
    # Raw predictions, as a training loop has them: no `id`, no `area`.
    return [
        {"image_id": 1, "category_id": 1, "bbox": [11, 11, 30, 30], "score": 0.9},
        # Two detections tied on score for one ground truth: order decides.
        {"image_id": 2, "category_id": 1, "bbox": [1, 1, 40, 40], "score": 0.8},
        {"image_id": 2, "category_id": 1, "bbox": [2, 2, 40, 40], "score": 0.8},
        # Unmatched: falls outside gt 2's box entirely.
        {"image_id": 2, "category_id": 2, "bbox": [5, 5, 10, 10], "score": 0.6},
        # Detection-only image: no ground truth at all in image 3.
        {"image_id": 3, "category_id": 1, "bbox": [5, 5, 10, 10], "score": 0.5},
    ]


def group_by_image(anns):
    by_image = {}
    for ann in anns:
        by_image.setdefault(ann["image_id"], []).append(ann)
    return by_image


def batch_eval(cats, imgs, gts, dts, **kwargs):
    gt = COCO({"images": imgs, "annotations": gts, "categories": cats})
    ev = COCOeval(gt, gt.load_res(dts), "bbox", **kwargs)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def streamed_eval(cats, imgs, gts, dts, batch_size=2, **kwargs):
    se = StreamingEval(cats, iou_type="bbox", **kwargs)
    gt_by_image, dt_by_image = group_by_image(gts), group_by_image(dts)
    # Reverse arrival order, in detector-sized batches: finalize() must lay
    # the images out by id whatever the batching.
    order = list(reversed(imgs))
    for i in range(0, len(order), batch_size):
        batch = order[i : i + batch_size]
        ids = [img["id"] for img in batch]
        se.update(
            batch,
            [a for img_id in ids for a in gt_by_image.get(img_id, [])],
            [d for img_id in ids for d in dt_by_image.get(img_id, [])],
        )
    ev = se.finalize()
    ev.accumulate()
    ev.summarize()
    return ev


class TestStreamingMatchesBatch:
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_equals_batch_for_any_batching(self, batch_size):
        cats, imgs, gts, dts = categories(), images(), gt_annotations(), dt_annotations()
        batch = batch_eval(cats, imgs, gts, dts)
        streamed = streamed_eval(cats, imgs, gts, dts, batch_size=batch_size)

        assert streamed.stats.tolist() == batch.stats.tolist()
        # Category 3 has no annotations on either side: a -1.0 slot in both,
        # not a missing key.
        assert streamed.get_results(per_class=True) == batch.get_results(per_class=True)
        assert streamed.eval["precision"].tolist() == batch.eval["precision"].tolist()
        assert streamed.eval["scores"].tolist() == batch.eval["scores"].tolist()
        # The lean cells are enough for stats; the full per-image records
        # would need the datasets, which a StreamingEval does not keep.
        assert streamed.eval_imgs == [] == streamed.evalImgs

    def test_lvis_federated_categories_equal_batch(self):
        cats = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        imgs = [
            {"id": 1, "width": 100, "height": 100, "neg_category_ids": [2]},
            {"id": 2, "width": 100, "height": 100, "not_exhaustive_category_ids": [1]},
            {"id": 3, "width": 100, "height": 100},
        ]
        gts = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
        ]
        dts = [
            {"id": 101, "image_id": 1, "category_id": 1, "bbox": [12, 12, 50, 50], "score": 0.9},
            # Confirmed-negative category: a scored false positive.
            {"id": 102, "image_id": 1, "category_id": 2, "bbox": [200, 200, 50, 50], "score": 0.8},
            # Not-exhaustive category: an ignored unmatched detection.
            {"id": 103, "image_id": 2, "category_id": 1, "bbox": [300, 300, 50, 50], "score": 0.7},
            # No label either way: the pair is dropped from evaluation.
            {"id": 104, "image_id": 3, "category_id": 2, "bbox": [50, 50, 50, 50], "score": 0.6},
        ]
        batch = batch_eval(cats, imgs, gts, dts, lvis_style=True)
        streamed = streamed_eval(cats, imgs, gts, dts, lvis_style=True)

        assert streamed.stats.tolist() == batch.stats.tolist()
        assert streamed.get_results() == batch.get_results()
        assert streamed.eval["precision"].tolist() == batch.eval["precision"].tolist()

    def test_empty_run_reports_the_sentinel(self):
        se = StreamingEval(categories(), iou_type="bbox")
        se.update([{"id": 99, "width": 10, "height": 10}], [], [])
        ev = se.finalize()
        ev.accumulate()
        ev.summarize()
        assert all(v == -1.0 for v in ev.stats.tolist())


class TestUnknownCategory:
    """A label outside `categories` raises instead of vanishing from every metric."""

    @staticmethod
    def streaming(**kwargs):
        return StreamingEval([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], iou_type="bbox", **kwargs)

    def test_detection_in_an_unlisted_category_raises_key_error(self):
        se = self.streaming()
        dt = [{"image_id": 2, "category_id": 7, "bbox": [0, 0, 5, 5], "score": 0.5}]
        with pytest.raises(KeyError, match=r"\[7\]"):
            se.update([{"id": 2, "width": 10, "height": 10}], [], dt)

    def test_ground_truth_in_an_unlisted_category_raises_key_error(self):
        se = self.streaming()
        gts = [
            {"id": 1, "image_id": 2, "category_id": 9, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 4, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0},
        ]
        # Every offender, sorted — not only the first.
        with pytest.raises(KeyError, match=r"\[4, 9\]"):
            se.update([{"id": 2, "width": 10, "height": 10}], gts, [])

    def test_a_rejected_batch_is_not_recorded(self):
        se = self.streaming()
        se.update(images()[:1], gt_annotations()[:1], dt_annotations()[:1])
        bad = [{"image_id": 2, "category_id": 7, "bbox": [0, 0, 5, 5], "score": 0.5}]
        with pytest.raises(KeyError):
            se.update(images()[1:2], [], bad)
        ev = se.finalize()
        assert list(ev.params.img_ids) == [1]

    def test_a_listed_category_outside_cat_ids_is_not_an_error(self):
        params = hotcoco.Params("bbox")
        params.cat_ids = [1]
        se = self.streaming(params=params)
        dt = [{"image_id": 2, "category_id": 2, "bbox": [0, 0, 5, 5], "score": 0.5}]
        se.update([{"id": 2, "width": 10, "height": 10}], [], dt)

    def test_pooled_categories_are_not_checked(self):
        params = hotcoco.Params("bbox")
        params.use_cats = False
        se = self.streaming(params=params)
        dt = [{"image_id": 2, "category_id": 7, "bbox": [0, 0, 5, 5], "score": 0.5}]
        se.update([{"id": 2, "width": 10, "height": 10}], [], dt)


def test_spent_streaming_eval_raises():
    se = StreamingEval(categories(), iou_type="bbox")
    se.update(images()[:1], gt_annotations()[:1], dt_annotations()[:1])
    se.finalize()
    with pytest.raises(RuntimeError, match="spent"):
        se.update(images()[:1], [], [])
    with pytest.raises(RuntimeError, match="spent"):
        se.finalize()


def test_nan_score_raises_like_load_res():
    se = StreamingEval(categories(), iou_type="bbox")
    with pytest.raises(RuntimeError, match="NaN"):
        se.update(images()[:1], [], [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": float("nan")}])


def test_non_dict_annotation_is_a_type_error():
    se = StreamingEval(categories(), iou_type="bbox")
    with pytest.raises(TypeError, match="dt_anns"):
        se.update(images()[:1], [], ["not a dict"])


def test_streaming_eval_is_exported_from_both_namespaces():
    assert hotcoco.StreamingEval is StreamingEval
    assert hotcoco.detection.StreamingEval is StreamingEval
