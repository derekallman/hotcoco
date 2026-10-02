"""`StreamingEval` reproduces the batch `COCO(dict)` + `COCOeval` pipeline.

Images go in one at a time, in any order; the finalized evaluator must give
the same stats and per-class results as a batch run over the same
annotations, in COCO and LVIS mode, and must raise once `finalize()` has
consumed it. No `data/` needed.
"""

import copy
import pickle

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


def stream_images(se, ids, dts=None):
    """Send the fixture images in ``ids`` to ``se`` in one batch."""
    gts, dts = gt_annotations(), dts if dts is not None else dt_annotations()
    se.update(
        [i for i in images() if i["id"] in ids],
        [a for a in gts if a["image_id"] in ids],
        [d for d in dts if d["image_id"] in ids],
    )
    return se


def finalized_stats(se):
    ev = se.finalize()
    ev.accumulate()
    ev.summarize()
    return ev.stats.tolist()


class TestMergeAndSerialize:
    """Shards merge to one stream; state survives bytes, pickle and deepcopy."""

    def test_merged_shards_equal_one_stream(self):
        whole = finalized_stats(stream_images(StreamingEval(categories()), {1, 2, 3}))
        a = stream_images(StreamingEval(categories()), {1})
        a.merge(stream_images(StreamingEval(categories()), {2, 3}))
        assert finalized_stats(a) == whole

    def test_merge_leaves_other_usable_and_other_wins_overlap(self):
        moved = [{**d, "bbox": [90, 90, 5, 5]} for d in dt_annotations()]
        a = stream_images(StreamingEval(categories()), {1, 2})
        b = stream_images(StreamingEval(categories()), {2, 3}, moved)
        a.merge(b)
        expected = stream_images(StreamingEval(categories()), {1})
        stream_images(expected, {2, 3}, moved)
        assert finalized_stats(a) == finalized_stats(expected)
        assert finalized_stats(b)  # not spent by being merged from

    def test_merge_mismatch_raises_and_names_the_field(self):
        a = stream_images(StreamingEval(categories()), {1})
        with pytest.raises(ValueError, match="categories"):
            a.merge(StreamingEval(categories()[:2]))
        with pytest.raises(ValueError, match="eval_mode"):
            a.merge(StreamingEval(categories(), lvis_style=True))
        params = hotcoco.Params()
        params.max_dets = [1, 10]
        with pytest.raises(ValueError, match="max_dets"):
            a.merge(StreamingEval(categories(), params=params))
        # The failed merges changed nothing.
        assert finalized_stats(a) == finalized_stats(stream_images(StreamingEval(categories()), {1}))

    def test_merge_with_spent_raises(self):
        a, b = StreamingEval(categories()), StreamingEval(categories())
        b.finalize()
        with pytest.raises(RuntimeError, match="spent"):
            a.merge(b)

    def test_merge_into_itself_is_an_error(self):
        a = StreamingEval(categories())
        with pytest.raises(ValueError, match="itself"):
            a.merge(a)

    def test_bytes_round_trip(self):
        se = stream_images(StreamingEval(categories()), {1, 2, 3})
        restored = StreamingEval.from_bytes(se.to_bytes())
        assert finalized_stats(restored) == finalized_stats(se)

    @pytest.mark.parametrize("clone", [lambda se: pickle.loads(pickle.dumps(se)), copy.deepcopy, copy.copy])
    def test_pickle_and_copy_round_trip(self, clone):
        se = stream_images(StreamingEval(categories()), {1, 2})
        twin = clone(se)
        stream_images(twin, {3})
        stream_images(se, {3})
        assert finalized_stats(twin) == finalized_stats(se)

    def test_copy_is_independent(self):
        se = stream_images(StreamingEval(categories()), {1})
        twin = copy.deepcopy(se)
        stream_images(twin, {2, 3})
        assert finalized_stats(se) == finalized_stats(stream_images(StreamingEval(categories()), {1}))

    def test_restored_lvis_state_keeps_its_mode(self):
        se = StreamingEval(categories(), lvis_style=True)
        with pytest.raises(ValueError, match="eval_mode"):
            copy.deepcopy(se).merge(StreamingEval(categories()))

    @pytest.mark.parametrize("bad", [b"", b"nope", b"HCSE", b"HCSE\x09\x00\x00\x00" + bytes(8)])
    def test_bad_bytes_raise_value_error(self, bad):
        with pytest.raises(ValueError, match="StreamingEval state"):
            StreamingEval.from_bytes(bad)

    def test_truncated_bytes_raise_value_error(self):
        data = stream_images(StreamingEval(categories()), {1, 2, 3}).to_bytes()
        for n in range(len(data)):
            with pytest.raises(ValueError):
                StreamingEval.from_bytes(data[:n])

    def test_spent_eval_cannot_be_serialized(self):
        se = StreamingEval(categories())
        se.finalize()
        with pytest.raises(RuntimeError, match="spent"):
            se.to_bytes()
        with pytest.raises(RuntimeError, match="spent"):
            pickle.dumps(se)


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
