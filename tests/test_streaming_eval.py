"""`StreamingEval` reproduces the batch `COCO(dict)` + `COCOeval` pipeline.

Images go in one at a time, in any order; the finalized evaluator must give
the same stats and per-class results as a batch run over the same
annotations, in COCO and LVIS mode, and must raise once `finalize()` has
consumed it. No `data/` needed.
"""

import copy
import pickle

import hotcoco
import numpy as np
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


def as_array(dts):
    """The ``(N, 7)`` array ``load_res`` accepts for a list of raw detections."""
    return np.array([[d["image_id"], *d["bbox"], d["score"], d["category_id"]] for d in dts], dtype=np.float64)


def stats_after(update_dt, cats=None, **kwargs):
    """Stats of a StreamingEval fed every fixture image in two batches."""
    se = StreamingEval(cats or categories(), **kwargs)
    for ids in ({1, 2}, {3}):
        im = [i for i in images() if i["id"] in ids]
        gt = [a for a in gt_annotations() if a["image_id"] in ids]
        se.update(im, gt, update_dt(ids))
    ev = se.finalize()
    ev.accumulate()
    ev.summarize()
    return ev.stats.tolist()


class TestArrayDetections:
    """``update`` takes the ``(N, 7)`` array ``load_res`` takes, without a dict per detection."""

    def test_array_equals_dict_form(self):
        dts = dt_annotations()
        by_dict = stats_after(lambda ids: [d for d in dts if d["image_id"] in ids])
        by_array = stats_after(lambda ids: as_array([d for d in dts if d["image_id"] in ids]))
        assert by_array == by_dict
        assert by_array[0] > 0

    def test_empty_array_is_an_image_with_no_detections(self):
        stats = stats_after(lambda ids: np.zeros((0, 7)))
        assert stats[0] == 0.0

    def test_six_columns_put_everything_in_category_one_like_load_res(self):
        dts = [d for d in dt_annotations() if d["category_id"] == 1]
        six = stats_after(lambda ids: as_array([d for d in dts if d["image_id"] in ids])[:, :6])
        seven = stats_after(lambda ids: as_array([d for d in dts if d["image_id"] in ids]))
        assert six == seven

    def test_wrong_column_count_raises(self):
        se = StreamingEval(categories())
        with pytest.raises(ValueError, match="6 or 7 columns"):
            se.update(images()[:1], [], np.zeros((2, 5)))

    def test_unlisted_type_raises_type_error(self):
        se = StreamingEval(categories())
        with pytest.raises(TypeError, match="list of dicts or a numpy"):
            se.update(images()[:1], [], "detections")
        with pytest.raises(TypeError, match="list of dicts or a numpy"):
            se.update(images()[:1], [], np.full((1, 7), "1"))

    def test_float32_array_equals_float64(self):
        """Detectors emit float32; widening it is exact, so nothing has to convert first."""
        dts = dt_annotations()
        by_f64 = stats_after(lambda ids: as_array([d for d in dts if d["image_id"] in ids]))
        by_f32 = stats_after(lambda ids: as_array([d for d in dts if d["image_id"] in ids]).astype(np.float32))
        assert by_f32 == by_f64

    def test_segmentation_with_a_dict_list_raises(self):
        se = StreamingEval(categories())
        with pytest.raises(TypeError, match="segmentation goes with"):
            se.update(images()[:1], [], [], segmentation=[])

    def test_segmentation_length_must_match_rows(self):
        se = StreamingEval(categories(), iou_type="segm")
        with pytest.raises(ValueError, match="segmentation has 1 entries for 2 rows"):
            se.update(images()[:1], [], np.zeros((2, 7)), segmentation=[{"size": [100, 100], "counts": "0"}])

    def test_segm_array_with_rle_list_equals_dict_form(self):
        def rle(box):
            m = np.zeros((100, 100), dtype=np.uint8, order="F")
            x, y, w, h = (int(v) for v in box)
            m[y : y + h, x : x + w] = 1
            r = hotcoco.mask.encode(m)
            return {"size": r["size"], "counts": r["counts"]}

        gts = [{**g, "segmentation": rle(g["bbox"])} for g in gt_annotations()]
        dts = dt_annotations()
        # Masks shifted off the boxes: a run that ignored `segmentation` and
        # derived masks from the boxes would score differently.
        for d in dts:
            x, y, w, h = d["bbox"]
            d["segmentation"] = rle([x + 12, y + 12, w, h])

        def run(update_dt):
            se = StreamingEval(categories(), iou_type="segm")
            for ids in ({1, 2}, {3}):
                se.update(
                    [i for i in images() if i["id"] in ids],
                    [a for a in gts if a["image_id"] in ids],
                    update_dt(ids),
                    **(
                        {}
                        if isinstance(update_dt(ids), list)
                        else {"segmentation": [d["segmentation"] for d in dts if d["image_id"] in ids]}
                    ),
                )
            ev = se.finalize()
            ev.accumulate()
            ev.summarize()
            return ev.stats.tolist()

        def run_without_masks():
            return run(
                lambda ids: [{k: v for k, v in d.items() if k != "segmentation"} for d in dts if d["image_id"] in ids]
            )

        by_dict = run(lambda ids: [d for d in dts if d["image_id"] in ids])
        by_array = run(lambda ids: as_array([d for d in dts if d["image_id"] in ids]))
        assert by_array == by_dict
        no_masks = run_without_masks()
        assert by_array != no_masks

    def test_array_detection_in_an_unlisted_category_raises_key_error(self):
        se = StreamingEval(categories())
        with pytest.raises(KeyError, match=r"\[7\]"):
            se.update(images()[:1], [], np.array([[1, 0, 0, 5, 5, 0.5, 7]], dtype=np.float64))


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


def via_pickle(se):
    return pickle.loads(pickle.dumps(se))


def via_bytes(se):
    return StreamingEval.from_bytes(se.to_bytes())


class TestMergeAndSerialize:
    """Shards merge to one stream; state survives bytes, pickle and deepcopy."""

    def test_merged_shards_equal_one_stream(self):
        whole = finalized_stats(stream_images(StreamingEval(categories()), {1, 2, 3}))
        shards = [stream_images(StreamingEval(categories()), ids) for ids in ({1}, {2}, {3})]
        assert finalized_stats(StreamingEval.merge(shards)) == whole

    def test_merge_leaves_its_inputs_usable_and_the_last_wins_an_overlap(self):
        moved = [{**d, "bbox": [90, 90, 5, 5]} for d in dt_annotations()]
        a = stream_images(StreamingEval(categories()), {1, 2})
        b = stream_images(StreamingEval(categories()), {2, 3}, moved)
        merged = StreamingEval.merge([a, b])
        expected = stream_images(StreamingEval(categories()), {1})
        stream_images(expected, {2, 3}, moved)
        assert finalized_stats(merged) == finalized_stats(expected)
        assert finalized_stats(a) == finalized_stats(stream_images(StreamingEval(categories()), {1, 2}))
        assert finalized_stats(b) == finalized_stats(stream_images(StreamingEval(categories()), {2, 3}, moved))

    def test_merge_returns_an_independent_evaluator(self):
        a = stream_images(StreamingEval(categories()), {1})
        merged = StreamingEval.merge([a])
        stream_images(merged, {2, 3})
        assert finalized_stats(a) == finalized_stats(stream_images(StreamingEval(categories()), {1}))
        assert finalized_stats(merged) == finalized_stats(stream_images(StreamingEval(categories()), {1, 2, 3}))

    def test_the_same_evaluator_twice_counts_its_images_once(self):
        a = stream_images(StreamingEval(categories()), {1, 2})
        assert finalized_stats(StreamingEval.merge([a, a])) == finalized_stats(a)

    def test_merge_mismatch_names_the_evaluator_and_the_field(self):
        a = stream_images(StreamingEval(categories()), {1})
        with pytest.raises(ValueError, match=r"evaluators\[1\].*categories"):
            StreamingEval.merge([a, StreamingEval(categories()[:2])])
        with pytest.raises(ValueError, match=r"evaluators\[2\].*eval_mode"):
            StreamingEval.merge([a, StreamingEval(categories()), StreamingEval(categories(), lvis_style=True)])
        params = hotcoco.Params()
        params.max_dets = [1, 10]
        with pytest.raises(ValueError, match="max_dets"):
            StreamingEval.merge([a, StreamingEval(categories(), params=params)])

    def test_category_order_does_not_block_a_merge(self):
        # The K axis is sorted by id, so list order changes no number.
        a = stream_images(StreamingEval(categories()), {1})
        b = stream_images(StreamingEval(categories()[::-1]), {2, 3})
        expected = stream_images(StreamingEval(categories()), {1, 2, 3})
        assert finalized_stats(StreamingEval.merge([a, b])) == finalized_stats(expected)

    @pytest.mark.parametrize("spent_at", [0, 1])
    def test_merge_with_spent_raises(self, spent_at):
        evaluators = [StreamingEval(categories()), StreamingEval(categories())]
        evaluators[spent_at].finalize()
        with pytest.raises(RuntimeError, match="spent"):
            StreamingEval.merge(evaluators)

    def test_merge_of_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="at least one"):
            StreamingEval.merge([])
        with pytest.raises(ValueError, match="at least one"):
            StreamingEval.merge(iter(()))

    def test_merge_reads_a_generator_of_restored_states(self):
        states = [stream_images(StreamingEval(categories()), {i}).to_bytes() for i in (1, 2, 3)]
        merged = StreamingEval.merge(StreamingEval.from_bytes(s) for s in states)
        assert finalized_stats(merged) == finalized_stats(stream_images(StreamingEval(categories()), {1, 2, 3}))

    def test_merge_names_what_it_takes(self):
        a = stream_images(StreamingEval(categories()), {1})
        # The old in-place spelling, `a.merge(b)`.
        with pytest.raises(TypeError, match=r"StreamingEval.merge\(\[a, b\]\)"):
            a.merge(StreamingEval(categories()))
        with pytest.raises(TypeError, match=r"evaluators\[1\] is not a StreamingEval"):
            StreamingEval.merge([a, "b"])

    def test_bytes_round_trip(self):
        se = stream_images(StreamingEval(categories()), {1, 2, 3})
        restored = via_bytes(se)
        assert finalized_stats(restored) == finalized_stats(se)

    @pytest.mark.parametrize("clone", [via_pickle, copy.deepcopy, copy.copy])
    def test_pickle_and_copy_round_trip(self, clone):
        se = stream_images(StreamingEval(categories()), {1, 2})
        twin = clone(se)
        stream_images(twin, {3})
        stream_images(se, {3})
        assert finalized_stats(twin) == finalized_stats(se)

    def test_class_is_pickled_by_its_public_path(self):
        assert StreamingEval.__module__ == "hotcoco"
        assert b"hotcoco.hotcoco" not in pickle.dumps(StreamingEval(categories()))

    def test_copy_is_independent(self):
        se = stream_images(StreamingEval(categories()), {1})
        twin = copy.deepcopy(se)
        stream_images(twin, {2, 3})
        assert finalized_stats(se) == finalized_stats(stream_images(StreamingEval(categories()), {1}))

    @pytest.mark.parametrize("restore", [via_bytes, via_pickle])
    def test_restored_evaluator_still_rejects_unknown_categories(self, restore):
        se = restore(stream_images(StreamingEval(categories()), {1}))
        dt = [{"image_id": 2, "category_id": 7, "bbox": [0, 0, 5, 5], "score": 0.5}]
        with pytest.raises(KeyError, match=r"\[7\]"):
            se.update(images()[1:2], [], dt)

    @pytest.mark.parametrize("restore", [via_bytes, via_pickle])
    def test_restored_evaluator_accepts_array_detections(self, restore):
        se = restore(stream_images(StreamingEval(categories()), {1}))
        rest = [d for d in dt_annotations() if d["image_id"] in {2, 3}]
        se.update(
            [i for i in images() if i["id"] in {2, 3}],
            [a for a in gt_annotations() if a["image_id"] in {2, 3}],
            as_array(rest),
        )
        assert finalized_stats(se) == finalized_stats(stream_images(StreamingEval(categories()), {1, 2, 3}))

    def test_restored_lvis_state_keeps_its_mode_and_numbers(self):
        cats = [{"id": 1, "name": "a", "frequency": "r"}, {"id": 2, "name": "b", "frequency": "f"}]
        imgs = [
            {"id": 1, "width": 100, "height": 100, "neg_category_ids": [2]},
            {"id": 2, "width": 100, "height": 100, "not_exhaustive_category_ids": [1]},
        ]
        gts = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
        ]
        dts = [
            {"image_id": 1, "category_id": 1, "bbox": [12, 12, 50, 50], "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": [200, 200, 50, 50], "score": 0.8},
            {"image_id": 2, "category_id": 1, "bbox": [300, 300, 50, 50], "score": 0.7},
        ]
        se = StreamingEval(cats, lvis_style=True)
        se.update(imgs, gts, dts)
        twin = copy.deepcopy(se)
        with pytest.raises(ValueError, match="eval_mode"):
            StreamingEval.merge([se, StreamingEval(cats)])
        # A different LVIS frequency would move APr/APc/APf.
        with pytest.raises(ValueError, match="categories"):
            StreamingEval.merge([se, StreamingEval([{**c, "frequency": "c"} for c in cats], lvis_style=True)])
        assert finalized_stats(twin) == finalized_stats(se)

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
    with pytest.raises(ValueError, match="NaN"):
        se.update(images()[:1], [], [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": float("nan")}])


def test_fractional_array_id_raises_like_load_res():
    """A float label tensor reaches ``update()`` as a fractional ``category_id``."""
    se = StreamingEval(categories(), iou_type="bbox")
    with pytest.raises(ValueError, match="ids must be finite, non-negative integers"):
        se.update(images()[:1], [], np.array([[1.0, 0, 0, 10, 10, 0.9, 1.5]]))


def test_non_dict_annotation_is_a_type_error():
    se = StreamingEval(categories(), iou_type="bbox")
    with pytest.raises(TypeError, match="dt_anns"):
        se.update(images()[:1], [], ["not a dict"])


def test_streaming_eval_is_exported_from_both_namespaces():
    assert hotcoco.StreamingEval is StreamingEval
    assert hotcoco.detection.StreamingEval is StreamingEval
