"""Python-side pins for the 2026-10 whole-project review fixes.

The Rust integration tests in ``crates/hotcoco/tests/review_fixes.rs`` prove
the mechanics; these check what a Python user sees — the exception, the
number — and compare against pycocotools where it has an answer. No ``data/``
needed.
"""

import numpy as np
import pycocotools.mask as pm
import pytest
from hotcoco import COCO, COCOeval, LVISResults, StreamingEval, mask


def square(x, y, s):
    return [x, y, x + s, y, x + s, y + s, x, y + s]


def gt_dataset(image):
    return {
        "images": [image],
        "categories": [{"id": 1, "name": "thing"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "area": 2500,
                "iscrowd": 0,
                "segmentation": [square(10, 10, 50)],
            }
        ],
    }


class TestSegmNeedsImageDims:
    """A polygon on an image without ``height``/``width`` used to rasterize
    onto a 0×0 canvas and report segm AP 0.000 with no warning."""

    def test_evaluate_raises_value_error_naming_the_image(self):
        gt = COCO(gt_dataset({"id": 1}))
        dt = gt.load_res([{"image_id": 1, "category_id": 1, "score": 0.9, "segmentation": [square(10, 10, 50)]}])
        ev = COCOeval(gt, dt, "segm")
        with pytest.raises(ValueError, match=r"height.*ids 1"):
            ev.evaluate()
        with pytest.raises(ValueError, match=r"height.*ids 1"):
            ev.run()
        # Checked at evaluate(), so switching iouType after construction cannot
        # slip past it.
        ev = COCOeval(gt, dt, "bbox")
        ev.params.iouType = "segm"
        with pytest.raises(ValueError, match=r"height.*ids 1"):
            ev.evaluate()
        # Box evaluation never reads the fields.
        dt = gt.load_res([{"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [10, 10, 50, 50]}])
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        assert ev.stats[0] == pytest.approx(1.0)

    def test_with_dims_the_same_data_is_a_perfect_match(self):
        gt = COCO(gt_dataset({"id": 1, "height": 100, "width": 100}))
        dt = gt.load_res([{"image_id": 1, "category_id": 1, "score": 0.9, "segmentation": [square(10, 10, 50)]}])
        ev = COCOeval(gt, dt, "segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        assert ev.stats[0] == pytest.approx(1.0)

    def test_streaming_update_raises(self):
        se = StreamingEval([{"id": 1, "name": "thing"}], iou_type="segm")
        ds = gt_dataset({"id": 1})
        with pytest.raises(RuntimeError, match="height"):
            se.update(
                ds["images"],
                ds["annotations"],
                [{"image_id": 1, "category_id": 1, "score": 0.9, "segmentation": [square(10, 10, 50)]}],
            )


class TestLVISResultsCapsPerImage:
    """``LVISResults`` accepted ``max_dets`` and ignored it; lvis-api keeps
    each image's top ``max_dets`` by score, across categories."""

    def _gt_and_dets(self):
        gt = COCO(
            {
                "images": [{"id": 1, "height": 100, "width": 100}, {"id": 2, "height": 100, "width": 100}],
                "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
                "annotations": [],
            }
        )
        box = [1, 1, 5, 5]
        dets = [
            {"image_id": 1, "category_id": 1, "bbox": box, "score": 0.2},
            {"image_id": 1, "category_id": 2, "bbox": box, "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": box, "score": 0.5},
            {"image_id": 1, "category_id": 2, "bbox": box, "score": 0.5},
            {"image_id": 2, "category_id": 1, "bbox": box, "score": 0.1},
        ]
        return gt, dets

    def test_keeps_top_max_dets_per_image_ties_in_file_order(self):
        gt, dets = self._gt_and_dets()
        res = LVISResults(gt, dets, max_dets=2)
        kept = sorted((a["image_id"], a["category_id"], a["score"]) for a in res.dataset["annotations"])
        # Image 1: 0.9, then the first of the two 0.5s (category 1). Image 2 is under the cap.
        assert kept == [(1, 1, 0.5), (1, 2, 0.9), (2, 1, 0.1)]
        direct = gt.load_res(dets).cap_detections_per_image(2)
        assert len(direct.dataset["annotations"]) == 3

    def test_negative_max_dets_keeps_everything(self):
        gt, dets = self._gt_and_dets()
        assert len(LVISResults(gt, dets, max_dets=-1).dataset["annotations"]) == 5
        assert len(gt.load_res(dets).cap_detections_per_image(None).dataset["annotations"]) == 5

    def test_none_and_integral_float_max_dets(self):
        """1.1 ignored ``max_dets``, so ``None`` and ``300.0`` worked; the cap
        made both raise. ``None`` keeps everything, like ``-1``; an integral
        float is that integer; a fractional one is an error."""
        import numpy as np

        gt, dets = self._gt_and_dets()

        def kept(max_dets):
            return len(LVISResults(gt, dets, max_dets=max_dets).dataset["annotations"])

        assert kept(None) == 5
        assert kept(2.0) == kept(np.float64(2.0)) == kept(np.int64(2)) == kept(2) == 3
        assert kept(-1.0) == 5
        with pytest.raises(ValueError, match="max_dets"):
            LVISResults(gt, dets, max_dets=2.5)


class TestStreamingGroundTruthFixups:
    def test_gt_without_ids_or_area_matches_batch(self):
        """torchvision-style targets: no ``id`` (all collided on 0 and resolved
        to the batch's last annotation) and no ``area`` (read as 0, so every
        object was ``small``)."""
        cats = [{"id": 1, "name": "a"}, {"id": 3, "name": "b"}]
        imgs = [{"id": 1, "width": 200, "height": 200}, {"id": 2, "width": 200, "height": 200}]
        # A 60×60 box is medium; a 10×10 box is small.
        gts_bare = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 60, 60], "iscrowd": 0},
            {"image_id": 2, "category_id": 3, "bbox": [100, 100, 10, 10], "iscrowd": 0},
        ]
        dts = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 60, 60], "score": 0.9},
            {"image_id": 2, "category_id": 3, "bbox": [100, 100, 10, 10], "score": 0.8},
        ]
        se = StreamingEval(cats, iou_type="bbox")
        se.update(imgs, gts_bare, dts)
        streamed = se.finalize()
        streamed.accumulate()
        streamed.summarize()

        gts_full = [dict(g, id=i + 1, area=g["bbox"][2] * g["bbox"][3]) for i, g in enumerate(gts_bare)]
        gt = COCO({"images": imgs, "categories": cats, "annotations": gts_full})
        batch = COCOeval(gt, gt.load_res(dts), "bbox")
        batch.evaluate()
        batch.accumulate()
        batch.summarize()

        assert streamed.stats.tolist() == batch.stats.tolist()
        assert batch.stats[0] == pytest.approx(1.0)
        assert batch.stats[3] == pytest.approx(1.0)  # APs: the 10×10 box
        assert batch.stats[4] == pytest.approx(1.0)  # APm: the 60×60 box
        assert batch.stats[5] == -1.0  # APl: no large ground truth


class TestMaskAgainstPycocotools:
    def test_short_rle_tail_is_background(self):
        """Counts summing to less than h·w: the omitted tail is background, as
        ``decode`` renders it. The walkers used to leak the last run's value
        over it and report IoU 5.0."""
        short = {"size": [10, 1], "counts": [0, 2]}
        full = {"size": [10, 1], "counts": [0, 10]}
        assert np.asarray(mask.iou([short], [full], [False])).tolist() == [[0.2]]
        union = mask.merge([short, full], intersect=False)
        assert mask.decode(union).tolist() == mask.decode(full).tolist()

    # Every reader of an RLE dict, each called on a 2×2 one.
    RLE_READERS = {
        "area": lambda r: mask.area(r),
        "area_list": lambda r: mask.area([r]),
        "decode": lambda r: mask.decode(r),
        "decode_list": lambda r: mask.decode([r]),
        "toBbox": lambda r: mask.toBbox(r),
        "iou": lambda r: mask.iou([r], [{"size": [2, 2], "counts": [0, 4]}], [False]),
        "merge": lambda r: mask.merge([r, {"size": [2, 2], "counts": [4]}]),
        "frPyObjects": lambda r: mask.frPyObjects(r, 2, 2),
    }

    @pytest.mark.parametrize("reader", list(RLE_READERS))
    @pytest.mark.parametrize(
        "counts, total",
        [([0, 100], 100), ([2**32 - 1, 2], 2**32 + 1)],  # the second wraps to 1 in u32
        ids=["overrun", "u32_wrap"],
    )
    def test_overrun_run_list_raises_as_its_string_does(self, reader, counts, total):
        """Runs summing past h·w are rejected, with the message the same runs
        give as a compressed string. They used to pass unchecked: ``area``
        returned 100 for a 4-pixel mask. pycocotools has no one answer to
        match — its ``frPyObjects`` takes these runs, ``area`` then returns
        100 and ``toBbox`` a box 50 pixels wide, ``decode`` raises, and
        ``iou`` and ``merge`` never return."""
        read = self.RLE_READERS[reader]
        runs = {"size": [2, 2], "counts": counts}
        string = {"size": [2, 2], "counts": pm.frPyObjects(runs, 2, 2)["counts"]}
        message = f"^invalid RLE: total counts {total} exceed h\\*w=4$"
        with pytest.raises(ValueError, match=message):
            read(runs)
        with pytest.raises(ValueError, match=message):
            read(string)

    @pytest.mark.parametrize("reader", list(RLE_READERS))
    @pytest.mark.parametrize("counts", [[0, 2], [1, 2], [0, 4], [4]], ids=["short", "short_odd", "exact", "exact_bg"])
    def test_run_list_within_bounds_reads_as_its_string_does(self, reader, counts):
        """Runs that stop short of h·w, or fill it, are still read, exactly as
        the same runs given as a compressed string are."""
        read = self.RLE_READERS[reader]
        runs = {"size": [2, 2], "counts": counts}
        string = {"size": [2, 2], "counts": pm.frPyObjects(runs, 2, 2)["counts"]}
        got, want = read(runs), read(string)
        assert type(got) is type(want)
        assert np.asarray(got).tolist() == np.asarray(want).tolist()

    def test_short_run_list_reads_as_pycocotools_does(self):
        runs = {"size": [2, 2], "counts": [1, 2]}
        ref = pm.frPyObjects(runs, 2, 2)
        assert mask.decode(runs).tolist() == pm.decode(ref).tolist() == [[0, 1], [1, 0]]
        assert mask.area(runs) == pm.area(ref) == 2
        assert mask.toBbox(runs).tolist() == pm.toBbox(ref).tolist()

    def test_mismatched_dims_is_minus_one(self):
        a = dict(pm.encode(np.asfortranarray(np.ones((4, 6), dtype=np.uint8))))
        b = dict(pm.encode(np.asfortranarray(np.ones((6, 4), dtype=np.uint8))))
        assert np.asarray(pm.iou([a], [b], [False])).tolist() == [[-1.0]]
        assert np.asarray(mask.iou([a], [b], [False])).tolist() == [[-1.0]]

    @pytest.mark.parametrize("box", [[0.5, 0.5, 2, 2], [1.2, 1.7, 3.3, 2.1], [3, 3, 4, 4], [-1.5, 2.5, 4, 3]])
    def test_fractional_box_rasterizes_like_pycocotools(self, box):
        boxes = np.array([box], dtype=np.float64)
        ref = pm.frPyObjects(boxes, 10, 10)[0]
        got = mask.frPyObjects(boxes, 10, 10)[0]
        assert mask.decode(got).tolist() == pm.decode(ref).tolist()
        assert mask.area(got) == pm.area(ref)
        # pycocotools' frPyObjects cannot take a list of boxes (its Cython
        # frBbox wants an array); hotcoco can, and both spellings must agree.
        assert mask.frPyObjects([box], 10, 10)[0]["counts"] == got["counts"]
