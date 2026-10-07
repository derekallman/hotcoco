"""Column-form inputs: arrays in, no Python dict per annotation.

``COCO.from_arrays`` must equal ``COCO(dict)`` over the same annotations, and
``update_anns(ids=, area=)`` must equal the dict edit it replaces. No ``data/``
needed.
"""

import inspect
import json

import hotcoco
import numpy as np
import pytest
from hotcoco import COCO, COCOeval

IMAGES = [{"id": i, "width": 100, "height": 100, "file_name": f"{i}.jpg"} for i in (1, 2, 3)]
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}]
IMAGE_IDS = np.array([1, 2, 2, 3], dtype=np.int64)
CATEGORY_IDS = np.array([1, 2, 1, 1], dtype=np.int64)
BOXES = np.array([[10, 10, 30, 30], [50, 50, 20, 20], [0, 0, 40, 40], [5, 5, 10, 10]], dtype=np.float64)


def dict_dataset(ids=None, area=None, iscrowd=None, segmentation=None):
    n = len(IMAGE_IDS)
    anns = []
    for i in range(n):
        b = [float(v) for v in BOXES[i]]
        ann = {
            "id": int(ids[i]) if ids is not None else i + 1,
            "image_id": int(IMAGE_IDS[i]),
            "category_id": int(CATEGORY_IDS[i]),
            "bbox": b,
            "area": float(area[i]) if area is not None else b[2] * b[3],
            "iscrowd": int(iscrowd[i]) if iscrowd is not None else 0,
        }
        if segmentation is not None:
            ann["segmentation"] = segmentation[i]
        anns.append(ann)
    return {"images": IMAGES, "annotations": anns, "categories": CATEGORIES}


def evaluated_stats(gt):
    dts = [
        {"image_id": 1, "category_id": 1, "bbox": [11, 11, 30, 30], "score": 0.9},
        {"image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.8},
        {"image_id": 2, "category_id": 1, "bbox": [1, 1, 40, 40], "score": 0.7},
    ]
    ev = COCOeval(gt, gt.load_res(dts), "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats.tolist()


class TestFromArrays:
    def test_equals_dict_form(self):
        by_arrays = COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES)
        by_dict = COCO(dict_dataset())
        assert by_arrays.dataset == by_dict.dataset
        assert evaluated_stats(by_arrays) == evaluated_stats(by_dict)
        assert evaluated_stats(by_arrays)[0] > 0

    def test_optional_columns(self):
        ids = np.array([11, 12, 13, 14])
        area = np.array([900.0, 400.0, 1600.0, 100.0])
        crowd = np.array([0, 0, 1, 0])
        by_arrays = COCO.from_arrays(
            IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, ids=ids, area=area * 2, iscrowd=crowd.astype(bool)
        )
        by_dict = COCO(dict_dataset(ids=ids, area=area * 2, iscrowd=crowd))
        assert by_arrays.dataset == by_dict.dataset
        assert by_arrays.dataset["annotations"][2]["iscrowd"] == 1

    def test_default_ids_and_area(self):
        anns = COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES).dataset["annotations"]
        assert [a["id"] for a in anns] == [1, 2, 3, 4]
        assert [a["area"] for a in anns] == [900.0, 400.0, 1600.0, 100.0]

    @pytest.mark.parametrize(
        "make",
        [
            lambda a: a.astype(np.int32),
            lambda a: a.astype(np.int16),
            lambda a: a.astype(np.uint8),
            lambda a: a.astype(np.uint64),
            list,
            lambda a: a.tolist(),
        ],
    )
    def test_other_integer_spellings(self, make):
        gt = COCO.from_arrays(IMAGES, CATEGORIES, make(IMAGE_IDS), make(CATEGORY_IDS), BOXES.tolist())
        assert gt.dataset == COCO(dict_dataset()).dataset

    @pytest.mark.parametrize("dtype", [np.float32, np.int64, np.int32])
    def test_other_box_dtypes(self, dtype):
        gt = COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES.astype(dtype))
        assert gt.dataset == COCO(dict_dataset()).dataset

    def test_rles_equal_dict_form(self):
        def rle(box):
            m = np.zeros((100, 100), dtype=np.uint8, order="F")
            x, y, w, h = (int(v) for v in box)
            m[y : y + h, x : x + w] = 1
            r = hotcoco.mask.encode(m)
            return {"size": r["size"], "counts": r["counts"]}

        rles = [rle(b) for b in BOXES]
        area = np.array([900.0, 400.0, 1600.0, 100.0])
        by_arrays = COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, area=area, segmentation=rles)
        assert by_arrays.dataset == COCO(dict_dataset(segmentation=rles)).dataset

    def test_polygons_equal_dict_form(self):
        polygons = [[[x, y, x + w, y, x + w, y + h, x, y + h]] for x, y, w, h in BOXES.tolist()]
        area = BOXES[:, 2] * BOXES[:, 3]
        by_arrays = COCO.from_arrays(
            IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, area=area, segmentation=polygons
        )
        assert by_arrays.dataset == COCO(dict_dataset(area=area, segmentation=polygons)).dataset

    def test_segmentation_without_area_raises(self):
        with pytest.raises(ValueError, match="area is required with segmentation"):
            COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, segmentation=[{}] * 4)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"category_ids": CATEGORY_IDS[:3]}, "category_ids has 3 entries but image_ids has 4"),
            ({"boxes": BOXES[:, :3]}, r"boxes must have shape \(4, 4\)"),
            ({"boxes": BOXES[:3]}, r"boxes must have shape \(4, 4\)"),
            ({"image_ids": np.array([1, 2, 2, -3])}, "negative"),
        ],
    )
    def test_bad_columns_raise_value_error(self, kwargs, match):
        args = {"image_ids": IMAGE_IDS, "category_ids": CATEGORY_IDS, "boxes": BOXES} | kwargs
        with pytest.raises(ValueError, match=match):
            COCO.from_arrays(IMAGES, CATEGORIES, **args)

    def test_optional_column_length_is_checked(self):
        with pytest.raises(ValueError, match="area has 2 entries"):
            COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, area=[1.0, 2.0])
        with pytest.raises(ValueError, match="ids has 1 entries"):
            COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, ids=[1])

    def test_non_numeric_column_raises_type_error(self):
        with pytest.raises(TypeError, match="image_ids must be"):
            COCO.from_arrays(IMAGES, CATEGORIES, "abc", CATEGORY_IDS, BOXES)

    @pytest.mark.parametrize("boxes", [np.zeros((0, 4)), []])
    def test_empty_dataset(self, boxes):
        gt = COCO.from_arrays(IMAGES, CATEGORIES, [], [], boxes)
        assert gt.dataset["annotations"] == []

    def test_unknown_keyword_raises_type_error(self):
        assert list(inspect.signature(COCO.from_arrays).parameters) == [
            "images",
            "categories",
            "image_ids",
            "category_ids",
            "boxes",
            "ids",
            "area",
            "iscrowd",
            "segmentation",
        ]
        with pytest.raises(TypeError, match="unexpected keyword argument 'crowd'"):
            COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, crowd=[0, 0, 0, 0])

    def test_option_columns_are_keyword_only(self):
        with pytest.raises(TypeError):
            COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, [1, 2, 3, 4])

    def test_none_options_mean_the_default(self):
        explicit = COCO.from_arrays(
            IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, ids=None, area=None, iscrowd=None, segmentation=None
        )
        assert explicit.dataset == COCO(dict_dataset()).dataset

    def test_segmentation_must_be_a_list(self):
        with pytest.raises(TypeError, match="segmentation must be a list"):
            COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, area=np.ones(4), segmentation="abcd")


class TestUpdateAnnsColumns:
    def fresh(self):
        return COCO(dict_dataset())

    def test_columns_equal_dict_edit(self):
        by_columns, by_dict = self.fresh(), self.fresh()
        ids, area = np.array([2, 4]), np.array([111.0, 222.0])
        by_columns.update_anns(ids=ids, area=area)
        by_dict.update_anns([{"id": 2, "area": 111.0}, {"id": 4, "area": 222.0}])
        assert by_columns.dataset == by_dict.dataset
        assert [a["area"] for a in by_columns.dataset["annotations"]] == [900.0, 111.0, 1600.0, 222.0]

    def test_unknown_id_raises_and_writes_nothing(self):
        gt = self.fresh()
        before = gt.dataset
        with pytest.raises(KeyError, match="99"):
            gt.update_anns(ids=np.array([1, 99]), area=np.array([5.0, 6.0]))
        assert gt.dataset == before

    def test_shared_dataset_is_copied_on_write(self):
        gt = self.fresh()
        twin = gt.load_res([{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0}])
        before = twin.dataset
        gt.update_anns(ids=[1], area=[1.0])
        assert twin.dataset == before

    def test_empty_columns_are_a_no_op(self):
        gt = self.fresh()
        before = gt.dataset
        gt.update_anns(ids=np.array([], dtype=np.int64), area=np.array([]))
        assert gt.dataset == before

    def test_forms_are_exclusive(self):
        gt = self.fresh()
        with pytest.raises(TypeError, match="either a list of dicts or ids="):
            gt.update_anns([{"id": 1, "area": 1.0}], ids=[1], area=[1.0])
        with pytest.raises(TypeError, match="pass a list of dicts, or ids="):
            gt.update_anns()
        with pytest.raises(TypeError, match="ids= needs a column"):
            gt.update_anns(ids=[1])
        with pytest.raises(TypeError, match="area= goes with ids="):
            gt.update_anns([{"id": 1}], area=[1.0])
        with pytest.raises(TypeError, match="create= goes with a list of dicts"):
            gt.update_anns(ids=[1], area=[1.0], create=True)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="area has 1 entries but ids has 2"):
            self.fresh().update_anns(ids=[1, 2], area=[1.0])

    def test_dict_form_still_positional(self):
        gt = self.fresh()
        gt.update_anns([{"id": 1, "area": 5.0}])
        assert gt.dataset["annotations"][0]["area"] == 5.0


class TestLoadResArrayIds:
    """``load_res`` and ``StreamingEval.update`` share the array parser, so this covers both."""

    @pytest.mark.parametrize("dtype", [np.float32, np.int64])
    def test_other_dtypes_read_as_float64(self, dtype):
        arr = np.array([[1, 10, 10, 30, 30, 1, 1], [2, 50, 50, 20, 20, 0, 2]])
        gt = COCO(dict_dataset())
        assert gt.load_res(arr.astype(dtype)).dataset == gt.load_res(arr.astype(np.float64)).dataset

    @pytest.mark.parametrize(("col", "value"), [(0, np.nan), (0, -1.0), (6, np.nan), (6, -2.0)])
    def test_nan_or_negative_id_raises(self, col, value):
        arr = np.array([[1, 10, 10, 30, 30, 0.9, 1]], dtype=np.float64)
        arr[0, col] = value
        with pytest.raises(ValueError, match="ids must be finite and non-negative"):
            COCO(dict_dataset()).load_res(arr)


class TestLoadResSegmentation:
    """``load_res(array, segmentation=)``: the array route for segm, as ``StreamingEval.update`` has it."""

    ARRAY = np.array([[1, 10, 10, 30, 30, 0.9, 1], [2, 50, 50, 20, 20, 0.8, 2]], dtype=np.float64)

    @staticmethod
    def masks():
        rles = []
        for x in (10, 50):
            m = np.zeros((100, 100), np.uint8, order="F")
            m[x : x + 10, x : x + 5] = 1
            rles.append(hotcoco.mask.encode(m))
        return rles

    def as_dicts(self, rles):
        return [
            {
                "image_id": int(r[0]),
                "bbox": r[1:5].tolist(),
                "score": float(r[5]),
                "category_id": int(r[6]),
                "segmentation": rle,
            }
            for r, rle in zip(self.ARRAY, rles)
        ]

    def test_equals_dicts_with_a_box_and_a_mask(self):
        gt, rles = COCO(dict_dataset()), self.masks()
        want = gt.load_res(self.as_dicts(rles)).dataset
        assert gt.load_res(self.ARRAY, segmentation=rles).dataset == want
        assert gt.loadRes(self.ARRAY, segmentation=rles).dataset == want

    def test_area_is_the_box_until_update_anns_sets_the_mask(self):
        """Each row has a box, so its area is the box's, as pycocotools' ``loadRes``
        gives a result with a ``bbox``. The column ``update_anns`` switches it."""
        dt = COCO(dict_dataset()).load_res(self.ARRAY, segmentation=self.masks())
        assert [a["area"] for a in dt.dataset["annotations"]] == [900.0, 400.0]
        dt.update_anns(ids=[1, 2], area=hotcoco.mask.area(self.masks()))
        assert [a["area"] for a in dt.dataset["annotations"]] == [50.0, 50.0]
        assert dt.dataset["annotations"][0]["segmentation"]["size"] == [100, 100]

    def test_only_with_an_array(self, tmp_path):
        gt, rles = COCO(dict_dataset()), self.masks()
        dicts = self.as_dicts(rles)
        with pytest.raises(TypeError, match="segmentation goes with a detection array"):
            gt.load_res(dicts, segmentation=rles)
        path = tmp_path / "dt.json"
        path.write_text(json.dumps([{k: v for k, v in d.items() if k != "segmentation"} for d in dicts]))
        with pytest.raises(TypeError, match="segmentation goes with a detection array"):
            gt.load_res(str(path), segmentation=rles)

    def test_one_entry_per_row(self):
        with pytest.raises(ValueError, match="segmentation has 2 entries for 1 rows"):
            COCO(dict_dataset()).load_res(self.ARRAY[:1], segmentation=self.masks())
