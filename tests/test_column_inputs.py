"""Column-form inputs: arrays in, no Python dict per annotation.

``COCO.from_arrays`` must equal ``COCO(dict)`` over the same annotations, and
``update_anns(ids=, area=)`` must equal the dict edit it replaces. No ``data/``
needed.
"""

import hotcoco
import numpy as np
import pytest
from hotcoco import COCO, COCOeval

IMAGES = [{"id": i, "width": 100, "height": 100, "file_name": f"{i}.jpg"} for i in (1, 2, 3)]
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}]
IMAGE_IDS = np.array([1, 2, 2, 3], dtype=np.int64)
CATEGORY_IDS = np.array([1, 2, 1, 1], dtype=np.int64)
BOXES = np.array([[10, 10, 30, 30], [50, 50, 20, 20], [0, 0, 40, 40], [5, 5, 10, 10]], dtype=np.float64)


def dict_dataset(ids=None, area=None, iscrowd=None, rles=None):
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
        if rles is not None:
            ann["segmentation"] = rles[i]
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

    @pytest.mark.parametrize("make", [lambda a: a.astype(np.int32), list, lambda a: a.tolist()])
    def test_other_integer_spellings(self, make):
        gt = COCO.from_arrays(IMAGES, CATEGORIES, make(IMAGE_IDS), make(CATEGORY_IDS), BOXES.tolist())
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
        by_arrays = COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, area=area, rles=rles)
        assert by_arrays.dataset == COCO(dict_dataset(rles=rles)).dataset

    def test_rles_without_area_raises(self):
        with pytest.raises(ValueError, match="area is required with rles"):
            COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, rles=[{}] * 4)

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

    def test_empty_dataset(self):
        gt = COCO.from_arrays(IMAGES, CATEGORIES, [], [], np.zeros((0, 4)))
        assert gt.dataset["annotations"] == []


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

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="area has 1 entries but ids has 2"):
            self.fresh().update_anns(ids=[1, 2], area=[1.0])

    def test_dict_form_still_positional(self):
        gt = self.fresh()
        gt.update_anns([{"id": 1, "area": 5.0}])
        assert gt.dataset["annotations"][0]["area"] == 5.0
