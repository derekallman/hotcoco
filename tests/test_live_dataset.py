"""``coco.dataset`` is the dataset, not a copy of it.

pycocotools builds and edits a ``COCO`` through its ``dataset`` dict: assign
one, append to its lists, change a field, then call ``createIndex()``.
torchmetrics' bbox+segm run changes every prediction's ``area`` through a fresh
``coco_preds.dataset`` access and evaluates with no ``createIndex()`` at all.
Each test here runs one such idiom through hotcoco and pycocotools and compares
what the two see afterwards.

The one documented difference: an edit made through a reference kept from
before the last call needs ``createIndex()`` to be seen, as pycocotools' own
index does for an appended record.
"""

import contextlib
import copy
import json
import warnings

import numpy as np
import pytest
from helpers import torchmetrics_dataset
from hotcoco import COCO, COCOeval, mask
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval


def dataset(predictions=False):
    """torchmetrics' shape on a 128px canvas with half-filled masks, so bbox and
    mask areas land in different size ranges, with RLE ``counts`` as str."""
    return torchmetrics_dataset(
        predictions, seed=3, canvas=128, origin=40, box=(30, 60), half_mask=True, bytes_counts=False
    )


def both(predictions=False):
    """The same dataset as a hotcoco ``COCO`` and a pycocotools one, each built
    the way pycocotools' own code builds one in memory."""
    ours, ref = COCO(), PyCOCO()
    ours.dataset = dataset(predictions)
    ours.createIndex()
    ref.dataset = dataset(predictions)
    ref.createIndex()
    return ours, ref


def stats(coco_gt, coco_dt, iou_type, cls=COCOeval):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev = cls(coco_gt, coco_dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return list(ev.stats)


def new_annotation(coco):
    return {
        "id": max(coco.getAnnIds()) + 1,
        "image_id": 0,
        "category_id": 2,
        "bbox": [1.0, 1.0, 10.0, 10.0],
        "area": 100.0,
        "iscrowd": 0,
    }


class TestIdentity:
    def test_same_dict_on_every_access(self):
        ours, ref = both()
        assert ours.dataset is ours.dataset
        assert ref.dataset is ref.dataset

    def test_an_assigned_dict_is_the_dataset(self):
        d = dataset()
        ours, ref = COCO(), PyCOCO()
        ours.dataset = d
        ref.dataset = d
        assert ours.dataset is d
        assert ref.dataset is d

    def test_an_assigned_dict_keeps_its_other_keys(self):
        """``info`` and ``licenses`` are not read, and pycocotools keeps them too."""
        d = {**dataset(), "info": {"year": 2026}, "licenses": []}
        ours = COCO()
        ours.dataset = d
        ours.createIndex()
        assert ours.dataset["info"] == {"year": 2026}
        assert ours.getAnnIds() == list(range(1, 25))


class TestEditsReachTheIndex:
    def test_append_then_create_index(self):
        ours, ref = both()
        ann = new_annotation(ours)
        for coco in (ours, ref):
            coco.dataset["annotations"].append(copy.deepcopy(ann))
            coco.createIndex()
        assert sorted(ours.getAnnIds()) == sorted(ref.getAnnIds())
        assert ours.loadAnns(ann["id"]) == ref.loadAnns(ann["id"])
        assert ours.getAnnIds(imgIds=0) == ref.getAnnIds(imgIds=0)
        assert set(ours.anns) == set(ref.anns)
        assert ours.anns[ann["id"]]["bbox"] == ann["bbox"]

    def test_remove_then_create_index(self):
        ours, ref = both()
        for coco in (ours, ref):
            del coco.dataset["annotations"][0]
            coco.dataset["images"].pop()
            coco.createIndex()
        assert ours.getAnnIds() == ref.getAnnIds()
        assert ours.getImgIds() == ref.getImgIds()
        assert set(ours.imgs) == set(ref.imgs)

    def test_a_field_edit_is_seen_without_create_index(self):
        """pycocotools' index holds the same dict objects as ``dataset``, so a
        field edit shows up in ``loadAnns`` at once; hotcoco reads the dict back
        before the call."""
        ours, ref = both()
        for coco in (ours, ref):
            coco.dataset["annotations"][0]["area"] = 5.0
        assert ours.loadAnns(1)[0]["area"] == ref.loadAnns(1)[0]["area"] == 5.0
        assert ours.anns[1]["area"] == 5.0

    def test_a_category_edit_is_seen(self):
        ours, ref = both()
        for coco in (ours, ref):
            coco.dataset["categories"][0]["name"] = "renamed"
            coco.createIndex()
        assert ours.getCatIds(catNms=["renamed"]) == ref.getCatIds(catNms=["renamed"]) == [1]
        assert ours.loadCats(1) == ref.loadCats(1)

    def test_an_edit_through_a_kept_reference_is_seen_after_create_index(self):
        ours, _ = both()
        d = ours.dataset
        ours.getAnnIds()  # reads the dict back; `d` is now a clean, kept reference
        d["annotations"][0]["area"] = 7.0
        ours.createIndex()
        assert ours.loadAnns(1)[0]["area"] == 7.0

    def test_an_edit_before_a_rust_side_edit_is_kept(self):
        ours, _ = both()
        ours.dataset["annotations"][1]["area"] = 3.0
        ours.update_anns([{"id": 1, "area": 9.0}])
        by_id = {a["id"]: a["area"] for a in ours.dataset["annotations"]}
        assert by_id[1] == 9.0 and by_id[2] == 3.0
        assert ours.loadAnns([1, 2]) == [ours.anns[1], ours.anns[2]]
        assert [a["area"] for a in ours.loadAnns([1, 2])] == [9.0, 3.0]

    def test_save_writes_the_edit(self, tmp_path):
        ours, _ = both()
        ours.dataset["annotations"][0]["area"] = 11.0
        ours.save(str(tmp_path / "out.json"))
        saved = json.loads((tmp_path / "out.json").read_text())
        assert saved["annotations"][0]["area"] == 11.0


class TestEditsReachTheEvaluator:
    def test_torchmetrics_area_switch_then_evaluate(self):
        """torchmetrics' bbox+segm loop: set every prediction's ``area`` from
        ``area_<iou_type>`` through a fresh ``dataset`` access, then build the
        evaluator. The small/medium/large AP and AR move with it."""
        gt_ours, gt_ref = both()
        dt_ours, dt_ref = both(predictions=True)
        for dt in (dt_ours, dt_ref):
            for ann in dt.dataset["annotations"]:
                ann["area"] = ann["area_bbox"]
        unswitched = stats(gt_ours, dt_ours, "bbox")
        for dt in (dt_ours, dt_ref):
            for ann in dt.dataset["annotations"]:
                ann["area"] = ann["area_segm"]
        got = stats(gt_ours, dt_ours, "bbox")
        assert got == stats(gt_ref, dt_ref, "bbox", cls=PyCOCOeval)
        assert got != unswitched, "the switch must change a size-range metric for this test to mean anything"

    def test_an_appended_ground_truth_is_evaluated(self):
        gt_ours, gt_ref = both()
        dt_ours, dt_ref = both(predictions=True)
        before = stats(gt_ours, dt_ours, "bbox")
        ann = {**new_annotation(gt_ours), "category_id": 1, "bbox": [60.0, 60.0, 50.0, 50.0], "area": 2500.0}
        for gt in (gt_ours, gt_ref):
            gt.dataset["annotations"].append(copy.deepcopy(ann))
            gt.createIndex()
        got = stats(gt_ours, dt_ours, "bbox")
        assert got == stats(gt_ref, dt_ref, "bbox", cls=PyCOCOeval)
        assert got != before

    def test_the_same_object_on_both_sides(self):
        """``COCOeval(coco, coco)``, ``healthcheck(coco)``, and ``merge([coco,
        coco])`` each read one object twice, after an edit to it."""
        ours, _ = both()
        ours.dataset["annotations"][0]["area"] = 13.0
        assert stats(ours, ours, "bbox")[0] == 1.0
        assert ours.healthcheck(ours)["errors"] == []
        merged = COCO.merge([ours, ours])
        assert merged.loadAnns(1)[0]["area"] == 13.0
        assert len(merged.getAnnIds()) == 2 * len(ours.getAnnIds())


class TestTheReadBackKeepsTheRest:
    """A `COCO` is more than its dataset: the loader's warnings and the LVIS
    per-image cap marker live beside it. Reading the dict back must carry them."""

    def test_the_lvis_cap_marker_survives_a_read(self):
        from hotcoco import LVISeval

        gt = COCO(
            {
                "images": [{"id": 1, "width": 400, "height": 400}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
                ],
                "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            }
        )
        # 349 confident misses in another category, then the one hit at the
        # lowest score. lvis-api's results cap is 300 per image across
        # categories and drops the hit; the evaluator's own max_dets is per
        # image and category and keeps it. AP tells which ran.
        junk = [
            {"image_id": 1, "category_id": 2, "bbox": [100 + 5 * (i % 50), 100 + 5 * (i // 50), 4, 4], "score": 0.9}
            for i in range(349)
        ]
        dets = junk + [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.1}]
        dt = gt.load_res(dets).cap_detections_per_image(None)

        def ap(dt):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ev = LVISeval(gt, dt, "bbox")
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
            return ev.stats[0]

        before = ap(dt)
        assert before > 0, "uncapped, the hit is evaluated"
        len(dt.dataset["annotations"])  # a read, not an edit
        assert ap(dt) == before

    def test_load_res_warnings_survive_a_read_and_are_not_repeated(self):
        gt, _ = both()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            dt = gt.load_res([{"image_id": 0, "category_id": 99, "bbox": [1.0, 1.0, 2.0, 2.0], "score": 0.5}])
            assert any("category_id 99" in w for w in dt.load_warnings)
            emitted = len(seen)
            len(dt.dataset["annotations"])
            dt.getAnnIds()
            assert any("category_id 99" in w for w in dt.load_warnings), "the loader's warning is kept"
            assert len(seen) == emitted, "the read-back repeats no warning"

    def test_a_duplicate_id_warns_once_through_assign_and_create_index(self):
        d = dataset()
        d["annotations"][1]["id"] = 1
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            coco = COCO()
            coco.dataset = d
            coco.createIndex()
            coco.getAnnIds()
            len(coco.dataset["annotations"])
            coco.getAnnIds()
        assert sum("duplicate" in str(w.message) for w in seen) == 1
        # An edit that makes a new problem is warned about, once.
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            coco.dataset["annotations"][2]["id"] = 1
            coco.getAnnIds()
            coco.getAnnIds()
        assert sum("duplicate" in str(w.message) for w in seen) == 1

    def test_an_edit_through_the_assigned_dict_is_seen(self):
        """The assigned dict is the caller's: an edit to it before the next call
        is read then. After a call it is a kept reference and needs
        ``createIndex()``, where pycocotools' shared record dicts need nothing."""
        ours, ref = COCO(), PyCOCO()
        pairs = ((ours, dataset()), (ref, dataset()))
        for coco, d in pairs:
            coco.dataset = d
            d["annotations"][0]["area"] = 5.0
        ref.createIndex()  # pycocotools has no index at all before this; hotcoco reads d at the call
        assert ours.loadAnns(1)[0]["area"] == ref.loadAnns(1)[0]["area"] == 5.0
        for coco, d in pairs:
            d["annotations"][0]["area"] = 6.0
            coco.createIndex()
        assert ours.loadAnns(1)[0]["area"] == ref.loadAnns(1)[0]["area"] == 6.0

    def test_update_anns_edits_the_dict_in_hand(self):
        ours, _ = both()
        d = ours.dataset
        first = d["annotations"][0]
        ours.update_anns([{"id": 1, "area": 9.0}])
        assert ours.dataset is d and first["area"] == 9.0
        ours.update_anns(ids=[2], area=[8.0])
        assert d["annotations"][1]["area"] == 8.0
        ann = new_annotation(ours)
        d["annotations"].append(ann)
        ours.createIndex()
        assert ann["id"] in ours.getAnnIds()


class TestBadEdits:
    def test_a_bad_edit_raises_where_it_is_read(self):
        ours, _ = both()
        ours.dataset["annotations"][2]["bbox"] = "x"
        with pytest.raises(ValueError, match=r"annotations\[2\].*bbox"):
            ours.getAnnIds()
        # Fixed in place, the next call reads the repaired dict.
        ours.dataset["annotations"][2]["bbox"] = [1.0, 1.0, 2.0, 2.0]
        assert ours.loadAnns(3)[0]["bbox"] == [1.0, 1.0, 2.0, 2.0]

    def test_a_list_that_is_not_a_list_raises_as_the_setter_does(self):
        ours, _ = both()
        ours.dataset["annotations"] = 5
        with pytest.raises(ValueError, match="annotations"):
            ours.createIndex()


def test_torchmetrics_bbox_and_segm_in_one_metric():
    """torchmetrics' ``iou_type=("bbox", "segm")`` on hotcoco against its
    pycocotools backend. This is the run RF-DETR splits in two today."""
    torch = pytest.importorskip("torch")
    tm = pytest.importorskip("torchmetrics.detection")
    pytest.importorskip("faster_coco_eval")  # the backend name the swap is built under
    from fuzz_torchmetrics import SWAPPED  # scripts/ is on pytest's path

    def tensors(anns, scores):
        xyxy = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        out = {
            "boxes": torch.tensor(xyxy, dtype=torch.float32),
            "labels": torch.tensor([a["category_id"] for a in anns]),
            "masks": torch.tensor(np.stack([mask.decode(a["segmentation"]) for a in anns])).bool(),
        }
        if scores:
            out["scores"] = torch.tensor([a["score"] for a in anns])
        return out

    def compute(backend):
        # Built under a name torchmetrics knows, then swapped, as the fuzzer does.
        metric = tm.MeanAveragePrecision(iou_type=("bbox", "segm"), backend="faster_coco_eval")
        assert hasattr(metric, "_coco_backend"), "the swap writes this attribute"
        metric._coco_backend = SWAPPED[backend]()
        metric.warn_on_many_detections = False
        gt, dt = dataset(), dataset(predictions=True)
        for image in gt["images"]:
            rows = [[a for a in ds["annotations"] if a["image_id"] == image["id"]] for ds in (gt, dt)]
            metric.update([tensors(rows[1], scores=True)], [tensors(rows[0], scores=False)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return metric.compute()

    got, ref = compute("hotcoco"), compute("pycocotools")
    for key in ref:
        assert torch.equal(got[key], ref[key]), key
    assert ref["segm_map_small"] != ref["bbox_map_small"], "the two areas must land in different size ranges"


# ---------------------------------------------------------------------------
# Every public method reads an edit back.
#
# The probe: an edit that makes a duplicate id, which the read-back warns about
# once. A method that read the data without the read-back emits nothing. The
# data is private to `live.rs` and reachable only through the accessor that
# reads back, so this is the test that would fail if that ever changed; a new
# public name needs a row or a reason.
# ---------------------------------------------------------------------------

ANN = {"image_id": 0, "bbox": [1.0, 1.0, 2.0, 2.0]}
DET = {"image_id": 0, "category_id": 1, "bbox": [1.0, 1.0, 2.0, 2.0], "score": 0.5}

READS = {
    "get_ann_ids": lambda c, p: c.get_ann_ids(),
    "getAnnIds": lambda c, p: c.getAnnIds(),
    "get_cat_ids": lambda c, p: c.get_cat_ids(),
    "getCatIds": lambda c, p: c.getCatIds(),
    "get_img_ids": lambda c, p: c.get_img_ids(),
    "getImgIds": lambda c, p: c.getImgIds(),
    "load_anns": lambda c, p: c.load_anns(1),
    "loadAnns": lambda c, p: c.loadAnns(1),
    "load_cats": lambda c, p: c.load_cats(1),
    "loadCats": lambda c, p: c.loadCats(1),
    "load_imgs": lambda c, p: c.load_imgs(0),
    "loadImgs": lambda c, p: c.loadImgs(0),
    "load_res": lambda c, p: c.load_res([DET]),
    "loadRes": lambda c, p: c.loadRes([DET]),
    "_load_res_capped": lambda c, p: c._load_res_capped([DET], 1),
    "cap_detections_per_image": lambda c, p: c.cap_detections_per_image(1),
    "ann_to_rle": lambda c, p: c.ann_to_rle(ANN),
    "annToRLE": lambda c, p: c.annToRLE(ANN),
    "ann_to_mask": lambda c, p: c.ann_to_mask(ANN),
    "annToMask": lambda c, p: c.annToMask(ANN),
    "stats": lambda c, p: c.stats(),
    "healthcheck": lambda c, p: c.healthcheck(),
    "filter": lambda c, p: c.filter(),
    "split": lambda c, p: c.split(),
    "sample": lambda c, p: c.sample(n=1),
    "save": lambda c, p: c.save(str(p / "out.json")),
    "to_yolo": lambda c, p: c.to_yolo(str(p / "yolo")),
    "to_voc": lambda c, p: c.to_voc(str(p / "voc")),
    "to_cvat": lambda c, p: c.to_cvat(str(p / "cvat.xml")),
    "to_dota": lambda c, p: c.to_dota(str(p / "dota")),
    "to_oid": lambda c, p: c.to_oid(str(p / "oid.csv")),
    "update_anns": lambda c, p: c.update_anns([{"id": 3, "area": 1.0}]),
    "create_index": lambda c, p: c.create_index(),
    "createIndex": lambda c, p: c.createIndex(),
    "merge": lambda c, p: COCO.merge([c]),
    "imgs": lambda c, p: c.imgs,
    "anns": lambda c, p: c.anns,
    "cats": lambda c, p: c.cats,
    "load_warnings": lambda c, p: c.load_warnings,
}

NOT_A_READ = {
    "dataset": "hands the dict out again; a dict already out is the truth",
    "image_dir": "a path on the wrapper, not data",
    "browse": "starts a server, and reads through the methods above",
    "load_res_oid": "reads a CSV first; the data is read through load_res_anns as load_res is",
    "from_arrays": "a classmethod building a new object",
    "from_cvat": "a classmethod building a new object",
    "from_dota": "a classmethod building a new object",
    "from_oid": "a classmethod building a new object",
    "from_voc": "a classmethod building a new object",
    "from_yolo": "a classmethod building a new object",
}


def test_every_public_name_has_a_row():
    public = {name for name in dir(COCO) if not name.startswith("__")}
    assert public == set(READS) | set(NOT_A_READ)


@pytest.mark.parametrize("name", sorted(READS))
def test_every_method_reads_the_edit_back(name, tmp_path):
    coco = COCO(dataset())
    coco.dataset["annotations"][1]["id"] = 1
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with contextlib.suppress(Exception):  # what the call does with the data is its own test
            READS[name](coco, tmp_path)
    assert any("duplicate" in str(w.message) for w in seen), f"{name} ran on stale data"
