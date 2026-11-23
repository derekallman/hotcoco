"""The calls Detectron2, MMDetection and torchvision make on pycocotools and
lvis-api, run against hotcoco behind `init_as_pycocotools()` / `init_as_lvis()`.

Neither framework is installed here, so each test repeats the calls its source
makes, in order, with the file and line beside them (branch heads read
2026-11-16). A line that changes upstream needs its row changed here; a new
call a framework starts making needs a new row. `docs/guide/frameworks.md`
may claim a framework works only as far as this file runs its calls.
"""

from __future__ import annotations

import copy
import json
import pickle
import re
import sys

import hotcoco
import numpy as np
import pytest


def _ours(name: str) -> bool:
    return name.split(".")[0] in ("pycocotools", "lvis")


@pytest.fixture
def no_reference_modules():
    """`pycocotools` and `lvis` out of `sys.modules`, and put back after."""
    saved = {name: mod for name, mod in sys.modules.items() if _ours(name)}
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [name for name in sys.modules if _ours(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def shims(no_reference_modules):
    """Both shims installed, and the real `pycocotools` / `lvis` put back after."""
    hotcoco.init_as_pycocotools()
    hotcoco.init_as_lvis()
    yield


def _dataset() -> dict:
    return {
        "info": {},
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
            {"id": 3, "width": 100, "height": 100, "file_name": "c.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 40, 40], "area": 1600, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }


def _lvis_dataset() -> dict:
    ds = _dataset()
    for img in ds["images"]:
        img["neg_category_ids"] = []
        img["not_exhaustive_category_ids"] = []
    for cat, freq in zip(ds["categories"], "fc"):
        cat["frequency"] = freq
    return ds


def _results() -> list[dict]:
    return [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.8},
        {"image_id": 2, "category_id": 1, "bbox": [5, 5, 40, 40], "score": 0.7},
    ]


@pytest.fixture
def ann_file(tmp_path):
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(_dataset()))
    return str(path)


@pytest.fixture
def lvis_file(tmp_path):
    path = tmp_path / "lvis.json"
    path.write_text(json.dumps(_lvis_dataset()))
    return str(path)


# ---------------------------------------------------------------------------
# MMDetection
# ---------------------------------------------------------------------------


def _mmdet_coco_class():
    """`mmdet/datasets/api_wrappers/coco_api.py` 8-45, as written there."""
    import pycocotools
    from pycocotools.coco import COCO as _COCO

    class COCO(_COCO):
        def __init__(self, annotation_file=None):
            if getattr(pycocotools, "__version__", "0") >= "12.0.2":
                raise AssertionError("mmdet warns here; hotcoco's version must not trip it")
            super().__init__(annotation_file=annotation_file)
            self.img_ann_map = self.imgToAnns
            self.cat_img_map = self.catToImgs

        def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
            return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

        def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
            return self.getCatIds(cat_names, sup_names, cat_ids)

        def get_img_ids(self, img_ids=[], cat_ids=[]):
            return self.getImgIds(img_ids, cat_ids)

        def load_anns(self, ids):
            return self.loadAnns(ids)

        def load_cats(self, ids):
            return self.loadCats(ids)

        def load_imgs(self, ids):
            return self.loadImgs(ids)

    return COCO


def test_mmdet_coco_wrapper_and_dataset_loading(shims, ann_file):
    COCO = _mmdet_coco_class()
    coco = COCO(ann_file)  # mmdet/datasets/coco.py:67
    cat_ids = coco.get_cat_ids(cat_names=("person", "dog"))  # :70
    assert cat_ids == [1, 2]
    cat_img_map = copy.deepcopy(coco.cat_img_map)  # :73
    assert set(cat_img_map[1]) == {1, 2}  # :186
    assert cat_img_map[1] == [1, 2] and cat_img_map[2] == [1]
    img_ids = coco.get_img_ids()  # :75
    assert img_ids == [1, 2, 3]
    info = coco.load_imgs([1])[0]  # :79
    info["img_id"] = 1
    ann_ids = coco.get_ann_ids(img_ids=[1])  # :82
    assert [a["id"] for a in coco.load_anns(ann_ids)] == [1, 2]  # :83
    assert coco.get_ann_ids(img_ids=3) == []  # coco_metric.py:169, a scalar id
    assert [a["id"] for a in coco.img_ann_map[1]] == [1, 2]
    assert coco.img_ann_map[3] == []


@pytest.mark.filterwarnings("ignore::hotcoco.ReferenceDeviationWarning")  # MMDetection's own maxDets
def test_mmdet_coco_metric(shims, ann_file):
    """`mmdet/evaluation/metrics/coco_metric.py` 413-532."""
    from pycocotools.cocoeval import COCOeval

    api = _mmdet_coco_class()(ann_file)
    cat_ids = api.get_cat_ids(cat_names=("person", "dog"))
    img_ids = api.get_img_ids()
    coco_dt = api.loadRes(_results())  # :462
    coco_eval = COCOeval(api, coco_dt, "bbox")  # :472
    coco_eval.params.catIds = cat_ids  # :474
    coco_eval.params.imgIds = img_ids
    coco_eval.params.maxDets = list((100, 300, 1000))
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    assert coco_eval.stats[0] == pytest.approx(1.0)  # :514
    precisions = coco_eval.eval["precision"]  # :523
    assert len(cat_ids) == precisions.shape[2]
    assert precisions[:, :, 0, 0, -1].mean() == pytest.approx(1.0)
    assert api.loadCats(1)[0]["name"] == "person"  # :532, a scalar id


def test_cocoeval_can_be_subclassed(shims, ann_file):
    """Detectron2 (`COCOevalMaxDets`, `COCOeval_opt`) and MMDetection
    (`COCOevalMP`) define subclasses at import, so the class statement itself
    has to work, and a subclass that overrides nothing has to evaluate."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    class COCOevalMaxDets(COCOeval):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.note = "mine"

    gt = COCO(ann_file)
    ev = COCOevalMaxDets(gt, gt.loadRes(_results()), "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    assert ev.note == "mine"
    assert ev.stats[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Detectron2
# ---------------------------------------------------------------------------


def test_detectron2_load_coco_json(shims, ann_file):
    """`detectron2/data/datasets/coco.py` 69-133."""
    from pycocotools.coco import COCO

    coco_api = COCO(ann_file)  # :74
    cat_ids = sorted(coco_api.getCatIds())  # :81
    assert [c["name"] for c in coco_api.loadCats(cat_ids)] == ["person", "dog"]
    img_ids = sorted(coco_api.imgs.keys())  # :106
    assert len(coco_api.loadImgs(img_ids)) == 3  # :115
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]  # :131, image 3 has none
    assert [len(a) for a in anns] == [2, 1, 0]
    assert sum(len(a) for a in anns) == len(coco_api.anns)  # :133
    assert anns[0][0].get("ignore", 0) == 0  # :176


def test_detectron2_evaluate_predictions_on_coco(shims, ann_file):
    """`detectron2/evaluation/coco_evaluation.py` 146-150, 497-502, 590-629."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)
    assert "annotations" in coco_gt.dataset  # :150
    ann_ids = coco_gt.getAnnIds(imgIds=1)  # :497, a scalar id
    assert all(obj["iscrowd"] == 0 for obj in coco_gt.loadAnns(ann_ids))  # :502
    coco_dt = coco_gt.loadRes(_results())  # :590
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")  # :591
    coco_eval.params.maxDets = [1, 10, 100]  # :604
    coco_eval.params.imgIds = [1, 2, 3]  # :607
    assert hasattr(coco_eval.params, "kpt_oks_sigmas")  # :612
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    assert coco_eval.stats[0] == pytest.approx(1.0)  # :349
    assert coco_eval.eval["precision"].shape[2] == 2  # :362
    copy.deepcopy(coco_eval.params)  # fast_eval_api.py:93


def test_detectron2_imports_mask_utils_from_the_evaluator_modules(shims):
    """`detectron2/evaluation/rotated_coco_evaluation.py` 7, which
    `detectron2.evaluation` imports, so nothing in it loads without the name."""
    from pycocotools.coco import maskUtils as from_coco
    from pycocotools.cocoeval import COCOeval, maskUtils  # :7

    assert maskUtils is from_coco is hotcoco.mask
    assert COCOeval is hotcoco.COCOeval


@pytest.mark.filterwarnings("ignore::hotcoco.ReferenceDeviationWarning")  # Detectron2's own maxDets
def test_detectron2_max_dets_summarize(shims, ann_file):
    """`COCOevalMaxDets.summarize`, `detectron2/evaluation/coco_evaluation.py`
    640-717: it indexes `eval` with numpy comparisons against `params.iouThrs`
    and assigns `self.stats`."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    class COCOevalMaxDets(COCOeval):  # :634
        def summarize(self):
            p = self.params
            assert "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) == "0.50:0.95"  # :652
            aind = [i for i, lbl in enumerate(p.areaRngLbl) if lbl == "all"]  # :657
            mind = [i for i, m in enumerate(p.maxDets) if m == self.params.maxDets[2]]  # :658
            s = self.eval["precision"]  # :661
            t = np.where(0.5 == p.iouThrs)[0]  # :664
            assert t.tolist() == [0]
            s = s[t][:, :, :, aind, mind]  # :665
            stats = np.zeros((12,))  # :682
            stats[1] = np.mean(s[s > -1])  # :674
            self.stats = stats  # :717

    coco_gt = COCO(ann_file)
    coco_eval = COCOevalMaxDets(coco_gt, coco_gt.loadRes(_results()), "bbox")
    coco_eval.params.maxDets = [1, 10, 250]  # :604
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    assert coco_eval.stats.shape == (12,)  # :349 reads it back
    assert coco_eval.stats[1] == pytest.approx(1.0)
    # The next run's own summary replaces the assigned one, as pycocotools' does.
    COCOeval.summarize(coco_eval)
    assert coco_eval.stats[0] == pytest.approx(1.0)
    # So does `summary_lines()`, which computes `stats` too.
    coco_eval.stats = "assigned"
    coco_eval.summary_lines()
    assert coco_eval.stats[0] == pytest.approx(1.0)
    # `run()` calls the subclass's steps, as lvis-api's `run()` does.
    coco_eval.run()
    assert coco_eval.stats[0] == 0.0 and coco_eval.stats[1] == pytest.approx(1.0)


def test_params_thresholds_are_arrays_under_the_pycocotools_names():
    """pycocotools holds `iouThrs` and `recThrs` as numpy arrays, and callers
    compare a scalar against them."""
    p = hotcoco.Params()
    assert isinstance(p.iouThrs, np.ndarray) and p.iouThrs.dtype == np.float64
    assert isinstance(p.recThrs, np.ndarray) and p.recThrs.shape == (101,)
    assert (0.75 == p.iouThrs).nonzero()[0].tolist() == [5]
    assert isinstance(p.iou_thrs, list) and p.iou_thrs == p.iouThrs.tolist()


def test_detectron2_mask_calls(shims):
    """`detectron2/structures/masks.py` 34-36 and `utils/visualizer.py` 147-149, 1088."""
    import pycocotools.mask as mask_util

    polygons = [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]]
    rles = mask_util.frPyObjects(polygons, 50, 60)
    rle = mask_util.merge(rles)
    assert mask_util.decode(rle).astype(bool).shape == (50, 60)
    assert mask_util.toBbox(rle).shape == (4,)
    assert mask_util.area(rles).item() == mask_util.area(rle)
    mask = np.zeros((50, 60), dtype=bool)
    mask[5:9, 5:9] = True
    encoded = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]  # coco_evaluation.py:418
    encoded["counts"] = encoded["counts"].decode("utf-8")  # :426
    assert mask_util.area(encoded) == 16


def test_detectron2_and_mmdet_lvis(shims, lvis_file, tmp_path):
    """`detectron2/data/datasets/lvis.py` 68-101, `evaluation/lvis_evaluation.py`
    72-75 and 263-377; `mmdet/evaluation/metrics/lvis_metric.py` 291-332 and 488-496."""
    import lvis
    from lvis import LVIS, LVISEval, LVISResults

    assert not getattr(lvis, "__version__", "0") >= "10.5.3"  # lvis_metric.py:27, warns if true
    lvis_api = LVIS(lvis_file)
    img_ids = sorted(lvis_api.imgs.keys())
    assert len(lvis_api.load_imgs(img_ids)) == 3
    assert [len(lvis_api.img_ann_map[i]) for i in img_ids] == [2, 1, 0]
    assert len(lvis_api.get_ann_ids()) > 0
    ann_ids = lvis_api.get_ann_ids(img_ids=[1])
    assert len(lvis_api.load_anns(ann_ids)) == 2
    assert set(copy.deepcopy(lvis_api.cat_img_map)[1]) == {1, 2}  # mmdet/datasets/lvis.py:295
    # lvis-api's own defaults: `None` is no filter, and `load_anns()` is everything.
    assert lvis_api.get_ann_ids(img_ids=None, cat_ids=None, area_rng=None) == [1, 2, 3]
    assert len(lvis_api.load_anns(None)) == 3
    assert lvis_api.load_cats([1])[0]["name"] == "person"  # lvis_metric.py:332

    results = LVISResults(lvis_api, _results(), max_dets=300)  # lvis_evaluation.py:371
    lvis_eval = LVISEval(lvis_api, results, "bbox")
    lvis_eval.run()
    lvis_eval.print_results()
    assert lvis_eval.get_results()["AP"] == pytest.approx(1.0)

    result_file = tmp_path / "results.json"
    result_file.write_text(json.dumps(_results()))
    lvis_dt = LVISResults(lvis_api, str(result_file))  # lvis_metric.py:291, a path
    lvis_eval = LVISEval(lvis_api, lvis_dt, "bbox")
    lvis_eval.params.imgIds = lvis_api.get_img_ids()  # :299
    lvis_eval.evaluate()
    lvis_eval.accumulate()
    lvis_eval.summarize()
    assert dict(lvis_eval.get_results().items())["AP"] == pytest.approx(1.0)
    assert lvis_eval.eval["precision"].shape[2] == 2  # :322

    # `LVISFixedAPMetric` is not supported past this point and its next lines
    # are not run: it sets `params.max_dets = -1` (:492), an int where hotcoco
    # holds a list, and reads `lvis_eval.results` (:496) as a dict, which is
    # a method here.
    fixed = LVISResults(lvis_api, _results(), max_dets=-1)  # :488
    lvis_eval = LVISEval(lvis_api, fixed, iou_type="bbox")
    lvis_eval.run()
    assert lvis_eval.get_results()["AP"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# torchvision references/detection
# ---------------------------------------------------------------------------


def test_torchvision_coco_utils_and_coco_eval(shims):
    """`coco_utils.py` 111-170 and `coco_eval.py` 17, 35, 186-192.

    torchvision's `CocoEvaluator` itself is not run: it builds `COCOeval`
    with no detections and assigns `cocoDt`, `evalImgs` and `_paramsEval`,
    which `hotcoco.integrations.CocoEvaluator` replaces."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_ds = COCO()  # coco_utils.py:123
    coco_ds.dataset = _dataset()  # :169
    coco_ds.createIndex()  # :170
    assert coco_ds.getAnnIds(imgIds=1, iscrowd=None) == [1, 2]  # :111

    coco_gt = copy.deepcopy(coco_ds)  # coco_eval.py:17
    assert coco_gt.getAnnIds() == [1, 2, 3]
    coco_dt = COCO.loadRes(coco_gt, _results())  # :35, called on the class
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = [1, 2, 3]  # :39
    coco_eval.evaluate()
    img_ids = coco_eval.params.imgIds
    grid = np.asarray(coco_eval.evalImgs).reshape(-1, len(coco_eval.params.areaRng), len(img_ids))  # :192
    assert grid.shape == (2, 4, 3)
    assert grid[0, 0, 0]["image_id"] == 1 and grid[0, 0, 0]["category_id"] == 1
    assert grid[1, 0, 0]["category_id"] == 2
    assert grid[1, 0, 1] is None  # image 2 has no dog
    assert grid[0, 0, 2] is None  # image 3 has nothing
    copy.deepcopy(coco_eval.params)  # :186


# ---------------------------------------------------------------------------
# Copies
# ---------------------------------------------------------------------------


def _public_state(obj) -> dict:
    def plain(value):
        return value.tolist() if isinstance(value, np.ndarray) else value

    return {
        name: plain(getattr(obj, name))
        for name in dir(obj)
        if not name.startswith("_") and not callable(getattr(obj, name))
    }


@pytest.mark.parametrize("clone", [copy.deepcopy, copy.copy, lambda o: pickle.loads(pickle.dumps(o))])
def test_params_copies_carry_every_field(clone):
    """Every public attribute, so a field added to `Params` and not to its
    pickled state fails here."""
    params = hotcoco.Params("keypoints")
    params.imgIds = [3, 1]
    params.catIds = [2]
    params.iouThrs = [0.3, 0.7]
    params.recThrs = [0.0, 0.5, 1.0]
    params.maxDets = [7, 50]
    params.areaRng = [[0, 1e10], [0, 50]]
    params.areaRngLbl = ["all", "tiny"]
    params.useCats = 0
    params.expand_dt = not params.expand_dt
    params.kpt_oks_sigmas = [0.1, 0.2]
    changed = _public_state(params)
    assert all(changed[name] != value for name, value in _public_state(hotcoco.Params("bbox")).items()), (
        "a field left at its default proves nothing about the copy"
    )
    assert _public_state(clone(params)) == changed


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))])
def test_coco_copies_are_independent_and_equal(clone, ann_file):
    coco = hotcoco.COCO(ann_file, image_dir="/images")
    coco.dataset["annotations"][0]["note"] = "custom key"
    twin = clone(coco)
    assert twin.dataset == coco.dataset
    assert twin.image_dir == "/images"
    twin.dataset["annotations"].pop()
    assert len(twin.getAnnIds()) == 2 and len(coco.getAnnIds()) == 3


def test_a_capped_lvis_results_copy_is_not_capped_again(lvis_file):
    """`LVISResults(..., max_dets=-1)` marks the set as already capped; a copy
    that lost the mark would be cut to 300 per image by `LVISeval`."""
    gt = hotcoco.COCO(lvis_file)
    many = [
        {"image_id": 1, "category_id": 1 + i % 2, "bbox": [i % 50, 0, 10, 10], "score": 1 / (i + 1)} for i in range(400)
    ]
    kept = {}
    for name, results in (
        ("plain", gt.loadRes(many)),
        ("marked", hotcoco.LVISResults(gt, many, max_dets=-1)),
        ("copied", copy.deepcopy(hotcoco.LVISResults(gt, many, max_dets=-1))),
    ):
        ev = hotcoco.LVISeval(gt, results, "bbox")
        ev.evaluate()
        kept[name] = sum(len(e["dtIds"]) for e in ev.eval_imgs if e["aRng"][0] == 0 and e["aRng"][1] > 1e9)
    assert kept == {"plain": 300, "marked": 400, "copied": 400}


def test_a_subclass_copy_keeps_its_class_and_attributes(ann_file):
    class Wrapped(hotcoco.COCO):
        def __init__(self, annotation_file=None):
            super().__init__(annotation_file=annotation_file)
            self.tag = "kept"

    twin = copy.deepcopy(Wrapped(ann_file))
    assert type(twin) is Wrapped and twin.tag == "kept"
    assert len(twin.getImgIds()) == 3


# ---------------------------------------------------------------------------
# Copies of an evaluator: the run is repeated on the copy
# ---------------------------------------------------------------------------

CLONES = [copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))]
STEPS = ["evaluate", "accumulate", "summarize"]


def _evaluator(ann_file, cls=None, **kwargs):
    gt = hotcoco.COCO(ann_file)
    return (cls or hotcoco.COCOeval)(gt, gt.loadRes(_results()), "bbox", **kwargs)


class Tagged(hotcoco.COCOeval):
    """Module-level, so pickle can name it."""


def _run_state(ev) -> dict:
    """Everything a caller can read off the run."""
    acc = ev.eval
    return {
        "params": _public_state(ev.params),
        "stats": np.asarray(ev.stats).tolist(),
        "eval": {k: np.asarray(acc[k]).tolist() for k in ("precision", "recall", "scores")} if acc else None,
        "eval_imgs": ev.eval_imgs,
    }


@pytest.mark.parametrize("clone", CLONES)
@pytest.mark.parametrize("reached", range(4))
def test_an_evaluator_copy_is_at_the_same_step_with_the_same_results(clone, reached, ann_file, capsys, recwarn):
    ev = _evaluator(ann_file)
    for step in STEPS[:reached]:
        getattr(ev, step)()
    capsys.readouterr()
    recwarn.clear()
    twin = clone(ev)
    # Repeating the run on the copy prints no summary and raises no warning.
    assert capsys.readouterr().out == ""
    assert [str(w.message) for w in recwarn.list] == []
    assert _run_state(twin) == _run_state(ev)
    # Each goes on from the step it had reached.
    for step in STEPS[reached:]:
        getattr(ev, step)()
        getattr(twin, step)()
    assert _run_state(twin) == _run_state(ev)
    assert twin.get_results() == ev.get_results() != {}
    # The copy has its own datasets and params.
    twin.params.imgIds = [1]
    assert ev.params.imgIds == [1, 2, 3]


@pytest.mark.parametrize("clone", CLONES)
def test_an_evaluator_copy_repeats_each_step_under_the_params_it_ran_with(clone, ann_file):
    ev = _evaluator(ann_file)
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.params.imgIds = [1]  # accumulate() over one of the evaluated images
    ev.accumulate()
    ev.summarize()
    ev.params.catIds = [2]  # edited after the run, not yet used
    twin = clone(ev)
    assert _run_state(twin) == _run_state(ev)
    assert len(twin.eval_imgs) == len(ev.eval_imgs) > 0
    assert twin.params.catIds == [2]


@pytest.mark.filterwarnings("ignore::hotcoco.ReferenceDeviationWarning")  # Open Images: an extension
@pytest.mark.parametrize("clone", CLONES)
def test_lvis_and_open_images_evaluator_copies(clone, ann_file, lvis_file):
    gt = hotcoco.COCO(lvis_file)
    lvis = hotcoco.LVISeval(gt, hotcoco.LVISResults(gt, _results()), "bbox")
    lvis.run()
    twin = clone(lvis)
    assert twin.get_results() == lvis.get_results() != {}
    assert len(twin.stats) == 13

    oid = _evaluator(ann_file, oid_style=True, hierarchy=hotcoco.Hierarchy.from_parent_map({1: 2}))
    oid.run()
    twin = clone(oid)
    assert _run_state(twin) == _run_state(oid)
    assert twin.get_results() == oid.get_results() != {}


@pytest.mark.parametrize("clone", CLONES)
def test_an_evaluator_subclass_copy_keeps_its_class_attributes_and_assigned_stats(clone, ann_file):
    ev = _evaluator(ann_file, cls=Tagged)
    ev.tag = "kept"
    ev.evaluate()
    ev.accumulate()
    ev.stats = np.arange(12.0)
    twin = clone(ev)
    assert type(twin) is Tagged and twin.tag == "kept"
    assert twin.stats.tolist() == list(range(12))


@pytest.mark.parametrize("clone", CLONES)
def test_a_finalized_stream_does_not_copy(clone):
    """It holds match results and no datasets, so there is no run to repeat."""
    stream = hotcoco.StreamingEval([{"id": 1, "name": "a"}], iou_type="bbox")
    stream.update(
        [{"id": 1, "width": 100, "height": 100}],
        [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0, "id": 1}],
        [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}],
    )
    ev = stream.finalize()
    with pytest.raises(TypeError, match="StreamingEval"):
        clone(ev)


@pytest.mark.parametrize(("real", "init"), [("pycocotools.coco", "init_as_pycocotools"), ("lvis", "init_as_lvis")])
def test_installing_a_shim_after_the_real_import_warns(no_reference_modules, real, init):
    """A module that imported the real library first keeps it, since the patch
    only reaches imports made after it. That run is on the reference, not on
    hotcoco, and nothing said so."""
    import importlib

    importlib.import_module(real)
    with pytest.warns(hotcoco.DropInWarning, match=re.escape(real.split(".")[0])):
        getattr(hotcoco, init)()


def test_installing_a_shim_first_is_silent(no_reference_modules):
    """The documented order, and a second call, say nothing."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            hotcoco.init_as_pycocotools()
            hotcoco.init_as_lvis()


@pytest.mark.parametrize("name", ["iouThrs", "recThrs"])
def test_an_in_place_edit_of_a_params_array_raises(name):
    """`params.iouThrs` is a copy, so `params.iouThrs[0] = 0.4` changed nothing
    and said nothing. Read-only, the edit raises; an edited copy assigned back
    is how to change it."""
    p = hotcoco.Params("bbox")
    with pytest.raises(ValueError, match="read-only"):
        getattr(p, name)[0] = 0.4
    edited = getattr(p, name).copy()
    edited[0] = 0.4
    setattr(p, name, edited)
    assert getattr(p, name)[0] == 0.4
