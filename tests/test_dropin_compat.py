"""pycocotools drop-in spellings that hotcoco used to reject.

Each test calls hotcoco and pycocotools with the same input and compares the
results. The leniency lives on the compatibility surfaces — ``COCO``'s camelCase
getters and ``hotcoco.mask`` — while ``hotcoco.primitives`` stays strict; the last
class pins that split.
"""

import contextlib
import io
import json
import pathlib

import hotcoco
import numpy as np
import pytest
from hotcoco import COCO, COCOeval, primitives
from hotcoco import mask as hm

pycocotools = pytest.importorskip("pycocotools")
from pycocotools import mask as pm  # noqa: E402
from pycocotools.coco import COCO as PCOCO  # noqa: E402


def dataset():
    return {
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
            {"id": 3, "width": 100, "height": 100, "file_name": "c.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 1},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }


def pycoco(ds, tmp_path):
    """pycocotools' COCO only loads from a file."""
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ds))
    with contextlib.redirect_stdout(io.StringIO()):
        return PCOCO(str(path))


@pytest.fixture
def both(tmp_path):
    return COCO(dataset()), pycoco(dataset(), tmp_path)


# ---------------------------------------------------------------------------
# 1. COCO(pathlib.Path)
# ---------------------------------------------------------------------------


def test_coco_accepts_pathlike(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(dataset()))
    assert isinstance(path, pathlib.Path)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = PCOCO(path)
    ours = COCO(path)
    assert ours.getAnnIds() == sorted(ref.getAnnIds())
    assert ours.getImgIds() == sorted(ref.getImgIds())


# ---------------------------------------------------------------------------
# 2. Integer flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [0, 1, False, True, np.int64(0), np.int64(1), np.bool_(True)])
def test_get_ann_ids_iscrowd_int(both, flag):
    ours, ref = both
    assert ours.getAnnIds(iscrowd=flag) == sorted(ref.getAnnIds(iscrowd=flag))


@pytest.mark.parametrize("intersect", [0, 1, np.int64(1)])
def test_mask_merge_int_intersect(intersect):
    boxes = np.array([[0, 0, 10, 10], [5, 5, 10, 10]], dtype=np.float64)
    rles = pm.frPyObjects(boxes, 20, 20)
    assert hm.merge(rles, intersect) == pm.merge(rles, intersect)


# ---------------------------------------------------------------------------
# 3. areaRng=[] means "no range filter"
# ---------------------------------------------------------------------------


def test_get_ann_ids_empty_area_rng(both):
    ours, ref = both
    assert ours.getAnnIds(areaRng=[]) == sorted(ref.getAnnIds(areaRng=[]))
    assert ours.getAnnIds(areaRng=()) == ours.getAnnIds()


# ---------------------------------------------------------------------------
# 4. Id lists: any iterable of ints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [set, frozenset, lambda ids: dict.fromkeys(ids).keys(), tuple, list, np.array],
    ids=["set", "frozenset", "dict_keys", "tuple", "list", "ndarray"],
)
def test_id_lists_accept_any_iterable(both, make):
    ours, ref = both
    ids = make([1, 2])
    assert sorted(ours.getAnnIds(imgIds=ids)) == sorted(ref.getAnnIds(imgIds=ids))
    assert sorted(ours.getAnnIds(catIds=ids)) == sorted(ref.getAnnIds(catIds=ids))
    assert sorted(ours.getImgIds(imgIds=ids)) == sorted(ref.getImgIds(imgIds=ids))
    assert sorted(ours.getImgIds(catIds=make([1]))) == sorted(ref.getImgIds(catIds=make([1])))
    assert sorted(ours.getCatIds(catIds=ids)) == sorted(ref.getCatIds(catIds=ids))
    by_id = lambda xs: sorted(xs, key=lambda x: x["id"])  # noqa: E731
    assert by_id(ours.loadAnns(ids)) == by_id(ref.loadAnns(ids))
    assert by_id(ours.loadImgs(ids)) == by_id(ref.loadImgs(ids))
    assert by_id(ours.loadCats(ids)) == by_id(ref.loadCats(ids))
    # The snake_case methods share the extractor.
    assert sorted(ours.get_ann_ids(img_ids=ids)) == sorted(ref.getAnnIds(imgIds=ids))


def test_id_list_rejects_non_int_items(both):
    ours, _ = both
    with pytest.raises(TypeError):
        ours.getAnnIds(imgIds=["a"])


# ---------------------------------------------------------------------------
# 5. mask.iou on boxes
# ---------------------------------------------------------------------------

BOXES = [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 10.0, 10.0], [50.0, 50.0, 4.0, 4.0]]


@pytest.mark.parametrize(
    "dt, gt",
    [
        (BOXES, BOXES[:2]),
        (np.array(BOXES), np.array(BOXES[:2])),
        (np.array(BOXES, dtype=np.int64), np.array(BOXES[:2], dtype=np.int64)),
        (np.array(BOXES), BOXES[:2]),  # pycocotools converts a box list to an array first
        ([np.array(b) for b in BOXES], BOXES[:2]),
    ],
    ids=["lists", "float-arrays", "int-arrays", "array-and-list", "list-of-arrays"],
)
@pytest.mark.parametrize("iscrowd", [[0, 0], [0, 1], [False, True], np.array([1, 0])])
def test_mask_iou_on_boxes(dt, gt, iscrowd):
    ours = hm.iou(dt, gt, iscrowd)
    ref = pm.iou(dt, gt, iscrowd)
    assert isinstance(ours, np.ndarray)
    assert ours.dtype == np.float64
    np.testing.assert_array_equal(ours, ref)


def test_mask_iou_on_rles_unchanged():
    rles = pm.frPyObjects(np.array(BOXES), 64, 64)
    ours = hm.iou(rles, rles[:2], [0, 1])
    assert isinstance(ours, np.ndarray)
    np.testing.assert_array_equal(ours, pm.iou(rles, rles[:2], [0, 1]))


def test_mask_iou_rejects_mixed_kinds():
    rles = pm.frPyObjects(np.array(BOXES), 64, 64)
    with pytest.raises(Exception, match="same data type"):
        pm.iou(BOXES, rles, [0, 0, 0])
    with pytest.raises(TypeError, match="same kind"):
        hm.iou(BOXES, rles, [0, 0, 0])


def test_mask_iou_rejects_unrecognized_list():
    with pytest.raises(Exception, match="bounding box"):
        pm.iou([[1.0, 2.0]], BOXES, [0, 0, 0])
    with pytest.raises(TypeError, match="boxes"):
        hm.iou([[1.0, 2.0]], BOXES, [0, 0, 0])


def test_mask_iou_reads_a_generator_once():
    # The dispatch collects the argument once; re-iterating would see an
    # exhausted generator and return zero rows.
    ours = hm.iou((b for b in BOXES), BOXES[:2], [0, 1])
    np.testing.assert_array_equal(ours, pm.iou(BOXES, BOXES[:2], [0, 1]))


def test_mask_iou_empty_list_takes_other_kind():
    rles = pm.frPyObjects(np.array(BOXES), 64, 64)
    assert hm.iou([], BOXES, [0, 0, 0]).shape == (0, 3)
    assert hm.iou(rles, [], []).shape == (3, 0)


def test_mask_iou_takes_float_iscrowd():
    # A flag column read back from pandas is float.
    ours = hm.iou(BOXES, BOXES[:2], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(ours, pm.iou(BOXES, BOXES[:2], [0, 1]))


def test_mask_iou_empty_side_is_empty():
    # pycocotools returns `[]`; hotcoco returns a (D, G) array with a zero
    # dimension. Both have length zero and iterate as empty.
    assert len(pm.iou([], BOXES, [0, 0, 0])) == 0
    out = hm.iou([], BOXES, [0, 0, 0])
    assert out.shape == (0, 3)
    assert hm.iou(BOXES, [], []).shape == (3, 0)


# ---------------------------------------------------------------------------
# 6. run() and print_results() go through sys.stdout
# ---------------------------------------------------------------------------


def _eval(lvis=False):
    gt = COCO(dataset())
    dt = gt.loadRes([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9}])
    return COCOeval(gt, dt, "bbox", lvis_style=lvis) if lvis else COCOeval(gt, dt, "bbox")


def test_run_table_reaches_redirect_stdout():
    ev = _eval()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.run()
    lines = buf.getvalue().splitlines()
    assert len(lines) == 12
    assert lines == ev.summary_lines()


def test_run_then_print_results_reach_capsys(capsys):
    ev = _eval()
    ev.run()
    capsys.readouterr()
    ev.print_results()
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 12
    assert out[0].split() == ["AP", "=", f"{ev.stats[0]:.3f}"]


def test_print_results_before_summarize_goes_through_python(capfd):
    ev = _eval()
    with pytest.warns(UserWarning, match="No results to print"):
        ev.print_results()
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "No results" not in captured.err


def test_run_matches_pycocotools_stats(tmp_path):
    gt_ds = dataset()
    dets = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9}]
    from pycocotools.cocoeval import COCOeval as PCOCOeval

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ref_gt = pycoco(gt_ds, tmp_path)
        ref = PCOCOeval(ref_gt, ref_gt.loadRes(dets), "bbox")
        ref.evaluate()
        ref.accumulate()
        ref.summarize()
    ref_lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith(" Average")]

    gt = COCO(gt_ds)
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.run()
    ours = buf.getvalue().splitlines()
    # Values, not whole lines: the label padding differs ("Precision (AP)" vs
    # pycocotools' "Precision  (AP)"), which is a separate question.
    assert [ln.rsplit("=", 1)[1] for ln in ours] == [ln.rsplit("=", 1)[1] for ln in ref_lines]
    assert [ln.split("@")[1] for ln in ours] == [ln.split("@")[1] for ln in ref_lines]


# ---------------------------------------------------------------------------
# The canonical layer stays strict
# ---------------------------------------------------------------------------


class TestPrimitivesStayStrict:
    def test_mask_iou_rejects_boxes(self):
        with pytest.raises(TypeError):
            primitives.mask_iou(BOXES, BOXES, [False] * 3)

    def test_bbox_iou_takes_arrays(self):
        out = primitives.bbox_iou(np.array(BOXES), np.array(BOXES[:2]), [False, True])
        np.testing.assert_array_equal(out, pm.iou(BOXES, BOXES[:2], [0, 1]))

    def test_bbox_iou_rejects_rles(self):
        rles = pm.frPyObjects(np.array(BOXES), 64, 64)
        with pytest.raises(TypeError):
            primitives.bbox_iou(rles, rles, [False] * 3)

    def test_snake_case_getter_keeps_bool_flag(self):
        coco = COCO(dataset())
        assert coco.get_ann_ids(iscrowd=True) == [2]
        with pytest.raises(TypeError):
            coco.get_ann_ids(iscrowd=0)
        with pytest.raises((TypeError, ValueError)):
            coco.get_ann_ids(area_rng=[])

    def test_mask_bbox_iou_is_the_primitive(self):
        assert hotcoco.mask.bbox_iou is primitives.bbox_iou
