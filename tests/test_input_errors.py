"""One exception per kind of bad input, whichever path the input took.

A record reaches hotcoco as a Python dict, a JSON file, a numpy array row, or a
``StreamingEval`` batch. The same malformed value must get the same outcome on
every path that can express it: accepted everywhere, or rejected everywhere
with ``ValueError``. A JSON file cannot tell a wrong type from a bad value, so
inside a record both are ``ValueError``; ``TypeError`` is for an argument of the
wrong kind entirely, such as ``COCO(42)``.

The other kinds: a missing file is ``FileNotFoundError`` naming the file, a
call made before the step it needs is ``RuntimeError``, and an id the
evaluator does not know is ``KeyError``.

``AGENTS.md`` (Input errors) states the convention this file checks.
"""

import copy
import json
import math

import hotcoco
import numpy as np
import pytest
from hotcoco import COCO, COCOeval, StreamingEval, mask, panoptic, semantic

IMAGES = [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}]
CATS = [{"id": 1, "name": "cat"}]
GT = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
DT = {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}
MISSING = object()

# Why a path is left out of a row: the row's value is accepted there for a
# reason that is not the bug this file guards against.
NAN_TOKEN = "a file's NaN token reads as null, which a list of numbers cannot hold"
STREAMING_CATS = "StreamingEval is built with its categories, so an unknown one is a KeyError"


def with_field(record, field, value):
    record = copy.deepcopy(record)
    if value is MISSING:
        record.pop(field, None)
    else:
        record[field] = value
    return record


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def outcome(load, role, iou_type="bbox"):
    """``None`` if the input loads and evaluates, else the exception's class.

    ``role`` is what a loaded ``COCO`` is: ``"gt"``, evaluated against
    detections that are fine, or ``"dt"``, against ground truth that is. A
    ``StreamingEval`` is finalized.
    """
    try:
        loaded = load()
        if isinstance(loaded, StreamingEval):
            loaded.finalize().accumulate()
        elif role is not None:
            gt = loaded if role == "gt" else COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
            ev = COCOeval(gt, gt.load_res([DT]) if role == "gt" else loaded, iou_type)
            ev.evaluate()
            ev.accumulate()
    except Exception as e:
        return type(e)
    return None


def assert_agree(outcomes, expect=None):
    """Every path accepted, or every path raised ``ValueError``.

    ``expect`` pins which: ``"accept"`` or ``"reject"``. Without it the paths
    only have to agree.
    """
    shown = {path: (o.__name__ if o else "accepted") for path, o in outcomes.items()}
    assert len(set(outcomes.values())) == 1, f"paths disagree: {shown}"
    (result,) = set(outcomes.values())
    assert result in (None, ValueError), f"rejected with the wrong exception: {shown}"
    if expect is not None:
        assert (result is None) == (expect == "accept"), f"expected {expect}: {shown}"


def streamed(iou_type, gt_anns, dt_anns, *, images=IMAGES, cats=CATS, **kw):
    def load():
        evaluator = StreamingEval(list(cats), iou_type)
        evaluator.update(list(images), gt_anns, dt_anns, **kw)
        return evaluator

    return load


def numbers(value):
    """Whether ``value`` is a list of numbers, which an array row can hold."""
    return isinstance(value, list) and all(is_number(v) for v in value)


# ------------------------------------------------------------ ground truth

ARRAY_COLUMNS = {
    "image_id": "image_ids",
    "category_id": "category_ids",
    "id": "ids",
    "area": "area",
    "iscrowd": "iscrowd",
}


def gt_outcomes(tmp_path, field, value, skip=()):
    iou_type = "segm" if field == "segmentation" else "bbox"
    ann = with_field(GT, field, value)
    dataset = {"images": IMAGES, "annotations": [ann], "categories": CATS}
    paths = {"dict": lambda: COCO(dataset), "streaming": streamed(iou_type, [ann], [DT])}
    if value is not MISSING:
        path = tmp_path / "gt.json"
        path.write_text(json.dumps(dataset))
        paths["json"] = lambda: COCO(str(path))
    columns = {"image_ids": [1], "category_ids": [1], "boxes": [[10, 10, 20, 20]]}
    if field == "segmentation":
        columns.update(area=[400], segmentation=[value])
    elif field == "bbox" and numbers(value):
        columns["boxes"] = [value]
    elif field in ARRAY_COLUMNS and is_number(value):
        columns[ARRAY_COLUMNS[field]] = [value]
    else:
        columns = None
    if columns is not None:
        paths["arrays"] = lambda: COCO.from_arrays(IMAGES, CATS, **columns)
    return {name: outcome(load, "gt", iou_type) for name, load in paths.items() if name not in skip}


ID_CASES = [
    ("1.5", 1.5, "reject"),
    ("1.0", 1.0, "accept"),
    ("-1", -1, "reject"),
    ("'1'", "1", "reject"),
    ("None", None, "reject"),
    ("NaN", math.nan, "reject"),
    ("2**64", 2**64, "reject"),
    ("1e30", 1e30, "reject"),
]

GT_CASES = [
    *[
        (f"{field}={name}", field, value, expect, {})
        for field in ("image_id", "category_id", "id")
        for name, value, expect in ID_CASES
    ],
    # Python's True is the int 1, but a JSON true is not a number, so no path takes it.
    *[(f"{field}=True", field, True, "reject", {}) for field in ("image_id", "category_id", "id")],
    ("image_id missing", "image_id", MISSING, "reject", {}),
    ("category_id missing", "category_id", MISSING, "reject", {}),
    ("id missing", "id", MISSING, "accept", {}),
    ("category_id unknown", "category_id", 99, "accept", {"streaming": STREAMING_CATS}),
    ("bbox of 3", "bbox", [10, 10, 20], "reject", {}),
    ("bbox of 5", "bbox", [10, 10, 20, 20, 5], "reject", {}),
    ("bbox holding a str", "bbox", ["a", 10, 20, 20], "reject", {}),
    ("bbox holding NaN", "bbox", [10, 10, math.nan, 20], None, {"json": NAN_TOKEN}),
    ("bbox='x'", "bbox", "x", "reject", {}),
    ("bbox None", "bbox", None, "accept", {}),
    ("area='x'", "area", "x", "reject", {}),
    ("iscrowd=1.5", "iscrowd", 1.5, "reject", {}),
    ("iscrowd='yes'", "iscrowd", "yes", "reject", {}),
    ("iscrowd=None", "iscrowd", None, "reject", {}),
]


@pytest.mark.parametrize("field, value, expect, skip", [c[1:] for c in GT_CASES], ids=[c[0] for c in GT_CASES])
def test_ground_truth_field(tmp_path, field, value, expect, skip):
    assert_agree(gt_outcomes(tmp_path, field, value, skip=skip), expect)


# -------------------------------------------------------------- detections


def dt_outcomes(tmp_path, field, value, skip=()):
    iou_type = "segm" if field == "segmentation" else "bbox"
    det = with_field(DT, field, value)
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    path = tmp_path / "dt.json"
    path.write_text(json.dumps([det]))
    paths = {
        "list": lambda: gt.load_res([det]),
        "json": lambda: gt.load_res(str(path)),
        "streaming list": streamed(iou_type, [GT], [det]),
    }
    # [image_id, x, y, w, h, score, category_id]: a row holds numbers only, and
    # a box of four.
    row = {"image_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9, "category_id": 1}
    seg = {"segmentation": [value]} if field == "segmentation" else {}
    if field == "bbox" and numbers(value) and len(value) == 4 or field in row and field != "bbox" and is_number(value):
        row[field] = value
    elif not seg:
        row = None
    if row is not None:
        array = np.array([[row["image_id"], *row["bbox"], row["score"], row["category_id"]]], dtype=np.float64)
        paths["array"] = lambda: gt.load_res(array, **seg)
        paths["streaming array"] = streamed(iou_type, [GT], array, **seg)
    return {name: outcome(load, "dt", iou_type) for name, load in paths.items() if name not in skip}


DT_CASES = [
    *[
        (f"{field}={name}", field, value, expect, {})
        for field in ("image_id", "category_id")
        for name, value, expect in ID_CASES
    ],
    *[(f"{field}=True", field, True, "reject", {}) for field in ("image_id", "category_id")],
    ("image_id missing", "image_id", MISSING, "reject", {}),
    ("category_id missing", "category_id", MISSING, "reject", {}),
    ("image_id unknown", "image_id", 99, "accept", {}),
    (
        "category_id unknown",
        "category_id",
        99,
        "accept",
        {"streaming list": STREAMING_CATS, "streaming array": STREAMING_CATS},
    ),
    ("score=NaN", "score", math.nan, "reject", {}),
    ("score=inf", "score", math.inf, "accept", {}),
    ("score=-1", "score", -1, "accept", {}),
    ("score='0.9'", "score", "0.9", "reject", {}),
    ("score='NaN'", "score", "NaN", "reject", {}),
    ("score None", "score", None, "accept", {}),
    ("score missing", "score", MISSING, "accept", {}),
    ("bbox of 3", "bbox", [10, 10, 20], "reject", {}),
    ("bbox holding a str", "bbox", ["a", 10, 20, 20], "reject", {}),
    ("bbox holding NaN", "bbox", [10, 10, math.nan, 20], None, {"json": NAN_TOKEN}),
]


@pytest.mark.parametrize("field, value, expect, skip", [c[1:] for c in DT_CASES], ids=[c[0] for c in DT_CASES])
def test_detection_field(tmp_path, field, value, expect, skip):
    assert_agree(dt_outcomes(tmp_path, field, value, skip=skip), expect)


# ------------------------------------------------------------ segmentation

# Not pinned to accept or reject: a loaded RLE is kept as given until a mask is
# drawn from it, and how much is validated then is a separate question. Every
# path must still give the same answer.
SEG_CASES = [
    ("rle without size", {"counts": [10000]}, "reject", {}),
    ("rle size of 1", {"size": [100], "counts": [10000]}, "reject", {}),
    ("rle size negative", {"size": [-100, 100], "counts": [10000]}, "reject", {}),
    ("rle counts None", {"size": [100, 100], "counts": None}, "reject", {}),
    ("rle counts garbage", {"size": [100, 100], "counts": "\x01\x02!!"}, None, {}),
    ("rle counts short", {"size": [100, 100], "counts": [5, 5]}, None, {}),
    ("polygon of odd length", [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], None, {}),
    ("polygon holding a str", [["a", 2.0, 3.0, 4.0, 5.0, 6.0]], "reject", {}),
    ("polygon holding NaN", [[1.0, 2.0, math.nan, 4.0, 5.0, 6.0]], None, {"json": NAN_TOKEN}),
    ("segmentation='x'", "x", "reject", {}),
    ("segmentation=7", 7, "reject", {}),
]


@pytest.mark.parametrize("value, expect, skip", [c[1:] for c in SEG_CASES], ids=[c[0] for c in SEG_CASES])
def test_ground_truth_segmentation(tmp_path, value, expect, skip):
    assert_agree(gt_outcomes(tmp_path, "segmentation", value, skip=skip), expect)


@pytest.mark.parametrize("value, expect, skip", [c[1:] for c in SEG_CASES], ids=[c[0] for c in SEG_CASES])
def test_detection_segmentation(tmp_path, value, expect, skip):
    assert_agree(dt_outcomes(tmp_path, "segmentation", value, skip=skip), expect)


# ---------------------------------------------------- images and categories


def dataset_outcomes(tmp_path, images=IMAGES, cats=CATS, anns=(GT,)):
    dataset = {"images": list(images), "annotations": list(anns), "categories": list(cats)}
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(dataset))
    paths = {
        "dict": lambda: COCO(dataset),
        "json": lambda: COCO(str(path)),
        "streaming": streamed("bbox", list(anns), [DT], images=images, cats=cats),
    }
    return {name: outcome(load, "gt") for name, load in paths.items()}


DATASET_CASES = [
    ("image id=1.5", {"images": [with_field(IMAGES[0], "id", 1.5)]}),
    ("image id=-1", {"images": [with_field(IMAGES[0], "id", -1)]}),
    ("image id='1'", {"images": [with_field(IMAGES[0], "id", "1")]}),
    ("image id missing", {"images": [with_field(IMAGES[0], "id", MISSING)]}),
    ("image width='100'", {"images": [with_field(IMAGES[0], "width", "100")]}),
    ("image width=1.5", {"images": [with_field(IMAGES[0], "width", 1.5)]}),
    ("image neg_category_ids=[True]", {"images": [with_field(IMAGES[0], "neg_category_ids", [True])]}),
    (
        "image not_exhaustive_category_ids=[1.0]",
        {"images": [with_field(IMAGES[0], "not_exhaustive_category_ids", [1.0])]},
    ),
    ("image as a list", {"images": [[1, 2]]}),
    ("category id=1.5", {"cats": [with_field(CATS[0], "id", 1.5)]}),
    ("category id='1'", {"cats": [with_field(CATS[0], "id", "1")]}),
    ("category id missing", {"cats": [with_field(CATS[0], "id", MISSING)]}),
    ("category name=7", {"cats": [with_field(CATS[0], "name", 7)]}),
    ("annotation as a list", {"anns": [[1, 2]]}),
]


@pytest.mark.parametrize("parts", [c[1] for c in DATASET_CASES], ids=[c[0] for c in DATASET_CASES])
def test_dataset_record(tmp_path, parts):
    assert_agree(dataset_outcomes(tmp_path, **parts), "reject")


def test_a_dataset_list_that_is_not_a_list(tmp_path):
    dataset = {"images": {}, "annotations": [GT], "categories": CATS}
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(dataset))
    assert_agree(
        {"dict": outcome(lambda: COCO(dataset), None), "json": outcome(lambda: COCO(str(path)), None)}, "reject"
    )


def test_a_record_error_names_the_record_and_the_field():
    with pytest.raises(ValueError, match=r"annotations\[1\]: field 'bbox': "):
        COCO({"images": IMAGES, "annotations": [GT, with_field(GT, "bbox", "x")], "categories": CATS})
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    with pytest.raises(ValueError, match=r"res\[0\]: field 'image_id': expected a non-negative integer, got 1\.5"):
        gt.load_res([with_field(DT, "image_id", 1.5)])


# ------------------------------------------------------- arguments, not data


def test_an_argument_of_the_wrong_kind_is_a_type_error():
    """``TypeError`` is for the argument itself, not for what is inside it."""
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    for call in (
        lambda: COCO(42),
        lambda: COCO([GT]),
        lambda: gt.load_res(42),
        lambda: mask.decode(42),
        lambda: semantic.SemanticEval(3).update([[0, 1]], [[0, 1]]),
    ):
        with pytest.raises(TypeError):
            call()


# --------------------------------------------------------------------- files


def test_a_missing_file_is_file_not_found_naming_it(tmp_path):
    missing = tmp_path / "missing.json"
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    sem = semantic.SemanticEval(3)
    for call in (
        lambda: COCO(str(missing)),
        lambda: gt.load_res(str(missing)),
        lambda: panoptic.PanopticEval(str(missing), str(missing)),
        lambda: panoptic.pq_compute(str(missing), str(missing)),
        lambda: sem.update_png(str(missing), str(missing)),
        lambda: sem.update_folders(str(tmp_path / "no-such-folder"), str(tmp_path)),
        lambda: gt.save(str(tmp_path / "no-such-folder" / "out.json")),
    ):
        with pytest.raises(FileNotFoundError) as caught:
            call()
        assert str(tmp_path) in str(caught.value), caught.value


def test_a_file_that_does_not_parse_is_a_value_error(tmp_path):
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"images": [')
    not_png = tmp_path / "not.png"
    not_png.write_text("not a PNG")
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    for call in (
        lambda: COCO(str(truncated)),
        lambda: gt.load_res(str(truncated)),
        lambda: panoptic.PanopticEval(str(truncated), str(truncated)),
        lambda: semantic.SemanticEval(3).update_png(str(not_png), str(not_png)),
    ):
        with pytest.raises(ValueError):
            call()


# ----------------------------------------------- input errors, not RuntimeError


def test_input_errors_from_the_core_are_value_errors():
    """Each of these was a ``RuntimeError`` before the core named its kind."""
    small = mask.encode(np.asfortranarray(np.zeros((2, 2), dtype=np.uint8)))
    large = mask.encode(np.asfortranarray(np.zeros((3, 3), dtype=np.uint8)))
    with pytest.raises(ValueError, match="mismatched RLE dimensions"):
        mask.merge([small, large])
    one = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    two = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS + [{"id": 2, "name": "dog"}]})
    with pytest.raises(ValueError, match="different numbers of categories"):
        COCO.merge([one, two])


# ----------------------------------------------------------------- call order


def test_a_call_before_the_step_it_needs_is_a_runtime_error():
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    ev = COCOeval(gt, gt.load_res([DT]), "bbox")
    with pytest.raises(RuntimeError, match="evaluate"):
        ev.tide_errors()
    with pytest.raises(RuntimeError, match="evaluate"):
        hotcoco.compare(ev, ev)
    with pytest.raises(RuntimeError, match="run"):
        panoptic.PanopticEval(gt, gt).results()


# --------------------------------------------- what the rules must not catch


def test_a_lone_annotation_needs_no_category_id():
    """``ann_to_mask`` reads one annotation as pycocotools' ``annToMask`` does."""
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    polygon = [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0]]
    assert gt.annToMask({"image_id": 1, "segmentation": polygon}).sum() > 0
    assert gt.ann_to_rle({"image_id": 1, "segmentation": polygon})["size"] == [100, 100]


def test_only_a_records_own_nan_score_is_rejected(tmp_path):
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    path = tmp_path / "dt.json"
    path.write_text(
        '[\n{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "score": 0.5},\n'
        '{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "score": NaN}\n]'
    )
    with pytest.raises(ValueError, match="has a NaN score at line 3"):
        gt.load_res(str(path))
    # A `score` key in custom metadata is data like any other: null, not a string.
    path.write_text(
        '[{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "score": 0.5, "attributes": {"score": NaN}}]'
    )
    (det,) = gt.load_res(str(path)).dataset["annotations"]
    assert det["attributes"] == {"score": None}


def test_a_numpy_bool_is_not_an_id():
    """``numpy.True_`` is no ``bool`` to Python, but it is no id either, like ``True``.

    numpy 1.x still lets one be an index, under a deprecation warning, so the
    refusal cannot rely on ``__index__`` failing; the repr is ``True`` there.
    """
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    with pytest.raises(ValueError, match=r"field 'image_id': expected a non-negative integer, got (np\.)?True"):
        COCO({"images": IMAGES, "annotations": [with_field(GT, "image_id", np.True_)], "categories": CATS})
    with pytest.raises(ValueError, match=r"field 'category_id': expected a non-negative integer, got (np\.)?True"):
        gt.load_res([with_field(DT, "category_id", np.True_)])
    # A column spelled as a list or as a bool array fails the same way, by position.
    for column in ([True], [np.True_], np.array([True])):
        with pytest.raises(ValueError, match=r"image_ids\[0\]: expected a non-negative integer, got (np\.)?True"):
            COCO.from_arrays(IMAGES, CATS, column, [1], [[10, 10, 20, 20]])
    # A 0-d bool array is neither a `bool` nor a numpy bool scalar, and converts to a float too.
    with pytest.raises(ValueError, match=r"field 'id': expected a non-negative integer, got array\(True\)"):
        COCO({"images": IMAGES, "annotations": [with_field(GT, "id", np.array(True))], "categories": CATS})


def test_an_id_column_must_be_a_sequence():
    """A set has no order to pair with the other columns."""
    for column in ({1}, {1: 0}, (i for i in [1])):
        with pytest.raises(TypeError, match="image_ids must be a sequence"):
            COCO.from_arrays(IMAGES, CATS, column, [1], [[10, 10, 20, 20]])
    with pytest.raises(ValueError, match=r"image_ids\[0\]: expected a non-negative integer, got -1"):
        COCO.from_arrays(IMAGES, CATS, np.array([-1]), [1], [[10, 10, 20, 20]])
    with pytest.raises(TypeError, match=r"image_ids\[0\]: expected a non-negative integer, got '1'"):
        COCO.from_arrays(IMAGES, CATS, ["1"], [1], [[10, 10, 20, 20]])
