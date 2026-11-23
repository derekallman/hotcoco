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
from helpers import PanopticCase
from hotcoco import COCO, COCOeval, StreamingEval, mask, panoptic, primitives, semantic
from hotcoco.tracking import TrackingEval

IMAGES = [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}]
CATS = [{"id": 1, "name": "cat"}]
GT = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
DT = {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}
MISSING = object()

# Bad input is this file's subject, and the loader warns about much of what it
# tolerates: an unknown id, a non-finite token. What each case checks is the
# exception, so those warnings are expected here.
pytestmark = pytest.mark.filterwarnings("ignore::hotcoco.DatasetWarning")

# Why a path is left out of a row: the row's value is accepted there for a
# reason that is not the bug this file guards against.
NAN_TOKEN = "a file's NaN token reads as null, which a list of numbers cannot hold"
STREAMING_CATS = "StreamingEval is built with its categories, so an unknown one is a KeyError"
INF_TOKEN = "a file's Infinity token reads as null, which is a missing score; pycocotools reads it as inf"


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


def edited(coco, field, value):
    """pycocotools' edit-in-place path: ``with_field``'s rule applied to the
    first annotation of ``coco.dataset`` — ``MISSING`` deletes, anything else
    sets — then ``createIndex()``, where the dict is read back and a bad value
    raises."""
    target = coco.dataset["annotations"][0]
    if value is MISSING:
        target.pop(field, None)
    else:
        target[field] = value
    coco.createIndex()
    return coco


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
    paths = {
        "dict": lambda: COCO(dataset),
        "edited dict": lambda: edited(COCO({"images": IMAGES, "annotations": [GT], "categories": CATS}), field, value),
        "streaming": streamed(iou_type, [ann], [DT]),
    }
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
    # A bool inside a number field: `True` is 1 to Python, `true` is not a number to JSON.
    ("bbox holding a bool", "bbox", [True, 10, 20, 20], "reject", {}),
    ("bbox holding NaN", "bbox", [10, 10, math.nan, 20], None, {"json": NAN_TOKEN}),
    ("bbox='x'", "bbox", "x", "reject", {}),
    ("bbox None", "bbox", None, "accept", {}),
    ("area='x'", "area", "x", "reject", {}),
    ("area=True", "area", True, "reject", {}),
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
        "edited dict": lambda: edited(gt.load_res([DT]), field, value),
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
    ("score=inf", "score", math.inf, "accept", {"json": INF_TOKEN}),
    ("score=-1", "score", -1, "accept", {}),
    ("score='0.9'", "score", "0.9", "reject", {}),
    ("score='NaN'", "score", "NaN", "reject", {}),
    ("score=True", "score", True, "reject", {}),
    ("score None", "score", None, "reject", {}),
    ("score missing", "score", MISSING, "reject", {}),
    ("bbox of 3", "bbox", [10, 10, 20], "reject", {}),
    ("bbox holding a str", "bbox", ["a", 10, 20, 20], "reject", {}),
    ("bbox holding a bool", "bbox", [True, 10, 20, 20], "reject", {}),
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
    ("polygon holding a bool", [[True, 2.0, 3.0, 4.0, 5.0, 6.0]], "reject", {}),
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
        "image not_exhaustive_category_ids=[1.5]",
        {"images": [with_field(IMAGES[0], "not_exhaustive_category_ids", [1.5])]},
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


# ------------------------------------------------------- panoptic segments

SEGMENT = {"id": 1, "category_id": 1, "area": 4, "iscrowd": 0}


def segment_outcomes(tmp_path, field, value):
    """A ``segments_info`` entry in a panoptic dict, a panoptic JSON file, and
    ``primitives.match_segments``. The evaluators are only built: a bad record
    fails there, before any PNG is read."""
    seg = with_field(SEGMENT, field, value)
    dataset = {
        "images": [{"id": 1, "width": 2, "height": 2, "file_name": "1.jpg"}],
        "annotations": [{"image_id": 1, "file_name": "1.png", "segments_info": [seg]}],
        "categories": [{"id": 1, "name": "x", "isthing": 1}],
    }
    path = tmp_path / "panoptic.json"
    path.write_text(json.dumps(dataset))
    return {
        "dict": outcome(lambda: panoptic.PanopticEval(dataset, dataset), None),
        "json": outcome(lambda: panoptic.PanopticEval(str(path), str(path)), None),
        "match_segments": outcome(lambda: primitives.match_segments([seg], [seg], {(1, 1): 4}), None),
    }


SEGMENT_CASES = [
    ("id", 1.5, "reject"),
    ("id", -1, "reject"),
    ("id", "1", "reject"),
    ("id", True, "reject"),
    ("id", MISSING, "reject"),
    ("category_id", None, "reject"),
    ("category_id", MISSING, "reject"),
    ("area", -1, "reject"),
    ("area", "4", "reject"),
    ("iscrowd", "yes", "reject"),
    ("iscrowd", None, "reject"),
    ("iscrowd", 0.5, "reject"),
    ("bbox", "x", "reject"),
    ("bbox", [1, 2, 3], "reject"),
    ("bbox", [True, 2, 3, 4], "reject"),
    ("segmentation", 7, "reject"),
    ("bbox", None, "accept"),
    ("id", 1.0, "accept"),
    ("area", None, "accept"),
    ("area", MISSING, "accept"),
    ("iscrowd", MISSING, "accept"),
    ("iscrowd", 1.0, "accept"),
]


@pytest.mark.parametrize(
    "field, value, expect",
    SEGMENT_CASES,
    ids=[f"{f}={'missing' if v is MISSING else repr(v)}" for f, v, _ in SEGMENT_CASES],
)
def test_segment_record(tmp_path, field, value, expect):
    assert_agree(segment_outcomes(tmp_path, field, value), expect)


@pytest.mark.parametrize("value", [np.int64(1), np.uint32(1), np.float64(1.0)], ids=lambda v: type(v).__name__)
def test_a_numpy_scalar_is_an_id_in_a_segment(value):
    """A segment read from a dict follows the one id rule on both paths. A JSON file
    cannot hold a numpy scalar, so it is no path here."""
    seg = {"id": value, "category_id": value}
    dataset = {
        "images": [{"id": 1, "width": 2, "height": 2}],
        "annotations": [{"image_id": 1, "file_name": "1.png", "segments_info": [seg]}],
        "categories": [{"id": 1, "name": "x", "isthing": 1}],
    }
    panoptic.PanopticEval(dataset, dataset)
    assert primitives.match_segments([seg], [seg], {(1, 1): 4})["matched"] == [(1, 1, 1.0)]
    bad = {**dataset, "annotations": [{"image_id": 1, "segments_info": [{"id": 1.5, "category_id": 1}]}]}
    with pytest.raises(ValueError, match=r"^pred: annotations\[0\]: segments_info\[0\]: field 'id': "):
        panoptic.PanopticEval(dataset, bad)


def test_a_tuple_of_records_is_the_list_a_json_file_would_hold():
    """``json.dump`` writes a tuple as an array, so a dict holding one loads as the file does."""
    seg = {"id": 1, "category_id": 1}
    panoptic_dict = {
        "images": ({"id": 1, "width": 2, "height": 2},),
        "annotations": ({"image_id": 1, "file_name": "1.png", "segments_info": (seg,)},),
        "categories": ({"id": 1, "name": "x", "isthing": 1},),
    }
    panoptic.PanopticEval(panoptic_dict, panoptic_dict)
    coco = COCO({"images": tuple(IMAGES), "annotations": (GT,), "categories": tuple(CATS)})
    assert coco.getAnnIds() == [1]
    assert primitives.match_segments((seg,), (seg,), {(1, 1): 4})["matched"] == [(1, 1, 1.0)]


def test_an_isthing_of_none_is_absent_in_a_dict_as_null_is_in_a_file(tmp_path):
    """The JSON loader reads ``"isthing": null`` as no ``isthing``; a dict reads ``None`` the same."""
    dataset = {
        "images": [{"id": 1, "width": 2, "height": 2}],
        "annotations": [{"image_id": 1, "file_name": "1.png", "segments_info": []}],
        "categories": [{"id": 1, "name": "x", "isthing": None}],
    }
    path = tmp_path / "panoptic.json"
    path.write_text(json.dumps(dataset))
    for ev in (panoptic.PanopticEval(dataset, dataset), panoptic.PanopticEval(str(path), str(path))):
        assert ev.provenance() == "extension"
    detection = {"images": IMAGES, "annotations": [GT], "categories": [with_field(CATS[0], "isthing", None)]}
    assert COCO(detection).loadCats(1)[0].get("isthing") is None


# --------------------------------------------------------- tracking rows

TRACK_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
TRACK_GT = [1, 1, 10, 10, 20, 40, 1, 1, 1]
TRACK_PRED = [1, 5, 11, 10, 20, 40, 0.9, -1, -1, -1]  # a tracker file's 10 columns


def track_row(row, **changes):
    row = list(row)
    for column, value in changes.items():
        row[TRACK_COLUMNS.index(column)] = value
    return row


def track_outcomes(tmp_path, benchmark, gt, pred, num_frames=3):
    """One sequence's MOTChallenge rows through ``update`` as an array and as a list
    of rows, and through ``update_folders`` as the text files TrackEval reads."""

    def through(rows):
        def load():
            TrackingEval(benchmark).update("seq-a", rows(gt), rows(pred), num_frames=num_frames)

        return load

    def from_folders():
        seq = tmp_path / "gt" / "seq-a"
        (seq / "gt").mkdir(parents=True)
        (tmp_path / "pred").mkdir()
        text = lambda rows: "".join(",".join(str(v) for v in row) + "\n" for row in rows)  # noqa: E731
        (seq / "gt" / "gt.txt").write_text(text(gt))
        (seq / "seqinfo.ini").write_text(f"[Sequence]\nseqLength={num_frames}\n")
        (tmp_path / "pred" / "seq-a.txt").write_text(text(pred))
        TrackingEval(benchmark).update_folders(tmp_path / "gt", tmp_path / "pred")

    return {
        "array": outcome(through(lambda rows: np.array(rows, dtype=float)), None),
        "list": outcome(through(lambda rows: rows), None),
        "folders": outcome(from_folders, None),
    }


TRACK_CASES = [
    ("valid", "MOT17", [TRACK_GT], [TRACK_PRED], "accept"),
    ("fractional frame", None, [track_row(TRACK_GT, frame=1.5)], [TRACK_PRED], "reject"),
    ("frame 0", None, [TRACK_GT], [track_row(TRACK_PRED, frame=0)], "reject"),
    ("frame past the end", None, [track_row(TRACK_GT, frame=4)], [TRACK_PRED], "reject"),
    ("fractional id", None, [TRACK_GT], [track_row(TRACK_PRED, id=2.5)], "reject"),
    ("negative id", None, [track_row(TRACK_GT, id=-1)], [TRACK_PRED], "reject"),
    ("NaN box", None, [TRACK_GT], [track_row(TRACK_PRED, w=math.nan)], "reject"),
    ("infinite box", None, [track_row(TRACK_GT, x=math.inf)], [TRACK_PRED], "reject"),
    # Finite values whose far edge or corner area ((x + w) - x) * ((y + h) - y)
    # is not: two such boxes have an IoU of inf / inf, NaN, which TrackEval's
    # scipy assignment raises on. The last one's w * h is finite; its NaN
    # reached the assignment, which panicked.
    (
        "a box whose right edge overflows",
        None,
        [track_row(TRACK_GT, x=1e308, w=1e308)],
        [track_row(TRACK_PRED, x=1e308, w=1e308)],
        "reject",
    ),
    (
        "a box whose area overflows",
        None,
        [track_row(TRACK_GT, w=1e200, h=1e200)],
        [track_row(TRACK_PRED, w=1e200, h=1e200)],
        "reject",
    ),
    (
        "a box whose corner area overflows",
        None,
        [track_row(TRACK_GT, x=-2.040552280408471e307, y=0, w=8.988465674311579e307, h=2)],
        [track_row(TRACK_PRED, x=-2.040552280408471e307, y=0, w=8.988465674311579e307, h=2)],
        "reject",
    ),
    ("NaN ground-truth confidence", None, [track_row(TRACK_GT, conf=math.nan)], [TRACK_PRED], "reject"),
    ("NaN prediction confidence", None, [TRACK_GT], [track_row(TRACK_PRED, conf=math.nan)], "reject"),
    ("the same id twice in a frame", None, [TRACK_GT], [TRACK_PRED, track_row(TRACK_PRED, x=60)], "reject"),
    ("ground-truth class 14", "MOT17", [track_row(TRACK_GT, **{"class": 14})], [TRACK_PRED], "reject"),
    ("prediction class 2", "MOT17", [TRACK_GT], [track_row(TRACK_PRED, **{"class": 2})], "reject"),
    ("prediction class 2 under MOT15", "MOT15", [TRACK_GT], [track_row(TRACK_PRED, **{"class": 2})], "reject"),
    ("prediction class 2 without a benchmark", None, [TRACK_GT], [track_row(TRACK_PRED, **{"class": 2})], "accept"),
    ("7-column ground truth", "MOT17", [TRACK_GT[:7]], [TRACK_PRED], "reject"),
    ("6-column ground truth without a benchmark", None, [TRACK_GT[:6]], [TRACK_PRED], "accept"),
    ("6-column predictions", "MOT17", [TRACK_GT], [TRACK_PRED[:6]], "accept"),
    ("5-column predictions", None, [TRACK_GT], [TRACK_PRED[:5]], "reject"),
    ("no predictions", "MOT17", [TRACK_GT], [], "accept"),
]


@pytest.mark.parametrize("benchmark, gt, pred, expect", [c[1:] for c in TRACK_CASES], ids=[c[0] for c in TRACK_CASES])
def test_a_tracking_row(tmp_path, benchmark, gt, pred, expect):
    assert_agree(track_outcomes(tmp_path, benchmark, gt, pred), expect)


@pytest.mark.parametrize("num_frames", [2**64 - 1, 2**62, 10**17], ids=["wraps", "capacity overflow", "no allocation"])
def test_a_frame_count_no_memory_holds_is_a_value_error(tmp_path, num_frames):
    """The frame offsets of ``num_frames``, from the argument or ``seqLength``:
    one past ``usize`` wrapped to an empty table and panicked, and a size the
    system refuses aborted the interpreter."""
    assert_agree(track_outcomes(tmp_path, None, [TRACK_GT], [TRACK_PRED], num_frames=num_frames), "reject")


def test_a_bad_tracking_argument_is_a_value_error():
    row = [TRACK_GT]
    for call, message in (
        (lambda: TrackingEval("MOT99"), "unknown benchmark"),
        (lambda: TrackingEval(clear_threshold=1.5), "clear_threshold"),
        (lambda: TrackingEval(identity_threshold=math.nan), "identity_threshold"),
        (lambda: TrackingEval().update("s", row, row, num_frames=-1), "num_frames"),
        (lambda: TrackingEval().update("s", row, row, num_frames=1.5), "num_frames"),
        (lambda: TrackingEval().update("s", row, row, num_frames=True), "num_frames"),
        (lambda: TrackingEval().update("s", [TRACK_GT, [1]], row, num_frames=1), "rectangular"),
        # A bool is no frame or id in a row, as in any record; it read as 1.
        (lambda: TrackingEval().update("s", [track_row(TRACK_GT, frame=True)], row, num_frames=1), r"^gt\[0\]\[0\]"),
        (lambda: TrackingEval().update("s", row, np.ones((1, 7), bool), num_frames=1), r"^pred\[0\]\[0\]"),
    ):
        with pytest.raises(ValueError, match=message):
            call()
    ev = TrackingEval()
    ev.update("s", row, row, num_frames=1)
    with pytest.raises(ValueError, match="sequence s was already evaluated"):
        ev.update("s", row, row, num_frames=1)
    with pytest.raises(ValueError, match="itself"):
        ev.merge(ev)


def test_greedy_match_takes_the_ground_truth_count_from_its_flags_when_sim_is_empty():
    """``[]`` has no columns to count, so an image with no detections is still sized by
    its ground truths' flags, as ``np.zeros((0, G))`` would be."""
    m = primitives.greedy_match([], [], 0.5, iscrowd=[0, 1], gt_ignore=[False, False])
    assert m["gt_matches"].tolist() == [-1, -1] and m["dt_matches"].tolist() == []
    assert primitives.greedy_match([], [], 0.5, group_of=[1, 0, 0])["gt_matches"].tolist() == [-1, -1, -1]
    with pytest.raises(ValueError, match="gt_ignore and iscrowd must have the same length"):
        primitives.greedy_match([], [], 0.5, iscrowd=[0, 1], gt_ignore=[False])


def test_a_pair_with_no_union_is_one_error_on_both_paths(tmp_path):
    """The void id 0 listed as a segment with no stated area: its pixels are its area
    and its void overlap at once, so its union with pred 0 is empty. panopticapi
    divides by zero there; both paths here raise, naming the pair."""
    gt = np.ones((2, 2), dtype=np.uint32)
    gt[0] = 0
    segments = [{"id": 0, "category_id": 1}, {"id": 1, "category_id": 1}]
    paths = PanopticCase("no-union").add(1, gt, segments, gt, segments).write(tmp_path)
    message = "ground-truth segment 0 and predicted segment 0 have no union, so their IoU is not a number"
    with pytest.raises(ValueError, match=f"image 1: {message}"):
        panoptic.PanopticEval(paths["gt"][0], paths["pred"][0]).evaluate()
    with pytest.raises(ValueError, match=message):
        primitives.match_segments(segments, segments, primitives.segment_overlaps(gt, gt))


def test_bad_values_for_the_matchers_are_value_errors():
    seg = [{"id": 1, "category_id": 1}]
    # The control: well-formed, so accepted.
    primitives.match_segments(seg, seg, {(1, 1): 4, (0, 1): 4})
    for call, message in (
        (lambda: primitives.segment_overlaps([[0, 1]], [[0, 1, 2]]), "same shape"),
        (lambda: primitives.segment_overlaps([[0, -1]], [[0, 1]]), "gt pixel 1 has a negative label"),
        (lambda: primitives.segment_overlaps([[0, 1]], np.array([[0, 2**32]])), "pred pixel 1 has label 4294967296"),
        (lambda: primitives.match_segments(seg, [], {(1, 1): 4}), "segment 1 is in the label map but not in segments"),
        (
            lambda: primitives.match_segments(seg, seg, {(1, 0): 4}),
            "segment 1 is in segments_info but not in the label",
        ),
        (lambda: primitives.match_segments(seg, seg, {(1, 1): -4}), r"overlaps\[\(1, 1\)\]"),
        (lambda: primitives.match_segments(seg, seg, {(1, "a"): 4}), r"overlaps\[\(1, 'a'\)\]"),
        (lambda: primitives.match_segments(seg, seg, {1: 4}), "pair"),
        (lambda: primitives.match_segments(seg, seg, {(1, 1): 2**64 - 1, (0, 1): 5}), "more pixels"),
        # A stated area of 0 under 4 overlapping pixels: the union is 0.
        (lambda: primitives.match_segments([{"id": 1, "category_id": 1, "area": 0}], seg, {(1, 1): 4}), "not a number"),
        (lambda: primitives.greedy_match([[0.5, math.nan]], [0.9]), "sim contains NaN"),
        (lambda: primitives.greedy_match([[0.5]], [math.nan]), "scores contains NaN"),
        (lambda: primitives.greedy_match([[0.5]], [0.9], [0.5, math.nan]), "iou_thrs contains NaN"),
        (lambda: primitives.greedy_match([[0.5]], [0.9, 0.8]), "scores and sim's rows must have the same length"),
        (lambda: primitives.greedy_match([[0.5]], [0.9], iscrowd=[0, 1]), "iscrowd and sim's columns"),
        (lambda: primitives.greedy_match([[0.5]], [0.9], gt_ignore=[]), "gt_ignore and sim's columns"),
        (lambda: primitives.greedy_match([[0.5]], [0.9], group_of=[0, 1]), "group_of and sim's columns"),
        (lambda: primitives.greedy_match([], [], iscrowd=[0], group_of=[0, 1]), "group_of and iscrowd"),
        (lambda: primitives.greedy_match([[0.5]], [0.9], dt_ignore=[0, 1]), "dt_ignore and sim's rows"),
        (lambda: primitives.greedy_match([], [], dt_ignore=[True]), "dt_ignore and sim's rows"),
        (lambda: primitives.greedy_match([[0.5], [0.5, 0.5]], [0.9, 0.8]), "rectangular"),
    ):
        with pytest.raises(ValueError, match=message):
            call()


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
        lambda: primitives.segment_overlaps([[0.5]], [[0.5]]),
        lambda: primitives.segment_overlaps(["a"], ["b"]),
        lambda: primitives.match_segments([], [], [((1, 1), 4)]),
        lambda: primitives.match_segments(42, [], {}),
        lambda: primitives.greedy_match("x", []),
        lambda: primitives.greedy_match([[0.5]], [0.9], "0.5"),
        lambda: primitives.greedy_match([[0.5]], [0.9], iscrowd=42),
        lambda: primitives.greedy_match([[0.5]], [0.9], group_of=42),
        lambda: primitives.greedy_match([[0.5]], [0.9], dt_ignore=42),
        lambda: TrackingEval(17),
        lambda: TrackingEval().update("s", "rows", [], num_frames=1),
        lambda: TrackingEval().update("s", np.zeros(9), [], num_frames=1),
        lambda: TrackingEval().update("s", [], [], num_frames="3"),
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
        lambda: TrackingEval().update_folders(str(tmp_path / "no-such-folder"), str(tmp_path)),
    ):
        with pytest.raises(FileNotFoundError) as caught:
            call()
        assert str(tmp_path) in str(caught.value), caught.value


def test_a_file_that_does_not_parse_is_a_value_error(tmp_path):
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"images": [')
    not_png = tmp_path / "not.png"
    not_png.write_text("not a PNG")
    (tmp_path / "track" / "seq-a" / "gt").mkdir(parents=True)
    (tmp_path / "track" / "seq-a" / "gt" / "gt.txt").write_text("1,1,ten,10,20,40,1,1,1\n")
    (tmp_path / "track" / "seq-a" / "seqinfo.ini").write_text("[Sequence]\nseqLength=1\n")
    (tmp_path / "seq-a.txt").write_text("")
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    for call in (
        lambda: COCO(str(truncated)),
        lambda: gt.load_res(str(truncated)),
        lambda: panoptic.PanopticEval(str(truncated), str(truncated)),
        lambda: semantic.SemanticEval(3).update_png(str(not_png), str(not_png)),
        lambda: TrackingEval().update_folders(tmp_path / "track", tmp_path),
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
    # Python's step, as every PanopticEval reader names it, not Rust's `run()`.
    with pytest.raises(RuntimeError, match=r"^evaluate\(\) must be called before results\(\)"):
        panoptic.PanopticEval(gt, gt).results()
    with pytest.raises(RuntimeError, match="update"):
        TrackingEval().results()


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


def test_a_number_list_is_a_sequence_not_any_iterable():
    """A set has no order and a dict iterates its keys, so neither is a box;
    a JSON file cannot hold them, and the dict path refuses them too."""
    for value in ({10, 20, 30, 40}, {10: 0, 20: 0, 30: 0, 40: 0}, (x for x in (10, 20, 30, 40))):
        with pytest.raises(ValueError, match="field 'bbox': expected a sequence"):
            COCO({"images": IMAGES, "annotations": [with_field(GT, "bbox", value)], "categories": CATS})
    for value in ([10, 20, 30, 40], (10, 20, 30, 40), np.array([10, 20, 30, 40])):
        COCO({"images": IMAGES, "annotations": [with_field(GT, "bbox", value)], "categories": CATS})


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


def test_a_whole_float_is_an_id_in_an_lvis_list(tmp_path):
    """``neg_category_ids=[1.0]`` is ``[1]`` from a dict and a file alike, by the one id rule."""
    dataset = {"images": [with_field(IMAGES[0], "neg_category_ids", [1.0])], "annotations": [GT], "categories": CATS}
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(dataset))
    for coco in (COCO(dataset), COCO(str(path))):
        assert coco.loadImgs(1)[0]["neg_category_ids"] == [1]


def test_an_argument_no_matrix_can_hold_is_a_value_error():
    """These used to panic, or wrap and answer: a class count whose square
    overflows, counts that sum past 64 bits, costs with no finite assignment."""
    from hotcoco import metrics

    inf = math.inf
    for cost, maximize in [([[inf, inf], [inf, inf]], False), ([[-inf, 1], [1, 2]], False), ([[inf, 1], [1, 2]], True)]:
        with pytest.raises(ValueError, match="finite"):
            primitives.lsap(cost, maximize=maximize)
    rows, cols = primitives.lsap([[inf, 1], [1, inf]])
    assert (rows.tolist(), cols.tolist()) == ([0, 1], [1, 0])

    # 2**28 classes fit the arithmetic, but no system grants the ~2**59 bytes:
    # that allocation aborted the interpreter before it was checked.
    for too_many in (2**28, 2**32, 2**64 - 1):
        # The error names the argument as the caller spelled it.
        with pytest.raises(ValueError, match=f"^num_classes: {too_many} is too many classes"):
            metrics.confusion_matrix([0], [0], too_many)
        with pytest.raises(ValueError, match=f"^num_classes: {too_many} is too many classes"):
            metrics.pixel_confusion([0], [0], too_many)
        with pytest.raises(ValueError, match=f"^classes: {too_many} is too many classes"):
            semantic.SemanticEval(too_many)
    with pytest.raises(ValueError, match="more pixels"):
        metrics.mean_iou(np.full((2, 2), 2**62, dtype=np.uint64))

    ev = semantic.SemanticEval(3)
    with pytest.raises(ValueError, match="itself"):
        ev.merge(ev)


def test_an_empty_list_is_an_empty_label_map():
    """``numpy.asarray([])`` is float64, so the list spelling failed only when
    empty: "a label map holds integer class ids"."""
    from hotcoco import metrics

    assert primitives.segment_overlaps([], []) == {}
    assert metrics.pixel_confusion([], [], 2).tolist() == [[0, 0], [0, 0]]
    with pytest.raises(TypeError, match="integer"):
        metrics.pixel_confusion([[0.5]], [[0.5]], 2)
    with pytest.raises(TypeError, match="integer"):
        metrics.pixel_confusion(np.zeros((0, 2)), np.zeros((0, 2)), 2)


def test_a_panoptic_dict_with_no_annotations_is_still_panoptic():
    """With no annotation to carry ``segments_info`` the dict read as
    detection-style, and a folder beside it was refused."""
    empty = {
        "images": [{"id": 1, "height": 4, "width": 4, "file_name": "a.png"}],
        "annotations": [],
        "categories": [{"id": 1, "name": "a", "isthing": 1}],
    }
    ev = panoptic.PanopticEval(empty, empty, gt_folder="g", pred_folder="p")
    ev.evaluate()
    assert ev.eval_imgs == []
    ev = panoptic.PanopticEval(empty, empty)
    ev.evaluate()
    assert ev.eval_imgs == []


@pytest.mark.parametrize(
    "rle, match",
    [
        ({"size": [-5, 4], "counts": b"63204"}, "size"),
        ({"size": [2**32, 4], "counts": b"63204"}, "size"),
        ({"size": [5, 4], "counts": [6, -3, 2]}, "counts"),
        ({"size": [5, 4], "counts": [2**32]}, "counts"),
        ({"size": [2**31, 2**31], "counts": b"63204"}, "pixels"),
    ],
)
def test_an_rle_number_out_of_range_is_a_value_error(rle, match):
    """Each was ``OverflowError`` or a ``TypeError`` saying a list of ints is
    not a list of ints, and the last aborted the interpreter in ``decode``."""
    for read in (mask.decode, mask.area, mask.toBbox, lambda r: mask.merge([r]), lambda r: mask.iou([r], [r], [0])):
        with pytest.raises(ValueError, match=match):
            read(rle)


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.split(val_frac=-1),
        lambda c: c.split(val_frac=math.nan),
        lambda c: c.split(val_frac=0.9, test_frac=0.9),
        lambda c: c.split(test_frac=1.5),
        lambda c: c.sample(frac=2),
        lambda c: c.sample(frac=math.nan),
    ],
)
def test_a_fraction_outside_zero_to_one_is_a_value_error(call):
    """Each was clamped without a word: ``val_frac=-1`` an empty validation
    set, ``0.9 + 0.9`` a test set of a tenth, ``frac=nan`` an empty sample."""
    coco = COCO(
        {"images": [{"id": i, "width": 9, "height": 9} for i in range(1, 11)], "annotations": [], "categories": CATS}
    )
    with pytest.raises(ValueError, match="frac"):
        call(coco)
    train, val, test = coco.split(val_frac=0.5, test_frac=0.5)
    assert [len(x.dataset["images"]) for x in (train, val, test)] == [0, 5, 5]
    assert len(coco.sample(frac=1.0).dataset["images"]) == 10


@pytest.mark.parametrize("ids", [-1, [-1], [1, 2**70], {-1}])
def test_an_int_that_is_no_id_is_a_value_error(ids):
    """It was ``TypeError: expected an int or an iterable of ints, got list``
    for a list of ints."""
    coco = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    for lookup in (coco.getAnnIds, coco.loadAnns, coco.getImgIds, coco.loadImgs):
        with pytest.raises(ValueError, match="an id is an integer"):
            lookup(ids)
    with pytest.raises(TypeError, match="iterable of ints"):
        coco.loadAnns(["a"])


def test_an_analysis_argument_that_is_no_number_is_a_value_error():
    """NaN slipped past each range check: ``compare`` panicked on it,
    ``f_scores`` returned a key ``FNaN``, and ``calibration(n_bins=0)`` an
    ``ece`` of 0.0."""
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    ev = COCOeval(gt, gt.loadRes([DT]), "bbox")
    ev.params.maxDets = [1, 10, 100]
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        ev.run()
    with pytest.raises(ValueError, match="confidence"):
        hotcoco.compare(ev, ev, n_bootstrap=10, confidence=math.nan)
    with pytest.raises(ValueError, match="n_bootstrap"):
        hotcoco.compare(ev, ev, n_bootstrap=2**63)
    for beta in (math.nan, 0, -1, math.inf):
        with pytest.raises(ValueError, match="beta"):
            ev.f_scores(beta)
    with pytest.raises(ValueError, match="n_bins"):
        ev.calibration(n_bins=0)
    with pytest.raises(ValueError, match="iou_thr"):
        ev.confusion_matrix(iou_thr=math.nan)


def test_a_converter_names_the_file_it_could_not_read(tmp_path):
    """The message was the bare OS error, and a file that is not CVAT XML —
    a COCO JSON, an empty file — imported as an empty dataset."""
    nope = tmp_path / "nope"
    for read in (COCO.from_voc, COCO.from_cvat, COCO.from_dota, COCO.from_oid):
        with pytest.raises(FileNotFoundError, match="nope"):
            read(str(nope))
    not_cvat = tmp_path / "x.xml"
    for content in ("", "hello", '{"images": []}'):
        not_cvat.write_text(content)
        with pytest.raises(ValueError, match="x.xml"):
            COCO.from_cvat(str(not_cvat))
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "bad.txt").write_bytes(b"\xff\xfe 1 2 3")
    with pytest.raises(ValueError, match="bad.txt"):
        COCO.from_dota(str(tmp_path / "labels"))


def test_a_tracking_file_that_is_not_utf8_is_a_value_error(tmp_path):
    """Bad content, as a line that is not numbers is, and as the converters and
    the JSON loader raise it: `read_to_string`'s `InvalidData` was kept as an I/O
    error, a plain `OSError`."""
    seq = tmp_path / "gt" / "seq-a"
    (seq / "gt").mkdir(parents=True)
    (tmp_path / "pred").mkdir()
    (seq / "gt" / "gt.txt").write_bytes(b"1,1,0,0,10,10,1,1,1\n2,1,0,0,10,10,1,1,\xff\n")
    (seq / "seqinfo.ini").write_text("[Sequence]\nseqLength=3\n")
    (tmp_path / "pred" / "seq-a.txt").write_text("1,1,0,0,10,10,0.9\n")
    with pytest.raises(ValueError, match="gt.txt"):
        TrackingEval().update_folders(tmp_path / "gt", tmp_path / "pred")


@pytest.mark.parametrize("shape", [(2**32 + 5, 0), (0, 2**32 + 7), (2**32 + 5, 0, 2)])
def test_a_mask_side_no_rle_size_holds_is_a_value_error(shape):
    """An RLE ``size`` is two 32-bit sides. Cast, a longer side came back as the
    side modulo 2**32 on a mask of no pixels, where pycocotools keeps it."""
    from hotcoco import mask as hm

    with pytest.raises(ValueError, match=r"2\*\*32 - 1"):
        hm.encode(np.zeros(shape, np.uint8, order="F"))


def test_the_tracking_error_raised_is_the_first_failing_sequences(tmp_path):
    """Sequences are evaluated in parallel, but the error raised is the first
    failing sequence's in name order, every run: collected straight into a
    result, it was whichever worker failed first, and its kind varied."""
    gt_rows = "".join(f"{f},1,0,0,10,10,1,1,1\n" for f in range(1, 30_001))
    for name, gt_text in [("seq-a", gt_rows), ("seq-b", "1,1,0,0,10,10,1,1,1\nx\n")]:
        seq = tmp_path / "gt" / name
        (seq / "gt").mkdir(parents=True)
        (seq / "gt" / "gt.txt").write_text(gt_text)
        (seq / "seqinfo.ini").write_text("[Sequence]\nseqLength=30000\n")
    (tmp_path / "pred").mkdir()
    (tmp_path / "pred" / "seq-b.txt").write_text("1,1,0,0,10,10,0.9\n")
    raised = set()
    for _ in range(30):
        with pytest.raises(OSError) as caught:
            TrackingEval().update_folders(tmp_path / "gt", tmp_path / "pred")
        raised.add((type(caught.value), str(caught.value)))
    ((kind, message),) = raised
    assert kind is FileNotFoundError and "seq-a" in message


@pytest.mark.parametrize("score", [MISSING, None, math.nan], ids=["missing", "None", "NaN"])
def test_confusion_matrix_checks_scores_as_evaluate_does(score):
    """`confusion_matrix()` runs without `evaluate()`, so it never reached the
    score check and ranked a detection with no score, or a NaN one, as 0."""
    gt = COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})
    dt = COCO()
    dt.dataset = {"images": IMAGES, "annotations": [{**with_field(DT, "score", score), "id": 1}], "categories": CATS}
    with pytest.raises(ValueError, match="score"):
        COCOeval(gt, dt, "bbox").confusion_matrix()
