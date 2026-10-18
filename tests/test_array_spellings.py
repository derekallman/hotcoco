"""One array, every in-memory spelling, one answer.

The binding reads an array argument in place when it can and copies it when it
must. Each in-place branch is a fast path, and a fast path skips whatever check
the copy did for free. Every one of them has shipped a bug that way: a
Fortran-order buffer borrowed as row-major, a ``frombuffer`` view at an odd
offset borrowed as a slice of ``uint16``, a ``uint64`` cell of 2^63 that the
``astype(int64)`` copy had refused as negative. A review finds one such corner
per pass; this file is the gate that finds them all at once.

Every integer dtype, in every memory layout numpy can produce, through every
entry point that reads the array, must give exactly what a fresh C-contiguous
``int64`` copy of the same values gives. The readers under test live in
``crates/hotcoco-pyo3/src/convert.rs``: ``label_array`` (label maps),
``u64_matrix_arg`` (a confusion matrix), ``u64_vec`` (an id column),
``numeric_array`` (a float column, a box table, a detection table), and the
numpy scalars that ``extract_int`` reads inside a record.

``fuzz_dropin.py`` does this for a dataset's spellings; this is the array
half, small enough to run in CI. Add a row here for every spelling a review
finds, before fixing it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hotcoco import COCO, metrics, semantic

INT_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
FLOAT_DTYPES = [np.float16, np.float32, np.float64]
LAYOUTS = ["contiguous", "fortran", "strided", "transposed", "reversed", "misaligned"]

RNG = np.random.default_rng(7)
GT = RNG.integers(0, 3, (6, 10), dtype=np.int64)
PRED = RNG.integers(0, 3, (6, 10), dtype=np.int64)
IGNORE = 100  # fits every integer dtype, int8 included
GT[0, :3] = IGNORE  # ignored pixels on the ground-truth side only
CONFUSION = RNG.integers(0, 50, (4, 4), dtype=np.int64)

IMAGES = [{"id": i, "width": 100, "height": 100, "file_name": f"{i}.jpg"} for i in (1, 2, 3)]
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}]
IMAGE_IDS = np.array([1, 2, 2, 3], dtype=np.int64)
CATEGORY_IDS = np.array([1, 2, 1, 1], dtype=np.int64)
BOXES = np.array([[10, 10, 30, 30], [50, 50, 20, 20], [0, 0, 40, 40], [5, 5, 10, 10]], dtype=np.int64)
AREAS = np.array([90, 40, 120, 10], dtype=np.int64)  # fits every integer dtype
SCORES = np.array([0.5, 0.25, 0.75, 0.125], dtype=np.float64)  # exact in every float dtype


def spelled(values, dtype, layout):
    """``values`` as a ``dtype`` array in ``layout``, equal to ``values`` element for element.

    Each layout is a view that numpy reports as something other than a fresh array:
    column-major, a step through a wider buffer, the transpose of a transpose,
    negative strides, or a byte offset that no element of ``dtype`` can start at.
    """
    a = np.ascontiguousarray(values, dtype=dtype)
    if layout == "contiguous":
        return a
    if layout == "fortran":
        return np.asfortranarray(a)
    if layout == "strided":
        wide = np.zeros(a.shape[:-1] + (2 * a.shape[-1],), dtype=dtype)
        wide[..., ::2] = a
        return wide[..., ::2]
    if layout == "transposed":
        return np.ascontiguousarray(a.T).T
    if layout == "reversed":
        flipped = np.ascontiguousarray(a[(slice(None, None, -1),) * a.ndim])
        return flipped[(slice(None, None, -1),) * a.ndim]
    assert layout == "misaligned"
    off = np.frombuffer(b"\0" + a.tobytes(), dtype=dtype, offset=1).reshape(a.shape)
    assert off.flags.c_contiguous and off.flags.aligned == (a.itemsize == 1)
    return off


def test_a_spelling_is_a_view_in_the_named_layout():
    """The matrix is only as strong as its spellings: each must leave the plain path."""
    for layout in LAYOUTS[1:]:
        a = spelled(GT, np.int64, layout)
        assert np.array_equal(a, GT), layout
        assert not (a.flags.c_contiguous and a.flags.aligned), layout
    a = spelled(GT, np.int64, "fortran")
    assert a.flags.f_contiguous and a.ndim == 2


# --- label maps: SemanticEval.update and metrics.pixel_confusion -------------


def reference_confusion():
    ev = semantic.SemanticEval(3, ignore_index=IGNORE)
    ev.update(GT.copy(), PRED.copy())
    return ev.confusion()


def label_pair(dtype, layout, side, gt=GT, pred=PRED):
    """One side spelled, the other a plain copy.

    Spelling both sides the same way hides the bug this file exists for: two
    Fortran buffers read column-major still pair pixel ``k`` with pixel ``k``, and
    the counts come out right. Against a row-major partner they do not.
    """
    if side == "gt":
        return spelled(gt, dtype, layout), np.ascontiguousarray(pred)
    return np.ascontiguousarray(gt), spelled(pred, dtype, layout)


@pytest.mark.parametrize("side", ["gt", "pred"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_label_maps_count_the_same_in_every_spelling(dtype, layout, side):
    gt, pred = label_pair(dtype, layout, side)
    ev = semantic.SemanticEval(3, ignore_index=IGNORE)
    ev.update(gt, pred)
    assert np.array_equal(ev.confusion(), reference_confusion())
    assert np.array_equal(metrics.pixel_confusion(gt, pred, 3, ignore_index=IGNORE), reference_confusion())


@pytest.mark.parametrize("side", ["gt", "pred"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_batch_counts_the_same_in_every_spelling(dtype, layout, side):
    gt, pred = label_pair(dtype, layout, side, np.stack([GT, GT]), np.stack([PRED, PRED]))
    ev = semantic.SemanticEval(3, ignore_index=IGNORE)
    ev.update(gt, pred)
    assert np.array_equal(ev.confusion(), 2 * reference_confusion())


# --- a confusion matrix: metrics.mean_iou -------------------------------------


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_confusion_matrix_scores_the_same_in_every_spelling(dtype, layout):
    assert metrics.mean_iou(spelled(CONFUSION, dtype, layout)) == metrics.mean_iou(CONFUSION.tolist())


# --- columns: COCO.from_arrays and update_anns ---------------------------------

COLUMN_LAYOUTS = [layout for layout in LAYOUTS if layout not in ("fortran", "transposed")] + ["column"]


def spelled_column(values, dtype, layout):
    """A 1-D spelling; ``column`` is one column of a row-major table, the pandas case."""
    if layout == "column":
        table = np.zeros((len(values), 3), dtype=dtype)
        table[:, 1] = values
        return table[:, 1]
    return spelled(values, dtype, layout)


def anns_of(coco):
    return coco.loadAnns(coco.getAnnIds())


REFERENCE = COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS.tolist(), CATEGORY_IDS.tolist(), BOXES.tolist())


@pytest.mark.parametrize("layout", COLUMN_LAYOUTS)
@pytest.mark.parametrize("dtype", INT_DTYPES + FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_an_id_column_reads_the_same_in_every_spelling(dtype, layout):
    coco = COCO.from_arrays(
        IMAGES,
        CATEGORIES,
        spelled_column(IMAGE_IDS, dtype, layout),
        spelled_column(CATEGORY_IDS, dtype, layout),
        BOXES.tolist(),
        ids=spelled_column([7, 8, 9, 10], dtype, layout),
        area=spelled_column(AREAS, dtype, layout),
    )
    assert [a["image_id"] for a in anns_of(coco)] == IMAGE_IDS.tolist()
    assert [a["category_id"] for a in anns_of(coco)] == CATEGORY_IDS.tolist()
    assert coco.getAnnIds() == [7, 8, 9, 10]
    assert [a["area"] for a in anns_of(coco)] == AREAS.tolist()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", INT_DTYPES + FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_box_table_reads_the_same_in_every_spelling(dtype, layout):
    coco = COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, spelled(BOXES, dtype, layout))
    assert [a["bbox"] for a in anns_of(coco)] == [a["bbox"] for a in anns_of(REFERENCE)]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_detection_table_reads_the_same_in_every_spelling(dtype, layout):
    rows = np.column_stack([IMAGE_IDS, BOXES, SCORES, CATEGORY_IDS]).astype(np.float64)
    want = anns_of(REFERENCE.loadRes(rows.copy()))
    got = anns_of(REFERENCE.loadRes(spelled(rows, dtype, layout)))
    assert [(a["image_id"], a["bbox"], a["score"], a["category_id"]) for a in got] == [
        (a["image_id"], a["bbox"], a["score"], a["category_id"]) for a in want
    ]


@pytest.mark.parametrize("layout", COLUMN_LAYOUTS)
@pytest.mark.parametrize("dtype", INT_DTYPES + FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_update_anns_columns_read_the_same_in_every_spelling(dtype, layout):
    coco = COCO.from_arrays(IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES)
    coco.update_anns(ids=spelled_column([1, 2, 3, 4], dtype, layout), area=spelled_column(AREAS, dtype, layout))
    assert [a["area"] for a in anns_of(coco)] == AREAS.tolist()


# --- scalars inside a record: extract_int ---------------------------------------

SCALARS = [
    *[(np.dtype(d).name, np.dtype(d).type(1)) for d in INT_DTYPES],
    *[(np.dtype(d).name, np.dtype(d).type(1.0)) for d in FLOAT_DTYPES],
    ("0-d", np.array(1)),
    ("float", 1.0),
]


@pytest.mark.parametrize("value", [v for _, v in SCALARS], ids=[name for name, _ in SCALARS])
def test_a_numpy_scalar_is_an_id_in_a_record(value):
    coco = COCO(
        {
            "images": [{"id": value, "width": 10, "height": 10}],
            "categories": [{"id": value, "name": "x"}],
            "annotations": [{"id": value, "image_id": value, "category_id": value, "bbox": [0, 0, 1, 1]}],
        }
    )
    assert coco.getImgIds() == [1] and coco.getCatIds() == [1] and coco.getAnnIds() == [1]


@pytest.mark.parametrize("value", [np.uint8(IGNORE), np.int64(IGNORE), np.array(IGNORE)], ids=["uint8", "int64", "0-d"])
def test_a_numpy_scalar_option_reads_as_the_int(value):
    ev = semantic.SemanticEval(np.int32(3), ignore_index=value)
    ev.update(GT, PRED)
    assert np.array_equal(ev.confusion(), reference_confusion())
