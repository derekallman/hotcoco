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
``int64`` copy of the same values gives. Every array goes through one reader,
``numpy_array`` in ``crates/hotcoco-pyo3/src/convert.rs``, and this file
reaches it through each caller: ``label_array`` (label maps, for semantic
counts and ``primitives.segment_overlaps``), ``u64_matrix_arg`` (a confusion
matrix), ``u64_vec`` (an id column), ``f64_vec`` and ``f64_matrix_arg`` (a
float column, a box table, the similarity matrix, scores, and thresholds of
``primitives.greedy_match``), ``anns_from_array`` (a detection table),
``flag_vec`` (``iscrowd``, ``gt_ignore``, ``dt_ignore``), ``bool_vec`` (``matched``), and the
numpy scalars that ``extract_int`` reads inside a record.

``fuzz_dropin.py`` does this for a dataset's spellings; this is the array
half, small enough to run in CI. Add a row here for every spelling a review
finds, before fixing it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hotcoco import COCO, metrics, primitives, semantic

INT_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
FLOAT_DTYPES = [np.float16, np.float32, np.float64]
LAYOUTS = ["contiguous", "fortran", "strided", "transposed", "reversed", "misaligned", "bigendian"]

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
    negative strides, a byte offset that no element of ``dtype`` can start at, or
    the other byte order, which is the right dtype to Python and the wrong one to
    a Rust slice.
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
    if layout == "misaligned":
        off = np.frombuffer(b"\0" + a.tobytes(), dtype=dtype, offset=1).reshape(a.shape)
        assert off.flags.c_contiguous and off.flags.aligned == (a.itemsize == 1)
        return off
    assert layout == "bigendian"
    swapped = a.astype(a.dtype.newbyteorder(">"))
    assert swapped.dtype.isnative == (a.itemsize == 1)
    return swapped


def test_a_spelling_is_a_view_in_the_named_layout():
    """The matrix is only as strong as its spellings: each must leave the plain path."""
    for layout in LAYOUTS[1:]:
        a = spelled(GT, np.int64, layout)
        assert np.array_equal(a, GT), layout
        assert not (a.flags.c_contiguous and a.flags.aligned and a.dtype.isnative), layout
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


def reference_overlaps(gt=GT, pred=PRED):
    return primitives.segment_overlaps(gt.tolist(), pred.tolist())


@pytest.mark.parametrize("side", ["gt", "pred"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_label_maps_count_the_same_in_every_spelling(dtype, layout, side):
    gt, pred = label_pair(dtype, layout, side)
    ev = semantic.SemanticEval(3, ignore_index=IGNORE)
    ev.update(gt, pred)
    assert np.array_equal(ev.confusion(), reference_confusion())
    assert np.array_equal(metrics.pixel_confusion(gt, pred, 3, ignore_index=IGNORE), reference_confusion())
    assert primitives.segment_overlaps(gt, pred) == reference_overlaps()


@pytest.mark.parametrize("side", ["gt", "pred"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_bool_label_map_counts_the_same_in_every_spelling(layout, side):
    """A bool map is a two-class map; it takes the ``int64`` copy, never a borrow."""
    gt, pred = (GT % 2).astype(bool), (PRED % 2).astype(bool)
    want = semantic.SemanticEval(2, ignore_index=None)
    want.update(gt.astype(np.int64), pred.astype(np.int64))
    spelled_pair = label_pair(np.bool_, layout, side, gt, pred)
    ev = semantic.SemanticEval(2, ignore_index=None)
    ev.update(*spelled_pair)
    assert np.array_equal(ev.confusion(), want.confusion())
    assert primitives.segment_overlaps(*spelled_pair) == reference_overlaps(gt.astype(np.int64), pred.astype(np.int64))


@pytest.mark.parametrize("side", ["gt", "pred"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_batch_counts_the_same_in_every_spelling(dtype, layout, side):
    gt, pred = label_pair(dtype, layout, side, np.stack([GT, GT]), np.stack([PRED, PRED]))
    ev = semantic.SemanticEval(3, ignore_index=IGNORE)
    ev.update(gt, pred)
    assert np.array_equal(ev.confusion(), 2 * reference_confusion())
    assert primitives.segment_overlaps(gt, pred) == {k: 2 * n for k, n in reference_overlaps().items()}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [np.int32, np.int64], ids=lambda d: np.dtype(d).name)
def test_a_bad_label_is_named_by_its_row_major_position(dtype, layout):
    """The pixel an error names is ``gt.flat``'s index, whatever order the buffer is in,
    for each signed width read in place."""
    bad = GT.copy()
    bad[2, 3] = -1
    want = f"gt pixel {np.ravel_multi_index((2, 3), GT.shape)} has a negative label"
    with pytest.raises(ValueError, match=want):
        primitives.segment_overlaps(spelled(bad, dtype, layout), PRED)
    with pytest.raises(ValueError, match=want):
        metrics.pixel_confusion(spelled(bad, dtype, layout), PRED, 3, ignore_index=IGNORE)


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


FLAGS = np.array([0, 1, 1, 0], dtype=np.int64)
MATCHED = FLAGS.astype(bool).tolist()


@pytest.mark.parametrize("layout", COLUMN_LAYOUTS)
@pytest.mark.parametrize("dtype", [np.bool_] + INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_flag_column_reads_the_same_in_every_spelling(dtype, layout):
    coco = COCO.from_arrays(
        IMAGES, CATEGORIES, IMAGE_IDS, CATEGORY_IDS, BOXES, iscrowd=spelled_column(FLAGS, dtype, layout)
    )
    assert [a["iscrowd"] for a in anns_of(coco)] == FLAGS.tolist()


@pytest.mark.parametrize("layout", COLUMN_LAYOUTS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_metric_columns_read_the_same_in_every_spelling(dtype, layout):
    scores = spelled_column(SCORES, dtype, layout)
    matched = spelled_column(FLAGS, np.bool_, layout)
    assert metrics.average_precision(scores, matched, 2) == metrics.average_precision(SCORES.tolist(), MATCHED, 2)
    assert metrics.calibration_error(scores, matched) == metrics.calibration_error(SCORES.tolist(), MATCHED)


# --- greedy_match: a similarity matrix, scores, thresholds, and flags ------------

# Multiples of 1/8, exact in every float dtype. Chosen so that each threshold
# matches differently, and so does the transpose, which is what a column-major
# buffer read row-major would be.
SIM = np.array([[1.0, 0.75, 0.25, 0.0], [0.0, 0.75, 0.125, 0.5], [1.0, 1.0, 0.25, 0.0], [0.5, 0.875, 0.625, 0.0]])
THRESHOLDS = np.array([0.5, 0.625, 0.75])
DT_FLAGS = np.roll(FLAGS, 2)  # a pattern neither ground-truth flag column has


def greedy_reference(**flags):
    return primitives.greedy_match(SIM.tolist(), SCORES.tolist(), THRESHOLDS.tolist(), **flags)


def same_matches(got, want):
    return got.keys() == want.keys() and all(np.array_equal(got[k], want[k]) for k in want)


def test_the_similarity_matrix_tells_its_spellings_apart():
    assert not same_matches(primitives.greedy_match(SIM.T.copy(), SCORES, THRESHOLDS), greedy_reference())
    assert len({tuple(row) for row in greedy_reference()["dt_matches"]}) == len(THRESHOLDS)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_a_similarity_matrix_matches_the_same_in_every_spelling(dtype, layout):
    got = primitives.greedy_match(spelled(SIM, dtype, layout), SCORES, THRESHOLDS)
    assert same_matches(got, greedy_reference())


@pytest.mark.parametrize("layout", COLUMN_LAYOUTS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_greedy_match_columns_read_the_same_in_every_spelling(dtype, layout):
    got = primitives.greedy_match(SIM, spelled_column(SCORES, dtype, layout), spelled_column(THRESHOLDS, dtype, layout))
    assert same_matches(got, greedy_reference())


@pytest.mark.parametrize("layout", COLUMN_LAYOUTS)
@pytest.mark.parametrize("dtype", [np.bool_] + INT_DTYPES, ids=lambda d: np.dtype(d).name)
def test_greedy_match_flags_read_the_same_in_every_spelling(dtype, layout):
    crowd = np.roll(FLAGS, 1)
    want = greedy_reference(gt_ignore=FLAGS.tolist(), iscrowd=crowd.tolist(), dt_ignore=DT_FLAGS.tolist())
    got = primitives.greedy_match(
        SIM,
        SCORES,
        THRESHOLDS,
        gt_ignore=spelled_column(FLAGS, dtype, layout),
        iscrowd=spelled_column(crowd, dtype, layout),
        dt_ignore=spelled_column(DT_FLAGS, dtype, layout),
    )
    assert same_matches(got, want)
    assert not same_matches(want, greedy_reference()), "the flags must change the matches"
    without_seed = greedy_reference(gt_ignore=FLAGS.tolist(), iscrowd=crowd.tolist())
    assert not same_matches(want, without_seed), "the detection flags must change dt_ignore"


def degenerate(shape, dtype, width=5):
    """A zero-size array, or one with a length-one axis, in ``dtype``. The last is a
    one-row view with an odd row stride, which numpy still flags C-contiguous."""
    if shape == "odd stride":
        a = np.zeros((2, width), dtype)[::2]
        assert a.flags.c_contiguous and a.strides[0] != width * a.itemsize
        return a
    return np.zeros(shape, dtype)


@pytest.mark.parametrize("shape", [(0, 5), (1, 5), (5, 1), (1, 1), "odd stride"], ids=str)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint64, np.int32, np.int64], ids=lambda d: np.dtype(d).name)
def test_a_degenerate_shape_is_one_slice(shape, dtype):
    """Each width read in place trusts that what numpy calls C-contiguous is one slice
    to ndarray, length-one axes and all. The zero-size rows take the copy path instead
    (numpy flags every empty array aligned, so the reader does not borrow one) and pin
    that an empty map counts as nothing."""
    gt = degenerate(shape, dtype)
    ev = semantic.SemanticEval(3)
    ev.update(gt, gt)
    assert ev.confusion()[0, 0] == gt.size
    assert primitives.segment_overlaps(gt, gt) == ({(0, 0): gt.size} if gt.size else {})


@pytest.mark.parametrize("shape", [(0, 6), "odd stride"], ids=str)
def test_a_degenerate_detection_table_is_one_slice(shape):
    rows = degenerate(shape, np.float64, width=6)
    rows[...] = [1, 0, 0, 1, 1, 0.5]
    assert len(anns_of(REFERENCE.loadRes(rows))) == len(rows)


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
