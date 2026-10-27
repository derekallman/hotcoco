"""Semantic segmentation parity: hotcoco vs mmsegmentation, torchmetrics, detectron2.

The fixture (`fixtures/semantic_expected.json`) holds label-map pairs and what the
three references computed for them; `scripts/gen_semantic_fixtures.py` regenerates
it from the pinned sources, so this needs no torch and no network. The Rust suite
(`crates/hotcoco/tests/semantic.rs`) asserts the numbers through the core API; this
file asserts them through the bindings and the two input paths the bindings add:

- `SemanticEval.update` on numpy arrays of several dtypes, one image per call;
- `update_folders` on PNG files written by Pillow in the modes the datasets use:
  ``L`` (8-bit grayscale), ``P`` (palette, VOC), ``I;16`` (16-bit grayscale);
- `metrics.pixel_confusion` + `metrics.mean_iou`, the functional spelling.

mmseg is the reference hotcoco follows, so it is compared to 1e-9 on every number;
torchmetrics reduces in float32 and is compared to 1e-6; detectron2's mIoU must
*differ* on the cases with a predicted-but-absent class, which is what makes the
documented convention observable.

    uv run pytest tests/test_parity_semantic.py -v
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
from helpers import FIXTURES_DIR
from hotcoco import metrics, semantic
from PIL import Image

FIXTURE = json.loads((FIXTURES_DIR / "semantic_expected.json").read_text())
CASES = FIXTURE["cases"]
TOL = 1e-9


def evaluator(case, dtype=np.int64):
    ev = semantic.SemanticEval(
        case["num_classes"], ignore_index=case["ignore_index"], reduce_zero_label=case["reduce_zero_label"]
    )
    for gt, pred in zip(case["gt"], case["pred"]):
        ev.update(np.array(gt, dtype=dtype), np.array(pred, dtype=dtype))
    return ev


def close(got, want, tol=TOL):
    """``None`` in the fixture is a reference NaN: hotcoco's ``-1.0``."""
    if want is None:
        return got == -1.0
    return abs(got - want) <= tol


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"case{c['id']}")
def test_matches_mmseg(case):
    res = evaluator(case).results()
    m = case["expected"]["mmseg"]
    for c, (iou, acc) in enumerate(zip(m["IoU"], m["Acc"])):
        assert close(res["per_class"][c]["IoU"], iou), (c, res["per_class"][c]["IoU"], iou)
        assert close(res["per_class"][c]["Acc"], acc), (c, res["per_class"][c]["Acc"], acc)
    assert close(res["mIoU"], m["mIoU"])
    assert close(res["mAcc"], m["mAcc"])
    assert close(res["aAcc"], m["aAcc"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"case{c['id']}")
def test_matches_torchmetrics_within_float32(case):
    res = evaluator(case).results()
    t = case["expected"]["torchmetrics"]
    for c, iou in enumerate(t["IoU"]):
        if res["per_class"][c]["IoU"] != -1.0:  # torchmetrics prints 0 for a class on neither side
            assert close(res["per_class"][c]["IoU"], iou, 1e-6)
    if t["mIoU"] is not None:
        assert close(res["mIoU"], t["mIoU"], 1e-6)


def test_detectron2_differs_exactly_on_the_documented_corner():
    corner = 0
    for case in CASES:
        res = evaluator(case).results()
        d = case["expected"]["detectron2"]
        assert close(res["mAcc"], d["mACC"]), case["id"]
        assert close(res["aAcc"], d["pACC"]), case["id"]
        if res["aAcc"] != -1.0:  # detectron2 sums nothing to 0.0 on an all-ignored split
            assert close(res["fwIoU"], d["fwIoU"]), case["id"]
        pred_only = any(c["IoU"] == 0.0 and c["Acc"] == -1.0 for c in res["per_class"])
        if pred_only:
            corner += 1
            assert not close(res["mIoU"], d["mIoU"]), (case["id"], res["mIoU"], d["mIoU"])
        else:
            assert close(res["mIoU"], d["mIoU"]), (case["id"], res["mIoU"], d["mIoU"])
    assert corner >= FIXTURE["min_pred_only_cases"] * len(CASES)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_every_integer_dtype_gives_the_same_counts(dtype):
    case = next(c for c in CASES if c["ignore_index"] == 255 and not c["reduce_zero_label"])
    want = evaluator(case).confusion()
    assert np.array_equal(evaluator(case, dtype).confusion(), want)


def test_a_batch_axis_is_one_update():
    """An ``(N, H, W)`` batch counts like N calls; it is one update, not N."""
    case = CASES[0]
    gt, pred = np.array(case["gt"][0]), np.array(case["pred"][0])
    make = lambda: semantic.SemanticEval(case["num_classes"], ignore_index=case["ignore_index"])  # noqa: E731
    per_image = make()
    per_image.update(gt, pred)
    per_image.update(gt[::-1], pred[::-1])
    batched = make()
    batched.update(np.stack([gt, gt[::-1]]), np.stack([pred, pred[::-1]]))
    assert np.array_equal(batched.confusion(), per_image.confusion())
    assert batched.n_images == 1 and per_image.n_images == 2


@pytest.mark.parametrize("mode", ["L", "P", "I;16"])
def test_png_folders_match_arrays(tmp_path, mode):
    case = next(c for c in CASES if c["ignore_index"] == 255 and not c["reduce_zero_label"] and len(c["gt"]) > 1)
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    pil_dtype = np.uint16 if mode == "I;16" else np.uint8
    for i, (gt, pred) in enumerate(zip(case["gt"], case["pred"])):
        for folder, arr in ((gt_dir, gt), (pred_dir, pred)):
            # Pillow infers I;16 from uint16 and L from uint8; P is a relabeling of L.
            img = Image.fromarray(np.array(arr, dtype=pil_dtype))
            if mode == "P":
                img = Image.fromarray(np.array(arr, dtype=np.uint8), mode="P")
            if mode == "P":
                img.putpalette([0, 0, 0] * 256)
            img.save(folder / f"{i:03d}.png")
    ev = semantic.SemanticEval(case["num_classes"])
    assert ev.update_folders(gt_dir, pred_dir) == len(case["gt"])
    assert np.array_equal(ev.confusion(), evaluator(case).confusion())
    assert ev.results()["mIoU"] == evaluator(case).results()["mIoU"]


def test_metrics_functions_agree_with_the_evaluator():
    case = CASES[0]
    total = np.zeros((case["num_classes"],) * 2, dtype=np.uint64)
    for gt, pred in zip(case["gt"], case["pred"]):
        total += metrics.pixel_confusion(gt, pred, case["num_classes"], ignore_index=case["ignore_index"])
    ev = evaluator(case)
    assert np.array_equal(total, ev.confusion())
    scores = metrics.mean_iou(total)
    res = ev.results()
    assert scores["mIoU"] == res["mIoU"] and scores["fwIoU"] == res["fwIoU"]
    assert scores["IoU"] == [c["IoU"] for c in res["per_class"]]
    assert scores["Acc"] == [c["Acc"] for c in res["per_class"]]


def test_shards_merge_and_pickle():
    case = next(c for c in CASES if len(c["gt"]) > 2)
    whole = evaluator(case)
    shards = []
    for gt, pred in zip(case["gt"], case["pred"]):
        ev = semantic.SemanticEval(
            case["num_classes"], ignore_index=case["ignore_index"], reduce_zero_label=case["reduce_zero_label"]
        )
        ev.update(np.array(gt), np.array(pred))
        shards.append(pickle.loads(pickle.dumps(ev)))  # the DDP round trip
    merged = shards[0]
    for s in shards[1:]:
        merged.merge(s)
    assert np.array_equal(merged.confusion(), whole.confusion())
    assert merged.n_images == whole.n_images
    assert merged.report() == whole.report()


def test_report_contract():
    ev = semantic.SemanticEval(["road", "sky", "void"])
    ev.update(np.array([[0, 0], [1, 255]], dtype=np.uint8), np.array([[0, 1], [1, 1]], dtype=np.uint8))
    report = ev.report()
    assert report["task"] == "semantic"
    assert report["provenance"] == "parity_verified"
    assert list(report["metrics"]) == sorted(semantic.METRIC_NAMES)
    assert report["per_class"] == {"road": {"IoU": 0.5, "Acc": 0.5}, "sky": {"IoU": 0.5, "Acc": 1.0}}
    assert report["per_group"] == {} and report["curves"] == {}
    assert report["params"]["ignore_index"] == 255 and report["params"]["n_pixels"] == 3
    assert ev.reference_deviations() == []
    assert ev.stats == [report["metrics"][k] for k in semantic.METRIC_NAMES]


def test_bad_input_is_a_value_error():
    ev = semantic.SemanticEval(2)
    with pytest.raises(ValueError, match="same shape"):
        ev.update(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.uint8))
    with pytest.raises(ValueError, match="pred pixel 0 has label 9"):
        ev.update(np.zeros((2, 2), np.uint8), np.full((2, 2), 9, np.uint8))
    with pytest.raises(ValueError, match="negative"):
        ev.update(np.array([[-1]]), np.array([[0]]))
    with pytest.raises(TypeError, match="integer"):
        ev.update(np.zeros((2, 2)), np.zeros((2, 2)))
    assert ev.n_images == 0 and ev.confusion().sum() == 0
    with pytest.raises(ValueError, match="different classes"):
        ev.merge(semantic.SemanticEval(3))
    with pytest.raises(ValueError, match="square"):
        metrics.mean_iou([[1, 2, 3], [4, 5, 6]])
    # The uint64 matrices two evaluators hand back, subtracted the wrong way round.
    with pytest.raises(ValueError, match="not a pixel count"):
        metrics.mean_iou(np.array([[5, 3], [0, 4]], np.uint64) - np.array([[6, 3], [0, 4]], np.uint64))
    # An option pair with no reference meaning fails at construction, not one batch in.
    with pytest.raises(ValueError, match="needs ignore_index 255"):
        semantic.SemanticEval(2, ignore_index=None, reduce_zero_label=True)
    # A flag is not a class count, though Python would read it as 1.
    with pytest.raises(TypeError, match="number of classes"):
        semantic.SemanticEval(True)
    # A uint64 label of 2**63 or more is out of range, not negative, and is
    # read as any other label is: unchecked under an ignored pixel, and the
    # ignore index itself whatever its size.
    with pytest.raises(
        ValueError, match=r"gt pixel 1 has label 18446744073709551615; labels must be below num_classes \(2\)"
    ):
        ev.update(np.array([[0, 2**64 - 1]], np.uint64), np.array([[0, 0]]))
    ev.update(np.array([[0, 255]]), np.array([[0, 2**64 - 1]], np.uint64))
    wide = semantic.SemanticEval(2, ignore_index=2**64 - 1)
    wide.update(np.array([[0, 2**64 - 1]], np.uint64), np.array([[0, 1]]))
    assert ev.confusion().sum() == 1 and wide.confusion().sum() == 1
    # The same in the other byte order, which is a copy, not a borrow.
    with pytest.raises(ValueError, match="gt pixel 1 has label 9223372036854775808"):
        ev.update(np.array([[0, 2**63]], ">u8"), np.array([[0, 0]]))
    # rust-numpy views at most 32 dimensions; numpy allows 64.
    with pytest.raises(ValueError, match="gt has 33 dimensions; at most 32"):
        ev.update(np.zeros((1,) * 33, np.uint8), np.zeros((1,) * 33, np.uint8))


@pytest.mark.parametrize("view", ["strided", "fortran", "transposed", "misaligned"], ids=str)
def test_a_view_counts_like_its_contiguous_copy(view):
    """A C-contiguous, aligned buffer is read in place; any other layout is copied in row-major order.

    Fortran order is the trap: its buffer is contiguous too, in column-major order, and
    borrowing it against a C-order partner pairs each pixel with the wrong one. A
    ``frombuffer`` at an odd offset is contiguous but not a slice of ``uint16``.
    """
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 3, (4, 8), dtype=np.uint16)
    pred = rng.integers(0, 3, (4, 8), dtype=np.int64)
    if view == "strided":
        gt, pred = gt[:, ::2], pred[:, ::2]
    elif view == "fortran":
        gt = np.asfortranarray(gt)
    elif view == "transposed":
        gt, pred = gt.T, np.ascontiguousarray(pred.T)
    else:
        gt = np.frombuffer(b"\0" + gt.tobytes(), dtype=np.uint16, offset=1).reshape(gt.shape)
        assert gt.flags.c_contiguous and not gt.flags.aligned
    in_place, other = semantic.SemanticEval(3), semantic.SemanticEval(3)
    in_place.update(gt.copy(), pred.copy())  # a fresh array is C-contiguous and aligned
    other.update(gt, pred)
    assert np.array_equal(in_place.confusion(), other.confusion())
    assert in_place.confusion().sum() == gt.size
