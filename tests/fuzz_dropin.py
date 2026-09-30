"""Drop-in spelling fuzzer: one dataset, many in-memory representations.

`fuzz_parity.py` round-trips every dataset through JSON files, so it can never
see a `bytes` RLE `counts`, a numpy scalar id, a tuple bbox, a `bool` iscrowd,
or a missing optional key — the class of gap issue #5 reported, where the same
mask scored 0.0 or 1.0 depending on how `counts` was spelled.

This fuzzer builds a small dataset in memory, evaluates it through pycocotools
in its canonical spelling to get the reference numbers, then applies one
*spelling transform* at a time — a change pycocotools treats as equivalent, or
tolerates — and evaluates the transformed dataset through hotcoco. Each
(transform, iou_type) lands in one of these buckets:

- **silent**: hotcoco returned numbers that differ from the reference. The bug
  class that matters; the test fails on the first one and hypothesis minimizes.
- **loud**: hotcoco raised where pycocotools evaluated. A drop-in gap, reported
  in the summary at the end of the session, never a failure.
- **tolerant**: hotcoco matched the canonical numbers where pycocotools raised
  or changed its own answer. Fine, also summarized.
- **shared drift**: both implementations moved off the canonical numbers the
  same way. Not hotcoco's bug, but worth a look; summarized.
- **ok**: both agree.

A bug-hunting tool, not a CI gate.

Usage:
    uv run pytest tests/fuzz_dropin.py -x -q -p no:cacheprovider
"""

from __future__ import annotations

import copy
import random
import sys
import warnings
from collections import Counter, defaultdict

import hotcoco
import hypothesis.strategies as st
import numpy as np
import pycocotools.mask as pm
import pytest
from helpers import COCO_KEYPOINT_NAMES, COCO_SKELETON, compare_metrics, metric_names_for, suppress_output
from hypothesis import HealthCheck, given, settings
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

TOL = 1e-9  # for the eval-dict arrays; stats go through helpers.compare_metrics
IOU_TYPES = ("bbox", "segm", "keypoints")

# ---------------------------------------------------------------------------
# Canonical dataset strategy
# ---------------------------------------------------------------------------


def _rect_polygon(x, y, w, h):
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def _rle_str(x, y, w, h, img_w, img_h):
    m = np.zeros((img_h, img_w), dtype=np.uint8, order="F")
    m[y : y + h, x : x + w] = 1
    rle = pm.encode(m)
    return {"size": [img_h, img_w], "counts": rle["counts"].decode("ascii")}


@st.composite
def canonical_dataset(draw):
    """A small GT dict plus a list of detections, all in the plain-JSON spelling."""
    n_img = draw(st.integers(1, 3))
    n_cat = draw(st.integers(1, 3))
    images = []
    for i in range(n_img):
        images.append(
            {
                "id": i + 1,
                "width": draw(st.integers(8, 40)),
                "height": draw(st.integers(8, 40)),
                "file_name": f"{i + 1}.jpg",
            }
        )
    categories = [
        {
            "id": c + 1,
            "name": f"c{c + 1}",
            "supercategory": "s",
            "keypoints": COCO_KEYPOINT_NAMES,
            "skeleton": COCO_SKELETON,
        }
        for c in range(n_cat)
    ]

    def keypoints(x, y, w, h):
        # 17 keypoints, not fewer: pycocotools' computeOks hard-codes 17
        # sigmas and raises on any other count, which would silently skip
        # every keypoint transform (the reference must evaluate first).
        kps, n_vis = [], 0
        for _ in COCO_KEYPOINT_NAMES:
            v = draw(st.sampled_from([0, 1, 2, 2]))
            if v == 0:
                kps += [0, 0, 0]
            else:
                kps += [x + draw(st.integers(0, w)), y + draw(st.integers(0, h)), v]
                n_vis += 1
        return kps, n_vis

    def box(img):
        w = draw(st.integers(1, img["width"]))
        h = draw(st.integers(1, img["height"]))
        x = draw(st.integers(0, img["width"] - w))
        y = draw(st.integers(0, img["height"] - h))
        return x, y, w, h

    anns = []
    for a in range(draw(st.integers(1, 5))):
        img = images[draw(st.integers(0, n_img - 1))]
        x, y, w, h = box(img)
        use_rle = draw(st.booleans())
        seg = _rle_str(x, y, w, h, img["width"], img["height"]) if use_rle else _rect_polygon(x, y, w, h)
        kps, n_vis = keypoints(x, y, w, h)
        anns.append(
            {
                "id": a + 1,
                "image_id": img["id"],
                "category_id": draw(st.integers(1, n_cat)),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": int(draw(st.sampled_from([0, 0, 0, 1]))),
                "segmentation": seg,
                "keypoints": kps,
                "num_keypoints": n_vis,
            }
        )
    dets = []
    for _ in range(draw(st.integers(0, 6))):
        img = images[draw(st.integers(0, n_img - 1))]
        # Bias toward near-GT boxes so IoU thresholds actually matter.
        if anns and draw(st.booleans()):
            src = anns[draw(st.integers(0, len(anns) - 1))]
            if src["image_id"] == img["id"]:
                bx, by, bw, bh = src["bbox"]
                jitter = draw(st.integers(0, 2))
                x, y = bx, by
                w = max(1, min(img["width"] - x, bw + jitter))
                h = max(1, min(img["height"] - y, bh - jitter if bh > jitter else bh))
            else:
                x, y, w, h = box(img)
        else:
            x, y, w, h = box(img)
        use_rle = draw(st.booleans())
        seg = _rle_str(x, y, w, h, img["width"], img["height"]) if use_rle else _rect_polygon(x, y, w, h)
        kps, _ = keypoints(x, y, w, h)
        dets.append(
            {
                "image_id": img["id"],
                "category_id": draw(st.integers(1, n_cat)),
                "bbox": [x, y, w, h],
                "score": round(draw(st.floats(0.05, 1.0)), 3),
                "segmentation": seg,
                "keypoints": kps,
            }
        )
    gt = {"images": images, "annotations": anns, "categories": categories}
    return gt, dets


# ---------------------------------------------------------------------------
# Spelling transforms
# ---------------------------------------------------------------------------
#
# Each takes (gt, dets) — already deep-copied — and returns (gt, dets, params)
# where `params` is a dict of Params attributes to set (or None). Every
# transform is something pycocotools either treats as identical to the
# canonical spelling or tolerates without changing the numbers.

TRANSFORMS = {}


def transform(name, *, iou_types=IOU_TYPES):
    """Register a spelling. `iou_types` names where it is an equivalence."""

    def deco(fn):
        assert name not in TRANSFORMS, f"transform {name!r} registered twice"
        fn.iou_types = frozenset(iou_types)
        TRANSFORMS[name] = fn
        return fn

    return deco


def _walk_numbers(obj, fn):
    """Apply `fn` to every int/float leaf (not bool) in nested dict/list."""
    if isinstance(obj, dict):
        return {k: _walk_numbers(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_numbers(v, fn) for v in obj]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return fn(obj)
    return obj


def _cast_ints(kind):
    lo, hi = np.iinfo(kind).min, np.iinfo(kind).max

    def fn(v):
        # Narrow dtypes are realistic (uint8 category ids, int16 boxes from a
        # tensor) only for values that fit; leave the rest alone.
        return kind(v) if isinstance(v, int) and lo <= v <= hi else v

    return fn


def _cast_floats(kind):
    def fn(v):
        return kind(v) if isinstance(v, float) else v

    return fn


@transform("identity")
def _t_identity(gt, dets):
    return gt, dets, None


for _kind in (np.int64, np.int32, np.uint8, np.int16):

    @transform(f"ints_as_{_kind.__name__}")
    def _t_int_kind(gt, dets, _kind=_kind):
        f = _cast_ints(_kind)
        return _walk_numbers(gt, f), _walk_numbers(dets, f), None


@transform("ints_as_python_float")
def _t_ints_float(gt, dets):
    # `20.0` for a size or a bbox is how a JSON written by numpy/pandas reads;
    # ids stay ints here, `ids_as_python_float` covers those.
    def keep_ids(obj):
        # Floats for everything except identifiers, which pycocotools uses as dict keys.
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[k] = v if k in ("id", "image_id", "category_id") else keep_ids(v)
            return out
        if isinstance(obj, list):
            return [keep_ids(v) for v in obj]
        return float(obj) if isinstance(obj, int) and not isinstance(obj, bool) else obj

    return keep_ids(gt), keep_ids(dets), None


def _map_ids(gt, dets, fn):
    """Apply `fn` to every image id, annotation id, and image reference."""
    for img in gt["images"]:
        img["id"] = fn(img["id"])
    for a in gt["annotations"]:
        a["id"], a["image_id"] = fn(a["id"]), fn(a["image_id"])
    for d in dets:
        d["image_id"] = fn(d["image_id"])
    return gt, dets, None


@transform("ids_as_python_float")
def _t_ids_float(gt, dets):
    return _map_ids(gt, dets, float)


for _kind in (np.float32, np.float64):

    @transform(f"floats_as_{_kind.__name__}")
    def _t_float_kind(gt, dets, _kind=_kind):
        f = _cast_floats(_kind)
        return _walk_numbers(gt, f), _walk_numbers(dets, f), None


def field_transform(name, key, fn, *, gt_only=False, when=lambda v: True, iou_types=IOU_TYPES):
    """Register a spelling that rewrites one field on every record that has it."""

    @transform(name, iou_types=iou_types)
    def _t(gt, dets):
        for a in gt["annotations"] + ([] if gt_only else dets):
            if key in a and when(a[key]):
                a[key] = fn(a[key])
        return gt, dets, None


def _is_rle(s):
    return isinstance(s, dict)


def _is_polygon(s):
    return isinstance(s, list)


def _with_size(fn):
    return lambda s: {**s, "size": fn(s["size"])}


field_transform("bbox_as_tuple", "bbox", tuple)
field_transform("bbox_as_ndarray", "bbox", lambda b: np.asarray(b, dtype=np.float64))
field_transform("bbox_as_float32_ndarray", "bbox", lambda b: np.asarray(b, dtype=np.float32))
field_transform("score_as_float32", "score", np.float32)
field_transform("score_as_0d_ndarray", "score", np.array)
field_transform("iscrowd_as_bool", "iscrowd", bool, gt_only=True)
field_transform("iscrowd_as_np_bool", "iscrowd", np.bool_, gt_only=True)
field_transform("iscrowd_as_float", "iscrowd", float, gt_only=True)
field_transform("area_as_float32", "area", np.float32, gt_only=True)
field_transform(
    "rle_counts_bytes",
    "segmentation",
    lambda s: {**s, "counts": s["counts"].encode("ascii")},
    when=lambda s: _is_rle(s) and isinstance(s["counts"], str),
)
field_transform("rle_size_tuple", "segmentation", _with_size(tuple), when=_is_rle)
field_transform("rle_size_ndarray", "segmentation", _with_size(np.asarray), when=_is_rle)
field_transform("polygon_as_tuple", "segmentation", lambda s: tuple(tuple(p) for p in s), when=_is_polygon)
field_transform(
    "polygon_as_ndarray", "segmentation", lambda s: [np.asarray(p, dtype=np.float64) for p in s], when=_is_polygon
)
field_transform(
    "polygon_as_float32_ndarray",
    "segmentation",
    lambda s: [np.asarray(p, dtype=np.float32) for p in s],
    when=_is_polygon,
)
field_transform(
    "polygon_as_single_ndarray_2d", "segmentation", lambda s: np.asarray(s, dtype=np.float64), when=_is_polygon
)
field_transform("keypoints_as_ndarray", "keypoints", lambda k: np.asarray(k, dtype=np.float64))
field_transform("keypoints_as_float32_ndarray", "keypoints", lambda k: np.asarray(k, dtype=np.float32))
field_transform("keypoints_as_nx3_ndarray", "keypoints", lambda k: np.asarray(k, dtype=np.float64).reshape(-1, 3))
field_transform("keypoints_as_floats", "keypoints", lambda k: [float(v) for v in k])


@transform("iscrowd_missing_when_zero")
def _t_iscrowd_missing(gt, dets):
    for a in gt["annotations"]:
        if not a["iscrowd"]:
            del a["iscrowd"]
    return gt, dets, None


@transform("area_missing_on_gt")
def _t_area_missing(gt, dets):
    for a in gt["annotations"]:
        del a["area"]
    return gt, dets, None


@transform("image_size_missing", iou_types=("bbox", "keypoints"))
def _t_img_size_missing(gt, dets):
    # TorchMetrics emits bare {"id": i}. Only meaningful for bbox; segm needs
    # the size to rasterize polygons, so pycocotools raises there.
    for img in gt["images"]:
        del img["width"]
        del img["height"]
    return gt, dets, None


@transform("file_name_missing")
def _t_file_name_missing(gt, dets):
    for img in gt["images"]:
        del img["file_name"]
    return gt, dets, None


@transform("category_name_missing")
def _t_cat_name_missing(gt, dets):
    for c in gt["categories"]:
        del c["name"]
    return gt, dets, None


@transform("supercategory_missing")
def _t_supercat_missing(gt, dets):
    for c in gt["categories"]:
        del c["supercategory"]
    return gt, dets, None


@transform("extra_keys_everywhere")
def _t_extra_keys(gt, dets):
    junk = {"note": None, "meta": {"nested": [1, "x", b"raw"]}, "flag": True, "arr": np.arange(3)}
    for rec in gt["images"] + gt["annotations"] + gt["categories"] + dets:
        rec.update(copy.deepcopy(junk))
    gt["info"] = {"version": 1}
    gt["licenses"] = []
    return gt, dets, None


@transform("shuffled_records")
def _t_shuffled(gt, dets):
    rng = random.Random(0)
    for key in ("images", "annotations", "categories"):
        rng.shuffle(gt[key])
    rng.shuffle(dets)
    return gt, dets, None


@transform("ids_offset_large")
def _t_ids_large(gt, dets):
    return _map_ids(gt, dets, lambda i: i + 2**40)


@transform("ids_zero_based")
def _t_ids_zero(gt, dets):
    return _map_ids(gt, dets, lambda i: i - 1)


@transform("rle_uncompressed_list")
def _t_rle_uncompressed(gt, dets):
    for a in gt["annotations"] + dets:
        s = a["segmentation"]
        if isinstance(s, dict) and isinstance(s["counts"], str):
            h, w = s["size"]
            m = pm.decode({"size": [h, w], "counts": s["counts"].encode()})
            # Column-major run lengths starting with a background run.
            flat = np.asfortranarray(m).ravel(order="F")
            counts, cur, run = [], 0, 0
            for v in flat:
                if v == cur:
                    run += 1
                else:
                    counts.append(run)
                    cur, run = v, 1
            counts.append(run)
            s["counts"] = [int(c) for c in counts]
    return gt, dets, None


@transform("rle_uncompressed_ndarray_counts")
def _t_rle_uncompressed_np(gt, dets):
    gt, dets, _ = _t_rle_uncompressed(gt, dets)
    for a in gt["annotations"] + dets:
        s = a["segmentation"]
        if isinstance(s, dict) and isinstance(s["counts"], list):
            s["counts"] = np.asarray(s["counts"], dtype=np.int64)
    return gt, dets, None


@transform("num_keypoints_missing")
def _t_num_kp_missing(gt, dets):
    for a in gt["annotations"]:
        del a["num_keypoints"]
    return gt, dets, None


@transform("dets_without_keypoints", iou_types=("bbox", "segm"))
def _t_dets_no_kp(gt, dets):
    for d in dets:
        d.pop("keypoints", None)
    return gt, dets, None


@transform("dets_as_numpy_nx7", iou_types=("bbox",))
def _t_dets_numpy(gt, dets):
    # pycocotools' loadNumpyAnnotations path: rows of [image_id, x, y, w, h, score, category_id].
    rows = [[d["image_id"], *d["bbox"], d["score"], d["category_id"]] for d in dets]
    return gt, np.asarray(rows, dtype=np.float64).reshape(-1, 7), None


@transform("params_cat_ids_subset_ndarray")
def _t_params_cat_subset(gt, dets):
    ids = sorted(c["id"] for c in gt["categories"])
    return gt, dets, {"catIds": np.asarray(ids[: max(1, len(ids) - 1)])}


@transform("params_img_ids_subset_np_scalars")
def _t_params_img_subset(gt, dets):
    ids = sorted(i["id"] for i in gt["images"])
    return gt, dets, {"imgIds": [np.int64(i) for i in ids[: max(1, len(ids) - 1)]]}


@transform("dets_without_segmentation", iou_types=("bbox", "keypoints"))
def _t_dets_no_seg(gt, dets):
    # loadRes synthesizes a polygon from bbox — segm numbers change, so this
    # is only an equivalence off segm.
    for d in dets:
        d.pop("segmentation", None)
    return gt, dets, None


@transform("dets_with_id_and_area")
def _t_dets_with_id(gt, dets):
    for i, d in enumerate(dets):
        d["id"] = i + 1
        d["area"] = d["bbox"][2] * d["bbox"][3]
        d["iscrowd"] = 0
    return gt, dets, None


@transform("params_img_ids_ndarray")
def _t_params_img_np(gt, dets):
    return gt, dets, {"imgIds": np.asarray([i["id"] for i in gt["images"]])}


@transform("params_img_ids_np_scalars")
def _t_params_img_npscalar(gt, dets):
    return gt, dets, {"imgIds": [np.int64(i["id"]) for i in gt["images"]]}


@transform("params_cat_ids_tuple")
def _t_params_cat_tuple(gt, dets):
    return gt, dets, {"catIds": tuple(c["id"] for c in gt["categories"])}


@transform("params_max_dets_tuple", iou_types=("bbox", "segm"))
def _t_params_maxdets_tuple(gt, dets):
    return gt, dets, {"maxDets": (1, 10, 100)}


@transform("params_max_dets_tuple_kp", iou_types=("keypoints",))
def _t_params_maxdets_tuple_kp(gt, dets):
    # pycocotools' keypoint summary hard-codes maxDets=20 and reports -1 for
    # everything when the list lacks it; hotcoco reports at the configured cap
    # and warns. Deliberate, so only the keypoint default is an equivalence.
    return gt, dets, {"maxDets": (20,)}


@transform("params_iou_thrs_ndarray_default")
def _t_params_iou_np(gt, dets):
    return gt, dets, {"iouThrs": np.linspace(0.5, 0.95, 10)}


@transform("params_rec_thrs_list_default")
def _t_params_rec_list(gt, dets):
    return gt, dets, {"recThrs": [round(x, 2) for x in np.linspace(0, 1, 101)]}


@transform("params_area_rng_tuples")
def _t_params_area_tuples(gt, dets):
    return gt, dets, {"areaRng": [(0, 1e10), (0, 32**2), (32**2, 96**2), (96**2, 1e10)]}


@transform("params_use_cats_int")
def _t_params_usecats_int(gt, dets):
    return gt, dets, {"useCats": 1}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _py_eval(gt, dets, iou_type, params, *, direct):
    # pycocotools assigns keys on annotation dicts (`ignore`, `segmentation`,
    # loadRes' `id`/`area`) and never touches images, categories, or the values
    # inside a record, so per-record shallow copies keep the shared spelling
    # intact across its six evaluations.
    with suppress_output():
        coco = PyCOCO()
        coco.dataset = {**gt, "annotations": [dict(a) for a in gt["annotations"]]}
        coco.createIndex()
        if direct:
            dt = PyCOCO()
            dt.dataset = _dt_dataset(gt, dets)
            dt.createIndex()
        else:
            dt = coco.loadRes(dets if isinstance(dets, np.ndarray) else [dict(d) for d in dets])
        ev = PyCOCOeval(coco, dt, iou_type)
        if params:
            for k, v in params.items():
                setattr(ev.params, k, v)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return ev.stats.tolist(), _eval_arrays(ev)


def _eval_arrays(ev):
    return {k: np.asarray(ev.eval[k], dtype=np.float64) for k in ("precision", "recall", "scores")}


def _dt_dataset(gt, dets):
    """The 'build a COCO of detections by hand' path (issue #5's integration)."""
    anns = []
    for i, d in enumerate(dets):
        a = dict(d)
        a.setdefault("id", i + 1)
        b = a["bbox"]
        a.setdefault("area", float(b[2]) * float(b[3]))
        a.setdefault("iscrowd", 0)
        anns.append(a)
    return {"images": copy.deepcopy(gt["images"]), "categories": copy.deepcopy(gt["categories"]), "annotations": anns}


def _rs_eval(gt, dets, iou_type, params, *, direct):
    # No copies: hotcoco reads the dicts into Rust and never mutates them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with suppress_output():
            coco = hotcoco.COCO(gt)
            dt = hotcoco.COCO(_dt_dataset(gt, dets)) if direct else coco.loadRes(dets)
            ev = hotcoco.COCOeval(coco, dt, iou_type)
            if params:
                for k, v in params.items():
                    setattr(ev.params, k, v)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            return [float(x) for x in ev.stats], _eval_arrays(ev)


def _close(a, b, iou_type):
    """Stats (via the shared comparator) and the eval-dict arrays both agree."""
    sa, ea = a
    sb, eb = b
    if compare_metrics(sa, sb, metric_names_for(iou_type)):
        return False
    return all(ea[k].shape == eb[k].shape and np.allclose(ea[k], eb[k], atol=TOL, rtol=0) for k in ea)


# Session-wide tallies, printed at the end.
OUTCOMES: dict[str, Counter] = defaultdict(Counter)
EXAMPLES: dict[str, str] = {}


def _record(bucket, key, detail=None):
    OUTCOMES[bucket][key] += 1
    if detail and key not in EXAMPLES:
        EXAMPLES[key] = detail


def _run_transform(key, tgt, tdets, params, iou_type, reference, *, direct, py_stats=None):
    py_err = None
    if py_stats is None:
        try:
            py_stats = _py_eval(tgt, tdets, iou_type, params, direct=direct)
        except Exception as e:  # noqa: BLE001
            py_err = f"{type(e).__name__}: {e}"
    try:
        rs_stats = _rs_eval(tgt, tdets, iou_type, params, direct=direct)
        rs_err = None
    except Exception as e:  # noqa: BLE001
        rs_stats, rs_err = None, f"{type(e).__name__}: {e}"

    if rs_stats is None:
        if py_stats is None:
            _record("both_raise", key, f"py: {py_err} | rs: {rs_err}")
        else:
            _record("loud", key, rs_err)
        return

    # Two independent facts decide the bucket: does hotcoco match the
    # canonical numbers, and does it match what pycocotools said under this
    # spelling? pycocotools may change its own answer — a zero annotation id
    # reads as "unmatched" in its match matrix, an int16 bbox overflows its
    # area multiply — so matching neither is the only silent divergence.
    rs_matches_ref = _close(rs_stats, reference, iou_type)
    rs_matches_py = py_stats is not None and _close(rs_stats, py_stats, iou_type)
    py_matches_ref = py_stats is not None and _close(py_stats, reference, iou_type)
    if rs_matches_ref and py_matches_ref:
        _record("ok", key)
    elif rs_matches_ref:
        _record("tolerant", key, py_err or "pycocotools changed its own answer")
    elif rs_matches_py:
        _record("shared_drift", key, f"both moved to {py_stats[0][:3]} from {reference[0][:3]}")
    else:
        _record("silent", key, f"ref={reference[0][:3]} rs={rs_stats[0][:3]}")
        pytest.fail(
            f"SILENT DIVERGENCE {key}\n  reference: {reference[0]}\n  hotcoco:   {rs_stats[0]}\n"
            f"  pycocotools on the spelling: {py_err or py_stats[0]}\n"
            f"  gt={tgt}\n  dets={tdets}\n  params={params}"
        )


SETTINGS = {
    "max_examples": 120,
    "deadline": None,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much],
    "print_blob": True,
}


@given(data=canonical_dataset())
@settings(**SETTINGS)
def test_spellings_agree(data):
    gt, dets = data
    # Every transform is deterministic, so spell once per example and reuse
    # across the six (iou_type, path) evaluations.
    spelled = {name: fn(copy.deepcopy(gt), copy.deepcopy(dets)) for name, fn in TRANSFORMS.items()}
    for iou_type in IOU_TYPES:
        for direct in (False, True):
            if not direct and not dets:
                continue  # loadRes has nothing to load; the direct path covers empties
            try:
                reference = _py_eval(gt, dets, iou_type, None, direct=direct)
            except Exception:  # noqa: BLE001
                continue  # canonical spelling must evaluate for the test to mean anything
            for name, (tgt, tdets, params) in spelled.items():
                if iou_type not in TRANSFORMS[name].iou_types:
                    continue
                key = f"{name}/{iou_type}/{'direct' if direct else 'loadres'}"
                # `identity` is the canonical spelling; its pycocotools answer is the reference.
                py_stats = reference if name == "identity" else None
                _run_transform(key, tgt, tdets, params, iou_type, reference, direct=direct, py_stats=py_stats)


# ---------------------------------------------------------------------------
# hotcoco.mask spellings vs pycocotools.mask
# ---------------------------------------------------------------------------


@st.composite
def small_mask(draw):
    h = draw(st.integers(1, 24))
    w = draw(st.integers(1, 24))
    m = np.zeros((h, w), dtype=np.uint8)
    for _ in range(draw(st.integers(0, 3))):
        y0, x0 = draw(st.integers(0, h - 1)), draw(st.integers(0, w - 1))
        y1, x1 = draw(st.integers(y0, h - 1)), draw(st.integers(x0, w - 1))
        m[y0 : y1 + 1, x0 : x1 + 1] = 1
    return m


def _mask_spellings(m):
    """Arrays that hold the same pixels."""
    h, w = m.shape
    wide = np.zeros((h, w + 3), dtype=np.uint8)
    wide[:, 1 : w + 1] = m
    return {
        "uint8_F": np.asfortranarray(m),
        "uint8_C": np.ascontiguousarray(m),
        "bool_F": np.asfortranarray(m.astype(bool)),
        "bool_C": np.ascontiguousarray(m.astype(bool)),
        "sliced_view": wide[:, 1 : w + 1],
        "transposed_view": np.ascontiguousarray(m.T).T,
    }


def _rle_spellings(rle):
    """RLE dicts that mean the same mask."""
    b = rle["counts"]
    return {
        "bytes": {"size": list(rle["size"]), "counts": b},
        "str": {"size": list(rle["size"]), "counts": b.decode("ascii")},
        "size_tuple": {"size": tuple(rle["size"]), "counts": b},
        "size_ndarray": {"size": np.asarray(rle["size"]), "counts": b},
        "size_np_scalars": {"size": [np.int64(rle["size"][0]), np.int64(rle["size"][1])], "counts": b},
    }


@given(m=small_mask())
@settings(**SETTINGS)
def test_mask_spellings_agree(m):
    ref = pm.encode(np.asfortranarray(m))
    ref_area = int(pm.area(ref))
    ref_bbox = pm.toBbox(ref).tolist()
    ref_pixels = pm.decode(ref).tolist()

    for name, arr in _mask_spellings(m).items():
        key = f"mask.encode/{name}"
        try:
            got = hotcoco.mask.encode(arr)
        except Exception as e:  # noqa: BLE001
            _record("loud", key, f"{type(e).__name__}: {e}")
            continue
        if got["counts"] != ref["counts"] or list(got["size"]) != list(ref["size"]):
            _record("silent", key)
            pytest.fail(f"SILENT DIVERGENCE {key}: {got} != {ref} for\n{m}")
        _record("ok", key)

    h, w = ref["size"]
    # (hotcoco check against the reference, the same call on pycocotools)
    ops = {
        "decode": (lambda r: hotcoco.mask.decode(r).tolist() == ref_pixels, pm.decode),
        "area": (lambda r: int(hotcoco.mask.area(r)) == ref_area, pm.area),
        "toBbox": (lambda r: hotcoco.mask.toBbox(r).tolist() == ref_bbox, pm.toBbox),
        "iou_self": (
            lambda r: (
                abs(float(np.asarray(hotcoco.mask.iou([r], [r], [0])).ravel()[0]) - (1.0 if ref_area else 0.0)) < TOL
            ),
            lambda r: pm.iou([r], [r], [0]),
        ),
        "merge_single": (lambda r: hotcoco.mask.merge([r])["counts"] == ref["counts"], lambda r: pm.merge([r])),
        "frPyObjects": (
            lambda r: hotcoco.mask.frPyObjects(r, h, w)["counts"] == ref["counts"],
            lambda r: pm.frPyObjects(r, h, w),
        ),
    }
    for name, rle in _rle_spellings(ref).items():
        for op, (check, pm_call) in ops.items():
            key = f"mask.{op}/{name}"
            try:
                ok = check(rle)
            except Exception as e:  # noqa: BLE001
                try:
                    pm_call(rle)
                    _record("loud", key, f"{type(e).__name__}: {e}")
                except Exception:  # noqa: BLE001
                    _record("both_raise", key, f"{type(e).__name__}: {e}")
                continue
            if not ok:
                _record("silent", key)
                pytest.fail(f"SILENT DIVERGENCE {key} on rle={rle} mask=\n{m}")
            _record("ok", key)


# ---------------------------------------------------------------------------
# End-of-session report
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _report_at_end():
    yield
    _print_report()


def _print_report():
    out = sys.__stdout__
    print("\n" + "=" * 72, file=out)
    print("  drop-in spelling fuzzer — outcomes by (transform, iou_type, path)", file=out)
    print("=" * 72, file=out)
    for bucket in ("silent", "loud", "shared_drift", "tolerant", "both_raise"):
        keys = OUTCOMES.get(bucket)
        if not keys:
            continue
        label = {
            "silent": "SILENT: hotcoco returned different numbers",
            "loud": "LOUD: hotcoco raised where pycocotools evaluated",
            "shared_drift": "SHARED DRIFT: both moved off the canonical numbers together",
            "tolerant": "TOLERANT: hotcoco matched the canonical numbers where pycocotools raised or drifted",
            "both_raise": "BOTH RAISE",
        }[bucket]
        print(f"\n{label}", file=out)
        for key, n in sorted(keys.items()):
            print(f"  {key:<60} x{n}", file=out)
            if key in EXAMPLES:
                print(f"      e.g. {EXAMPLES[key][:160]}", file=out)
    n_ok = sum(OUTCOMES["ok"].values())
    print(f"\nok: {n_ok} (transform, iou_type, path) evaluations agreed", file=out)
