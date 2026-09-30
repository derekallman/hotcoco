"""Verify that __init__.pyi covers every public symbol in the hotcoco module.

Run with: uv run pytest tests/test_stubs.py -v
"""

from __future__ import annotations

import ast
import functools
import math
import sys
from pathlib import Path
from types import ModuleType

import hotcoco
import numpy as np
import pytest

STUB_PATH = Path(__file__).resolve().parent.parent / "python" / "hotcoco" / "__init__.pyi"


@functools.cache
def _parse_stub_names() -> dict[str, set[str]]:
    """Parse the .pyi file and return {class_name: {method_names}} and top-level names."""
    source = STUB_PATH.read_text()
    tree = ast.parse(source)

    top_level: set[str] = set()
    classes: dict[str, set[str]] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level.add(node.name)
        elif isinstance(node, ast.ClassDef):
            top_level.add(node.name)
            members: set[str] = set()
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
            classes[node.name] = members
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level.add(target.id)

    return {"__top__": top_level, **classes}


def _public_names(obj: object) -> set[str]:
    """Public attribute names (no underscore prefix), excluding submodules.

    A submodule becomes an attribute of its package the moment anything
    imports it, so `hotcoco.cli` is a name here only if some earlier test
    happened to import it. Enumerating them in a skip-list made this check
    depend on test ordering — it passed file-by-file and failed on the full
    suite. Submodules carry their own stubs (checked below) or are pure
    Python that pyright reads from source, so none of them belong to the
    top-level stub contract.
    """
    return {
        name for name in dir(obj) if not name.startswith("_") and not isinstance(getattr(obj, name, None), ModuleType)
    }


def test_stub_file_exists():
    assert STUB_PATH.exists(), f"Missing stub file: {STUB_PATH}"


def test_py_typed_exists():
    py_typed = STUB_PATH.parent / "py.typed"
    assert py_typed.exists(), "Missing py.typed marker"


def test_top_level_exports_covered():
    """Every public name in hotcoco should appear in the stub."""
    stub_names = _parse_stub_names()["__top__"]
    runtime_names = _public_names(hotcoco)

    # Submodules (detection, metrics, primitives, mask, plot, browse, cli, ...)
    # are already excluded by _public_names; their contents are checked by the
    # tests below, or read from source by pyright for the pure-Python ones.
    #
    # "annotations" is the _Feature object bound by `from __future__ import
    # annotations` in __init__.py — a language directive, not public API.
    # The LVIS drop-in surface (LVIS, LVISEval, LVISeval, LVISResults) and the
    # torchvision one (CocoDetection, CocoEvaluator) are deliberately NOT here:
    # they are documented imports, so the stub must carry them or pyright
    # rejects `from hotcoco import LVISEval` in every LVIS example.
    skip = {"annotations"}
    runtime_names -= skip

    missing = runtime_names - stub_names
    assert not missing, f"Public names missing from stubs: {sorted(missing)}"


@pytest.mark.parametrize("name", ["COCO", "COCOeval", "Params", "Hierarchy"])
def test_members_covered(name):
    """Each stubbed top-level object and its stub must list the same members.

    One parametrized case per object rather than identical functions: adding
    a class to the stub should be a one-line edit here, not a copy-paste that
    invites a drifting message. (`mask` moved to its own module stub — see
    test_mask_stub_matches_runtime.)

    Both directions are checked, and the second is the one that bites. A member
    missing from the stub is merely invisible to autocomplete; a member the stub
    invents that the extension never defines is worse, because an IDE offers it
    and the call raises `AttributeError` at runtime.
    """
    stub_members = _parse_stub_names().get(name, set())
    runtime_members = _public_names(getattr(hotcoco, name))

    missing = runtime_members - stub_members
    assert not missing, f"{name} members missing from stubs: {sorted(missing)}"

    # `_public_names` drops underscore-prefixed names, so compare like with like —
    # otherwise every stubbed `__init__` reads as a phantom.
    phantom = {m for m in stub_members if not m.startswith("_")} - runtime_members
    assert not phantom, f"{name} members stubbed but not defined: {sorted(phantom)}"


@pytest.mark.parametrize("name", ["LVISeval", "LVISResults", "CocoDetection", "CocoEvaluator"])
def test_dropin_class_methods_covered(name):
    """The LVIS / torchvision drop-in classes must be stubbed with their methods.

    Only the missing direction: these stubs also declare instance attributes
    (``self.coco``, ``self.ids``, ...) that ``dir()`` on the class cannot see,
    so the phantom check that works for the PyO3 classes would false-positive.
    The presence of the name itself is enforced by
    test_top_level_exports_covered, which no longer skip-lists this surface.
    """
    stub_members = _parse_stub_names().get(name)
    assert stub_members is not None, f"{name} missing from __init__.pyi"
    runtime_members = _public_names(getattr(hotcoco, name))
    missing = runtime_members - stub_members
    assert not missing, f"{name} members missing from stubs: {sorted(missing)}"


def test_mask_stub_matches_runtime():
    """mask.pyi must list exactly the runtime `hotcoco.mask` functions.

    The mask surface is stubbed as a submodule (mask.pyi) rather than a class
    in __init__.pyi, so `import hotcoco.mask` — the pycocotools migration
    form — type-checks. Both directions, same reasoning as
    test_members_covered.
    """
    stub_names = _stub_function_names("mask.pyi")
    runtime_members = _public_names(hotcoco.mask)

    missing = runtime_members - stub_names
    assert not missing, f"mask members missing from mask.pyi: {sorted(missing)}"

    phantom = {m for m in stub_names if not m.startswith("_")} - runtime_members
    assert not phantom, f"mask members stubbed but not defined: {sorted(phantom)}"


# ---------------------------------------------------------------------------
# Family namespaces
# ---------------------------------------------------------------------------


def test_detection_namespace_importable_both_ways():
    """`import hotcoco.detection` and `from hotcoco import detection` must agree.

    PyO3's `add_submodule` makes a submodule reachable as an *attribute* without
    registering it in `sys.modules`, so the `import x.y` form fails while
    `from x import y` works. That bit `hotcoco.mask`; this test keeps it from
    biting the family namespaces as panoptic, tracking, and concepts land.
    """

    import hotcoco.detection
    from hotcoco import detection

    assert "hotcoco.detection" in sys.modules
    assert detection is hotcoco.detection


def test_mask_importable_both_ways():
    """The regression that motivated the check above."""
    import hotcoco.mask
    from hotcoco import mask

    assert mask is hotcoco.mask
    assert hasattr(hotcoco.mask, "iou")


def test_detection_namespace_reexports_are_the_same_objects():
    """The namespace is additive sugar, not a parallel implementation."""
    from hotcoco import detection

    for name in detection.__all__:
        assert getattr(detection, name) is getattr(hotcoco, name), f"hotcoco.detection.{name} is not hotcoco.{name}"


def test_detection_stub_matches_runtime():
    stub = ast.parse((STUB_PATH.parent / "detection.pyi").read_text())
    stubbed: set[str] = set()
    for node in ast.iter_child_nodes(stub):
        if isinstance(node, ast.ImportFrom):
            stubbed.update(alias.asname or alias.name for alias in node.names)

    from hotcoco import detection

    missing = set(detection.__all__) - stubbed
    assert not missing, f"Names missing from detection.pyi: {sorted(missing)}"


# ---------------------------------------------------------------------------
# The functional layer
# ---------------------------------------------------------------------------


@functools.cache
def _stub_function_names(filename: str) -> set[str]:
    """Top-level `def` names declared in a sibling stub file."""
    tree = ast.parse((STUB_PATH.parent / filename).read_text())
    return {n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef)}


def test_functional_layer_importable_both_ways():
    """`import hotcoco.metrics` must work, not only `from hotcoco import metrics`."""

    import hotcoco.metrics
    import hotcoco.primitives
    from hotcoco import metrics, primitives

    assert "hotcoco.metrics" in sys.modules
    assert "hotcoco.primitives" in sys.modules
    assert metrics is hotcoco.metrics
    assert primitives is hotcoco.primitives


def test_metrics_stub_matches_runtime():
    from hotcoco import metrics

    missing = set(metrics.__all__) - _stub_function_names("metrics.pyi")
    assert not missing, f"Names missing from metrics.pyi: {sorted(missing)}"


def test_metrics_facade_reexports_every_extension_function():
    """The facade is hand-maintained, so check the *other* direction too.

    `test_metrics_stub_matches_runtime` asserts stub >= __all__. Without this,
    adding a #[pyfunction] to crates/hotcoco-pyo3/src/metrics.rs and forgetting
    the two-line edit in metrics.py leaves it unreachable from Python with the
    whole suite green.
    """
    from hotcoco import metrics
    from hotcoco.hotcoco import metrics as _ext

    exported = {n for n in dir(_ext) if not n.startswith("_")}
    missing = exported - set(metrics.__all__)
    assert not missing, f"In the extension but not re-exported by metrics.py: {sorted(missing)}"


def test_primitives_stub_matches_runtime():
    from hotcoco import primitives

    missing = set(primitives.__all__) - _stub_function_names("primitives.pyi")
    assert not missing, f"Names missing from primitives.pyi: {sorted(missing)}"


def test_functional_layer_needs_no_evaluator():
    """The whole point: metric functions callable on bare arrays.

    If these ever start requiring a COCOeval, the functional layer has collapsed
    back into the god object 1.0 pulled them out of.
    """
    from hotcoco import metrics, primitives

    assert metrics.average_precision([0.9, 0.1], [True, False], num_gt=2) > 0.0
    ece, mce = metrics.calibration_error([0.9] * 10, [True] * 5 + [False] * 5)
    assert ece == mce  # single occupied bin
    assert metrics.confusion_matrix([0, None], [0, 1], num_classes=2).shape == (3, 3)
    assert len(metrics.calibration_curve([0.5], [True], n_bins=4)) == 4
    rows, _ = primitives.lsap([[1.0, 2.0], [3.0, 4.0]])
    assert len(rows) == 2


def test_metric_functions_accept_numpy_arrays():
    """The module docstring advertises "lists or numpy arrays" — hold it to that.

    float64/bool arrays take the fast path in the bindings; float32 and strided
    views take the per-element fallback. All four must produce exactly the
    answer the list path produces.
    """
    from hotcoco import metrics

    scores = [0.9, 0.8, 0.7, 0.3]
    matched = [True, False, True, True]
    expected_ap = metrics.average_precision(scores, matched, num_gt=4)
    expected_cal = metrics.calibration_error(scores, matched)

    # Fast path: float64 and bool ndarrays.
    np_scores = np.array(scores)
    np_matched = np.array(matched)
    assert metrics.average_precision(np_scores, np_matched, num_gt=4) == expected_ap
    assert metrics.calibration_error(np_scores, np_matched) == expected_cal

    # Fallback path: float32 still works, at the old per-element cost.
    assert metrics.average_precision(np_scores.astype(np.float32), np_matched, num_gt=4) == expected_ap

    # Strided views must be read honestly, not rejected or miscopied.
    every_other = metrics.average_precision(np_scores[::2], np_matched[::2], num_gt=2)
    assert every_other == metrics.average_precision(scores[::2], matched[::2], num_gt=2)

    # Optional array parameters go through the same conversion.
    ig = np.array([False, True, False, False])
    assert metrics.average_precision(np_scores, np_matched, num_gt=4, ignored=ig) == metrics.average_precision(
        scores, matched, num_gt=4, ignored=ig.tolist()
    )

    curve_list = metrics.precision_recall_curve([1.0, 2.0], [0.0, 1.0], num_gt=3)
    curve_np = metrics.precision_recall_curve(np.array([1.0, 2.0]), np.array([0.0, 1.0]), num_gt=3)
    assert curve_list == curve_np

    # A 2-D array is not a flat argument list — it must raise, not flatten.

    with pytest.raises(TypeError):
        metrics.average_precision(np.zeros((2, 2)), np_matched, num_gt=4)


def test_metric_functions_reject_mismatched_arrays():
    """Parallel arrays of different lengths are a caller bug, not a silent truncation."""
    from hotcoco import metrics

    with pytest.raises(ValueError):
        metrics.average_precision([0.9, 0.8], [True], num_gt=1)
    with pytest.raises(ValueError):
        metrics.calibration_error([0.9, 0.8], [True])
    with pytest.raises(ValueError):
        metrics.confusion_matrix([0, 1], [0], num_classes=2)


def test_lsap_rejects_ragged_and_nan():

    from hotcoco import primitives

    with pytest.raises(ValueError):
        primitives.lsap([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError):
        primitives.lsap([[1.0, math.nan], [3.0, 4.0]])


def test_lvis_dropin_matches_lvis_api_spelling():
    """`from lvis import LVISEval` must work — capital E, as lvis-api spells it.

    hotcoco names the class `LVISeval` after pycocotools' `COCOeval`, but
    lvis-api exports `LVISEval`, and that is what Detectron2 and MMDetection
    import. Without the alias, `init_as_lvis()` registered a `lvis` module that
    the canonical import could not use.
    """

    hotcoco.init_as_lvis()
    from lvis import LVIS, LVISEval, LVISResults

    assert LVISEval is hotcoco.LVISeval
    assert LVIS is hotcoco.COCO
    assert LVISResults is hotcoco.LVISResults


def test_dropin_supports_the_import_as_binding_form():
    """`import pycocotools.coco as pc` must work, not only `from … import`.

    The `import a.b as x` form binds via `getattr(a, "b")`, not
    `sys.modules["a.b"]`, so the sys.modules patch alone leaves it broken —
    found by the 1.0 third-party-consumer smoke test. The patch functions now
    also set the submodule names as attributes on the hotcoco module.

    The patch is undone on exit: unlike `lvis`, `pycocotools` is really
    installed in this venv, and the differential-parity tests in this same
    pytest process must keep importing the real one.
    """

    saved = {name: sys.modules.get(name) for name in list(sys.modules) if name.split(".")[0] in ("pycocotools", "lvis")}
    try:
        hotcoco.init_as_pycocotools()
        import pycocotools.coco as pc
        import pycocotools.cocoeval as pce
        import pycocotools.mask as pm

        assert pc.COCO is hotcoco.COCO
        assert pce.COCOeval is hotcoco.COCOeval
        assert pm.encode is hotcoco.mask.encode

        hotcoco.init_as_lvis()
        import lvis.eval as le

        assert le.LVISEval is hotcoco.LVISeval
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in ("pycocotools", "lvis"):
                del sys.modules[name]
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod


def _tiny_dataset():
    return {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }


def test_query_methods_accept_scalar_ids_like_pycocotools():
    """`coco.getAnnIds(img_id)` with a bare int must work — `_isArrayLike` parity.

    torchvision's ``CocoDetection`` calls exactly that; found by the 1.0
    third-party-consumer smoke test.
    """

    coco = hotcoco.COCO(_tiny_dataset())
    assert coco.get_ann_ids(1) == coco.get_ann_ids([1]) == [1, 2]
    assert coco.getAnnIds(1) == [1, 2]
    assert coco.get_img_ids(1, 1) == [1]
    assert coco.load_anns(1)[0]["id"] == 1
    assert coco.loadImgs(1)[0]["id"] == 1
    # A bare string must not be split into characters.
    assert coco.get_cat_ids("person") == [1]
    assert coco.getCatIds(catNms="dog") == [2]
    # The camelCase aliases must accept pycocotools' *keyword* spellings —
    # Detectron2 calls `getAnnIds(imgIds=…)`.
    assert coco.getAnnIds(imgIds=1) == [1, 2]
    assert coco.getAnnIds(imgIds=[1], catIds=[2]) == [2]
    assert coco.getImgIds(catIds=1) == [1]


def test_dataset_assignment_construction_flow():
    """`coco = COCO(); coco.dataset = d; coco.createIndex()` must work.

    That is how pycocotools consumers construct in-memory datasets —
    torchmetrics' pycocotools backend uses it verbatim, with image entries
    that carry only an ``id``. Found by the 1.0 smoke test.
    """

    coco = hotcoco.COCO()
    ds = _tiny_dataset()
    ds["images"] = [{"id": 1}]  # torchmetrics builds images as bare ids
    coco.dataset = ds
    coco.createIndex()
    assert coco.get_img_ids() == [1]
    assert coco.get_ann_ids(1) == [1, 2]


def test_cocoeval_accepts_pycocotools_constructor_keywords():
    """`COCOeval(cocoGt=gt, cocoDt=dt, iouType="bbox")` must work.

    pycocotools spells the keywords camelCase and torchmetrics passes
    ``iouType=``; found by the 1.0 smoke test. Mixing both spellings of one
    argument is an error, as is an unknown keyword.
    """

    gt = hotcoco.COCO(_tiny_dataset())
    dt = gt.load_res([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9}])

    ev = hotcoco.COCOeval(cocoGt=gt, cocoDt=dt, iouType="bbox")
    ev.run()
    assert ev.stats[0] >= 0.0

    with pytest.raises(TypeError):
        hotcoco.COCOeval(gt, dt, iou_type="bbox", iouType="bbox")
    with pytest.raises(TypeError):
        hotcoco.COCOeval(cocoGt=gt, cocoDt=dt, iouType="bbox", bogus=1)
    with pytest.raises(TypeError):
        hotcoco.COCOeval(cocoGt=gt, cocoDt=dt)
