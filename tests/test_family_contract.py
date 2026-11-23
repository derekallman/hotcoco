"""What every evaluator family promises about its readers.

Two rules, checked for every family at once so a new family cannot go its own
way:

1. A reader called before the evaluator has anything to report raises
   ``RuntimeError``. The exceptions are the ``COCOeval`` members that copy
   pycocotools, which callers of pycocotools rely on; ``PYCOCOTOOLS_READERS``
   names each one.
2. ``results()`` carries the same three keys beside the family's numbers:
   ``hotcoco_version``, ``provenance`` and ``reference_deviations``.

``test_every_reader_has_a_row`` fails on a public member that is in neither
list, so a new reader has to be classified before it ships.
"""

from __future__ import annotations

import warnings

import hotcoco
import numpy as np
import pytest
from hotcoco import COCO, COCOeval, __version__, panoptic, semantic, tracking

IMAGES = [{"id": 1, "width": 100, "height": 100}]
CATS = [{"id": 1, "name": "thing", "isthing": 1}]
GT = {
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [10, 10, 40, 40],
    "area": 1600,
    "iscrowd": 0,
    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
}
DT = {"image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "score": 0.9}


def _gt():
    return COCO({"images": IMAGES, "annotations": [GT], "categories": CATS})


def _detection():
    gt = _gt()
    return COCOeval(gt, gt.load_res([DT]), "bbox")


def _run_detection(ev):
    ev.evaluate()
    ev.accumulate()
    ev.summarize()


def _panoptic():
    return panoptic.PanopticEval(_gt(), _gt())


def _semantic():
    return semantic.SemanticEval(["a", "b"])


def _update_semantic(ev):
    ev.update(np.array([[0, 1]], dtype=np.uint8), np.array([[0, 0]], dtype=np.uint8))


def _tracking():
    return tracking.TrackingEval("MOT17")


def _update_tracking(ev):
    gt = np.array([[1, 1, 0, 0, 10, 10, 1, 1, 1]], dtype=np.float64)
    pred = np.array([[1, 5, 0, 0, 10, 10, 0.9]], dtype=np.float64)
    ev.update("seq", gt, pred, num_frames=1)


# family -> (a fresh evaluator, the step that gives it something to report)
FAMILIES = {
    "detection": (_detection, _run_detection),
    "panoptic": (_panoptic, lambda ev: ev.evaluate()),
    "semantic": (_semantic, _update_semantic),
    "tracking": (_tracking, _update_tracking),
}

# Readers of a result: each raises ``RuntimeError`` before there is one.
RESULT_READERS = {
    "detection": ["results", "report", "save_results"],
    "panoptic": ["stats", "results", "report", "summarize", "summary_lines", "eval_imgs"],
    "semantic": ["stats", "results", "report", "summarize", "summary_lines"],
    "tracking": ["compute", "sequence_curves"],
}
RESULT_READERS["semantic"].append("compute")

# ``COCOeval`` members that answer before a run the way pycocotools' do, or
# with an empty value and a ``CallOrderWarning``; ``tests/test_run_state.py``
# and ``tests/test_warning_categories.py`` pin what each returns.
PYCOCOTOOLS_READERS = {
    "stats": "pycocotools: an empty list until summarize()",
    "eval": "pycocotools: empty until accumulate()",
    "evalImgs": "pycocotools: empty until evaluate()",
    "eval_imgs": "the snake_case spelling of evalImgs",
    "ious": "pycocotools: empty until evaluate()",
    "summarize": "pycocotools prints nothing useful and returns; hotcoco warns",
    "summary_lines": "summarize() without the print",
    "accumulate": "pycocotools prints a message and returns; hotcoco warns",
    "get_results": "shipped returning {} with a warning",
    "print_results": "shipped printing nothing with a warning",
    "f_scores": "shipped returning {} with a warning",
}

# Members that are not readers of a result: steps of the pipeline, the
# configuration, the inputs, and analyses with their own tests.
NOT_RESULT_READERS = {
    "detection": {
        "evaluate",
        "run",
        "params",
        "cocoGt",
        "cocoDt",
        "coco_gt",
        "coco_dt",
        "provenance",
        "reference_deviations",
        "is_benchmark_standard",
        "metric_keys",
        "metric_defs",
        "virtual_cat_names",
        # Analyses: each raises RuntimeError before evaluate(), in test_run_state.py.
        "calibration",
        "confusion_matrix",
        "image_diagnostics",
        "slice_by",
        "tide_errors",
    },
    "panoptic": {"evaluate", "run", "provenance", "reference_deviations"},
    "semantic": {
        "update",
        "update_png",
        "update_folders",
        "merge",
        "provenance",
        "reference_deviations",
        # The evaluator's own state, true at any time: an empty matrix is what
        # has been counted.
        "confusion",
        "n_images",
        "num_classes",
        "class_names",
    },
    "tracking": {"update", "update_folders", "merge", "benchmark", "clear_threshold", "identity_threshold"},
}
# Panoptic's `compute()` evaluates first when `evaluate()` has not run, so it
# is a step, not a reader that raises.
NOT_RESULT_READERS["panoptic"].add("compute")


def _read(ev, name, tmp_path):
    member = getattr(ev, name)
    if not callable(member):
        return member
    if name == "save_results":
        return member(str(tmp_path / "results.json"))
    if name == "sequence_curves":
        return member("seq")
    return member()


CASES = [(family, name) for family, names in RESULT_READERS.items() for name in names]


@pytest.mark.parametrize(("family", "name"), CASES, ids=[f"{f}-{n}" for f, n in CASES])
def test_a_reader_raises_before_there_is_a_result(family, name, tmp_path):
    make, _ = FAMILIES[family]
    with pytest.raises(RuntimeError, match="before"):
        _read(make(), name, tmp_path)


@pytest.mark.parametrize(("family", "name"), CASES, ids=[f"{f}-{n}" for f, n in CASES])
def test_the_same_reader_answers_after(family, name, tmp_path):
    """Otherwise the test above passes on a reader that always raises."""
    make, step = FAMILIES[family]
    ev = make()
    step(ev)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _read(ev, name, tmp_path)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_reader_has_a_row(family):
    make, _ = FAMILIES[family]
    public = {n for n in dir(type(make())) if not n.startswith("_")}
    known = set(RESULT_READERS[family]) | NOT_RESULT_READERS[family]
    if family == "detection":
        known |= set(PYCOCOTOOLS_READERS)
    assert public - known == set(), "classify the new member in this file"
    assert known - public == set(), "a row names a member that is gone"


@pytest.mark.parametrize("family", ["detection", "panoptic", "semantic"])
def test_results_carries_the_same_three_keys(family):
    make, step = FAMILIES[family]
    ev = make()
    step(ev)
    results = ev.results()
    assert results["hotcoco_version"] == __version__
    assert results["provenance"] == ev.provenance()
    assert results["reference_deviations"] == ev.reference_deviations()


def test_the_three_keys_report_a_deviation():
    """A list that is always empty would pass the test above."""
    ev = tracking.TrackingEval()  # no benchmark: a ground-truth row with conf 0 is kept
    gt = np.array([[1, 1, 0, 0, 10, 10, 0, 1, 1]], dtype=np.float64)
    pred = np.array([[1, 5, 0, 0, 10, 10, 0.9]], dtype=np.float64)
    ev.update("seq", gt, pred, num_frames=1)
    with pytest.warns(hotcoco.ReferenceDeviationWarning, match="conf"):
        report = ev.compute()
    assert report.provenance == "extension"
    assert len(report.deviations) == 1

    gt = _gt()
    det = COCOeval(gt, gt.load_res([DT]), "bbox")
    det.params.iou_thrs = [0.5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run_detection(det)
    results = det.results()
    assert results["provenance"] == "extension"
    assert results["reference_deviations"] == det.reference_deviations() != []


# ---------------------------------------------------------------------------
# compute() returns one Report, the same shape in every family
# ---------------------------------------------------------------------------

REPORT_MEMBERS = {
    "task",
    "provenance",
    "hotcoco_version",
    "deviations",
    "metrics",
    "per_class",
    "per_group",
    "curves",
    "params",
    "to_dict",
    "save",
    "load",
}
COMPUTING = ["panoptic", "semantic", "tracking"]


def _computed(family):
    make, step = FAMILIES[family]
    ev = make()
    step(ev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", hotcoco.ReferenceDeviationWarning)
        return ev, ev.compute()


@pytest.mark.parametrize("family", COMPUTING)
def test_compute_returns_the_one_report(family):
    ev, report = _computed(family)
    assert type(report) is hotcoco.Report
    assert {n for n in dir(report) if not n.startswith("_")} == REPORT_MEMBERS
    assert report.task == family
    assert report.hotcoco_version == __version__
    assert report.metrics is report.metrics, "the same dict on every read"
    assert all(v is None or isinstance(v, float) for v in report.metrics.values())
    assert all(isinstance(c, np.ndarray) and c.dtype == np.float64 for c in report.curves.values())


@pytest.mark.parametrize("family", ["panoptic", "semantic"])
def test_a_report_agrees_with_the_readers_it_replaces(family):
    """`compute()` is the readers in one object: the table `summary_lines()` gives,
    the deviations, and `report()`'s metrics with None for its -1."""
    ev, report = _computed(family)
    assert str(report).splitlines() == ev.summary_lines()
    assert report.deviations == ev.reference_deviations()
    assert report.provenance == ev.provenance()
    legacy = ev.report()
    assert report.metrics == {k: (None if v == -1.0 else v) for k, v in legacy["metrics"].items()}


def test_not_computed_is_none_in_a_report_and_minus_one_in_report():
    """An empty stuff split is "not computed". `report()` keeps the -1 it always
    returned; the Report says None, which fails loudly when averaged."""
    ev = _panoptic()
    ev.evaluate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", hotcoco.ReferenceDeviationWarning)
        report = ev.compute()
    assert ev.report()["metrics"]["PQ_st"] == -1.0
    assert report.metrics["PQ_st"] is None
    assert report.metrics["PQ_th"] is not None


def test_a_negative_mota_stays_a_number():
    """MOTA can be negative; "not computed" is decided where a family makes its
    numbers, never by the sign, so a negative MOTA is not None."""
    ev = tracking.TrackingEval()
    gt = np.array([[1, 1, 0, 0, 10, 10, 1, 1, 1]], dtype=np.float64)
    pred = np.array([[1, k, 50 * k, 50 * k, 10, 10, 0.9] for k in range(2, 6)], dtype=np.float64)
    ev.update("seq", gt, pred, num_frames=1)
    report = ev.compute()
    assert report.metrics["MOTA"] is not None and report.metrics["MOTA"] < 0
    assert report.per_group["seq"]["MOTA"] == report.metrics["MOTA"]


@pytest.mark.parametrize("family", COMPUTING)
def test_a_report_round_trips(family, tmp_path):
    import pickle

    _, report = _computed(family)
    path = tmp_path / "report.json"
    report.save(path)
    loaded = hotcoco.Report.load(path)
    assert loaded.to_dict() == report.to_dict()
    assert str(loaded) == str(report)
    assert pickle.loads(pickle.dumps(report)).to_dict() == report.to_dict()


def test_a_report_has_no_public_constructor():
    """Provenance is the producer's: a caller cannot make a Report that claims a
    parity check nobody ran."""
    with pytest.raises(TypeError):
        hotcoco.Report()


def test_report_keeps_its_shape_through_1_5():
    """`report()` returns the dict it always did, -1 included, so a caller does not
    see a value change type under the same name. It is deprecated in 1.6."""
    ev = _detection()
    _run_detection(ev)
    legacy = ev.report()
    assert set(legacy) == {"task", "provenance", "metrics", "per_class", "per_group", "curves", "params"}
    assert legacy["metrics"]["APl"] == -1.0
