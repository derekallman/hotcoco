"""Python-side pins for what a step reads belonging to the current run.

The Rust tests in ``crates/hotcoco/tests/run_state.rs`` prove the mechanics;
these check what a Python user sees — the exception, the warning, the number —
and pin that pycocotools keeps a stale ``E.stats`` after a new ``evaluate()``,
the behavior hotcoco's drop-in attributes still share. No ``data/`` needed.
"""

import contextlib
import io
import warnings

import pytest
from hotcoco import COCO, CallOrderWarning, COCOeval, Hierarchy, ReferenceDeviationWarning, StreamingEval


def two_category_dataset():
    """Ten images, two categories: category 1 detected exactly, category 2
    shifted far enough to miss at the stricter thresholds."""
    images = [{"id": i, "width": 100, "height": 100} for i in range(1, 11)]
    gt, dt = [], []
    for img in range(1, 11):
        x = 5.0 * img
        box = [x, 10.0, 30.0, 30.0]
        for cat in (1, 2):
            gt.append(
                {"id": 2 * img - 2 + cat, "image_id": img, "category_id": cat, "bbox": box, "area": 900, "iscrowd": 0}
            )
        dt.append({"image_id": img, "category_id": 1, "bbox": box, "score": 0.9})
        dt.append({"image_id": img, "category_id": 2, "bbox": [x + 6.0, 10.0, 30.0, 30.0], "score": 0.8})
    categories = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    return {"images": images, "annotations": gt, "categories": categories}, dt


def quiet(fn, *args):
    """Call ``fn`` with its stdout swallowed and its warnings silenced."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args)


def summarized(ev, cat_ids=None):
    if cat_ids is not None:
        ev.params.catIds = cat_ids
    quiet(ev.evaluate)
    quiet(ev.accumulate)
    quiet(ev.summarize)
    return ev


def fresh():
    gt_dict, dt_list = two_category_dataset()
    gt = COCO(gt_dict)
    return COCOeval(gt, gt.loadRes(dt_list), "bbox")


def run(cat_ids=None):
    return summarized(fresh(), cat_ids)


def test_a_new_evaluate_retires_the_previous_summary():
    ev = run()
    cat_1_alone = run([1]).results()["metrics"]["AP"]
    assert ev.results()["metrics"]["AP"] != pytest.approx(cat_1_alone)

    ev.params.catIds = [1]
    ev.evaluate()
    with pytest.raises(RuntimeError, match=r"evaluate\(\) has run since the last accumulate\(\)"):
        ev.results()
    with pytest.raises(RuntimeError, match=r"call accumulate\(\) and summarize\(\) before report\(\)"):
        ev.report()
    # The warnings carry core's reason, the one `results()` raises with.
    reason = r"evaluate\(\) has run since the last accumulate\(\), so call accumulate\(\)"
    with pytest.warns(CallOrderWarning, match=r"get_results\(\) called before summarize\(\).*" + reason):
        assert ev.get_results() == {}
    with pytest.warns(CallOrderWarning, match=r"f_scores\(\) called before accumulate\(\).*" + reason):
        assert ev.f_scores() == {}
    with pytest.warns(CallOrderWarning, match="No results to print.*" + reason):
        ev.print_results()

    ev.accumulate()
    with pytest.raises(RuntimeError, match=r"call summarize\(\) before results\(\)"):
        ev.results(per_class=True)
    quiet(ev.summarize)
    assert ev.results()["metrics"]["AP"] == pytest.approx(cat_1_alone)


def test_params_changed_after_a_run_retire_it():
    ev = run()
    f1 = ev.f_scores()
    ev.params.areaRng = ev.params.areaRng[::-1]
    ev.params.areaRngLbl = ev.params.areaRngLbl[::-1]
    with pytest.raises(RuntimeError, match=r"params\.area_ranges changed after accumulate\(\)"):
        ev.results()
    with pytest.warns(CallOrderWarning, match=r"f_scores\(\) called before"):
        assert ev.f_scores() == {}

    quiet(ev.evaluate)
    quiet(ev.accumulate)
    assert ev.f_scores() == f1


def test_print_results_sees_params_edited_from_python():
    ev = run()
    ev.params.iouThrs = ev.params.iouThrs[::-1]
    with pytest.warns(CallOrderWarning, match="No results to print"):
        ev.print_results()


def test_summarize_warns_about_the_params_it_reports():
    """The table comes from the params ``accumulate()`` recorded, and so do the
    reference-deviation warnings printed above it."""
    ev = fresh()
    ev.evaluate()
    ev.accumulate()
    ev.params.maxDets = [1, 10, 300]
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()) as out:
        warnings.simplefilter("error", ReferenceDeviationWarning)
        ev.summarize()
    assert "maxDets=100" in out.getvalue()


@pytest.mark.parametrize("method", ["summarize", "summary_lines"])
def test_summarizing_a_retired_accumulation_warns_and_keeps_pycocotools_numbers(method):
    """After a new ``evaluate()``, ``summarize()`` still reports the last
    ``accumulate()``, as pycocotools does, and says so."""
    ev = run()
    before = list(ev.stats)
    ev.params.catIds = [1]
    quiet(ev.evaluate)
    with pytest.warns(
        CallOrderWarning,
        match=rf"{method}\(\) reports the last accumulate\(\), which does not describe the current run: "
        r"evaluate\(\) has run since the last accumulate\(\)",
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            getattr(ev, method)()
    assert list(ev.stats) == before


def test_summarizing_the_current_run_does_not_warn():
    ev = fresh()
    ev.evaluate()
    ev.accumulate()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("error", CallOrderWarning)
        ev.summarize()
        ev.summary_lines()


def test_a_fresh_run_reports_as_before():
    """The gate is invisible to the pipeline everyone runs."""
    ev = run()
    with warnings.catch_warnings():
        warnings.simplefilter("error", CallOrderWarning)
        assert ev.get_results()["AP"] == ev.stats[0]
        assert ev.results(per_class=True)["per_class"]
        assert set(ev.f_scores()) == {"F1", "F1_50", "F1_75"}
        ev.report()


def test_stats_keeps_pycocotools_behavior_after_a_new_evaluate():
    """``E.stats`` is pycocotools' attribute: after a new ``evaluate()`` both keep
    the earlier run's array. Deliberately left that way for drop-in compatibility,
    and open to revisiting; when it changes, this test changes with it."""
    pycocotools_coco = pytest.importorskip("pycocotools.coco")
    pycocotools_eval = pytest.importorskip("pycocotools.cocoeval")
    gt_dict, dt_list = two_category_dataset()
    gt = quiet(pycocotools_coco.COCO)
    gt.dataset = gt_dict
    quiet(gt.createIndex)
    reference = summarized(pycocotools_eval.COCOeval(gt, quiet(gt.loadRes, dt_list), "bbox"))
    for ev in (run(), reference):
        before = list(ev.stats)
        ev.params.catIds = [1]
        quiet(ev.evaluate)
        assert list(ev.stats) == before, type(ev).__module__


def test_open_images_expand_dt_applies_to_the_run_it_is_set_for():
    box = [10, 10, 30, 30]
    images = [{"id": i, "width": 100, "height": 100} for i in range(1, 5)]
    gt = COCO(
        {
            "images": images,
            "categories": [{"id": 1, "name": "animal"}, {"id": 2, "name": "dog"}],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 2, "bbox": box, "area": 900, "iscrowd": 0} for i in range(1, 5)
            ],
        }
    )
    dt = gt.loadRes([{"image_id": i, "category_id": 2, "bbox": box, "score": 0.9} for i in range(1, 5)])
    hierarchy = Hierarchy.from_parent_map({2: 1})

    def ap(ev):
        quiet(ev.run)
        return ev.stats[0]

    def fresh(expand_dt):
        ev = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
        ev.params.expand_dt = expand_dt
        return ap(ev)

    without, with_ = fresh(False), fresh(True)
    assert without != with_

    ev = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev.params.expand_dt = True
    assert ap(ev) == with_
    ev.params.expand_dt = False
    assert ap(ev) == without
    assert len(ev.cocoDt.dataset["annotations"]) == 4


def analyses(ev):
    tide = ev.tide_errors(pos_thr=0.75)
    calibration = ev.calibration(iou_threshold=0.75)
    diagnostics = ev.image_diagnostics(iou_thr=0.75)
    return (
        tide["ap_base"],
        tide["counts"],
        calibration["ece"],
        calibration["num_detections"],
        len(diagnostics["dt_status"]),
    )


@pytest.mark.parametrize(
    "name, value", [("catIds", [1]), ("imgIds", [1, 2]), ("maxDets", [1, 2, 3]), ("areaRng", [[0, 1e10], [0, 5]])]
)
def test_analyses_answer_for_the_params_evaluate_saw(name, value):
    """``params`` edited after ``evaluate()``: TIDE mixed the new ``catIds``
    with misses counted over the old, and a new ``maxDets`` emptied all three,
    calibration reporting an ``ece`` of 0.0 over no detections."""
    ev = fresh()
    quiet(ev.evaluate)
    before = analyses(ev)
    assert before[3] == 20 and before[4] == 20
    setattr(ev.params, name, value)
    assert analyses(ev) == before


def test_tide_refuses_a_pooled_run():
    """``useCats = 0`` has no category to tell Cls from Loc by; TIDE returned
    an ``ap_base`` of 0.0 and calibration listed category 18446744073709551615."""
    ev = fresh()
    ev.params.useCats = 0
    quiet(ev.evaluate)
    with pytest.raises(ValueError, match="use_cats"):
        ev.tide_errors()
    calibration = ev.calibration()
    assert calibration["num_detections"] == 20
    assert calibration["per_category"] == {}


def test_accumulate_reads_img_ids_edited_after_evaluate():
    """``imgIds`` narrowed between ``evaluate()`` and ``accumulate()`` was
    ignored: every image's AP under params that named two."""
    gt_dict, dt_list = two_category_dataset()
    # Category 2 is detected on images 1 and 2 only.
    dt_list = [d for d in dt_list if d["category_id"] == 1 or d["image_id"] <= 2]
    gt = COCO(gt_dict)

    def ap(evaluated, accumulated):
        ev = COCOeval(gt, gt.loadRes(dt_list), "bbox")
        ev.params.imgIds = evaluated
        quiet(ev.evaluate)
        ev.params.imgIds = accumulated
        quiet(ev.accumulate)
        quiet(ev.summarize)
        return ev.stats[0]

    everything = list(range(1, 11))
    whole, two = ap(everything, everything), ap([1, 2], [1, 2])
    assert abs(whole - two) > 0.1
    assert ap(everything, [1, 2]) == two
    assert ap(everything, everything[::-1]) == whole


def test_a_finalized_stream_cannot_be_evaluated_again():
    """``finalize().run()`` matched the evaluator's empty datasets and replaced
    the streamed AP with -1.0, silently."""
    gt_dict, dt_list = two_category_dataset()
    se = StreamingEval(gt_dict["categories"])
    se.update(gt_dict["images"], gt_dict["annotations"], dt_list)
    ev = se.finalize()
    quiet(ev.accumulate)
    quiet(ev.summarize)
    streamed = ev.stats[0]
    assert streamed > 0
    for call in (ev.evaluate, ev.run, ev.tide_errors, ev.calibration, ev.image_diagnostics):
        with pytest.raises(RuntimeError, match="StreamingEval"):
            call()
    assert ev.stats[0] == streamed
    assert ev.results()["metrics"]["AP"] == streamed


def test_open_images_confusion_matrix_is_the_same_after_evaluate():
    """``evaluate()`` swaps in hierarchy-expanded datasets, and the matrix
    matched against those: two perfect detections read as each ground truth
    missed and its parent's copy predicted as the child."""
    gt = COCO(
        {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "categories": [
                {"id": 2, "name": "dog", "supercategory": "animal"},
                {"id": 3, "name": "cat", "supercategory": "animal"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 3, "bbox": [60, 60, 30, 30], "area": 900, "iscrowd": 0},
            ],
        }
    )
    dt = gt.loadRes(
        [
            {"image_id": 1, "category_id": 2, "bbox": [10, 10, 30, 30], "score": 0.9},
            {"image_id": 1, "category_id": 3, "bbox": [60, 60, 30, 30], "score": 0.8},
        ]
    )
    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    before = ev.confusion_matrix()
    assert before["matrix"].tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    quiet(ev.evaluate)
    after = ev.confusion_matrix()
    by_id = dict(zip(after["cat_ids"], range(len(after["cat_ids"])), strict=True))
    dog, cat = by_id[2], by_id[3]
    assert after["matrix"][dog, dog] == 1 and after["matrix"][cat, cat] == 1
    assert after["matrix"].sum() == 2


def test_tide_classifies_at_the_threshold_it_matched_at():
    """``pos_thr`` snaps to the grid for the matches; the false positives were
    still classified at the value passed, and the result reported that one."""
    gt_dict, dt_list = two_category_dataset()
    gt = COCO(gt_dict)

    def tide(thrs, **kwargs):
        ev = COCOeval(gt, gt.loadRes(dt_list), "bbox")
        ev.params.iouThrs = thrs
        quiet(ev.evaluate)
        result = ev.tide_errors(**kwargs)
        return result["pos_thr"], result["ap_base"], result["counts"]

    snapped = tide([0.7, 0.75])
    assert snapped[0] == 0.7
    assert snapped == tide([0.5, 0.7], pos_thr=0.7)


@pytest.mark.parametrize("name", ["maxDets", "areaRng"])
def test_an_empty_axis_is_an_error_not_a_panic(name):
    """Each reader indexed the missing slot and panicked."""
    ev = fresh()
    setattr(ev.params, name, [])
    if name == "areaRng":
        ev.params.areaRngLbl = []
    quiet(ev.evaluate)
    quiet(ev.accumulate)
    quiet(ev.summarize)
    for read in (ev.report, ev.results, lambda: ev.results(per_class=True)):
        with pytest.raises(ValueError, match="is empty"):
            read()
    assert quiet(ev.f_scores) == {}
    assert quiet(ev.get_results) == {}
    if name == "areaRng":
        assert ev.calibration()["num_detections"] == 0
        assert ev.image_diagnostics()["dt_status"] == {}


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        (lambda p: setattr(p, "iouThrs", [0.5, 0.6]), r"params\.iou_thrs 0\.6"),
        (lambda p: setattr(p, "areaRng", [[0, 1e10], [0, 7]]), r"params\.area_ranges \[0, 7\]"),
        (lambda p: setattr(p, "maxDets", [1, 500]), r"params\.max_dets 500 \(evaluate\(\) kept 10 per image\)"),
        (lambda p: setattr(p, "catIds", [1, 2, 99]), r"params\.cat_ids 99"),
        (lambda p: setattr(p, "imgIds", [1, 2, 77]), r"params\.img_ids 77"),
        (lambda p: setattr(p, "useCats", 0), r"params\.use_cats false"),
    ],
    ids=["iou_thrs", "area_ranges", "max_dets", "cat_ids", "img_ids", "use_cats"],
)
def test_accumulate_warns_about_params_evaluate_did_not_run(edit, named):
    """The slots stay -1 and drop out of every mean, which nothing else says."""
    ev = fresh()
    ev.params.iouThrs = [0.5, 0.75]
    ev.params.maxDets = [1, 10]
    ev.evaluate()
    edit(ev.params)
    if "area" in named:
        ev.params.areaRngLbl = ["all", "tiny"]
    with pytest.warns(CallOrderWarning, match=r"accumulate\(\) was asked for " + named + r".*Call evaluate\(\) again"):
        ev.accumulate()


def test_accumulate_is_quiet_about_params_narrowed_or_reordered():
    ev = fresh()
    ev.evaluate()
    ev.params.iouThrs = [0.75, 0.5]
    ev.params.maxDets = [10, 1]
    ev.params.catIds = [2]
    ev.params.imgIds = [1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ev.accumulate()
