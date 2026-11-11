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
from hotcoco import COCO, CallOrderWarning, COCOeval, Hierarchy, ReferenceDeviationWarning


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
