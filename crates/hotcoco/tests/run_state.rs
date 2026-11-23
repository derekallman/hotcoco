//! What a step reads belongs to the current run.
//!
//! `evaluate()`, `accumulate()`, and `summarize()` each leave output behind
//! that the next one reads, and a new `evaluate()` leaves the old output in
//! place, as pycocotools' attributes do. These tests pin that hotcoco's own
//! readers of that output — `report()`, `results()`, `get_results()`,
//! `f_scores()`, `print_results_lines()` — answer only for the current run,
//! never for an earlier one, and never index an accumulation with `params`
//! edited after it. Each test is a reproducer from the 2026-11 audit of
//! implicit state machines, where the answer used to be a plausible number
//! from the wrong run.

#![allow(clippy::unwrap_used)]

use std::collections::HashMap;

use hotcoco::error::Error;
use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image};
use hotcoco::{COCO, COCOeval, Hierarchy, Params, StreamingEval};

fn image(id: u64) -> Image {
    Image {
        id,
        width: 100,
        height: 100,
        ..Default::default()
    }
}

fn cat(id: u64) -> Category {
    Category {
        id,
        name: format!("cat_{id}"),
        ..Default::default()
    }
}

fn gt_box(id: u64, image_id: u64, category_id: u64, bbox: [f64; 4]) -> Annotation {
    Annotation {
        id,
        image_id,
        category_id,
        bbox: Some(bbox),
        area: Some(bbox[2] * bbox[3]),
        ..Default::default()
    }
}

fn det_box(image_id: u64, category_id: u64, bbox: [f64; 4], score: f64) -> Annotation {
    Annotation {
        image_id,
        category_id,
        bbox: Some(bbox),
        score: Some(score),
        ..Default::default()
    }
}

/// Ten images, two categories: category 1 detected exactly, category 2
/// shifted far enough to miss at the stricter thresholds. So AP over both
/// categories and AP over category 1 alone differ by a wide margin.
fn two_category_pair() -> (COCO, COCO) {
    let mut gt_anns = Vec::new();
    let mut dt_anns = Vec::new();
    for img in 1..=10u64 {
        let x = 5.0 * img as f64;
        let b = [x, 10.0, 30.0, 30.0];
        gt_anns.push(gt_box(2 * img - 1, img, 1, b));
        gt_anns.push(gt_box(2 * img, img, 2, b));
        dt_anns.push(det_box(img, 1, b, 0.9));
        dt_anns.push(det_box(img, 2, [x + 6.0, 10.0, 30.0, 30.0], 0.8));
    }
    let gt = COCO::from_dataset(Dataset {
        images: (1..=10).map(image).collect(),
        annotations: gt_anns,
        categories: vec![cat(1), cat(2)],
        ..Default::default()
    });
    let dt = gt.load_res_anns(dt_anns).unwrap();
    (gt, dt)
}

fn fresh() -> COCOeval {
    let (gt, dt) = two_category_pair();
    COCOeval::new(gt, dt, IouType::Bbox)
}

fn summarized(cat_ids: &[u64]) -> COCOeval {
    let mut ev = fresh();
    ev.params.cat_ids = cat_ids.to_vec();
    ev.evaluate().unwrap();
    ev.accumulate();
    ev.summarize_lines();
    ev
}

fn ap(ev: &COCOeval) -> f64 {
    ev.results(false).unwrap().metrics["AP"]
}

fn invalid_state(result: hotcoco::error::Result<impl std::fmt::Debug>) -> String {
    match result {
        Err(Error::InvalidState(msg)) => msg,
        other => panic!("expected InvalidState, got {other:?}"),
    }
}

/// `evaluate()` again under new `params`, and nothing after it: `results()`
/// used to return the earlier run's AP under the new configuration.
#[test]
fn a_new_evaluate_retires_the_previous_summary() {
    let mut ev = summarized(&[]);
    let both = ap(&ev);
    let cat_1_alone = ap(&summarized(&[1]));
    assert!(
        (both - cat_1_alone).abs() > 0.1,
        "the fixture must tell them apart"
    );

    ev.params.cat_ids = vec![1];
    ev.evaluate().unwrap();

    let msg = invalid_state(ev.results(false));
    assert!(
        msg.contains("evaluate() has run since the last accumulate()")
            && msg.contains("call accumulate() and summarize() before results()"),
        "{msg}"
    );
    invalid_state(ev.report());
    assert!(ev.get_results(None, false).is_empty());
    assert!(ev.get_results(None, true).is_empty());
    assert!(ev.f_scores(1.0).is_empty());
    assert_eq!(ev.print_results_lines(), Vec::<String>::new());

    // The drop-in attributes keep pycocotools' behavior: `E.stats` still
    // holds the earlier run's array after a new `evaluate()`. Deliberate, and
    // open to revisiting — `tests/test_run_state.py` pins pycocotools doing
    // the same.
    assert_eq!(ev.stats().unwrap()[0], both);
    assert!(ev.accumulated().is_some());

    ev.accumulate();
    assert!(!ev.f_scores(1.0).is_empty(), "f_scores reads accumulate()");
    let msg = invalid_state(ev.report());
    assert!(
        msg.contains("accumulate() has run since the last summarize()")
            && msg.contains("call summarize() before report()"),
        "{msg}"
    );
    assert!(ev.get_results(None, false).is_empty());

    ev.summarize_lines();
    assert_eq!(ap(&ev), cat_1_alone);
}

/// `evaluate()` and `accumulate()` again, no `summarize()`: `report()` used to
/// put the earlier run's headline AP beside the new run's per-class AP.
#[test]
fn a_report_never_mixes_two_runs() {
    let mut ev = summarized(&[]);
    ev.params.cat_ids = vec![1];
    ev.evaluate().unwrap();
    ev.accumulate();
    invalid_state(ev.report());
    invalid_state(ev.results(true));

    ev.summarize_lines();
    let report = ev.report().unwrap();
    assert_eq!(report.metric("AP"), Some(report.per_class["cat_1"]["AP"]));
    assert!(!report.per_class.contains_key("cat_2"));
}

/// `summarize()` after a new `evaluate()` summarizes the retired accumulation;
/// the message names the step that is actually missing, `accumulate()`.
#[test]
fn a_retired_run_names_the_step_that_is_missing() {
    let mut ev = summarized(&[]);
    ev.evaluate().unwrap();
    ev.summarize_lines();
    let msg = invalid_state(ev.results(false));
    assert_eq!(
        msg,
        "evaluate() has run since the last accumulate(), \
         so call accumulate() and summarize() before results()"
    );
}

/// `accumulate()` and `summarize()` with no `evaluate()` before them summarize
/// nothing: still no run, though `stats` holds a row of `-1.0`.
#[test]
fn accumulating_before_evaluate_is_not_a_run() {
    let mut ev = fresh();
    ev.accumulate();
    ev.summarize_lines();
    assert!(ev.stats().is_some());
    let msg = invalid_state(ev.report());
    assert_eq!(
        msg,
        "call evaluate(), accumulate(), and summarize() before report()"
    );
    assert!(ev.f_scores(1.0).is_empty());
}

/// `params` are compared as stored, bit for bit: a NaN in them is the same
/// NaN after `accumulate()`, not a change that retires every run.
#[test]
fn params_holding_nan_still_match_their_accumulation() {
    let mut ev = fresh();
    ev.params.kpt_oks_sigmas = vec![f64::NAN; 17];
    ev.evaluate().unwrap();
    ev.accumulate();
    ev.summarize_lines();
    ev.report().unwrap();
    assert!(!ev.f_scores(1.0).is_empty());
}

/// `summarize()` warns about the configuration it reports, the one
/// `accumulate()` recorded, not `params` edited after it.
#[test]
fn summary_warnings_describe_the_summarized_params() {
    let mut ev = fresh();
    ev.evaluate().unwrap();
    ev.accumulate();
    ev.params.max_dets = vec![1, 10, 300];
    assert!(
        !ev.reference_deviations().is_empty(),
        "the edited params deviate"
    );
    assert_eq!(ev.summary_deviations(), Vec::<String>::new());
    assert!(
        ev.summarize_lines()
            .iter()
            .any(|l| l.contains("maxDets=100"))
    );
}

/// `check_accumulated()` and `check_summarized()` are why `f_scores()` and
/// `get_results()` return nothing, for a caller — the Python binding — to
/// pass on: the same reasons `report()` gives, without a method to name.
#[test]
fn the_checks_name_what_the_readers_wait_for() {
    let reason = |r: hotcoco::error::Result<()>| r.unwrap_err().to_string();
    let ev = fresh();
    assert_eq!(
        reason(ev.check_summarized()),
        "call evaluate(), accumulate(), and summarize() first"
    );
    assert_eq!(
        reason(ev.check_accumulated()),
        "call evaluate() and accumulate() first"
    );

    let mut ev = summarized(&[]);
    ev.check_summarized().unwrap();
    ev.evaluate().unwrap();
    assert_eq!(
        reason(ev.check_summarized()),
        "evaluate() has run since the last accumulate(), so call accumulate() and summarize() first"
    );
    ev.accumulate();
    ev.check_accumulated().unwrap();
    ev.params.max_dets = vec![1, 10, 50];
    assert_eq!(
        reason(ev.check_accumulated()),
        "params.max_dets changed after accumulate(), so call evaluate() and accumulate() again"
    );
}

/// Nothing run yet: every step is named.
#[test]
fn a_fresh_evaluator_names_every_step() {
    let ev = fresh();
    let msg = invalid_state(ev.report());
    assert_eq!(
        msg,
        "call evaluate(), accumulate(), and summarize() before report()"
    );
}

/// `params` edited after `accumulate()`: the axes it laid out no longer match
/// `params`, so an index read from `params` lands in the wrong slot. Reversed
/// area ranges used to send F1 to the `large` slot, which has no ground
/// truth: `-1.0` for every F-score.
#[test]
fn params_edited_after_accumulate_are_never_indexed_against_it() {
    let mut ev = summarized(&[]);
    let stats = ev.stats().unwrap().to_vec();
    let f_scores = ev.f_scores(1.0);
    assert!(f_scores["F1"] > 0.0);

    ev.params.area_ranges.reverse();

    // hotcoco's own readers refuse: their answer would describe one
    // configuration and label it with another.
    assert!(ev.f_scores(1.0).is_empty());
    assert!(ev.get_results(None, false).is_empty());
    let msg = invalid_state(ev.report());
    assert!(
        msg.contains("params.area_ranges changed after accumulate()")
            && msg.contains("call evaluate(), accumulate(), and summarize() again"),
        "{msg}"
    );

    // `summarize()` reads the axes the accumulation recorded, so it gives
    // that accumulation's numbers, not a reshuffle of them.
    ev.summarize_lines();
    assert_eq!(ev.stats().unwrap(), stats.as_slice());
    let iou_thrs = ev.accumulated().unwrap().params.iou_thrs.clone();
    assert_eq!(iou_thrs, Params::new(IouType::Bbox).iou_thrs);

    // A fresh run under the edited params answers again.
    ev.evaluate().unwrap();
    ev.accumulate();
    assert_eq!(ev.f_scores(1.0), f_scores);
    ev.summarize_lines();
    assert_eq!(ev.stats().unwrap(), stats.as_slice());
    ev.report().unwrap();
}

/// IoU thresholds reordered after `accumulate()`: `summarize()` finds AP50 in
/// the accumulation's own grid, not at `params`' new position for 0.5.
#[test]
fn summarize_reads_iou_rows_from_the_accumulation() {
    let mut ev = summarized(&[]);
    let stats = ev.stats().unwrap().to_vec();
    ev.params.iou_thrs.reverse();
    ev.summarize_lines();
    assert_eq!(ev.stats().unwrap(), stats.as_slice());
}

/// A finalized `StreamingEval` is an evaluated run: accumulate and summarize
/// it, and every reader answers.
#[test]
fn a_finalized_stream_is_an_evaluated_run() {
    let (gt, dt) = two_category_pair();
    let mut se = StreamingEval::new(
        Params::new(IouType::Bbox),
        hotcoco::EvalMode::Coco,
        gt.dataset.categories.clone(),
    )
    .unwrap();
    se.update(
        gt.dataset.images.clone(),
        gt.dataset.annotations.clone(),
        dt.dataset.annotations.clone(),
    )
    .unwrap();
    let mut ev = se.finalize();
    let msg = invalid_state(ev.report());
    assert!(
        msg.starts_with("call accumulate() and summarize()"),
        "{msg}"
    );
    ev.accumulate();
    ev.summarize_lines();
    assert_eq!(ap(&ev), ap(&summarized(&[])));
}

// ---------------------------------------------------------------------------
// Open Images: each evaluate() expands the datasets as constructed
// ---------------------------------------------------------------------------

/// Category 2 is a child of category 1; ground truth and detections are all
/// category 2, so expanding the detections decides whether category 1 is
/// detected at all.
fn oid_pair() -> (COCO, COCO, Hierarchy) {
    let b = [10.0, 10.0, 30.0, 30.0];
    let gt = COCO::from_dataset(Dataset {
        images: (1..=4).map(image).collect(),
        annotations: (1..=4).map(|i| gt_box(i, i, 2, b)).collect(),
        categories: vec![cat(1), cat(2)],
        ..Default::default()
    });
    let dt = gt
        .load_res_anns((1..=4).map(|i| det_box(i, 2, b, 0.9)).collect())
        .unwrap();
    let hierarchy = Hierarchy::from_parent_map(HashMap::from([(2, 1)]));
    (gt, dt, hierarchy)
}

fn oid_ap(ev: &mut COCOeval) -> f64 {
    ev.evaluate().unwrap();
    ev.accumulate();
    ev.summarize_lines();
    ev.stats().unwrap()[0]
}

/// `expand_dt` turned off on an evaluator that already ran with it on used to
/// keep the expanded detections: the first run's setting stuck.
#[test]
fn open_images_expand_dt_applies_to_the_run_it_is_set_for() {
    let (gt, dt, hierarchy) = oid_pair();
    let fresh = |expand_dt: bool| {
        let mut ev = COCOeval::new_oid(gt.clone(), dt.clone(), Some(hierarchy.clone()));
        ev.params.expand_dt = expand_dt;
        oid_ap(&mut ev)
    };
    let (without, with) = (fresh(false), fresh(true));
    assert!(
        (with - without).abs() > 0.1,
        "the fixture must tell them apart"
    );

    let mut ev = COCOeval::new_oid(gt.clone(), dt.clone(), Some(hierarchy.clone()));
    ev.params.expand_dt = true;
    assert_eq!(oid_ap(&mut ev), with);
    ev.params.expand_dt = false;
    assert_eq!(oid_ap(&mut ev), without);
    assert_eq!(ev.coco_dt().dataset.annotations.len(), 4);
    ev.params.expand_dt = true;
    assert_eq!(oid_ap(&mut ev), with);
    assert_eq!(ev.coco_dt().dataset.annotations.len(), 8);
    assert_eq!(
        ev.coco_gt().dataset.annotations.len(),
        8,
        "the ground truth expands once per run, not once per call"
    );
}

// ---------------------------------------------------------------------------
// The analyses of evaluate()'s cells describe the run evaluate() made
// ---------------------------------------------------------------------------

/// What TIDE, calibration, and the per-image diagnostics say about a run.
fn analyses(ev: &COCOeval) -> (f64, u64, u64, usize, usize) {
    let tide = ev.tide_errors(0.75, 0.1).unwrap();
    let calibration = ev.calibration(10, 0.75).unwrap();
    let diagnostics = ev.image_diagnostics(0.75, 0.5).unwrap();
    (
        tide.ap_base,
        tide.counts["Loc"],
        tide.counts["Miss"],
        calibration.num_detections,
        diagnostics.images.len(),
    )
}

/// `params` edited after `evaluate()`: TIDE used to score the new `cat_ids`
/// against misses counted over the old ones, and a new `max_dets` emptied all
/// three analyses, calibration reporting a perfect `ece` of 0.0 over no
/// detections. They read `evaluate()`'s cells, so they answer for its params,
/// as `eval_imgs()` does.
#[test]
fn analyses_answer_for_the_params_evaluate_saw() {
    let mut ev = fresh();
    ev.evaluate().unwrap();
    let as_evaluated = analyses(&ev);
    assert_eq!(as_evaluated, (0.5, 10, 0, 20, 10), "the fixture");

    ev.params.cat_ids = vec![1];
    assert_eq!(analyses(&ev), as_evaluated, "cat_ids edited");
    ev.params.img_ids = vec![1, 2];
    assert_eq!(analyses(&ev), as_evaluated, "img_ids edited");
    ev.params.max_dets = vec![1, 2, 3];
    assert_eq!(analyses(&ev), as_evaluated, "max_dets edited");
    ev.params.area_ranges.reverse();
    assert_eq!(analyses(&ev), as_evaluated, "area_ranges edited");

    ev.evaluate().unwrap();
    assert_ne!(analyses(&ev), as_evaluated, "a new evaluate() is a new run");
}

/// With `use_cats` off there is one pooled cell per image and no category to
/// tell `Cls` from `Loc` by: TIDE used to return an `ap_base` of 0.0 and zero
/// for every delta, and calibration listed the pooled cell as a category.
#[test]
fn pooled_runs_have_no_per_category_analysis() {
    let mut ev = fresh();
    ev.params.use_cats = false;
    ev.evaluate().unwrap();
    match ev.tide_errors(0.5, 0.1) {
        Err(Error::InvalidInput(msg)) => assert!(msg.contains("use_cats"), "{msg}"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    let calibration = ev.calibration(10, 0.5).unwrap();
    assert_eq!(calibration.num_detections, 20);
    assert!(calibration.per_category.is_empty());
}

/// `img_ids` narrowed between `evaluate()` and `accumulate()`: `accumulate()`
/// used to ignore the edit and report every evaluated image under `params`
/// that named two. It resolves the image axis by value, as it does the
/// category, area, and threshold axes.
#[test]
fn accumulate_reads_img_ids_edited_after_evaluate() {
    // Image 1 is detected exactly, image 2 not at all.
    let b = [10.0, 10.0, 30.0, 30.0];
    let gt = COCO::from_dataset(Dataset {
        images: (1..=2).map(image).collect(),
        annotations: vec![gt_box(1, 1, 1, b), gt_box(2, 2, 1, b)],
        categories: vec![cat(1)],
        ..Default::default()
    });
    let dt = gt.load_res_anns(vec![det_box(1, 1, b, 0.9)]).unwrap();
    let ap_over = |evaluated: &[u64], accumulated: &[u64]| {
        let mut ev = COCOeval::new(gt.clone(), dt.clone(), IouType::Bbox);
        ev.params.img_ids = evaluated.to_vec();
        ev.evaluate().unwrap();
        ev.params.img_ids = accumulated.to_vec();
        ev.accumulate();
        ev.summarize_lines();
        ap(&ev)
    };
    let (both, first) = (ap_over(&[1, 2], &[1, 2]), ap_over(&[1], &[1]));
    assert!(
        (both - first).abs() > 0.1,
        "the fixture must tell them apart"
    );

    assert_eq!(ap_over(&[1, 2], &[1]), first);
    assert_eq!(ap_over(&[1, 2], &[2, 1]), both, "order is not a change");
    // An image `evaluate()` never visited has no cells to add.
    assert_eq!(ap_over(&[1], &[1, 2]), first);
}

/// The evaluator `finalize()` returns holds match results and no datasets.
/// `evaluate()` on it used to match the empty datasets and replace the
/// streamed run with a row of `-1.0`, and the analyses that rebuild per-image
/// records answered over no cells — a calibration error of 0.0.
#[test]
fn a_finalized_stream_cannot_be_evaluated_again() {
    let (gt, dt) = two_category_pair();
    let mut se = StreamingEval::new(
        Params::new(IouType::Bbox),
        hotcoco::EvalMode::Coco,
        gt.dataset.categories.clone(),
    )
    .unwrap();
    se.update(
        gt.dataset.images.clone(),
        gt.dataset.annotations.clone(),
        dt.dataset.annotations.clone(),
    )
    .unwrap();
    let mut ev = se.finalize();
    ev.accumulate();
    ev.summarize_lines();
    let streamed = ap(&ev);

    for msg in [
        invalid_state(ev.evaluate()),
        invalid_state(ev.run()),
        invalid_state(ev.tide_errors(0.5, 0.1)),
        invalid_state(ev.calibration(10, 0.5)),
        invalid_state(ev.image_diagnostics(0.5, 0.5)),
        invalid_state(ev.confusion_matrix(0.5, None, None)),
    ] {
        assert!(msg.contains("StreamingEval"), "{msg}");
    }
    // No IoUs are kept: empty, as before `evaluate()`, not a key per cell
    // with nothing in it, which reads as "no overlaps".
    assert!(ev.iou_matrices().is_empty());
    // `f_scores` reads the accumulation, which the stream has.
    assert!(!ev.f_scores(1.0).is_empty());
    assert_eq!(ap(&ev), streamed, "the streamed run is still the run");
}

/// `eval_imgs_grid` is pycocotools' `evalImgs` layout: category by area range
/// by image over the ids `evaluate()` resolved, with a hole wherever an image
/// has nothing of the category. `eval_imgs` holds the same records, no holes.
#[test]
fn the_eval_imgs_grid_is_categories_by_areas_by_images() {
    let mut ev = fresh();
    assert!(ev.eval_imgs_grid().is_empty(), "nothing before evaluate()");
    // A category no image has: a whole K slot of holes.
    ev.params.cat_ids = vec![3, 1, 2];
    ev.evaluate().unwrap();

    let img_ids: Vec<u64> = (1..=10).collect();
    let cat_ids = [1, 2, 3];
    let (k, a, i) = (cat_ids.len(), ev.params.area_ranges.len(), img_ids.len());

    let grid = ev.eval_imgs_grid();
    assert_eq!(grid.len(), k * a * i);
    for (slot, entry) in grid.iter().enumerate() {
        let Some(record) = entry else { continue };
        let (ki, ai, ii) = (slot / (a * i), slot / i % a, slot % i);
        assert_eq!(record.category_id, cat_ids[ki]);
        assert_eq!(record.image_id, img_ids[ii]);
        assert_eq!(record.area_rng, ev.params.area_ranges[ai].range);
    }
    let filled = grid.iter().flatten().count();
    assert_eq!(filled, ev.eval_imgs().iter().flatten().count());
    assert!(
        filled > 0 && filled < grid.len(),
        "the fixture needs both a record and a hole"
    );

    // Laid out over the run's ids, whatever `params` says afterwards.
    ev.params.img_ids.clear();
    ev.params.cat_ids.reverse();
    assert_eq!(ev.eval_imgs_grid().len(), k * a * i);
}

/// `accumulate()` leaves a slot `evaluate()` never matched at `-1`, which is
/// quiet; `unevaluated_params` is what names it.
#[test]
fn unevaluated_params_names_each_axis_evaluate_did_not_run() {
    let mut ev = fresh();
    assert_eq!(
        ev.unevaluated_params(),
        Vec::<String>::new(),
        "nothing before evaluate()"
    );
    ev.params.iou_thrs = vec![0.5, 0.75];
    ev.params.max_dets = vec![1, 10];
    ev.evaluate().unwrap();
    assert_eq!(ev.unevaluated_params(), Vec::<String>::new());

    // Narrowing or reordering asks for nothing new.
    ev.params.iou_thrs = vec![0.75];
    ev.params.max_dets = vec![1];
    ev.params.area_ranges.reverse();
    let evaluated_imgs = ev.params.img_ids.clone();
    ev.params.img_ids.truncate(1);
    assert_eq!(ev.unevaluated_params(), Vec::<String>::new());

    ev.params.iou_thrs = vec![0.75, 0.6];
    assert_eq!(ev.unevaluated_params(), ["iou_thrs 0.6"]);
    ev.params.iou_thrs = vec![0.75];

    ev.params.max_dets = vec![1, 100];
    assert_eq!(
        ev.unevaluated_params(),
        ["max_dets 100 (evaluate() kept 10 per image)"]
    );
    ev.params.max_dets = vec![1];

    ev.params.area_ranges[0].range = [0.0, 7.0];
    assert_eq!(ev.unevaluated_params(), ["area_ranges [0, 7]"]);
    ev.params.area_ranges[0].range = [7.0, 9.0];
    ev.params.cat_ids.push(99);
    ev.params.img_ids = evaluated_imgs;
    ev.params.img_ids.push(77);
    assert_eq!(
        ev.unevaluated_params(),
        ["area_ranges [7, 9]", "cat_ids 99", "img_ids 77"]
    );

    ev.params.use_cats = false;
    assert!(
        ev.unevaluated_params()
            .contains(&"use_cats false".to_string())
    );
}

#[test]
fn replay_describes_only_the_current_run() {
    let mut ev = fresh();
    let new = ev.replay().expect("a dataset-backed evaluator replays");
    assert!(new.evaluated.is_none() && new.accumulated.is_none() && !new.summarized);

    ev.params.max_dets = vec![1, 10, 100];
    ev.evaluate().unwrap();
    ev.params.img_ids.truncate(1);
    ev.accumulate();
    ev.summarize_lines();
    let done = ev.replay().unwrap();
    assert!(done.summarized);
    assert!(done.evaluated.unwrap().img_ids.len() > 1);
    assert_eq!(done.accumulated.unwrap().img_ids.len(), 1);

    // A new evaluate() retires the accumulation and summary still on hand.
    ev.evaluate().unwrap();
    let rerun = ev.replay().unwrap();
    assert!(rerun.evaluated.is_some() && rerun.accumulated.is_none() && !rerun.summarized);

    // Repeating the steps on the replayed datasets gives the same summary.
    let original = summarized(&[]);
    let r = original.replay().unwrap();
    let mut twin = COCOeval::new(r.coco_gt, r.coco_dt, IouType::Bbox);
    twin.params = r.evaluated.unwrap();
    twin.evaluate().unwrap();
    twin.params = r.accumulated.unwrap();
    twin.accumulate();
    twin.summarize_lines();
    assert!(original.stats().is_some());
    assert_eq!(twin.stats(), original.stats());
}
