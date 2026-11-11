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
    assert_eq!(report.metrics["AP"], report.per_class["cat_1"]["AP"]);
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
