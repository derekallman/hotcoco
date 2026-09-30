//! Regression tests for the 1.0 preflight detection fixes.
//!
//! Each test pins a behavioral fix: IoU row/column alignment when an
//! annotation has no usable geometry, sentinel degradation for missing
//! area-label / max-det lookups in `summarize`, `compare()` parameter
//! validation, graceful handling of empty `max_dets`, F-score key naming, and
//! the self-explaining `EvalParams` archive.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use hotcoco::detection::{CompareOpts, compare};
use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image};
use hotcoco::{AreaRange, COCO, COCOeval};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn image(id: u64) -> Image {
    Image {
        id,
        width: 200,
        height: 200,
        ..Default::default()
    }
}

fn cat(id: u64, name: &str) -> Category {
    Category {
        id,
        name: name.to_string(),
        ..Default::default()
    }
}

/// Ground-truth annotation with geometry; `bbox: None` models a real dataset's
/// geometry-less record.
fn ann(id: u64, bbox: Option<[f64; 4]>) -> Annotation {
    Annotation {
        id,
        image_id: 1,
        category_id: 1,
        bbox,
        area: bbox.map(|b| b[2] * b[3]),
        ..Default::default()
    }
}

/// Detection — an [`ann`] carrying a score.
fn det(id: u64, bbox: Option<[f64; 4]>, score: f64) -> Annotation {
    Annotation {
        score: Some(score),
        ..ann(id, bbox)
    }
}

fn coco_from(anns: Vec<Annotation>) -> COCO {
    COCO::from_dataset(Dataset {
        images: vec![image(1)],
        annotations: anns,
        categories: vec![cat(1, "thing")],
        ..Default::default()
    })
}

fn fixture_eval() -> COCOeval {
    let gt = COCO::new(&fixtures_dir().join("gt.json")).unwrap();
    let dt = gt.load_res(&fixtures_dir().join("dt.json")).unwrap();
    COCOeval::new(gt, dt, IouType::Bbox)
}

// ---------------------------------------------------------------------------
// A.3 — IoU row/column alignment with geometry-less annotations
// ---------------------------------------------------------------------------

/// A bbox-less ground truth between two valid ones must occupy a zero *column*
/// in the pair's IoU matrix, not vanish from it. Before the fix, the matrix
/// builders dropped it while `gather_pair` still counted it, so every later
/// ground truth read its neighbor's IoU column and the detection sitting
/// exactly on GT 3 went unmatched.
#[test]
fn bboxless_gt_between_valid_gts_does_not_shift_iou_columns() {
    let gt = coco_from(vec![
        ann(1, Some([0.0, 0.0, 10.0, 10.0])),
        ann(2, None), // no geometry — the column that used to vanish
        ann(3, Some([50.0, 50.0, 10.0, 10.0])),
    ]);
    let dt = coco_from(vec![
        det(101, Some([0.0, 0.0, 10.0, 10.0]), 0.9), // IoU 1.0 with GT 1
        det(102, Some([50.0, 50.0, 10.0, 10.0]), 0.8), // IoU 1.0 with GT 3
    ]);

    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    ev.evaluate();

    let diag = ev.image_diagnostics(0.5, 0.5).unwrap();
    assert_eq!(
        diag.annotations.dt_match.get(&101),
        Some(&1),
        "detection 101 sits exactly on GT 1"
    );
    assert_eq!(
        diag.annotations.dt_match.get(&102),
        Some(&3),
        "detection 102 sits exactly on GT 3 and must not read GT 2's missing column"
    );
    // The geometry-less GT scores as an unmatched ground truth, not as a shift.
    assert_eq!(
        diag.annotations.gt_status.len(),
        3,
        "all three ground truths are classified"
    );
}

/// The detection-side twin: a bbox-less detection between two valid ones must
/// occupy a zero *row*, so the lower-scoring valid detection still reads its
/// own IoU row rather than falling off the end of a shortened matrix.
#[test]
fn bboxless_dt_between_valid_dts_does_not_shift_iou_rows() {
    let gt = coco_from(vec![
        ann(1, Some([0.0, 0.0, 10.0, 10.0])),
        ann(3, Some([50.0, 50.0, 10.0, 10.0])),
    ]);
    let dt = coco_from(vec![
        det(101, Some([0.0, 0.0, 10.0, 10.0]), 0.9), // IoU 1.0 with GT 1
        det(102, None, 0.85),                        // no geometry — zero row
        det(103, Some([50.0, 50.0, 10.0, 10.0]), 0.8), // IoU 1.0 with GT 3
    ]);

    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    ev.evaluate();

    let diag = ev.image_diagnostics(0.5, 0.5).unwrap();
    assert_eq!(diag.annotations.dt_match.get(&101), Some(&1));
    assert_eq!(
        diag.annotations.dt_match.get(&103),
        Some(&3),
        "detection 103's row must not shift onto detection 102's missing slot"
    );
    use hotcoco::detection::DtStatus;
    assert_eq!(
        diag.annotations.dt_status.get(&102),
        Some(&DtStatus::Fp),
        "a geometry-less detection is an unmatched detection, not a shift"
    );
}

/// Same alignment contract on the direct `eval_imgs` surface: with the
/// geometry-less GT present, both detections match at every IoU threshold and
/// the matched ids are the geometrically correct ones.
#[test]
fn eval_imgs_matches_are_aligned_with_geometry_gaps() {
    let gt = coco_from(vec![
        ann(1, Some([0.0, 0.0, 10.0, 10.0])),
        ann(2, None),
        ann(3, Some([50.0, 50.0, 10.0, 10.0])),
    ]);
    let dt = coco_from(vec![
        det(101, Some([0.0, 0.0, 10.0, 10.0]), 0.9),
        det(102, Some([50.0, 50.0, 10.0, 10.0]), 0.8),
    ]);

    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    ev.evaluate();

    // area = "all" cell at the default cap.
    let cell = ev
        .eval_imgs()
        .iter()
        .flatten()
        .find(|e| e.area_rng == [0.0, 1e10])
        .expect("the all-area cell exists");

    for t_idx in 0..10 {
        assert!(cell.dt_matched[(t_idx, 0)], "dt 101 matched at t={t_idx}");
        assert!(cell.dt_matched[(t_idx, 1)], "dt 102 matched at t={t_idx}");
        assert_eq!(cell.dt_matches[(t_idx, 0)], 1);
        assert_eq!(cell.dt_matches[(t_idx, 1)], 3);
    }
}

// ---------------------------------------------------------------------------
// B — summarize: missing area label / max_det degrade to the -1.0 sentinel
// ---------------------------------------------------------------------------

/// Renaming the "small" area range must make `APs`/`ARs` report the `-1.0`
/// "not computed" sentinel — not silently reuse index 0, which is the "all"
/// slice wearing a per-size metric's name.
#[test]
fn missing_area_label_reports_sentinel_not_all_slice() {
    let mut ev = fixture_eval();
    for ar in &mut ev.params.area_ranges {
        if ar.label == "small" {
            ar.label = "tiny".to_string();
        }
    }
    ev.run();

    let results = ev.get_results(None, false);
    assert!(
        results["AP"] >= 0.0,
        "headline AP is computable on the fixture"
    );
    assert_eq!(
        results["APs"], -1.0,
        "no 'small' range exists, so APs is not computed — it must not report the 'all' slice"
    );
    assert_eq!(results["ARs"], -1.0);
    // Pre-fix, APs silently equaled AP (both read index 0).
    assert_ne!(results["APs"], results["AP"]);
}

/// Empty `max_dets` must degrade — no metric is computable, every stat is the
/// sentinel — rather than panic, matching how the missing-area-label and
/// missing-threshold branches behave.
#[test]
fn empty_max_dets_degrades_to_sentinels_without_panicking() {
    let mut ev = fixture_eval();
    ev.params.max_dets = Vec::new();
    ev.run(); // pre-fix: assert! panic inside evaluate()

    let stats = ev.stats().expect("summarize ran");
    assert!(!stats.is_empty());
    assert!(
        stats.iter().all(|&v| v == -1.0),
        "with no max-det slots nothing is computable; got {stats:?}"
    );
}

// ---------------------------------------------------------------------------
// B — compare() validates the axes its shared catalog reads
// ---------------------------------------------------------------------------

#[test]
fn compare_rejects_mismatched_grids_and_ranges() {
    let mut ev_a = fixture_eval();
    ev_a.evaluate();

    // iou_thrs
    let mut ev_b = fixture_eval();
    ev_b.params.iou_thrs = vec![0.5];
    ev_b.evaluate();
    let err = compare(&ev_a, &ev_b, &CompareOpts::default()).unwrap_err();
    assert!(err.to_string().contains("iou_thrs"), "got: {err}");

    // rec_thrs
    let mut ev_b = fixture_eval();
    ev_b.params.rec_thrs = vec![0.0, 0.5, 1.0];
    ev_b.evaluate();
    let err = compare(&ev_a, &ev_b, &CompareOpts::default()).unwrap_err();
    assert!(err.to_string().contains("rec_thrs"), "got: {err}");

    // max_dets
    let mut ev_b = fixture_eval();
    ev_b.params.max_dets = vec![50];
    ev_b.evaluate();
    let err = compare(&ev_a, &ev_b, &CompareOpts::default()).unwrap_err();
    assert!(err.to_string().contains("max_dets"), "got: {err}");

    // area_ranges (bounds differ, labels identical — the label-only check
    // missed exactly this case elsewhere)
    let mut ev_b = fixture_eval();
    ev_b.params.area_ranges[1] = AreaRange {
        label: "small".to_string(),
        range: [0.0, 100.0],
    };
    ev_b.evaluate();
    let err = compare(&ev_a, &ev_b, &CompareOpts::default()).unwrap_err();
    assert!(err.to_string().contains("area_ranges"), "got: {err}");

    // Identical params still compare fine.
    let mut ev_b = fixture_eval();
    ev_b.evaluate();
    assert!(compare(&ev_a, &ev_b, &CompareOpts::default()).is_ok());
}

// ---------------------------------------------------------------------------
// E — F-score key naming: integer betas undecorated
// ---------------------------------------------------------------------------

#[test]
fn f_score_keys_use_minimal_digits() {
    let mut ev = fixture_eval();
    ev.run();

    let f2 = ev.f_scores(2.0);
    assert!(
        f2.contains_key("F2") && f2.contains_key("F2_50") && f2.contains_key("F2_75"),
        "integer beta must print undecorated; got keys {:?}",
        f2.keys().collect::<Vec<_>>()
    );
    let fh = ev.f_scores(0.5);
    assert!(fh.contains_key("F0.5") && fh.contains_key("F0.5_50"));
}

// ---------------------------------------------------------------------------
// E — EvalParams archives the whole configuration, deviations included
// ---------------------------------------------------------------------------

/// A saved `Extension` run must carry its own explanation: the archive holds
/// the recall grid, `use_cats`, the OKS sigmas, and the deviation strings that
/// drove the provenance bit.
#[test]
fn eval_params_archive_is_self_explaining() {
    let mut ev = fixture_eval();
    ev.params.iou_thrs = vec![0.25, 0.75]; // a deliberate deviation
    ev.run();

    let results = ev.results(false).unwrap();
    assert_eq!(results.params.recall_thresholds, ev.params.rec_thrs);
    assert!(results.params.use_cats);
    assert_eq!(results.params.kpt_oks_sigmas, ev.params.kpt_oks_sigmas);
    assert!(
        !results.params.reference_deviations.is_empty(),
        "custom iou_thrs is a deviation and the archive must say so"
    );
    assert!(
        results
            .params
            .reference_deviations
            .iter()
            .any(|d| d.contains("iou_thrs")),
        "the deviation names the parameter: {:?}",
        results.params.reference_deviations
    );

    // And the archived strings are the same ones the live predicate returns.
    assert_eq!(
        results.params.reference_deviations,
        ev.reference_deviations()
    );
}

// ---------------------------------------------------------------------------
// Found by tests/fuzz_dropin.py: a keypoint GT without `num_keypoints`
// read the field as 0 and was ignored — every ground truth in a file that
// omits the field, so keypoint AP scored a dataset with nothing to match.
// ---------------------------------------------------------------------------

fn keypoint_pair(with_num_keypoints: bool) -> (COCO, COCO) {
    // 17 COCO keypoints, five labeled, the rest absent.
    let mut kps = vec![0.0; 51];
    for k in 0..5 {
        kps[k * 3] = 10.0 + k as f64;
        kps[k * 3 + 1] = 10.0 + k as f64;
        kps[k * 3 + 2] = 2.0;
    }
    let gt_ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: Some([5.0, 5.0, 20.0, 20.0]),
        area: Some(400.0),
        keypoints: Some(kps),
        num_keypoints: with_num_keypoints.then_some(5),
        ..Default::default()
    };
    let dt_ann = Annotation {
        score: Some(0.9),
        num_keypoints: None,
        ..gt_ann.clone()
    };
    (coco_from(vec![gt_ann]), coco_from(vec![dt_ann]))
}

#[test]
fn num_keypoints_is_derived_when_absent() {
    let ann = Annotation {
        keypoints: Some(vec![1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 3.0, 1.0]),
        ..Default::default()
    };
    assert_eq!(
        ann.num_visible_keypoints(),
        2,
        "derived from visibility flags"
    );
    let explicit = Annotation {
        num_keypoints: Some(7),
        ..ann.clone()
    };
    assert_eq!(
        explicit.num_visible_keypoints(),
        7,
        "the field wins when present"
    );
    assert_eq!(Annotation::default().num_visible_keypoints(), 0);

    let stats = |with_field: bool| {
        let (gt, dt) = keypoint_pair(with_field);
        let mut ev = COCOeval::new(gt, dt, IouType::Keypoints);
        ev.evaluate();
        ev.accumulate();
        ev.summarize_lines();
        ev.stats().unwrap().to_vec()
    };
    let with = stats(true);
    let without = stats(false);
    // A lone true positive's AP is the mean of `coco_precision(1, 0)` over the
    // grid, to within an ulp of the mean's rounding.
    let lone_tp = hotcoco::metrics::counts::average_precision(
        &[1.0],
        &[true],
        None,
        1,
        &hotcoco::params::default_rec_thrs(),
    );
    assert!(
        (with[0] - lone_tp).abs() <= f64::EPSILON,
        "identical keypoints must score a lone TP's AP: {} vs {lone_tp}",
        with[0]
    );
    assert_eq!(
        with, without,
        "omitting num_keypoints must not change a single metric"
    );
}
