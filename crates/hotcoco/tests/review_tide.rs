//! TIDE Loc/Cls fix parity with tidecv's `BestGTMatch` rule.
//!
//! tidecv fixes a `Loc` or `Cls` false positive into a true positive only when
//! (1) its target ground truth was *not* already matched, and (2) it is the
//! highest-scoring `Loc`/`Cls` error targeting that ground truth — the winner is
//! picked across both error types, since `best_score` lives on the GT. Every
//! other such error is suppressed (dropped from the AP data, neither TP nor FP).
//! A fixed `Cls` error is credited to the **ground truth's** category, not the
//! detection's.
//!
//! Every expected number below is tidecv 1.0.1's own output, produced by this
//! script (`uv run python`, from the repo root):
//!
//! ```python
//! from tidecv import TIDE, Data
//!
//! CASES = {  # (gts: [(cat, box)], dets: [(cat, score, box)])
//!     "reviewer_loc": (
//!         [(1, [10, 10, 40, 40]), (1, [120, 120, 40, 40])],
//!         [(1, 0.9, [10, 10, 40, 40]), (1, 0.8, [30, 30, 40, 40]),
//!          (1, 0.7, [300, 300, 40, 40]), (1, 0.6, [120, 120, 40, 40])],
//!     ),
//!     "cls_lands_in_gt_cat": (
//!         [(1, [0, 0, 50, 50]), (2, [100, 0, 50, 50])],
//!         [(2, 0.9, [0, 0, 50, 50]), (2, 0.5, [100, 0, 50, 50])],
//!     ),
//!     "two_loc_one_gt": (
//!         [(1, [0, 0, 40, 40]), (1, [200, 200, 40, 40]), (1, [400, 0, 40, 40])],
//!         [(1, 0.9, [20, 20, 40, 40]), (1, 0.8, [300, 400, 40, 40]),
//!          (1, 0.7, [20, 0, 40, 40]), (1, 0.6, [200, 200, 40, 40])],
//!     ),
//!     "cls_outranks_loc_on_one_gt": (
//!         [(1, [0, 0, 40, 40]), (1, [200, 200, 40, 40]), (2, [400, 400, 40, 40])],
//!         [(2, 0.9, [0, 0, 40, 40]), (1, 0.8, [20, 20, 40, 40]),
//!          (1, 0.6, [200, 200, 40, 40]), (2, 0.5, [400, 400, 40, 40])],
//!     ),
//! }
//!
//! for name, (gts, dets) in CASES.items():
//!     gt, pr = Data("gt"), Data("pr")
//!     for c in (1, 2):
//!         gt.add_class(c, f"c{c}"); pr.add_class(c, f"c{c}")
//!     gt.add_image(1, "i"); pr.add_image(1, "i")
//!     for c, b in gts:
//!         gt.add_ground_truth(1, c, b)
//!     for c, s, b in dets:
//!         pr.add_detection(1, c, s, b)
//!     tide = TIDE()
//!     tide.evaluate(gt, pr, mode=TIDE.BOX, pos_threshold=0.5, background_threshold=0.1)
//!     run = list(tide.runs.values())[0]
//!     me = run.fix_main_errors()
//!     print(name, "ap_base=%.6f" % (run.ap / 100),
//!           {k.short_name: round(v / 100, 6) for k, v in me.items()},
//!           {k.short_name: len(v) for k, v in run.error_dict.items()})
//! ```
//!
//! Every category in these cases holds ground truth on purpose: tidecv averages
//! a GT-less category that has detections in as AP 0, hotcoco (like COCO) leaves
//! it out of the mean, and that separate, pre-existing difference is not what
//! these tests are about.

#![allow(clippy::unwrap_used)]

use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image};
use hotcoco::{COCO, COCOeval, TideErrors};

const TOL: f64 = 1e-6;

fn ann(id: u64, cat: u64, bbox: [f64; 4], score: Option<f64>) -> Annotation {
    Annotation {
        id,
        image_id: 1,
        category_id: cat,
        bbox: Some(bbox),
        area: Some(bbox[2] * bbox[3]),
        score,
        ..Default::default()
    }
}

fn dataset(annotations: Vec<Annotation>) -> Dataset {
    Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img1.jpg".into(),
            height: 640,
            width: 640,
            ..Default::default()
        }],
        annotations,
        categories: (1..=2)
            .map(|id| Category {
                id,
                name: format!("c{id}"),
                ..Default::default()
            })
            .collect(),
        licenses: vec![],
    }
}

/// `gts`: `(cat, box)`; `dets`: `(cat, score, box)`. Detection ids are assigned
/// in list order, which is also tidecv's insertion order.
fn tide(gts: &[(u64, [f64; 4])], dets: &[(u64, f64, [f64; 4])]) -> TideErrors {
    let gt = gts
        .iter()
        .enumerate()
        .map(|(i, &(c, b))| ann(i as u64 + 1, c, b, None))
        .collect();
    let dt = dets
        .iter()
        .enumerate()
        .map(|(i, &(c, s, b))| ann(i as u64 + 101, c, b, Some(s)))
        .collect();
    let mut ev = COCOeval::new(
        COCO::from_dataset(dataset(gt)),
        COCO::from_dataset(dataset(dt)),
        IouType::Bbox,
    );
    ev.evaluate();
    ev.tide_errors(0.5, 0.1).expect("tide_errors")
}

fn assert_close(te: &TideErrors, key: &str, want: f64) {
    let got = te.delta_ap[key];
    assert!(
        (got - want).abs() < TOL,
        "dAP[{key}] = {got:.6}, tidecv = {want:.6}"
    );
}

/// The reviewer's reproducer. The `Loc` error targets a ground truth an earlier
/// detection already matched, so tidecv suppresses it rather than promoting it;
/// promoting it gave three TPs against two GTs (dAP 0.2475).
#[test]
fn loc_error_on_a_matched_gt_is_suppressed_not_promoted() {
    let te = tide(
        &[
            (1, [10.0, 10.0, 40.0, 40.0]),
            (1, [120.0, 120.0, 40.0, 40.0]),
        ],
        &[
            (1, 0.9, [10.0, 10.0, 40.0, 40.0]),
            (1, 0.8, [30.0, 30.0, 40.0, 40.0]), // IoU 1/7 with GT 1, which is taken
            (1, 0.7, [300.0, 300.0, 40.0, 40.0]),
            (1, 0.6, [120.0, 120.0, 40.0, 40.0]),
        ],
    );
    assert_eq!(te.counts["Loc"], 1);
    assert_eq!(te.counts["Bkg"], 1);
    assert!((te.ap_base - 0.752475).abs() < TOL, "{}", te.ap_base);
    assert_close(&te, "Loc", 0.082508);
    assert_close(&te, "Bkg", 0.082508);
    assert_close(&te, "Miss", 0.0);
}

/// A fixed `Cls` error becomes a TP for the ground truth's category (1), and
/// leaves the detection's own category (2) without its false positive. Crediting
/// it to category 2 instead gave dAP 0.25: category 2 went 0.5 → 1.0 while
/// category 1 stayed at 0.
#[test]
fn cls_fix_is_credited_to_the_gt_category() {
    let te = tide(
        &[(1, [0.0, 0.0, 50.0, 50.0]), (2, [100.0, 0.0, 50.0, 50.0])],
        &[
            (2, 0.9, [0.0, 0.0, 50.0, 50.0]), // IoU 1.0 with the category-1 GT
            (2, 0.5, [100.0, 0.0, 50.0, 50.0]),
        ],
    );
    assert_eq!(te.counts["Cls"], 1);
    assert!((te.ap_base - 0.25).abs() < TOL, "{}", te.ap_base);
    assert_close(&te, "Cls", 0.75);
}

/// Two `Loc` errors target the same unmatched GT; only the higher-scoring one
/// (0.9) becomes a TP and the other (0.7) is dropped. Promoting both inflated
/// the curve with a third TP for a GT that can be recovered once.
#[test]
fn only_the_best_scoring_error_per_gt_is_promoted() {
    let te = tide(
        &[
            (1, [0.0, 0.0, 40.0, 40.0]),
            (1, [200.0, 200.0, 40.0, 40.0]),
            (1, [400.0, 0.0, 40.0, 40.0]),
        ],
        &[
            (1, 0.9, [20.0, 20.0, 40.0, 40.0]), // IoU 1/7 with GT 1
            (1, 0.8, [300.0, 400.0, 40.0, 40.0]),
            (1, 0.7, [20.0, 0.0, 40.0, 40.0]), // IoU 1/3 with GT 1
            (1, 0.6, [200.0, 200.0, 40.0, 40.0]),
        ],
    );
    assert_eq!(te.counts["Loc"], 2);
    assert_eq!(te.counts["Miss"], 1);
    assert!((te.ap_base - 0.084158).abs() < TOL, "{}", te.ap_base);
    assert_close(&te, "Loc", 0.470297);
    assert_close(&te, "Bkg", 0.028053);
    // tidecv fixes a Miss by dropping the GT from the denominator
    // (`MissedError.fix()` returns `(class, -1)`). Injecting a top-scoring TP
    // instead read 0.383168 here.
    assert_close(&te, "Miss", 0.042079);
}

/// The per-GT winner is chosen across `Loc` and `Cls` together. A `Cls` error
/// (0.9) outranks a `Loc` error (0.8) on the same GT, so fixing `Loc` alone
/// suppresses the `Loc` error rather than promoting it, and fixing `Cls` adds
/// the TP to category 1 while the `Loc` error stays a false positive there.
#[test]
fn per_gt_winner_spans_loc_and_cls() {
    let te = tide(
        &[
            (1, [0.0, 0.0, 40.0, 40.0]),
            (1, [200.0, 200.0, 40.0, 40.0]),
            (2, [400.0, 400.0, 40.0, 40.0]),
        ],
        &[
            (2, 0.9, [0.0, 0.0, 40.0, 40.0]),   // Cls on GT 1
            (1, 0.8, [20.0, 20.0, 40.0, 40.0]), // Loc on GT 1
            (1, 0.6, [200.0, 200.0, 40.0, 40.0]),
            (2, 0.5, [400.0, 400.0, 40.0, 40.0]),
        ],
    );
    assert_eq!(te.counts["Cls"], 1);
    assert_eq!(te.counts["Loc"], 1);
    assert!((te.ap_base - 0.376238).abs() < TOL, "{}", te.ap_base);
    assert_close(&te, "Loc", 0.126238);
    assert_close(&te, "Cls", 0.541254);
    assert_close(&te, "Miss", 0.0);
}
