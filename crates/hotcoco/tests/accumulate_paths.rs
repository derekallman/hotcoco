//! `accumulate()` picks one of three sweeps per category, and all three must
//! write the same numbers.
//!
//! - **Fused:** every area range and IoU threshold in one pass. Taken when
//!   thresholds × active area ranges fit 64 lanes — every default run.
//! - **Per area range:** one pass a range. Taken past 64 lanes.
//! - **Per threshold:** the cumulative arrays Open Images' all-points AP
//!   needs. Taken in Open Images mode only.
//!
//! No real dataset reaches the second (val2017 parity runs 40 lanes), so
//! nothing but this file holds it to the first. A threshold's curve does not
//! depend on which other thresholds are evaluated beside it, so the same
//! ten thresholds are read out of a ten-threshold run (fused) and out of a
//! seventeen-threshold run (68 lanes, per range) and compared bit for bit.
//! Open Images mode on ground truth with no crowd or group-of boxes matches
//! as COCO does, which holds the third path to the first the same way.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image};
use hotcoco::{COCO, COCOeval};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Ground truth in every area range and detections on it: jittered copies
/// with tied scores, plus noise. `crowds` adds `iscrowd` regions.
fn datasets(rng: &mut StdRng, crowds: bool) -> (Arc<COCO>, Arc<COCO>) {
    let images: Vec<Image> = (1..=30)
        .map(|id| Image {
            id,
            width: 640,
            height: 480,
            ..Default::default()
        })
        .collect();
    let categories: Vec<Category> = (1..=6)
        .map(|id| Category {
            id,
            name: format!("c{id}"),
            ..Default::default()
        })
        .collect();
    let (mut gts, mut dts) = (Vec::new(), Vec::new());
    for img in 1..=29u64 {
        for _ in 0..rng.random_range(0..10) {
            let (w, h) = (rng.random_range(4.0..300.0), rng.random_range(4.0..300.0));
            let (x, y) = (rng.random_range(0.0..300.0), rng.random_range(0.0..180.0));
            let cat = rng.random_range(1..=5);
            gts.push(Annotation {
                id: gts.len() as u64 + 1,
                image_id: img,
                category_id: cat,
                bbox: Some([x, y, w, h]),
                area: Some(w * h),
                iscrowd: crowds && rng.random_bool(0.08),
                ..Default::default()
            });
            for _ in 0..rng.random_range(0..4) {
                let j: [f64; 4] = std::array::from_fn(|_| rng.random_range(0.8..1.2));
                dts.push(Annotation {
                    image_id: img,
                    category_id: if rng.random_bool(0.9) { cat } else { 6 },
                    bbox: Some([x * j[0], y * j[1], w * j[2], h * j[3]]),
                    score: Some(f64::from(rng.random_range(0..12u8)) / 11.0),
                    ..Default::default()
                });
            }
        }
    }
    let gt = Arc::new(COCO::from_dataset(Dataset {
        images,
        annotations: gts,
        categories,
        ..Default::default()
    }));
    let dt = Arc::new(gt.load_res_anns(dts).unwrap());
    (gt, dt)
}

/// The default ten thresholds, and seventeen that hold them: with four area
/// ranges, 40 lanes and 68.
fn thresholds() -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let ten: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * f64::from(i)).collect();
    let mut seventeen = ten.clone();
    seventeen.extend((0..7).map(|i| 0.52 + 0.07 * f64::from(i)));
    seventeen.sort_by(f64::total_cmp);
    let at = ten
        .iter()
        .map(|t| seventeen.iter().position(|s| s == t).unwrap())
        .collect();
    (ten, seventeen, at)
}

fn run(mut ev: COCOeval, iou_thrs: &[f64]) -> COCOeval {
    ev.params.iou_thrs = iou_thrs.to_vec();
    ev.evaluate().unwrap();
    ev.accumulate();
    ev
}

/// Require threshold `ta[i]` of `a` and `tb[i]` of `b` equal, bit for bit, in
/// precision, scores, and recall.
fn assert_same_thresholds(label: &str, a: &COCOeval, ta: &[usize], b: &COCOeval, tb: &[usize]) {
    let (a, b) = (a.accumulated().unwrap(), b.accumulated().unwrap());
    let (sa, sb) = (&a.shape, &b.shape);
    assert_eq!(
        (sa.r, sa.k, sa.a, sa.m),
        (sb.r, sb.k, sb.a, sb.m),
        "{label}"
    );
    let mut some_precision = false;
    for (&t_a, &t_b) in ta.iter().zip(tb) {
        for k in 0..sa.k {
            for ar in 0..sa.a {
                for m in 0..sa.m {
                    let (x, y) = (
                        a.recall[sa.recall_idx(t_a, k, ar, m)],
                        b.recall[sb.recall_idx(t_b, k, ar, m)],
                    );
                    assert_eq!(x.to_bits(), y.to_bits(), "{label}: recall {x} vs {y}");
                    for r in 0..sa.r {
                        let (i, j) = (
                            sa.precision_idx(t_a, r, k, ar, m),
                            sb.precision_idx(t_b, r, k, ar, m),
                        );
                        for (name, x, y) in [
                            ("precision", a.precision[i], b.precision[j]),
                            ("scores", a.scores[i], b.scores[j]),
                        ] {
                            assert_eq!(
                                x.to_bits(),
                                y.to_bits(),
                                "{label}: {name} at t={t_a}/{t_b} r={r} k={k} a={ar} m={m}: {x} vs {y}"
                            );
                        }
                        some_precision |= a.precision[i] > 0.0;
                    }
                }
            }
        }
    }
    assert!(
        some_precision,
        "{label}: nothing was matched, so nothing was compared"
    );
}

#[test]
fn the_per_range_sweep_writes_what_the_fused_sweep_writes() {
    let mut rng = StdRng::seed_from_u64(0xACC);
    let (ten, seventeen, at) = thresholds();
    let all: Vec<usize> = (0..10).collect();
    for trial in 0..4 {
        let (gt, dt) = datasets(&mut rng, true);
        let new = || COCOeval::new(Arc::clone(&gt), Arc::clone(&dt), IouType::Bbox);
        let fused = run(new(), &ten);
        let per_range = run(new(), &seventeen);
        assert_same_thresholds(&format!("trial {trial}"), &fused, &all, &per_range, &at);
    }
}

#[test]
fn the_per_threshold_sweep_writes_what_the_fused_sweep_writes() {
    let mut rng = StdRng::seed_from_u64(0x01D);
    let (ten, _, _) = thresholds();
    let all: Vec<usize> = (0..10).collect();
    for trial in 0..4 {
        // No crowd and no group-of box: Open Images matches as COCO does.
        let (gt, dt) = datasets(&mut rng, false);
        let fused = run(
            COCOeval::new(Arc::clone(&gt), Arc::clone(&dt), IouType::Bbox),
            &ten,
        );
        let mut oid = COCOeval::new_oid(Arc::clone(&gt), Arc::clone(&dt), None);
        // Open Images' own defaults differ; the comparison is of the sweep.
        oid.params.area_ranges = fused.params.area_ranges.clone();
        oid.params.max_dets = fused.params.max_dets.clone();
        oid.params.rec_thrs = fused.params.rec_thrs.clone();
        let per_threshold = run(oid, &ten);
        assert_same_thresholds(
            &format!("trial {trial}"),
            &fused,
            &all,
            &per_threshold,
            &all,
        );
    }
}
