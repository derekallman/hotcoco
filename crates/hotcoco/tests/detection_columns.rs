//! `DetectionColumns` against the records it stands for.
//!
//! The Python binding's `load_res(array)` keeps box detections as
//! [`DetectionColumns`] and evaluates them with `COCOeval::new_columns`,
//! building the results `COCO` only when something else reads it. Every test
//! here loads the same rows both ways — the columns, and
//! `gt.load_res_anns(COCO::detections_from_rows(rows))` — and requires the
//! same errors, the same warnings, the same results `COCO`, and the same
//! evaluation, bit for bit, under every configuration that changes which
//! detections a cell reads or how they rank.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image};
use hotcoco::{AreaRange, COCO, COCOeval, DetectionColumns};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Ground truth with what changes matching: crowd regions, objects in every
/// area range, an image with no annotations, and two categories with none.
fn ground_truth(rng: &mut StdRng) -> Arc<COCO> {
    let images: Vec<Image> = (1..=40)
        .map(|id| Image {
            id,
            width: 640,
            height: 480,
            ..Default::default()
        })
        .collect();
    let categories: Vec<Category> = (1..=7)
        .map(|id| Category {
            id,
            name: format!("c{id}"),
            ..Default::default()
        })
        .collect();
    let mut annotations = Vec::new();
    for img in 1..=39u64 {
        for _ in 0..rng.random_range(0..9) {
            let (w, h) = (rng.random_range(4.0..300.0), rng.random_range(4.0..300.0));
            let (x, y) = (rng.random_range(0.0..300.0), rng.random_range(0.0..180.0));
            annotations.push(Annotation {
                id: annotations.len() as u64 + 1,
                image_id: img,
                category_id: rng.random_range(1..=5),
                bbox: Some([x, y, w, h]),
                area: Some(w * h),
                iscrowd: rng.random_bool(0.05),
                ..Default::default()
            });
        }
    }
    Arc::new(COCO::from_dataset(Dataset {
        images,
        annotations,
        categories,
        ..Default::default()
    }))
}

/// `n` detection rows, `ncols` wide: mostly jittered ground-truth boxes,
/// some noise, scores drawn from a few values so ties are common, and rows
/// on an image and in a category the ground truth does not have.
fn rows(gt: &COCO, rng: &mut StdRng, n: usize, ncols: usize) -> Vec<f64> {
    let anns = &gt.dataset.annotations;
    let mut out = Vec::with_capacity(n * ncols);
    for _ in 0..n {
        let (image_id, cat, bbox) = if !anns.is_empty() && rng.random_bool(0.6) {
            let a = &anns[rng.random_range(0..anns.len())];
            let b = a.bbox.unwrap();
            let jitter: [f64; 4] = std::array::from_fn(|_| rng.random_range(0.85..1.15));
            let cat = if rng.random_bool(0.9) {
                a.category_id
            } else {
                rng.random_range(1..=7)
            };
            let bbox = std::array::from_fn(|k| b[k] * jitter[k]);
            (a.image_id, cat, bbox)
        } else {
            let image_id = if rng.random_bool(0.02) {
                999
            } else {
                rng.random_range(1..=40)
            };
            let cat = if rng.random_bool(0.02) {
                77
            } else {
                rng.random_range(1..=7)
            };
            let (w, h) = (rng.random_range(2.0..200.0), rng.random_range(2.0..200.0));
            (
                image_id,
                cat,
                [
                    rng.random_range(0.0..400.0),
                    rng.random_range(0.0..300.0),
                    w,
                    h,
                ],
            )
        };
        let score = f64::from(rng.random_range(0..12u8)) / 11.0;
        out.extend_from_slice(&[image_id as f64, bbox[0], bbox[1], bbox[2], bbox[3], score]);
        if ncols == 7 {
            out.push(cat as f64);
        }
    }
    out
}

fn records(gt: &COCO, rows: &[f64], ncols: usize) -> hotcoco::error::Result<COCO> {
    gt.load_res_anns(COCO::detections_from_rows(rows, ncols)?)
}

fn columns(gt: &Arc<COCO>, rows: &[f64], ncols: usize) -> hotcoco::error::Result<DetectionColumns> {
    DetectionColumns::check_rows(rows, ncols)?;
    DetectionColumns::from_checked_rows(Arc::clone(gt), rows, ncols)
}

/// Run both evaluators under `configure` and require every output equal.
fn assert_same_evaluation(
    label: &str,
    gt: &Arc<COCO>,
    rows: &[f64],
    ncols: usize,
    configure: impl Fn(&mut COCOeval),
) {
    let mut by_records = COCOeval::new(
        Arc::clone(gt),
        records(gt, rows, ncols).unwrap(),
        IouType::Bbox,
    );
    let mut by_columns = COCOeval::new_columns(
        Arc::clone(gt),
        Arc::new(columns(gt, rows, ncols).unwrap()),
        IouType::Bbox,
    );
    for ev in [&mut by_records, &mut by_columns] {
        configure(ev);
        ev.evaluate().unwrap();
        ev.accumulate();
        ev.summarize();
    }
    let (a, b) = (
        by_records.accumulated().unwrap(),
        by_columns.accumulated().unwrap(),
    );
    for (name, x, y) in [
        ("precision", &a.precision, &b.precision),
        ("recall", &a.recall, &b.recall),
        ("scores", &a.scores, &b.scores),
    ] {
        assert_eq!(x.len(), y.len(), "{label}: {name} length");
        for (i, (p, q)) in x.iter().zip(y).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "{label}: {name}[{i}]: {p} vs {q}");
        }
    }
    let (sa, sb) = (by_records.stats().unwrap(), by_columns.stats().unwrap());
    assert_eq!(
        sa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        sb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{label}: stats"
    );
    // The per-image records are rebuilt from the same source `evaluate()`
    // read: the ids, scores, and match flags of every cell.
    assert_eq!(
        format!("{:?}", by_records.eval_imgs()),
        format!("{:?}", by_columns.eval_imgs()),
        "{label}: eval_imgs"
    );
    // And what reads the records gets the same results set.
    assert_eq!(
        format!("{:?}", by_records.coco_dt().dataset.annotations),
        format!("{:?}", by_columns.coco_dt().dataset.annotations),
        "{label}: the results COCO"
    );
}

#[test]
fn columns_evaluate_exactly_as_the_records_they_stand_for() {
    let mut rng = StdRng::seed_from_u64(0xC0_1A_5E);
    for trial in 0..6 {
        let gt = ground_truth(&mut rng);
        let ncols = if trial % 3 == 2 { 6 } else { 7 };
        let n = [0, 1, 37, 400, 1500, 3000][trial];
        let rows = rows(&gt, &mut rng, n, ncols);
        let label = |what: &str| format!("trial {trial} ({n} rows, {ncols} cols): {what}");

        assert_same_evaluation(&label("defaults"), &gt, &rows, ncols, |_| {});
        assert_same_evaluation(&label("use_cats off"), &gt, &rows, ncols, |ev| {
            ev.params.use_cats = false;
        });
        assert_same_evaluation(&label("small max_dets"), &gt, &rows, ncols, |ev| {
            ev.params.max_dets = vec![1, 3, 7];
        });
        assert_same_evaluation(
            &label("img_ids and cat_ids subsets"),
            &gt,
            &rows,
            ncols,
            |ev| {
                ev.params.img_ids = (1..=40).step_by(3).collect();
                ev.params.cat_ids = vec![2, 3, 7];
            },
        );
        assert_same_evaluation(
            &label("cat_ids unsorted, use_cats off"),
            &gt,
            &rows,
            ncols,
            |ev| {
                ev.params.use_cats = false;
                ev.params.cat_ids = vec![5, 1, 3];
            },
        );
        assert_same_evaluation(&label("thresholds and areas"), &gt, &rows, ncols, |ev| {
            ev.params.iou_thrs = vec![0.3, 0.55, 0.9];
            ev.params.area_ranges = vec![
                AreaRange {
                    label: "all".into(),
                    range: [0.0, 1e10],
                },
                AreaRange {
                    label: "mid".into(),
                    range: [900.0, 20_000.0],
                },
            ];
        });
        // A geometry other than boxes reads the records, built on demand.
        assert_same_evaluation(&label("evaluated as segm"), &gt, &rows, ncols, |ev| {
            ev.params.iou_type = IouType::Segm;
        });
    }
}

#[test]
fn columns_load_with_the_same_warnings_and_results_set() {
    let mut rng = StdRng::seed_from_u64(7);
    let gt = ground_truth(&mut rng);
    let mut rows = rows(&gt, &mut rng, 500, 7);
    // Corner pairs on most rows, to set off the census: past a 640×480 image
    // read as [x, y, w, h], inside it read as [x1, y1, x2, y2].
    for row in rows.chunks_exact_mut(7).take(400) {
        row[1..5].copy_from_slice(&[400.0, 300.0, 600.0, 450.0]);
    }
    let by_records = records(&gt, &rows, 7).unwrap();
    let by_columns = columns(&gt, &rows, 7).unwrap();
    let warned = |what: &str| by_records.load_warnings().iter().any(|w| w.contains(what));
    assert!(
        warned("[x1, y1, x2, y2]") && warned("image_id 999") && warned("category_id 77"),
        "the rows should set off every load warning: {:?}",
        by_records.load_warnings()
    );
    assert_eq!(by_records.load_warnings(), by_columns.load_warnings());
    let built = by_columns.coco();
    assert_eq!(built.load_warnings(), by_records.load_warnings());
    assert_eq!(
        format!("{:?}", built.dataset),
        format!("{:?}", by_records.dataset)
    );
}

#[test]
fn columns_refuse_what_the_records_refuse_with_the_same_error() {
    let mut rng = StdRng::seed_from_u64(11);
    let gt = ground_truth(&mut rng);
    let good = rows(&gt, &mut rng, 50, 7);
    let mut cases = Vec::new();
    for (col, value) in [(5, f64::NAN), (0, -1.0), (0, 2.5), (6, f64::NAN), (6, 1e30)] {
        let mut bad = good.clone();
        bad[7 * 31 + col] = value;
        cases.push((format!("column {col} = {value}"), bad));
    }
    for (label, bad) in cases {
        let want = records(&gt, &bad, 7).err().map(|e| e.to_string());
        let got = columns(&gt, &bad, 7).err().map(|e| e.to_string());
        assert!(want.is_some(), "{label}: the records should refuse it");
        assert_eq!(got, want, "{label}");
    }
}
