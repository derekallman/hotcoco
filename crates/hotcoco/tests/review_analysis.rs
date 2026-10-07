//! Regression tests for the 2026-10 review of the detection analysis layer.
//!
//! Each test pins a case where an analysis surface — `accumulate()`,
//! `compare()`, `calibration()`, or the Open Images hierarchy expansion —
//! produced a plausible wrong number instead of the right one or an error.

use std::collections::HashMap;

use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image, Segmentation};
use hotcoco::{COCO, COCOeval, CompareOpts, Hierarchy, compare};

fn image(id: u64) -> Image {
    Image {
        id,
        file_name: format!("{id}.jpg"),
        width: 640,
        height: 640,
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

fn det_box(id: u64, image_id: u64, category_id: u64, bbox: [f64; 4], score: f64) -> Annotation {
    Annotation {
        score: Some(score),
        ..gt_box(id, image_id, category_id, bbox)
    }
}

fn dataset(images: Vec<Image>, annotations: Vec<Annotation>, categories: Vec<Category>) -> Dataset {
    Dataset {
        info: None,
        images,
        annotations,
        categories,
        licenses: vec![],
    }
}

/// Four images, two categories. Every image has one ground truth per category
/// and one detection on it, shifted so its IoU lands between 0.5 and 0.75 on
/// some images and above 0.9 on others — AP50 and AP75 differ, so a threshold
/// row read under the wrong label shows up as a different number.
fn two_category_eval(cat_ids: Option<Vec<u64>>, use_cats: bool) -> COCOeval {
    let images: Vec<Image> = (1..=4).map(image).collect();
    let mut gts = Vec::new();
    let mut dts = Vec::new();
    let mut id = 1;
    for img in 1..=4u64 {
        for c in 1..=2u64 {
            let b = [10.0, 10.0, 100.0, 100.0];
            gts.push(gt_box(id, img, c, b));
            // IoU 0.667 on odd images (TP at 0.5, FP at 0.75), 0.905 on even.
            let shift = if img % 2 == 1 { 20.0 } else { 5.0 };
            let score = 0.5 + 0.1 * img as f64 - 0.05 * c as f64;
            dts.push(det_box(
                id + 100,
                img,
                c,
                [b[0] + shift, b[1], b[2], b[3]],
                score,
            ));
            id += 1;
        }
    }
    let cats = vec![cat(1), cat(2)];
    let gt = COCO::from_dataset(dataset(images.clone(), gts, cats.clone()));
    let dt = COCO::from_dataset(dataset(images, dts, cats));
    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    if let Some(ids) = cat_ids {
        ev.params.cat_ids = ids;
    }
    ev.params.use_cats = use_cats;
    ev.evaluate().expect("evaluable inputs");
    ev
}

fn summarized(mut ev: COCOeval) -> Vec<f64> {
    ev.accumulate();
    ev.summarize();
    ev.stats().expect("summarized").to_vec()
}

// --- 1. accumulate() reads the threshold rows evaluate() matched on ---------

/// Growing `iou_thrs` after `evaluate()` used to size the T axis from the new
/// grid and read rows the cells never had — a panic in debug builds, garbage in
/// release. The rows that were evaluated must be read under their own
/// threshold, and the ones that were not must stay "not computed".
#[test]
fn accumulate_after_growing_iou_thrs_reads_evaluated_rows() {
    let reference = summarized(two_category_eval(None, true));
    assert_ne!(
        reference[1], reference[2],
        "fixture: AP50 must differ from AP75"
    );

    let mut ev = two_category_eval(None, true);
    let evaluated = ev.params.iou_thrs.clone();
    let mut grown = vec![0.3];
    grown.extend(&evaluated);
    grown.push(0.97);
    assert_eq!(grown.len(), 12);
    ev.params.iou_thrs = grown;
    ev.accumulate();

    let acc = ev.accumulated().expect("accumulated");
    assert_eq!(acc.shape.t, 12, "the T axis follows params.iou_thrs");
    // The two thresholds evaluate() never matched at are "not computed".
    for t_idx in [0, 11] {
        for k in 0..acc.shape.k {
            let i = acc.recall_idx(t_idx, k, 0, acc.shape.m - 1);
            assert_eq!(
                acc.recall[i], -1.0,
                "threshold row {t_idx} was never evaluated and must read -1.0"
            );
        }
    }

    ev.summarize();
    let stats = ev.stats().expect("summarized");
    // AP, AP50, AP75 come from the evaluated rows, under their own labels.
    for (idx, name) in [(0, "AP"), (1, "AP50"), (2, "AP75")] {
        assert_eq!(
            stats[idx], reference[idx],
            "{name} after growing iou_thrs must equal the untouched run"
        );
    }
}

/// Reordering `iou_thrs` after `evaluate()` used to read the rows in their
/// evaluate-time order under the new labels: AP75 reported the 0.5 row.
#[test]
fn accumulate_after_reordering_iou_thrs_keeps_each_row_under_its_label() {
    let reference = summarized(two_category_eval(None, true));

    let mut ev = two_category_eval(None, true);
    ev.params.iou_thrs = vec![0.75, 0.5];
    let stats = summarized(ev);
    assert_eq!(stats[1], reference[1], "AP50 must read the 0.5 row");
    assert_eq!(stats[2], reference[2], "AP75 must read the 0.75 row");
}

/// The per-threshold analyses read the same evaluate-time rows `accumulate()`
/// does. Each used to index the cells by a threshold's position in
/// `params.iou_thrs` as it stands, so after a reorder `0.75` read the `0.5` row.
#[test]
fn analyses_after_reordering_iou_thrs_read_each_threshold_by_value() {
    let reference = two_category_eval(None, true);
    let mut ev = two_category_eval(None, true);
    ev.params.iou_thrs = vec![0.75, 0.5];

    let cal = |e: &COCOeval| {
        let c = e.calibration(10, 0.75).expect("calibration");
        (c.ece, c.mce, c.num_detections, c.per_category)
    };
    assert_eq!(cal(&ev), cal(&reference), "calibration at 0.75");

    let tide = |e: &COCOeval| {
        let t = e.tide_errors(0.75, 0.1).expect("tide");
        (t.delta_ap, t.counts, t.ap_base)
    };
    assert_eq!(tide(&ev), tide(&reference), "TIDE at 0.75");

    let diag = |e: &COCOeval| {
        let d = e.image_diagnostics(0.75, 0.5).expect("diagnostics");
        let mut per_image: Vec<_> = d
            .images
            .iter()
            .map(|(&id, s)| (id, s.tp, s.fp, s.fn_count))
            .collect();
        per_image.sort_unstable();
        (d.iou_thr, per_image)
    };
    let (thr, per_image) = diag(&ev);
    assert_eq!(thr, 0.75);
    assert_eq!(per_image, diag(&reference).1, "diagnostics at 0.75");
    assert!(
        per_image.iter().any(|&(_, _, fp, _)| fp > 0),
        "fixture: odd images are FPs at 0.75"
    );
}

/// A threshold added to `params.iou_thrs` after `evaluate()` has no row in the
/// cells. The analyses used to index past the end of the matrix; they must say
/// the threshold was never evaluated instead.
#[test]
fn analyses_reject_a_threshold_evaluate_never_matched_at() {
    let mut ev = two_category_eval(None, true);
    let mut grown = ev.params.iou_thrs.clone();
    grown.push(0.97);
    ev.params.iou_thrs = grown;

    let errors = [
        ev.calibration(10, 0.97)
            .map(|_| ())
            .expect_err("calibration"),
        ev.tide_errors(0.97, 0.1).map(|_| ()).expect_err("tide"),
        ev.image_diagnostics(0.97, 0.5)
            .map(|_| ())
            .expect_err("diagnostics"),
    ];
    for err in errors {
        assert!(
            err.to_string().contains("evaluate()"),
            "error must say the threshold was not evaluated, got: {err}"
        );
    }
    // A threshold that was evaluated still works on the grown grid.
    ev.calibration(10, 0.5).expect("0.5 was evaluated");
}

// --- 2. compare() refuses runs over different category sets -----------------

#[test]
fn compare_rejects_different_cat_ids() {
    let ev_a = two_category_eval(None, true);
    let ev_b = two_category_eval(Some(vec![1]), true);
    let err = compare(&ev_a, &ev_b, &CompareOpts::default())
        .expect_err("person-only AP minus all-class mAP is not a delta");
    assert!(
        err.to_string().contains("cat_ids"),
        "error must name the field, got: {err}"
    );
}

#[test]
fn compare_rejects_different_use_cats() {
    let ev_a = two_category_eval(None, true);
    let ev_b = two_category_eval(None, false);
    let err = compare(&ev_a, &ev_b, &CompareOpts::default())
        .expect_err("per-class and class-agnostic AP are different metrics");
    assert!(
        err.to_string().contains("use_cats"),
        "error must name the field, got: {err}"
    );
}

/// The K axis is averaged, never read by position across the two runs, so the
/// same categories in a different order compare cleanly.
#[test]
fn compare_accepts_reordered_cat_ids() {
    let ev_a = two_category_eval(None, true);
    let ev_b = two_category_eval(Some(vec![2, 1]), true);
    let result =
        compare(&ev_a, &ev_b, &CompareOpts::default()).expect("same category set in another order");
    for (key, delta) in &result.deltas {
        assert!(delta.abs() < 1e-12, "{key}: expected no delta, got {delta}");
    }
    for c in &result.per_category {
        assert_eq!(c.delta, 0.0, "category {} paired by id", c.cat_id);
    }
}

/// A pooled (`use_cats = false`) run has one K slot holding every category, so
/// it has no per-category AP. Reading that slot under `params.cat_ids[0]` used to
/// report the pooled AP as the first category's.
#[test]
fn compare_of_pooled_runs_has_no_per_category_rows() {
    let ev_a = two_category_eval(None, false);
    let ev_b = two_category_eval(None, false);
    let result = compare(&ev_a, &ev_b, &CompareOpts::default()).expect("same configuration");
    assert!(
        result.per_category.is_empty(),
        "pooled AP must not appear under a category id, got: {:?}",
        result
            .per_category
            .iter()
            .map(|c| c.cat_id)
            .collect::<Vec<_>>()
    );
}

// --- 3. calibration() does not report a category it has no detections for ---

#[test]
fn calibration_omits_category_without_detections() {
    let images = vec![image(1), image(2)];
    let gts = vec![
        gt_box(1, 1, 1, [10.0, 10.0, 50.0, 50.0]),
        gt_box(2, 2, 1, [10.0, 10.0, 50.0, 50.0]),
        // Category 2 has ground truth and no detections at all.
        gt_box(3, 1, 2, [200.0, 200.0, 50.0, 50.0]),
    ];
    let dts = vec![
        det_box(10, 1, 1, [10.0, 10.0, 50.0, 50.0], 0.9),
        det_box(11, 2, 1, [300.0, 300.0, 50.0, 50.0], 0.4),
    ];
    let cats = vec![cat(1), cat(2)];
    let gt = COCO::from_dataset(dataset(images.clone(), gts, cats.clone()));
    let dt = COCO::from_dataset(dataset(images, dts, cats));
    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    ev.evaluate().expect("evaluable inputs");

    let cal = ev.calibration(10, 0.5).expect("calibration");
    assert!(
        cal.per_category.contains_key(&1),
        "category 1 has detections"
    );
    assert!(
        !cal.per_category.contains_key(&2),
        "category 2 has no detections, so it has no calibration error; got {:?}",
        cal.per_category.get(&2)
    );
}

// --- 4. Open Images hierarchy expansion keeps distinct detections distinct ---

/// cat (1) and dog (2) are both children of animal (3).
fn animal_hierarchy() -> Hierarchy {
    let mut parents = HashMap::new();
    parents.insert(1, 3);
    parents.insert(2, 3);
    Hierarchy::from_parent_map(parents)
}

fn scores_of(coco: &COCO, category_id: u64) -> Vec<f64> {
    let mut scores: Vec<f64> = coco
        .dataset
        .annotations
        .iter()
        .filter(|a| a.category_id == category_id)
        .map(|a| a.score.expect("detection"))
        .collect();
    scores.sort_by(f64::total_cmp);
    scores
}

/// Two detections on the same box in sibling classes are two detections of
/// their shared ancestor, each at its own score. The old key had no score, so
/// file order decided: cat@0.3 listed first left a single animal@0.3 and the
/// dog's 0.9 was lost.
#[test]
fn expansion_keeps_each_child_score_on_its_ancestor_copy() {
    let b = [10.0, 10.0, 100.0, 100.0];
    let dts = vec![det_box(1, 1, 1, b, 0.3), det_box(2, 1, 2, b, 0.9)];
    let dt = COCO::from_dataset(dataset(vec![image(1)], dts, vec![cat(1), cat(2), cat(3)]));
    let hierarchy = animal_hierarchy();

    let expanded = hotcoco::detection::expand::expand_annotations(&dt, &hierarchy);
    assert_eq!(scores_of(&expanded, 3), vec![0.3, 0.9]);

    // Expanding the expanded set again — what a second `evaluate()` does — adds
    // nothing.
    let twice = hotcoco::detection::expand::expand_annotations(&expanded, &hierarchy);
    assert_eq!(
        twice.dataset.annotations.len(),
        expanded.dataset.annotations.len(),
        "a second expansion must be a no-op"
    );
}

fn polygon_ann(id: u64, category_id: u64, x: f64) -> Annotation {
    Annotation {
        id,
        image_id: 1,
        category_id,
        segmentation: Some(Segmentation::Polygon(vec![vec![
            x,
            10.0,
            x + 20.0,
            10.0,
            x + 20.0,
            30.0,
            x,
            30.0,
        ]])),
        area: Some(400.0),
        ..Default::default()
    }
}

/// Segmentation-only annotations used to all key as box `[0, 0, 0, 0]`, so
/// two different masks in one image expanded to a single ancestor copy.
#[test]
fn expansion_keeps_distinct_bboxless_masks_distinct() {
    let gt = COCO::from_dataset(dataset(
        vec![image(1)],
        vec![polygon_ann(1, 1, 10.0), polygon_ann(2, 1, 200.0)],
        vec![cat(1), cat(2), cat(3)],
    ));
    let hierarchy = animal_hierarchy();
    let expanded = hotcoco::detection::expand::expand_annotations(&gt, &hierarchy);
    let animals = expanded
        .dataset
        .annotations
        .iter()
        .filter(|a| a.category_id == 3)
        .count();
    assert_eq!(animals, 2, "two different masks are two ancestor objects");

    let twice = hotcoco::detection::expand::expand_annotations(&expanded, &hierarchy);
    assert_eq!(
        twice.dataset.annotations.len(),
        expanded.dataset.annotations.len()
    );
}

/// `evaluate()` replaces `coco_dt` with its expansion, so running it twice
/// expands an already-expanded set. That must change neither the detections
/// nor the numbers.
#[test]
fn oid_evaluate_twice_with_expand_dt_is_idempotent() {
    let b = [10.0, 10.0, 100.0, 100.0];
    let cats = vec![cat(1), cat(2), cat(3)];
    let gt = COCO::from_dataset(dataset(
        vec![image(1)],
        vec![gt_box(1, 1, 2, b)],
        cats.clone(),
    ));
    let dt = COCO::from_dataset(dataset(
        vec![image(1)],
        vec![det_box(1, 1, 1, b, 0.3), det_box(2, 1, 2, b, 0.9)],
        cats,
    ));
    let mut ev = COCOeval::new_oid(gt, dt, Some(animal_hierarchy()));
    ev.params.expand_dt = true;

    ev.evaluate().expect("evaluable inputs");
    let n_dt = ev.coco_dt().dataset.annotations.len();
    let n_gt = ev.coco_gt().dataset.annotations.len();
    ev.accumulate();
    ev.summarize();
    let first = ev.stats().expect("summarized").to_vec();

    ev.evaluate().expect("evaluable inputs");
    assert_eq!(ev.coco_dt().dataset.annotations.len(), n_dt);
    assert_eq!(ev.coco_gt().dataset.annotations.len(), n_gt);
    ev.accumulate();
    ev.summarize();
    assert_eq!(ev.stats().expect("summarized"), first.as_slice());
    assert_eq!(scores_of(ev.coco_dt(), 3), vec![0.3, 0.9]);
}

/// Two ground truths on the same box in sibling classes are two objects of
/// their shared ancestor, as the TF Object Detection API's
/// `OIDHierarchicalLabelsExpansion` expands each box row on its own. The
/// dedup used to compare each copy against the copies already made, so the
/// second animal vanished and recall at animal could reach 1.0 with one
/// detection. Copies are now compared only against the input, which is what
/// keeps a pre-expanded dataset (and a second `evaluate()`) from doubling.
#[test]
fn expansion_keeps_sibling_ground_truths_on_one_box_distinct() {
    let b = [10.0, 10.0, 100.0, 100.0];
    let gt = COCO::from_dataset(dataset(
        vec![image(1)],
        vec![gt_box(1, 1, 1, b), gt_box(2, 1, 2, b)],
        vec![cat(1), cat(2), cat(3)],
    ));
    let hierarchy = animal_hierarchy();
    let expanded = hotcoco::detection::expand::expand_annotations(&gt, &hierarchy);
    let animals = |c: &COCO| {
        c.dataset
            .annotations
            .iter()
            .filter(|a| a.category_id == 3)
            .count()
    };
    assert_eq!(animals(&expanded), 2, "one animal per source box");
    let twice = hotcoco::detection::expand::expand_annotations(&expanded, &hierarchy);
    assert_eq!(animals(&twice), 2, "re-expanding adds nothing");
}

/// A group-of box and a single-object box on the same coordinates are
/// different ground truths, so an existing group-of animal does not stand in
/// for the copy of a single dog.
#[test]
fn expansion_does_not_treat_a_group_of_box_as_its_single_twin() {
    let b = [10.0, 10.0, 100.0, 100.0];
    let group = Annotation {
        is_group_of: Some(true),
        ..gt_box(2, 1, 3, b)
    };
    let gt = COCO::from_dataset(dataset(
        vec![image(1)],
        vec![gt_box(1, 1, 1, b), group],
        vec![cat(1), cat(2), cat(3)],
    ));
    let expanded = hotcoco::detection::expand::expand_annotations(&gt, &animal_hierarchy());
    let singles = expanded
        .dataset
        .annotations
        .iter()
        .filter(|a| a.category_id == 3 && a.is_group_of != Some(true))
        .count();
    assert_eq!(singles, 1, "the dog's animal copy is not the group-of box");
}
