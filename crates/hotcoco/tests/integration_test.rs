#![allow(clippy::unwrap_used)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use hotcoco::convert::{
    coco_to_cvat, coco_to_dota, coco_to_oid, coco_to_voc, coco_to_yolo, cvat_to_coco, dota_to_coco,
    oid_results_to_anns, oid_to_coco, voc_to_coco, yolo_to_coco,
};
use hotcoco::params::IouType;
use hotcoco::report::Provenance;
use hotcoco::types::{Annotation, Category, Dataset, Image, Segmentation};
use hotcoco::{COCO, COCOeval, Hierarchy, quality};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

// ---------------------------------------------------------------------------
// Inline dataset builders
//
// A bare `Annotation` literal is thirteen lines, so a two-GT/three-detection
// scenario runs past a hundred and the geometry — the only part that decides
// what the test exercises — ends up buried in field boilerplate. That is not
// hypothetical: `test_oid_group_of_multi_match` sat at IoU 0.16 against a
// threshold of 0.5 for a whole release, never reaching the code path it was
// written for, because 6400/40000 is invisible inside that much noise.
//
// Keep each box on one line so the geometry is readable at a glance, and state
// the IoU in a comment wherever a test depends on clearing a threshold.
// ---------------------------------------------------------------------------

/// Ground-truth annotation. Chain `.group_of()` / `.crowd()` / `.in_cat()` as needed.
fn ann(id: u64, bbox: [f64; 4]) -> Annotation {
    Annotation {
        id,
        image_id: 1,
        category_id: 1,
        bbox: Some(bbox),
        area: Some(bbox[2] * bbox[3]),
        ..Default::default()
    }
}

/// Detection annotation — an [`ann`] carrying a score.
fn det(id: u64, bbox: [f64; 4], score: f64) -> Annotation {
    Annotation {
        score: Some(score),
        ..ann(id, bbox)
    }
}

/// Modifiers for the builders above. Add variants as tests need them —
/// `-D warnings` rejects any that sit unused.
trait AnnExt {
    fn group_of(self) -> Self;
    /// Mark as a COCO crowd region (`iscrowd`).
    fn crowd(self) -> Self;
    /// Move to image `img_id`; both builders default to image 1.
    fn in_img(self, img_id: u64) -> Self;
    /// Move to category `cat_id`; both builders default to category 1.
    fn in_cat(self, cat_id: u64) -> Self;
    /// Attach a segmentation mask.
    fn mask(self, seg: Segmentation) -> Self;
    /// Attach keypoints `[x, y, v, …]`; sets `num_keypoints` to the count of
    /// entries with `v > 0`, which is what gates GT-ignore in keypoint eval.
    fn kpts(self, kpts: Vec<f64>) -> Self;
    /// Override `area` (the builders default it to the bbox area) — for masks
    /// whose pixel count differs from their bbox.
    fn with_area(self, area: f64) -> Self;
}

impl AnnExt for Annotation {
    fn group_of(mut self) -> Self {
        self.is_group_of = Some(true);
        self
    }

    fn crowd(mut self) -> Self {
        self.iscrowd = true;
        self
    }

    fn in_img(mut self, img_id: u64) -> Self {
        self.image_id = img_id;
        self
    }

    fn in_cat(mut self, cat_id: u64) -> Self {
        self.category_id = cat_id;
        self
    }

    fn mask(mut self, seg: Segmentation) -> Self {
        self.segmentation = Some(seg);
        self
    }

    fn kpts(mut self, kpts: Vec<f64>) -> Self {
        let visible = kpts.iter().skip(2).step_by(3).filter(|&&v| v > 0.0).count();
        self.num_keypoints = Some(visible as u32);
        self.keypoints = Some(kpts);
        self
    }

    fn with_area(mut self, area: f64) -> Self {
        self.area = Some(area);
        self
    }
}

fn img(id: u64) -> Image {
    Image {
        id,
        file_name: format!("img{id}.jpg"),
        height: 640,
        width: 640,
        ..Default::default()
    }
}

fn cat(id: u64, name: &str) -> Category {
    Category {
        id,
        name: name.into(),
        ..Default::default()
    }
}

/// Assemble a dataset from the pieces above.
fn dataset(images: Vec<Image>, categories: Vec<Category>, annotations: Vec<Annotation>) -> Dataset {
    Dataset {
        info: None,
        images,
        annotations,
        categories,
        licenses: vec![],
    }
}

/// Axis-aligned IoU, for asserting the precondition a scenario depends on.
///
/// Tests that turn on a detection clearing an IoU threshold should assert that it
/// does, rather than trusting the reader to multiply. Independent of
/// `primitives::sim` on purpose — an oracle that shares code with its subject
/// checks nothing.
fn iou_of(a: [f64; 4], b: [f64; 4]) -> f64 {
    let (ax2, ay2) = (a[0] + a[2], a[1] + a[3]);
    let (bx2, by2) = (b[0] + b[2], b[1] + b[3]);
    let iw = (ax2.min(bx2) - a[0].max(b[0])).max(0.0);
    let ih = (ay2.min(by2) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Intersection-over-area of `a` — `inter / area(a)`, where `a` is the detection.
///
/// This is the measure COCO uses for crowd regions and Open Images uses for
/// group-of boxes: "the area of intersection of the detection and the box divided
/// by the area of the detection". A detection wholly inside the box scores 1.0
/// however small it is, which is the entire point and the thing plain IoU misses.
fn ioa_of(a: [f64; 4], b: [f64; 4]) -> f64 {
    let (ax2, ay2) = (a[0] + a[2], a[1] + a[3]);
    let (bx2, by2) = (b[0] + b[2], b[1] + b[3]);
    let iw = (ax2.min(bx2) - a[0].max(b[0])).max(0.0);
    let ih = (ay2.min(by2) - a[1].max(b[1])).max(0.0);
    let area_a = a[2] * a[3];
    if area_a > 0.0 { iw * ih / area_a } else { 0.0 }
}

/// Pixel-exact uncompressed RLE for the rectangle `x..x+rw` by `y..y+rh` in an
/// `h`-tall, `w`-wide image.
///
/// Built by construction, independent of `crate::mask` on purpose (an oracle
/// that shares code with its subject checks nothing). COCO RLE runs scan
/// column-major and alternate `[zeros, ones, zeros, …]` starting with zeros,
/// so a solid rectangle is: `x·h + y` leading zeros, then per column `rh` ones
/// separated by `h − rh` zeros, then the zeros after the last set pixel.
/// Rectangle masks make every intersection/union a product of side lengths, so
/// the segm tests below can state their IoUs as exact fractions.
fn rect_mask(h: u32, w: u32, x: u32, y: u32, rw: u32, rh: u32) -> Segmentation {
    assert!(x + rw <= w && y + rh <= h, "rectangle must fit the image");
    assert!(rw > 0 && rh > 0, "rectangle must have pixels");
    let mut counts = vec![x * h + y];
    for col in 0..rw {
        counts.push(rh);
        if col + 1 < rw {
            counts.push(h - rh);
        }
    }
    counts.push((h - y - rh) + (w - x - rw) * h);
    Segmentation::UncompressedRle {
        size: [h, w],
        counts,
    }
}

#[test]
fn test_load_gt() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");
    assert_eq!(coco.dataset.images.len(), 3);
    assert_eq!(coco.dataset.annotations.len(), 5);
    assert_eq!(coco.dataset.categories.len(), 2);
}

#[test]
fn test_load_res() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");
    assert_eq!(coco_dt.dataset.annotations.len(), 7);
    // All annotations should have scores
    for ann in &coco_dt.dataset.annotations {
        assert!(ann.score.is_some());
    }
}

#[test]
fn test_load_gt_tolerates_non_finite_floats() {
    // Python's json module emits bare NaN/Infinity/-Infinity (non-standard JSON)
    // and reads them back happily, so pycocotools accepts such files. serde_json
    // rejects them ("expected value" / "invalid number"). hotcoco should tolerate
    // them by normalizing non-finite values to null (-> None on Option<f64>),
    // matching pycocotools — WITHOUT corrupting strings that merely contain the
    // substring "Infinity"/"NaN".
    let json = r#"{
  "images": [
    {"id": 1, "width": 100, "height": 100, "file_name": "Infinity_scan.jpg"}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "area": NaN, "score": Infinity},
    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0], "area": 12.0, "score": -Infinity}
  ],
  "categories": [{"id": 1, "name": "thing"}]
}"#;
    let path = std::env::temp_dir().join("hotcoco_nan_gt_test.json");
    std::fs::write(&path, json).expect("write temp fixture");

    let coco = COCO::new(&path).expect("Failed to load GT with non-finite floats");

    assert_eq!(coco.dataset.images.len(), 1);
    assert_eq!(coco.dataset.annotations.len(), 2);
    assert_eq!(coco.dataset.categories.len(), 1);

    // Non-finite Option<f64> fields normalize to None.
    let a0 = &coco.dataset.annotations[0];
    assert_eq!(a0.area, None, "NaN area should become None");
    assert_eq!(a0.score, None, "Infinity score should become None");
    let a1 = &coco.dataset.annotations[1];
    assert_eq!(a1.area, Some(12.0), "finite area must be preserved");
    assert_eq!(a1.score, None, "-Infinity score should become None");

    // A string value that merely contains "Infinity" must be left untouched.
    assert_eq!(coco.dataset.images[0].file_name, "Infinity_scan.jpg");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_bbox_evaluation_runs() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();

    let eval = coco_eval.accumulated().expect("Accumulate should set eval");

    // Verify dimensions
    assert_eq!(eval.shape.t, 10); // IoU thresholds
    assert_eq!(eval.shape.r, 101); // recall thresholds
    assert_eq!(eval.shape.k, 2); // categories
    assert_eq!(eval.shape.a, 4); // area ranges
    assert_eq!(eval.shape.m, 3); // max_dets

    // The precision array should have valid values (not all -1)
    let has_valid = eval.precision.iter().any(|&v| v >= 0.0);
    assert!(has_valid, "Should have some valid precision values");

    // Check that recall is non-negative for at least some entries
    let has_recall = eval.recall.iter().any(|&v| v >= 0.0);
    assert!(has_recall, "Should have some valid recall values");

    // For perfect matches at IoU=0.5 (dt bboxes closely match gt),
    // we should get high AP values
    // At IoU=0.5, our detections are good matches
    let ap_50_idx = eval.precision_idx(0, 0, 0, 0, 2); // t=0 (IoU=0.5), r=0, k=0 (cat), a=0 (all), m=2 (maxDet=100)
    let ap_50 = eval.precision[ap_50_idx];
    assert!(
        ap_50 > 0.0,
        "AP@0.5 for category 'cat' should be positive, got {}",
        ap_50
    );
}

#[test]
fn test_get_ann_ids_filtering() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    // Filter by image
    let ids = coco.get_ann_ids(&[1], &[], None, None);
    assert_eq!(ids.len(), 2);

    // Filter by category
    let ids = coco.get_ann_ids(&[], &[1], None, None);
    assert_eq!(ids.len(), 3); // 3 annotations with cat_id=1

    // Filter by both
    let ids = coco.get_ann_ids(&[2], &[1], None, None);
    assert_eq!(ids.len(), 2); // img 2 has 2 cat_id=1 annotations

    // Filter by area range
    let ids = coco.get_ann_ids(&[], &[], Some([500.0, 2000.0]), None);
    assert_eq!(ids.len(), 2); // area 900 and 1600
}

/// The `is_crowd` filter of `get_ann_ids`, in both polarities — previously the
/// one filter parameter with no test at all.
#[test]
fn test_get_ann_ids_iscrowd_filter() {
    let coco = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            ann(1, [0.0, 0.0, 10.0, 10.0]),
            ann(2, [20.0, 20.0, 10.0, 10.0]).crowd(),
            ann(3, [40.0, 40.0, 10.0, 10.0]),
        ],
    ));

    assert_eq!(coco.get_ann_ids(&[], &[], None, Some(true)), vec![2]);
    assert_eq!(coco.get_ann_ids(&[], &[], None, Some(false)), vec![1, 3]);
    assert_eq!(coco.get_ann_ids(&[], &[], None, None).len(), 3);
    // Composes with the image filter.
    assert_eq!(coco.get_ann_ids(&[1], &[], None, Some(true)), vec![2]);
}

#[test]
fn test_summarize_prints() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();

    // `summarize_lines` is what `summarize()` prints; asserting on the strings
    // means this covers the formatting contract instead of only "did not panic",
    // which is all it did before.
    let lines = coco_eval.summarize_lines();
    assert_eq!(
        lines.len(),
        12,
        "bbox summary is 12 lines, got:\n{}",
        lines.join("\n")
    );

    for line in &lines {
        assert!(
            line.starts_with(" Average Precision") || line.starts_with(" Average Recall"),
            "unexpected summary line: {line:?}"
        );
        assert!(
            line.contains("IoU=") && line.contains("area=") && line.contains("maxDets="),
            "summary line is missing its parameter annotation: {line:?}"
        );
        // Every line ends in a formatted metric — `-1.000` for "not computed",
        // otherwise a value in [0, 1]. A NaN or an unformatted float shows up here.
        let value = line
            .rsplit('=')
            .next()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or_else(|| panic!("summary line has no parseable value: {line:?}"));
        assert!(
            value == -1.0 || (0.0..=1.0).contains(&value),
            "summary value {value} is neither the -1.0 sentinel nor in [0, 1]: {line:?}"
        );
    }

    // stats() is populated as a side effect and must agree with what was printed.
    assert_eq!(coco_eval.stats().expect("summarize sets stats").len(), 12);
}

/// Regression test for the iscrowd-vs-gt_ignore matching bug.
///
/// When a non-crowd GT is area-ignored (area outside the evaluated range),
/// it can be matched once by a detection (making that detection "ignored"),
/// but must NOT be re-matched by additional detections. Only crowd GTs
/// allow re-matching. The bug let area-ignored non-crowd GTs absorb
/// multiple detections as "ignored" instead of counting them as FP,
/// which inflated AP for medium/large area ranges.
#[test]
fn test_area_ignored_gt_does_not_absorb_multiple_detections() {
    // One image, one category, custom area range [500, 1e10].
    // GT_A: area 400, non-crowd → area-ignored (below 500)
    // GT_B: area 10000 → in range
    let gt_dataset = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            ann(1, [10.0, 10.0, 20.0, 20.0]),   // area 400 → area-ignored
            ann(2, [50.0, 50.0, 100.0, 100.0]), // area 10000 → in range
        ],
    );

    // DT1: matches GT_A exactly (IoU 1.0), area=400 (out of range), score=0.9
    //       → matches area-ignored GT_A → DT1 is "ignored"
    // DT2: overlaps GT_A (IoU 0.8, asserted below), area=500 (in range), score=0.8
    //       With fix: GT_A already matched, not crowd → can't re-match → FP
    //       With bug: GT_A is "ignorable" → re-match → DT2 also "ignored"
    // DT3: matches GT_B perfectly, area=10000 (in range), score=0.7 → TP
    //
    // Crucially, DT2 (the FP) has higher score than DT3 (the TP), so
    // the FP appears before the TP in the precision-recall curve,
    // reducing AP from 1.0 to ~0.5.
    assert!((iou_of([10.0, 10.0, 25.0, 20.0], [10.0, 10.0, 20.0, 20.0]) - 0.8).abs() < 1e-12);
    let dt_dataset = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            det(101, [10.0, 10.0, 20.0, 20.0], 0.9),
            det(102, [10.0, 10.0, 25.0, 20.0], 0.8),
            det(103, [50.0, 50.0, 100.0, 100.0], 0.7),
        ],
    );

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    // Custom area range: [500, 1e10] so GT_A (area=400) is area-ignored
    coco_eval.params.area_ranges = vec![hotcoco::AreaRange {
        label: "custom".into(),
        range: [500.0, 1e10],
    }];
    coco_eval.evaluate();
    coco_eval.accumulate();

    let eval = coco_eval.accumulated().unwrap();
    let m_idx = eval.shape.m - 1;

    // Correct behavior:
    //   Sorted by score: DT1(0.9), DT2(0.8), DT3(0.7)
    //   DT1 matches area-ignored GT_A → DT1 is "ignored".
    //   DT2 overlaps GT_A (IoU≈0.8) but GT_A already matched and not crowd → skip.
    //   DT2 unmatched, area=500 in range → FP.
    //   DT3 matches GT_B → TP.
    //   Non-ignored dets: DT2(FP, score=0.8), DT3(TP, score=0.7).
    //   AP@0.5 ≈ 0.5 (FP before TP in ranking).
    //
    // Buggy behavior (gt_ignore instead of iscrowd):
    //   DT2 re-matches area-ignored GT_A → DT2 also "ignored".
    //   Non-ignored dets: only DT3(TP). AP@0.5 = 1.0.
    let ap_sum: f64 = (0..eval.shape.r)
        .map(|r| {
            let idx = eval.precision_idx(0, r, 0, 0, m_idx);
            let p = eval.precision[idx];
            if p < 0.0 { 0.0 } else { p }
        })
        .sum();
    let ap = ap_sum / eval.shape.r as f64;

    assert!(
        ap < 0.9,
        "AP should be ~0.5 (with FP counted), got {ap:.4}. \
         If AP ≈ 1.0, area-ignored non-crowd GT is incorrectly absorbing multiple detections."
    );
    assert!(ap > 0.3, "AP should be ~0.5, got {ap:.4}");
}

/// Helper to run bbox eval and return the 12 summary stats.
fn run_bbox_eval(coco_gt: COCO, coco_dt: COCO) -> Vec<f64> {
    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();
    coco_eval.summarize();
    coco_eval
        .stats()
        .expect("summarize should set stats")
        .to_vec()
}

/// Edge case test fixture covering crowd re-matching, bbox at origin,
/// tied scores, all-FP images, all-miss images, area boundaries, and
/// empty categories.
#[test]
fn test_edge_cases() {
    let gt_path = fixtures_dir().join("edge_gt.json");
    let dt_path = fixtures_dir().join("edge_dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load edge GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load edge DT");

    let stats = run_bbox_eval(coco_gt, coco_dt);
    assert_eq!(stats.len(), 12);

    // Expected values from running the evaluator on the edge fixtures.
    // The fixture exercises:
    // - Crowd GT re-matching (image 1): 3 dets match crowd, none are FP
    // - Bbox at origin [0,0,20,20] (image 2): correct zero-length RLE run handling
    // - Tied scores 0.5 (image 3): deterministic matching of 2 non-overlapping GTs
    // - All-FP (image 4): cat=2 dets with no GT → precision=0 for cat 2
    // - All-miss (image 5): 2 GTs with no dets → recall=0
    // - Area boundaries (image 6): 32²=1024 (medium), 96²=9216 (large boundary)
    // - Empty category (cat 3): no GT, no DT → does not affect metrics
    #[rustfmt::skip]
    let expected: &[f64] = &[
        0.712871,  // AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        0.712871,  // AP @[ IoU=0.50      | area=   all | maxDets=100 ]
        0.712871,  // AP @[ IoU=0.75      | area=   all | maxDets=100 ]
        0.663366,  // AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
        1.000000,  // AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
        1.000000,  // AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
        0.428571,  // AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
        0.714286,  // AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
        0.714286,  // AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        0.666667,  // AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
        1.000000,  // AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
        1.000000,  // AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
    ];

    let tol = 1e-4;
    for (i, (&got, &exp)) in stats.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < tol,
            "stats[{i}] mismatch: got {got:.6}, expected {exp:.6}"
        );
    }
}

/// Test that crowd GTs allow re-matching by multiple detections.
/// All detections overlapping a crowd GT should be "ignored" (not FP).
#[test]
fn test_crowd_rematching() {
    let gt_dataset = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![ann(1, [10.0, 10.0, 50.0, 50.0]).crowd()],
    );

    // 3 detections, all nested inside the crowd region.
    let dt_dataset = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            det(101, [10.0, 10.0, 50.0, 50.0], 0.9),
            det(102, [12.0, 12.0, 48.0, 48.0], 0.8),
            det(103, [15.0, 15.0, 45.0, 45.0], 0.7),
        ],
    );

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();

    let eval = coco_eval.accumulated().unwrap();
    let m_idx = eval.shape.m - 1; // maxDets=100

    // With only a crowd GT and no non-crowd GTs:
    // - All 3 detections should match the crowd GT (re-matching allowed)
    // - All 3 detections become "ignored" (matched to crowd)
    // - No non-ignored detections remain → no FPs
    // - But also no non-crowd GT → recall is -1 (undefined)
    //
    // The key assertion: recall should be -1 (no non-crowd GT to measure against)
    // and precision entries should all be -1 (no valid recall points).
    let recall_idx = eval.recall_idx(0, 0, 0, m_idx); // t=0, k=0, a=0, m=maxDets
    let recall = eval.recall[recall_idx];
    assert!(
        recall < 0.0,
        "Recall should be -1 (no non-crowd GT), got {recall:.4}"
    );

    // Verify no FPs: if crowd re-matching is broken, some detections would be
    // FP and precision would show valid (non-negative) values at some recall points.
    // With correct behavior, all precision values should be -1.
    let all_neg = (0..eval.shape.r).all(|r| {
        let idx = eval.precision_idx(0, r, 0, 0, m_idx);
        eval.precision[idx] < 0.0
    });
    assert!(
        all_neg,
        "All precision values should be -1 (no non-crowd GT), \
         but some are non-negative — crowd re-matching may be broken"
    );
}

/// Test that 0-based annotation/image/category IDs work correctly.
/// Previously, `dt_matches`/`gt_matches` used 0 as the "unmatched" sentinel,
/// so a valid match to annotation id=0 was treated as unmatched → false positive.
#[test]
fn test_zero_based_ids() {
    let gt_path = fixtures_dir().join("zero_gt.json");
    let dt_path = fixtures_dir().join("zero_dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load zero GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load zero DT");

    let stats = run_bbox_eval(coco_gt, coco_dt);

    // Perfect detections for all 3 GTs → AP and AR should be 1.0
    let ap = stats[0]; // AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
    let ap50 = stats[1]; // AP @[ IoU=0.50 | area=all | maxDets=100 ]
    let ar100 = stats[8]; // AR @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
    assert!(
        (ap - 1.0).abs() < 1e-6,
        "AP should be 1.0 for perfect detections with 0-based IDs, got {ap:.6}"
    );
    assert!(
        (ap50 - 1.0).abs() < 1e-6,
        "AP@0.5 should be 1.0, got {ap50:.6}"
    );
    assert!(
        (ar100 - 1.0).abs() < 1e-6,
        "AR@100 should be 1.0, got {ar100:.6}"
    );
}

/// Test dataset statistics computed from the gt.json fixture.
///
/// gt.json has 3 images (all 100x100), 5 annotations (0 crowd), 2 categories.
/// Annotations: img1/cat(400), img1/dog(900), img2/cat(1600), img2/cat(400), img3/dog(2500).
#[test]
fn test_dataset_stats() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");
    let stats = coco.stats();

    assert_eq!(stats.image_count, 3);
    assert_eq!(stats.annotation_count, 5);
    assert_eq!(stats.category_count, 2);
    assert_eq!(stats.crowd_count, 0);

    // per_category sorted by ann_count desc: cat(3) then dog(2)
    assert_eq!(stats.per_category.len(), 2);
    assert_eq!(stats.per_category[0].name, "cat");
    assert_eq!(stats.per_category[0].ann_count, 3);
    assert_eq!(stats.per_category[0].img_count, 2); // imgs 1 and 2
    assert_eq!(stats.per_category[0].crowd_count, 0);
    assert_eq!(stats.per_category[1].name, "dog");
    assert_eq!(stats.per_category[1].ann_count, 2);
    assert_eq!(stats.per_category[1].img_count, 2); // imgs 1 and 3
    assert_eq!(stats.per_category[1].crowd_count, 0);

    // All images are 100x100
    assert_eq!(stats.image_width.min, 100.0);
    assert_eq!(stats.image_width.max, 100.0);
    assert_eq!(stats.image_width.mean, 100.0);
    assert_eq!(stats.image_width.median, 100.0);
    assert_eq!(stats.image_height.min, 100.0);
    assert_eq!(stats.image_height.max, 100.0);

    // areas: 400, 900, 1600, 400, 2500 → sorted: 400, 400, 900, 1600, 2500
    assert_eq!(stats.annotation_area.min, 400.0);
    assert_eq!(stats.annotation_area.max, 2500.0);
    let expected_mean = (400.0 + 900.0 + 1600.0 + 400.0 + 2500.0) / 5.0;
    assert!((stats.annotation_area.mean - expected_mean).abs() < 1e-9);
    assert_eq!(stats.annotation_area.median, 900.0); // middle of 5 values
}

/// Test that `load_res` unconditionally reassigns annotation IDs.
#[test]
fn test_zero_based_ids_load_res() {
    let gt_path = fixtures_dir().join("zero_gt.json");
    let dt_path = fixtures_dir().join("zero_dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load zero GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load zero DT");

    // load_res should assign IDs 1, 2, 3 unconditionally
    let ids: Vec<u64> = coco_dt.dataset.annotations.iter().map(|a| a.id).collect();
    assert_eq!(
        ids,
        vec![1, 2, 3],
        "load_res should assign 1-indexed IDs unconditionally"
    );
}

// ---------------------------------------------------------------------------
// Dataset operation tests: filter, merge, split, sample
// ---------------------------------------------------------------------------

/// gt.json: 3 images, 5 annotations, 2 categories (cat=1, dog=2)
/// - img1: ann1(cat), ann2(dog)
/// - img2: ann3(cat,area=1600), ann4(cat,area=400)
/// - img3: ann5(dog,area=2500)

#[test]
fn test_filter_by_cat() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    // Filter to cat_id=1 ("cat") only
    let filtered = coco.filter(Some(&[1]), None, None, true);

    // Annotations: 1(cat,img1), 3(cat,img2), 4(cat,img2) — 3 anns
    assert_eq!(filtered.annotations.len(), 3);
    // Images: img1 and img2 (img3 has only dog anns)
    assert_eq!(filtered.images.len(), 2);
    let img_ids: HashSet<u64> = filtered.images.iter().map(|i| i.id).collect();
    assert!(img_ids.contains(&1));
    assert!(img_ids.contains(&2));
    // Categories: only "cat"
    assert_eq!(filtered.categories.len(), 1);
    assert_eq!(filtered.categories[0].name, "cat");
}

#[test]
fn test_filter_drop_vs_keep_empty() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    // Filter to cat_id=1 with drop_empty_images=false: all 3 images kept
    let kept = coco.filter(Some(&[1]), None, None, false);
    assert_eq!(kept.images.len(), 3);
    assert_eq!(kept.annotations.len(), 3);

    // Same filter with drop_empty_images=true: only images with cat anns
    let dropped = coco.filter(Some(&[1]), None, None, true);
    assert_eq!(dropped.images.len(), 2);
    assert_eq!(dropped.annotations.len(), 3);
}

#[test]
fn test_filter_area_rng() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    // Keep only annotations with area in [500, 2000]
    // gt.json areas: 400, 900, 1600, 400, 2500
    // In range: 900(img1,dog), 1600(img2,cat)
    let filtered = coco.filter(None, None, Some([500.0, 2000.0]), true);
    assert_eq!(filtered.annotations.len(), 2);
    for ann in &filtered.annotations {
        let area = ann.area.unwrap_or(0.0);
        assert!((500.0..=2000.0).contains(&area), "area {area} out of range");
    }
}

#[test]
fn test_merge_same_cats() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    // Merge gt with a copy: should double images and annotations
    let ds1 = &coco.dataset;
    let ds2 = &coco.dataset;
    let merged = COCO::merge(&[ds1, ds2]).expect("Merge should succeed");

    // Image and annotation counts doubled
    assert_eq!(merged.images.len(), ds1.images.len() * 2);
    assert_eq!(merged.annotations.len(), ds1.annotations.len() * 2);

    // All image IDs must be globally unique
    let img_ids: HashSet<u64> = merged.images.iter().map(|i| i.id).collect();
    assert_eq!(
        img_ids.len(),
        merged.images.len(),
        "Image IDs must be unique"
    );

    // All annotation IDs must be globally unique
    let ann_ids: HashSet<u64> = merged.annotations.iter().map(|a| a.id).collect();
    assert_eq!(
        ann_ids.len(),
        merged.annotations.len(),
        "Ann IDs must be unique"
    );

    // Categories unchanged
    assert_eq!(merged.categories.len(), ds1.categories.len());
}

#[test]
fn test_merge_different_cats_error() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    // Build a dataset with a different category taxonomy
    let mut ds2 = coco.dataset.clone();
    ds2.categories[0].name = "horse".into();

    let result = COCO::merge(&[&coco.dataset, &ds2]);
    assert!(result.is_err(), "Merging different taxonomies should fail");
}

#[test]
fn test_split_coverage() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    // 3 images, val_frac=0.33 → 1 val, 2 train (round(3*0.33)=1)
    let (train, val, test) = coco.split(0.33, None, 42);
    assert!(test.is_none());

    let all_ids: HashSet<u64> = coco.dataset.images.iter().map(|i| i.id).collect();
    let train_ids: HashSet<u64> = train.images.iter().map(|i| i.id).collect();
    let val_ids: HashSet<u64> = val.images.iter().map(|i| i.id).collect();

    // No overlap
    assert!(
        train_ids.is_disjoint(&val_ids),
        "train and val must not overlap"
    );
    // Union covers all
    let union: HashSet<u64> = train_ids.union(&val_ids).copied().collect();
    assert_eq!(union, all_ids, "train+val must cover all images");

    // Annotation image_ids all reference valid images in their split
    for ann in &train.annotations {
        assert!(train_ids.contains(&ann.image_id));
    }
    for ann in &val.annotations {
        assert!(val_ids.contains(&ann.image_id));
    }
}

#[test]
fn test_split_determinism() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    let (train1, val1, _) = coco.split(0.33, None, 42);
    let (train2, val2, _) = coco.split(0.33, None, 42);

    let train1_ids: Vec<u64> = {
        let mut v: Vec<u64> = train1.images.iter().map(|i| i.id).collect();
        v.sort_unstable();
        v
    };
    let train2_ids: Vec<u64> = {
        let mut v: Vec<u64> = train2.images.iter().map(|i| i.id).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(train1_ids, train2_ids, "Same seed must produce same split");

    let val1_ids: Vec<u64> = {
        let mut v: Vec<u64> = val1.images.iter().map(|i| i.id).collect();
        v.sort_unstable();
        v
    };
    let val2_ids: Vec<u64> = {
        let mut v: Vec<u64> = val2.images.iter().map(|i| i.id).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(val1_ids, val2_ids, "Same seed must produce same split");

    // Different seeds must actually change the partition. The 3-image fixture
    // is too small to guarantee that for any given seed pair, so use a bigger
    // synthetic dataset; the shuffle is deterministic, so if *some* seed pair
    // in this range coincided we could simply pick another — finding none at
    // all would mean the seed is ignored, which is what this half asserts.
    // (The previous version computed a split for seed 99 and asserted nothing.)
    let images: Vec<Image> = (1..=12).map(img).collect();
    let big = COCO::from_dataset(dataset(images, vec![cat(1, "thing")], vec![]));
    let val_ids = |seed: u64| -> HashSet<u64> {
        let (_, val, _) = big.split(0.5, None, seed);
        val.images.iter().map(|i| i.id).collect()
    };
    let baseline = val_ids(0);
    assert!(
        (1..8).any(|seed| val_ids(seed) != baseline),
        "seeds 1..8 all reproduced seed 0's split — the seed is being ignored"
    );
}

#[test]
fn test_sample_n() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    let sampled = coco.sample(Some(2), None, 42);
    assert_eq!(sampled.images.len(), 2);

    // All annotations belong to sampled images
    let img_ids: HashSet<u64> = sampled.images.iter().map(|i| i.id).collect();
    for ann in &sampled.annotations {
        assert!(img_ids.contains(&ann.image_id));
    }

    // Categories preserved in full
    assert_eq!(sampled.categories.len(), coco.dataset.categories.len());
}

/// Regression test: sparse evaluate() + grouped accumulate() must produce identical metrics
/// to the previous dense implementation.
///
/// Expected values were captured from the edge fixtures on the `main` branch before the
/// sparse refactor and cross-validated with the `test_edge_cases` test above.
/// Running `data/bench_parity.py` against val2017 further confirms parity with pycocotools
/// (bbox ≤1e-4, segm ≤2e-4, keypoints exact).
#[test]
fn empty_image_category_pairs_do_not_change_metrics() {
    // The premise the sparse evaluation path rests on: `collect_sparse_pairs`
    // drives evaluation from the annotation index, skipping every (image,
    // category) pair with neither ground truth nor a detection. That is only
    // sound if such pairs contribute nothing — so pad the dataset with them and
    // require the metrics to come out bit-identical.
    //
    // This test used to be `test_evaluate_sparse_matches_dense`, which claimed to
    // compare the sparse path against "the previously-verified dense
    // implementation". There is no dense implementation — it was removed — so the
    // body simply re-asserted `test_edge_cases`' twelve constants against the same
    // fixture, and no second computation ever ran.
    let coco_gt = COCO::new(&fixtures_dir().join("edge_gt.json")).expect("Failed to load edge GT");
    let coco_dt = coco_gt
        .load_res(&fixtures_dir().join("edge_dt.json"))
        .expect("Failed to load edge DT");
    let baseline = run_bbox_eval(coco_gt, coco_dt);

    // Same data, plus images that carry no annotations at all and a category that
    // appears in no annotation — pure empty pairs, nothing else changed.
    let mut gt = COCO::new(&fixtures_dir().join("edge_gt.json"))
        .expect("Failed to load edge GT")
        .dataset
        .clone();
    let next_img_id = gt.images.iter().map(|i| i.id).max().unwrap_or(0) + 1;
    for k in 0..5 {
        gt.images.push(img(next_img_id + k));
    }
    let next_cat_id = gt.categories.iter().map(|c| c.id).max().unwrap_or(0) + 1;
    gt.categories.push(cat(next_cat_id, "never_annotated"));

    let padded_gt = COCO::from_dataset(gt);
    let padded_dt = padded_gt
        .load_res(&fixtures_dir().join("edge_dt.json"))
        .expect("Failed to load edge DT against padded GT");
    let padded = run_bbox_eval(padded_gt, padded_dt);

    assert_eq!(baseline.len(), padded.len());
    for (i, (&base, &pad)) in baseline.iter().zip(padded.iter()).enumerate() {
        assert_eq!(
            base, pad,
            "stats[{i}] changed when empty (image, category) pairs were added: \
             {base} -> {pad}"
        );
    }
}

#[test]
fn test_sample_determinism() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");

    let s1 = coco.sample(Some(2), None, 42);
    let s2 = coco.sample(Some(2), None, 42);

    let ids1: HashSet<u64> = s1.images.iter().map(|i| i.id).collect();
    let ids2: HashSet<u64> = s2.images.iter().map(|i| i.id).collect();
    assert_eq!(ids1, ids2, "Same seed must produce same sample");
}

/// A 10-image dataset with one annotation per image, for the split/sample
/// tests that need more images than the 3-image `gt.json` fixture offers.
fn ten_image_coco() -> COCO {
    let images: Vec<Image> = (1..=10).map(img).collect();
    let annotations: Vec<Annotation> = (1..=10)
        .map(|i| ann(i, [0.0, 0.0, 10.0, 10.0]).in_img(i))
        .collect();
    COCO::from_dataset(dataset(images, vec![cat(1, "thing")], annotations))
}

/// Three-way `split(val_frac, test_frac = Some(..))` — previously untested.
#[test]
fn test_split_three_way() {
    let coco = ten_image_coco();
    let (train, val, test) = coco.split(0.2, Some(0.3), 7);
    let test = test.expect("test_frac requested a third split");

    // 10 images: round(10·0.2) = 2 val, round(10·0.3) = 3 test, 5 train.
    assert_eq!(val.images.len(), 2);
    assert_eq!(test.images.len(), 3);
    assert_eq!(train.images.len(), 5);

    let ids = |d: &Dataset| -> HashSet<u64> { d.images.iter().map(|i| i.id).collect() };
    let (tr, va, te) = (ids(&train), ids(&val), ids(&test));
    assert!(tr.is_disjoint(&va) && tr.is_disjoint(&te) && va.is_disjoint(&te));
    let mut all = tr.clone();
    all.extend(&va);
    all.extend(&te);
    assert_eq!(all.len(), 10, "the three splits must cover every image");

    // Annotations follow their images, and categories survive in every split.
    for (split, split_ids) in [(&train, &tr), (&val, &va), (&test, &te)] {
        for a in &split.annotations {
            assert!(split_ids.contains(&a.image_id));
        }
        assert_eq!(split.categories.len(), 1);
    }
}

/// Fraction-based `sample(frac = Some(..))` — previously only the `n` form was
/// tested.
#[test]
fn test_sample_frac() {
    let coco = ten_image_coco();

    let half = coco.sample(None, Some(0.5), 42);
    assert_eq!(half.images.len(), 5);
    let img_ids: HashSet<u64> = half.images.iter().map(|i| i.id).collect();
    for a in &half.annotations {
        assert!(img_ids.contains(&a.image_id));
    }
    assert_eq!(half.categories.len(), coco.dataset.categories.len());

    // The count truncates: 10 · 0.35 = 3.5 → 3 images.
    assert_eq!(coco.sample(None, Some(0.35), 42).images.len(), 3);
    // A fraction over 1.0 clamps to the whole dataset.
    assert_eq!(coco.sample(None, Some(1.5), 42).images.len(), 10);
}

// ---------------------------------------------------------------------------
// LVIS federated evaluation tests
// ---------------------------------------------------------------------------

/// LVIS test 1: neg_category_ids — unmatched DTs on an image where the
/// category is confirmed absent must count as FP.
///
/// Image 1 carries the only GT and a perfect detection (score 0.9); image 2
/// has no GT but lists category 1 in `neg_category_ids`, and the detector
/// fires there with a *higher* score (0.95). Counted as an FP, that detection
/// outranks the TP: the PR curve is (r=0, p=0) then (r=1, p=0.5), whose
/// envelope is 0.5 at every recall point → AP = 0.5 exactly, at every IoU
/// threshold.
///
/// The old assertion was `ap <= 0.0` against an empty-GT fixture, which passed
/// on the `-1.0` "not computed" sentinel (no GT anywhere → nothing computable)
/// — it could not tell "FP counted" from "FP dropped" from "nothing ran".
/// Here each failure mode lands on a distinct value: FP counted → 0.5, FP
/// wrongly dropped → 1.0, nothing computed → −1.0.
#[test]
fn test_lvis_neg_category_counts_as_fp() {
    let gt_ds = dataset(
        vec![
            img(1),
            Image {
                neg_category_ids: vec![1],
                ..img(2)
            },
        ],
        vec![Category {
            frequency: Some("r".into()),
            ..cat(1, "cat1")
        }],
        vec![ann(1, [0.0, 0.0, 20.0, 20.0])],
    );
    let dt_ds = dataset(
        vec![img(1), img(2)],
        vec![cat(1, "cat1")],
        vec![
            det(101, [0.0, 0.0, 20.0, 20.0], 0.9), // TP on image 1
            det(102, [0.0, 0.0, 20.0, 20.0], 0.95).in_img(2), // FP on the neg-cat image
        ],
    );

    let coco_gt = COCO::from_dataset(gt_ds);
    let coco_dt = COCO::from_dataset(dt_ds);

    let mut ev = COCOeval::new_lvis(coco_gt, coco_dt, IouType::Bbox);
    ev.run();

    let results = ev.get_results(None, false);
    let ap = results["AP"];
    assert!(
        (ap - 0.5).abs() < 1e-9,
        "the neg-category FP must halve AP to exactly 0.5 \
         (1.0 means it was dropped; -1.0 means nothing was computed): got {ap}"
    );
}

/// LVIS test 2: unlisted category — DT fires on an image where the category
/// has neither GT nor a neg/not_exhaustive listing. The DT pair should be
/// silently dropped (not included in evaluation at all), so AP is unaffected.
#[test]
fn test_lvis_unlisted_category_not_penalized() {
    // Image A: has GT + matching DT (correct).
    // Image B: no GT, cat not listed anywhere, but DT fires.
    // Expected: the DT on image B is dropped; AP equals the single-image case.
    let gt_ds = dataset(
        vec![img(1), img(2)], // neither image lists neg or not_exhaustive
        vec![Category {
            frequency: Some("f".into()),
            ..cat(1, "cat1")
        }],
        vec![ann(1, [0.0, 0.0, 20.0, 20.0])], // GT only on image A
    );
    let dt_ds = dataset(
        vec![img(1), img(2)],
        vec![cat(1, "cat1")],
        vec![
            det(101, [0.0, 0.0, 20.0, 20.0], 0.9), // matches GT on image A
            det(102, [0.0, 0.0, 20.0, 20.0], 0.8).in_img(2), // fires on image B — should be dropped
        ],
    );

    let coco_gt_two = COCO::from_dataset(gt_ds.clone());
    let coco_dt_two = COCO::from_dataset(dt_ds);

    let mut ev_two = COCOeval::new_lvis(coco_gt_two, coco_dt_two, IouType::Bbox);
    ev_two.run();

    // Baseline: only image A with its GT and matching DT (perfect AP = 1.0).
    let gt_ds_one = dataset(
        vec![img(1)],
        vec![Category {
            frequency: Some("f".into()),
            ..cat(1, "cat1")
        }],
        vec![ann(1, [0.0, 0.0, 20.0, 20.0])],
    );
    let dt_ds_one = dataset(
        vec![img(1)],
        vec![cat(1, "cat1")],
        vec![det(101, [0.0, 0.0, 20.0, 20.0], 0.9)],
    );

    let mut ev_one = COCOeval::new_lvis(
        COCO::from_dataset(gt_ds_one),
        COCO::from_dataset(dt_ds_one),
        IouType::Bbox,
    );
    ev_one.run();

    let ap_two = ev_two.get_results(None, false)["AP"];
    let ap_one = ev_one.get_results(None, false)["AP"];

    assert!(
        (ap_two - ap_one).abs() < 1e-6,
        "Unlisted DT on image B should not change AP: two-image AP={ap_two:.6}, one-image AP={ap_one:.6}"
    );
}

/// LVIS test 3: not_exhaustive_category_ids — unmatched DTs in a
/// not-exhaustively-checked image are ignored (not FP).
#[test]
fn test_lvis_not_exhaustive_unmatched_ignored() {
    // 1 image, 1 category.
    // GT: 1 annotation (area=400).
    // DT: 2 detections — DT1 matches GT (TP), DT2 is unmatched.
    // Image has cat 1 in not_exhaustive_category_ids.
    // DT2 must be ignored → precision at recall=1 stays 1.0 → AP = 1.0.
    let gt_ds = dataset(
        vec![Image {
            not_exhaustive_category_ids: vec![1], // not_exhaustive for cat 1
            ..img(1)
        }],
        vec![Category {
            frequency: Some("c".into()),
            ..cat(1, "cat1")
        }],
        vec![ann(1, [0.0, 0.0, 20.0, 20.0])], // area 400
    );
    let dt_ds = dataset(
        vec![img(1)],
        vec![cat(1, "cat1")],
        vec![
            det(101, [0.0, 0.0, 20.0, 20.0], 0.9), // matches GT
            det(102, [0.0, 0.0, 10.0, 10.0], 0.5), // area 100, unmatched — should be ignored
        ],
    );

    let mut ev = COCOeval::new_lvis(
        COCO::from_dataset(gt_ds),
        COCO::from_dataset(dt_ds),
        IouType::Bbox,
    );
    ev.run();

    let ap = ev.get_results(None, false)["AP"];
    assert!(
        (ap - 1.0).abs() < 1e-6,
        "Unmatched DT in not_exhaustive image should be ignored; AP should be 1.0, got {ap}"
    );
}

// ============================================================
// confusion_matrix tests
// ============================================================

/// All DTs match their correct category → pure diagonal matrix.
#[test]
fn test_confusion_matrix_perfect() {
    // 2 categories: cat(1)=idx 0, dog(2)=idx 1; background=idx 2
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            ann(1, [0.0, 0.0, 50.0, 50.0]),            // cat GT
            ann(2, [60.0, 0.0, 50.0, 50.0]).in_cat(2), // dog GT
        ],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            det(101, [0.0, 0.0, 50.0, 50.0], 0.9), // cat DT → matches cat GT
            det(102, [60.0, 0.0, 50.0, 50.0], 0.8).in_cat(2), // dog DT → matches dog GT
        ],
    ));

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    let cm = ev.confusion_matrix(0.5, None, None);

    assert_eq!(cm.num_cats, 2);
    assert_eq!(cm.cat_ids, vec![1, 2]);

    // Diagonal TPs
    assert_eq!(cm.get(0, 0), 1, "cat→cat TP should be 1");
    assert_eq!(cm.get(1, 1), 1, "dog→dog TP should be 1");

    // No cross-category confusion
    assert_eq!(cm.get(0, 1), 0, "cat should not be predicted as dog");
    assert_eq!(cm.get(1, 0), 0, "dog should not be predicted as cat");

    // No FPs or FNs
    assert_eq!(cm.get(0, 2), 0, "no missed cats");
    assert_eq!(cm.get(1, 2), 0, "no missed dogs");
    assert_eq!(cm.get(2, 0), 0, "no spurious cat predictions");
    assert_eq!(cm.get(2, 1), 0, "no spurious dog predictions");
}

/// DT of category dog overlaps GT of category cat → off-diagonal confusion cell.
#[test]
fn test_confusion_matrix_class_confusion() {
    // 1 GT: cat(1) at [0,0,50,50]
    // 1 DT: dog(2) at same location → IoU=1.0 with cat GT → recorded as gt=cat, pred=dog
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![det(101, [0.0, 0.0, 50.0, 50.0], 0.9).in_cat(2)],
    ));

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    let cm = ev.confusion_matrix(0.5, None, None);

    // GT cat (idx 0) was predicted as dog (idx 1) → off-diagonal confusion
    assert_eq!(cm.get(0, 1), 1, "GT cat predicted as dog should be 1");
    // No FN (GT was matched, just to wrong category)
    assert_eq!(cm.get(0, 2), 0, "GT cat should not be a missed FN");
    // No FP (DT matched a GT)
    assert_eq!(cm.get(2, 1), 0, "dog DT should not be a spurious FP");
    // No TP for cat
    assert_eq!(cm.get(0, 0), 0);
}

/// DT with no nearby GT → lands in the background (FP) row.
#[test]
fn test_confusion_matrix_fp_background() {
    // No GT annotations; one spurious DT
    let coco_gt = COCO::from_dataset(dataset(vec![img(1)], vec![cat(1, "cat")], vec![]));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![det(101, [0.0, 0.0, 50.0, 50.0], 0.9)],
    ));

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    let cm = ev.confusion_matrix(0.5, None, None);

    // num_cats=1, k=2: cat=0, background=1
    assert_eq!(cm.num_cats, 1);
    // FP: background row (1), cat col (0)
    assert_eq!(
        cm.get(1, 0),
        1,
        "spurious cat DT should count as FP (background row)"
    );
    // No FN
    assert_eq!(cm.get(0, 1), 0);
}

/// GT with no matching DT → lands in the background (FN) column.
#[test]
fn test_confusion_matrix_fn_missed() {
    // One GT, no DTs
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(vec![img(1)], vec![cat(1, "cat")], vec![]));

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    let cm = ev.confusion_matrix(0.5, None, None);

    // num_cats=1, k=2: cat=0, background=1
    // FN: cat row (0), background col (1)
    assert_eq!(
        cm.get(0, 1),
        1,
        "missed cat GT should count as FN (background col)"
    );
    // No FP
    assert_eq!(cm.get(1, 0), 0);
}

/// Same data: matches at iou_thr=0.5, misses at iou_thr=0.9.
///
/// GT=[0,0,100,100], DT=[50,0,50,100] → IoU = 0.5 exactly.
#[test]
fn test_confusion_matrix_iou_threshold() {
    // IoU between GT [0,0,100,100] and DT [50,0,50,100]:
    //   intersection = 50×100 = 5000
    //   union = 10000 + 5000 - 5000 = 10000
    //   IoU = 0.5
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 100.0, 100.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![det(101, [50.0, 0.0, 50.0, 100.0], 0.9)],
    ));

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    // At threshold 0.5: IoU (0.5) >= 0.5 → TP
    let cm_50 = ev.confusion_matrix(0.5, None, None);
    assert_eq!(cm_50.get(0, 0), 1, "should match at iou_thr=0.5");
    assert_eq!(cm_50.get(0, 1), 0);
    assert_eq!(cm_50.get(1, 0), 0);

    // At threshold 0.9: IoU (0.5) < 0.9 → FP + FN
    let cm_90 = ev.confusion_matrix(0.9, None, None);
    assert_eq!(cm_90.get(0, 0), 0, "should not match at iou_thr=0.9");
    assert_eq!(cm_90.get(0, 1), 1, "GT should be FN");
    assert_eq!(cm_90.get(1, 0), 1, "DT should be FP");
}

/// Low-score DT dropped by min_score → GT becomes a missed detection (FN).
#[test]
fn test_confusion_matrix_min_score() {
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![det(101, [0.0, 0.0, 50.0, 50.0], 0.3)],
    ));

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    // Without min_score: DT matches GT → TP
    let cm_no_filter = ev.confusion_matrix(0.5, None, None);
    assert_eq!(cm_no_filter.get(0, 0), 1, "should TP without score filter");
    assert_eq!(cm_no_filter.get(0, 1), 0, "no FN without score filter");

    // With min_score=0.5: DT (score=0.3) is dropped → GT missed → FN
    let cm_filtered = ev.confusion_matrix(0.5, None, Some(0.5));
    assert_eq!(
        cm_filtered.get(0, 0),
        0,
        "DT below min_score should be dropped"
    );
    assert_eq!(
        cm_filtered.get(0, 1),
        1,
        "GT should become FN when DT is filtered out"
    );
    assert_eq!(cm_filtered.get(1, 0), 0, "no FP when DT is filtered out");
}

/// Only the top-K detections by score are kept; lower-scoring DTs are excluded.
#[test]
fn test_confusion_matrix_max_det() {
    // 2 GTs: cat at [0,0,50,50], dog at [60,0,50,50]
    // 2 DTs: cat (score=0.9) and dog (score=0.5)
    // With max_det=1: only cat DT kept → cat GT matches, dog GT missed (FN)
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            ann(1, [0.0, 0.0, 50.0, 50.0]),
            ann(2, [60.0, 0.0, 50.0, 50.0]).in_cat(2),
        ],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            det(101, [0.0, 0.0, 50.0, 50.0], 0.9),
            det(102, [60.0, 0.0, 50.0, 50.0], 0.5).in_cat(2),
        ],
    ));

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    // max_det=2 (default): both DTs included → both TPs
    let cm_full = ev.confusion_matrix(0.5, Some(2), None);
    assert_eq!(cm_full.get(0, 0), 1, "cat TP with max_det=2");
    assert_eq!(cm_full.get(1, 1), 1, "dog TP with max_det=2");
    assert_eq!(cm_full.get(0, 2), 0, "no missed cat with max_det=2");
    assert_eq!(cm_full.get(1, 2), 0, "no missed dog with max_det=2");

    // max_det=1: only cat DT (score=0.9) kept; dog DT dropped
    let cm_1det = ev.confusion_matrix(0.5, Some(1), None);
    // num_cats=2, k=3: cat=0, dog=1, background=2
    assert_eq!(cm_1det.get(0, 0), 1, "cat GT matches cat DT → TP");
    assert_eq!(
        cm_1det.get(1, 2),
        1,
        "dog GT has no DT → FN (background col)"
    );
    assert_eq!(
        cm_1det.get(2, 1),
        0,
        "no spurious dog FP (DT was truncated)"
    );
}

// ============================================================
// tide_errors tests
// ============================================================

/// Run evaluate() and return tide_errors at the default thresholds.
fn run_tide(coco_gt: COCO, coco_dt: COCO) -> hotcoco::TideErrors {
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.tide_errors(0.5, 0.1).expect("tide_errors failed")
}

/// Test 1: all DTs are perfect TPs → all ΔAP = 0, all counts = 0.
#[test]
fn test_tide_all_correct() {
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![det(101, [0.0, 0.0, 50.0, 50.0], 0.9)],
    ));

    let te = run_tide(coco_gt, coco_dt);

    for (key, &val) in &te.delta_ap {
        assert!(
            val.abs() < 1e-6,
            "delta_ap[{key}] should be 0 for perfect detections, got {val}"
        );
    }
    for (key, &val) in &te.counts {
        assert_eq!(val, 0, "counts[{key}] should be 0 for perfect detections");
    }
    assert!(
        te.ap_base > 0.99,
        "ap_base should be ~1.0, got {}",
        te.ap_base
    );
}

/// Test 2: DT at right location but wrong class → Cls error with positive ΔAP.
#[test]
fn test_tide_cls_error() {
    // GT cat(1) at [0,0,50,50]; GT dog(2) at [60,0,50,50].
    // DT dog(2) at [0,0,50,50] (score=0.9): FP for dog category because no dog GT overlaps.
    // Cross-IoU with cat(1) GT = 1.0 ≥ pos_thr → Cls.
    // dog(2) has 1 GT so it contributes to ΔAP: fixing Cls converts FP→TP, AP goes 0→1 for dog.
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            ann(1, [0.0, 0.0, 50.0, 50.0]),            // cat GT
            ann(2, [60.0, 0.0, 50.0, 50.0]).in_cat(2), // dog GT (no overlap with DT)
        ],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![det(101, [0.0, 0.0, 50.0, 50.0], 0.9).in_cat(2)],
    ));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Cls"], 1, "should be 1 Cls error");
    assert_eq!(te.counts["Loc"], 0);
    assert_eq!(te.counts["Bkg"], 0);
    assert!(
        te.delta_ap["Cls"] > 0.0,
        "fixing Cls should improve AP (dog AP goes 0→1), got {}",
        te.delta_ap["Cls"]
    );
}

/// Test 3: DT right class, IoU = 0.3 (≥ bg_thr=0.1, < pos_thr=0.5) → Loc error.
#[test]
fn test_tide_loc_error() {
    // GT: [0,0,50,50] area=2500; DT: [25,0,50,50] area=2500
    // IoU = intersection/union = 25*50 / (50*50 + 50*50 - 25*50) = 1250/3750 = 1/3 ≈ 0.333
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![det(101, [25.0, 0.0, 50.0, 50.0], 0.9)],
    ));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Loc"], 1, "should be 1 Loc error");
    assert_eq!(te.counts["Cls"], 0);
    assert_eq!(te.counts["Bkg"], 0);
    assert!(
        te.delta_ap["Loc"] > 0.0,
        "fixing Loc should improve AP, got {}",
        te.delta_ap["Loc"]
    );
}

/// Test 4: DT wrong class AND poor localization (IoU = 0.3 with other-class GT) → Both error.
#[test]
fn test_tide_both_error() {
    // GT: cat(1) at [0,0,50,50]; DT: dog(2) at [25,0,50,50] → IoU≈0.333 with cat GT
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![det(101, [25.0, 0.0, 50.0, 50.0], 0.9).in_cat(2)],
    ));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Both"], 1, "should be 1 Both error");
    assert_eq!(te.counts["Cls"], 0, "not Cls because IoU < pos_thr");
    assert_eq!(te.counts["Loc"], 0, "not Loc because different class");
}

/// Test 5: two DTs for same GT → first is TP, second is Dupe.
#[test]
fn test_tide_dupe_error() {
    // GT: [0,0,50,50]; DT1(score=0.9): exact match (TP); DT2(score=0.7): same box (Dupe)
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![
            det(101, [0.0, 0.0, 50.0, 50.0], 0.9), // TP
            det(102, [0.0, 0.0, 50.0, 50.0], 0.7), // Dupe
        ],
    ));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Dupe"], 1, "second DT should be Dupe");
    assert_eq!(te.counts["Bkg"], 0);
    assert_eq!(te.counts["Cls"], 0);
}

/// Test 6: DT with IoU < bg_thr with all GTs → Bkg error.
#[test]
fn test_tide_bkg_error() {
    // GT: [0,0,10,10]; DT: [90,90,10,10] — no overlap at all → IoU=0 < bg_thr=0.1
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 10.0, 10.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![det(101, [90.0, 90.0, 10.0, 10.0], 0.9)],
    ));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Bkg"], 1, "far-away DT should be Bkg error");
    assert_eq!(te.counts["Loc"], 0);
    assert_eq!(te.counts["Cls"], 0);
}

/// Test 7: GT with no DT → Miss error, ΔAP["Miss"] > 0.
#[test]
fn test_tide_miss_error() {
    // GT: [0,0,50,50]; no DT at all
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(vec![img(1)], vec![cat(1, "cat")], vec![]));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Miss"], 1, "GT with no DT should be Miss");
    assert!(
        te.delta_ap["Miss"] > 0.0,
        "fixing Miss should improve AP, got {}",
        te.delta_ap["Miss"]
    );
    assert!(
        (te.delta_ap["Miss"] - 1.0).abs() < 1e-6,
        "injecting 1 perfect TP should give AP=1, delta=1.0, got {}",
        te.delta_ap["Miss"]
    );
}

/// Test 8: DT with same-class IoU ∈ [bg_thr, pos_thr] AND cross-class IoU ≥ pos_thr
/// → classified as Loc (tidecv priority: Loc > Cls, matching BoxError > ClassError).
#[test]
fn test_tide_priority_loc_over_cls() {
    // Setup: two images.
    // Image 1: GT cat(1) [0,0,50,50] matched by DT cat(1) score=0.95 (TP).
    // Image 2: GT cat(1) [0,0,30,30] (small), GT dog(2) [0,0,50,50].
    //   DT cat(1) at [0,0,50,50] (score=0.9):
    //     - same-class IoU with cat GT [0,0,30,30] = 900/2500 = 0.36 ∈ [bg_thr=0.1, pos_thr=0.5] → Loc
    //     - cross-class IoU with dog GT [0,0,50,50] = 1.0 ≥ pos_thr=0.5 → would be Cls if Loc lost
    // tidecv/hotcoco priority: Loc fires first → Loc wins.
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1), img(2)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            ann(1, [0.0, 0.0, 50.0, 50.0]),                     // img1 cat TP
            ann(2, [0.0, 0.0, 30.0, 30.0]).in_img(2),           // img2 cat (small)
            ann(3, [0.0, 0.0, 50.0, 50.0]).in_img(2).in_cat(2), // img2 dog
        ],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1), img(2)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            det(101, [0.0, 0.0, 50.0, 50.0], 0.95),          // img1 cat TP
            det(102, [0.0, 0.0, 50.0, 50.0], 0.9).in_img(2), // img2 cat FP: Loc wins over Cls
        ],
    ));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(
        te.counts["Loc"], 1,
        "Loc should win over Cls by priority (tidecv: BoxError first)"
    );
    assert_eq!(
        te.counts["Cls"], 0,
        "Cls should not fire when same-class IoU ≥ bg_thr triggers Loc first"
    );
}

/// Test 9: DT with IoU ≥ bg_thr to both correct-class and wrong-class GT → Loc wins.
#[test]
fn test_tide_priority_loc_over_both() {
    // DT cat(1) at [15,0,50,50]: overlaps both cat GT [0,0,50,50] (same-class) and dog GT [10,0,50,50].
    // Same-class IoU = intersection of [15,0,65,50] and [0,0,50,50] = [15,0,50,50] = 35*50=1750
    //   / union ([0,0,65,50] area=3250) = 1750/3250 ≈ 0.538 but < pos_thr = just right...
    // Let me use simpler numbers.
    // GT cat(1): [0,0,50,50]; GT dog(2): [60,0,50,50] (no overlap with DT).
    // DT cat(1): [25,0,50,50] → same-class IoU ≈ 0.333 ≥ bg_thr=0.1 → Loc
    //             cross-IoU with dog GT [60,0,50,50] = 0 (no overlap) → can't be Both
    // So Loc wins.
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            ann(1, [0.0, 0.0, 50.0, 50.0]),            // cat
            ann(2, [60.0, 0.0, 50.0, 50.0]).in_cat(2), // dog (no overlap with DT)
        ],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![det(101, [25.0, 0.0, 50.0, 50.0], 0.9)],
    ));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Loc"], 1, "same-class overlap ≥ bg_thr → Loc");
    assert_eq!(te.counts["Both"], 0);
}

/// Test 10: ΔAP["FP"] ≥ max of individual FP ΔAPs.
#[test]
fn test_tide_delta_ap_fp_ge_individuals() {
    // Multiple FP error types in one scene
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            ann(1, [0.0, 0.0, 50.0, 50.0]),
            ann(2, [60.0, 0.0, 50.0, 50.0]).in_cat(2),
        ],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat"), cat(2, "dog")],
        vec![
            // DT1 cat: matches cat GT (TP)
            det(101, [0.0, 0.0, 50.0, 50.0], 0.95),
            // DT2 dog: far away → Bkg
            det(102, [150.0, 150.0, 10.0, 10.0], 0.8).in_cat(2),
            // DT3 cat: wrong class vs dog GT at [60,0,50,50] → cross-IoU=1.0 → Cls
            det(103, [60.0, 0.0, 50.0, 50.0], 0.7),
        ],
    ));

    let te = run_tide(coco_gt, coco_dt);

    let fp_delta = te.delta_ap["FP"];
    let max_individual = te.delta_ap["Cls"]
        .max(te.delta_ap["Loc"])
        .max(te.delta_ap["Both"])
        .max(te.delta_ap["Dupe"])
        .max(te.delta_ap["Bkg"]);

    assert!(
        fp_delta >= max_individual - 1e-9,
        "ΔAP[FP]={fp_delta:.4} should be ≥ max individual={max_individual:.4}"
    );
}

/// Test 11: category with GTs but zero DTs → only Miss errors, no NaN in ΔAP.
#[test]
fn test_tide_empty_category() {
    // cat(1): 1 GT, 0 DTs → Miss=1, all ΔAP values finite
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "cat")],
        vec![ann(1, [0.0, 0.0, 50.0, 50.0])],
    ));
    let coco_dt = COCO::from_dataset(dataset(vec![img(1)], vec![cat(1, "cat")], vec![]));

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Miss"], 1, "one missed GT");
    assert_eq!(te.counts["Bkg"], 0);

    for (key, &val) in &te.delta_ap {
        assert!(
            val.is_finite(),
            "delta_ap[{key}] should be finite, got {val}"
        );
        assert!(
            val >= 0.0,
            "delta_ap[{key}] should be non-negative, got {val}"
        );
    }
    // Fixing Miss should recover to AP=1.0 from baseline AP=0.0 → delta=1.0
    assert!(
        (te.delta_ap["Miss"] - 1.0).abs() < 1e-6,
        "delta_ap[Miss] should be 1.0, got {}",
        te.delta_ap["Miss"]
    );
}

// ---------------------------------------------------------------------------
// COCO ↔ YOLO conversion tests
// ---------------------------------------------------------------------------

fn make_test_dataset_basic() -> Dataset {
    Dataset {
        info: None,
        images: vec![
            Image {
                id: 1,
                file_name: "img1.jpg".into(),
                width: 100,
                height: 200,
                ..Default::default()
            },
            Image {
                id: 2,
                file_name: "img2.jpg".into(),
                width: 400,
                height: 300,
                ..Default::default()
            },
        ],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 20.0, 30.0, 40.0]),
                area: Some(1200.0),
                ..Default::default()
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 2,
                bbox: Some([50.0, 60.0, 20.0, 25.0]),
                area: Some(500.0),
                ..Default::default()
            },
            Annotation {
                id: 3,
                image_id: 2,
                category_id: 1,
                bbox: Some([0.0, 0.0, 200.0, 150.0]),
                area: Some(30000.0),
                ..Default::default()
            },
        ],
        categories: vec![
            Category {
                id: 1,
                name: "cat".into(),
                ..Default::default()
            },
            Category {
                id: 2,
                name: "dog".into(),
                ..Default::default()
            },
        ],
        licenses: vec![],
    }
}

/// Shared round-trip assertion for the converter tests: after export → import,
/// every annotation must come back with the same geometry, keyed by
/// (image, category name).
///
/// `img_key` normalizes image identity across formats — YOLO and DOTA
/// reconstruct file names from bare stems, so they compare by [`file_stem_key`];
/// the XML/CSV formats keep the full name ([`file_name_key`]). `geom` picks the
/// compared array (bbox, obb) and `tol` is one absolute tolerance per
/// component, because precision is format- and component-specific: VOC rounds
/// to integer pixels, CVAT prints 2 decimals, DOTA prints corners to 1 decimal
/// yet round-trips angles much tighter.
///
/// Assumes each (image, category) pair holds at most one annotation, which is
/// what the shared fixtures provide — with duplicates the sorted pairing would
/// be ambiguous.
fn assert_geometry_round_trip<const N: usize>(
    original: &Dataset,
    recovered: &Dataset,
    img_key: fn(&Image) -> String,
    geom: fn(&Annotation) -> [f64; N],
    tol: [f64; N],
) {
    assert_eq!(recovered.images.len(), original.images.len(), "image count");
    assert_eq!(
        recovered.annotations.len(),
        original.annotations.len(),
        "annotation count"
    );
    assert_eq!(
        recovered.categories.len(),
        original.categories.len(),
        "category count"
    );

    let keyed = |ds: &Dataset| -> Vec<(String, String, [f64; N])> {
        let cat_name: HashMap<u64, &str> = ds
            .categories
            .iter()
            .map(|c| (c.id, c.name.as_str()))
            .collect();
        let img_name: HashMap<u64, String> = ds.images.iter().map(|i| (i.id, img_key(i))).collect();
        let mut rows: Vec<(String, String, [f64; N])> = ds
            .annotations
            .iter()
            .map(|a| {
                (
                    img_name[&a.image_id].clone(),
                    cat_name[&a.category_id].to_string(),
                    geom(a),
                )
            })
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        rows
    };

    for ((o_img, o_cat, o_geo), (r_img, r_cat, r_geo)) in
        keyed(original).iter().zip(keyed(recovered).iter())
    {
        assert_eq!(o_img, r_img, "image mismatch");
        assert_eq!(o_cat, r_cat, "category mismatch");
        for i in 0..N {
            assert!(
                (o_geo[i] - r_geo[i]).abs() <= tol[i],
                "geom[{i}] mismatch for {o_img}/{o_cat}: orig={} recovered={}",
                o_geo[i],
                r_geo[i]
            );
        }
    }
}

/// Image key for formats that reconstruct file names from bare stems.
fn file_stem_key(img: &Image) -> String {
    std::path::Path::new(&img.file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(img.file_name.as_str())
        .to_string()
}

/// Image key for formats that preserve the full file name.
fn file_name_key(img: &Image) -> String {
    img.file_name.clone()
}

fn bbox_of(a: &Annotation) -> [f64; 4] {
    a.bbox.expect("annotation should carry a bbox")
}

fn obb_of(a: &Annotation) -> [f64; 5] {
    a.obb.expect("annotation should carry an obb")
}

#[test]
fn test_coco_to_yolo_basic() {
    let dataset = make_test_dataset_basic();
    let dir = tempfile::tempdir().expect("tempdir");
    let stats = coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    assert_eq!(stats.images, 2);
    assert_eq!(stats.annotations, 3);
    assert_eq!(stats.skipped_crowd, 0);
    assert_eq!(stats.skipped_no_bbox, 0);

    // data.yaml: assert the category names by parsing with the crate's own
    // reader rather than string-matching one legal spelling of the YAML —
    // `names:` may be written flow or block style and both must read back.
    let yaml = std::fs::read_to_string(dir.path().join("data.yaml")).expect("data.yaml");
    assert!(yaml.contains("nc: 2"), "yaml: {yaml}");
    let dims: HashMap<String, (u32, u32)> = [
        ("img1".to_string(), (100u32, 200u32)),
        ("img2".to_string(), (400u32, 300u32)),
    ]
    .into_iter()
    .collect();
    let parsed = yolo_to_coco(dir.path(), &dims).expect("re-import of our own export");
    let names: Vec<&str> = parsed.categories.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["cat", "dog"], "yaml was: {yaml}");

    // img1.txt: 2 annotations
    let txt1 = std::fs::read_to_string(dir.path().join("img1.txt")).expect("img1.txt");
    let lines1: Vec<&str> = txt1.lines().collect();
    assert_eq!(lines1.len(), 2, "img1.txt should have 2 lines");

    // img2.txt: 1 annotation
    let txt2 = std::fs::read_to_string(dir.path().join("img2.txt")).expect("img2.txt");
    let lines2: Vec<&str> = txt2.lines().collect();
    assert_eq!(lines2.len(), 1, "img2.txt should have 1 line");

    // Spot-check: ann id=1, bbox=[10,20,30,40], img width=100, height=200
    // cx = (10+15)/100 = 0.25, cy = (20+20)/200 = 0.2, w=0.3, h=0.2, class=0
    let first_line = lines1[0];
    let parts: Vec<f64> = first_line
        .split_whitespace()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    assert!((parts[0] - 0.25).abs() < 1e-5, "cx mismatch: {}", parts[0]);
    assert!((parts[1] - 0.2).abs() < 1e-5, "cy mismatch: {}", parts[1]);
    assert!((parts[2] - 0.3).abs() < 1e-5, "nw mismatch: {}", parts[2]);
    assert!((parts[3] - 0.2).abs() < 1e-5, "nh mismatch: {}", parts[3]);
}

#[test]
fn test_coco_to_yolo_category_remapping() {
    // COCO cat IDs {1, 3, 7} → YOLO class IDs {0, 1, 2} after sorting by ID
    let dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img.jpg".into(),
            width: 200,
            height: 200,
            ..Default::default()
        }],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 7,
                bbox: Some([10.0, 10.0, 40.0, 40.0]),
                area: Some(1600.0),
                ..Default::default()
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 3,
                bbox: Some([60.0, 60.0, 20.0, 20.0]),
                area: Some(400.0),
                ..Default::default()
            },
        ],
        // Unsorted in dataset; coco_to_yolo must sort by ID
        categories: vec![
            Category {
                id: 7,
                name: "bird".into(),
                ..Default::default()
            },
            Category {
                id: 1,
                name: "cat".into(),
                ..Default::default()
            },
            Category {
                id: 3,
                name: "dog".into(),
                ..Default::default()
            },
        ],
        licenses: vec![],
    };

    let dir = tempfile::tempdir().expect("tempdir");
    coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    // Sorted order: cat(1), dog(3), bird(7) — asserted through the crate's own
    // data.yaml reader, so the assertion survives a formatting change.
    let dims: HashMap<String, (u32, u32)> = [("img".to_string(), (200u32, 200u32))]
        .into_iter()
        .collect();
    let parsed = yolo_to_coco(dir.path(), &dims).expect("re-import of our own export");
    let names: Vec<&str> = parsed.categories.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["cat", "dog", "bird"]);

    let txt = std::fs::read_to_string(dir.path().join("img.txt")).expect("img.txt");
    let lines: Vec<&str> = txt.lines().collect();
    assert_eq!(lines.len(), 2);

    // category_id=7 → class_idx=2 (sorted: cat→0, dog→1, bird→2)
    let class0: usize = lines[0].split_whitespace().next().unwrap().parse().unwrap();
    assert_eq!(
        class0, 2,
        "cat_id=7 should map to class_idx=2, got {class0}"
    );

    // category_id=3 → class_idx=1
    let class1: usize = lines[1].split_whitespace().next().unwrap().parse().unwrap();
    assert_eq!(
        class1, 1,
        "cat_id=3 should map to class_idx=1, got {class1}"
    );
}

#[test]
fn test_coco_to_yolo_crowd_skipped() {
    let dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img.jpg".into(),
            width: 100,
            height: 100,
            ..Default::default()
        }],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                iscrowd: true, // should be skipped
                ..Default::default()
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 20.0, 20.0]),
                area: Some(400.0),
                ..Default::default()
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "thing".into(),
            ..Default::default()
        }],
        licenses: vec![],
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let stats = coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    assert_eq!(stats.skipped_crowd, 1);
    assert_eq!(stats.annotations, 1);

    let txt = std::fs::read_to_string(dir.path().join("img.txt")).expect("img.txt");
    assert_eq!(txt.lines().count(), 1, "only one non-crowd annotation");
}

#[test]
fn test_coco_to_yolo_missing_bbox() {
    let dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img.jpg".into(),
            width: 100,
            height: 100,
            ..Default::default()
        }],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: None, // no bbox — should be skipped
                area: Some(400.0),
                ..Default::default()
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 20.0, 20.0]),
                area: Some(400.0),
                ..Default::default()
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "thing".into(),
            ..Default::default()
        }],
        licenses: vec![],
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let stats = coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    assert_eq!(stats.skipped_no_bbox, 1);
    assert_eq!(stats.annotations, 1);
}

#[test]
fn test_coco_to_yolo_empty_image() {
    // Image with no annotations → empty .txt must still be created
    let dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "empty.jpg".into(),
            width: 640,
            height: 480,
            ..Default::default()
        }],
        annotations: vec![],
        categories: vec![Category {
            id: 1,
            name: "thing".into(),
            ..Default::default()
        }],
        licenses: vec![],
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let stats = coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    assert_eq!(stats.images, 1);
    assert_eq!(stats.annotations, 0);

    let txt_path = dir.path().join("empty.txt");
    assert!(txt_path.exists(), "empty.txt should be created");
    let content = std::fs::read_to_string(&txt_path).expect("empty.txt");
    assert!(content.is_empty(), "empty.txt should have no content");
}

#[test]
fn test_yolo_to_coco_basic() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Write data.yaml
    std::fs::write(dir.path().join("data.yaml"), "nc: 2\nnames: [cat, dog]\n")
        .expect("write data.yaml");

    // Write label files
    std::fs::write(
        dir.path().join("img1.txt"),
        "0 0.250000 0.200000 0.300000 0.200000\n1 0.700000 0.725000 0.200000 0.250000\n",
    )
    .expect("write img1.txt");
    std::fs::write(
        dir.path().join("img2.txt"),
        "0 0.250000 0.250000 0.500000 0.500000\n",
    )
    .expect("write img2.txt");

    let dims: HashMap<String, (u32, u32)> = [
        ("img1".to_string(), (100u32, 200u32)),
        ("img2".to_string(), (400u32, 300u32)),
    ]
    .into_iter()
    .collect();

    let dataset = yolo_to_coco(dir.path(), &dims).expect("yolo_to_coco");

    assert_eq!(dataset.images.len(), 2);
    assert_eq!(dataset.annotations.len(), 3);
    assert_eq!(dataset.categories.len(), 2);

    // Categories: id=1→cat, id=2→dog
    assert_eq!(dataset.categories[0].id, 1);
    assert_eq!(dataset.categories[0].name, "cat");
    assert_eq!(dataset.categories[1].id, 2);
    assert_eq!(dataset.categories[1].name, "dog");

    // Find the img1 image and check dims
    let img1 = dataset
        .images
        .iter()
        .find(|i| i.file_name == "img1")
        .unwrap();
    assert_eq!(img1.width, 100);
    assert_eq!(img1.height, 200);

    // Check bbox reconstruction for first annotation of img1:
    // YOLO: class=0, cx=0.25, cy=0.2, w=0.3, h=0.2; image 100×200
    // COCO: x=(0.25-0.15)*100=10, y=(0.2-0.2)*200=0... wait let me recalculate
    // cx=0.25 → x = (0.25 - 0.3/2)*100 = (0.25-0.15)*100 = 10
    // cy=0.20 → y = (0.20 - 0.2/2)*200 = (0.20-0.10)*200 = 20
    // bw = 0.3*100 = 30, bh = 0.2*200 = 40
    let ann = dataset
        .annotations
        .iter()
        .find(|a| a.image_id == img1.id && a.category_id == 1)
        .unwrap();
    let bbox = ann.bbox.unwrap();
    assert!((bbox[0] - 10.0).abs() < 1e-4, "x: {}", bbox[0]);
    assert!((bbox[1] - 20.0).abs() < 1e-4, "y: {}", bbox[1]);
    assert!((bbox[2] - 30.0).abs() < 1e-4, "w: {}", bbox[2]);
    assert!((bbox[3] - 40.0).abs() < 1e-4, "h: {}", bbox[3]);
}

#[test]
fn test_yolo_round_trip() {
    let original = make_test_dataset_basic();
    let dir = tempfile::tempdir().expect("tempdir");

    // COCO → YOLO
    coco_to_yolo(&original, dir.path()).expect("coco_to_yolo");

    // Build image_dims from original dataset for the round-trip
    let dims: HashMap<String, (u32, u32)> = original
        .images
        .iter()
        .map(|img| {
            let stem = std::path::Path::new(&img.file_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(img.file_name.as_str())
                .to_string();
            (stem, (img.width, img.height))
        })
        .collect();

    // YOLO → COCO
    let recovered = yolo_to_coco(dir.path(), &dims).expect("yolo_to_coco");

    // YOLO reconstructs file names from bare stems; 6-decimal normalized
    // coordinates round-trip within 1e-4 of a pixel here.
    assert_geometry_round_trip(&original, &recovered, file_stem_key, bbox_of, [1e-4; 4]);
}

// ── VOC conversion tests ─────────────────────────────────────────────────────

#[test]
fn test_coco_to_voc_basic() {
    let dataset = make_test_dataset_basic();
    let dir = tempfile::tempdir().expect("tempdir");
    let stats = coco_to_voc(&dataset, dir.path()).expect("coco_to_voc");

    assert_eq!(stats.images, 2);
    assert_eq!(stats.annotations, 3);
    assert_eq!(stats.crowd_as_difficult, 0);
    assert_eq!(stats.skipped_no_bbox, 0);

    // Annotations/ directory should exist
    let ann_dir = dir.path().join("Annotations");
    assert!(ann_dir.is_dir(), "Annotations/ directory should exist");

    // labels.txt should list categories sorted by COCO ID
    let labels = std::fs::read_to_string(dir.path().join("labels.txt")).expect("labels.txt");
    assert_eq!(labels.trim(), "cat\ndog");

    // img1.xml should exist with 2 objects
    let xml1 = std::fs::read_to_string(ann_dir.join("img1.xml")).expect("img1.xml");
    assert!(xml1.contains("<filename>img1.jpg</filename>"), "filename");
    assert!(xml1.contains("<width>100</width>"), "width");
    assert!(xml1.contains("<height>200</height>"), "height");

    // Spot-check first annotation: bbox=[10,20,30,40] under the VOC 1-based
    // inclusive convention → xmin=11, ymin=21, xmax=40, ymax=60
    assert!(xml1.contains("<xmin>11</xmin>"), "xmin");
    assert!(xml1.contains("<ymin>21</ymin>"), "ymin");
    assert!(xml1.contains("<xmax>40</xmax>"), "xmax");
    assert!(xml1.contains("<ymax>60</ymax>"), "ymax");
    assert!(xml1.contains("<name>cat</name>"), "cat object");
    assert!(xml1.contains("<name>dog</name>"), "dog object");

    // img2.xml should have 1 object
    let xml2 = std::fs::read_to_string(ann_dir.join("img2.xml")).expect("img2.xml");
    assert!(xml2.contains("<name>cat</name>"), "cat object in img2");
    // bbox=[0,0,200,150] → xmin=0, ymin=0, xmax=200, ymax=150
    assert!(xml2.contains("<xmax>200</xmax>"), "xmax img2");
    assert!(xml2.contains("<ymax>150</ymax>"), "ymax img2");
}

/// A VOC2012 `<part>` describes a sub-region of its object — a person's head,
/// hand, or foot. Its `<name>` and `<bndbox>` belong to the part, and neither may
/// reach the object being built.
///
/// The `<part>` always follows the object's own `<bndbox>` in VOC2012, so a
/// parser that skips `<name>` but not `<bndbox>` reports the *last part's* box as
/// the person's — silently, on the single most common class in the dataset.
#[test]
fn test_voc_part_elements_do_not_overwrite_the_object() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ann_dir = dir.path().join("Annotations");
    std::fs::create_dir_all(&ann_dir).expect("mkdir");

    let xml = r"<annotation>
  <filename>person.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>person</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>50</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
    <part>
      <name>head</name>
      <bndbox>
        <xmin>150</xmin>
        <ymin>60</ymin>
        <xmax>220</xmax>
        <ymax>130</ymax>
      </bndbox>
    </part>
    <part>
      <name>foot</name>
      <bndbox>
        <xmin>110</xmin>
        <ymin>360</ymin>
        <xmax>180</xmax>
        <ymax>400</ymax>
      </bndbox>
    </part>
  </object>
</annotation>";
    std::fs::write(ann_dir.join("person.xml"), xml).expect("write xml");

    let dataset = voc_to_coco(dir.path()).expect("voc_to_coco");

    // Parts are not objects.
    assert_eq!(
        dataset.categories.len(),
        1,
        "only `person` is an object; `head` and `foot` are parts. Got: {:?}",
        dataset
            .categories
            .iter()
            .map(|c| &c.name)
            .collect::<Vec<_>>()
    );
    assert_eq!(dataset.categories[0].name, "person");
    assert_eq!(dataset.annotations.len(), 1);

    // xmin=100, ymin=50, xmax=300, ymax=400 → 1-based inclusive VOC coords
    // become [99, 49, 201, 351].
    assert_eq!(
        dataset.annotations[0].bbox,
        Some([99.0, 49.0, 201.0, 351.0]),
        "the person's own box, not the last part's"
    );
}

#[test]
fn test_voc_to_coco_basic() {
    // Write a known VOC XML and parse it back
    let dir = tempfile::tempdir().expect("tempdir");
    let ann_dir = dir.path().join("Annotations");
    std::fs::create_dir_all(&ann_dir).expect("mkdir");

    let xml = r"<annotation>
  <folder>Annotations</folder>
  <filename>test.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>person</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>100</xmin>
      <ymin>50</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
  <object>
    <name>car</name>
    <pose>Left</pose>
    <truncated>1</truncated>
    <difficult>1</difficult>
    <bndbox>
      <xmin>400</xmin>
      <ymin>200</ymin>
      <xmax>600</xmax>
      <ymax>450</ymax>
    </bndbox>
  </object>
</annotation>";
    std::fs::write(ann_dir.join("test.xml"), xml).expect("write xml");

    let dataset = voc_to_coco(dir.path()).expect("voc_to_coco");

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "test.jpg");
    assert_eq!(dataset.images[0].width, 640);
    assert_eq!(dataset.images[0].height, 480);

    assert_eq!(dataset.annotations.len(), 2);
    // Categories sorted alphabetically: car=1, person=2
    assert_eq!(dataset.categories.len(), 2);
    assert_eq!(dataset.categories[0].name, "car");
    assert_eq!(dataset.categories[1].name, "person");

    // person: xmin=100, ymin=50, xmax=300, ymax=400 → 1-based inclusive VOC
    // coords become bbox=[99, 49, 201, 351]
    let person_ann = dataset
        .annotations
        .iter()
        .find(|a| {
            a.category_id
                == dataset
                    .categories
                    .iter()
                    .find(|c| c.name == "person")
                    .unwrap()
                    .id
        })
        .expect("person annotation");
    let bbox = person_ann.bbox.unwrap();
    assert_eq!(bbox, [99.0, 49.0, 201.0, 351.0]);

    // car: xmin=400, ymin=200, xmax=600, ymax=450 → bbox=[399, 199, 201, 251]
    let car_ann = dataset
        .annotations
        .iter()
        .find(|a| {
            a.category_id
                == dataset
                    .categories
                    .iter()
                    .find(|c| c.name == "car")
                    .unwrap()
                    .id
        })
        .expect("car annotation");
    let bbox = car_ann.bbox.unwrap();
    assert_eq!(bbox, [399.0, 199.0, 201.0, 251.0]);

    // <difficult> imports to iscrowd, the inverse of the export mapping
    assert!(!person_ann.iscrowd, "difficult=0 → iscrowd=false");
    assert!(car_ann.iscrowd, "difficult=1 → iscrowd=true");
}

#[test]
fn test_voc_round_trip() {
    let original = make_test_dataset_basic();
    let dir = tempfile::tempdir().expect("tempdir");

    // COCO → VOC
    coco_to_voc(&original, dir.path()).expect("coco_to_voc");

    // VOC → COCO. VOC uses integer coords, so the tolerance is 1.0 pixel.
    let recovered = voc_to_coco(dir.path()).expect("voc_to_coco");
    assert_geometry_round_trip(&original, &recovered, file_name_key, bbox_of, [1.0; 4]);
}

#[test]
fn test_coco_to_voc_crowd_as_difficult() {
    let dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img.jpg".into(),
            width: 100,
            height: 100,
            ..Default::default()
        }],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 20.0, 30.0, 40.0]),
                area: Some(1200.0),
                iscrowd: true,
                ..Default::default()
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 60.0, 10.0, 10.0]),
                area: Some(100.0),
                ..Default::default()
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "thing".into(),
            ..Default::default()
        }],
        licenses: vec![],
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let stats = coco_to_voc(&dataset, dir.path()).expect("coco_to_voc");

    assert_eq!(stats.annotations, 2, "both annotations should be written");
    assert_eq!(stats.crowd_as_difficult, 1, "one crowd annotation");

    let xml = std::fs::read_to_string(dir.path().join("Annotations/img.xml")).expect("img.xml");
    // The crowd annotation should have <difficult>1</difficult>
    assert!(
        xml.contains("<difficult>1</difficult>"),
        "crowd → difficult=1"
    );
    assert!(
        xml.contains("<difficult>0</difficult>"),
        "non-crowd → difficult=0"
    );
}

#[test]
fn test_voc_labels_txt_ordering() {
    // When labels.txt is present, it should determine category ordering
    let dir = tempfile::tempdir().expect("tempdir");
    let ann_dir = dir.path().join("Annotations");
    std::fs::create_dir_all(&ann_dir).expect("mkdir");

    // Write labels.txt with non-alphabetical order
    std::fs::write(dir.path().join("labels.txt"), "zebra\napple\n").expect("labels.txt");

    let xml = r"<annotation>
  <filename>img.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  <object>
    <name>apple</name>
    <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
  </object>
  <object>
    <name>zebra</name>
    <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>100</xmax><ymax>100</ymax></bndbox>
  </object>
</annotation>";
    std::fs::write(ann_dir.join("img.xml"), xml).expect("write xml");

    let dataset = voc_to_coco(dir.path()).expect("voc_to_coco");

    // labels.txt ordering: zebra=1, apple=2
    assert_eq!(dataset.categories[0].name, "zebra");
    assert_eq!(dataset.categories[0].id, 1);
    assert_eq!(dataset.categories[1].name, "apple");
    assert_eq!(dataset.categories[1].id, 2);
}

// ── CVAT conversion tests ────────────────────────────────────────────────────

#[test]
fn test_coco_to_cvat_basic() {
    let dataset = make_test_dataset_basic();
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("annotations.xml");
    let stats = coco_to_cvat(&dataset, &out).expect("coco_to_cvat");

    assert_eq!(stats.images, 2);
    assert_eq!(stats.boxes, 3);
    assert_eq!(stats.polygons, 0);
    assert_eq!(stats.skipped_no_geometry, 0);

    let xml = std::fs::read_to_string(&out).expect("read xml");
    assert!(xml.contains("<version>1.1</version>"), "version");
    assert!(xml.contains("<name>cat</name>"), "cat label");
    assert!(xml.contains("<name>dog</name>"), "dog label");
    assert!(xml.contains("name=\"img1.jpg\""), "image name");
    // ann id=1: bbox=[10,20,30,40] → xtl=10, ytl=20, xbr=40, ybr=60
    assert!(xml.contains("xtl=\"10.00\""), "xtl");
    assert!(xml.contains("ytl=\"20.00\""), "ytl");
    assert!(xml.contains("xbr=\"40.00\""), "xbr");
    assert!(xml.contains("ybr=\"60.00\""), "ybr");
}

#[test]
fn test_cvat_to_coco_basic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("annotations.xml");
    let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>person</name></label>
        <label><name>car</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="test.jpg" width="640" height="480">
    <box label="person" xtl="100" ytl="50" xbr="300" ybr="400" occluded="0"/>
    <box label="car" xtl="400" ytl="200" xbr="600" ybr="450" occluded="0"/>
  </image>
</annotations>"#;
    std::fs::write(&xml_path, xml).expect("write xml");

    let (dataset, stats) = cvat_to_coco(&xml_path).expect("cvat_to_coco");
    assert_eq!(stats.images, 1);
    assert_eq!(stats.boxes, 2);

    assert_eq!(dataset.images.len(), 1);
    assert_eq!(dataset.images[0].file_name, "test.jpg");
    assert_eq!(dataset.images[0].width, 640);
    assert_eq!(dataset.images[0].height, 480);

    assert_eq!(dataset.annotations.len(), 2);
    // Categories from meta: person=1, car=2 (meta ordering preserved)
    assert_eq!(dataset.categories[0].name, "person");
    assert_eq!(dataset.categories[1].name, "car");

    // person: xtl=100, ytl=50, xbr=300, ybr=400 → bbox=[100, 50, 200, 350]
    let person_cat = dataset
        .categories
        .iter()
        .find(|c| c.name == "person")
        .unwrap();
    let person_ann = dataset
        .annotations
        .iter()
        .find(|a| a.category_id == person_cat.id)
        .expect("person annotation");
    let bbox = person_ann.bbox.unwrap();
    assert_eq!(bbox, [100.0, 50.0, 200.0, 350.0]);
}

#[test]
fn test_cvat_round_trip_boxes() {
    let original = make_test_dataset_basic();
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("output.xml");

    coco_to_cvat(&original, &xml_path).expect("coco_to_cvat");
    let (recovered, _stats) = cvat_to_coco(&xml_path).expect("cvat_to_coco");

    // CVAT prints float coords to 2 decimals — round-trip within 0.01.
    assert_geometry_round_trip(&original, &recovered, file_name_key, bbox_of, [0.01; 4]);
}

#[test]
fn test_cvat_polygons() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("poly.xml");
    let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><labels><label><name>shape</name></label></labels></task></meta>
  <image id="0" name="img.jpg" width="100" height="100">
    <polygon label="shape" points="10.0,20.0;50.0,20.0;50.0,80.0;10.0,80.0" occluded="0"/>
  </image>
</annotations>"#;
    std::fs::write(&xml_path, xml).expect("write xml");

    let (dataset, _stats) = cvat_to_coco(&xml_path).expect("cvat_to_coco");
    assert_eq!(dataset.annotations.len(), 1);

    let ann = &dataset.annotations[0];
    // bbox should be [10, 20, 40, 60]
    let bbox = ann.bbox.unwrap();
    assert!((bbox[0] - 10.0).abs() < 1e-6, "x");
    assert!((bbox[1] - 20.0).abs() < 1e-6, "y");
    assert!((bbox[2] - 40.0).abs() < 1e-6, "w");
    assert!((bbox[3] - 60.0).abs() < 1e-6, "h");

    // area via shoelace: 40 * 60 = 2400
    assert!((ann.area.unwrap() - 2400.0).abs() < 1e-6, "area");

    // segmentation should be a polygon
    match &ann.segmentation {
        Some(hotcoco::types::Segmentation::Polygon(polys)) => {
            assert_eq!(polys.len(), 1);
            assert_eq!(
                polys[0],
                vec![10.0, 20.0, 50.0, 20.0, 50.0, 80.0, 10.0, 80.0]
            );
        }
        other => panic!("expected Polygon segmentation, got: {other:?}"),
    }
}

#[test]
fn test_cvat_skips_unsupported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let xml_path = dir.path().join("mixed.xml");
    let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><labels><label><name>thing</name></label></labels></task></meta>
  <image id="0" name="img.jpg" width="100" height="100">
    <box label="thing" xtl="10" ytl="10" xbr="50" ybr="50" occluded="0"/>
    <polyline label="thing" points="10,10;50,50" occluded="0"/>
    <points label="thing" points="25,25" occluded="0"/>
  </image>
</annotations>"#;
    std::fs::write(&xml_path, xml).expect("write xml");

    let (dataset, stats) = cvat_to_coco(&xml_path).expect("cvat_to_coco");
    // Only the box should be imported; polyline and points are unsupported
    // shape kinds — skipped and counted.
    assert_eq!(dataset.annotations.len(), 1);
    assert_eq!(stats.skipped_unsupported, 2);
    assert_eq!(
        dataset.annotations[0].bbox.unwrap(),
        [10.0, 10.0, 40.0, 40.0]
    );
}

// ── f_scores tests ────────────────────────────────────────────────────────────

fn make_perfect_eval() -> COCOeval {
    // One image, one GT bbox, one perfectly matching DT.
    let bbox = [10.0, 10.0, 50.0, 50.0];
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![ann(1, bbox)],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![det(1, bbox, 1.0)],
    ));
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev
}

#[test]
fn test_f_scores_empty_before_accumulate() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("load DT");
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    assert!(ev.f_scores(1.0).is_empty());
}

#[test]
fn test_f_scores_keys_and_range() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("load DT");
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();

    let f1 = ev.f_scores(1.0);
    assert_eq!(f1.len(), 3);
    assert!(f1.contains_key("F1") && f1.contains_key("F1_50") && f1.contains_key("F1_75"));
    for (k, v) in &f1 {
        assert!((0.0..=1.0).contains(v), "{k} = {v} outside [0, 1]");
    }

    // beta variant gets correct key prefix
    let fb = ev.f_scores(0.5);
    assert!(fb.contains_key("F0.5") && fb.contains_key("F0.5_50") && fb.contains_key("F0.5_75"));
}

#[test]
fn test_f_scores_perfect_detection() {
    let scores = make_perfect_eval().f_scores(1.0);
    assert!((scores["F1"] - 1.0).abs() < 1e-9, "F1={}", scores["F1"]);
    assert!(
        (scores["F1_50"] - 1.0).abs() < 1e-9,
        "F1_50={}",
        scores["F1_50"]
    );
}

// ─── results export ──────────────────────────────────────────────────

#[test]
fn test_results_returns_metrics() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.run();

    // Without per-class
    let results = ev
        .results(false)
        .expect("results() should succeed after run()");
    assert_eq!(results.metrics.len(), 12);
    assert!(results.per_class.is_none());
    assert_eq!(results.params.iou_thresholds.len(), 10);
    assert_eq!(results.params.max_dets, vec![1, 10, 100]);
    assert_eq!(results.params.eval_mode, "coco");

    // With per-class
    let results = ev.results(true).expect("results() should succeed");
    assert!(results.per_class.is_some());
    assert!(!results.per_class.as_ref().unwrap().is_empty());
}

#[test]
fn test_results_errors_before_summarize() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    assert!(ev.results(false).is_err());
}

#[test]
fn test_results_save_roundtrip() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.run();

    let results = ev.results(true).unwrap();
    let json = results.to_json().unwrap();

    // Verify it's valid JSON with expected structure
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed["params"]["iou_type"].is_string());
    assert!(parsed["metrics"]["AP"].is_number());
    assert!(parsed["per_class"].is_object());

    // Save to file and verify roundtrip
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("results.json");
    results.save(&path).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    let file_parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
    assert_eq!(parsed, file_parsed);
}

#[test]
fn test_healthcheck_structural_errors() {
    let path = fixtures_dir().join("healthcheck_bad.json");
    let dataset: hotcoco::Dataset =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let report = quality::healthcheck(&dataset);

    let codes: Vec<&str> = report.errors.iter().map(|f| f.code).collect();
    assert!(
        codes.contains(&"duplicate_image_id"),
        "should detect duplicate image IDs"
    );
    assert!(
        codes.contains(&"duplicate_ann_id"),
        "should detect duplicate annotation IDs"
    );
    assert!(
        codes.contains(&"orphan_image_id"),
        "should detect orphan image_id in annotations"
    );
    assert!(
        codes.contains(&"orphan_category_id"),
        "should detect orphan category_id in annotations"
    );
    assert!(
        codes.contains(&"zero_dimensions"),
        "should detect zero height/width on images"
    );
}

#[test]
fn test_healthcheck_clean_dataset() {
    let path = fixtures_dir().join("gt.json");
    let dataset: hotcoco::Dataset =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let report = quality::healthcheck(&dataset);

    assert!(
        report.errors.is_empty(),
        "clean dataset should have no errors: {:?}",
        report.errors
    );
}

#[test]
fn test_healthcheck_quality_warnings() {
    let path = fixtures_dir().join("healthcheck_quality.json");
    let dataset: hotcoco::Dataset =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let report = quality::healthcheck(&dataset);

    let codes: Vec<&str> = report.warnings.iter().map(|f| f.code).collect();
    assert!(
        codes.contains(&"degenerate_bbox"),
        "should detect zero-width bbox"
    );
    assert!(
        codes.contains(&"bbox_out_of_bounds"),
        "should detect bbox extending outside image"
    );
    assert!(
        codes.contains(&"extreme_aspect_ratio"),
        "should detect extreme aspect ratio"
    );
    assert!(
        codes.contains(&"near_duplicate"),
        "should detect near-duplicate overlapping annotations"
    );
}

#[test]
fn test_healthcheck_summary() {
    let path = fixtures_dir().join("gt.json");
    let dataset: hotcoco::Dataset =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let report = quality::healthcheck(&dataset);

    assert_eq!(report.summary.num_images, 3);
    assert_eq!(report.summary.num_annotations, 5);
    assert_eq!(report.summary.num_categories, 2);
    assert_eq!(report.summary.images_without_annotations, 0);
    // cat: 3 annotations, dog: 2 annotations
    assert_eq!(report.summary.category_counts[0].0, "cat");
    assert_eq!(report.summary.category_counts[0].1, 3);
    assert_eq!(report.summary.category_counts[1].0, "dog");
    assert_eq!(report.summary.category_counts[1].1, 2);
    // imbalance: 3/2 = 1.5
    assert!((report.summary.imbalance_ratio - 1.5).abs() < 1e-9);
}

#[test]
fn test_healthcheck_compatibility() {
    let gt: hotcoco::Dataset = serde_json::from_str(
        r#"{
        "images": [
            {"id": 1, "file_name": "a.jpg", "height": 100, "width": 100}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0}
        ],
        "categories": [{"id": 1, "name": "cat"}]
    }"#,
    )
    .unwrap();

    let dt: hotcoco::Dataset = serde_json::from_str(
        r#"{
        "images": [
            {"id": 1, "file_name": "a.jpg", "height": 100, "width": 100}
        ],
        "annotations": [
            {"id": 1, "image_id": 999, "category_id": 1, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0, "score": 0.9},
            {"id": 2, "image_id": 1, "category_id": 999, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0, "score": 0.8},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0, "score": 1.5}
        ],
        "categories": [{"id": 1, "name": "cat"}]
    }"#,
    )
    .unwrap();

    let report = quality::healthcheck_compatibility(&gt, &dt);

    let codes: Vec<&str> = report
        .errors
        .iter()
        .map(|f| f.code)
        .chain(report.warnings.iter().map(|f| f.code))
        .collect();
    assert!(
        codes.contains(&"dt_orphan_image_id"),
        "should detect DT referencing unknown image ID"
    );
    assert!(
        codes.contains(&"dt_orphan_category_id"),
        "should detect DT referencing unknown category ID"
    );
    assert!(
        codes.contains(&"dt_missing_score"),
        "should detect DT missing score"
    );
    assert!(
        codes.contains(&"dt_score_out_of_range"),
        "should detect DT with score > 1.0"
    );
}

#[test]
fn test_accumulate_unchanged_after_refactor() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).unwrap();
    let coco_dt = coco_gt.load_res(&dt_path).unwrap();

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();

    // The name promises numeric stability across a refactor, so pin the numbers.
    // Asserting only the array shape and "some value is non-negative" would pass
    // with every metric wrong, which is what it did before.
    //
    // These twelve come from **pycocotools** on this same fixture, not from a
    // recording of hotcoco's own output — a snapshot would pin whatever the code
    // happened to do on the day it was written, including a bug. Regenerate with:
    //
    //   uv run python -c "from pycocotools.coco import COCO; \
    //     from pycocotools.cocoeval import COCOeval; \
    //     gt=COCO('crates/hotcoco/tests/fixtures/gt.json'); \
    //     dt=gt.loadRes('crates/hotcoco/tests/fixtures/dt.json'); \
    //     e=COCOeval(gt,dt,'bbox'); e.evaluate(); e.accumulate(); e.summarize()"
    ev.summarize();
    let stats = ev.stats().expect("summarize sets stats");

    #[rustfmt::skip]
    let expected: &[f64] = &[
        0.908416,  // AP  @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        1.000000,  // AP  @[ IoU=0.50      | area=   all | maxDets=100 ]
        1.000000,  // AP  @[ IoU=0.75      | area=   all | maxDets=100 ]
        0.925743,  // AP  @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
        0.900000,  // AP  @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
       -1.000000,  // AP  @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
        0.791667,  // AR  @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
        0.908333,  // AR  @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
        0.908333,  // AR  @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        0.925000,  // AR  @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
        0.900000,  // AR  @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
       -1.000000,  // AR  @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
    ];

    assert_eq!(stats.len(), expected.len());
    for (i, (&got, &exp)) in stats.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "stats[{i}]: got {got:.6}, expected {exp:.6}"
        );
    }

    let eval = ev.accumulated().unwrap();
    assert_eq!(eval.shape.t, 10);
    assert_eq!(eval.shape.k, 2);
}

#[test]
fn test_slice_by_full_dataset_matches_normal_eval() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");

    // Load twice since COCO doesn't implement Clone.
    let coco_gt1 = COCO::new(&gt_path).unwrap();
    let coco_dt1 = coco_gt1.load_res(&dt_path).unwrap();
    let coco_gt2 = COCO::new(&gt_path).unwrap();
    let coco_dt2 = coco_gt2.load_res(&dt_path).unwrap();

    let all_img_ids: Vec<u64> = coco_gt1.dataset.images.iter().map(|i| i.id).collect();

    let mut ev = COCOeval::new(coco_gt1, coco_dt1, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    let normal_results = ev.get_results(None, false);

    let mut ev2 = COCOeval::new(coco_gt2, coco_dt2, IouType::Bbox);
    ev2.evaluate();
    let sliced = ev2
        .slice_by(
            vec![("all_images".to_string(), all_img_ids)]
                .into_iter()
                .collect(),
        )
        .unwrap();

    for (key, &val) in &sliced.overall.metrics {
        let normal_val = normal_results.get(key).copied().unwrap();
        assert!(
            (val - normal_val).abs() < 1e-12,
            "overall {} mismatch: {} vs {}",
            key,
            val,
            normal_val
        );
    }

    let all_slice = sliced
        .slices
        .iter()
        .find(|s| s.name == "all_images")
        .unwrap();
    for (key, &val) in &all_slice.metrics {
        let normal_val = normal_results.get(key).copied().unwrap();
        assert!(
            (val - normal_val).abs() < 1e-12,
            "all_images slice {} mismatch: {} vs {}",
            key,
            val,
            normal_val
        );
    }

    for &d in all_slice.delta.values() {
        assert!(d.abs() < 1e-12, "delta should be zero for full dataset");
    }
}

#[test]
fn test_slice_by_disjoint_halves() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).unwrap();
    let coco_dt = coco_gt.load_res(&dt_path).unwrap();

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();

    let sliced = ev
        .slice_by(
            vec![
                ("first_two".to_string(), vec![1, 2]),
                ("last_one".to_string(), vec![3]),
            ]
            .into_iter()
            .collect(),
        )
        .unwrap();

    assert_eq!(sliced.slices.len(), 2);
    let first = sliced
        .slices
        .iter()
        .find(|s| s.name == "first_two")
        .unwrap();
    let last = sliced.slices.iter().find(|s| s.name == "last_one").unwrap();
    assert_eq!(first.num_images, 2);
    assert_eq!(last.num_images, 1);

    // A slice's metrics are defined as "evaluate only these images" — so they
    // must equal a from-scratch evaluation restricted to the same ids, metric
    // by metric (sentinels included). The old assertion — `any(v >= 0.0)` —
    // passed for any slice with one computed number, wrong or not.
    let eval_subset = |img_ids: Vec<u64>| -> std::collections::BTreeMap<String, f64> {
        let gt = COCO::new(&gt_path).unwrap();
        let dt = gt.load_res(&dt_path).unwrap();
        let mut e = COCOeval::new(gt, dt, IouType::Bbox);
        e.params.img_ids = img_ids;
        e.run();
        e.get_results(None, false)
    };

    for (slice, ids) in [(first, vec![1, 2]), (last, vec![3])] {
        let expected = eval_subset(ids);
        assert_eq!(
            slice.metrics.len(),
            expected.len(),
            "{}: metric key sets must match",
            slice.name
        );
        for (key, &val) in &slice.metrics {
            let exp = expected[key];
            assert!(
                (val - exp).abs() < 1e-12,
                "{} {key}: slice reported {val}, independent eval of the same \
                 images gives {exp}",
                slice.name
            );
        }
    }

    // And the two halves must actually disagree somewhere — disjoint image
    // sets with different detections should not produce identical metrics.
    assert!(
        first
            .metrics
            .iter()
            .any(|(k, v)| (v - last.metrics[k]).abs() > 1e-12),
        "disjoint halves reported identical metrics: {:?}",
        first.metrics
    );
}

#[test]
fn test_slice_by_reserved_name_rejected() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).unwrap();
    let coco_dt = coco_gt.load_res(&dt_path).unwrap();

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();

    let result = ev.slice_by(
        vec![("_overall".to_string(), vec![1])]
            .into_iter()
            .collect(),
    );
    assert!(result.is_err(), "_overall should be a reserved name");
}

#[test]
fn test_slice_by_requires_evaluate() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).unwrap();
    let coco_dt = coco_gt.load_res(&dt_path).unwrap();

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    let result = ev.slice_by(vec![("slice".to_string(), vec![1])].into_iter().collect());
    assert!(result.is_err(), "should error when evaluate() not called");
}

// ========== Open Images / Hierarchy tests ==========

#[test]
fn test_is_group_of_deserialization() {
    // With is_group_of present
    let json = r#"{
        "id": 1, "image_id": 1, "category_id": 1,
        "bbox": [0,0,10,10], "area": 100, "iscrowd": 0,
        "is_group_of": true
    }"#;
    let ann: Annotation = serde_json::from_str(json).unwrap();
    assert_eq!(ann.is_group_of, Some(true));

    // Without is_group_of (should default to None)
    let json2 = r#"{
        "id": 2, "image_id": 1, "category_id": 1,
        "bbox": [0,0,10,10], "area": 100, "iscrowd": 0
    }"#;
    let ann2: Annotation = serde_json::from_str(json2).unwrap();
    assert_eq!(ann2.is_group_of, None);

    // With is_group_of = false
    let json3 = r#"{
        "id": 3, "image_id": 1, "category_id": 1,
        "bbox": [0,0,10,10], "area": 100, "iscrowd": 0,
        "is_group_of": false
    }"#;
    let ann3: Annotation = serde_json::from_str(json3).unwrap();
    assert_eq!(ann3.is_group_of, Some(false));
}

#[test]
fn test_hierarchy_from_parent_map() {
    // Dog(1) -> Animal(2) -> Entity(3)
    // Cat(4) -> Animal(2)
    let mut pm: HashMap<u64, u64> = HashMap::new();
    pm.insert(1, 2); // Dog -> Animal
    pm.insert(4, 2); // Cat -> Animal
    pm.insert(2, 3); // Animal -> Entity

    let h = Hierarchy::from_parent_map(pm);

    // Dog ancestors: [Dog, Animal, Entity]
    let dog_ancestors = h.ancestors(1);
    assert_eq!(dog_ancestors.len(), 3);
    assert_eq!(dog_ancestors[0], 1);
    assert_eq!(dog_ancestors[1], 2);
    assert_eq!(dog_ancestors[2], 3);

    // Cat ancestors: [Cat, Animal, Entity]
    let cat_ancestors = h.ancestors(4);
    assert_eq!(cat_ancestors.len(), 3);
    assert_eq!(cat_ancestors[0], 4);
    assert_eq!(cat_ancestors[1], 2);
    assert_eq!(cat_ancestors[2], 3);

    // Animal ancestors: [Animal, Entity]
    let animal_ancestors = h.ancestors(2);
    assert_eq!(animal_ancestors.len(), 2);

    // Entity ancestors: [Entity]
    let entity_ancestors = h.ancestors(3);
    assert_eq!(entity_ancestors.len(), 1);
    assert_eq!(entity_ancestors[0], 3);

    // Children checks
    let animal_children = h.children(2);
    assert_eq!(animal_children.len(), 2);
    assert!(animal_children.contains(&1)); // Dog
    assert!(animal_children.contains(&4)); // Cat

    // Parent checks
    assert_eq!(h.parent(1), Some(2));
    assert_eq!(h.parent(3), None); // root
}

/// KNOWN BUG — reported to the coordinator, deliberately not fixed here:
/// `Hierarchy::from_parent_map` never terminates on a cyclic parent map. The
/// ancestor precomputation (`detection/hierarchy.rs`, the
/// `while let Some(&parent) = parent_map.get(&current)` walk) chases parent
/// links with no visited set, so `1 → 2 → 1` pushes ancestors forever until
/// OOM. The input is user-reachable: an explicit parent map, mutually
/// referencing `supercategory` names via `from_categories`, or a cyclic OID
/// hierarchy JSON.
///
/// `#[ignore]` keeps the suite green while pinning the expected behavior;
/// un-ignore once `from_parent_map` detects cycles (error or break — either
/// satisfies this test as written).
#[test]
fn test_hierarchy_cyclic_parent_map_terminates() {
    let mut pm: HashMap<u64, u64> = HashMap::new();
    pm.insert(1, 2);
    pm.insert(2, 1);

    let h = Hierarchy::from_parent_map(pm);
    // Termination is the real assertion; if construction returns at all, the
    // ancestor list must not have looped.
    assert!(
        h.ancestors(1).len() <= 2,
        "a 2-node cycle cannot yield more than 2 ancestors: {:?}",
        h.ancestors(1)
    );
}

#[test]
fn test_hierarchy_from_categories_supercategory() {
    let cats = vec![
        Category {
            id: 1,
            name: "dog".into(),
            supercategory: Some("animal".into()),
            ..Default::default()
        },
        Category {
            id: 2,
            name: "animal".into(),
            ..Default::default()
        },
        Category {
            id: 3,
            name: "cat".into(),
            supercategory: Some("animal".into()),
            ..Default::default()
        },
    ];

    let h = Hierarchy::from_categories(&cats);

    // Dog -> Animal
    assert_eq!(h.parent(1), Some(2));
    // Cat -> Animal
    assert_eq!(h.parent(3), Some(2));
    // Animal is root
    assert_eq!(h.parent(2), None);

    // Dog ancestors: [Dog, Animal]
    let dog_ancestors = h.ancestors(1);
    assert_eq!(dog_ancestors.len(), 2);
    assert_eq!(dog_ancestors[0], 1);
    assert_eq!(dog_ancestors[1], 2);
}

#[test]
fn test_hierarchy_virtual_nodes() {
    // "vehicle" supercategory doesn't match any category name -> virtual node
    let cats = vec![
        Category {
            id: 1,
            name: "car".into(),
            supercategory: Some("vehicle".into()),
            ..Default::default()
        },
        Category {
            id: 2,
            name: "truck".into(),
            supercategory: Some("vehicle".into()),
            ..Default::default()
        },
    ];

    let h = Hierarchy::from_categories(&cats);

    // Both car and truck should have the same parent (a virtual node)
    let car_parent = h.parent(1).unwrap();
    let truck_parent = h.parent(2).unwrap();
    assert_eq!(car_parent, truck_parent);
    // The semantic contract: the virtual node collides with no real category
    // and carries the supercategory's name. (Pinning `>= u64::MAX - 10` tied
    // the test to the current countdown allocation scheme instead.)
    assert!(
        ![1, 2].contains(&car_parent),
        "virtual node id {car_parent} collides with a real category id"
    );
    assert_eq!(
        h.virtual_names.get(&car_parent).map(String::as_str),
        Some("vehicle"),
        "virtual node should be named after the unmatched supercategory"
    );

    // Car ancestors: [car, vehicle_virtual]
    assert_eq!(h.ancestors(1).len(), 2);
}

#[test]
fn test_hierarchy_from_oid_json() {
    let label_to_id: HashMap<String, u64> = vec![
        ("/m/entity".to_string(), 100),
        ("/m/animal".to_string(), 200),
        ("/m/dog".to_string(), 300),
        ("/m/cat".to_string(), 400),
    ]
    .into_iter()
    .collect();

    let json = r#"{
        "LabelName": "/m/entity",
        "Subcategory": [
            {
                "LabelName": "/m/animal",
                "Subcategory": [
                    { "LabelName": "/m/dog" },
                    { "LabelName": "/m/cat" }
                ]
            }
        ]
    }"#;

    let h = Hierarchy::from_oid_json(json, &label_to_id).unwrap();

    // Dog -> Animal -> Entity
    assert_eq!(h.parent(300), Some(200));
    assert_eq!(h.parent(200), Some(100));
    assert_eq!(h.parent(100), None);

    let dog_ancestors = h.ancestors(300);
    assert_eq!(dog_ancestors.len(), 3);
    assert_eq!(dog_ancestors[0], 300);
    assert_eq!(dog_ancestors[1], 200);
    assert_eq!(dog_ancestors[2], 100);
}

#[test]
fn test_hierarchy_from_oid_json_unknown_labels_skipped() {
    // Only /m/dog is known; /m/entity and /m/animal get virtual nodes
    let label_to_id: HashMap<String, u64> = vec![("/m/dog".to_string(), 1)].into_iter().collect();

    let json = r#"{
        "LabelName": "/m/entity",
        "Subcategory": [
            {
                "LabelName": "/m/animal",
                "Subcategory": [
                    { "LabelName": "/m/dog" }
                ]
            }
        ]
    }"#;

    let h = Hierarchy::from_oid_json(json, &label_to_id).unwrap();

    // Dog (id=1) should have a parent: a virtual node standing in for the
    // unknown "/m/animal" label — asserted by name, not by the current
    // "counts down from u64::MAX − 1" id allocation.
    let dog_parent = h.parent(1).unwrap();
    assert_ne!(dog_parent, 1, "parent must not collide with the real id");
    assert_eq!(
        h.virtual_names.get(&dog_parent).map(String::as_str),
        Some("/m/animal"),
        "parent should be the virtual node for the unknown /m/animal label"
    );

    // Dog ancestors: [dog, virtual_animal, virtual_entity]
    assert_eq!(h.ancestors(1).len(), 3);
}

#[test]
fn test_gt_expansion_basic() {
    // Dog(1) -> Animal(2): a Dog GT should expand to Dog + Animal
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img.jpg".into(),
            height: 100,
            width: 100,
            ..Default::default()
        }],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1, // Dog
            bbox: Some([10.0, 10.0, 20.0, 20.0]),
            area: Some(400.0),
            ..Default::default()
        }],
        categories: vec![
            Category {
                id: 1,
                name: "dog".into(),
                ..Default::default()
            },
            Category {
                id: 2,
                name: "animal".into(),
                ..Default::default()
            },
        ],
        licenses: vec![],
    };

    let coco = COCO::from_dataset(gt_dataset);

    let mut pm: HashMap<u64, u64> = HashMap::new();
    pm.insert(1, 2); // Dog -> Animal
    let hierarchy = Hierarchy::from_parent_map(pm);

    let expanded = hotcoco::detection::expand::expand_annotations(&coco, &hierarchy);

    // Should have 2 annotations: original Dog + expanded Animal
    assert_eq!(
        expanded.dataset.annotations.len(),
        2,
        "Dog GT should expand to Dog + Animal"
    );

    let cat_ids: HashSet<u64> = expanded
        .dataset
        .annotations
        .iter()
        .map(|a| a.category_id)
        .collect();
    assert!(cat_ids.contains(&1), "should contain Dog");
    assert!(cat_ids.contains(&2), "should contain Animal");

    // Both annotations should share the same bbox
    for ann in &expanded.dataset.annotations {
        assert_eq!(ann.bbox, Some([10.0, 10.0, 20.0, 20.0]));
        assert_eq!(ann.image_id, 1);
    }
}

#[test]
fn test_gt_expansion_idempotent() {
    // If the annotation already has ancestors present, expanding again shouldn't duplicate
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img.jpg".into(),
            height: 100,
            width: 100,
            ..Default::default()
        }],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1, // Dog
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                ..Default::default()
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 2, // Animal (already present with same bbox)
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                ..Default::default()
            },
        ],
        categories: vec![
            Category {
                id: 1,
                name: "dog".into(),
                ..Default::default()
            },
            Category {
                id: 2,
                name: "animal".into(),
                ..Default::default()
            },
        ],
        licenses: vec![],
    };

    let coco = COCO::from_dataset(gt_dataset);

    let mut pm: HashMap<u64, u64> = HashMap::new();
    pm.insert(1, 2); // Dog -> Animal
    let hierarchy = Hierarchy::from_parent_map(pm);

    let expanded = hotcoco::detection::expand::expand_annotations(&coco, &hierarchy);

    // Should still have exactly 2 annotations — no duplicates
    assert_eq!(
        expanded.dataset.annotations.len(),
        2,
        "pre-expanded input should stay at 2 annotations"
    );
}

#[test]
fn test_oid_group_of_multi_match() {
    // One ordinary GT + one group-of GT. DT1 matches the ordinary GT. DT2 and DT3
    // both lie inside the group-of box; the Open Images protocol scores the
    // *best* of them as a single true positive and ignores the rest:
    //
    //   "If at least one detection is inside group-of box a single True Positive
    //    is scored. ... Multiple correct detections inside the same group-of box
    //    is still count as a single True Positive."
    //
    // DT3 is deliberately small — IoA 1.00 but IoU 0.16 — because that is the
    // case the old implementation got wrong. It matched group-of boxes on plain
    // IoU, so any detection smaller than the box (i.e. the normal case: one
    // object inside a crowd) fell through as a false positive.
    const ORDINARY_GT: [f64; 4] = [300.0, 300.0, 100.0, 100.0];
    const GROUP_OF_GT: [f64; 4] = [0.0, 0.0, 200.0, 200.0];
    const ABSORBED_A: [f64; 4] = [0.0, 0.0, 200.0, 200.0];
    const ABSORBED_B: [f64; 4] = [10.0, 10.0, 80.0, 80.0];

    // Preconditions, asserted rather than trusted. Note these are IoA, the
    // measure the protocol specifies — and that DT3 would fail an IoU test.
    assert!(ioa_of(ABSORBED_A, GROUP_OF_GT) >= 0.5);
    assert!(ioa_of(ABSORBED_B, GROUP_OF_GT) >= 0.5);
    assert!(
        iou_of(ABSORBED_B, GROUP_OF_GT) < 0.5,
        "DT3 must fail plain IoU, or this test cannot detect a regression to it"
    );
    assert!(ioa_of(ORDINARY_GT, GROUP_OF_GT) < 0.5);

    let cats = vec![cat(1, "person")];
    let gt_dataset = dataset(
        vec![img(1)],
        cats.clone(),
        vec![
            ann(1, ORDINARY_GT),            // one ground truth
            ann(2, GROUP_OF_GT).group_of(), // one more: a group-of box counts once
        ],
    );
    let dt_dataset = dataset(
        vec![img(1)],
        cats,
        vec![
            det(1, ORDINARY_GT, 0.9), // matches the ordinary GT -> TP
            det(2, ABSORBED_A, 0.8),  // best detection in the group box -> the one TP
            det(3, ABSORBED_B, 0.7),  // also inside it -> absorbed, neither TP nor FP
        ],
    );

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);
    let mut ev = COCOeval::new_oid(coco_gt, coco_dt, None);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().unwrap();
    // Two ground truths, both found: the ordinary GT by DT1, the group-of box by
    // DT2. DT3 is ignored, so it cannot depress precision.
    assert!(
        stats[0] > 0.99,
        "AP should be ~1.0: both GTs found, surplus group-of detection ignored, got {:.4}",
        stats[0]
    );

    // The mechanism. OID runs a single IoU threshold, so t_idx is 0.
    let e = ev
        .eval_imgs()
        .iter()
        .flatten()
        .next()
        .expect("one image/category cell");
    let ignored: Vec<u64> = e
        .dt_ids
        .iter()
        .zip(e.dt_ignore.row(0))
        .filter(|&(_, &ig)| ig)
        .map(|(&id, _)| id)
        .collect();
    assert_eq!(
        ignored,
        vec![3],
        "only the surplus detection is ignored; the best one scores the box"
    );
    assert_eq!(
        e.dt_matches[(0, 1)],
        2,
        "DT2 should be paired with the group-of GT (id 2)"
    );

    // The group-of box is counted, and counted exactly once.
    assert_eq!(
        e.gt_in_denominator.iter().filter(|&&x| x).count(),
        2,
        "ordinary GT + group-of box = 2 ground truths in the denominator"
    );

    // Recall stays bounded. This configuration measured 4.0 when the group-of
    // pass credited every overlapping detection against a denominator of 1.
    let acc = ev.accumulated().expect("accumulate() was called");
    let a_idx = ev.params.all_area_idx();
    let m_idx = acc.shape.m - 1;
    for t in 0..acc.shape.t {
        for k in 0..acc.shape.k {
            let r = acc.recall[acc.shape.recall_idx(t, k, a_idx, m_idx)];
            assert!(
                r <= 1.0,
                "recall must not exceed 1.0 (t={t}, k={k}), got {r}"
            );
        }
    }
}

#[test]
fn test_oid_group_of_matches_on_ioa_not_iou() {
    // Regression test for the defect that made group-of handling near-inoperative:
    // the IoU matrix was built with the crowd flag forced off under OID, so
    // group-of boxes were matched on plain IoU. Any detection *smaller* than the
    // group box — one object inside a crowd, i.e. the normal case — fell through
    // as a false positive instead of being absorbed.
    //
    // The scoring here is deliberate. The small detection outranks the ordinary
    // true positive, so a regression puts its false positive *before* full recall
    // where the interpolated precision cannot hide it. With the order reversed
    // this test reports AP 1.0 either way and proves nothing — which is exactly
    // how the original group-of tests passed for the whole life of the feature.
    const ORDINARY_GT: [f64; 4] = [300.0, 300.0, 100.0, 100.0];
    const GROUP_OF_GT: [f64; 4] = [0.0, 0.0, 200.0, 200.0];
    const INSIDE: [f64; 4] = [10.0, 10.0, 80.0, 80.0];

    assert!(
        ioa_of(INSIDE, GROUP_OF_GT) >= 0.5,
        "the protocol's measure must accept this detection"
    );
    assert!(
        iou_of(INSIDE, GROUP_OF_GT) < 0.5,
        "and plain IoU must reject it, or the test cannot tell the two apart"
    );

    let cats = vec![cat(1, "person")];
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        cats.clone(),
        vec![ann(1, ORDINARY_GT), ann(2, GROUP_OF_GT).group_of()],
    ));
    let coco_dt = COCO::from_dataset(dataset(
        vec![img(1)],
        cats,
        vec![
            det(1, INSIDE, 0.9),      // inside the group box; scores it
            det(2, ORDINARY_GT, 0.8), // the ordinary true positive
        ],
    ));

    let mut ev = COCOeval::new_oid(coco_gt, coco_dt, None);
    ev.run();

    // Both ground truths found, no false positives -> AP 1.0.
    // Matched on IoU instead, the first detection is an FP at recall 0 and AP
    // collapses to ~0.25.
    let stats = ev.stats().unwrap();
    assert!(
        stats[0] > 0.99,
        "detection inside a group-of box must be absorbed, not counted FP; got AP {:.4}",
        stats[0]
    );
}

#[test]
fn test_oid_undetected_group_of_is_a_miss() {
    // A group-of box nothing was detected inside. The Open Images protocol:
    // "Otherwise, the group-of box is counted as a single False Negative."
    //
    // Contrast with Open Images V2 (TF's `group_of_weight = 0.0`), where an
    // undetected group-of box costs nothing. Both are real protocols; hotcoco
    // follows the Challenge one, as FiftyOne does. Naming the alternative here so
    // a future reader does not "fix" this back to V2.
    const FOUND_GT: [f64; 4] = [0.0, 0.0, 100.0, 100.0];
    const MISSED_GROUP_OF: [f64; 4] = [400.0, 400.0, 100.0, 100.0];

    // The detection must not reach the group-of box, or this tests nothing.
    assert!(ioa_of(FOUND_GT, MISSED_GROUP_OF) < 0.5);

    let cats = vec![cat(1, "person")];
    let coco_gt = COCO::from_dataset(dataset(
        vec![img(1)],
        cats.clone(),
        vec![ann(1, FOUND_GT), ann(2, MISSED_GROUP_OF).group_of()],
    ));
    let coco_dt = COCO::from_dataset(dataset(vec![img(1)], cats, vec![det(1, FOUND_GT, 0.9)]));

    let mut ev = COCOeval::new_oid(coco_gt, coco_dt, None);
    ev.run();

    // Two ground truths in the denominator, one found. Recall caps at 0.5 and
    // precision is 1.0 up to there, so all-points AP is exactly 0.5.
    let stats = ev.stats().unwrap();
    assert!(
        (stats[0] - 0.5).abs() < 1e-9,
        "AP should be 0.5: the undetected group-of box is a miss, got {:.6}",
        stats[0]
    );

    let e = ev
        .eval_imgs()
        .iter()
        .flatten()
        .next()
        .expect("one image/category cell");
    assert_eq!(
        e.num_gt_in_denominator(),
        2,
        "the group-of box counts toward the denominator even though nothing hit it"
    );
    assert!(
        !e.gt_matched[(0, 1)],
        "the group-of box should be unmatched — that is what makes it a miss"
    );
}

#[test]
fn test_oid_hierarchy_evaluation() {
    // Hierarchy: Poodle(1) -> Dog(2) -> Animal(3)
    // GT: one Poodle annotation. DT: one Dog prediction at same bbox.
    // After expansion, Dog GT exists → Dog detection should get AP=1.0 at Dog level.
    let mut parent_map = HashMap::new();
    parent_map.insert(1, 2);
    parent_map.insert(2, 3);
    let hierarchy = Hierarchy::from_parent_map(parent_map);

    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img1.jpg".into(),
            height: 640,
            width: 640,
            ..Default::default()
        }],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1, // Poodle
            bbox: Some([10.0, 10.0, 100.0, 100.0]),
            area: Some(10000.0),
            ..Default::default()
        }],
        categories: vec![
            Category {
                id: 1,
                name: "poodle".into(),
                supercategory: Some("dog".into()),
                ..Default::default()
            },
            Category {
                id: 2,
                name: "dog".into(),
                supercategory: Some("animal".into()),
                ..Default::default()
            },
            Category {
                id: 3,
                name: "animal".into(),
                ..Default::default()
            },
        ],
        licenses: vec![],
    };

    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 2, // Dog prediction
            bbox: Some([10.0, 10.0, 100.0, 100.0]),
            area: Some(10000.0),
            score: Some(0.9),
            ..Default::default()
        }],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);
    let mut ev = COCOeval::new_oid(coco_gt, coco_dt, Some(hierarchy));
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let results = ev.results(true).unwrap();
    let per_class = results.per_class.as_ref().unwrap();

    // Dog AP should be 1.0 (detection matches expanded Dog GT)
    let dog_ap = per_class
        .iter()
        .find(|(k, _)| k.contains("dog"))
        .map(|(_, &v)| v)
        .expect("dog should be in per-class results");
    assert!(
        (dog_ap - 1.0).abs() < 1e-6,
        "Dog AP should be 1.0, got {dog_ap:.6}"
    );
}

#[test]
fn test_oid_dt_expansion() {
    // Hierarchy: Dog(1) -> Animal(2)
    // GT: Animal. DT: Dog.
    // Without DT expansion: Dog detection doesn't match Animal GT → low AP.
    // With DT expansion: Dog detection expanded to Animal → matches → better AP.
    let mut parent_map = HashMap::new();
    parent_map.insert(1, 2);
    let hierarchy = Hierarchy::from_parent_map(parent_map);

    let img = Image {
        id: 1,
        file_name: "img1.jpg".into(),
        height: 640,
        width: 640,
        ..Default::default()
    };
    let cats = vec![
        Category {
            id: 1,
            name: "dog".into(),
            supercategory: Some("animal".into()),
            ..Default::default()
        },
        Category {
            id: 2,
            name: "animal".into(),
            ..Default::default()
        },
    ];

    let gt_dataset = Dataset {
        info: None,
        images: vec![img.clone()],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 2, // Animal GT
            bbox: Some([10.0, 10.0, 100.0, 100.0]),
            area: Some(10000.0),
            ..Default::default()
        }],
        categories: cats.clone(),
        licenses: vec![],
    };

    let dt_dataset = Dataset {
        info: None,
        images: vec![img],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1, // Dog prediction
            bbox: Some([10.0, 10.0, 100.0, 100.0]),
            area: Some(10000.0),
            score: Some(0.9),
            ..Default::default()
        }],
        categories: cats,
        licenses: vec![],
    };

    // Without DT expansion
    let coco_gt = COCO::from_dataset(gt_dataset.clone());
    let coco_dt = COCO::from_dataset(dt_dataset.clone());
    let mut ev1 = COCOeval::new_oid(coco_gt, coco_dt, Some(hierarchy.clone()));
    ev1.evaluate();
    ev1.accumulate();
    ev1.summarize();
    let stats_no_expand = ev1.stats().unwrap().to_vec();

    // With DT expansion
    let coco_gt2 = COCO::from_dataset(gt_dataset);
    let coco_dt2 = COCO::from_dataset(dt_dataset);
    let mut ev2 = COCOeval::new_oid(coco_gt2, coco_dt2, Some(hierarchy));
    ev2.params.expand_dt = true;
    ev2.evaluate();
    ev2.accumulate();
    ev2.summarize();
    let stats_expand = ev2.stats().unwrap().to_vec();

    // With DT expansion, AP should be higher
    assert!(
        stats_expand[0] > stats_no_expand[0],
        "DT expansion should improve AP: expand={:.4} vs no_expand={:.4}",
        stats_expand[0],
        stats_no_expand[0]
    );
}

#[test]
fn test_oid_auto_derive_hierarchy() {
    // No explicit hierarchy — derive from supercategory fields.
    // Dog(1) has supercategory "animal" which matches Animal(2).
    // GT: Dog. DT: Animal prediction. After expansion, Animal GT exists → match.
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img1.jpg".into(),
            height: 640,
            width: 640,
            ..Default::default()
        }],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([10.0, 10.0, 100.0, 100.0]),
            area: Some(10000.0),
            ..Default::default()
        }],
        categories: vec![
            Category {
                id: 1,
                name: "dog".into(),
                supercategory: Some("animal".into()),
                ..Default::default()
            },
            Category {
                id: 2,
                name: "animal".into(),
                ..Default::default()
            },
        ],
        licenses: vec![],
    };

    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 2, // Animal prediction
            bbox: Some([10.0, 10.0, 100.0, 100.0]),
            area: Some(10000.0),
            score: Some(0.9),
            ..Default::default()
        }],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);

    // No hierarchy arg → auto-derive from supercategory
    let mut ev = COCOeval::new_oid(coco_gt, coco_dt, None);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().unwrap();
    // Dog GT expanded to Animal, Animal detection matches → AP > 0
    assert!(
        stats[0] > 0.0,
        "Auto-derived hierarchy should enable hierarchical matching, got AP={:.4}",
        stats[0]
    );
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

#[test]
fn test_calibration_basic() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();

    let cal = ev.calibration(10, 0.5).expect("calibration should succeed");

    assert_eq!(cal.n_bins, 10);
    assert_eq!(cal.bins.len(), 10);
    assert!((cal.iou_threshold - 0.5).abs() < 1e-9);
    assert!(cal.num_detections > 0, "should have some detections");
    assert!(cal.ece >= 0.0, "ECE must be non-negative");
    assert!(cal.mce >= 0.0, "MCE must be non-negative");
    assert!(cal.mce >= cal.ece, "MCE must be >= ECE");

    // Per-category should have entries for both categories
    assert!(!cal.per_category.is_empty());

    // Bin counts should sum to num_detections
    let total: usize = cal.bins.iter().map(|b| b.count).sum();
    assert_eq!(total, cal.num_detections);
}

#[test]
fn test_calibration_requires_evaluate() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    let result = ev.calibration(10, 0.5);
    assert!(result.is_err(), "should fail before evaluate()");
}

#[test]
fn test_calibration_invalid_iou_threshold() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();

    let result = ev.calibration(10, 0.42);
    assert!(
        result.is_err(),
        "should fail with non-standard IoU threshold"
    );
}

/// Unnormalized scores are rejected rather than silently producing an ECE above 1.
///
/// Binning saturates an out-of-range score into an end bin while keeping its raw
/// magnitude in that bin's mean, so a model exporting logits would otherwise get a
/// plausible-looking number that means nothing. See
/// `metrics::calibration::tests::out_of_range_scores_escape_their_bin` for the
/// mechanism this guards.
#[test]
fn test_calibration_rejects_unnormalized_scores() {
    let cats = vec![cat(1, "person")];
    let gt = dataset(
        vec![img(1)],
        cats.clone(),
        vec![ann(1, [10.0, 10.0, 50.0, 50.0])],
    );
    // A logit, not a probability.
    let dt = dataset(
        vec![img(1)],
        cats,
        vec![det(1, [10.0, 10.0, 50.0, 50.0], 7.4)],
    );

    let mut ev = COCOeval::new(
        COCO::from_dataset(gt),
        COCO::from_dataset(dt),
        IouType::Bbox,
    );
    ev.evaluate();

    let err = ev
        .calibration(10, 0.5)
        .expect_err("a score of 7.4 is not a confidence and must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("7.4") && msg.contains("[0, 1]"),
        "error should name the offending score and the required range, got: {msg}"
    );
}

#[test]
fn test_calibration_known_values() {
    // Construct a scenario with known calibration:
    // 2 high-confidence TPs (0.9) and 2 low-confidence FPs (0.2)
    // High bin: avg_conf=0.9, avg_acc=1.0, gap=0.1
    // Low bin:  avg_conf=0.2, avg_acc=0.0, gap=0.2
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "a.jpg".into(),
            height: 100,
            width: 100,
            ..Default::default()
        }],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 30.0, 30.0]),
                area: Some(900.0),
                ..Default::default()
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([60.0, 60.0, 30.0, 30.0]),
                area: Some(900.0),
                ..Default::default()
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "obj".into(),
            supercategory: Some(String::new()),
            ..Default::default()
        }],
        licenses: vec![],
    };

    let dt_dataset = Dataset {
        info: None,
        images: vec![],
        annotations: vec![
            // TP: matches GT 1
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 30.0, 30.0]),
                area: Some(900.0),
                score: Some(0.9),
                ..Default::default()
            },
            // TP: matches GT 2
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([60.0, 60.0, 30.0, 30.0]),
                area: Some(900.0),
                score: Some(0.9),
                ..Default::default()
            },
            // FP: no matching GT
            Annotation {
                id: 3,
                image_id: 1,
                category_id: 1,
                bbox: Some([0.0, 0.0, 5.0, 5.0]),
                area: Some(25.0),
                score: Some(0.2),
                ..Default::default()
            },
            // FP: no matching GT
            Annotation {
                id: 4,
                image_id: 1,
                category_id: 1,
                bbox: Some([90.0, 90.0, 5.0, 5.0]),
                area: Some(25.0),
                score: Some(0.2),
                ..Default::default()
            },
        ],
        categories: vec![],
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();

    let cal = ev.calibration(10, 0.5).expect("calibration should succeed");

    // 4 non-ignored detections total
    assert_eq!(cal.num_detections, 4);

    // Bin [0.1, 0.2): 0 detections (score 0.2 goes to bin index 2)
    // Bin [0.2, 0.3): 2 FPs → avg_conf=0.2, avg_acc=0.0
    // Bin [0.9, 1.0): 2 TPs → avg_conf=0.9, avg_acc=1.0
    let low_bin = &cal.bins[2]; // [0.2, 0.3)
    assert_eq!(low_bin.count, 2);
    assert!((low_bin.avg_accuracy - 0.0).abs() < 1e-9);

    let high_bin = &cal.bins[9]; // [0.9, 1.0)
    assert_eq!(high_bin.count, 2);
    assert!((high_bin.avg_accuracy - 1.0).abs() < 1e-9);

    // ECE = (2/4)*|0.0-0.2| + (2/4)*|1.0-0.9| = 0.5*0.2 + 0.5*0.1 = 0.15
    assert!(
        (cal.ece - 0.15).abs() < 1e-9,
        "Expected ECE=0.15, got {:.6}",
        cal.ece
    );
    // MCE = max(0.2, 0.1) = 0.2
    assert!(
        (cal.mce - 0.2).abs() < 1e-9,
        "Expected MCE=0.2, got {:.6}",
        cal.mce
    );
}

// ---------------------------------------------------------------------------
// Compare / bootstrap validation
// ---------------------------------------------------------------------------

/// Build a synthetic dataset with `n` images, 1 category, 1 GT bbox per image.
///
/// Returns (gt_dataset, dt_good_dataset, dt_weak_dataset) where:
/// - dt_good has perfect-overlap detections on all images (all TPs, AP≈1.0)
/// - dt_weak has detections on only the first half of images (misses half, adds FPs)
///
/// The AP difference between them is real and measurable.
fn make_compare_fixtures(n: usize) -> (Dataset, Dataset, Dataset) {
    let cat = Category {
        id: 1,
        name: "thing".into(),
        ..Default::default()
    };

    let images: Vec<Image> = (1..=n)
        .map(|i| Image {
            id: i as u64,
            file_name: format!("img{i}.jpg"),
            height: 200,
            width: 200,
            ..Default::default()
        })
        .collect();

    let gt_anns: Vec<Annotation> = (1..=n)
        .map(|i| Annotation {
            id: i as u64,
            image_id: i as u64,
            category_id: 1,
            bbox: Some([10.0, 10.0, 50.0, 50.0]),
            area: Some(2500.0),
            ..Default::default()
        })
        .collect();

    let make_ann = |id: u64, image_id: u64, bbox: [f64; 4], score: f64| Annotation {
        id,
        image_id,
        category_id: 1,
        bbox: Some(bbox),
        area: Some(bbox[2] * bbox[3]),
        score: Some(score),
        ..Default::default()
    };

    // Good model: perfect TP on every image
    let dt_good_anns: Vec<Annotation> = (1..=n)
        .map(|i| make_ann(i as u64, i as u64, [10.0, 10.0, 50.0, 50.0], 0.9))
        .collect();

    // Weak model: TPs on first half only, plus an FP on each detected image
    let half = n / 2;
    let mut dt_weak_anns: Vec<Annotation> = Vec::new();
    for i in 1..=half {
        // TP
        dt_weak_anns.push(make_ann(i as u64, i as u64, [10.0, 10.0, 50.0, 50.0], 0.8));
        // FP (wrong location)
        dt_weak_anns.push(make_ann(
            (n + i) as u64,
            i as u64,
            [120.0, 120.0, 30.0, 30.0],
            0.6,
        ));
    }

    let dt_good = Dataset {
        info: None,
        images: images.clone(),
        annotations: dt_good_anns,
        categories: vec![cat.clone()],
        licenses: vec![],
    };
    let dt_weak = Dataset {
        info: None,
        images: images.clone(),
        annotations: dt_weak_anns,
        categories: vec![cat.clone()],
        licenses: vec![],
    };

    let gt = Dataset {
        info: None,
        images,
        annotations: gt_anns,
        categories: vec![cat],
        licenses: vec![],
    };

    (gt, dt_good, dt_weak)
}

#[test]
fn test_compare_bootstrap_ci_contains_point_estimate() {
    // Two genuinely different models: high-score vs low-score detections.
    // The point estimate delta should lie within the bootstrap CI.
    let (gt_ds, dt_good_ds, dt_weak_ds) = make_compare_fixtures(30);

    let gt_a = COCO::from_dataset(gt_ds.clone());
    let dt_a = COCO::from_dataset(dt_good_ds);
    let mut ev_a = COCOeval::new(gt_a, dt_a, IouType::Bbox);
    ev_a.evaluate();

    let gt_b = COCO::from_dataset(gt_ds);
    let dt_b = COCO::from_dataset(dt_weak_ds);
    let mut ev_b = COCOeval::new(gt_b, dt_b, IouType::Bbox);
    ev_b.evaluate();

    let opts = hotcoco::CompareOpts {
        n_bootstrap: 200,
        seed: 42,
        confidence: 0.95,
    };
    let result = hotcoco::compare(&ev_a, &ev_b, &opts).unwrap();

    // There should be a real difference (weak model has lower AP)
    let ap_delta = result.deltas["AP"];
    assert!(
        ap_delta < -0.01,
        "Expected negative AP delta (weak vs good), got {ap_delta}"
    );

    let ci = result.ci.as_ref().unwrap();
    for (key, boot_ci) in ci {
        let delta = result.deltas[key];
        // Skip metrics with no data (delta = 0 for sentinel values)
        if delta.abs() < 1e-15 {
            continue;
        }
        // Point estimate should be within the CI
        assert!(
            boot_ci.lower <= delta && delta <= boot_ci.upper,
            "{key}: point estimate {delta:.6} outside CI [{:.6}, {:.6}]",
            boot_ci.lower,
            boot_ci.upper,
        );
        // CI should have positive width
        assert!(
            boot_ci.upper >= boot_ci.lower,
            "{key}: inverted CI [{:.6}, {:.6}]",
            boot_ci.lower,
            boot_ci.upper,
        );
        // std_err should be non-negative
        assert!(boot_ci.std_err >= 0.0, "{key}: negative std_err");
    }

    // AP CI should indicate the difference is significant (CI entirely below zero)
    let ap_ci = &ci["AP"];
    assert!(
        ap_ci.upper < 0.0,
        "Expected AP CI entirely below zero for weak-vs-good, got [{:.6}, {:.6}]",
        ap_ci.lower,
        ap_ci.upper,
    );
    assert!(
        ap_ci.prob_positive < 0.1,
        "Expected low prob_positive for negative delta, got {:.3}",
        ap_ci.prob_positive,
    );
}

#[test]
fn test_compare_bootstrap_coverage() {
    // Run bootstrap with many different seeds and check that the CI
    // covers the "true" delta (point estimate) at roughly the stated rate.
    // With 95% confidence and 50 trials, we expect ~47-48 to cover.
    let (gt_ds, dt_good_ds, dt_weak_ds) = make_compare_fixtures(30);

    // Compute the "true" delta (no bootstrap, full dataset)
    let gt_a = COCO::from_dataset(gt_ds.clone());
    let dt_a = COCO::from_dataset(dt_good_ds.clone());
    let mut ev_a = COCOeval::new(gt_a, dt_a, IouType::Bbox);
    ev_a.evaluate();

    let gt_b = COCO::from_dataset(gt_ds.clone());
    let dt_b = COCO::from_dataset(dt_weak_ds.clone());
    let mut ev_b = COCOeval::new(gt_b, dt_b, IouType::Bbox);
    ev_b.evaluate();

    let baseline = hotcoco::compare(&ev_a, &ev_b, &hotcoco::CompareOpts::default()).unwrap();
    let true_ap_delta = baseline.deltas["AP"];

    // Run 50 bootstrap trials with different seeds
    let n_trials = 50;
    let mut covers = 0;
    for seed in 0..n_trials {
        let gt_a = COCO::from_dataset(gt_ds.clone());
        let dt_a = COCO::from_dataset(dt_good_ds.clone());
        let mut ev_a = COCOeval::new(gt_a, dt_a, IouType::Bbox);
        ev_a.evaluate();

        let gt_b = COCO::from_dataset(gt_ds.clone());
        let dt_b = COCO::from_dataset(dt_weak_ds.clone());
        let mut ev_b = COCOeval::new(gt_b, dt_b, IouType::Bbox);
        ev_b.evaluate();

        let opts = hotcoco::CompareOpts {
            n_bootstrap: 200,
            seed,
            confidence: 0.95,
        };
        let result = hotcoco::compare(&ev_a, &ev_b, &opts).unwrap();
        let ci = result.ci.as_ref().unwrap();
        let ap_ci = &ci["AP"];
        if ap_ci.lower <= true_ap_delta && true_ap_delta <= ap_ci.upper {
            covers += 1;
        }

        // The interval must also be *informative*. Coverage bounded only from
        // below is satisfied by a degenerate [-1, 1] interval (an AP delta
        // cannot leave [-1, 1], so a whole-range CI covers 100% of the time and
        // measures nothing) — the exact failure mode of feeding `confidence`
        // in the wrong unit, which silently degrades to [min, max].
        let width = ap_ci.upper - ap_ci.lower;
        assert!(
            width.is_finite() && width >= 0.0,
            "seed {seed}: CI [{:.6}, {:.6}] is not a finite interval",
            ap_ci.lower,
            ap_ci.upper
        );
        assert!(
            width < 1.0,
            "seed {seed}: CI width {width:.4} spans most of the possible delta \
             range — an uninformative interval that coverage alone cannot catch"
        );
    }

    // With 95% nominal coverage and 50 trials, expect ~47.5 covers.
    // Allow a wide margin (80-100%) since 50 trials has high variance
    // and our dedup-bootstrap is slightly conservative (wider CIs).
    let coverage = covers as f64 / n_trials as f64;
    assert!(
        coverage >= 0.80,
        "Bootstrap coverage {:.1}% ({covers}/{n_trials}) is too low — expected ≥80% for 95% CI",
        coverage * 100.0
    );
}

// =============================================================================
// OBB (Oriented Bounding Box) evaluation tests
// =============================================================================

#[test]
fn test_obb_eval_basic() {
    // GT: one rotated box, DT: same box with high score → AP ≈ 1.0
    let gt_dataset = Dataset {
        info: None,
        images: vec![img(1)],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([90.0, 90.0, 220.0, 120.0]),
            area: Some(20000.0),
            obb: Some([200.0, 150.0, 200.0, 100.0, 0.3]),
            ..Default::default()
        }],
        categories: vec![cat(1, "vehicle")],
        licenses: vec![],
    };

    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([90.0, 90.0, 220.0, 120.0]),
            area: Some(20000.0),
            obb: Some([200.0, 150.0, 200.0, 100.0, 0.3]),
            score: Some(0.99),
            ..Default::default()
        }],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Obb);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().unwrap();
    // 12 metrics (same as bbox)
    assert_eq!(stats.len(), 12, "OBB should produce 12 metrics");
    // AP should be 1.0 (perfect detection)
    assert!(
        stats[0] > 0.99,
        "AP should be ~1.0 for identical OBBs, got {}",
        stats[0]
    );
}

#[test]
fn test_obb_eval_no_overlap() {
    // GT and DT have non-overlapping OBBs → AP = 0 (or -1)
    let gt_dataset = Dataset {
        info: None,
        images: vec![img(1)],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([0.0, 0.0, 50.0, 50.0]),
            area: Some(2500.0),
            obb: Some([25.0, 25.0, 50.0, 50.0, 0.0]),
            ..Default::default()
        }],
        categories: vec![cat(1, "vehicle")],
        licenses: vec![],
    };

    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([700.0, 500.0, 50.0, 50.0]),
            area: Some(2500.0),
            obb: Some([725.0, 525.0, 50.0, 50.0, 0.0]),
            score: Some(0.9),
            ..Default::default()
        }],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Obb);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().unwrap();
    assert_eq!(stats.len(), 12);
    // There *is* a ground truth (2500 px², medium), so AP is a computed 0.0 —
    // not the -1.0 "not computed" sentinel the old `<= 0.0` also accepted.
    assert_eq!(
        stats[0], 0.0,
        "AP must be exactly 0.0 for non-overlapping OBBs (-1.0 would mean the \
         cell was never computed)"
    );
    assert_eq!(stats[4], 0.0, "APm: the medium GT was evaluated and missed");
    assert_eq!(stats[3], -1.0, "APs: no small GT, so the sentinel");
}

#[test]
fn test_dota_round_trip_integration() {
    let dataset = Dataset {
        info: None,
        images: vec![img(1)],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([90.0, 90.0, 220.0, 120.0]),
            area: Some(20000.0),
            obb: Some([200.0, 150.0, 200.0, 100.0, 0.0]),
            ..Default::default()
        }],
        categories: vec![cat(1, "vehicle")],
        licenses: vec![],
    };

    let tmp = tempfile::TempDir::new().unwrap();
    let label_dir = tmp.path().join("labels");

    // Export COCO → DOTA
    let stats = coco_to_dota(&dataset, &label_dir).unwrap();
    assert_eq!(stats.images, 1);
    assert_eq!(stats.annotations, 1);

    // Import DOTA → COCO
    let mut dims = HashMap::new();
    dims.insert("obb_test".into(), (800u32, 600u32));
    let result = dota_to_coco(&label_dir, None, &dims).unwrap();

    assert_eq!(result.categories[0].name, "vehicle");

    // DOTA reconstructs file names from stems and prints corners to 1 decimal
    // place (0.2 tolerance on center/extent); the recovered angle is far
    // tighter than the coordinate rounding.
    assert_geometry_round_trip(
        &dataset,
        &result,
        file_stem_key,
        obb_of,
        [0.2, 0.2, 0.2, 0.2, 0.01],
    );
}

/// Empty `max_dets` degrades instead of panicking: `evaluate()` runs under
/// `Params::max_det()`'s fallback cap (100), and with no M slots to report,
/// every summary stat is the `-1.0` "not computed" sentinel — the same
/// degradation a missing area label or IoU threshold gets. (This used to be an
/// `assert!` panic, the only hard stop on a path whose siblings all degrade.)
/// The deeper end-to-end coverage lives in `tests/detection_fixes.rs`.
#[test]
fn test_empty_max_dets_degrades_gracefully() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.params.max_dets = vec![];
    coco_eval.run(); // must not panic

    let stats = coco_eval.stats().expect("summarize ran");
    assert!(!stats.is_empty());
    assert!(
        stats.iter().all(|&v| v == -1.0),
        "no max-det slots means nothing is computable: {stats:?}"
    );
}

// ---------------------------------------------------------------------------
// Threshold-epsilon policy (pycocotools' `min(t, 1-1e-10)` match floor)
// ---------------------------------------------------------------------------

/// One image, one GT and one DT, both category 1, bboxes as given.
fn one_box_pair(
    image_id: u64,
    gt_bbox: [f64; 4],
    dt_bbox: [f64; 4],
) -> (Image, Annotation, Annotation) {
    let img = Image {
        id: image_id,
        file_name: format!("img{image_id}.jpg"),
        height: 200,
        width: 200,
        ..Default::default()
    };
    let gt = Annotation {
        id: image_id * 10,
        image_id,
        category_id: 1,
        bbox: Some(gt_bbox),
        area: Some(gt_bbox[2] * gt_bbox[3]),
        ..Default::default()
    };
    let dt = Annotation {
        id: image_id * 10 + 1,
        image_id,
        category_id: 1,
        bbox: Some(dt_bbox),
        area: Some(dt_bbox[2] * dt_bbox[3]),
        score: Some(0.9),
        ..Default::default()
    };
    (img, gt, dt)
}

/// At `iou_thr == 1.0`, pycocotools searches from `min(t, 1-1e-10)`, so a pair
/// whose IoU lies in `[1-1e-10, 1.0)` still matches. hotcoco applies the same
/// clamp in the detection `evaluate()` path via
/// `primitives::greedy::coco_match_floor`.
///
/// Image 1 is *near*-identical: a 100x100 GT against a 100 x 100.000000005 DT,
/// giving IoU = 1/(1 + 5e-11) ~= 1 - 5e-11, which sits above the 1-1e-10 floor
/// but strictly below 1.0. Image 2 is exactly identical (IoU exactly 1.0).
///
/// Clamped (correct): both match, AP = 1.0.
/// Unclamped (pre-1.0 behavior): only image 2 matches, AP = 0.5.
#[test]
fn test_match_floor_clamped_at_iou_threshold_one() {
    let (img1, gt1, dt1) = one_box_pair(
        1,
        [0.0, 0.0, 100.0, 100.0],
        [0.0, 0.0, 100.0, 100.000000005],
    );
    let (img2, gt2, dt2) = one_box_pair(2, [0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]);

    let categories = vec![Category {
        id: 1,
        name: "thing".into(),
        ..Default::default()
    }];

    let gt_dataset = Dataset {
        info: None,
        images: vec![img1.clone(), img2.clone()],
        annotations: vec![gt1, gt2],
        categories: categories.clone(),
        licenses: vec![],
    };
    let dt_dataset = Dataset {
        info: None,
        images: vec![img1, img2],
        annotations: vec![dt1, dt2],
        categories,
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.params.iou_thrs = vec![1.0];
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let ap = ev.stats().expect("stats")[0];
    assert!(
        (ap - 1.0).abs() < 1e-12,
        "AP at iou_thr=1.0 should be 1.0 with the pycocotools match floor \
         (got {ap}); 0.5 means the near-identical pair was not matched, i.e. \
         the min(t, 1-1e-10) clamp is missing"
    );
}

/// The clamp must be inert below 1.0 — otherwise it would perturb the default
/// 0.50:0.05:0.95 sweep, which is the whole reason it is safe to apply.
#[test]
fn test_match_floor_is_inert_below_one() {
    use hotcoco::primitives::greedy::coco_match_floor;
    for t in [0.0, 0.1, 0.5, 0.75, 0.95, 0.99, 0.999999] {
        assert_eq!(coco_match_floor(t), t, "clamp must not fire at t={t}");
    }
    assert_eq!(coco_match_floor(1.0), 1.0 - 1e-10);
}

// ---------------------------------------------------------------------------
// Public API surface
// ---------------------------------------------------------------------------

/// The crate-root API surface 1.0 commits to.
///
/// Naming each type *is* the assertion — this file stops compiling if a path
/// stops resolving. These are the paths real code uses, and unlike the module
/// paths they are stable from 1.0 onward.
///
/// 1.0 renamed `eval` to `detection` and moved `hierarchy`, `healthcheck`, the
/// statistics DTOs, and `counts` — with **no compatibility aliases**. 0.x is
/// pre-release under SemVer, so the old module paths carried no stability
/// promise; carrying them through 1.x would have meant a multi-year obligation
/// for a surface nothing depended on. The crate-root re-exports below absorbed
/// every one of those moves, so most code needed no edit at all.
///
/// `cargo-semver-checks` (`just semver`) covers the same ground mechanically
/// against the published release; this catches it without needing the network.
#[test]
fn crate_root_api_surface_resolves() {
    let gt_path = fixtures_dir().join("gt.json");
    let coco = COCO::new(&gt_path).expect("Failed to load GT");
    let hierarchy = Hierarchy::from_categories(&coco.dataset.categories);
    let _expanded = hotcoco::detection::expand::expand_annotations(&coco, &hierarchy);

    let _: Option<hotcoco::COCOeval> = None;
    let _: Option<hotcoco::EvalImg> = None;
    let _: Option<hotcoco::AccumulatedEval> = None;
    let _: Option<hotcoco::EvalShape> = None;
    let _: Option<hotcoco::EvalMode> = None;
    let _: Option<hotcoco::EvalParams> = None;
    let _: Option<hotcoco::EvalResults> = None;
    let _: Option<hotcoco::ConfusionMatrix> = None;
    let _: Option<hotcoco::TideErrors> = None;
    let _: Option<hotcoco::SliceResult> = None;
    let _: Option<hotcoco::ComparisonResult> = None;
    let _: Option<hotcoco::BootstrapCI> = None;
    let _: Option<hotcoco::CalibrationBin> = None;
    let _: Option<hotcoco::Hierarchy> = None;
    let _: Option<hotcoco::HealthReport> = None;
    let _: Option<hotcoco::Finding> = None;
    let _: Option<hotcoco::Layer> = None;
    let _: Option<hotcoco::DatasetSummary> = None;
    let _: Option<hotcoco::SummaryStats> = None;
    let _: Option<hotcoco::CategoryStats> = None;
    let _: Option<hotcoco::DatasetStats> = None;
    let _: Option<hotcoco::EvalReport> = None;
    let _: Option<hotcoco::Provenance> = None;

    // The functional layer is reachable by its own path, not only through a driver.
    let ap = hotcoco::metrics::counts::average_precision(&[0.9], &[true], None, 1, &[0.0, 1.0]);
    assert!(ap.is_finite());
    let bins = hotcoco::metrics::calibration::calibration_curve(&[0.9], &[true], 4);
    assert_eq!(bins.len(), 4);
    let cm = hotcoco::metrics::confusion::confusion_matrix(&[Some(0)], &[Some(0)], 2);
    assert_eq!(cm.len(), 9);
    let (rows, _) = hotcoco::primitives::assign::lsap(&[1.0, 2.0, 3.0, 4.0], 2, 2, false);
    assert_eq!(rows.len(), 2);
}

// ---------------------------------------------------------------------------
// EvalReport
// ---------------------------------------------------------------------------

/// The standard bbox fixture pair, loaded but not evaluated.
///
/// Left unrun because the provenance tests below adjust `params` first, which
/// only means anything before `run()`. Callers that just want the numbers use
/// [`bbox_eval_on_fixtures`].
fn load_bbox_fixtures() -> COCOeval {
    let coco_gt = COCO::new(&fixtures_dir().join("gt.json")).expect("Failed to load GT");
    let coco_dt = coco_gt
        .load_res(&fixtures_dir().join("dt.json"))
        .expect("Failed to load DT");
    COCOeval::new(coco_gt, coco_dt, IouType::Bbox)
}

/// [`load_bbox_fixtures`] run to completion.
fn bbox_eval_on_fixtures() -> COCOeval {
    let mut ev = load_bbox_fixtures();
    ev.run();
    ev
}

/// `results()` is a projection of `report()`, so the two can never disagree
/// about a metric. Asserting it here is what makes the projection worth having.
#[test]
fn test_report_and_results_agree_on_metrics() {
    let ev = bbox_eval_on_fixtures();
    let report = ev.report().expect("report");
    let results = ev.results(true).expect("results");

    assert_eq!(report.metrics.len(), results.metrics.len());
    for (key, value) in &results.metrics {
        assert_eq!(
            report.metric(key),
            Some(*value),
            "metric {key} differs between report() and results()"
        );
    }

    let per_class = results.per_class.expect("per_class requested");
    assert_eq!(per_class.len(), report.per_class.len());
    for (name, ap) in &per_class {
        assert_eq!(report.per_class[name]["AP"], *ap, "per-class AP for {name}");
    }
}

/// The serialized shape of `EvalResults` is a compatibility contract — users
/// parse saved result files — so re-basing it on `EvalReport` must not move it.
#[test]
fn test_eval_results_json_shape_is_stable() {
    let ev = bbox_eval_on_fixtures();
    let parsed: serde_json::Value =
        serde_json::from_str(&ev.results(true).expect("results").to_json().expect("json"))
            .expect("valid json");

    let obj = parsed.as_object().expect("top level is an object");
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "hotcoco_version",
            "metrics",
            "params",
            "per_class",
            "provenance"
        ],
        "EvalResults gained or lost a top-level key"
    );

    // Provenance has to survive into the archived artifact, not just the live
    // process: this is the file users keep, diff, and come back to.
    assert_eq!(
        parsed["provenance"], "parity_verified",
        "default bbox params on the standard fixture are reference-comparable"
    );

    let mut param_keys: Vec<&str> = parsed["params"]
        .as_object()
        .expect("params object")
        .keys()
        .map(String::as_str)
        .collect();
    param_keys.sort_unstable();
    assert_eq!(
        param_keys,
        [
            "area_ranges",
            "eval_mode",
            "iou_thresholds",
            "iou_type",
            "kpt_oks_sigmas",
            "max_dets",
            "recall_thresholds",
            "reference_deviations",
            "use_cats"
        ],
        "EvalParams gained or lost a key"
    );

    // A parity-verified run archives an *empty* deviation list — present, so a
    // reader can tell "no deviations" from "not recorded".
    assert_eq!(
        parsed["params"]["reference_deviations"],
        serde_json::json!([]),
        "default bbox params must archive an empty deviation list"
    );

    // `per_class: None` must stay absent rather than serialize as null.
    let without: serde_json::Value =
        serde_json::from_str(&ev.results(false).expect("results").to_json().expect("json"))
            .expect("valid json");
    assert!(
        without
            .as_object()
            .expect("object")
            .get("per_class")
            .is_none(),
        "per_class must be omitted, not null, when not requested"
    );
}

/// bbox/segm/keypoints are checked against pycocotools; oriented boxes have no
/// reference implementation, so they must not claim to be benchmark-standard.
#[test]
fn test_report_provenance_marks_obb_as_extension() {
    let ev = bbox_eval_on_fixtures();
    let report = ev.report().expect("report");
    assert_eq!(report.task, "detection");
    assert_eq!(report.provenance, hotcoco::Provenance::ParityVerified);
    assert!(report.provenance.is_benchmark_standard());

    let coco_gt = COCO::new(&fixtures_dir().join("gt.json")).expect("GT");
    let coco_dt = coco_gt
        .load_res(&fixtures_dir().join("dt.json"))
        .expect("DT");
    let mut obb = COCOeval::new(coco_gt, coco_dt, IouType::Obb);
    obb.run();
    let obb_report = obb.report().expect("report");
    assert_eq!(obb_report.provenance, hotcoco::Provenance::Extension);
    assert!(
        !obb_report.provenance.is_benchmark_standard(),
        "OBB has no reference implementation and must not read as leaderboard-comparable"
    );
}

/// Curves carry the aggregate slice a chart draws — one PR curve per IoU
/// threshold plus the shared x-axis — not the full T*R*K*A*M tensor.
#[test]
fn test_report_curves_are_the_aggregate_slice() {
    let ev = bbox_eval_on_fixtures();
    let report = ev.report().expect("report");
    let acc = ev.accumulated().expect("accumulated");

    assert_eq!(
        report.curves.len(),
        ev.params.iou_thrs.len() + 1,
        "one curve per IoU threshold, plus rec_thrs"
    );
    let rec_thrs = &report.curves["rec_thrs"];
    assert_eq!(rec_thrs.len(), acc.shape.r);
    for thr in &ev.params.iou_thrs {
        let curve = &report.curves[&format!("pr@{thr:.2}")];
        assert_eq!(
            curve.len(),
            acc.shape.r,
            "curve is indexed by recall threshold"
        );
    }

    // Precision is a fraction or the -1 "no data" sentinel — never anything else.
    for (name, curve) in &report.curves {
        if name == "rec_thrs" {
            continue;
        }
        for &v in curve {
            assert!((0.0..=1.0).contains(&v) || v == -1.0, "{name} has {v}");
        }
    }
}

#[test]
fn test_report_requires_summarize() {
    let coco_gt = COCO::new(&fixtures_dir().join("gt.json")).expect("GT");
    let coco_dt = coco_gt
        .load_res(&fixtures_dir().join("dt.json"))
        .expect("DT");
    let ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    assert!(
        ev.report().is_err(),
        "report() before summarize() must error"
    );
}

// ---------------------------------------------------------------------------
// End-to-end invariants
//
// Properties that must hold for any dataset, checkable without a reference
// implementation. This is the only kind of check available on the surfaces
// `report()` marks `Provenance::Extension` — Open Images and oriented boxes have
// no reference to compare against, which is exactly why a wrong number there
// survived until an invariant was written down. `recall <= 1.0` is not a
// hypothetical: Open Images returned 4.0.
// ---------------------------------------------------------------------------

/// Run every fixture pair we have through an evaluation and assert the
/// dataset-independent properties of the result.
fn assert_eval_invariants(ev: &COCOeval, label: &str) {
    let acc = ev
        .accumulated()
        .unwrap_or_else(|| panic!("{label}: accumulate() must have been called"));
    let a_idx = ev.params.all_area_idx();

    for (i, &r) in acc.recall.iter().enumerate() {
        // -1.0 is "not computed for this configuration" and is not a low score.
        // Compare against it exactly: a `>= 0.0` guard would let a genuine sign
        // bug hide behind the sentinel.
        assert!(
            r == -1.0 || (0.0..=1.0).contains(&r),
            "{label}: recall {r} outside [0,1] at flat index {i}"
        );
    }

    for (i, &p) in acc.precision.iter().enumerate() {
        assert!(
            p == -1.0 || (0.0..=1.0).contains(&p),
            "{label}: precision {p} outside [0,1] at flat index {i}"
        );
    }

    // Max-detections caps are nested: a larger cap keeps a superset of each
    // image's detections while `num_gt` is unchanged, so recall cannot fall.
    // (Stated over the "all" area range, where every fixture has ground truth.)
    for t in 0..acc.shape.t {
        for k in 0..acc.shape.k {
            let at = |m: usize| acc.recall[acc.shape.recall_idx(t, k, a_idx, m)];
            for m in 1..acc.shape.m {
                let (prev, cur) = (at(m - 1), at(m));
                if prev == -1.0 || cur == -1.0 {
                    continue;
                }
                assert!(
                    cur >= prev - 1e-12,
                    "{label}: recall fell from {prev} to {cur} when raising maxDets \
                     (t={t}, k={k}, m={} -> {m})",
                    m - 1
                );
            }
        }
    }
}

#[test]
fn eval_invariants_hold_across_fixtures() {
    for (gt_name, dt_name) in [
        ("gt.json", "dt.json"),
        ("edge_gt.json", "edge_dt.json"),
        ("zero_gt.json", "zero_dt.json"),
    ] {
        let gt = COCO::new(&fixtures_dir().join(gt_name)).unwrap();
        let dt = gt.load_res(&fixtures_dir().join(dt_name)).unwrap();
        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.run();

        assert_eval_invariants(&ev, gt_name);

        // Summary metrics obey the same domain as the cells they average.
        for (i, &v) in ev.stats().unwrap().iter().enumerate() {
            assert!(
                v == -1.0 || (0.0..=1.0).contains(&v),
                "{gt_name}: stats[{i}] = {v} outside [0,1] and not the -1.0 sentinel"
            );
        }
    }
}

#[test]
fn eval_invariants_hold_for_open_images() {
    // Open Images has no reference implementation, so invariants are the only
    // check that reaches it. This is the configuration that returned recall 4.0.
    const ORDINARY_GT: [f64; 4] = [300.0, 300.0, 100.0, 100.0];
    const GROUP_OF_GT: [f64; 4] = [0.0, 0.0, 200.0, 200.0];

    let cats = vec![cat(1, "person")];
    let gt = dataset(
        vec![img(1)],
        cats.clone(),
        vec![ann(1, ORDINARY_GT), ann(2, GROUP_OF_GT).group_of()],
    );
    let dt = dataset(
        vec![img(1)],
        cats,
        vec![
            det(1, ORDINARY_GT, 0.9),
            det(2, [0.0, 0.0, 200.0, 200.0], 0.85),
            det(3, [10.0, 10.0, 190.0, 190.0], 0.8),
            det(4, [5.0, 5.0, 195.0, 195.0], 0.75),
        ],
    );

    let mut ev = COCOeval::new_oid(COCO::from_dataset(gt), COCO::from_dataset(dt), None);
    ev.run();

    assert_eval_invariants(&ev, "open images group-of");
}

// ---------------------------------------------------------------------------
// Comparability predicate
//
// `reference_deviations()` is the single fact behind both the summarize()
// warnings and `Provenance`. Only its OBB branch had a test; the rest were
// unverified, which is exactly the shape of defect this repo keeps producing —
// a guard that looks green and checks nothing. One test per branch, each
// asserting that a run which *should* be downgraded actually is.
// ---------------------------------------------------------------------------

/// Run to completion and report whether `report()` calls the result comparable.
fn provenance_of(mut ev: COCOeval) -> Provenance {
    ev.run();
    ev.report().expect("report() succeeds").provenance
}

#[test]
fn default_params_are_parity_verified() {
    // The control. Without this, every assertion below could pass because
    // `report()` downgrades unconditionally.
    assert_eq!(
        provenance_of(load_bbox_fixtures()),
        Provenance::ParityVerified
    );
}

/// `provenance()` is the accessor renderers read; `report()` is what everything
/// else reads. They must be the same bit.
///
/// Worth its own test because the failure is silent and asymmetric: renderers
/// never build a report, so a drift would put a parity-verified badge on an
/// extension run with every assertion above still green.
#[test]
fn provenance_accessor_agrees_with_report() {
    let mut verified = load_bbox_fixtures();
    verified.run();
    assert_eq!(
        verified.provenance(),
        verified.report().expect("report() succeeds").provenance,
        "accessor and report disagree on a default run"
    );
    assert_eq!(verified.provenance(), Provenance::ParityVerified);

    let mut extension = load_bbox_fixtures();
    extension.params.iou_thrs = vec![0.5, 0.75];
    extension.run();
    assert_eq!(
        extension.provenance(),
        extension.report().expect("report() succeeds").provenance,
        "accessor and report disagree on a custom-parameter run"
    );
    // The sharp case: still `IouType::Bbox` in plain COCO mode. A renderer that
    // derived provenance from the eval mode or geometry would call this verified.
    assert_eq!(extension.provenance(), Provenance::Extension);
}

#[test]
fn custom_iou_thrs_downgrade_to_extension() {
    let mut ev = load_bbox_fixtures();
    ev.params.iou_thrs = vec![0.5, 0.75];
    assert_eq!(provenance_of(ev), Provenance::Extension);
}

#[test]
fn custom_max_dets_downgrade_to_extension() {
    let mut ev = load_bbox_fixtures();
    ev.params.max_dets = vec![1, 10, 50];
    assert_eq!(provenance_of(ev), Provenance::Extension);
}

#[test]
fn custom_area_range_labels_downgrade_to_extension() {
    let mut ev = load_bbox_fixtures();
    for (i, ar) in ev.params.area_ranges.iter_mut().enumerate() {
        ar.label = format!("bucket{i}");
    }
    assert_eq!(provenance_of(ev), Provenance::Extension);
}

/// The gap the audit found: the Python `areaRng` setter preserves labels, so a
/// user redefining "small" changed APs/APm/APl while the label-only check saw
/// nothing and the run still claimed parity.
#[test]
fn custom_area_range_bounds_downgrade_even_with_default_labels() {
    let mut ev = load_bbox_fixtures();
    let labels_before: Vec<String> = ev
        .params
        .area_ranges
        .iter()
        .map(|ar| ar.label.clone())
        .collect();

    for ar in &mut ev.params.area_ranges {
        if ar.label == "small" {
            ar.range = [0.0, 100.0];
        }
    }

    let labels_after: Vec<String> = ev
        .params
        .area_ranges
        .iter()
        .map(|ar| ar.label.clone())
        .collect();
    assert_eq!(
        labels_before, labels_after,
        "the point of this test is that labels are unchanged"
    );

    assert_eq!(provenance_of(ev), Provenance::Extension);
}

#[test]
fn custom_rec_thrs_downgrade_to_extension() {
    let mut ev = load_bbox_fixtures();
    // The 11-point VOC grid instead of COCO's 101 points.
    ev.params.rec_thrs = (0..=10).map(|i| f64::from(i) / 10.0).collect();
    assert_eq!(provenance_of(ev), Provenance::Extension);
}

#[test]
fn class_agnostic_pooling_downgrades_to_extension() {
    let mut ev = load_bbox_fixtures();
    ev.params.use_cats = false;
    assert_eq!(provenance_of(ev), Provenance::Extension);
}

#[test]
fn custom_kpt_oks_sigmas_downgrade_to_extension() {
    // The bbox fixture is fine here: the guard is `iou_type == Keypoints &&
    // sigmas != default`, which does not consult the annotations.
    let gt = COCO::new(&fixtures_dir().join("gt.json")).unwrap();
    let dt = gt.load_res(&fixtures_dir().join("dt.json")).unwrap();
    let mut ev = COCOeval::new(gt, dt, IouType::Keypoints);
    ev.params.kpt_oks_sigmas = vec![0.05; ev.params.kpt_oks_sigmas.len()];
    assert_eq!(provenance_of(ev), Provenance::Extension);
}

#[test]
fn open_images_downgrades_to_extension() {
    let gt = COCO::new(&fixtures_dir().join("gt.json")).unwrap();
    let dt = gt.load_res(&fixtures_dir().join("dt.json")).unwrap();
    assert_eq!(
        provenance_of(COCOeval::new_oid(gt, dt, None)),
        Provenance::Extension
    );
}

/// Evaluation must be independent of the rayon thread count, bitwise.
///
/// It is today — verified across 1/2/4/8 threads on val2017 — because no float
/// accumulation runs in parallel: every `par_iter` either collects by index or
/// reduces `u64` counters, and the `f64` sums in `summarize` and `counts` walk a
/// deterministically-ordered slice. But that holds by discipline, not by a guard,
/// and `tests/architecture.rs` does not forbid a future `par_iter().sum::<f64>()`.
/// Float addition is not associative, so such a reduction would make AP depend on
/// how rayon happened to split the work.
#[test]
fn evaluation_is_independent_of_thread_count() {
    fn eval_on(threads: usize) -> Vec<f64> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(|| {
            let gt = COCO::new(&fixtures_dir().join("edge_gt.json")).unwrap();
            let dt = gt.load_res(&fixtures_dir().join("edge_dt.json")).unwrap();
            run_bbox_eval(gt, dt)
        })
    }

    let single = eval_on(1);
    for threads in [2usize, 3, 4, 8] {
        let many = eval_on(threads);
        assert_eq!(single.len(), many.len());
        for (i, (&a, &b)) in single.iter().zip(many.iter()).enumerate() {
            // Bitwise, not approximate: the claim is determinism, and a tolerance
            // would pass for exactly the parallel float reduction this forbids.
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "stats[{i}] differs between 1 thread ({a}) and {threads} threads ({b})"
            );
        }
    }
}

/// `tide_errors` must be independent of the rayon thread count, bitwise, even
/// with tied detection scores spanning multiple images and categories.
///
/// This guards the ordering invariant `tide.rs`'s `Classified::merge` and
/// `classify_detections` doc comments describe: rayon's fold/reduce always
/// merges a `self` covering the earlier contiguous range of `cells` with an
/// `other` covering the later range, which is what keeps
/// `CatData::rank_by_score_desc`'s stable sort tie-breaking equal to a
/// sequential pass no matter how many threads did the work. Every detection
/// below shares its score with another detection in the same category (0.5
/// within category 1, 0.7 within category 2), and the tied detections
/// disagree on match outcome (TP vs. the background FP at image 3 / image 4)
/// — so a tie-break that silently reordered under a different thread count
/// would move a TP relative to an FP and change the resulting AP outright,
/// not just perturb its last bit by chance. Category 1 also carries an
/// uncovered ground truth (image 7 has none of the tied detections, so it is
/// a genuine Miss), which exercises the `ApScratch` buffers the Miss fix
/// reuses.
#[test]
fn tide_errors_is_deterministic_across_thread_counts_with_tied_scores() {
    fn build_gt() -> COCO {
        COCO::from_dataset(dataset(
            (1..=7).map(img).collect(),
            vec![cat(1, "cat1"), cat(2, "cat2")],
            vec![
                ann(1, [0.0, 0.0, 50.0, 50.0]).in_img(1).in_cat(1),
                ann(2, [0.0, 0.0, 50.0, 50.0]).in_img(3).in_cat(1),
                ann(3, [0.0, 0.0, 50.0, 50.0]).in_img(5).in_cat(1),
                // No detection lands in image 7 at all — a genuine Miss.
                ann(4, [0.0, 0.0, 50.0, 50.0]).in_img(7).in_cat(1),
                ann(5, [0.0, 0.0, 50.0, 50.0]).in_img(2).in_cat(2),
                ann(6, [0.0, 0.0, 50.0, 50.0]).in_img(4).in_cat(2),
                ann(7, [0.0, 0.0, 50.0, 50.0]).in_img(6).in_cat(2),
            ],
        ))
    }

    fn build_dt() -> COCO {
        COCO::from_dataset(dataset(
            (1..=7).map(img).collect(),
            vec![cat(1, "cat1"), cat(2, "cat2")],
            vec![
                // Category 1, all tied at score 0.5: TP, FP (background), TP.
                det(101, [0.0, 0.0, 50.0, 50.0], 0.5).in_img(1).in_cat(1),
                det(102, [200.0, 200.0, 10.0, 10.0], 0.5)
                    .in_img(3)
                    .in_cat(1),
                det(103, [0.0, 0.0, 50.0, 50.0], 0.5).in_img(5).in_cat(1),
                // Category 2, all tied at score 0.7: TP, FP (background), TP.
                det(104, [0.0, 0.0, 50.0, 50.0], 0.7).in_img(2).in_cat(2),
                det(105, [300.0, 300.0, 10.0, 10.0], 0.7)
                    .in_img(4)
                    .in_cat(2),
                det(106, [0.0, 0.0, 50.0, 50.0], 0.7).in_img(6).in_cat(2),
            ],
        ))
    }

    fn tide_on(threads: usize) -> hotcoco::TideErrors {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        pool.install(|| {
            let mut ev = COCOeval::new(build_gt(), build_dt(), IouType::Bbox);
            ev.evaluate();
            ev.tide_errors(0.5, 0.1).expect("tide_errors failed")
        })
    }

    let baseline = tide_on(1);
    for threads in [2usize, 4] {
        let many = tide_on(threads);

        assert_eq!(
            baseline.ap_base.to_bits(),
            many.ap_base.to_bits(),
            "ap_base differs between 1 thread ({}) and {threads} threads ({})",
            baseline.ap_base,
            many.ap_base
        );

        assert_eq!(
            baseline.delta_ap.keys().collect::<Vec<_>>(),
            many.delta_ap.keys().collect::<Vec<_>>(),
            "delta_ap keys differ between 1 thread and {threads} threads"
        );
        for (key, &a) in &baseline.delta_ap {
            let b = many.delta_ap[key];
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "delta_ap[{key}] differs between 1 thread ({a}) and {threads} threads ({b})"
            );
        }

        assert_eq!(
            baseline.counts, many.counts,
            "counts differ between 1 thread and {threads} threads"
        );
    }
}

/// A NaN detection score is rejected rather than silently corrupting the ranking.
///
/// Every ranking path sorts with `partial_cmp(..).unwrap_or(Equal)`, a comparator
/// that is not transitive once NaN is present: the sort does not panic, it
/// produces an arbitrary order, and AP becomes a function of the sort
/// implementation rather than of the detections.
///
/// Only reachable programmatically. Loading from a *file* never gets here —
/// `sanitize_non_finite` rewrites bare `NaN` to `null` first, which is the
/// pycocotools-compatible behavior and is covered by
/// `test_load_gt_tolerates_non_finite_floats`.
#[test]
fn nan_detection_score_is_rejected() {
    let cats = vec![cat(1, "person")];
    let gt = dataset(
        vec![img(1)],
        cats.clone(),
        vec![ann(1, [10.0, 10.0, 50.0, 50.0])],
    );
    let coco_gt = COCO::from_dataset(gt);

    let mut bad = det(1, [10.0, 10.0, 50.0, 50.0], 0.9);
    bad.score = Some(f64::NAN);

    // `COCO` is not Debug, so match on the Result rather than using expect_err.
    let msg = match coco_gt.load_res_anns(vec![bad]) {
        Ok(_) => panic!("a NaN score must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("NaN score"),
        "error should name the problem, got: {msg}"
    );

    // A finite score on the same path still works.
    assert!(
        coco_gt
            .load_res_anns(vec![det(1, [10.0, 10.0, 50.0, 50.0], 0.9)])
            .is_ok(),
        "finite scores must still load"
    );
}

/// The five FP error types partition every unmatched, non-ignored detection.
///
/// `classify_fp` is total — it returns one of Cls/Loc/Both/Dupe/Bkg for every
/// false positive — so their counts must sum to exactly the number of FPs, and
/// `counts` must hold no sixth key. That second half matters because `counts` is a
/// `HashMap<String, u64>` keyed by a stringified enum: a typo'd or renamed key
/// would silently create a new bucket rather than fail to compile, and every
/// existing test asserts individual counts, which cannot see an extra one.
#[test]
fn tide_fp_types_partition_the_false_positives() {
    const FP_TYPES: [&str; 5] = ["Cls", "Loc", "Both", "Dupe", "Bkg"];

    let gt = COCO::new(&fixtures_dir().join("edge_gt.json")).unwrap();
    let dt = gt.load_res(&fixtures_dir().join("edge_dt.json")).unwrap();
    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    ev.evaluate();
    let te = ev.tide_errors(0.5, 0.1).expect("tide_errors");

    // No key outside the documented set. `Miss` is a false *negative* and is
    // counted separately, so it is allowed but not part of the FP partition.
    const ALL_KEYS: [&str; 6] = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"];
    for key in te.counts.keys() {
        assert!(
            ALL_KEYS.contains(&key.as_str()),
            "unexpected error-type key {key:?} in counts; the enum and the map have drifted"
        );
    }

    // Count the detections that are genuinely false positives at this threshold:
    // not matched, not ignored. OID's group-of absorption and LVIS's
    // not-exhaustive rule both work by setting `dt_ignore`, so this stays correct
    // for those modes too.
    let t_idx = ev
        .params
        .iou_thr_idx(0.5)
        .expect("0.5 is in the default grid");
    let target_area = ev.params.area_ranges[ev.params.all_area_idx()].range;

    let mut expected_fps = 0u64;
    for e in ev.eval_imgs().iter().flatten() {
        if e.area_rng != target_area {
            continue;
        }
        for d in 0..e.dt_ids.len() {
            if !e.dt_matched[(t_idx, d)] && !e.dt_ignore[(t_idx, d)] {
                expected_fps += 1;
            }
        }
    }

    let classified: u64 = FP_TYPES
        .iter()
        .map(|k| te.counts.get(*k).copied().unwrap_or(0))
        .sum();

    assert_eq!(
        classified, expected_fps,
        "FP types sum to {classified} but there are {expected_fps} unmatched, \
         non-ignored detections; counts = {:?}",
        te.counts
    );

    // `FP` and `FN` are tidecv's special oracles. Both are pure improvements —
    // suppressing FPs can only raise precision, shrinking the denominator can
    // only raise recall — so neither delta may be negative. And since `FP`
    // suppresses *every* false positive, it dominates each per-type fix that is
    // itself a pure suppression (Bkg/Both/Dupe; not Loc/Cls, whose fixes gain
    // recall by flipping to TP).
    let fp = te.delta_ap["FP"];
    let fnv = te.delta_ap["FN"];
    assert!(fp >= 0.0, "FP oracle must not lower AP, got {fp}");
    assert!(fnv >= 0.0, "FN oracle must not lower AP, got {fnv}");
    for k in ["Bkg", "Both", "Dupe"] {
        let per_type = te.delta_ap[k];
        assert!(
            fp >= per_type - 1e-12,
            "suppressing all FPs must dominate suppressing only {k}: {fp} < {per_type}"
        );
    }
}

/// `max_dets` order must not change any number.
///
/// Before `Params::max_det()` owned the cap, `evaluate()` stamped eval_imgs
/// with `max_dets.last()` while `image_diagnostics` filtered on the maximum:
/// with `[100, 10, 1]` the filter matched nothing and diagnostics came back
/// empty. pycocotools sorts `maxDets` in-place; hotcoco keeps the caller's
/// order on the M axis, so equality here is by value, not by index.
#[test]
fn test_max_dets_order_is_irrelevant() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");

    let run = |max_dets: Vec<usize>| {
        let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
        let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");
        let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
        ev.params.max_dets = max_dets;
        ev.evaluate();
        ev.accumulate();
        ev.summarize();
        ev
    };

    let sorted = run(vec![1, 10, 100]);
    let unsorted = run(vec![100, 10, 1]);

    // Summary metrics look up the M axis by value, so the stats vectors must
    // be identical element-for-element.
    assert_eq!(
        sorted.stats().expect("summarize sets stats"),
        unsorted.stats().expect("summarize sets stats"),
        "stats must not depend on max_dets order"
    );

    // The diagnostics filter is the site that used to disagree with the
    // evaluate() cap: it must find eval_imgs, not an empty intersection.
    let diag_sorted = sorted
        .image_diagnostics(0.5, 0.5)
        .expect("diagnostics on sorted max_dets");
    let diag_unsorted = unsorted
        .image_diagnostics(0.5, 0.5)
        .expect("diagnostics on unsorted max_dets");
    assert!(
        !diag_unsorted.images.is_empty(),
        "diagnostics must see eval_imgs regardless of max_dets order"
    );
    assert_eq!(
        diag_sorted.images.len(),
        diag_unsorted.images.len(),
        "diagnostics coverage must not depend on max_dets order"
    );
}

/// GT annotations feed the matcher in JSON array order, exactly as pycocotools
/// builds `_gts` — the index must not re-sort them by id.
///
/// The order is observable: the greedy scan takes the *later* GT on an exact
/// IoU tie (`>=`, same as pycocotools), so two identical boxes whose ids are
/// reversed relative to array order pick opposite winners under the two
/// orderings. Official COCO files are id-ordered, which is why no parity run
/// can see this; converted or merged files are where it bites.
#[test]
fn test_gt_annotations_keep_json_array_order() {
    let gt_json = r#"{
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "thing"}],
        "annotations": [
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
        ]
    }"#;
    let dt_json = r#"[
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}
    ]"#;

    let dir = tempfile::tempdir().expect("create temp dir");
    let gt_path = dir.path().join("gt.json");
    let dt_path = dir.path().join("dt.json");
    std::fs::write(&gt_path, gt_json).expect("write GT fixture");
    std::fs::write(&dt_path, dt_json).expect("write DT fixture");

    let coco_gt = COCO::new(&gt_path).expect("load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("load DT");
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();

    let all_idx = ev.params.all_area_idx();
    let all_rng = ev.params.area_ranges[all_idx].range;
    let cell = ev
        .eval_imgs()
        .iter()
        .flatten()
        .find(|e| e.area_rng == all_rng)
        .expect("one populated cell at area=all");

    // Array order [2, 1], not id order [1, 2].
    assert_eq!(
        cell.gt_ids,
        vec![2, 1],
        "GT ids must stay in JSON array order"
    );
    // On an exact IoU tie the later GT in array order wins — id 1 here.
    // Under id-sorted order the winner flips to id 2.
    assert!(cell.dt_matched[(0, 0)], "detection must match at IoU 0.5");
    assert_eq!(
        cell.dt_matches[(0, 0)],
        1,
        "tie must resolve to the later GT in array order, as pycocotools does"
    );
}

/// Per-class AP, the F-scores and `report()`'s PR curves must read the M-axis
/// slot holding `max_det()`, not the last one.
///
/// `Params::max_dets` is the caller's list and hotcoco does not sort it (the
/// accumulated M axis follows the caller's order), so `[100, 10, 1]` puts the cap
/// at slot 0 while `[1, 10, 100]` puts it at slot 2. Three sites took
/// `shape.m - 1` regardless, which meant a single `report()` disagreed with
/// itself: the headline `AP` was computed at `maxDets = 100` and every per-class
/// `AP` beside it at `maxDets = 1`.
///
/// The scenario needs more than one detection per image per class, or the cap
/// makes no difference and the test passes against the defect.
#[test]
fn per_class_metrics_follow_max_det_not_the_last_slot() {
    let gt_ds = dataset(
        vec![img(1)],
        vec![cat(1, "a"), cat(2, "b")],
        vec![
            ann(1, [0.0, 0.0, 10.0, 10.0]),
            ann(2, [20.0, 0.0, 10.0, 10.0]),
            ann(3, [0.0, 20.0, 10.0, 10.0]).in_cat(2),
            ann(4, [20.0, 20.0, 10.0, 10.0]).in_cat(2),
        ],
    );
    // Two true positives per class. At maxDets = 1 only the top-scoring one
    // survives, halving recall and therefore AP.
    let dets = vec![
        det(1, [0.0, 0.0, 10.0, 10.0], 0.9),
        det(2, [20.0, 0.0, 10.0, 10.0], 0.8),
        det(3, [0.0, 20.0, 10.0, 10.0], 0.7).in_cat(2),
        det(4, [20.0, 20.0, 10.0, 10.0], 0.6).in_cat(2),
    ];

    let run = |max_dets: Vec<usize>| {
        let gt = COCO::from_dataset(gt_ds.clone());
        let dt = gt.load_res_anns(dets.clone()).unwrap();
        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.params.max_dets = max_dets;
        ev.run();
        let report = ev.report().unwrap();
        let per_class: Vec<f64> = report
            .per_class
            .values()
            .filter_map(|m| m.get("AP").copied())
            .collect();
        let f1 = ev.f_scores(1.0).get("F1").copied().unwrap();
        let curve = report.curves["pr@0.50"].clone();
        (report.metrics["AP"], per_class, f1, curve)
    };

    let (ap_sorted, per_class_sorted, f1_sorted, curve_sorted) = run(vec![1, 10, 100]);
    let (ap_unsorted, per_class_unsorted, f1_unsorted, curve_unsorted) = run(vec![100, 10, 1]);

    // The cap is the same value either way, so every one of these is the same
    // number — the ordering of `max_dets` is not a metric input.
    assert_eq!(
        ap_sorted, ap_unsorted,
        "headline AP must not depend on order"
    );
    assert_eq!(
        per_class_sorted, per_class_unsorted,
        "per-class AP diverged"
    );
    assert_eq!(f1_sorted, f1_unsorted, "F1 diverged");
    assert_eq!(curve_sorted, curve_unsorted, "PR curve diverged");

    // And the per-class values must agree with the headline they sit beside,
    // which is what a reader compares them against.
    assert!(
        !per_class_unsorted.is_empty(),
        "no per-class AP was reported"
    );
    for ap in &per_class_unsorted {
        assert!(
            (ap - ap_unsorted).abs() < 1e-12,
            "per-class AP {ap} disagrees with headline AP {ap_unsorted}"
        );
    }
}

/// `metric_defs()` is the catalog `metric_keys()` and `stats()` are projections
/// of, so all three must stay index-parallel. Renderers read `defs[i]` to label
/// `stats[i]`; if the two lists could differ in length or order, every label
/// would be one row off with nothing to catch it.
#[test]
fn metric_defs_align_with_metric_keys_and_stats() {
    let gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "a")],
        vec![ann(1, [0.0, 0.0, 10.0, 10.0])],
    ));
    let dt = gt
        .load_res_anns(vec![det(1, [0.0, 0.0, 10.0, 10.0], 0.9)])
        .unwrap();
    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    ev.run();

    let defs = ev.metric_defs();
    let keys = ev.metric_keys();
    let stats = ev.stats().unwrap();

    assert_eq!(defs.len(), keys.len());
    assert_eq!(defs.len(), stats.len());
    for (d, &k) in defs.iter().zip(&keys) {
        assert_eq!(d.name, k);
    }

    // The fields a renderer needs are readable, and describe the row they label.
    let ap50 = defs.iter().find(|d| d.name == "AP50").unwrap();
    assert!(ap50.ap);
    assert_eq!(ap50.iou_thr, Some(0.5));
    assert_eq!(ap50.area_lbl, "all");
    assert_eq!(ap50.max_det, 100);
    assert!(ap50.freq_group.is_none());

    // LVIS is the mode that populates the frequency axis.
    let gt = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "a")],
        vec![ann(1, [0.0, 0.0, 10.0, 10.0])],
    ));
    let dt = gt.load_res_anns(vec![]).unwrap();
    let lvis = COCOeval::new_lvis(gt, dt, IouType::Bbox);
    let apr = lvis
        .metric_defs()
        .into_iter()
        .find(|d| d.name == "APr")
        .unwrap();
    assert_eq!(apr.freq_group, Some(hotcoco::FreqGroup::Rare));
}

// ---------------------------------------------------------------------------
// Open Images CSV conversion
// ---------------------------------------------------------------------------

/// Write `contents` to a file inside `dir` and return its path.
fn write_csv(dir: &std::path::Path, name: &str, contents: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write csv");
    path
}

/// Image dimensions keyed by Open Images image ID.
fn oid_dims(entries: &[(&str, (u32, u32))]) -> HashMap<String, (u32, u32)> {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect()
}

#[test]
fn test_oid_to_coco_column_order() {
    // Open Images orders columns XMin,XMax,YMin,YMax — XMax before YMin. Every
    // number below is distinct so a transposed read cannot coincidentally pass.
    let tmp = tempfile::tempdir().expect("tempdir");
    let csv = write_csv(
        tmp.path(),
        "boxes.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf\n\
         abc123,/m/01,0.1,0.5,0.3,0.4,0\n",
    );

    let dims = oid_dims(&[("abc123", (1000, 500))]);
    let ds = oid_to_coco(&csv, None, &dims).expect("oid_to_coco");

    assert_eq!(ds.images.len(), 1);
    assert_eq!(ds.annotations.len(), 1);
    let bbox = ds.annotations[0].bbox.expect("bbox");
    // x = 0.1*1000, y = 0.3*500, w = (0.5-0.1)*1000, h = (0.4-0.3)*500
    assert!((bbox[0] - 100.0).abs() < 1e-6, "x: {}", bbox[0]);
    assert!((bbox[1] - 150.0).abs() < 1e-6, "y: {}", bbox[1]);
    assert!((bbox[2] - 400.0).abs() < 1e-6, "w: {}", bbox[2]);
    assert!((bbox[3] - 50.0).abs() < 1e-6, "h: {}", bbox[3]);
}

#[test]
fn test_oid_header_drives_parsing_not_position() {
    // The same row under the full V6 layout and under a deliberately shuffled
    // header must produce identical boxes. This is the guard that makes the
    // XMin,XMax,YMin,YMax ordering trap unreachable.
    let tmp = tempfile::tempdir().expect("tempdir");
    let dims = oid_dims(&[("abc123", (1000, 500))]);

    let v6 = write_csv(
        tmp.path(),
        "v6.csv",
        "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,\
         IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n\
         abc123,xclick,/m/01,1,0.1,0.5,0.3,0.4,0,0,0,0,0\n",
    );
    let shuffled = write_csv(
        tmp.path(),
        "shuffled.csv",
        "IsGroupOf,YMax,LabelName,XMin,ImageID,YMin,XMax\n\
         0,0.4,/m/01,0.1,abc123,0.3,0.5\n",
    );

    let a = oid_to_coco(&v6, None, &dims).expect("v6");
    let b = oid_to_coco(&shuffled, None, &dims).expect("shuffled");
    assert_eq!(a.annotations[0].bbox, b.annotations[0].bbox);

    // The V6 `Confidence` column is ground-truth provenance, always 1 — it must
    // not be mistaken for a detection score.
    assert_eq!(a.annotations[0].score, None);
}

#[test]
fn test_oid_missing_required_column_errors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let csv = write_csv(
        tmp.path(),
        "bad.csv",
        "ImageID,LabelName,XMin,YMin,YMax\nabc,/m/01,0.1,0.3,0.4\n",
    );
    let err = oid_to_coco(&csv, None, &HashMap::new()).expect_err("must reject missing XMax");
    assert!(err.to_string().contains("xmax"), "error was: {err}");
}

#[test]
fn test_oid_group_of_becomes_is_group_of() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let csv = write_csv(
        tmp.path(),
        "boxes.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf\n\
         img1,/m/01,0.0,0.5,0.0,0.5,1\n\
         img1,/m/01,0.5,1.0,0.5,1.0,0\n",
    );

    let ds = oid_to_coco(&csv, None, &HashMap::new()).expect("oid_to_coco");
    assert_eq!(ds.annotations[0].is_group_of, Some(true));
    assert_eq!(ds.annotations[1].is_group_of, Some(false));
    // group-of is Open Images' own flag, distinct from COCO's iscrowd.
    assert!(!ds.annotations[0].iscrowd);
}

#[test]
fn test_oid_class_descriptions_resolve_mids() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let csv = write_csv(
        tmp.path(),
        "boxes.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax\nimg1,/m/0cmf2,0.1,0.2,0.1,0.2\n",
    );
    // Headerless, and one name carries a comma inside quotes.
    let desc = write_csv(
        tmp.path(),
        "class-descriptions-boxable.csv",
        "/m/0cmf2,Beer\n/m/0dv5r,\"Camera, still\"\n",
    );

    let ds = oid_to_coco(&csv, Some(&desc), &HashMap::new()).expect("oid_to_coco");
    assert_eq!(ds.categories.len(), 1);
    assert_eq!(ds.categories[0].name, "Beer");

    // Without the map, the MID stands in as the name.
    let plain = oid_to_coco(&csv, None, &HashMap::new()).expect("oid_to_coco");
    assert_eq!(plain.categories[0].name, "/m/0cmf2");
}

#[test]
fn test_oid_without_dims_keeps_normalized_coords() {
    // No image sizes available: boxes stay in [0,1] against a 1x1 image. IoU and
    // IoA are ratios of areas scaled identically on both axes, so Open Images AP
    // is unchanged — only absolute areas lose meaning.
    let tmp = tempfile::tempdir().expect("tempdir");
    let csv = write_csv(
        tmp.path(),
        "boxes.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax\nimg1,/m/01,0.25,0.75,0.25,0.75\n",
    );

    let ds = oid_to_coco(&csv, None, &HashMap::new()).expect("oid_to_coco");
    assert_eq!(ds.images[0].width, 1);
    assert_eq!(ds.images[0].height, 1);
    let bbox = ds.annotations[0].bbox.expect("bbox");
    assert!((bbox[2] - 0.5).abs() < 1e-9, "w: {}", bbox[2]);
    assert!((bbox[3] - 0.5).abs() < 1e-9, "h: {}", bbox[3]);
}

#[test]
fn test_oid_ids_are_deterministic() {
    // IDs come from sorted distinct values, not file order, so two orderings of
    // the same rows produce the same dataset.
    let tmp = tempfile::tempdir().expect("tempdir");
    let header = "ImageID,LabelName,XMin,XMax,YMin,YMax\n";
    let row_a = "aaa,/m/02,0.1,0.2,0.1,0.2\n";
    let row_b = "bbb,/m/01,0.3,0.4,0.3,0.4\n";

    let fwd = write_csv(tmp.path(), "fwd.csv", &format!("{header}{row_a}{row_b}"));
    let rev = write_csv(tmp.path(), "rev.csv", &format!("{header}{row_b}{row_a}"));

    let a = oid_to_coco(&fwd, None, &HashMap::new()).expect("fwd");
    let b = oid_to_coco(&rev, None, &HashMap::new()).expect("rev");

    let names = |ds: &hotcoco::Dataset| -> Vec<(u64, String)> {
        ds.categories
            .iter()
            .map(|c| (c.id, c.name.clone()))
            .collect()
    };
    assert_eq!(names(&a), names(&b));
    let files = |ds: &hotcoco::Dataset| -> Vec<(u64, String)> {
        ds.images
            .iter()
            .map(|i| (i.id, i.file_name.clone()))
            .collect()
    };
    assert_eq!(files(&a), files(&b));
}

#[test]
fn test_oid_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let csv = write_csv(
        tmp.path(),
        "boxes.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf\n\
         img1,/m/01,0.100000,0.500000,0.300000,0.400000,1\n\
         img2,/m/02,0.200000,0.600000,0.100000,0.900000,0\n",
    );
    let dims = oid_dims(&[("img1", (1000, 500)), ("img2", (640, 480))]);

    let ds = oid_to_coco(&csv, None, &dims).expect("oid_to_coco");
    let out = tmp.path().join("out.csv");
    let stats = coco_to_oid(&ds, &out).expect("coco_to_oid");

    assert_eq!(stats.images, 2);
    assert_eq!(stats.annotations, 2);
    assert_eq!(stats.group_of, 1);
    assert_eq!(stats.skipped_no_bbox, 0);

    let back = oid_to_coco(&out, None, &dims).expect("re-import");
    // 6-decimal normalized CSV coords land within 1e-3 of a pixel here.
    assert_geometry_round_trip(&ds, &back, file_name_key, bbox_of, [1e-3; 4]);
    // The group-of flag is OID's own semantics on top of the geometry.
    for (before, after) in ds.annotations.iter().zip(&back.annotations) {
        assert_eq!(before.is_group_of, after.is_group_of);
    }
}

#[test]
fn test_oid_round_trips_category_names_containing_commas() {
    // Open Images' own descriptions include names like "Camera, still". Written
    // bare, that name adds a field to the row and re-imports as "Camera" — a
    // corruption that looks like a plausible category rather than an error.
    let tmp = tempfile::tempdir().expect("tempdir");
    let csv = write_csv(
        tmp.path(),
        "boxes.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax\nimg1,/m/0dv5r,0.1,0.5,0.3,0.4\n",
    );
    let desc = write_csv(tmp.path(), "desc.csv", "/m/0dv5r,\"Camera, still\"\n");

    let ds = oid_to_coco(&csv, Some(&desc), &HashMap::new()).expect("oid_to_coco");
    assert_eq!(ds.categories[0].name, "Camera, still");

    let out = tmp.path().join("out.csv");
    coco_to_oid(&ds, &out).expect("coco_to_oid");

    let back = oid_to_coco(&out, None, &HashMap::new()).expect("re-import");
    assert_eq!(back.categories.len(), 1);
    assert_eq!(back.categories[0].name, "Camera, still");
    assert_eq!(back.annotations.len(), 1);
    let bbox = back.annotations[0].bbox.expect("bbox");
    assert!(
        (bbox[2] - 0.4).abs() < 1e-5,
        "w survived quoting: {}",
        bbox[2]
    );
}

#[test]
fn test_oid_results_align_with_ground_truth_ids() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let gt_csv = write_csv(
        tmp.path(),
        "gt.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf\n\
         zzz,/m/01,0.1,0.5,0.3,0.4,0\n\
         aaa,/m/02,0.1,0.5,0.3,0.4,0\n",
    );
    let dims = oid_dims(&[("zzz", (1000, 500)), ("aaa", (1000, 500))]);
    let gt = oid_to_coco(&gt_csv, None, &dims).expect("gt");

    let dt_csv = write_csv(
        tmp.path(),
        "dt.csv",
        "ImageID,LabelName,Score,XMin,XMax,YMin,YMax\nzzz,/m/01,0.9,0.1,0.5,0.3,0.4\n",
    );
    let anns = oid_results_to_anns(&gt, &dt_csv, None).expect("results");

    assert_eq!(anns.len(), 1);
    // "zzz" sorts after "aaa", so a detection reader that numbered images in file
    // order would attach this to image 1 instead of image 2.
    let gt_img = gt
        .images
        .iter()
        .find(|i| i.file_name == "zzz")
        .expect("gt image");
    assert_eq!(anns[0].image_id, gt_img.id);
    assert_eq!(anns[0].score, Some(0.9));
    let bbox = anns[0].bbox.expect("bbox");
    assert!((bbox[0] - 100.0).abs() < 1e-6, "x: {}", bbox[0]);
    assert!((bbox[3] - 50.0).abs() < 1e-6, "h: {}", bbox[3]);
}

#[test]
fn test_oid_results_reject_unknown_references() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let gt_csv = write_csv(
        tmp.path(),
        "gt.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax\nimg1,/m/01,0.1,0.5,0.3,0.4\n",
    );
    let gt = oid_to_coco(&gt_csv, None, &HashMap::new()).expect("gt");

    let bad_img = write_csv(
        tmp.path(),
        "bad_img.csv",
        "ImageID,LabelName,Score,XMin,XMax,YMin,YMax\nnope,/m/01,0.9,0.1,0.5,0.3,0.4\n",
    );
    let err = oid_results_to_anns(&gt, &bad_img, None).expect_err("unknown ImageID");
    assert!(err.to_string().contains("nope"), "error was: {err}");

    let bad_cat = write_csv(
        tmp.path(),
        "bad_cat.csv",
        "ImageID,LabelName,Score,XMin,XMax,YMin,YMax\nimg1,/m/99,0.9,0.1,0.5,0.3,0.4\n",
    );
    let err = oid_results_to_anns(&gt, &bad_cat, None).expect_err("unknown LabelName");
    assert!(err.to_string().contains("/m/99"), "error was: {err}");
}

#[test]
fn test_oid_end_to_end_open_images_eval() {
    // The point of the converter: CSV in, Open Images AP out, without the caller
    // writing a parser.
    let tmp = tempfile::tempdir().expect("tempdir");
    let gt_csv = write_csv(
        tmp.path(),
        "gt.csv",
        "ImageID,LabelName,XMin,XMax,YMin,YMax,IsGroupOf\n\
         img1,/m/01,0.1,0.5,0.1,0.5,0\n\
         img2,/m/01,0.2,0.6,0.2,0.6,0\n",
    );
    let dims = oid_dims(&[("img1", (640, 480)), ("img2", (640, 480))]);
    let gt = oid_to_coco(&gt_csv, None, &dims).expect("gt");

    let dt_csv = write_csv(
        tmp.path(),
        "dt.csv",
        "ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n\
         img1,/m/01,0.9,0.1,0.5,0.1,0.5\n\
         img2,/m/01,0.8,0.2,0.6,0.2,0.6\n",
    );
    let dt_anns = oid_results_to_anns(&gt, &dt_csv, None).expect("dt");

    let coco_gt = COCO::from_dataset(gt);
    let coco_dt = coco_gt.load_res_anns(dt_anns).expect("load_res_anns");

    let mut ev = COCOeval::new_oid(coco_gt, coco_dt, None);
    ev.run();
    let stats = ev.stats().expect("stats after run");
    assert!(
        stats[0] > 0.99,
        "perfect detections should score ~1.0: {stats:?}"
    );
}

// ---------------------------------------------------------------------------
// Segmentation (mask IoU) and keypoints (OKS) end-to-end evaluation
//
// Until these tests, `IouType::Segm` appeared zero times in this suite and
// `Keypoints` once (a provenance check) — `cargo test` never ran mask IoU or
// OKS through evaluate → accumulate → summarize, and real-data parity is
// local-only, so CI stayed green through a segm/keypoints regression. The
// fixtures are hand-computable: rectangle masks (exact pixel counts) and
// single-keypoint OKS values derived in the comments.
// ---------------------------------------------------------------------------

/// Shared P/R arithmetic for the two-detection fixtures below.
///
/// Both the segm and keypoints tests stage the same shape: two GTs, DT1 a
/// perfect match (similarity 1.0, score 0.9) and DT2 clearing only the 0.50
/// threshold (score 0.8). Per IoU threshold on the 0.50:0.05:0.95 grid:
///
/// - t = 0.50: both DTs are TPs. The PR curve reaches (recall 1.0,
///   precision 1.0) → 101-point AP = 1.0, recall = 1.0.
/// - t ≥ 0.55 (9 thresholds): DT1 TP, DT2 FP, ranked [TP@0.9, FP@0.8].
///   Recall stops at 0.5 with precision 1.0 there; the interpolated precision
///   envelope is 1.0 for the 51 recall grid points ≤ 0.50 and 0 above
///   → AP = 51/101, recall = 0.5.
///
/// So AP = (101 + 9·51)/1010 = 56/101 ≈ 0.554455, AP50 = 1.0,
/// AP75 = 51/101 ≈ 0.504950, and mean recall over the sweep
/// = (1.0 + 9·0.5)/10 = 0.55.
const AP_FULL_SWEEP: f64 = (101.0 + 9.0 * 51.0) / 1010.0;
const AP_AT_075: f64 = 51.0 / 101.0;
const AR_FULL_SWEEP: f64 = 0.55;

/// Assert a stats vector against hand-derived expectations, treating the
/// `-1.0` "not computed" sentinel as exact — a sentinel that arrives as a
/// nearby score, or vice versa, must fail even inside the tolerance.
fn assert_stats(stats: &[f64], expected: &[f64], names: &[&str]) {
    assert_eq!(stats.len(), expected.len(), "stats length");
    for (i, (&got, &exp)) in stats.iter().zip(expected.iter()).enumerate() {
        let name = names[i];
        if exp == -1.0 {
            assert_eq!(got, -1.0, "{name}: expected the -1.0 sentinel, got {got}");
        } else {
            assert!(
                (got - exp).abs() < 1e-9,
                "{name}: got {got:.9}, expected {exp:.9}"
            );
            assert_ne!(got, -1.0, "{name}: real score expected, got the sentinel");
        }
    }
}

const BBOX_SEGM_KEYS: [&str; 12] = [
    "AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
];

/// Segm end-to-end over pixel-exact rectangle masks.
///
/// Geometry (640×640 image, uncompressed RLEs from [`rect_mask`]):
/// - GT1 mask `[0,0,20,20]` (400 px); DT1 mask identical, score 0.9 → IoU 1.0.
/// - GT2 mask `[50,50,20,20]` (400 px); DT2 mask `[50,50,10,20]` (200 px),
///   score 0.8 → inter = 10·20 = 200, union = 400 + 200 − 200 = 400,
///   IoU = exactly 0.5 → matches at t = 0.50 only.
///
/// DT2's *bbox* is identical to GT2's (IoU 1.0), so a regression that silently
/// scored segm with bbox IoU reports AP = 1.0 here and fails.
///
/// Expected stats follow [`AP_FULL_SWEEP`]'s derivation. All GT areas are
/// 400 px² < 32² → "small" equals "all"; medium/large have no ground truth →
/// the −1.0 sentinel. AR@1 keeps only DT1 (top score), a TP at every
/// threshold over 2 GTs → 0.5.
#[test]
fn test_segm_eval_end_to_end() {
    let gt = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            ann(1, [0.0, 0.0, 20.0, 20.0]).mask(rect_mask(640, 640, 0, 0, 20, 20)),
            ann(2, [50.0, 50.0, 20.0, 20.0]).mask(rect_mask(640, 640, 50, 50, 20, 20)),
        ],
    );
    let dt = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            det(101, [0.0, 0.0, 20.0, 20.0], 0.9).mask(rect_mask(640, 640, 0, 0, 20, 20)),
            // bbox deliberately equals GT2's; only the mask is half-width.
            det(102, [50.0, 50.0, 20.0, 20.0], 0.8)
                .mask(rect_mask(640, 640, 50, 50, 10, 20))
                .with_area(200.0),
        ],
    );

    let mut ev = COCOeval::new(
        COCO::from_dataset(gt),
        COCO::from_dataset(dt),
        IouType::Segm,
    );
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().expect("summarize sets stats");
    let expected = [
        AP_FULL_SWEEP, // AP
        1.0,           // AP50 — both DTs match at t=0.50
        AP_AT_075,     // AP75 — DT2 (IoU 0.5) misses
        AP_FULL_SWEEP, // APs — every GT is small, so small == all
        -1.0,          // APm — no medium GT
        -1.0,          // APl — no large GT
        0.5,           // AR1 — top-1 is DT1, a TP at every t, over 2 GTs
        AR_FULL_SWEEP, // AR10
        AR_FULL_SWEEP, // AR100
        AR_FULL_SWEEP, // ARs
        -1.0,          // ARm
        -1.0,          // ARl
    ];
    assert_stats(stats, &expected, &BBOX_SEGM_KEYS);
}

/// The segm crowd branch: a crowd GT's similarity column is IoA
/// (intersection ÷ *detection* area), and detections matching it are ignored
/// rather than FPs.
///
/// Geometry (640×640, pixel-exact RLEs; bbox-IoA preconditions asserted with
/// the test-local oracle):
/// - GT1 mask `[0,0,50,50]`, `iscrowd` (2500 px).
/// - GT2 mask `[60,60,20,20]` (400 px).
/// - DT1 mask `[0,0,10,10]` (100 px), score 0.9 — wholly inside the crowd:
///   IoA = 100/100 = 1.0, while plain IoU = 100/2500 = 0.04.
/// - DT2 mask `[60,60,20,20]`, score 0.8 → IoU 1.0 with GT2.
///
/// With the IoA branch working, DT1 matches the crowd at every threshold and
/// is ignored; DT2 is the only ranked detection over the single counted GT →
/// AP = 1.0 across the sweep. If the crowd column used plain IoU, DT1 would be
/// an FP ranked above the TP and every AP would drop to 51/101 ≈ 0.505.
///
/// AR@1 pins the other side: the top-1 detection is the *ignored* DT1, so no
/// TP survives the cut and recall is a real 0.0 — not the −1.0 sentinel.
#[test]
fn test_segm_eval_crowd_uses_ioa() {
    assert!((ioa_of([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 50.0, 50.0]) - 1.0).abs() < 1e-12);
    assert!(iou_of([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 50.0, 50.0]) < 0.05);

    let gt = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            ann(1, [0.0, 0.0, 50.0, 50.0])
                .crowd()
                .mask(rect_mask(640, 640, 0, 0, 50, 50)),
            ann(2, [60.0, 60.0, 20.0, 20.0]).mask(rect_mask(640, 640, 60, 60, 20, 20)),
        ],
    );
    let dt = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            det(101, [0.0, 0.0, 10.0, 10.0], 0.9).mask(rect_mask(640, 640, 0, 0, 10, 10)),
            det(102, [60.0, 60.0, 20.0, 20.0], 0.8).mask(rect_mask(640, 640, 60, 60, 20, 20)),
        ],
    );

    let mut ev = COCOeval::new(
        COCO::from_dataset(gt),
        COCO::from_dataset(dt),
        IouType::Segm,
    );
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().expect("summarize sets stats");
    let expected = [
        1.0,  // AP — the crowd absorbs DT1; DT2 is a clean TP
        1.0,  // AP50
        1.0,  // AP75
        1.0,  // APs — GT2 (400 px) is the only counted GT
        -1.0, // APm — the 2500 px crowd is in range but ignored → nothing counted
        -1.0, // APl
        0.0,  // AR1 — top-1 (DT1) is ignored: zero TPs over one counted GT
        1.0,  // AR10
        1.0,  // AR100
        1.0,  // ARs
        -1.0, // ARm
        -1.0, // ARl
    ];
    assert_stats(stats, &expected, &BBOX_SEGM_KEYS);
}

/// Polygon segmentations run the `fr_poly` rasterization path end-to-end.
///
/// Identical GT and DT polygons give IoU exactly 1.0 whatever the rasterizer
/// does with boundary pixels, so AP pins to 1.0 without depending on fill
/// rules. The `area` field (2000 px², from the 50×40 bbox) is what drives area
/// slicing: medium is computed, small degrades to the −1.0 sentinel.
#[test]
fn test_segm_eval_polygon_end_to_end() {
    let poly = || Segmentation::Polygon(vec![vec![10.0, 10.0, 60.0, 10.0, 60.0, 50.0, 10.0, 50.0]]);
    let gt = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![ann(1, [10.0, 10.0, 50.0, 40.0]).mask(poly())],
    );
    let dt = dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![det(101, [10.0, 10.0, 50.0, 40.0], 0.9).mask(poly())],
    );

    let mut ev = COCOeval::new(
        COCO::from_dataset(gt),
        COCO::from_dataset(dt),
        IouType::Segm,
    );
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().expect("summarize sets stats");
    assert!(
        (stats[0] - 1.0).abs() < 1e-9,
        "AP for identical polygons: {}",
        stats[0]
    );
    assert_eq!(stats[3], -1.0, "APs: no small GT → sentinel");
    assert!(
        (stats[4] - 1.0).abs() < 1e-9,
        "APm: the 2000 px² GT is medium"
    );
}

/// Keypoints (OKS) end-to-end: evaluate → accumulate → summarize.
///
/// Fixture (one image, the default 17 COCO sigmas; σ₀ = 0.026 is the nose):
/// - GT1: 17 visible keypoints at `(100+i, 100)`, area 5000 (bbox 100×50).
///   DT1 (score 0.9) repeats them exactly → every per-keypoint e = 0 →
///   OKS = 1.0.
/// - GT2: only keypoint 0 visible, at `(300, 300)`; area 5000;
///   `num_keypoints` = 1. DT2 (score 0.8) puts keypoint 0 at `(304.2, 300)`,
///   off by d = 4.2. Per the pycocotools definition
///   (`vars = (2σ)²`, `e = d²/vars/(area+ε)/2`, `OKS = mean(exp(−e))`):
///   vars = 0.052² = 0.002704;
///   e = 4.2² / (0.002704 · 5000 · 2) = 17.64 / 27.04 ≈ 0.652367;
///   OKS = exp(−0.652367) ≈ 0.5208 → matches at t = 0.50, misses ≥ 0.55.
/// - Cross terms: the two poses are ~200–280 px apart, so every cross-OKS is
///   at most e^(−40) ≈ 0 and greedy matching pairs DT1↔GT1, DT2↔GT2.
///
/// The P/R arithmetic is then [`AP_FULL_SWEEP`]'s. Both GT areas are 5000 px²
/// → "medium" (32²..96²): APm/ARm equal all, large is the −1.0 sentinel.
/// Keypoint summaries have 10 rows (no small range), all at maxDets = 20.
#[test]
fn test_keypoints_eval_end_to_end() {
    // The OKS the fixture depends on, recomputed independently of the kernel
    // (area + f64::EPSILON ≈ area at this scale).
    let e = (4.2_f64 * 4.2) / (0.052_f64 * 0.052) / 5000.0 / 2.0;
    let oks = (-e).exp();
    assert!(
        oks > 0.51 && oks < 0.54,
        "fixture OKS must sit between the 0.50 and 0.55 thresholds: {oks}"
    );

    let mut gt1_kpts = Vec::with_capacity(51);
    for i in 0..17 {
        gt1_kpts.extend_from_slice(&[100.0 + i as f64, 100.0, 2.0]);
    }
    let mut gt2_kpts = vec![0.0; 51];
    gt2_kpts[0] = 300.0;
    gt2_kpts[1] = 300.0;
    gt2_kpts[2] = 2.0;
    let mut dt2_kpts = vec![0.0; 51];
    dt2_kpts[0] = 304.2; // d = 4.2 from GT2's visible keypoint
    dt2_kpts[1] = 300.0;

    let gt = dataset(
        vec![img(1)],
        vec![cat(1, "person")],
        vec![
            ann(1, [80.0, 80.0, 100.0, 50.0]).kpts(gt1_kpts.clone()),
            ann(2, [280.0, 280.0, 100.0, 50.0]).kpts(gt2_kpts),
        ],
    );
    let dt = dataset(
        vec![img(1)],
        vec![cat(1, "person")],
        vec![
            det(101, [80.0, 80.0, 100.0, 50.0], 0.9).kpts(gt1_kpts),
            det(102, [280.0, 280.0, 100.0, 50.0], 0.8).kpts(dt2_kpts),
        ],
    );

    let mut ev = COCOeval::new(
        COCO::from_dataset(gt),
        COCO::from_dataset(dt),
        IouType::Keypoints,
    );
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().expect("summarize sets stats");
    let keys = [
        "AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl",
    ];
    let expected = [
        AP_FULL_SWEEP, // AP
        1.0,           // AP50 — OKS 0.52 clears 0.50
        AP_AT_075,     // AP75 — OKS 0.52 misses 0.75
        AP_FULL_SWEEP, // APm — both GTs are medium, so medium == all
        -1.0,          // APl — no large GT
        AR_FULL_SWEEP, // AR (maxDets=20)
        1.0,           // AR50
        0.5,           // AR75
        AR_FULL_SWEEP, // ARm
        -1.0,          // ARl
    ];
    assert_stats(stats, &expected, &keys);
}

// --- update_anns: targeted annotation edits without a dataset rebuild -------

#[test]
fn update_anns_replaces_by_id_and_reindexes() {
    let mut coco = COCO::from_dataset(dataset(
        vec![img(1), img(2)],
        vec![cat(1, "thing")],
        vec![ann(1, [0.0, 0.0, 10.0, 10.0]), ann(2, [0.0, 0.0, 4.0, 4.0])],
    ));

    // Move annotation 2 to image 2 and give it a new area — the move is what
    // forces the re-index, and a stale index would show up here.
    let moved = ann(2, [0.0, 0.0, 4.0, 4.0]).in_img(2).with_area(999.0);
    coco.update_anns(vec![moved]).unwrap();

    assert_eq!(coco.get_ann(2).unwrap().area, Some(999.0));
    assert_eq!(coco.get_ann_ids_for_img(1), &[1]);
    assert_eq!(coco.get_ann_ids_for_img(2), &[2]);
    assert_eq!(coco.get_ann_ids_for_img_cat(2, 1), &[2]);
    // The area filter answers from the replaced record, not the loaded one.
    assert_eq!(
        coco.get_ann_ids(&[], &[], Some([500.0, 2000.0]), None),
        vec![2]
    );
}

#[test]
fn update_anns_edit_in_place_keeps_the_indices_current() {
    // An edit that moves nothing skips the re-index — the lookups must still
    // answer from the replaced record.
    let mut coco = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![ann(1, [0.0, 0.0, 10.0, 10.0])],
    ));

    coco.update_anns(vec![ann(1, [0.0, 0.0, 10.0, 10.0]).with_area(7.0)])
        .unwrap();

    assert_eq!(coco.get_ann(1).unwrap().area, Some(7.0));
    assert_eq!(coco.get_ann_ids_for_img(1), &[1]);
    assert_eq!(coco.get_ann_ids_for_img_cat(1, 1), &[1]);
    assert_eq!(coco.load_anns(&[1])[0].area, Some(7.0));
}

#[test]
fn update_anns_unknown_id_errors_and_writes_nothing() {
    let mut coco = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![ann(1, [0.0, 0.0, 10.0, 10.0])],
    ));

    let edit = ann(1, [0.0, 0.0, 10.0, 10.0]).with_area(1.0);
    let bogus = ann(7, [0.0, 0.0, 10.0, 10.0]).with_area(2.0);
    let err = coco.update_anns(vec![edit, bogus]).unwrap_err();

    assert_eq!(err.0, vec![7]);
    // The valid edit in the same call must not have landed either.
    assert_eq!(coco.get_ann(1).unwrap().area, Some(100.0));
    assert_eq!(coco.dataset.annotations.len(), 1);
}

#[test]
fn update_anns_keeps_the_last_of_duplicate_ids() {
    // pycocotools parity: the id lookup holds the last occurrence, so that is
    // the record `update_anns` replaces.
    let mut coco = COCO::from_dataset(dataset(
        vec![img(1)],
        vec![cat(1, "thing")],
        vec![
            ann(1, [0.0, 0.0, 1.0, 1.0]).with_area(1.0),
            ann(1, [0.0, 0.0, 2.0, 2.0]).with_area(2.0),
        ],
    ));

    coco.update_anns(vec![ann(1, [0.0, 0.0, 2.0, 2.0]).with_area(42.0)])
        .unwrap();

    assert_eq!(coco.dataset.annotations[0].area, Some(1.0));
    assert_eq!(coco.dataset.annotations[1].area, Some(42.0));
}

#[test]
fn update_anns_does_not_re_report_duplicate_ids() {
    // Reporting duplicate ids is a load-time event: `create_index` prints it and
    // appends it to `load_warnings`. A mutator re-indexes as often as a caller
    // edits, so it must not grow that list per edit.
    let mut coco = COCO::from_dataset(dataset(
        vec![img(1), img(2)],
        vec![cat(1, "thing")],
        vec![
            ann(1, [0.0, 0.0, 1.0, 1.0]),
            ann(1, [0.0, 0.0, 2.0, 2.0]),
            ann(2, [0.0, 0.0, 3.0, 3.0]),
        ],
    ));
    assert_eq!(coco.load_warnings().len(), 1, "the load reports them once");

    for _ in 0..3 {
        // A move, so the re-index actually runs.
        coco.update_anns(vec![ann(2, [0.0, 0.0, 3.0, 3.0]).in_img(2)])
            .unwrap();
        coco.update_anns(vec![ann(2, [0.0, 0.0, 3.0, 3.0]).in_img(1)])
            .unwrap();
    }
    assert_eq!(coco.load_warnings().len(), 1, "and the edits stay quiet");

    // An explicit `create_index()` still reports, the pycocotools idiom intact.
    coco.create_index();
    assert_eq!(coco.load_warnings().len(), 2);
}
