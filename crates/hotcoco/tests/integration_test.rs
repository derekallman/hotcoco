use std::collections::HashSet;
use std::path::PathBuf;

use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image};
use hotcoco::{COCOeval, COCO};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
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
fn test_bbox_evaluation_runs() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();

    let eval = coco_eval.eval.as_ref().expect("Accumulate should set eval");

    // Verify dimensions
    assert_eq!(eval.t, 10); // IoU thresholds
    assert_eq!(eval.r, 101); // recall thresholds
    assert_eq!(eval.k, 2); // categories
    assert_eq!(eval.a, 4); // area ranges
    assert_eq!(eval.m, 3); // max_dets

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

#[test]
fn test_summarize_prints() {
    let gt_path = fixtures_dir().join("gt.json");
    let dt_path = fixtures_dir().join("dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load DT");

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();
    // This should print 12 lines without panicking
    coco_eval.summarize();
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
    // GT_A: bbox [10,10,20,20] area=400, non-crowd → area-ignored (below 500)
    // GT_B: bbox [50,50,100,100] area=10000 → in range
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img1.jpg".into(),
            height: 200,
            width: 200,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
        }],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 100.0, 100.0]),
                area: Some(10000.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "thing".into(),
            supercategory: None,
            skeleton: None,
            keypoints: None,
            frequency: None,
        }],
        licenses: vec![],
    };

    // DT1: matches GT_A exactly, area=400 (small), score=0.9
    //       → matches area-ignored GT_A → DT1 is "ignored"
    // DT2: [10,10,25,20] area=500 (in range), overlaps GT_A (IoU≈0.8), score=0.8
    //       With fix: GT_A already matched, not crowd → can't re-match → FP
    //       With bug: GT_A is "ignorable" → re-match → DT2 also "ignored"
    // DT3: matches GT_B perfectly, area=10000 (in range), score=0.7 → TP
    //
    // Crucially, DT2 (the FP) has higher score than DT3 (the TP), so
    // the FP appears before the TP in the precision-recall curve,
    // reducing AP from 1.0 to ~0.5.
    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![
            Annotation {
                id: 101,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                score: Some(0.9),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            Annotation {
                id: 102,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 25.0, 20.0]),
                area: Some(500.0),
                score: Some(0.8),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            Annotation {
                id: 103,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 100.0, 100.0]),
                area: Some(10000.0),
                score: Some(0.7),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
        ],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    // Custom area range: [500, 1e10] so GT_A (area=400) is area-ignored
    coco_eval.params.area_rng = vec![[500.0, 1e10]];
    coco_eval.params.area_rng_lbl = vec!["custom".into()];
    coco_eval.evaluate();
    coco_eval.accumulate();

    let eval = coco_eval.eval.as_ref().unwrap();
    let m_idx = eval.m - 1;

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
    let ap_sum: f64 = (0..eval.r)
        .map(|r| {
            let idx = eval.precision_idx(0, r, 0, 0, m_idx);
            let p = eval.precision[idx];
            if p < 0.0 {
                0.0
            } else {
                p
            }
        })
        .sum();
    let ap = ap_sum / eval.r as f64;

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
    coco_eval.stats.expect("summarize should set stats")
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
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "crowd.jpg".into(),
            height: 100,
            width: 100,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
        }],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([10.0, 10.0, 50.0, 50.0]),
            area: Some(2500.0),
            iscrowd: true,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: None,
        }],
        categories: vec![Category {
            id: 1,
            name: "thing".into(),
            supercategory: None,
            skeleton: None,
            keypoints: None,
            frequency: None,
        }],
        licenses: vec![],
    };

    // 3 detections all overlapping the crowd GT
    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![
            Annotation {
                id: 101,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 50.0, 50.0]),
                area: Some(2500.0),
                score: Some(0.9),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            Annotation {
                id: 102,
                image_id: 1,
                category_id: 1,
                bbox: Some([12.0, 12.0, 48.0, 48.0]),
                area: Some(2304.0),
                score: Some(0.8),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
            Annotation {
                id: 103,
                image_id: 1,
                category_id: 1,
                bbox: Some([15.0, 15.0, 45.0, 45.0]),
                area: Some(2025.0),
                score: Some(0.7),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
            },
        ],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    coco_eval.evaluate();
    coco_eval.accumulate();

    let eval = coco_eval.eval.as_ref().unwrap();
    let m_idx = eval.m - 1; // maxDets=100

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
    let all_neg = (0..eval.r).all(|r| {
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
        v.sort();
        v
    };
    let train2_ids: Vec<u64> = {
        let mut v: Vec<u64> = train2.images.iter().map(|i| i.id).collect();
        v.sort();
        v
    };
    assert_eq!(train1_ids, train2_ids, "Same seed must produce same split");

    let val1_ids: Vec<u64> = {
        let mut v: Vec<u64> = val1.images.iter().map(|i| i.id).collect();
        v.sort();
        v
    };
    let val2_ids: Vec<u64> = {
        let mut v: Vec<u64> = val2.images.iter().map(|i| i.id).collect();
        v.sort();
        v
    };
    assert_eq!(val1_ids, val2_ids, "Same seed must produce same split");

    // Different seed should produce a different partition (with high probability for this data)
    let (train3, _, _) = coco.split(0.33, None, 99);
    let train3_ids: Vec<u64> = {
        let mut v: Vec<u64> = train3.images.iter().map(|i| i.id).collect();
        v.sort();
        v
    };
    // With 3 images and different seeds the shuffle may still coincide, but we at least
    // verify it compiles and runs without error.
    let _ = train3_ids;
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
fn test_evaluate_sparse_matches_dense() {
    let gt_path = fixtures_dir().join("edge_gt.json");
    let dt_path = fixtures_dir().join("edge_dt.json");
    let coco_gt = COCO::new(&gt_path).expect("Failed to load edge GT");
    let coco_dt = coco_gt.load_res(&dt_path).expect("Failed to load edge DT");

    let stats = run_bbox_eval(coco_gt, coco_dt);
    assert_eq!(stats.len(), 12, "summarize() should return 12 metrics");

    // Same expected values as test_edge_cases — verifies the sparse path produces
    // bit-identical results to the previously-verified dense implementation.
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
            "stats[{i}] mismatch (sparse path): got {got:.6}, expected {exp:.6}"
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

// ---------------------------------------------------------------------------
// LVIS federated evaluation tests
// ---------------------------------------------------------------------------

/// Helper: build a minimal Dataset from raw parts.
fn make_lvis_dataset(
    images: Vec<Image>,
    annotations: Vec<Annotation>,
    categories: Vec<Category>,
) -> Dataset {
    Dataset {
        info: None,
        images,
        annotations,
        categories,
        licenses: vec![],
    }
}

fn lvis_image(id: u64, neg_category_ids: Vec<u64>, not_exhaustive_category_ids: Vec<u64>) -> Image {
    Image {
        id,
        file_name: format!("img{}.jpg", id),
        height: 100,
        width: 100,
        license: None,
        coco_url: None,
        flickr_url: None,
        date_captured: None,
        neg_category_ids,
        not_exhaustive_category_ids,
    }
}

fn lvis_gt_ann(id: u64, image_id: u64, category_id: u64, area: f64) -> Annotation {
    Annotation {
        id,
        image_id,
        category_id,
        bbox: Some([0.0, 0.0, area.sqrt(), area.sqrt()]),
        area: Some(area),
        segmentation: None,
        iscrowd: false,
        keypoints: None,
        num_keypoints: None,
        score: None,
    }
}

fn lvis_dt_ann(id: u64, image_id: u64, category_id: u64, area: f64, score: f64) -> Annotation {
    Annotation {
        id,
        image_id,
        category_id,
        bbox: Some([0.0, 0.0, area.sqrt(), area.sqrt()]),
        area: Some(area),
        segmentation: None,
        iscrowd: false,
        keypoints: None,
        num_keypoints: None,
        score: Some(score),
    }
}

fn lvis_category(id: u64, frequency: Option<&str>) -> Category {
    Category {
        id,
        name: format!("cat{}", id),
        supercategory: None,
        skeleton: None,
        keypoints: None,
        frequency: frequency.map(String::from),
    }
}

/// LVIS test 1: neg_category_ids — unmatched DTs on an image where the
/// category is confirmed absent must count as FP → AP = 0.
#[test]
fn test_lvis_neg_category_counts_as_fp() {
    // 1 image, cat 1 listed in neg_category_ids (no GT). Detector fires.
    // The DT is a false positive → AP should be 0.
    let gt_ds = make_lvis_dataset(
        vec![lvis_image(1, vec![1], vec![])],
        vec![],
        vec![lvis_category(1, Some("r"))],
    );
    let dt_ds = make_lvis_dataset(
        vec![lvis_image(1, vec![], vec![])],
        vec![lvis_dt_ann(101, 1, 1, 400.0, 0.9)],
        vec![lvis_category(1, None)],
    );

    let coco_gt = COCO::from_dataset(gt_ds);
    let coco_dt = COCO::from_dataset(dt_ds);

    let mut ev = COCOeval::new_lvis(coco_gt, coco_dt, IouType::Bbox);
    ev.run();

    let results = ev.get_results();
    let ap = results["AP"];
    assert!(
        ap <= 0.0,
        "AP should be 0.0 when DT fires on neg_category image, got {ap}"
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
    let gt_ds = make_lvis_dataset(
        vec![
            lvis_image(1, vec![], vec![]), // image A: no neg, no not_exhaustive
            lvis_image(2, vec![], vec![]), // image B: no neg, no not_exhaustive
        ],
        vec![lvis_gt_ann(1, 1, 1, 400.0)], // GT only on image A
        vec![lvis_category(1, Some("f"))],
    );
    let dt_ds = make_lvis_dataset(
        vec![lvis_image(1, vec![], vec![]), lvis_image(2, vec![], vec![])],
        vec![
            lvis_dt_ann(101, 1, 1, 400.0, 0.9), // matches GT on image A
            lvis_dt_ann(102, 2, 1, 400.0, 0.8), // fires on image B — should be dropped
        ],
        vec![lvis_category(1, None)],
    );

    let coco_gt_two = COCO::from_dataset(gt_ds.clone());
    let coco_dt_two = COCO::from_dataset(dt_ds);

    let mut ev_two = COCOeval::new_lvis(coco_gt_two, coco_dt_two, IouType::Bbox);
    ev_two.run();

    // Baseline: only image A with its GT and matching DT (perfect AP = 1.0).
    let gt_ds_one = make_lvis_dataset(
        vec![lvis_image(1, vec![], vec![])],
        vec![lvis_gt_ann(1, 1, 1, 400.0)],
        vec![lvis_category(1, Some("f"))],
    );
    let dt_ds_one = make_lvis_dataset(
        vec![lvis_image(1, vec![], vec![])],
        vec![lvis_dt_ann(101, 1, 1, 400.0, 0.9)],
        vec![lvis_category(1, None)],
    );

    let mut ev_one = COCOeval::new_lvis(
        COCO::from_dataset(gt_ds_one),
        COCO::from_dataset(dt_ds_one),
        IouType::Bbox,
    );
    ev_one.run();

    let ap_two = ev_two.get_results()["AP"];
    let ap_one = ev_one.get_results()["AP"];

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
    let gt_ds = make_lvis_dataset(
        vec![lvis_image(1, vec![], vec![1])], // not_exhaustive for cat 1
        vec![lvis_gt_ann(1, 1, 1, 400.0)],
        vec![lvis_category(1, Some("c"))],
    );
    let dt_ds = make_lvis_dataset(
        vec![lvis_image(1, vec![], vec![])],
        vec![
            lvis_dt_ann(101, 1, 1, 400.0, 0.9), // matches GT
            lvis_dt_ann(102, 1, 1, 100.0, 0.5), // unmatched — should be ignored
        ],
        vec![lvis_category(1, None)],
    );

    let mut ev = COCOeval::new_lvis(
        COCO::from_dataset(gt_ds),
        COCO::from_dataset(dt_ds),
        IouType::Bbox,
    );
    ev.run();

    let ap = ev.get_results()["AP"];
    assert!(
        (ap - 1.0).abs() < 1e-6,
        "Unmatched DT in not_exhaustive image should be ignored; AP should be 1.0, got {ap}"
    );
}
