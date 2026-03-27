use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use hotcoco::convert::{
    coco_to_cvat, coco_to_dota, coco_to_voc, coco_to_yolo, cvat_to_coco, dota_to_coco, voc_to_coco,
    yolo_to_coco,
};
use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image};
use hotcoco::{healthcheck, COCOeval, Hierarchy, COCO};

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
                obb: None,
                is_group_of: None,
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
                obb: None,
                is_group_of: None,
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
                obb: None,
                is_group_of: None,
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
                obb: None,
                is_group_of: None,
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
                obb: None,
                is_group_of: None,
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
            if p < 0.0 {
                0.0
            } else {
                p
            }
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
            obb: None,
            is_group_of: None,
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
                obb: None,
                is_group_of: None,
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
                obb: None,
                is_group_of: None,
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
                obb: None,
                is_group_of: None,
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
        obb: None,
        is_group_of: None,
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
        obb: None,
        is_group_of: None,
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

    let results = ev.get_results(None, false);
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

    let ap = ev.get_results(None, false)["AP"];
    assert!(
        (ap - 1.0).abs() < 1e-6,
        "Unmatched DT in not_exhaustive image should be ignored; AP should be 1.0, got {ap}"
    );
}

// ============================================================
// Helpers shared by confusion_matrix tests
// ============================================================

fn cm_image(id: u64) -> Image {
    Image {
        id,
        file_name: format!("img{id}.jpg"),
        height: 200,
        width: 200,
        license: None,
        coco_url: None,
        flickr_url: None,
        date_captured: None,
        neg_category_ids: vec![],
        not_exhaustive_category_ids: vec![],
    }
}

fn cm_category(id: u64, name: &str) -> Category {
    Category {
        id,
        name: name.into(),
        supercategory: None,
        skeleton: None,
        keypoints: None,
        frequency: None,
    }
}

fn cm_gt_ann(id: u64, img_id: u64, cat_id: u64, bbox: [f64; 4]) -> Annotation {
    Annotation {
        id,
        image_id: img_id,
        category_id: cat_id,
        bbox: Some(bbox),
        area: Some(bbox[2] * bbox[3]),
        iscrowd: false,
        segmentation: None,
        keypoints: None,
        num_keypoints: None,
        score: None,
        obb: None,
        is_group_of: None,
    }
}

fn cm_dt_ann(id: u64, img_id: u64, cat_id: u64, bbox: [f64; 4], score: f64) -> Annotation {
    Annotation {
        id,
        image_id: img_id,
        category_id: cat_id,
        bbox: Some(bbox),
        area: Some(bbox[2] * bbox[3]),
        iscrowd: false,
        segmentation: None,
        keypoints: None,
        num_keypoints: None,
        score: Some(score),
        obb: None,
        is_group_of: None,
    }
}

fn cm_coco(images: Vec<Image>, anns: Vec<Annotation>, cats: Vec<Category>) -> COCO {
    COCO::from_dataset(Dataset {
        info: None,
        images,
        annotations: anns,
        categories: cats,
        licenses: vec![],
    })
}

// ============================================================
// confusion_matrix tests
// ============================================================

/// All DTs match their correct category → pure diagonal matrix.
#[test]
fn test_confusion_matrix_perfect() {
    // 2 categories: cat(1)=idx 0, dog(2)=idx 1; background=idx 2
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![
            cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0]),  // cat GT
            cm_gt_ann(2, 1, 2, [60.0, 0.0, 50.0, 50.0]), // dog GT
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![
            cm_dt_ann(101, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.9), // cat DT → matches cat GT
            cm_dt_ann(102, 1, 2, [60.0, 0.0, 50.0, 50.0], 0.8), // dog DT → matches dog GT
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 2, [0.0, 0.0, 50.0, 50.0], 0.9)],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![], // no GTs
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.9)],
        vec![cm_category(1, "cat")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![], // no detections
        vec![cm_category(1, "cat")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 100.0, 100.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 1, [50.0, 0.0, 50.0, 100.0], 0.9)],
        vec![cm_category(1, "cat")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.3)],
        vec![cm_category(1, "cat")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![
            cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0]),
            cm_gt_ann(2, 1, 2, [60.0, 0.0, 50.0, 50.0]),
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![
            cm_dt_ann(101, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.9),
            cm_dt_ann(102, 1, 2, [60.0, 0.0, 50.0, 50.0], 0.5),
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );

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
//
// Reuse the cm_* helpers defined above:
//   cm_image, cm_category, cm_gt_ann, cm_dt_ann, cm_coco

/// Run evaluate() and return tide_errors at the default thresholds.
fn run_tide(coco_gt: COCO, coco_dt: COCO) -> hotcoco::TideErrors {
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.tide_errors(0.5, 0.1).expect("tide_errors failed")
}

/// Test 1: all DTs are perfect TPs → all ΔAP = 0, all counts = 0.
#[test]
fn test_tide_all_correct() {
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.9)],
        vec![cm_category(1, "cat")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![
            cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0]),  // cat GT
            cm_gt_ann(2, 1, 2, [60.0, 0.0, 50.0, 50.0]), // dog GT (no overlap with DT)
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 2, [0.0, 0.0, 50.0, 50.0], 0.9)],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 1, [25.0, 0.0, 50.0, 50.0], 0.9)],
        vec![cm_category(1, "cat")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 2, [25.0, 0.0, 50.0, 50.0], 0.9)],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Both"], 1, "should be 1 Both error");
    assert_eq!(te.counts["Cls"], 0, "not Cls because IoU < pos_thr");
    assert_eq!(te.counts["Loc"], 0, "not Loc because different class");
}

/// Test 5: two DTs for same GT → first is TP, second is Dupe.
#[test]
fn test_tide_dupe_error() {
    // GT: [0,0,50,50]; DT1(score=0.9): exact match (TP); DT2(score=0.7): same box (Dupe)
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![
            cm_dt_ann(101, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.9), // TP
            cm_dt_ann(102, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.7), // Dupe
        ],
        vec![cm_category(1, "cat")],
    );

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Dupe"], 1, "second DT should be Dupe");
    assert_eq!(te.counts["Bkg"], 0);
    assert_eq!(te.counts["Cls"], 0);
}

/// Test 6: DT with IoU < bg_thr with all GTs → Bkg error.
#[test]
fn test_tide_bkg_error() {
    // GT: [0,0,10,10]; DT: [90,90,10,10] — no overlap at all → IoU=0 < bg_thr=0.1
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 10.0, 10.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 1, [90.0, 90.0, 10.0, 10.0], 0.9)],
        vec![cm_category(1, "cat")],
    );

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Bkg"], 1, "far-away DT should be Bkg error");
    assert_eq!(te.counts["Loc"], 0);
    assert_eq!(te.counts["Cls"], 0);
}

/// Test 7: GT with no DT → Miss error, ΔAP["Miss"] > 0.
#[test]
fn test_tide_miss_error() {
    // GT: [0,0,50,50]; no DT at all
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(vec![cm_image(1)], vec![], vec![cm_category(1, "cat")]);

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
    let coco_gt = cm_coco(
        vec![cm_image(1), cm_image(2)],
        vec![
            cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0]), // img1 cat TP
            cm_gt_ann(2, 2, 1, [0.0, 0.0, 30.0, 30.0]), // img2 cat (small)
            cm_gt_ann(3, 2, 2, [0.0, 0.0, 50.0, 50.0]), // img2 dog
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1), cm_image(2)],
        vec![
            cm_dt_ann(101, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.95), // img1 cat TP
            cm_dt_ann(102, 2, 1, [0.0, 0.0, 50.0, 50.0], 0.9),  // img2 cat FP: Loc wins over Cls
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![
            cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0]),  // cat
            cm_gt_ann(2, 1, 2, [60.0, 0.0, 50.0, 50.0]), // dog (no overlap with DT)
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(101, 1, 1, [25.0, 0.0, 50.0, 50.0], 0.9)],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );

    let te = run_tide(coco_gt, coco_dt);

    assert_eq!(te.counts["Loc"], 1, "same-class overlap ≥ bg_thr → Loc");
    assert_eq!(te.counts["Both"], 0);
}

/// Test 10: ΔAP["FP"] ≥ max of individual FP ΔAPs.
#[test]
fn test_tide_delta_ap_fp_ge_individuals() {
    // Multiple FP error types in one scene
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![
            cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0]),
            cm_gt_ann(2, 1, 2, [60.0, 0.0, 50.0, 50.0]),
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![
            // DT1 cat: matches cat GT (TP)
            cm_dt_ann(101, 1, 1, [0.0, 0.0, 50.0, 50.0], 0.95),
            // DT2 dog: far away → Bkg
            cm_dt_ann(102, 1, 2, [150.0, 150.0, 10.0, 10.0], 0.8),
            // DT3 cat: wrong class vs dog GT at [60,0,50,50] → cross-IoU=1.0 → Cls
            cm_dt_ann(103, 1, 1, [60.0, 0.0, 50.0, 50.0], 0.7),
        ],
        vec![cm_category(1, "cat"), cm_category(2, "dog")],
    );

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
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, [0.0, 0.0, 50.0, 50.0])],
        vec![cm_category(1, "cat")],
    );
    let coco_dt = cm_coco(vec![cm_image(1)], vec![], vec![cm_category(1, "cat")]);

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
                license: None,
                coco_url: None,
                flickr_url: None,
                date_captured: None,
                neg_category_ids: vec![],
                not_exhaustive_category_ids: vec![],
            },
            Image {
                id: 2,
                file_name: "img2.jpg".into(),
                width: 400,
                height: 300,
                license: None,
                coco_url: None,
                flickr_url: None,
                date_captured: None,
                neg_category_ids: vec![],
                not_exhaustive_category_ids: vec![],
            },
        ],
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 20.0, 30.0, 40.0]),
                area: Some(1200.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 2,
                bbox: Some([50.0, 60.0, 20.0, 25.0]),
                area: Some(500.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 3,
                image_id: 2,
                category_id: 1,
                bbox: Some([0.0, 0.0, 200.0, 150.0]),
                area: Some(30000.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
        ],
        categories: vec![
            Category {
                id: 1,
                name: "cat".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
            Category {
                id: 2,
                name: "dog".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
        ],
        licenses: vec![],
    }
}

#[test]
fn test_coco_to_yolo_basic() {
    let dataset = make_test_dataset_basic();
    let dir = tempfile::tempdir().expect("tempdir");
    let stats = coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    assert_eq!(stats.images, 2);
    assert_eq!(stats.annotations, 3);
    assert_eq!(stats.skipped_crowd, 0);
    assert_eq!(stats.missing_bbox, 0);

    // data.yaml
    let yaml = std::fs::read_to_string(dir.path().join("data.yaml")).expect("data.yaml");
    assert!(yaml.contains("nc: 2"), "yaml: {yaml}");
    assert!(yaml.contains("names: [cat, dog]"), "yaml: {yaml}");

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
                category_id: 7,
                bbox: Some([10.0, 10.0, 40.0, 40.0]),
                area: Some(1600.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 3,
                bbox: Some([60.0, 60.0, 20.0, 20.0]),
                area: Some(400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
        ],
        // Unsorted in dataset; coco_to_yolo must sort by ID
        categories: vec![
            Category {
                id: 7,
                name: "bird".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
            Category {
                id: 1,
                name: "cat".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
            Category {
                id: 3,
                name: "dog".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
        ],
        licenses: vec![],
    };

    let dir = tempfile::tempdir().expect("tempdir");
    coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    let yaml = std::fs::read_to_string(dir.path().join("data.yaml")).expect("data.yaml");
    // Sorted order: cat(1), dog(3), bird(7)
    assert!(yaml.contains("names: [cat, dog, bird]"), "yaml: {yaml}");

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
                iscrowd: true, // should be skipped
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 20.0, 20.0]),
                area: Some(400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
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
                bbox: None, // no bbox — should be skipped
                area: Some(400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 20.0, 20.0]),
                area: Some(400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
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

    let dir = tempfile::tempdir().expect("tempdir");
    let stats = coco_to_yolo(&dataset, dir.path()).expect("coco_to_yolo");

    assert_eq!(stats.missing_bbox, 1);
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
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
        }],
        annotations: vec![],
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

    // Categories must round-trip (sorted by original COCO ID)
    assert_eq!(recovered.categories.len(), original.categories.len());

    // Each annotation's bbox must round-trip within floating-point tolerance
    assert_eq!(recovered.annotations.len(), original.annotations.len());

    // Build a lookup of original bboxes by (image filename stem, category name)
    // to compare against recovered bboxes
    let cat_id_to_name: HashMap<u64, &str> = original
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    let img_id_to_stem: HashMap<u64, String> = original
        .images
        .iter()
        .map(|img| {
            let stem = std::path::Path::new(&img.file_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(img.file_name.as_str())
                .to_string();
            (img.id, stem)
        })
        .collect();

    // Collect original (stem, cat_name, bbox) triples
    let mut orig_bboxes: Vec<(String, String, [f64; 4])> = original
        .annotations
        .iter()
        .map(|ann| {
            let stem = img_id_to_stem[&ann.image_id].clone();
            let cat = cat_id_to_name[&ann.category_id].to_string();
            (stem, cat, ann.bbox.unwrap())
        })
        .collect();
    orig_bboxes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Collect recovered (stem, cat_name, bbox) triples
    let rec_cat_id_to_name: HashMap<u64, &str> = recovered
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    let rec_img_id_to_stem: HashMap<u64, &str> = recovered
        .images
        .iter()
        .map(|img| (img.id, img.file_name.as_str()))
        .collect();

    let mut rec_bboxes: Vec<(String, String, [f64; 4])> = recovered
        .annotations
        .iter()
        .map(|ann| {
            let stem = rec_img_id_to_stem[&ann.image_id].to_string();
            let cat = rec_cat_id_to_name[&ann.category_id].to_string();
            (stem, cat, ann.bbox.unwrap())
        })
        .collect();
    rec_bboxes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    for ((o_stem, o_cat, o_bbox), (r_stem, r_cat, r_bbox)) in
        orig_bboxes.iter().zip(rec_bboxes.iter())
    {
        assert_eq!(o_stem, r_stem, "stem mismatch");
        assert_eq!(o_cat, r_cat, "category mismatch");
        for i in 0..4 {
            assert!(
                (o_bbox[i] - r_bbox[i]).abs() < 1e-4,
                "bbox[{i}] mismatch for {o_stem}/{o_cat}: orig={} recovered={}",
                o_bbox[i],
                r_bbox[i]
            );
        }
    }
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
    assert_eq!(stats.missing_bbox, 0);

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

    // Spot-check first annotation: bbox=[10,20,30,40] → xmin=10, ymin=20, xmax=40, ymax=60
    assert!(xml1.contains("<xmin>10</xmin>"), "xmin");
    assert!(xml1.contains("<ymin>20</ymin>"), "ymin");
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

#[test]
fn test_voc_to_coco_basic() {
    // Write a known VOC XML and parse it back
    let dir = tempfile::tempdir().expect("tempdir");
    let ann_dir = dir.path().join("Annotations");
    std::fs::create_dir_all(&ann_dir).expect("mkdir");

    let xml = r#"<annotation>
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
</annotation>"#;
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

    // person: xmin=100, ymin=50, xmax=300, ymax=400 → bbox=[100, 50, 200, 350]
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
    assert_eq!(bbox, [100.0, 50.0, 200.0, 350.0]);

    // car: xmin=400, ymin=200, xmax=600, ymax=450 → bbox=[400, 200, 200, 250]
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
    assert_eq!(bbox, [400.0, 200.0, 200.0, 250.0]);

    // difficult flag is dropped (not mapped to iscrowd)
    assert!(!person_ann.iscrowd);
    assert!(!car_ann.iscrowd);
}

#[test]
fn test_voc_round_trip() {
    let original = make_test_dataset_basic();
    let dir = tempfile::tempdir().expect("tempdir");

    // COCO → VOC
    coco_to_voc(&original, dir.path()).expect("coco_to_voc");

    // VOC → COCO
    let recovered = voc_to_coco(dir.path()).expect("voc_to_coco");

    assert_eq!(recovered.images.len(), original.images.len());
    assert_eq!(recovered.annotations.len(), original.annotations.len());
    assert_eq!(recovered.categories.len(), original.categories.len());

    // Build lookup for original bboxes by (image filename, category name)
    let cat_id_to_name: HashMap<u64, &str> = original
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    let img_id_to_fname: HashMap<u64, &str> = original
        .images
        .iter()
        .map(|img| (img.id, img.file_name.as_str()))
        .collect();

    let mut orig_bboxes: Vec<(String, String, [f64; 4])> = original
        .annotations
        .iter()
        .map(|ann| {
            let fname = img_id_to_fname[&ann.image_id].to_string();
            let cat = cat_id_to_name[&ann.category_id].to_string();
            (fname, cat, ann.bbox.unwrap())
        })
        .collect();
    orig_bboxes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let rec_cat_id_to_name: HashMap<u64, &str> = recovered
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    let rec_img_id_to_fname: HashMap<u64, &str> = recovered
        .images
        .iter()
        .map(|img| (img.id, img.file_name.as_str()))
        .collect();

    let mut rec_bboxes: Vec<(String, String, [f64; 4])> = recovered
        .annotations
        .iter()
        .map(|ann| {
            let fname = rec_img_id_to_fname[&ann.image_id].to_string();
            let cat = rec_cat_id_to_name[&ann.category_id].to_string();
            (fname, cat, ann.bbox.unwrap())
        })
        .collect();
    rec_bboxes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // VOC uses integer coords, so round-trip tolerance is 1.0 pixel
    for ((o_fname, o_cat, o_bbox), (r_fname, r_cat, r_bbox)) in
        orig_bboxes.iter().zip(rec_bboxes.iter())
    {
        assert_eq!(o_fname, r_fname, "filename mismatch");
        assert_eq!(o_cat, r_cat, "category mismatch");
        for i in 0..4 {
            assert!(
                (o_bbox[i] - r_bbox[i]).abs() <= 1.0,
                "bbox[{i}] mismatch for {o_fname}/{o_cat}: orig={} recovered={}",
                o_bbox[i],
                r_bbox[i]
            );
        }
    }
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
                bbox: Some([10.0, 20.0, 30.0, 40.0]),
                area: Some(1200.0),
                iscrowd: true,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 60.0, 10.0, 10.0]),
                area: Some(100.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
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

    let xml = r#"<annotation>
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
</annotation>"#;
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

    let dataset = cvat_to_coco(&xml_path).expect("cvat_to_coco");

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
    let recovered = cvat_to_coco(&xml_path).expect("cvat_to_coco");

    assert_eq!(recovered.images.len(), original.images.len());
    assert_eq!(recovered.annotations.len(), original.annotations.len());
    assert_eq!(recovered.categories.len(), original.categories.len());

    // Build lookup for comparison
    let cat_id_to_name: HashMap<u64, &str> = original
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    let img_id_to_fname: HashMap<u64, &str> = original
        .images
        .iter()
        .map(|img| (img.id, img.file_name.as_str()))
        .collect();
    let mut orig_bboxes: Vec<(String, String, [f64; 4])> = original
        .annotations
        .iter()
        .map(|ann| {
            let fname = img_id_to_fname[&ann.image_id].to_string();
            let cat = cat_id_to_name[&ann.category_id].to_string();
            (fname, cat, ann.bbox.unwrap())
        })
        .collect();
    orig_bboxes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let rec_cat: HashMap<u64, &str> = recovered
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    let rec_img: HashMap<u64, &str> = recovered
        .images
        .iter()
        .map(|img| (img.id, img.file_name.as_str()))
        .collect();
    let mut rec_bboxes: Vec<(String, String, [f64; 4])> = recovered
        .annotations
        .iter()
        .map(|ann| {
            let fname = rec_img[&ann.image_id].to_string();
            let cat = rec_cat[&ann.category_id].to_string();
            (fname, cat, ann.bbox.unwrap())
        })
        .collect();
    rec_bboxes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // CVAT uses float coords — round-trip should be exact within formatting precision
    for ((o_f, o_c, o_b), (r_f, r_c, r_b)) in orig_bboxes.iter().zip(rec_bboxes.iter()) {
        assert_eq!(o_f, r_f, "filename mismatch");
        assert_eq!(o_c, r_c, "category mismatch");
        for i in 0..4 {
            assert!(
                (o_b[i] - r_b[i]).abs() < 0.01,
                "bbox[{i}] mismatch for {o_f}/{o_c}: orig={} recovered={}",
                o_b[i],
                r_b[i]
            );
        }
    }
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

    let dataset = cvat_to_coco(&xml_path).expect("cvat_to_coco");
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

    let dataset = cvat_to_coco(&xml_path).expect("cvat_to_coco");
    // Only the box should be imported; polyline and points are unsupported
    assert_eq!(dataset.annotations.len(), 1);
    assert_eq!(
        dataset.annotations[0].bbox.unwrap(),
        [10.0, 10.0, 40.0, 40.0]
    );
}

// ── f_scores tests ────────────────────────────────────────────────────────────

fn make_perfect_eval() -> COCOeval {
    // One image, one GT bbox, one perfectly matching DT.
    let bbox = [10.0, 10.0, 50.0, 50.0];
    let coco_gt = cm_coco(
        vec![cm_image(1)],
        vec![cm_gt_ann(1, 1, 1, bbox)],
        vec![cm_category(1, "thing")],
    );
    let coco_dt = cm_coco(
        vec![cm_image(1)],
        vec![cm_dt_ann(1, 1, 1, bbox, 1.0)],
        vec![cm_category(1, "thing")],
    );
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
    assert!(f1.contains_key("F1") && f1.contains_key("F150") && f1.contains_key("F175"));
    for (k, v) in &f1 {
        assert!((0.0..=1.0).contains(v), "{k} = {v} outside [0, 1]");
    }

    // beta variant gets correct key prefix
    let fb = ev.f_scores(0.5);
    assert!(fb.contains_key("F0.5") && fb.contains_key("F0.550") && fb.contains_key("F0.575"));
}

#[test]
fn test_f_scores_perfect_detection() {
    let scores = make_perfect_eval().f_scores(1.0);
    assert!((scores["F1"] - 1.0).abs() < 1e-9, "F1={}", scores["F1"]);
    assert!(
        (scores["F150"] - 1.0).abs() < 1e-9,
        "F150={}",
        scores["F150"]
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
    let report = healthcheck::healthcheck(&dataset);

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
    let report = healthcheck::healthcheck(&dataset);

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
    let report = healthcheck::healthcheck(&dataset);

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
    let report = healthcheck::healthcheck(&dataset);

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

    let report = healthcheck::healthcheck_compatibility(&gt, &dt);

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

    let eval = ev.accumulated().unwrap();
    assert_eq!(eval.shape.t, 10);
    assert_eq!(eval.shape.k, 2);
    let valid_count = eval.precision.iter().filter(|&&v| v >= 0.0).count();
    assert!(valid_count > 0, "should have valid precision values");
    let valid_recall = eval.recall.iter().filter(|&&v| v >= 0.0).count();
    assert!(valid_recall > 0, "should have valid recall values");
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

    assert!(first.metrics.values().any(|&v| v >= 0.0));
    assert!(last.metrics.values().any(|&v| v >= 0.0));
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

#[test]
fn test_hierarchy_from_categories_supercategory() {
    let cats = vec![
        Category {
            id: 1,
            name: "dog".into(),
            supercategory: Some("animal".into()),
            skeleton: None,
            keypoints: None,
            frequency: None,
        },
        Category {
            id: 2,
            name: "animal".into(),
            supercategory: None,
            skeleton: None,
            keypoints: None,
            frequency: None,
        },
        Category {
            id: 3,
            name: "cat".into(),
            supercategory: Some("animal".into()),
            skeleton: None,
            keypoints: None,
            frequency: None,
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
    let dog_anc = h.ancestors(1);
    assert_eq!(dog_anc.len(), 2);
    assert_eq!(dog_anc[0], 1);
    assert_eq!(dog_anc[1], 2);
}

#[test]
fn test_hierarchy_virtual_nodes() {
    // "vehicle" supercategory doesn't match any category name -> virtual node
    let cats = vec![
        Category {
            id: 1,
            name: "car".into(),
            supercategory: Some("vehicle".into()),
            skeleton: None,
            keypoints: None,
            frequency: None,
        },
        Category {
            id: 2,
            name: "truck".into(),
            supercategory: Some("vehicle".into()),
            skeleton: None,
            keypoints: None,
            frequency: None,
        },
    ];

    let h = Hierarchy::from_categories(&cats);

    // Both car and truck should have the same parent (a virtual node)
    let car_parent = h.parent(1).unwrap();
    let truck_parent = h.parent(2).unwrap();
    assert_eq!(car_parent, truck_parent);
    // Virtual node ID should be very large (u64::MAX - 1)
    assert!(car_parent >= u64::MAX - 10);

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

    let dog_anc = h.ancestors(300);
    assert_eq!(dog_anc.len(), 3);
    assert_eq!(dog_anc[0], 300);
    assert_eq!(dog_anc[1], 200);
    assert_eq!(dog_anc[2], 100);
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

    // Dog (id=1) should have a parent (virtual animal node)
    let dog_parent = h.parent(1).unwrap();
    assert!(dog_parent >= u64::MAX - 10, "parent should be virtual node");

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
            category_id: 1, // Dog
            bbox: Some([10.0, 10.0, 20.0, 20.0]),
            area: Some(400.0),
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: None,
            obb: None,
            is_group_of: None,
        }],
        categories: vec![
            Category {
                id: 1,
                name: "dog".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
            Category {
                id: 2,
                name: "animal".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
        ],
        licenses: vec![],
    };

    let coco = COCO::from_dataset(gt_dataset);

    let mut pm: HashMap<u64, u64> = HashMap::new();
    pm.insert(1, 2); // Dog -> Animal
    let hierarchy = Hierarchy::from_parent_map(pm);

    let expanded = hotcoco::eval::expand::expand_gt(&coco, &hierarchy);

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
                category_id: 1, // Dog
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 2, // Animal (already present with same bbox)
                bbox: Some([10.0, 10.0, 20.0, 20.0]),
                area: Some(400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
        ],
        categories: vec![
            Category {
                id: 1,
                name: "dog".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
            Category {
                id: 2,
                name: "animal".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
        ],
        licenses: vec![],
    };

    let coco = COCO::from_dataset(gt_dataset);

    let mut pm: HashMap<u64, u64> = HashMap::new();
    pm.insert(1, 2); // Dog -> Animal
    let hierarchy = Hierarchy::from_parent_map(pm);

    let expanded = hotcoco::eval::expand::expand_gt(&coco, &hierarchy);

    // Should still have exactly 2 annotations — no duplicates
    assert_eq!(
        expanded.dataset.annotations.len(),
        2,
        "pre-expanded input should stay at 2 annotations"
    );
}

#[test]
fn test_oid_group_of_multi_match() {
    // One non-group GT + one group-of GT, three DTs overlapping the group-of GT.
    // All DTs should be TPs (one matches non-group, others match group-of).
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img1.jpg".into(),
            height: 640,
            width: 640,
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
                bbox: Some([300.0, 300.0, 100.0, 100.0]),
                area: Some(10000.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None, // NOT group-of — provides recall denominator
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([0.0, 0.0, 200.0, 200.0]),
                area: Some(40000.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: Some(true), // Group-of GT
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "person".into(),
            supercategory: None,
            skeleton: None,
            keypoints: None,
            frequency: None,
        }],
        licenses: vec![],
    };

    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: Some([300.0, 300.0, 100.0, 100.0]),
                area: Some(10000.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: Some(0.9),
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([10.0, 10.0, 80.0, 80.0]),
                area: Some(6400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: Some(0.8),
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 3,
                image_id: 1,
                category_id: 1,
                bbox: Some([50.0, 50.0, 80.0, 80.0]),
                area: Some(6400.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: Some(0.7),
                obb: None,
                is_group_of: None,
            },
        ],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);
    let mut ev = COCOeval::new_oid(coco_gt, coco_dt, None);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().unwrap();
    // DT1 matches non-group GT. DT2+DT3 match group-of GT (multi-match).
    // All 3 DTs are TPs. Recall denominator = 1 (only non-group GT). AP should be 1.0.
    assert!(
        stats[0] > 0.99,
        "AP should be ~1.0 with group-of multi-match, got {:.4}",
        stats[0]
    );
}

#[test]
fn test_oid_group_of_no_fn_penalty() {
    // Group-of GT with NO detections — should NOT count as FN
    let gt_dataset = Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img1.jpg".into(),
            height: 640,
            width: 640,
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
                bbox: Some([0.0, 0.0, 100.0, 100.0]),
                area: Some(10000.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None, // NOT group-of
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([400.0, 400.0, 100.0, 100.0]),
                area: Some(10000.0),
                iscrowd: false,
                segmentation: None,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: Some(true), // Group-of, no detection nearby
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "person".into(),
            supercategory: None,
            skeleton: None,
            keypoints: None,
            frequency: None,
        }],
        licenses: vec![],
    };

    let dt_dataset = Dataset {
        info: None,
        images: gt_dataset.images.clone(),
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([0.0, 0.0, 100.0, 100.0]),
            area: Some(10000.0),
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: Some(0.9),
            obb: None,
            is_group_of: None,
        }],
        categories: gt_dataset.categories.clone(),
        licenses: vec![],
    };

    let coco_gt = COCO::from_dataset(gt_dataset);
    let coco_dt = COCO::from_dataset(dt_dataset);
    let mut ev = COCOeval::new_oid(coco_gt, coco_dt, None);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    let stats = ev.stats().unwrap();
    // Detection matches non-group GT perfectly. Group-of GT unmatched but NOT FN.
    // AP should be 1.0.
    assert!(
        stats[0] > 0.99,
        "AP should be ~1.0 (unmatched group-of is not FN), got {:.4}",
        stats[0]
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
            category_id: 1, // Poodle
            bbox: Some([10.0, 10.0, 100.0, 100.0]),
            area: Some(10000.0),
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: None,
            obb: None,
            is_group_of: None,
        }],
        categories: vec![
            Category {
                id: 1,
                name: "poodle".into(),
                supercategory: Some("dog".into()),
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
            Category {
                id: 2,
                name: "dog".into(),
                supercategory: Some("animal".into()),
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
            Category {
                id: 3,
                name: "animal".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
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
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: Some(0.9),
            obb: None,
            is_group_of: None,
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
        license: None,
        coco_url: None,
        flickr_url: None,
        date_captured: None,
        neg_category_ids: vec![],
        not_exhaustive_category_ids: vec![],
    };
    let cats = vec![
        Category {
            id: 1,
            name: "dog".into(),
            supercategory: Some("animal".into()),
            skeleton: None,
            keypoints: None,
            frequency: None,
        },
        Category {
            id: 2,
            name: "animal".into(),
            supercategory: None,
            skeleton: None,
            keypoints: None,
            frequency: None,
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
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: None,
            obb: None,
            is_group_of: None,
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
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: Some(0.9),
            obb: None,
            is_group_of: None,
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
            bbox: Some([10.0, 10.0, 100.0, 100.0]),
            area: Some(10000.0),
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: None,
            obb: None,
            is_group_of: None,
        }],
        categories: vec![
            Category {
                id: 1,
                name: "dog".into(),
                supercategory: Some("animal".into()),
                skeleton: None,
                keypoints: None,
                frequency: None,
            },
            Category {
                id: 2,
                name: "animal".into(),
                supercategory: None,
                skeleton: None,
                keypoints: None,
                frequency: None,
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
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: Some(0.9),
            obb: None,
            is_group_of: None,
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
                bbox: Some([10.0, 10.0, 30.0, 30.0]),
                area: Some(900.0),
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([60.0, 60.0, 30.0, 30.0]),
                area: Some(900.0),
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
                score: None,
                obb: None,
                is_group_of: None,
            },
        ],
        categories: vec![Category {
            id: 1,
            name: "obj".into(),
            supercategory: Some("".into()),
            skeleton: None,
            keypoints: None,
            frequency: None,
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
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
                score: Some(0.9),
                obb: None,
                is_group_of: None,
            },
            // TP: matches GT 2
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: Some([60.0, 60.0, 30.0, 30.0]),
                area: Some(900.0),
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
                score: Some(0.9),
                obb: None,
                is_group_of: None,
            },
            // FP: no matching GT
            Annotation {
                id: 3,
                image_id: 1,
                category_id: 1,
                bbox: Some([0.0, 0.0, 5.0, 5.0]),
                area: Some(25.0),
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
                score: Some(0.2),
                obb: None,
                is_group_of: None,
            },
            // FP: no matching GT
            Annotation {
                id: 4,
                image_id: 1,
                category_id: 1,
                bbox: Some([90.0, 90.0, 5.0, 5.0]),
                area: Some(25.0),
                segmentation: None,
                iscrowd: false,
                keypoints: None,
                num_keypoints: None,
                score: Some(0.2),
                obb: None,
                is_group_of: None,
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
        supercategory: None,
        skeleton: None,
        keypoints: None,
        frequency: None,
    };

    let images: Vec<Image> = (1..=n)
        .map(|i| Image {
            id: i as u64,
            file_name: format!("img{i}.jpg"),
            height: 200,
            width: 200,
            license: None,
            coco_url: None,
            flickr_url: None,
            date_captured: None,
            neg_category_ids: vec![],
            not_exhaustive_category_ids: vec![],
        })
        .collect();

    let gt_anns: Vec<Annotation> = (1..=n)
        .map(|i| Annotation {
            id: i as u64,
            image_id: i as u64,
            category_id: 1,
            bbox: Some([10.0, 10.0, 50.0, 50.0]),
            area: Some(2500.0),
            iscrowd: false,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
            score: None,
            obb: None,
            is_group_of: None,
        })
        .collect();

    let make_ann = |id: u64, image_id: u64, bbox: [f64; 4], score: f64| Annotation {
        id,
        image_id,
        category_id: 1,
        bbox: Some(bbox),
        area: Some(bbox[2] * bbox[3]),
        iscrowd: false,
        segmentation: None,
        keypoints: None,
        num_keypoints: None,
        score: Some(score),
        obb: None,
        is_group_of: None,
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

fn obb_test_image() -> Image {
    Image {
        id: 1,
        file_name: "obb_test.png".into(),
        width: 800,
        height: 600,
        license: None,
        coco_url: None,
        flickr_url: None,
        date_captured: None,
        neg_category_ids: vec![],
        not_exhaustive_category_ids: vec![],
    }
}

fn obb_test_category() -> Category {
    Category {
        id: 1,
        name: "vehicle".into(),
        supercategory: None,
        skeleton: None,
        keypoints: None,
        frequency: None,
    }
}

#[test]
fn test_obb_eval_basic() {
    // GT: one rotated box, DT: same box with high score → AP ≈ 1.0
    let gt_dataset = Dataset {
        info: None,
        images: vec![obb_test_image()],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([90.0, 90.0, 220.0, 120.0]),
            area: Some(20000.0),
            segmentation: None,
            iscrowd: false,
            keypoints: None,
            num_keypoints: None,
            obb: Some([200.0, 150.0, 200.0, 100.0, 0.3]),
            score: None,
            is_group_of: None,
        }],
        categories: vec![obb_test_category()],
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
            segmentation: None,
            iscrowd: false,
            keypoints: None,
            num_keypoints: None,
            obb: Some([200.0, 150.0, 200.0, 100.0, 0.3]),
            score: Some(0.99),
            is_group_of: None,
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
        images: vec![obb_test_image()],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([0.0, 0.0, 50.0, 50.0]),
            area: Some(2500.0),
            segmentation: None,
            iscrowd: false,
            keypoints: None,
            num_keypoints: None,
            obb: Some([25.0, 25.0, 50.0, 50.0, 0.0]),
            score: None,
            is_group_of: None,
        }],
        categories: vec![obb_test_category()],
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
            segmentation: None,
            iscrowd: false,
            keypoints: None,
            num_keypoints: None,
            obb: Some([725.0, 525.0, 50.0, 50.0, 0.0]),
            score: Some(0.9),
            is_group_of: None,
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
    // AP should be 0 or -1 (no match)
    assert!(
        stats[0] <= 0.0,
        "AP should be 0 for non-overlapping OBBs, got {}",
        stats[0]
    );
}

#[test]
fn test_dota_round_trip_integration() {
    let dataset = Dataset {
        info: None,
        images: vec![obb_test_image()],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([90.0, 90.0, 220.0, 120.0]),
            area: Some(20000.0),
            segmentation: None,
            iscrowd: false,
            keypoints: None,
            num_keypoints: None,
            obb: Some([200.0, 150.0, 200.0, 100.0, 0.0]),
            score: None,
            is_group_of: None,
        }],
        categories: vec![obb_test_category()],
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
    let result = dota_to_coco(&label_dir, &dims, None).unwrap();

    assert_eq!(result.images.len(), 1);
    assert_eq!(result.annotations.len(), 1);
    assert_eq!(result.categories.len(), 1);
    assert_eq!(result.categories[0].name, "vehicle");

    let ann = &result.annotations[0];
    let obb = ann.obb.unwrap();
    // Round-trip tolerance: DOTA uses 1 decimal place formatting
    assert!((obb[0] - 200.0).abs() < 0.2, "cx round-trip: {}", obb[0]);
    assert!((obb[1] - 150.0).abs() < 0.2, "cy round-trip: {}", obb[1]);
    assert!((obb[2] - 200.0).abs() < 0.2, "w round-trip: {}", obb[2]);
    assert!((obb[3] - 100.0).abs() < 0.2, "h round-trip: {}", obb[3]);
    assert!(obb[4].abs() < 0.01, "angle round-trip: {}", obb[4]);
}
