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
