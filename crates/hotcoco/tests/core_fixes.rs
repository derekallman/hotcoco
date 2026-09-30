//! Regression tests for the 1.0 preflight core-data hardening pass.
//!
//! One test per fix: untrusted JSON (RLE strings, dims, polygon coordinates,
//! ids) must produce a clean `Err` or a documented degradation — never a
//! debug panic, an integer overflow, or plausible-looking garbage.

#![allow(clippy::unwrap_used)]

use std::io::Write;

use hotcoco::types::{Annotation, Category, Dataset, Image, Rle, Segmentation};
use hotcoco::{COCO, mask};

fn gt_dataset() -> Dataset {
    Dataset {
        info: None,
        images: vec![Image {
            id: 1,
            file_name: "img1.jpg".into(),
            height: 20,
            width: 20,
            ..Default::default()
        }],
        annotations: vec![Annotation {
            id: 1,
            image_id: 1,
            category_id: 1,
            bbox: Some([1.0, 1.0, 4.0, 4.0]),
            area: Some(16.0),
            ..Default::default()
        }],
        categories: vec![Category {
            id: 1,
            name: "thing".into(),
            ..Default::default()
        }],
        licenses: vec![],
    }
}

// ── RLE string decoding ──────────────────────────────────────────────────────

/// A LEB-style run with endless continuation bits used to grow the shift past
/// 63 and overflow `<<` (debug panic, masked shift in release).
#[test]
fn rle_from_string_unbounded_continuation_errors() {
    // 'P' = 48 + 32: payload 0, continuation bit set. 20 of them would push
    // the shift to 100.
    let s = "P".repeat(20);
    assert!(mask::rle_from_string(&s, 10, 10).is_err());
}

/// A decoded count above u32::MAX was truncated by `as u32` *before* the
/// total-vs-h*w validation, so 2^33 wrapped to 0 and "passed".
#[test]
fn rle_from_string_count_above_u32_max_errors() {
    // Groups (5 bits each, LSB first): six zero groups with continuation,
    // then value 8 => x = 8 << 30 = 2^33.
    let s = "PPPPPP8";
    let res = mask::rle_from_string(s, 10, 10);
    let err = res.unwrap_err().to_string();
    assert!(err.contains("u32::MAX"), "unexpected error: {err}");
}

/// Still accepts everything a valid encoder produces.
#[test]
fn rle_string_roundtrip_still_works() {
    let rle = Rle {
        h: 100,
        w: 100,
        counts: vec![10, 20, 30, 40, 50, 60, 9790],
    };
    let s = mask::rle_to_string(&rle);
    let back = mask::rle_from_string(&s, 100, 100).unwrap();
    assert_eq!(back.counts, rle.counts);
}

// ── Zero-length runs ─────────────────────────────────────────────────────────

/// `to_bbox` computed `cc + c - 1` on a zero-length foreground run — an
/// underflow. Zero-length odd runs are legal in decoded RLE strings.
#[test]
fn to_bbox_zero_length_foreground_run() {
    let rle = Rle {
        h: 5,
        w: 5,
        counts: vec![5, 0, 20],
    };
    assert_eq!(mask::to_bbox(&rle), [0.0, 0.0, 0.0, 0.0]);

    // Zero run between real runs: bbox is that of the real foreground only.
    let rle2 = Rle {
        h: 5,
        w: 5,
        counts: vec![5, 0, 1, 3, 16],
    };
    // Foreground pixels: flat index 6..9 → column 1, rows 1..4.
    assert_eq!(mask::to_bbox(&rle2), [1.0, 1.0, 1.0, 3.0]);
}

// ── Dimension overflow (h * w computed in u32 with JSON-supplied dims) ───────

#[test]
fn huge_dims_error_instead_of_overflowing() {
    let h = 100_000;
    let w = 100_000; // h*w = 1e10 > u32::MAX
    assert!(mask::fr_poly(&[1.0, 1.0, 5.0, 1.0, 3.0, 4.0], h, w).is_err());
    assert!(mask::fr_polys(&[vec![1.0, 1.0, 5.0, 1.0, 3.0, 4.0]], h, w).is_err());
    assert!(mask::fr_bbox(&[1.0, 1.0, 2.0, 2.0], h, w).is_err());
    assert!(mask::encode(&[0u8; 16], h, w).is_err());
    let r = Rle {
        h,
        w,
        counts: vec![0],
    };
    assert!(mask::merge(&[r.clone(), r], false).is_err());
}

// ── Polygon coordinates far outside the image ────────────────────────────────

/// ±1e9 coordinates overflowed the rasterizer's i32 edge walk and reserved
/// multi-GB boundary buffers. They now clamp; the mask stays inside the image.
#[test]
fn fr_poly_extreme_coords_no_panic() {
    let poly = vec![1e9, 1e9, -1e9, 5.0, 3.0, -1e9];
    let rle = mask::fr_poly(&poly, 100, 100).unwrap();
    assert!(mask::area(&rle) <= 100 * 100);

    // Non-finite coordinates must not panic either.
    let weird = vec![f64::NAN, 2.0, f64::INFINITY, 5.0, 3.0, f64::NEG_INFINITY];
    let rle2 = mask::fr_poly(&weird, 100, 100).unwrap();
    assert!(mask::area(&rle2) <= 100 * 100);

    // A normal polygon still rasterizes identically to the reference value.
    let tri = vec![2.0, 2.0, 7.0, 2.0, 4.0, 7.0];
    let rle3 = mask::fr_poly(&tri, 10, 10).unwrap();
    assert_eq!(mask::area(&rle3), 12);
}

#[test]
fn fr_bbox_non_finite_coords_no_panic() {
    let rle = mask::fr_bbox(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0], 10, 10).unwrap();
    assert_eq!(mask::area(&rle), 0);
}

// ── encode / merge validation ────────────────────────────────────────────────

/// `encode` used to `assert_eq!` on the length — a panic on untrusted input
/// where the sibling `rle_from_string` returns `Err`.
#[test]
fn encode_length_mismatch_errors() {
    assert!(mask::encode(&[0u8; 5], 3, 4).is_err());
    assert!(mask::encode(&[0u8; 12], 3, 4).is_ok());
}

/// `merge` accepted RLEs with differing dims and stamped the first mask's dims
/// on the result.
#[test]
fn merge_mismatched_dims_errors() {
    let a = Rle {
        h: 3,
        w: 4,
        counts: vec![12],
    };
    let b = Rle {
        h: 4,
        w: 3,
        counts: vec![12],
    };
    let err = mask::merge(&[a.clone(), b], false).unwrap_err().to_string();
    assert!(err.contains("dimensions"), "unexpected error: {err}");
    assert!(mask::merge(&[a.clone(), a], false).is_ok());
}

// ── Rle::new is a real validated constructor now ─────────────────────────────

#[test]
fn rle_new_validates_in_release_builds_too() {
    assert!(Rle::new(3, 4, vec![6, 6]).is_ok());
    assert!(Rle::new(3, 4, vec![6, 7]).is_err());
}

// ── corners_to_obb length check ──────────────────────────────────────────────

#[test]
fn corners_to_obb_length_checked() {
    use hotcoco::geometry::corners_to_obb;
    assert!(corners_to_obb(&[0.0; 7]).is_err());
    assert!(corners_to_obb(&[0.0; 9]).is_err());
    let obb = corners_to_obb(&[0.0, 0.0, 4.0, 0.0, 4.0, 3.0, 0.0, 3.0]).unwrap();
    assert!((obb[2] - 4.0).abs() < 1e-9);
    assert!((obb[3] - 3.0).abs() < 1e-9);
}

// ── COCO::merge id-offset overflow ───────────────────────────────────────────

#[test]
fn coco_merge_id_offset_overflow_errors() {
    let mut ds1 = gt_dataset();
    ds1.images[0].id = u64::MAX;
    ds1.annotations[0].image_id = u64::MAX;
    let ds2 = gt_dataset();
    // Second dataset's ids get offset by u64::MAX → must error, not wrap into
    // silent id collisions.
    assert!(COCO::merge(&[&ds1, &ds2]).is_err());

    // Sane inputs still merge.
    let a = gt_dataset();
    let b = gt_dataset();
    let merged = COCO::merge(&[&a, &b]).unwrap();
    assert_eq!(merged.images.len(), 2);
    let mut ids: Vec<u64> = merged.images.iter().map(|i| i.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 2, "image ids must stay unique after merge");
}

// ── loadRes kind inference (pycocotools parity) ──────────────────────────────

/// pycocotools' loadRes checks `bbox`, then `segmentation`, then `keypoints`.
/// A results file whose annotations carry both a segmentation and keypoints
/// must be treated as segmentation results.
#[test]
fn load_res_segmentation_wins_over_keypoints() {
    let gt = COCO::from_dataset(gt_dataset());

    // A 4x4 box at (2,2) as a compressed RLE on the 20x20 image.
    let box_rle = mask::fr_bbox(&[2.0, 2.0, 4.0, 4.0], 20, 20).unwrap();
    let counts = mask::rle_to_string(&box_rle);

    let det = Annotation {
        image_id: 1,
        category_id: 1,
        segmentation: Some(Segmentation::CompressedRle {
            size: [20, 20],
            counts,
        }),
        // Keypoint extent is deliberately different from the mask bbox.
        keypoints: Some(vec![0.0, 0.0, 2.0, 10.0, 10.0, 2.0]),
        score: Some(0.9),
        ..Default::default()
    };
    let res = gt.load_res_anns(vec![det]).unwrap();
    let ann = &res.dataset.annotations[0];
    // Derived from the mask, not from the keypoint extent [0,0,10,10].
    assert_eq!(ann.bbox, Some([2.0, 2.0, 4.0, 4.0]));
    assert_eq!(ann.area, Some(16.0));
}

/// Empty keypoints used to fold to a [inf, inf, -inf, -inf] bbox with
/// infinite area, flowing into eval unflagged.
#[test]
fn load_res_empty_keypoints_derives_nothing() {
    let gt = COCO::from_dataset(gt_dataset());
    let det = Annotation {
        image_id: 1,
        category_id: 1,
        keypoints: Some(vec![]),
        score: Some(0.5),
        ..Default::default()
    };
    let res = gt.load_res_anns(vec![det]).unwrap();
    let ann = &res.dataset.annotations[0];
    assert_eq!(
        ann.bbox, None,
        "no bbox may be derived from empty keypoints"
    );
    assert_eq!(
        ann.area, None,
        "no area may be derived from empty keypoints"
    );
}

// ── Duplicate annotation ids ─────────────────────────────────────────────────

/// pycocotools' createIndex is last-write-wins in `anns` and appends every
/// occurrence to `imgToAnns`; both behaviors are kept, but the condition is
/// surfaced through `load_warnings()` instead of being silent.
#[test]
fn duplicate_ann_ids_warn_and_match_pycocotools() {
    let mut ds = gt_dataset();
    ds.annotations.push(Annotation {
        id: 1, // duplicate of the existing annotation id
        image_id: 1,
        category_id: 1,
        bbox: Some([5.0, 5.0, 2.0, 2.0]),
        area: Some(4.0),
        ..Default::default()
    });
    let coco = COCO::from_dataset(ds);

    // Warning surfaced.
    assert!(
        coco.load_warnings()
            .iter()
            .any(|w| w.contains("duplicate annotation id")),
        "expected a duplicate-id warning, got {:?}",
        coco.load_warnings()
    );
    // Last write wins in the id lookup.
    let ann = coco.get_ann(1).unwrap();
    assert_eq!(ann.bbox, Some([5.0, 5.0, 2.0, 2.0]));
    // Per-image list keeps both occurrences.
    assert_eq!(coco.get_ann_ids_for_img(1), &[1, 1]);
}

// ── Load warnings for results validation ─────────────────────────────────────

#[test]
fn load_res_unknown_image_id_recorded_in_warnings() {
    let gt = COCO::from_dataset(gt_dataset());
    let det = Annotation {
        image_id: 999, // not in GT
        category_id: 1,
        bbox: Some([0.0, 0.0, 1.0, 1.0]),
        score: Some(0.5),
        ..Default::default()
    };
    let res = gt.load_res_anns(vec![det]).unwrap();
    assert!(
        res.load_warnings()
            .iter()
            .any(|w| w.contains("image_id 999")),
        "expected an unknown-image_id warning, got {:?}",
        res.load_warnings()
    );

    // A clean load carries no warnings.
    let clean = COCO::from_dataset(gt_dataset());
    assert!(clean.load_warnings().is_empty());
}

// ── Missing-area convention ──────────────────────────────────────────────────

/// An annotation without an `area` never matches an explicit area range
/// (pycocotools raises KeyError there; it never fabricates area = 0.0).
#[test]
fn missing_area_excluded_from_area_range() {
    let mut ds = gt_dataset();
    ds.annotations.push(Annotation {
        id: 2,
        image_id: 1,
        category_id: 1,
        bbox: Some([0.0, 0.0, 3.0, 3.0]),
        area: None,
        ..Default::default()
    });
    let coco = COCO::from_dataset(ds);

    // Without a range: both annotations.
    assert_eq!(coco.get_ann_ids(&[], &[], None, None), vec![1, 2]);
    // With a range that would have matched a fabricated 0.0 area: only the
    // annotation that actually has an area.
    assert_eq!(coco.get_ann_ids(&[], &[], Some([0.0, 1e10]), None), vec![1]);
    let filtered = coco.filter(None, None, Some([0.0, 1e10]), false);
    assert_eq!(filtered.annotations.len(), 1);
    assert_eq!(filtered.annotations[0].id, 1);
}

// ── Custom-key preservation ──────────────────────────────────────────────────

/// Unknown JSON keys on images, annotations, and categories survive
/// load → filter → save (pycocotools preserves them because it stores raw
/// dicts). Exercises the JSON loader end to end.
#[test]
fn extra_keys_survive_load_and_filter() {
    let json = r#"{
        "images": [{"id": 1, "file_name": "a.jpg", "height": 10, "width": 10,
                    "camera": "rig-3"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1.0, 1.0, 2.0, 2.0], "area": 4.0,
                         "confidence_source": "human", "track_id": 17}],
        "categories": [{"id": 1, "name": "thing", "taxonomy_code": "T-9"}]
    }"#;

    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(json.as_bytes()).unwrap();
    let coco = COCO::new(tmp.path()).unwrap();

    assert_eq!(
        coco.dataset.images[0].extra.get("camera"),
        Some(&serde_json::Value::from("rig-3"))
    );
    assert_eq!(
        coco.dataset.annotations[0].extra.get("track_id"),
        Some(&serde_json::Value::from(17))
    );
    assert_eq!(
        coco.dataset.categories[0].extra.get("taxonomy_code"),
        Some(&serde_json::Value::from("T-9"))
    );

    // Survives filter (clone-based) and merge.
    let filtered = coco.filter(Some(&[1]), None, None, false);
    assert_eq!(
        filtered.annotations[0].extra.get("confidence_source"),
        Some(&serde_json::Value::from("human"))
    );
    let merged = COCO::merge(&[&filtered]).unwrap();
    assert_eq!(
        merged.images[0].extra.get("camera"),
        Some(&serde_json::Value::from("rig-3"))
    );
}

// ── Manual load-timing guard ─────────────────────────────────────────────────

/// Not a pass/fail gate — prints load time for the flatten-field change.
/// Run with:
/// `cargo test -p hotcoco --release --test core_fixes -- --ignored --nocapture`
#[test]
#[ignore = "needs data/annotations/instances_val2017.json; timing is machine-dependent"]
fn load_timing_val2017() {
    let path = std::path::Path::new("../../data/annotations/instances_val2017.json");
    if !path.exists() {
        eprintln!("val2017 annotations not present; skipping");
        return;
    }
    // Warm-up + 3 timed runs.
    let _ = COCO::new(path).unwrap();
    for i in 0..3 {
        let t = std::time::Instant::now();
        let coco = COCO::new(path).unwrap();
        eprintln!(
            "run {i}: loaded {} anns in {:?}",
            coco.dataset.annotations.len(),
            t.elapsed()
        );
    }
}

// ---------------------------------------------------------------------------
// Issue #5: a category record without `name` (pycocotools tolerates it,
// TorchMetrics emits it) loads with the `cat_<id>` placeholder, from both
// the JSON loader and `from_dataset`, and the load is flagged.
// ---------------------------------------------------------------------------

#[test]
fn category_without_name_loads_with_placeholder() {
    let json = r#"{"images":[{"id":1,"height":4,"width":4}],
                   "categories":[{"id":7},{"id":8,"name":"dog"}],
                   "annotations":[]}"#;
    let coco = load_json(json).unwrap();
    assert_eq!(coco.get_cat(7).unwrap().name, "cat_7");
    assert_eq!(coco.get_cat(8).unwrap().name, "dog");
    assert_eq!(coco.cat_name(7), "cat_7", "cat_name and the record agree");
    assert_eq!(
        coco.cat_name(99),
        "cat_99",
        "unknown id uses the same placeholder"
    );
    assert_eq!(coco.load_warnings().len(), 1);
    assert!(coco.load_warnings()[0].contains("without a name"));

    let mut ds = gt_dataset();
    ds.categories.push(Category {
        id: 3,
        ..Default::default()
    });
    let coco = COCO::from_dataset(ds);
    assert_eq!(coco.get_cat(3).unwrap().name, "cat_3");
}

// ---------------------------------------------------------------------------
// Integral floats spell ints in a JSON written through pandas or numpy, and
// pycocotools accepts them because `1.0 == 1`. The loader applies the same
// rule as the Python dict path; fractional and negative values still fail.
// ---------------------------------------------------------------------------

fn load_json(json: &str) -> hotcoco::error::Result<COCO> {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(json.as_bytes()).unwrap();
    COCO::new(f.path())
}

#[test]
fn json_integral_floats_are_ints() {
    let coco = load_json(
        r#"{"images":[{"id":1.0,"height":4.0,"width":4.0,"license":2.0},{"id":2}],
            "categories":[{"id":1.0,"name":"a"}],
            "annotations":[{"id":1.0,"image_id":1.0,"category_id":1.0,"bbox":[0,0,1,1],
                            "area":1,"iscrowd":1.0,"is_group_of":1,"num_keypoints":3.0}]}"#,
    )
    .unwrap();
    let img = coco.get_img(1).unwrap();
    assert_eq!((img.height, img.width, img.license), (4, 4, Some(2)));
    let bare = coco.get_img(2).unwrap();
    assert_eq!(
        (bare.height, bare.width),
        (0, 0),
        "size is optional, as on the dict path"
    );
    let ann = coco.get_ann(1).unwrap();
    assert_eq!((ann.image_id, ann.category_id), (1, 1));
    assert!(ann.iscrowd, "iscrowd 1.0 is true");
    assert_eq!(
        ann.is_group_of,
        Some(true),
        "is_group_of 1 is true, as on the dict path"
    );
    assert_eq!(ann.num_keypoints, Some(3));
    assert_eq!(coco.get_cat(1).unwrap().name, "a");

    for bad in [r#"{"id":1.5}"#, r#"{"id":-1}"#, r#"{"id":"1"}"#] {
        let json = format!(r#"{{"images":[{bad}],"categories":[],"annotations":[]}}"#);
        assert!(load_json(&json).is_err(), "{bad} must not load");
    }
    let fractional_flag = r#"{"images":[{"id":1}],"categories":[],
        "annotations":[{"id":1,"image_id":1,"category_id":1,"iscrowd":0.5}]}"#;
    assert!(
        load_json(fractional_flag).is_err(),
        "iscrowd 0.5 is not a flag"
    );
}
