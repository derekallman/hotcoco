//! Regression tests for the 2026-10 whole-project review.
//!
//! Each test pins a fix for a case where hotcoco produced a plausible wrong
//! number where pycocotools raises, or where its reference-free paths diverged
//! from the reference they were meant to track. The reviewer's reproducers are
//! the test inputs.

#![allow(clippy::unwrap_used)]

use hotcoco::params::IouType;
use hotcoco::types::{Annotation, Category, Dataset, Image, Segmentation};
use hotcoco::{COCO, COCOeval, EvalMode, Params, StreamingEval};

fn image(id: u64, dims: Option<(u32, u32)>) -> Image {
    let (height, width) = dims.unwrap_or_default();
    Image {
        id,
        width,
        height,
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

fn det_box(image_id: u64, category_id: u64, bbox: [f64; 4], score: f64) -> Annotation {
    Annotation {
        image_id,
        category_id,
        bbox: Some(bbox),
        score: Some(score),
        ..Default::default()
    }
}

/// A square polygon with the box's corners, the way a COCO file spells it.
fn square(x: f64, y: f64, s: f64) -> Vec<f64> {
    vec![x, y, x + s, y, x + s, y + s, x, y + s]
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

fn stats(ev: &mut COCOeval) -> Vec<f64> {
    ev.evaluate().expect("evaluable inputs");
    ev.accumulate();
    ev.summarize();
    ev.stats().unwrap().to_vec()
}

// ---------------------------------------------------------------------------
// Missing image height/width: an error, not segm AP 0
// ---------------------------------------------------------------------------

/// A polygon on an image record without `height`/`width` used to rasterize
/// onto a 0×0 canvas: an empty mask, segm AP 0.000, and no warning.
/// pycocotools raises `KeyError` at the same point.
#[test]
fn check_mask_dims_rejects_polygons_on_dimensionless_images() {
    let poly = Segmentation::Polygon(vec![square(10.0, 10.0, 50.0)]);
    let gt = COCO::from_dataset(Dataset {
        images: vec![image(1, None), image(2, Some((100, 100)))],
        annotations: vec![
            Annotation {
                segmentation: Some(poly.clone()),
                ..gt_box(1, 1, 1, [10.0, 10.0, 50.0, 50.0])
            },
            Annotation {
                segmentation: Some(poly),
                ..gt_box(2, 2, 1, [10.0, 10.0, 50.0, 50.0])
            },
        ],
        categories: vec![cat(1)],
        ..Default::default()
    });

    let err = gt.check_mask_dims(&[1, 2], None).unwrap_err().to_string();
    assert!(err.contains("ids 1)"), "names the offending image: {err}");
    assert!(!err.contains("ids 1, 2"), "image 2 has dims: {err}");
    gt.check_mask_dims(&[2], None).unwrap();

    // `check_inputs` reads the iou type as it stands at evaluate time, so a
    // bbox evaluator switched to segm after construction is still checked.
    let dt = gt.load_res_anns(vec![]).unwrap();
    let mut ev = COCOeval::new(gt.clone(), dt, IouType::Bbox);
    ev.check_inputs().unwrap();
    ev.params.iou_type = IouType::Segm;
    assert!(ev.check_inputs().is_err());

    // `evaluate()` and `run()` make the same check themselves, so a Rust
    // caller gets the error rather than a 0.000: nothing is evaluated.
    assert!(ev.evaluate().is_err());
    assert!(ev.run().is_err());
    assert!(!ev.evaluated());
    ev.params.iou_type = IouType::Bbox;
    ev.run().expect("box evaluation never reads the dims");

    // The annotation on the dimensionless image has no mask at all — never a
    // 0×0 one that compares as empty against everything.
    assert!(gt.ann_to_rle(&gt.dataset.annotations[0]).is_none());
    assert!(gt.ann_to_rle(&gt.dataset.annotations[1]).is_some());
}

/// RLE carries its own size and box evaluation never reads the image's, so
/// neither trips the check.
#[test]
fn check_mask_dims_accepts_rle_and_dimensionless_bbox_only_datasets_in_bbox_mode() {
    let rle = COCO::from_dataset(Dataset {
        images: vec![image(1, None)],
        annotations: vec![Annotation {
            segmentation: Some(Segmentation::UncompressedRle {
                size: [10, 10],
                counts: vec![0, 100],
            }),
            ..gt_box(1, 1, 1, [0.0, 0.0, 10.0, 10.0])
        }],
        categories: vec![cat(1)],
        ..Default::default()
    });
    rle.check_mask_dims(&[1], None).unwrap();

    // Boxes do rasterize in segm mode, so a bbox-only dataset is rejected
    // there — but it is still a perfectly good bbox dataset.
    let boxes = COCO::from_dataset(Dataset {
        images: vec![image(1, None)],
        annotations: vec![gt_box(1, 1, 1, [0.0, 0.0, 10.0, 10.0])],
        categories: vec![cat(1)],
        ..Default::default()
    });
    assert!(boxes.check_mask_dims(&[1], None).is_err());
    let dt = boxes
        .load_res_anns(vec![det_box(1, 1, [0.0, 0.0, 10.0, 10.0], 0.9)])
        .unwrap();
    let s = stats(&mut COCOeval::new(boxes, dt, IouType::Bbox));
    assert_close(s[0], 1.0);
}

/// Only images the evaluation covers are checked, as pycocotools only reads
/// `height`/`width` for those. An annotation whose `image_id` has no image
/// record is never evaluated (the default `img_ids` are the image records), so
/// it is not an error.
#[test]
fn check_inputs_ignores_orphan_annotations() {
    let poly = Segmentation::Polygon(vec![square(10.0, 10.0, 50.0)]);
    let gt = COCO::from_dataset(Dataset {
        images: vec![image(1, Some((100, 100)))],
        annotations: vec![
            Annotation {
                segmentation: Some(poly.clone()),
                ..gt_box(1, 1, 1, [10.0, 10.0, 50.0, 50.0])
            },
            // No image 99 in the dataset.
            Annotation {
                segmentation: Some(poly.clone()),
                ..gt_box(2, 99, 1, [10.0, 10.0, 50.0, 50.0])
            },
        ],
        categories: vec![cat(1)],
        ..Default::default()
    });
    let dt = gt
        .load_res_anns(vec![Annotation {
            segmentation: Some(poly),
            ..det_box(1, 1, [10.0, 10.0, 50.0, 50.0], 0.9)
        }])
        .unwrap();
    let mut ev = COCOeval::new(gt, dt, IouType::Segm);
    ev.check_inputs().unwrap();
    assert_close(stats(&mut ev)[0], 1.0);
}

/// A dimensionless image that `params.img_ids` leaves out is never rasterized,
/// so it does not block the images that are evaluated.
#[test]
fn check_inputs_ignores_dimensionless_images_outside_img_ids() {
    let poly = Segmentation::Polygon(vec![square(10.0, 10.0, 50.0)]);
    let gt = COCO::from_dataset(Dataset {
        images: vec![image(1, Some((100, 100))), image(2, None)],
        annotations: vec![
            Annotation {
                segmentation: Some(poly.clone()),
                ..gt_box(1, 1, 1, [10.0, 10.0, 50.0, 50.0])
            },
            Annotation {
                segmentation: Some(poly.clone()),
                ..gt_box(2, 2, 1, [10.0, 10.0, 50.0, 50.0])
            },
        ],
        categories: vec![cat(1)],
        ..Default::default()
    });
    let dt = gt
        .load_res_anns(vec![Annotation {
            segmentation: Some(poly),
            ..det_box(1, 1, [10.0, 10.0, 50.0, 50.0], 0.9)
        }])
        .unwrap();
    let mut ev = COCOeval::new(gt, dt, IouType::Segm);
    assert!(
        ev.check_inputs().is_err(),
        "image 2 is in the default scope"
    );
    ev.params.img_ids = vec![1];
    ev.check_inputs().unwrap();
    assert_close(stats(&mut ev)[0], 1.0);
}

#[test]
fn streaming_segm_update_errors_on_dimensionless_image() {
    let mut se =
        StreamingEval::new(Params::new(IouType::Segm), EvalMode::Coco, vec![cat(1)]).unwrap();
    let poly = Segmentation::Polygon(vec![square(10.0, 10.0, 50.0)]);
    let gt = Annotation {
        segmentation: Some(poly.clone()),
        ..gt_box(1, 1, 1, [10.0, 10.0, 50.0, 50.0])
    };
    let dt = Annotation {
        segmentation: Some(poly),
        ..det_box(1, 1, [10.0, 10.0, 50.0, 50.0], 0.9)
    };

    let err = se
        .update(vec![image(1, None)], vec![gt.clone()], vec![dt.clone()])
        .unwrap_err()
        .to_string();
    assert!(err.contains("height"), "{err}");

    // The same batch with dims is a perfect match.
    se.update(vec![image(1, Some((100, 100)))], vec![gt], vec![dt])
        .unwrap();
    let mut ev = se.finalize();
    ev.accumulate();
    ev.summarize();
    assert_close(ev.stats().unwrap()[0], 1.0);
}

// ---------------------------------------------------------------------------
// Streaming ground truth: ids assigned, area derived
// ---------------------------------------------------------------------------

/// Targets from a data loader carry no `id`. Every ground truth then shared
/// id 0, and the id index resolved all of them to the batch's last annotation
/// — possibly in another image and category. Here image 1's only GT is a
/// `[0, 0, 10, 10]` box in category 1 and image 2's is `[100, 100, 10, 10]`
/// in category 3; before the fix image 1's detection was scored against image
/// 2's box and missed.
#[test]
fn streaming_update_assigns_gt_ids() {
    let cats = vec![cat(1), cat(3)];
    let images = vec![image(1, Some((200, 200))), image(2, Some((200, 200)))];
    let gts = vec![
        Annotation {
            id: 0,
            area: None,
            ..gt_box(0, 1, 1, [0.0, 0.0, 10.0, 10.0])
        },
        Annotation {
            id: 0,
            area: None,
            ..gt_box(0, 2, 3, [100.0, 100.0, 10.0, 10.0])
        },
    ];
    let dts = vec![
        det_box(1, 1, [0.0, 0.0, 10.0, 10.0], 0.9),
        det_box(2, 3, [100.0, 100.0, 10.0, 10.0], 0.8),
    ];

    let mut se =
        StreamingEval::new(Params::new(IouType::Bbox), EvalMode::Coco, cats.clone()).unwrap();
    se.update(images.clone(), gts.clone(), dts.clone()).unwrap();
    let mut streamed = se.finalize();
    streamed.accumulate();
    streamed.summarize();

    // The batch path with proper ids and areas is the reference.
    let gt_coco = COCO::from_dataset(Dataset {
        images,
        annotations: vec![
            gt_box(1, 1, 1, [0.0, 0.0, 10.0, 10.0]),
            gt_box(2, 2, 3, [100.0, 100.0, 10.0, 10.0]),
        ],
        categories: cats,
        ..Default::default()
    });
    let dt_coco = gt_coco.load_res_anns(dts).unwrap();
    let batch = stats(&mut COCOeval::new(gt_coco, dt_coco, IouType::Bbox));

    assert_eq!(streamed.stats().unwrap(), batch.as_slice());
    assert_close(batch[0], 1.0);
}

/// Ground truth without `area` was read as area 0 by the matcher, so every
/// object landed in `small` while its detection (area derived as `w × h`)
/// landed where it belonged: APm/APl went to -1 and APs counted everything.
#[test]
fn streaming_update_derives_missing_gt_area_from_bbox() {
    // A 60×60 box: medium (32² ≤ area < 96²).
    let bbox = [10.0, 10.0, 60.0, 60.0];
    let images = vec![image(1, Some((200, 200)))];
    let gt_no_area = Annotation {
        area: None,
        ..gt_box(1, 1, 1, bbox)
    };
    let dts = vec![det_box(1, 1, bbox, 0.9)];

    let mut se =
        StreamingEval::new(Params::new(IouType::Bbox), EvalMode::Coco, vec![cat(1)]).unwrap();
    se.update(images.clone(), vec![gt_no_area], dts.clone())
        .unwrap();
    let mut streamed = se.finalize();
    streamed.accumulate();
    streamed.summarize();
    let s = streamed.stats().unwrap();

    // AP, APs, APm, APl at indices 0, 3, 4, 5.
    assert_close(s[0], 1.0);
    assert_eq!(s[3], -1.0, "no small ground truth: APs is not computed");
    assert_close(s[4], 1.0);
    assert_eq!(s[5], -1.0, "no large ground truth: APl is not computed");
}

/// Segm streaming derives an area-less polygon's area from its mask, and the
/// IoUs from the same mask: the batch rasterizes each such polygon once and
/// keeps the RLE. The triangle's box is `large` and its mask `medium`, so a
/// box-derived area, or a mask lost between the two uses, would show up here.
#[test]
fn streaming_segm_derives_area_from_the_mask_it_matches_on() {
    let tri = vec![10.0, 10.0, 110.0, 10.0, 10.0, 110.0];
    let bbox = [10.0, 10.0, 100.0, 100.0];
    let poly = Segmentation::Polygon(vec![tri.clone()]);
    let images = vec![image(1, Some((200, 200)))];
    let gt = Annotation {
        area: None,
        segmentation: Some(poly.clone()),
        ..gt_box(1, 1, 1, bbox)
    };
    let dt = Annotation {
        segmentation: Some(poly),
        ..det_box(1, 1, bbox, 0.9)
    };

    let mut se =
        StreamingEval::new(Params::new(IouType::Segm), EvalMode::Coco, vec![cat(1)]).unwrap();
    se.update(images.clone(), vec![gt.clone()], vec![dt.clone()])
        .unwrap();
    let mut streamed = se.finalize();
    streamed.accumulate();
    streamed.summarize();

    let mask_area = hotcoco::mask::area(&hotcoco::mask::fr_poly(&tri, 200, 200).unwrap()) as f64;
    let gt_coco = COCO::from_dataset(Dataset {
        images,
        annotations: vec![Annotation {
            area: Some(mask_area),
            ..gt
        }],
        categories: vec![cat(1)],
        ..Default::default()
    });
    let dt_coco = gt_coco.load_res_anns(vec![dt]).unwrap();
    let batch = stats(&mut COCOeval::new(gt_coco, dt_coco, IouType::Segm));

    assert_eq!(streamed.stats().unwrap(), batch.as_slice());
    assert_close(batch[4], 1.0);
    assert_eq!(batch[5], -1.0, "the mask is medium, not large");
}

/// An authored `area` wins over the derived one: COCO files carry mask areas
/// that differ from the box's.
#[test]
fn fill_missing_areas_keeps_authored_values() {
    let mut coco = COCO::from_dataset(Dataset {
        images: vec![image(1, Some((100, 100)))],
        annotations: vec![
            Annotation {
                area: Some(7.0),
                ..gt_box(1, 1, 1, [0.0, 0.0, 10.0, 10.0])
            },
            Annotation {
                area: None,
                ..gt_box(2, 1, 1, [0.0, 0.0, 10.0, 10.0])
            },
            // Mask only: the pixel count.
            Annotation {
                id: 3,
                image_id: 1,
                category_id: 1,
                segmentation: Some(Segmentation::Polygon(vec![square(0.0, 0.0, 10.0)])),
                ..Default::default()
            },
        ],
        categories: vec![cat(1)],
        ..Default::default()
    });
    coco.fill_missing_areas();
    let areas: Vec<Option<f64>> = coco.dataset.annotations.iter().map(|a| a.area).collect();
    assert_eq!(areas, vec![Some(7.0), Some(100.0), Some(100.0)]);
}

/// COCO's instance `area` is the mask's pixel count, so a ground truth with a
/// mask takes its area from the mask even when it also has a box. A 40×40 box
/// (1600, medium) around a 30×30 polygon (900, small) is a small object, as
/// the detections' mask-derived areas say.
#[test]
fn fill_missing_areas_prefers_the_mask_over_the_box() {
    let poly = Segmentation::Polygon(vec![square(10.0, 10.0, 30.0)]);
    let gt_ann = Annotation {
        area: None,
        segmentation: Some(poly.clone()),
        ..gt_box(1, 1, 1, [10.0, 10.0, 40.0, 40.0])
    };
    let images = vec![image(1, Some((200, 200)))];
    let mut coco = COCO::from_dataset(Dataset {
        images: images.clone(),
        annotations: vec![gt_ann.clone()],
        categories: vec![cat(1)],
        ..Default::default()
    });
    coco.fill_missing_areas();
    assert_eq!(coco.dataset.annotations[0].area, Some(900.0));

    let dt = Annotation {
        segmentation: Some(poly),
        ..det_box(1, 1, [10.0, 10.0, 30.0, 30.0], 0.9)
    };
    let mut se =
        StreamingEval::new(Params::new(IouType::Segm), EvalMode::Coco, vec![cat(1)]).unwrap();
    se.update(images, vec![gt_ann], vec![dt]).unwrap();
    let mut ev = se.finalize();
    ev.accumulate();
    ev.summarize();
    let s = ev.stats().unwrap();
    // AP, APs, APm at indices 0, 3, 4.
    assert_close(s[0], 1.0);
    assert_close(s[3], 1.0);
    assert_eq!(s[4], -1.0, "no medium ground truth: APm is not computed");
}

// ---------------------------------------------------------------------------
// The K axis belongs to the accumulation, not to `params` as they stand later
// ---------------------------------------------------------------------------

fn lvis_cat(id: u64, frequency: &str) -> Category {
    Category {
        frequency: Some(frequency.to_string()),
        ..cat(id)
    }
}

/// Three LVIS categories, one per frequency bucket, each with a perfect
/// detection — so AP of any bucket is 1.0 and a bucket outside the K axis is
/// `-1.0`. Nothing in between.
fn lvis_pair() -> (COCO, COCO) {
    let boxes = [
        [0.0, 0.0, 20.0, 20.0],
        [50.0, 50.0, 20.0, 20.0],
        [100.0, 100.0, 20.0, 20.0],
    ];
    let gt = COCO::from_dataset(Dataset {
        images: vec![image(1, Some((200, 200)))],
        annotations: (1..=3)
            .map(|c| gt_box(c, 1, c, boxes[c as usize - 1]))
            .collect(),
        categories: vec![lvis_cat(1, "r"), lvis_cat(2, "c"), lvis_cat(3, "f")],
        ..Default::default()
    });
    let dt = gt
        .load_res_anns(
            (1..=3)
                .map(|c| det_box(1, c, boxes[c as usize - 1], 0.9))
                .collect(),
        )
        .unwrap();
    (gt, dt)
}

fn lvis_results(ev: &COCOeval) -> std::collections::BTreeMap<String, f64> {
    ev.get_results(None, false)
}

/// `params.cat_ids` narrowed between `evaluate()` and `accumulate()` makes a
/// one-slot K axis. The frequency groups used to be frozen at `evaluate()`
/// with positions 0..3, so APf read position 2 of a one-slot array: another
/// cell's precision, or out of bounds.
#[test]
fn lvis_frequency_groups_follow_the_accumulated_k_axis() {
    let (gt, dt) = lvis_pair();
    let mut ev = COCOeval::new_lvis(gt, dt, IouType::Bbox);
    ev.evaluate().expect("evaluable inputs");
    ev.params.cat_ids = vec![3];
    ev.accumulate();
    ev.summarize();
    let r = lvis_results(&ev);
    assert_eq!(r["APr"], -1.0, "category 1 is not on the K axis");
    assert_eq!(r["APc"], -1.0, "category 2 is not on the K axis");
    assert_close(r["APf"], 1.0);

    // Reordered rather than narrowed: every bucket must still find its own
    // category, not the one that used to sit at its position.
    let (gt, dt) = lvis_pair();
    let mut ev = COCOeval::new_lvis(gt, dt, IouType::Bbox);
    ev.evaluate().expect("evaluable inputs");
    ev.params.cat_ids = vec![3, 1];
    ev.accumulate();
    ev.summarize();
    let r = lvis_results(&ev);
    assert_close(r["APr"], 1.0);
    assert_eq!(r["APc"], -1.0);
    assert_close(r["APf"], 1.0);
}

/// A pooled (`use_cats = false`) LVIS run has one K slot and no category on
/// it. The batch path reported the frequency groups as `-1.0`; the streaming
/// path built them from `params.cat_ids` anyway and indexed past the slot.
#[test]
fn streaming_lvis_without_categories_reports_no_frequency_groups() {
    let (gt, _) = lvis_pair();
    let boxes = [
        [0.0, 0.0, 20.0, 20.0],
        [50.0, 50.0, 20.0, 20.0],
        [100.0, 100.0, 20.0, 20.0],
    ];
    let dts: Vec<Annotation> = (1..=3)
        .map(|c| det_box(1, c, boxes[c as usize - 1], 0.9))
        .collect();

    let mut params = EvalMode::Lvis.default_params(IouType::Bbox);
    params.use_cats = false;
    let mut se = StreamingEval::new(params, EvalMode::Lvis, gt.dataset.categories.clone()).unwrap();
    se.update(
        gt.dataset.images.clone(),
        gt.dataset.annotations.clone(),
        dts.clone(),
    )
    .unwrap();
    let mut streamed = se.finalize();
    streamed.accumulate();
    streamed.summarize();

    let dt = gt.load_res_anns(dts).unwrap();
    let mut batch = COCOeval::new_lvis(gt, dt, IouType::Bbox);
    batch.params.use_cats = false;
    batch.run().expect("evaluable inputs");

    let s = lvis_results(&streamed);
    let b = lvis_results(&batch);
    assert_eq!(s, b);
    assert_eq!(s["APr"], -1.0);
    assert_eq!(s["APc"], -1.0);
    assert_eq!(s["APf"], -1.0);
    assert_close(s["AP"], 1.0);
}

/// With `use_cats = false` the single K slot is the pool of every category.
/// `per_class` used to zip that one value against `params.cat_ids` and report
/// it under the first category's name.
#[test]
fn pooled_run_has_no_per_class_entries() {
    let (gt, dt) = lvis_pair();
    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    ev.params.use_cats = false;
    ev.run().expect("evaluable inputs");

    let r = ev.get_results(None, true);
    let per_class: Vec<&String> = r.keys().filter(|k| k.starts_with("AP/")).collect();
    assert!(
        per_class.is_empty(),
        "pooled AP under a class name: {per_class:?}"
    );
    assert!(
        ev.results(true)
            .unwrap()
            .per_class
            .is_none_or(|m| m.is_empty())
    );

    // And with categories on, every one is named.
    let (gt, dt) = lvis_pair();
    let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
    ev.run().expect("evaluable inputs");
    let r = ev.get_results(None, true);
    assert_eq!(r.keys().filter(|k| k.starts_with("AP/")).count(), 3);
}

/// pycocotools rasterizes only the annotations `_prepare` loads — those in
/// `params.catIds` when `useCats` is on — so a polygon in a category the run
/// leaves out never needs a canvas. The check used to cover every category
/// and failed the whole segm run on it. With `use_cats` off every category is
/// loaded, so the same polygon is an error again.
#[test]
fn check_inputs_ignores_dimensionless_images_outside_cat_ids() {
    let poly = Segmentation::Polygon(vec![square(10.0, 10.0, 50.0)]);
    let gt = COCO::from_dataset(Dataset {
        images: vec![image(1, Some((100, 100))), image(2, None)],
        annotations: vec![
            Annotation {
                segmentation: Some(poly.clone()),
                ..gt_box(1, 1, 1, [10.0, 10.0, 50.0, 50.0])
            },
            // Category 2, on the image without `height`/`width`.
            Annotation {
                segmentation: Some(poly.clone()),
                ..gt_box(2, 2, 2, [10.0, 10.0, 50.0, 50.0])
            },
        ],
        categories: vec![cat(1), cat(2)],
        ..Default::default()
    });
    let dt = gt
        .load_res_anns(vec![Annotation {
            segmentation: Some(poly),
            ..det_box(1, 1, [10.0, 10.0, 50.0, 50.0], 0.9)
        }])
        .unwrap();
    let mut ev = COCOeval::new(gt, dt, IouType::Segm);
    assert!(
        ev.check_inputs().is_err(),
        "category 2 is in the default scope"
    );

    ev.params.cat_ids = vec![1];
    ev.check_inputs().unwrap();
    assert_close(stats(&mut ev)[0], 1.0);

    ev.params.use_cats = false;
    assert!(
        ev.check_inputs().is_err(),
        "use_cats = false pools every category, category 2 included"
    );
}
