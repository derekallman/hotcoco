//! Reference-parity regressions from the 2026-10 whole-project review.
//!
//! Each expected value below was produced by the reference implementation named
//! in the test's doc comment, not by hotcoco.

#![allow(clippy::unwrap_used)]

use hotcoco::mask;

// ── Polygons that extend far outside the image ──────────────────────────────

/// `fr_poly` used to clamp every upsampled vertex to one image-extent beyond
/// the image, which bends the edges of a polygon reaching further out than
/// that: `[0,0,300,100,0,100]` on 100×100 came out at area 7530 against the
/// reference's 8350. Expected strings are `pycocotools.mask.frPyObjects`
/// output (pycocotools 2.0.x). Every edge here has a slope whose reduced
/// denominator is odd, so no interpolated coordinate lands on `.5` and the
/// arm64-vs-x86 FMA difference in `interp` cannot change a pixel.
#[test]
fn fr_poly_far_out_of_image_matches_pycocotools() {
    let cases: [(&[f64], u32, u32, u64, &str); 5] = [
        (
            &[0.0, 0.0, 300.0, 100.0, 0.0, 100.0],
            100,
            100,
            8350,
            "0X61kL00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00001O00",
        ),
        (
            &[-500.0, -20.0, 50.0, 40.0, 600.0, 90.0, 20.0, 130.0],
            100,
            100,
            6026,
            "S1Q2S10000000000000001O00000000000000001O00000000000000001O00000000000000001O0000000000000000001O0000000000000000001O000000000000000000001O000000000000000000001O000000000000000000001O00000000000000000000",
        ),
        (
            &[0.0, -200.0, 60.0, 90.0, -100.0, 90.0],
            80,
            60,
            3973,
            "0`V31okL4L5K5K5K5K5K4L5K5K5K5K5K4L5K5K5Kb2",
        ),
        (
            &[20.0, -400.0, 80.0, 70.0, -30.0, 70.0],
            100,
            100,
            5289,
            "0V2n000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003M8H8H8H8H7I8H8H8H^l1",
        ),
        (
            &[5.0, 5.0, 95.0, -295.0, 95.0, 95.0],
            100,
            100,
            4452,
            "g?2o24O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1O1c?",
        ),
    ];
    for (poly, h, w, area, counts) in cases {
        let rle = mask::fr_poly(poly, h, w).unwrap();
        assert_eq!(mask::area(&rle), area, "area of {poly:?} on {h}x{w}");
        assert_eq!(
            mask::rle_to_string(&rle),
            counts,
            "counts of {poly:?} on {h}x{w}"
        );
    }
}

/// The memory bound survives: a polygon whose boundary walk would run to
/// billions of points (1000 vertices alternating ±1e9) falls back to the
/// image-extent clamp instead of reserving gigabytes, and stays in the image.
#[test]
fn fr_poly_absurd_many_vertex_polygon_stays_bounded() {
    let poly: Vec<f64> = (0..2000)
        .map(|i| if (i / 2 + i % 2) % 2 == 0 { 1e9 } else { -1e9 })
        .collect();
    let rle = mask::fr_poly(&poly, 100, 100).unwrap();
    assert!(mask::area(&rle) <= 100 * 100);
    assert_eq!(rle.counts.iter().map(|&c| c as u64).sum::<u64>(), 100 * 100);
}

// ── LVIS caps detections per image, not per (image, category) ───────────────

use hotcoco::types::{Annotation, Category, Dataset, Image};
use hotcoco::{COCO, COCOeval, EvalMode, IouType, StreamingEval};

fn lvis_image(id: u64, neg: Vec<u64>) -> Image {
    Image {
        id,
        file_name: format!("img{id}.jpg"),
        height: 200,
        width: 200,
        neg_category_ids: neg,
        ..Default::default()
    }
}

fn lvis_ann(image_id: u64, category_id: u64, bbox: [f64; 4], score: Option<f64>) -> Annotation {
    Annotation {
        image_id,
        category_id,
        bbox: Some(bbox),
        area: Some(bbox[2] * bbox[3]),
        score,
        ..Default::default()
    }
}

/// `tests/test_parity_lvis.py::_build_over_300_per_image`, in Rust: 302
/// detections on image 1 across four categories. lvis-api's `LVISResults`
/// keeps the top 300 by score over the whole image (stable, so ties keep file
/// order), which drops cat 1's only detection — tied with the 300th score but
/// later in the file. lvis-api 0.5.3 reports AP = 2/3 and APr = 0 here; the
/// old per-cell cap kept the detection and reported AP = APr = 1.
fn over_300_fixture() -> (Vec<Image>, Vec<Category>, Vec<Annotation>, Vec<Annotation>) {
    let images = vec![lvis_image(1, vec![2]), lvis_image(2, vec![])];
    let categories = ["r", "c", "f", "f"]
        .iter()
        .enumerate()
        .map(|(i, f)| Category {
            id: i as u64 + 1,
            name: format!("cat{}", i + 1),
            frequency: Some((*f).into()),
            ..Default::default()
        })
        .collect();
    let mut gt = vec![
        lvis_ann(1, 1, [10.0, 10.0, 40.0, 40.0], None),
        lvis_ann(1, 3, [100.0, 100.0, 50.0, 50.0], None),
        lvis_ann(2, 4, [20.0, 20.0, 60.0, 60.0], None),
    ];
    for (i, a) in gt.iter_mut().enumerate() {
        a.id = i as u64 + 1;
    }
    let mut dt = vec![lvis_ann(1, 3, [100.0, 100.0, 50.0, 50.0], Some(0.995))];
    for i in 0..200u32 {
        let b = [f64::from(i % 150), f64::from((i * 7) % 150), 20.0, 20.0];
        dt.push(lvis_ann(1, 4, b, Some(f64::from(9900 - i) / 1e4)));
    }
    for i in 0..100u32 {
        let b = [f64::from((i * 3) % 150), f64::from(i % 150), 15.0, 15.0];
        dt.push(lvis_ann(1, 2, b, Some(f64::from(9600 - i) / 1e4)));
    }
    dt.push(lvis_ann(
        1,
        1,
        [10.0, 10.0, 40.0, 40.0],
        Some(f64::from(9600 - 98) / 1e4),
    ));
    dt.push(lvis_ann(2, 4, [20.0, 20.0, 60.0, 60.0], Some(0.9)));
    (images, categories, gt, dt)
}

#[test]
fn lvis_caps_detections_per_image_like_lvis_api() {
    let (images, categories, gt_anns, dt_anns) = over_300_fixture();
    let gt = COCO::from_dataset(Dataset {
        images: images.clone(),
        annotations: gt_anns.clone(),
        categories: categories.clone(),
        ..Default::default()
    });
    let dt = gt.load_res_anns(dt_anns.clone()).unwrap();
    let mut ev = COCOeval::new_lvis(gt, dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    let stats = ev.stats().unwrap().to_vec();
    // Order: AP, AP50, AP75, APs, APm, APl, APr, APc, APf, AR@300, ...
    assert!((stats[0] - 2.0 / 3.0).abs() < 1e-12, "AP {}", stats[0]);
    assert_eq!(stats[6], 0.0, "APr");
    assert!((stats[9] - 2.0 / 3.0).abs() < 1e-12, "AR@300 {}", stats[9]);

    // The streaming path runs the same `evaluate()` per batch, so an image
    // over the cap streams to the same numbers.
    let mut se = StreamingEval::new(
        EvalMode::Lvis.default_params(IouType::Bbox),
        EvalMode::Lvis,
        categories,
    )
    .unwrap();
    se.update(images, gt_anns, dt_anns).unwrap();
    let mut streamed = se.finalize();
    streamed.accumulate();
    streamed.summarize();
    assert_eq!(streamed.stats().unwrap(), stats.as_slice());
}

/// A results set already capped by `cap_detections_per_image` (Python
/// `LVISResults`) is evaluated as is, as lvis-api's `LVISEval` takes an
/// `LVISResults` unchanged: `max_dets=-1` (`None`) and `1000` both keep cat 1's
/// tied detection, so AP = APr = 1 where the default 300 cap gives 2/3 and 0.
/// A cap of 300 matches the uncapped-input result.
#[test]
fn lvis_keeps_a_results_set_capped_by_the_caller() {
    let (images, categories, gt_anns, dt_anns) = over_300_fixture();
    let gt = COCO::from_dataset(Dataset {
        images,
        annotations: gt_anns,
        categories,
        ..Default::default()
    });
    let run = |max_det: Option<usize>| {
        let dt = gt
            .load_res_anns(dt_anns.clone())
            .unwrap()
            .cap_detections_per_image(max_det);
        let mut ev = COCOeval::new_lvis(gt.clone(), dt, IouType::Bbox);
        ev.evaluate();
        ev.accumulate();
        ev.summarize();
        ev.stats().unwrap().to_vec()
    };
    for max_det in [None, Some(1000)] {
        let stats = run(max_det);
        assert!(
            (stats[0] - 1.0).abs() < 1e-12,
            "{max_det:?}: AP {}",
            stats[0]
        );
        assert!(
            (stats[6] - 1.0).abs() < 1e-12,
            "{max_det:?}: APr {}",
            stats[6]
        );
    }
    let stats = run(Some(300));
    assert!((stats[0] - 2.0 / 3.0).abs() < 1e-12, "AP {}", stats[0]);
    assert_eq!(stats[6], 0.0, "APr");
}
