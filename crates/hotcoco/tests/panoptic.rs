//! End-to-end panoptic evaluation: hand-worked numbers through
//! `PanopticEval`, the PNG and mask paths agreeing on the same pixels, and
//! the input errors the reference raises on.
//!
//! Parity against panopticapi itself is `tests/test_parity_panoptic.py`
//! (synthetic, in CI) and `scripts/parity_panoptic.py` (val2017, local).

use std::path::Path;

use hotcoco::panoptic::{PanopticAnnotation, PanopticDataset, PanopticEval, SegmentInfo};
use hotcoco::types::{Annotation, Category, Dataset, Image, Segmentation};
use hotcoco::{Provenance, Rle};

/// A row-major `h × w` id map, as a test writes it.
struct Map {
    h: u32,
    w: u32,
    ids: Vec<u32>,
}

impl Map {
    fn filled(h: u32, w: u32, id: u32) -> Self {
        Map {
            h,
            w,
            ids: vec![id; (h * w) as usize],
        }
    }

    /// Paint the rectangle `[y0, y1) × [x0, x1)` with `id`.
    fn rect(mut self, y0: u32, y1: u32, x0: u32, x1: u32, id: u32) -> Self {
        for y in y0..y1 {
            for x in x0..x1 {
                self.ids[(y * self.w + x) as usize] = id;
            }
        }
        self
    }

    fn area(&self, id: u32) -> u64 {
        self.ids.iter().filter(|&&v| v == id).count() as u64
    }

    /// Write as a COCO panoptic PNG: RGB with `id2rgb`.
    fn write_png(&self, path: &Path) {
        let file = std::fs::File::create(path).expect("create png");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), self.w, self.h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().expect("header");
        let data: Vec<u8> = self
            .ids
            .iter()
            .flat_map(|&id| {
                [
                    (id & 0xff) as u8,
                    ((id >> 8) & 0xff) as u8,
                    ((id >> 16) & 0xff) as u8,
                ]
            })
            .collect();
        writer.write_image_data(&data).expect("write");
    }

    /// The mask of `id` as an uncompressed (column-major) RLE.
    fn rle(&self, id: u32) -> Segmentation {
        let mut mask = vec![0u8; (self.h * self.w) as usize];
        for y in 0..self.h {
            for x in 0..self.w {
                if self.ids[(y * self.w + x) as usize] == id {
                    mask[(x * self.h + y) as usize] = 1;
                }
            }
        }
        let rle: Rle = hotcoco::mask::encode(&mask, self.h, self.w).expect("encode");
        Segmentation::UncompressedRle {
            size: [self.h, self.w],
            counts: rle.counts,
        }
    }
}

fn category(id: u64, isthing: Option<bool>) -> Category {
    Category {
        id,
        name: format!("c{id}"),
        isthing,
        ..Default::default()
    }
}

fn segment(id: u64, category_id: u64, area: Option<u64>, iscrowd: bool) -> SegmentInfo {
    SegmentInfo {
        id,
        category_id,
        area,
        bbox: None,
        iscrowd,
        segmentation: None,
    }
}

/// A PNG-backed dataset of one image under `dir`.
fn png_dataset(
    dir: &Path,
    name: &str,
    map: &Map,
    segments: Vec<SegmentInfo>,
    categories: Vec<Category>,
) -> PanopticDataset {
    let folder = dir.join(name);
    std::fs::create_dir_all(&folder).expect("folder");
    map.write_png(&folder.join("1.png"));
    PanopticDataset {
        images: vec![Image {
            id: 1,
            height: map.h,
            width: map.w,
            ..Default::default()
        }],
        annotations: vec![PanopticAnnotation {
            image_id: 1,
            file_name: Some("1.png".to_string()),
            segments_info: segments,
        }],
        categories,
        ..Default::default()
    }
    .with_folder(folder)
}

/// The same image as detection-style annotations carrying RLEs, one per id.
fn mask_dataset(map: &Map, segments: &[SegmentInfo], categories: Vec<Category>) -> Dataset {
    Dataset {
        images: vec![Image {
            id: 1,
            height: map.h,
            width: map.w,
            ..Default::default()
        }],
        annotations: segments
            .iter()
            .map(|s| Annotation {
                id: s.id,
                image_id: 1,
                category_id: s.category_id,
                iscrowd: s.iscrowd,
                segmentation: Some(map.rle(s.id as u32)),
                ..Default::default()
            })
            .collect(),
        categories,
        ..Default::default()
    }
}

/// Ground truth: segment 1 (cat 10, thing) is the 4×4 block at the top left,
/// segment 2 (cat 20, stuff) is the right half, and the rest is void.
/// Prediction: segment 7 covers segment 1 shifted one column right (IoU 3/5),
/// segment 8 covers the right half exactly, segment 9 (cat 10) sits on void.
fn worked_example() -> (Map, Map, Vec<SegmentInfo>, Vec<SegmentInfo>, Vec<Category>) {
    let gt = Map::filled(8, 8, 0).rect(0, 4, 0, 4, 1).rect(0, 8, 4, 8, 2);
    let pred = Map::filled(8, 8, 0)
        .rect(0, 4, 1, 5, 7)
        .rect(0, 8, 4, 8, 8)
        .rect(5, 7, 1, 3, 9)
        // pred 7 was painted over by 8 in column 4: redo 8 after 7 so 7 keeps [1,4) only.
        .rect(0, 8, 4, 8, 8);
    let gt_segs = vec![
        segment(1, 10, Some(gt.area(1)), false),
        segment(2, 20, Some(gt.area(2)), false),
    ];
    let pred_segs = vec![
        segment(7, 10, None, false),
        segment(8, 20, None, false),
        segment(9, 10, None, false),
    ];
    let cats = vec![category(10, Some(true)), category(20, Some(false))];
    (gt, pred, gt_segs, pred_segs, cats)
}

#[test]
fn worked_example_scores_by_hand() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (gt, pred, gt_segs, pred_segs, cats) = worked_example();
    // pred 7 covers columns 1..4 of rows 0..4 (12 px) over gt 1 (16 px):
    // intersection 12, union 12 + 16 - 12 - 0 = 16 -> 0.75 > 0.5, matched.
    assert_eq!(pred.area(7), 12);
    let gt_ds = png_dataset(dir.path(), "gt", &gt, gt_segs, cats.clone());
    let pred_ds = png_dataset(dir.path(), "pred", &pred, pred_segs, Vec::new());
    let mut ev = PanopticEval::new(gt_ds, pred_ds);
    ev.run().expect("runs");
    let r = ev.result().expect("result");

    // cat 10: 1 TP (iou 0.75), 1 FP (pred 9 on void: 4/4 void -> ignored!).
    // pred 9 lies entirely on void, so it is ignored, not a false positive.
    let c10 = &r.per_category[&10];
    assert_eq!((c10.tp, c10.fp, c10.fn_), (1, 0, 0));
    assert!((c10.iou - 0.75).abs() < 1e-12);
    let c20 = &r.per_category[&20];
    assert_eq!((c20.tp, c20.fp, c20.fn_), (1, 0, 0));
    assert_eq!(c20.iou, 1.0);

    // PQ = mean(0.75, 1.0) = 0.875; SQ the same; RQ = 1.0.
    assert!((r.all.scores.pq - 0.875).abs() < 1e-12);
    assert!((r.all.scores.sq - 0.875).abs() < 1e-12);
    assert_eq!(r.all.scores.rq, 1.0);
    assert_eq!(r.all.n, 2);
    assert!((r.things.scores.pq - 0.75).abs() < 1e-12);
    assert_eq!(r.things.n, 1);
    assert_eq!(r.stuff.scores.pq, 1.0);
    assert_eq!(r.stuff.n, 1);
    assert_eq!(r.n_images, 1);

    // The same pixels as semantic segmentation. Ground-truth void (rows 4..8,
    // cols 0..4, 16 px — pred 9's 4 px among them) is ignored. Category 10
    // (16 px): 12 predicted 10, 4 predicted void (column 0). Category 20
    // (32 px): all 32 predicted 20. Rows and columns are category rank; the
    // last column is the predicted void.
    assert_eq!(r.pixels.num_classes(), 2);
    assert_eq!(r.pixels.unlabeled_pred(), Some(2));
    assert_eq!(r.pixels.get(0, 0), 12);
    assert_eq!(r.pixels.get(0, 2), 4);
    assert_eq!(r.pixels.get(1, 1), 32);
    assert_eq!(r.pixels.total(), 48);
    let sem = r.semantic();
    assert_eq!(sem.per_class.len(), 2, "the void column is not a class");
    assert_eq!(sem.per_class[0].iou, 0.75); // 12 / (16 + 12 - 12)
    assert_eq!(sem.per_class[0].acc, 0.75); // 12 / 16
    assert_eq!(sem.per_class[1].iou, 1.0);
    assert_eq!(sem.miou, 0.875);
    assert!((sem.aacc - 44.0 / 48.0).abs() < 1e-12);
    assert!((sem.fwiou - 44.0 / 48.0).abs() < 1e-12);

    assert_eq!(ev.provenance(), Provenance::ParityVerified);
    let report = ev.report().expect("report");
    assert_eq!(report.task, "panoptic");
    assert!((report.metric("PQ").expect("PQ") - 0.875).abs() < 1e-12);
    assert!((report.metric("PQ_th").expect("PQ_th") - 0.75).abs() < 1e-12);
    assert_eq!(report.metric("mIoU"), Some(0.875));
    assert!((report.metric("aAcc").expect("aAcc") - 44.0 / 48.0).abs() < 1e-12);
    assert_eq!(report.per_class["c10"]["PQ"], 0.75);
    assert_eq!(report.per_class["c10"]["IoU"], 0.75);
    assert_eq!(report.per_class["c20"]["Acc"], 1.0);
    assert_eq!(report.per_group["things"]["n"], 1.0);

    let lines = ev.summarize_lines();
    assert_eq!(lines[0], "          |    PQ     SQ     RQ     N");
    assert_eq!(lines[2], "All       |  87.5   87.5  100.0     2");

    // `results()` is panopticapi's shape: flat splits, per-class scores with
    // the counts beside them, `fn` spelled as the reference spells it.
    let results = ev.results().expect("results");
    let json: serde_json::Value =
        serde_json::from_str(&results.to_json().expect("json")).expect("parses");
    assert_eq!(json["All"]["pq"], 0.875);
    assert_eq!(json["Things"]["n"], 1);
    assert_eq!(json["per_class"]["10"]["tp"], 1);
    assert_eq!(json["per_class"]["10"]["fn"], 0);
    assert_eq!(json["per_class"]["10"]["pq"], 0.75);
    assert_eq!(json["provenance"], "parity_verified");
    assert!(json["per_class"].get("20").is_some());
}

#[test]
fn match_records_name_every_segment_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (gt, pred, gt_segs, pred_segs, cats) = worked_example();
    let mut ev = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs, cats),
        png_dataset(dir.path(), "pred", &pred, pred_segs, Vec::new()),
    );
    assert!(ev.eval_imgs().is_empty(), "no records before run()");
    ev.run().expect("runs");

    let [rec] = ev.eval_imgs() else {
        panic!("one record per image: {:?}", ev.eval_imgs())
    };
    assert_eq!(rec.image_id, 1);
    // pred 7 covers 12 of segment 1's 16 pixels and nothing else: 0.75.
    assert_eq!(rec.matches.matched, vec![(1, 7, 0.75), (2, 8, 1.0)]);
    assert_eq!(rec.matches.missed, Vec::<u64>::new());
    assert_eq!(rec.matches.false_positives, Vec::<u64>::new());
    // pred 9 lies entirely on void.
    assert_eq!(rec.matches.ignored, vec![9]);

    // A crowd segment is in no list; predictions it absorbs are ignored.
    let gt = Map::filled(4, 8, 0).rect(0, 4, 0, 4, 1).rect(0, 2, 6, 8, 2);
    let pred = Map::filled(4, 8, 0)
        .rect(0, 3, 0, 4, 7)
        .rect(0, 4, 6, 8, 8)
        .rect(3, 4, 4, 6, 9);
    let mut ev = PanopticEval::new(
        png_dataset(
            dir.path(),
            "gt2",
            &gt,
            vec![
                segment(1, 10, Some(gt.area(1)), true),
                segment(2, 10, Some(gt.area(2)), false),
                segment(3, 10, Some(5), false),
            ],
            vec![category(10, Some(true))],
        ),
        png_dataset(
            dir.path(),
            "pred2",
            &pred,
            vec![
                segment(7, 10, None, false),
                segment(8, 10, None, false),
                segment(9, 10, None, false),
            ],
            Vec::new(),
        ),
    );
    ev.run().expect("runs");
    let rec = &ev.eval_imgs()[0];
    assert_eq!(rec.matches.matched, vec![(2, 8, 1.0)]);
    // Segment 3 is in the JSON and not the PNG: a miss, as panopticapi has it.
    assert_eq!(rec.matches.missed, vec![3]);
    // pred 9: 2 of its 2 pixels on void beside the crowd -> ignored, not FP.
    assert_eq!(rec.matches.ignored, vec![7, 9]);
    assert_eq!(rec.matches.false_positives, Vec::<u64>::new());
    let c = &ev.result().expect("result").per_category[&10];
    assert_eq!(
        (c.tp, c.fp, c.fn_),
        (
            rec.matches.matched.len() as u64,
            rec.matches.false_positives.len() as u64,
            rec.matches.missed.len() as u64
        )
    );
}

/// The PNG path labels a segment by its id and the mask path by its position,
/// and `matched` follows the label: the order each path sums the IoUs in.
/// Ids the mask path is given twice stay two segments, under one id.
#[test]
fn match_records_follow_each_paths_labels() {
    let dir = tempfile::tempdir().expect("tempdir");
    let map = Map::filled(4, 4, 0)
        .rect(0, 4, 0, 2, 30)
        .rect(0, 4, 2, 4, 10);
    let segs = vec![segment(30, 1, None, false), segment(10, 1, None, false)];
    let cats = vec![category(1, Some(true))];

    let mut png = PanopticEval::new(
        png_dataset(dir.path(), "gt", &map, segs.clone(), cats.clone()),
        png_dataset(dir.path(), "pred", &map, segs.clone(), Vec::new()),
    );
    png.run().expect("png path runs");
    assert_eq!(
        png.eval_imgs()[0].matches.matched,
        vec![(10, 10, 1.0), (30, 30, 1.0)]
    );

    let mut masks = PanopticEval::new(
        PanopticDataset::from_dataset(&mask_dataset(&map, &segs, cats.clone())),
        PanopticDataset::from_dataset(&mask_dataset(&map, &segs, Vec::new())),
    );
    masks.run().expect("mask path runs");
    assert_eq!(
        masks.eval_imgs()[0].matches.matched,
        vec![(30, 30, 1.0), (10, 10, 1.0)]
    );

    // A converter's `id: 0` on every annotation: two segments, both id 0.
    let mut zeros = mask_dataset(&map, &segs, cats);
    for ann in &mut zeros.annotations {
        ann.id = 0;
    }
    let mut ev = PanopticEval::new(
        PanopticDataset::from_dataset(&zeros),
        PanopticDataset::from_dataset(&zeros),
    );
    ev.run().expect("repeated ids run on the mask path");
    assert_eq!(
        ev.eval_imgs()[0].matches.matched,
        vec![(0, 0, 1.0), (0, 0, 1.0)]
    );
    assert_eq!(ev.result().expect("result").per_category[&1].tp, 2);
}

#[test]
fn png_and_mask_paths_agree() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (gt, pred, gt_segs, pred_segs, cats) = worked_example();
    let mut from_png = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs.clone(), cats.clone()),
        png_dataset(dir.path(), "pred", &pred, pred_segs.clone(), Vec::new()),
    );
    from_png.run().expect("png path runs");

    let mut from_masks = PanopticEval::new(
        PanopticDataset::from_dataset(&mask_dataset(&gt, &gt_segs, cats.clone())),
        PanopticDataset::from_dataset(&mask_dataset(&pred, &pred_segs, Vec::new())),
    );
    from_masks.run().expect("mask path runs");

    assert_eq!(from_png.result(), from_masks.result());

    // Mixed: ground truth from PNG files, prediction from masks.
    let mut mixed = PanopticEval::new(
        png_dataset(dir.path(), "gt2", &gt, gt_segs, cats.clone()),
        PanopticDataset::from_dataset(&mask_dataset(&pred, &pred_segs, Vec::new())),
    );
    mixed.run().expect("mixed runs");
    assert_eq!(mixed.result(), from_png.result());
}

#[test]
fn a_dataset_without_isthing_is_an_extension() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (gt, pred, gt_segs, pred_segs, _) = worked_example();
    let cats = vec![category(10, Some(true)), category(20, None)];
    let mut ev = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs, cats),
        png_dataset(dir.path(), "pred", &pred, pred_segs, Vec::new()),
    );
    assert_eq!(ev.provenance(), Provenance::Extension);
    assert_eq!(ev.reference_deviations().len(), 1);
    ev.run().expect("runs");
    let r = ev.result().expect("result");
    // All still averages both; stuff has nothing that says it is stuff.
    assert_eq!(r.all.n, 2);
    assert_eq!(r.things.n, 1);
    assert_eq!(r.stuff.n, 0);
    assert_eq!(r.stuff.scores.pq, -1.0);
    let report = ev.report().expect("report");
    assert_eq!(report.provenance, Provenance::Extension);
    assert!(
        !report.per_group.contains_key("stuff"),
        "no sentinel groups in a report"
    );
    assert_eq!(
        report.metric("PQ_st"),
        Some(-1.0),
        "headline keeps the sentinel"
    );
    assert!(ev.summarize_lines()[4].starts_with("Stuff     |     -      -      -     0"));
}

#[test]
fn reference_errors_are_errors_here() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (gt, pred, gt_segs, pred_segs, cats) = worked_example();

    // A ground-truth image with no prediction.
    let mut ev = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs.clone(), cats.clone()),
        PanopticDataset::default().with_folder(dir.path().join("pred")),
    );
    let err = ev.run().expect_err("no prediction");
    assert!(
        err.to_string()
            .contains("no prediction for the image with id 1"),
        "{err}"
    );
    assert!(ev.result().is_none());

    // A predicted segment in the PNG but not the JSON (drop segment 9).
    let mut short = pred_segs.clone();
    short.retain(|s| s.id != 9);
    let mut ev = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs.clone(), cats.clone()),
        png_dataset(dir.path(), "pred", &pred, short, Vec::new()),
    );
    let err = ev.run().expect_err("png id missing from json");
    assert!(
        err.to_string()
            .contains("segment 9 is in the label map but not in segments_info"),
        "{err}"
    );

    // A predicted segment in the JSON but not the PNG.
    let mut extra = pred_segs.clone();
    extra.push(segment(11, 10, None, false));
    let mut ev = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs.clone(), cats.clone()),
        png_dataset(dir.path(), "pred", &pred, extra, Vec::new()),
    );
    let err = ev.run().expect_err("json id missing from png");
    assert!(
        err.to_string()
            .contains("segment 11 is in segments_info but not in the label map"),
        "{err}"
    );

    // An unknown prediction category.
    let mut odd = pred_segs.clone();
    odd[0].category_id = 99;
    let mut ev = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs.clone(), cats.clone()),
        png_dataset(dir.path(), "pred", &pred, odd, Vec::new()),
    );
    let err = ev.run().expect_err("unknown category");
    assert!(err.to_string().contains("unknown category_id 99"), "{err}");

    // Maps of different sizes.
    let small = Map::filled(4, 4, 0).rect(0, 4, 0, 4, 7);
    let mut ev = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs.clone(), cats.clone()),
        png_dataset(
            dir.path(),
            "pred",
            &small,
            vec![segment(7, 10, None, false)],
            Vec::new(),
        ),
    );
    let err = ev.run().expect_err("size mismatch");
    assert!(
        err.to_string().contains("8×8 but the prediction is 4×4"),
        "{err}"
    );

    // A mask-path image with no size.
    let mut sizeless = mask_dataset(&pred, &pred_segs, Vec::new());
    sizeless.images[0].height = 0;
    let mut ev = PanopticEval::new(
        PanopticDataset::from_dataset(&mask_dataset(&gt, &gt_segs, cats.clone())),
        PanopticDataset::from_dataset(&sizeless),
    );
    // The ground truth's images win, so this still runs; drop them too.
    ev.run().expect("ground truth supplies the size");
    let mut gt_sizeless = mask_dataset(&gt, &gt_segs, cats);
    gt_sizeless.images[0].height = 0;
    let mut ev = PanopticEval::new(
        PanopticDataset::from_dataset(&gt_sizeless),
        PanopticDataset::from_dataset(&sizeless),
    );
    let err = ev.run().expect_err("no size anywhere");
    assert!(err.to_string().contains("has no height and width"), "{err}");
}

#[test]
fn crowd_void_and_threshold_rules_through_the_driver() {
    let dir = tempfile::tempdir().expect("tempdir");
    // gt: crowd 1 (cat 10) on the left half, segment 2 (cat 10) a 2×2 block on
    // the right, void elsewhere.
    let gt = Map::filled(4, 8, 0).rect(0, 4, 0, 4, 1).rect(0, 2, 6, 8, 2);
    // pred: 7 (cat 10) mostly on the crowd -> ignored; 8 (cat 10) covers
    // segment 2 plus 4 void pixels -> iou = 4 / (8 + 4 - 4 - 4) = 1.0;
    // 9 (cat 10) half on the crowd, half on void -> ignored (crowd + void = all).
    let pred = Map::filled(4, 8, 0)
        .rect(0, 3, 0, 4, 7)
        .rect(0, 4, 6, 8, 8)
        .rect(3, 4, 2, 6, 9);
    let gt_segs = vec![
        segment(1, 10, Some(gt.area(1)), true),
        segment(2, 10, Some(gt.area(2)), false),
    ];
    let pred_segs = vec![
        segment(7, 10, None, false),
        segment(8, 10, None, false),
        segment(9, 10, None, false),
    ];
    let cats = vec![category(10, Some(true))];
    let mut ev = PanopticEval::new(
        png_dataset(dir.path(), "gt", &gt, gt_segs, cats),
        png_dataset(dir.path(), "pred", &pred, pred_segs, Vec::new()),
    );
    ev.run().expect("runs");
    let c = &ev.result().expect("result").per_category[&10];
    assert_eq!((c.tp, c.fp, c.fn_), (1, 0, 0), "{c:?}");
    assert_eq!(c.iou, 1.0);
}

#[test]
fn loads_a_panoptic_json_with_the_default_folder() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (gt, _, gt_segs, _, cats) = worked_example();
    let ds = png_dataset(dir.path(), "panoptic_x", &gt, gt_segs, cats);
    let json_path = dir.path().join("panoptic_x.json");
    std::fs::write(&json_path, ds.to_json().expect("json")).expect("write");
    let loaded = PanopticDataset::from_file(&json_path).expect("load");
    assert_eq!(
        loaded.folder.as_deref(),
        Some(dir.path().join("panoptic_x").as_path())
    );
    assert_eq!(loaded.annotations.len(), 1);
    assert_eq!(loaded.categories[0].isthing, Some(true));

    // Scoring it against itself is perfect.
    let mut ev = PanopticEval::new(loaded.clone(), loaded);
    ev.run().expect("runs");
    assert_eq!(ev.result().expect("result").all.scores.pq, 1.0);
}
