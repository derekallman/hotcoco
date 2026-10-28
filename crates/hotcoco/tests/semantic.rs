//! Semantic segmentation parity: hotcoco vs mmsegmentation, torchmetrics, and
//! detectron2 on frozen label maps.
//!
//! `tests/fixtures/semantic_expected.json` (repo root) holds label-map pairs
//! and what each reference computed for them; `scripts/gen_semantic_fixtures.py`
//! regenerates it from the pinned sources. The Python suite runs the same
//! fixture through the bindings and the PNG path.
//!
//! What is asserted, and against whom:
//!
//! - per-class IoU and Acc, `mIoU`, `mAcc`, `aAcc` against mmseg to 1e-9 — the
//!   reference whose conventions hotcoco follows;
//! - per-class IoU and `mIoU` against torchmetrics to 1e-6, which reduces in
//!   `float32`;
//! - `fwIoU`, `aAcc`, `mAcc` against detectron2 to 1e-9, and `mIoU` against it
//!   only where no class is predicted-but-absent — on the cases that have one,
//!   detectron2 must *differ*, which is what makes the convention observable.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use hotcoco::Provenance;
use hotcoco::metrics::is_computed;
use hotcoco::semantic::{FolderPairing, METRIC_NAMES, SemanticEval};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    min_pred_only_cases: f64,
    min_boundary_differs: f64,
}

#[derive(Deserialize)]
struct Case {
    id: usize,
    num_classes: usize,
    ignore_index: u64,
    reduce_zero_label: bool,
    gt: Vec<Vec<Vec<i64>>>,
    pred: Vec<Vec<Vec<i64>>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    mmseg: Mmseg,
    torchmetrics: Torchmetrics,
    detectron2: Detectron2,
    boundary_iou: BoundaryIou,
}

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct Mmseg {
    IoU: Vec<Option<f64>>,
    Acc: Vec<Option<f64>>,
    Dice: Vec<Option<f64>>,
    aAcc: Option<f64>,
    mIoU: Option<f64>,
    mAcc: Option<f64>,
    mDice: Option<f64>,
}

/// The paper's definition through the authors' `mask_to_boundary`.
#[derive(Deserialize)]
#[allow(non_snake_case)]
struct BoundaryIou {
    BIoU: Vec<Option<f64>>,
    mBIoU: Option<f64>,
}

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct Torchmetrics {
    IoU: Vec<f64>,
    mIoU: Option<f64>,
}

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct Detectron2 {
    mIoU: Option<f64>,
    fwIoU: Option<f64>,
    mACC: Option<f64>,
    pACC: Option<f64>,
    mBIoU: Option<f64>,
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/semantic_expected.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture file"))
        .expect("fixture JSON")
}

/// An empty evaluator with the case's options.
fn fresh(case: &Case) -> SemanticEval {
    SemanticEval::new(case.num_classes)
        .ignore_index(Some(case.ignore_index))
        .reduce_zero_label(case.reduce_zero_label)
}

fn flat(map: &[Vec<i64>]) -> Vec<i64> {
    map.iter().flatten().copied().collect()
}

fn evaluator(case: &Case) -> SemanticEval {
    let mut ev = fresh(case);
    for (gt, pred) in case.gt.iter().zip(&case.pred) {
        ev.update(&flat(gt), &flat(pred), gt.len(), gt[0].len())
            .unwrap();
    }
    ev
}

/// NaN in a reference is the sentinel here: both mean "not computed".
fn close(got: f64, want: Option<f64>, tol: f64) -> bool {
    match want {
        Some(w) if w.is_nan() => !is_computed(got),
        Some(w) => (got - w).abs() <= tol,
        None => !is_computed(got),
    }
}

#[test]
fn matches_mmseg_per_class_and_in_the_means() {
    for case in &fixture().cases {
        let s = evaluator(case).scores();
        let m = &case.expected.mmseg;
        for (c, ((iou, acc), dice)) in m.IoU.iter().zip(&m.Acc).zip(&m.Dice).enumerate() {
            assert!(
                close(s.per_class[c].iou, *iou, 1e-9),
                "case {}: class {c} IoU {} vs mmseg {iou:?}",
                case.id,
                s.per_class[c].iou
            );
            assert!(
                close(s.per_class[c].acc, *acc, 1e-9),
                "case {}: class {c} Acc {} vs mmseg {acc:?}",
                case.id,
                s.per_class[c].acc
            );
            assert!(
                close(s.per_class[c].dice, *dice, 1e-9),
                "case {}: class {c} Dice {} vs mmseg {dice:?}",
                case.id,
                s.per_class[c].dice
            );
        }
        assert!(
            close(s.mdice, m.mDice, 1e-9),
            "case {}: mDice {} vs {:?}",
            case.id,
            s.mdice,
            m.mDice
        );
        assert!(
            close(s.miou, m.mIoU, 1e-9),
            "case {}: mIoU {} vs {:?}",
            case.id,
            s.miou,
            m.mIoU
        );
        assert!(
            close(s.macc, m.mAcc, 1e-9),
            "case {}: mAcc {} vs {:?}",
            case.id,
            s.macc,
            m.mAcc
        );
        assert!(
            close(s.aacc, m.aAcc, 1e-9),
            "case {}: aAcc {} vs {:?}",
            case.id,
            s.aacc,
            m.aAcc
        );
    }
}

/// Per-class boundary IoU as the paper defines it, through the authors'
/// `mask_to_boundary`; detectron2's label-map erosion differs on most cases.
#[test]
fn boundary_iou_matches_the_authors_kernel_and_not_detectron2() {
    let fixture = fixture();
    let mut differs = 0;
    for case in &fixture.cases {
        let ev = evaluator(case);
        let b = ev.boundary_scores().expect("boundary IoU is on by default");
        let want = &case.expected.boundary_iou;
        for (c, biou) in want.BIoU.iter().enumerate() {
            assert!(
                close(b.per_class[c], *biou, 1e-9),
                "case {}: class {c} BIoU {} vs {biou:?}",
                case.id,
                b.per_class[c]
            );
        }
        assert!(
            close(b.mbiou, want.mBIoU, 1e-9),
            "case {}: mBIoU {} vs {:?}",
            case.id,
            b.mbiou,
            want.mBIoU
        );
        if let Some(d) = case.expected.detectron2.mBIoU
            && is_computed(b.mbiou)
            && (b.mbiou - d).abs() > 1e-9
        {
            differs += 1;
        }
    }
    assert!(
        differs as f64 >= fixture.min_boundary_differs * fixture.cases.len() as f64,
        "detectron2 agreed on all but {differs} cases"
    );
}

#[test]
fn boundary_iou_off_leaves_the_rest_alone() {
    let case = fixture().cases.pop().unwrap();
    let on = evaluator(&case);
    let mut off = fresh(&case).boundary_dilation(None);
    for (gt, pred) in case.gt.iter().zip(&case.pred) {
        off.update(&flat(gt), &flat(pred), gt.len(), gt[0].len())
            .unwrap();
    }
    assert!(off.boundary_scores().is_none());
    assert_eq!(off.scores(), on.scores());
    let (a, b) = (on.stats(), off.stats());
    assert_eq!(&a[..5], &b[..5]);
    assert!(is_computed(a[5]) && b[5] == -1.0);
    assert!(
        !off.report().metrics.contains_key("mBIoU"),
        "a metric not computed is not reported"
    );
    assert!(on.report().metrics.contains_key("mBIoU"));
    let mut on = on;
    assert!(
        on.merge(&off).is_err(),
        "boundary settings must agree to merge"
    );
    assert!(
        SemanticEval::new(2)
            .boundary_dilation(Some(-0.1))
            .check_options()
            .is_err()
    );
}

#[test]
fn matches_torchmetrics_within_float32() {
    for case in &fixture().cases {
        let s = evaluator(case).scores();
        let t = &case.expected.torchmetrics;
        for (c, iou) in t.IoU.iter().enumerate() {
            // torchmetrics reports 0 for a class on neither side; hotcoco has
            // no number for it. Compare where both have one.
            if is_computed(s.per_class[c].iou) {
                assert!(
                    (s.per_class[c].iou - iou).abs() <= 1e-6,
                    "case {}: class {c} IoU {} vs torchmetrics {iou}",
                    case.id,
                    s.per_class[c].iou
                );
            }
        }
        if let Some(want) = t.mIoU {
            assert!(
                (s.miou - want).abs() <= 1e-6,
                "case {}: mIoU {} vs {}",
                case.id,
                s.miou,
                want
            );
        }
    }
}

#[test]
fn matches_detectron2_except_on_the_documented_corner() {
    let fx = fixture();
    let mut corner = 0usize;
    for case in &fx.cases {
        let s = evaluator(case).scores();
        let d = &case.expected.detectron2;
        // Over an all-ignored split detectron2 sums nothing and reports 0.0 where
        // its other means are NaN; hotcoco reports "not computed" for all four.
        if is_computed(s.aacc) {
            assert!(
                close(s.fwiou, d.fwIoU, 1e-9),
                "case {}: fwIoU {} vs {:?}",
                case.id,
                s.fwiou,
                d.fwIoU
            );
        }
        assert!(
            close(s.aacc, d.pACC, 1e-9),
            "case {}: aAcc {} vs {:?}",
            case.id,
            s.aacc,
            d.pACC
        );
        assert!(
            close(s.macc, d.mACC, 1e-9),
            "case {}: mAcc {} vs {:?}",
            case.id,
            s.macc,
            d.mACC
        );
        let pred_only = s
            .per_class
            .iter()
            .any(|c| c.iou == 0.0 && !is_computed(c.acc));
        if pred_only {
            corner += 1;
            assert!(
                !close(s.miou, d.mIoU, 1e-9),
                "case {}: a predicted-but-absent class should make detectron2's mIoU differ ({} vs {:?})",
                case.id,
                s.miou,
                d.mIoU
            );
        } else {
            assert!(
                close(s.miou, d.mIoU, 1e-9),
                "case {}: mIoU {} vs {:?}",
                case.id,
                s.miou,
                d.mIoU
            );
        }
    }
    assert!(
        corner as f64 >= fx.min_pred_only_cases * fx.cases.len() as f64,
        "only {corner} of {} cases hit the convention corner",
        fx.cases.len()
    );
}

#[test]
fn shards_merge_to_the_whole() {
    for case in fixture().cases.iter().filter(|c| c.gt.len() > 1) {
        let whole = evaluator(case);
        let mut parts: Vec<SemanticEval> = case
            .gt
            .iter()
            .zip(&case.pred)
            .map(|(gt, pred)| {
                let mut ev = fresh(case);
                ev.update(&flat(gt), &flat(pred), gt.len(), gt[0].len())
                    .unwrap();
                ev
            })
            .collect();
        let mut merged = parts.remove(0);
        for p in &parts {
            merged.merge(p).unwrap();
        }
        assert_eq!(merged, whole, "case {}", case.id);
    }
}

#[test]
fn merge_refuses_a_different_configuration() {
    let mut a = SemanticEval::new(3);
    let b = SemanticEval::new(4);
    assert!(
        a.merge(&b)
            .unwrap_err()
            .to_string()
            .contains("different classes")
    );
    let c = SemanticEval::new(3).ignore_index(None);
    assert!(
        a.merge(&c)
            .unwrap_err()
            .to_string()
            .contains("ignore_index")
    );
}

#[test]
fn a_rejected_pair_counts_nothing() {
    let mut ev = SemanticEval::new(2);
    ev.update(&[0u8, 1], &[0u8, 1], 1, 2).unwrap();
    let before = ev.clone();
    let err = ev
        .update(&[0u8, 1], &[0u8, 7], 1, 2)
        .unwrap_err()
        .to_string();
    assert!(err.contains("pred pixel 1 has label 7"), "{err}");
    assert_eq!(ev, before);
    let err = ev
        .update(&[0u8, 1, 1], &[0u8, 1], 1, 3)
        .unwrap_err()
        .to_string();
    assert!(err.contains("same number of pixels"), "{err}");
    assert_eq!(ev, before);
}

#[test]
fn an_ignored_pixel_is_neither_right_nor_wrong_whatever_it_predicts() {
    // Padding carries the ignore index on both sides; the prediction there
    // is out of range as a class and must not be checked.
    let mut ev = SemanticEval::new(2);
    ev.update(&[255u8, 0, 1], &[255u8, 0, 1], 1, 3).unwrap();
    assert_eq!(ev.confusion().total(), 2);
    assert_eq!(ev.stats()[0], 1.0);
    let mut ev = SemanticEval::new(2).reduce_zero_label(true);
    ev.update(&[0u8, 1, 2], &[255u8, 0, 1], 1, 3).unwrap();
    assert_eq!(ev.confusion().total(), 2);
    assert_eq!(ev.stats()[0], 1.0);
}

#[test]
fn reduce_zero_label_shifts_the_ground_truth_only() {
    // File labels 0 (other), 1, 2 over two classes; predictions already 0/1.
    let mut ev = SemanticEval::new(2).reduce_zero_label(true);
    ev.update(&[0u8, 1, 2, 2], &[1u8, 0, 1, 1], 1, 4).unwrap();
    assert_eq!(ev.confusion().total(), 3);
    assert_eq!(ev.confusion().get(0, 0), 1);
    assert_eq!(ev.confusion().get(1, 1), 2);
    assert_eq!(ev.scores().miou, 1.0);
}

/// mmseg relabels `0 → 255` and `255 → 254 → 255`, then masks `ignore_index`
/// on the shifted label. So a file label of 255 is unlabeled too, a shifted
/// label equal to the ignore index is masked, and an ignore index other than
/// 255 has no reference behavior and is refused.
#[test]
fn reduce_zero_label_follows_mmseg_on_255() {
    let mut ev = SemanticEval::new(300).reduce_zero_label(true);
    // File labels: 0 and 255 unlabeled; 256 shifts to 255 and is masked; 1 → 0.
    ev.update(&[0u16, 255, 256, 1], &[0u16, 0, 0, 0], 1, 4)
        .unwrap();
    assert_eq!(ev.confusion().total(), 1);
    assert_eq!(ev.confusion().get(0, 0), 1);

    let mut ev = SemanticEval::new(2)
        .reduce_zero_label(true)
        .ignore_index(Some(100));
    let err = ev
        .update(&[1u8, 2], &[0u8, 1], 1, 2)
        .unwrap_err()
        .to_string();
    assert!(err.contains("needs ignore_index 255"), "{err}");
    assert_eq!(ev.n_images(), 0);
}

#[test]
fn report_has_the_contract_shape() {
    let mut ev = SemanticEval::with_class_names(vec!["road".into(), "sky".into(), "void".into()]);
    ev.update(&[0u8, 0, 1, 255], &[0u8, 1, 1, 1], 1, 4).unwrap();
    let report = ev.report();
    assert_eq!(report.task, "semantic");
    assert_eq!(report.provenance, Provenance::ParityVerified);
    assert_eq!(
        report
            .metrics
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        {
            let mut names = METRIC_NAMES.to_vec();
            names.sort_unstable();
            names
        }
    );
    assert_eq!(report.per_class["road"]["IoU"], 0.5);
    assert_eq!(report.per_class["road"]["Acc"], 0.5);
    assert!(
        !report.per_class.contains_key("void"),
        "a class on neither side has no row"
    );
    assert!(report.per_group.is_empty() && report.curves.is_empty());
    assert_eq!(report.params["n_images"], 1);
    assert_eq!(report.params["n_pixels"], 3);
    assert_eq!(report.params["ignore_index"], 255);

    let results = ev.results();
    assert_eq!(results.per_class.len(), 3);
    assert_eq!(results.per_class[2].iou, -1.0);
    assert_eq!(results.n_pixels, 3);
    let json: serde_json::Value = serde_json::from_str(&results.to_json().unwrap()).unwrap();
    assert_eq!(json["mIoU"], results.miou);
    assert_eq!(json["per_class"][0]["IoU"], 0.5);
    assert_eq!(json["per_class"][0]["tp"], 1);
}

#[test]
fn summary_prints_one_row_per_class_and_the_means() {
    let mut ev = SemanticEval::new(2);
    ev.update(&[0u8, 1], &[0u8, 1], 1, 2).unwrap();
    let lines = ev.summarize_lines();
    assert_eq!(lines.len(), 2 + 2 + 2, "{lines:?}");
    assert!(
        lines[2].starts_with("class_0") && lines[2].contains("100.00"),
        "{lines:?}"
    );
    assert!(
        lines[5].starts_with("mIoU 100.00") && lines[5].contains("1 images, 2 pixels"),
        "{lines:?}"
    );
}

/// A 2×2 8-bit label map, grayscale or palette.
fn write_png(path: &std::path::Path, data: &[u8], color: png::ColorType) {
    let mut enc = png::Encoder::new(std::fs::File::create(path).unwrap(), 2, 2);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    if color == png::ColorType::Indexed {
        enc.set_palette(vec![0u8; 768]);
    }
    enc.write_header().unwrap().write_image_data(data).unwrap();
}

/// The folder path reads PNG files of every label-map kind and pairs them by
/// name; the mask of a mismatch names the files.
#[test]
fn folders_pair_by_file_name() {
    let dir = tempfile::tempdir().unwrap();
    let (gt_dir, pred_dir) = (dir.path().join("gt"), dir.path().join("pred"));
    std::fs::create_dir_all(&gt_dir).unwrap();
    std::fs::create_dir_all(&pred_dir).unwrap();
    write_png(
        &gt_dir.join("a.png"),
        &[0, 0, 1, 255],
        png::ColorType::Grayscale,
    );
    write_png(
        &pred_dir.join("a.png"),
        &[0, 1, 1, 1],
        png::ColorType::Indexed,
    );
    write_png(
        &gt_dir.join("b.png"),
        &[1, 1, 1, 1],
        png::ColorType::Indexed,
    );
    write_png(
        &pred_dir.join("b.png"),
        &[1, 1, 0, 1],
        png::ColorType::Grayscale,
    );
    std::fs::write(gt_dir.join("notes.txt"), "ignored").unwrap();

    let mut ev = SemanticEval::new(2);
    assert_eq!(
        ev.update_folders(&gt_dir, &pred_dir, &FolderPairing::default())
            .unwrap(),
        2
    );
    let mut by_array = SemanticEval::new(2);
    by_array
        .update(&[0u8, 0, 1, 255], &[0u8, 1, 1, 1], 1, 4)
        .unwrap();
    by_array
        .update(&[1u8, 1, 1, 1], &[1u8, 1, 0, 1], 1, 4)
        .unwrap();
    assert_eq!(ev, by_array);

    write_png(
        &gt_dir.join("c.png"),
        &[0, 0, 0, 0],
        png::ColorType::Grayscale,
    );
    let err = SemanticEval::new(2)
        .update_folders(&gt_dir, &pred_dir, &FolderPairing::default())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("1 ground-truth file(s) have no prediction") && err.contains("c.png"),
        "{err}"
    );
    std::fs::remove_file(gt_dir.join("c.png")).unwrap();

    // A bad pair poisons the whole call, and nothing is counted.
    write_png(
        &pred_dir.join("b.png"),
        &[1, 1, 9, 1],
        png::ColorType::Grayscale,
    );
    let mut fresh = SemanticEval::new(2);
    let err = fresh
        .update_folders(&gt_dir, &pred_dir, &FolderPairing::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("b.png") && err.contains("label 9"), "{err}");
    assert_eq!(fresh, SemanticEval::new(2));
}

fn write_gray_png(path: &std::path::Path, data: &[u8]) {
    write_png(path, data, png::ColorType::Grayscale);
}

/// Cityscapes' layout: a folder per city, several files per frame telling
/// each other apart by suffix, and the predictions flat under the image's
/// name.
#[test]
fn folders_pair_recursively_by_stem_and_suffix() {
    let dir = tempfile::tempdir().unwrap();
    let gt = dir.path().join("gtFine").join("val");
    let pred = dir.path().join("preds");
    for city in ["aachen", "bremen"] {
        std::fs::create_dir_all(gt.join(city)).unwrap();
    }
    std::fs::create_dir_all(&pred).unwrap();
    let label = "_gtFine_labelTrainIds.png";
    let image = "_leftImg8bit.png";
    write_gray_png(
        &gt.join("aachen")
            .join(format!("aachen_000000_000019{label}")),
        &[0, 0, 1, 255],
    );
    // Another file of the frame: the wrong suffix, and not a label map.
    write_gray_png(
        &gt.join("aachen")
            .join("aachen_000000_000019_gtFine_color.png"),
        &[9, 9, 9, 9],
    );
    write_gray_png(
        &gt.join("bremen")
            .join(format!("bremen_000001_000019{label}")),
        &[1, 1, 1, 1],
    );
    write_gray_png(
        &pred.join(format!("aachen_000000_000019{image}")),
        &[0, 1, 1, 1],
    );
    write_gray_png(
        &pred.join(format!("bremen_000001_000019{image}")),
        &[1, 1, 0, 1],
    );

    let pairing = FolderPairing {
        recursive: true,
        gt_suffix: label.into(),
        pred_suffix: image.into(),
    };
    let mut ev = SemanticEval::new(2);
    assert_eq!(ev.update_folders(&gt, &pred, &pairing).unwrap(), 2);
    let mut by_array = SemanticEval::new(2);
    by_array
        .update(&[0u8, 0, 1, 255], &[0u8, 1, 1, 1], 1, 4)
        .unwrap();
    by_array
        .update(&[1u8, 1, 1, 1], &[1u8, 1, 0, 1], 1, 4)
        .unwrap();
    assert_eq!(ev, by_array);

    // Without `recursive` the city folders are not entered.
    let flat = FolderPairing {
        recursive: false,
        ..pairing.clone()
    };
    let err = SemanticEval::new(2)
        .update_folders(&gt, &pred, &flat)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no files ending in"), "{err}");

    // Two predictions with one stem are ambiguous, wherever they sit.
    std::fs::create_dir_all(pred.join("again")).unwrap();
    write_gray_png(
        &pred
            .join("again")
            .join(format!("aachen_000000_000019{image}")),
        &[0, 1, 1, 1],
    );
    let err = SemanticEval::new(2)
        .update_folders(&gt, &pred, &pairing)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("two prediction files") && err.contains("aachen_000000_000019"),
        "{err}"
    );

    // The default pairing is the `.png` files at the top of each folder, by name.
    assert_eq!(
        FolderPairing::default(),
        FolderPairing {
            recursive: false,
            gt_suffix: ".png".into(),
            pred_suffix: ".png".into()
        }
    );
}

/// A label error names both files and its side, so a ground-truth file with
/// the wrong id scheme is blamed on the ground truth, not the prediction.
#[test]
fn a_png_label_error_names_both_files_and_the_side() {
    let dir = tempfile::tempdir().unwrap();
    let (gt, pred) = (dir.path().join("gt.png"), dir.path().join("pred.png"));
    write_gray_png(&gt, &[0, 26, 0, 0]);
    write_gray_png(&pred, &[0, 0, 0, 0]);
    let mut ev = SemanticEval::new(19);
    let err = ev.update_png(&gt, &pred).unwrap_err().to_string();
    assert!(err.contains("gt.png against"), "{err}");
    assert!(err.contains("pred.png: gt pixel 1 has label 26"), "{err}");
}

/// A file that cannot be read keeps its I/O kind; only a file that is not a
/// label map is bad input.
#[test]
fn an_unreadable_label_map_is_an_io_error() {
    let dir = tempfile::tempdir().unwrap();
    let (gt, pred) = (dir.path().join("gt.png"), dir.path().join("pred.png"));
    write_gray_png(&gt, &[0, 0, 0, 0]);
    std::fs::create_dir(&pred).unwrap();
    let err = SemanticEval::new(2).update_png(&gt, &pred).unwrap_err();
    assert!(matches!(err, hotcoco::Error::Io(_)), "{err}");
    assert!(err.to_string().contains("pred.png"), "{err}");
}

/// A truncated or empty file is bad input, not an unreadable one: the decoder
/// reports the missing bytes as an end-of-file error, which is about the
/// file's content, and `ValueError` is what a file that is not a label map
/// raises on every other path.
#[test]
fn a_truncated_label_map_is_bad_input() {
    let dir = tempfile::tempdir().unwrap();
    let (gt, pred) = (dir.path().join("gt.png"), dir.path().join("pred.png"));
    write_gray_png(&gt, &[0, 0, 0, 0]);
    let bytes = std::fs::read(&gt).unwrap();
    std::fs::write(&pred, &bytes[..bytes.len() / 2]).unwrap();
    let err = SemanticEval::new(2).update_png(&gt, &pred).unwrap_err();
    assert!(matches!(err, hotcoco::Error::InvalidInput(_)), "{err}");
    std::fs::write(&pred, b"").unwrap();
    let err = SemanticEval::new(2).update_png(&gt, &pred).unwrap_err();
    assert!(matches!(err, hotcoco::Error::InvalidInput(_)), "{err}");
}
