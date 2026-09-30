use std::collections::HashMap;
use std::collections::hash_map::Entry;

use serde::Serialize;

use super::COCOeval;

/// Status of a detection annotation at a specific IoU threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DtStatus {
    /// True positive — matched a ground truth annotation.
    Tp,
    /// False positive — no matching ground truth.
    Fp,
}

/// Status of a ground truth annotation at a specific IoU threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GtStatus {
    /// Matched by a detection.
    Matched,
    /// False negative — no detection matched this ground truth.
    Fn,
}

/// Per-annotation TP/FP/FN index and matching pairs.
#[derive(Debug, Clone, Serialize)]
pub struct AnnotationIndex {
    /// Detection annotation ID → TP or FP.
    pub dt_status: HashMap<u64, DtStatus>,
    /// Ground truth annotation ID → Matched or FN.
    pub gt_status: HashMap<u64, GtStatus>,
    /// TP detection → matched GT annotation ID.
    pub dt_match: HashMap<u64, u64>,
    /// Matched GT → the detection that matched it.
    pub gt_match: HashMap<u64, u64>,
}

/// Error profile classification for an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ErrorProfile {
    /// No errors: all GTs matched, no spurious detections.
    Perfect,
    /// More false positives than false negatives.
    FpHeavy,
    /// More false negatives than false positives.
    FnHeavy,
    /// Both FP and FN present in roughly equal proportion.
    Mixed,
}

/// Per-image evaluation summary with scores and error profile.
#[derive(Debug, Clone, Serialize)]
pub struct ImageSummary {
    pub tp: u32,
    pub fp: u32,
    pub fn_count: u32,
    /// F1 score: `2*tp / (2*tp + fp + fn)`. 1.0 for images with no annotations and no detections.
    pub f1: f64,
    /// Average precision at the selected IoU threshold, computed from this image's detections only.
    pub ap: f64,
    pub error_profile: ErrorProfile,
}

/// Type of suspected label error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LabelErrorType {
    /// A high-confidence detection overlaps a GT of a different category.
    /// Suggests the GT label is wrong.
    WrongLabel,
    /// A high-confidence detection has no nearby GT at all.
    /// Suggests a missing annotation in the ground truth.
    MissingAnnotation,
}

/// A suspected label error in the ground truth.
#[derive(Debug, Clone, Serialize)]
pub struct LabelError {
    pub image_id: u64,
    /// The false-positive detection that triggered the flag.
    pub dt_id: u64,
    pub dt_score: f64,
    pub dt_category_id: u64,
    /// The GT annotation suspected of being wrong (None for MissingAnnotation).
    pub gt_id: Option<u64>,
    pub gt_category_id: Option<u64>,
    /// Bbox IoU between the detection and the GT (0.0 for MissingAnnotation).
    pub iou: f64,
    pub error_type: LabelErrorType,
}

/// Per-image diagnostic results: annotation index, image scores, and label error candidates.
///
/// Produced by [`COCOeval::image_diagnostics`]: per-annotation TP/FP/FN
/// classification plus per-image F1/AP scores, error profiles, and label error
/// detection.
#[derive(Debug, Clone, Serialize)]
pub struct ImageDiagnostics {
    /// Per-annotation TP/FP/FN classification and matching pairs.
    pub annotations: AnnotationIndex,
    /// Per-image summary with TP/FP/FN counts, F1, AP, and error profile.
    pub images: HashMap<u64, ImageSummary>,
    /// Suspected label errors, sorted by detection score descending.
    pub label_errors: Vec<LabelError>,
    /// The IoU threshold used (snapped to nearest in params).
    pub iou_thr: f64,
}

/// Bbox IoU at or above which a high-confidence FP overlapping an undetected
/// GT of a *different* category is flagged as a suspected wrong label. The
/// value mirrors TIDE's positive-match threshold (`pos_thr` = 0.5): the
/// detection localizes the box well enough that only the class disagrees.
const WRONG_LABEL_IOU: f64 = 0.5;

/// Bbox IoU below which a high-confidence FP counts as overlapping nothing and
/// is flagged as a suspected missing annotation. Mirrors TIDE's background
/// threshold (`bg_thr` = 0.1): under it, a detection is "on background" rather
/// than a poor localization of something annotated.
const MISSING_ANNOTATION_IOU: f64 = 0.1;

/// How many times one error kind must outnumber the other before an image's
/// profile reads [`ErrorProfile::FpHeavy`]/[`ErrorProfile::FnHeavy`] rather
/// than [`ErrorProfile::Mixed`]. 2× is a judgment call — far enough from
/// parity that the label survives one stray detection on a small image.
const PROFILE_DOMINANCE_FACTOR: u32 = 2;

/// Bbox IoU between two `[x, y, w, h]` boxes — the shared scalar kernel.
///
/// Diagnostics compares detections against plain (non-crowd) ground truth, so
/// the crowd flag is pinned off here rather than repeated at each call site.
/// Named distinctly from [`crate::primitives::sim::bbox_iou`], which is the D×G
/// matrix kernel.
fn bbox_iou_plain(a: [f64; 4], b: [f64; 4]) -> f64 {
    crate::primitives::sim::bbox_iou_pair(a, b, false)
}

/// Compute AP from detections for a single image given the GT count.
///
/// Reuses the standard COCO 101-point interpolation with monotone precision correction
/// via [`crate::metrics::counts::average_precision`].
///
/// `scores` and `matched` are parallel arrays over this image's non-ignored
/// detections, in any order — [`average_precision`](crate::metrics::counts::average_precision)
/// ranks them itself. They are collected as two vectors during the classification
/// walk rather than as one vector of pairs, so no per-image split is needed here;
/// the caller's own score sort is gone for the same reason, since it was ranking
/// the detections a second time with the identical stable comparator.
///
/// `n_gt` is the total number of non-ignored GT annotations for this image.
/// `rec_thrs` is the caller's recall grid — passed in rather than defaulted, so a
/// custom-grid evaluator does not report per-image AP on a different axis than the
/// AP it prints, and so the grid is built once per call rather than once per image.
///
/// An image with no ground truth scores `1.0` when nothing was predicted: this is
/// a per-image *quality* score, where a correctly-empty image is perfect. (TIDE's
/// corpus AP deliberately uses the opposite convention for `n_gt == 0` — see the
/// [`counts`](crate::metrics::counts) module note.)
fn compute_image_ap(scores: &[f64], matched: &[bool], n_gt: u32, rec_thrs: &[f64]) -> f64 {
    if n_gt == 0 {
        return if scores.is_empty() { 1.0 } else { 0.0 };
    }

    crate::metrics::counts::average_precision(scores, matched, None, n_gt as usize, rec_thrs)
}

/// One image's running tally during the classification walk.
///
/// `scores`/`matched` are two vectors rather than one of pairs because that is
/// the shape [`compute_image_ap`] takes — building pairs here only to split them
/// again per image allocated twice more for no reason.
#[derive(Default)]
struct ImageTally {
    tp: u32,
    fp: u32,
    fn_count: u32,
    scores: Vec<f64>,
    matched: Vec<bool>,
}

/// A high-confidence false positive, and what it might be evidence of.
struct FpDt {
    dt_id: u64,
    score: f64,
    cat_id: u64,
    bbox: [f64; 4],
}

/// An undetected ground truth, as a candidate mislabel.
struct FnGt {
    gt_id: u64,
    cat_id: u64,
    bbox: [f64; 4],
}

/// Everything in one image that the label-error scan compares against.
///
/// One map keyed by image rather than three: the FP list, the FN list and the
/// all-GT list are always read together for the same image, so keeping them in
/// step is this type's job rather than the caller's.
#[derive(Default)]
struct ImageCandidates {
    fps: Vec<FpDt>,
    fn_gts: Vec<FnGt>,
    /// Every GT bbox in the image, matched or not — the `missing_annotation`
    /// check asks whether a detection overlaps *anything*, not just failures.
    all_gt_bboxes: Vec<[f64; 4]>,
}

impl COCOeval {
    /// Compute per-image diagnostics: annotation TP/FP/FN index, per-image F1 and AP
    /// scores, error profiles, and label error candidates.
    ///
    /// Requires [`evaluate`](COCOeval::evaluate) to have been called first.
    ///
    /// # Arguments
    ///
    /// - `iou_thr` — IoU threshold for TP/FP classification (snapped to nearest in params).
    /// - `score_thr` — minimum detection confidence to consider for label error detection.
    ///
    /// # Label error detection
    ///
    /// Two types of suspected GT errors are flagged:
    ///
    /// - **Wrong label**: a high-confidence FP detection (score ≥ `score_thr`) that overlaps
    ///   an unmatched (FN) GT of a *different* category with bbox IoU ≥ 0.5.
    /// - **Missing annotation**: a high-confidence FP detection with no nearby GT at all
    ///   (max bbox IoU < 0.1 against all GTs in the image).
    pub fn image_diagnostics(
        &self,
        iou_thr: f64,
        score_thr: f64,
    ) -> crate::error::Result<ImageDiagnostics> {
        if !self.evaluated() {
            return Err("image_diagnostics() requires evaluate() to be called first".into());
        }

        // Snap to nearest IoU threshold — the reported `iou_thr` says which one.
        let t_idx = self.params.nearest_iou_thr_idx(iou_thr);

        let (annotations, tallies) = self.classify_annotations(t_idx);
        let images = summarize_images(tallies, &self.params.rec_thrs);
        let label_errors = self.find_label_errors(&annotations, score_thr);

        Ok(ImageDiagnostics {
            annotations,
            images,
            label_errors,
            iou_thr: self.params.iou_thrs[t_idx],
        })
    }

    /// Pass 1 — walk every in-scope cell, sorting detections into TP/FP and
    /// ground truths into matched/FN, and tallying both per image.
    ///
    /// An annotation is classified once even though it can appear in several
    /// cells; the first cell to reach it wins.
    fn classify_annotations(&self, t_idx: usize) -> (AnnotationIndex, HashMap<u64, ImageTally>) {
        let mut index = AnnotationIndex {
            dt_status: HashMap::new(),
            gt_status: HashMap::new(),
            dt_match: HashMap::new(),
            gt_match: HashMap::new(),
        };
        let mut tallies: HashMap<u64, ImageTally> = HashMap::new();

        // `default_cells` owns the (area = "all", default max_det) predicate.
        for eval_img in self.default_cells() {
            // Shape trust, not bounds checks: `evaluate()` builds every
            // `EvalImg` with one threshold row per `params.iou_thrs` entry and
            // one column per `dt_ids`/`gt_ids` element, and `t_idx` comes from
            // `nearest_iou_thr_idx` over the same list. `accumulate` and TIDE
            // index the identical shapes unguarded; runtime guards here would
            // only hide a real shape defect as silently skipped cells.
            debug_assert!(t_idx < eval_img.dt_matched.num_rows());

            let matched = eval_img.dt_matched.row(t_idx);
            let ignored = eval_img.dt_ignore.row(t_idx);
            let matches = eval_img.dt_matches.row(t_idx);
            debug_assert_eq!(matched.len(), matches.len());
            debug_assert_eq!(eval_img.dt_ids.len(), matched.len());
            debug_assert_eq!(eval_img.dt_ids.len(), ignored.len());

            // Resolved once per cell rather than once per annotation — every
            // annotation in this cell belongs to the same image.
            let tally = tallies.entry(eval_img.image_id).or_default();

            // `entry`, not `contains_key` then `insert`: one hash lookup instead
            // of two, and the "already seen" test and the write cannot drift apart.
            for (d, &did) in eval_img.dt_ids.iter().enumerate() {
                if ignored[d] {
                    continue;
                }
                let Entry::Vacant(slot) = index.dt_status.entry(did) else {
                    continue;
                };

                let is_tp = matched[d];
                if is_tp {
                    slot.insert(DtStatus::Tp);
                    index.dt_match.insert(did, matches[d]);
                    index.gt_match.insert(matches[d], did);
                    tally.tp += 1;
                } else {
                    slot.insert(DtStatus::Fp);
                    tally.fp += 1;
                }

                tally.scores.push(eval_img.dt_scores[d]);
                tally.matched.push(is_tp);
            }

            let gt_matched_at_t = eval_img.gt_matched.row(t_idx);
            debug_assert_eq!(eval_img.gt_ids.len(), gt_matched_at_t.len());
            for (g, &gid) in eval_img.gt_ids.iter().enumerate() {
                let Entry::Vacant(slot) = index.gt_status.entry(gid) else {
                    continue;
                };
                // `counts_as_miss`, not `!gt_ignore` — otherwise per-image
                // diagnostics disagree with `stats[0]` about whether an undetected
                // Open Images group-of box is a false negative.
                if !eval_img.counts_as_miss(g) {
                    continue;
                }
                if gt_matched_at_t[g] {
                    slot.insert(GtStatus::Matched);
                } else {
                    slot.insert(GtStatus::Fn);
                    tally.fn_count += 1;
                }
            }
        }

        (index, tallies)
    }

    /// Pass 3 — flag ground truths that look wrong, judged by the detections that
    /// disagree with them.
    ///
    /// Only high-confidence false positives are evidence: a low-scoring FP is far
    /// likelier to be the model's mistake than the annotator's.
    fn find_label_errors(&self, index: &AnnotationIndex, score_thr: f64) -> Vec<LabelError> {
        let mut by_image: HashMap<u64, ImageCandidates> = HashMap::new();

        for (&dt_id, &status) in &index.dt_status {
            if status != DtStatus::Fp {
                continue;
            }
            let Some(ann) = self.coco_dt.get_ann(dt_id) else {
                continue;
            };
            let (Some(score), Some(bbox)) = (ann.score, ann.bbox) else {
                continue;
            };
            if score < score_thr {
                continue;
            }
            by_image.entry(ann.image_id).or_default().fps.push(FpDt {
                dt_id,
                score,
                cat_id: ann.category_id,
                bbox,
            });
        }

        for (&gt_id, &status) in &index.gt_status {
            let Some(ann) = self.coco_gt.get_ann(gt_id) else {
                continue;
            };
            let Some(bbox) = ann.bbox else {
                continue;
            };
            let cand = by_image.entry(ann.image_id).or_default();
            cand.all_gt_bboxes.push(bbox);
            if status == GtStatus::Fn {
                cand.fn_gts.push(FnGt {
                    gt_id,
                    cat_id: ann.category_id,
                    bbox,
                });
            }
        }

        let mut errors: Vec<LabelError> = by_image
            .iter()
            .flat_map(|(&img_id, cand)| {
                cand.fps
                    .iter()
                    .filter_map(move |fp| classify_label_error(img_id, fp, cand))
            })
            .collect();

        // `by_image` iterates in hash order, so this sort is what makes the output
        // deterministic — the `dt_id` tiebreak is load-bearing, not cosmetic.
        errors.sort_by(|a, b| {
            b.dt_score
                .partial_cmp(&a.dt_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.dt_id.cmp(&b.dt_id))
        });
        errors
    }
}

/// Pass 2 — turn each image's tally into its reported summary.
fn summarize_images(
    tallies: HashMap<u64, ImageTally>,
    rec_thrs: &[f64],
) -> HashMap<u64, ImageSummary> {
    tallies
        .into_iter()
        .map(|(img_id, t)| {
            // F1 straight from the counts: algebraically 2PR/(P+R), but computed as
            // one division of exact integers rather than two divisions fed into
            // `f_beta`. Same value in real arithmetic, and this form has no
            // rounding to differ in — these numbers are gate-compared.
            //
            // The empty-image convention (nothing to detect, nothing predicted, so
            // nothing to get wrong) lives here rather than in `metrics::counts`,
            // per that module's rule: the shared formulas take no policy flag.
            let denom = 2 * t.tp + t.fp + t.fn_count;
            let f1 = if denom == 0 {
                1.0
            } else {
                (2 * t.tp) as f64 / denom as f64
            };

            // No sort here: `average_precision` ranks its input with the same
            // stable comparator this used, so sorting first only sorted an array
            // that was about to be sorted again.
            let n_gt = t.tp + t.fn_count; // total non-ignored GT for this image
            let ap = compute_image_ap(&t.scores, &t.matched, n_gt, rec_thrs);

            let error_profile = match (t.fp, t.fn_count) {
                (0, 0) => ErrorProfile::Perfect,
                (f, n) if f > PROFILE_DOMINANCE_FACTOR * n => ErrorProfile::FpHeavy,
                (f, n) if n > PROFILE_DOMINANCE_FACTOR * f => ErrorProfile::FnHeavy,
                _ => ErrorProfile::Mixed,
            };

            (
                img_id,
                ImageSummary {
                    tp: t.tp,
                    fp: t.fp,
                    fn_count: t.fn_count,
                    f1,
                    ap,
                    error_profile,
                },
            )
        })
        .collect()
}

/// Decide whether one high-confidence false positive indicts the ground truth.
///
/// `WrongLabel` is checked first and returns outright: a detection that lands on
/// a mislabeled box is not also evidence that the box is missing.
fn classify_label_error(img_id: u64, fp: &FpDt, cand: &ImageCandidates) -> Option<LabelError> {
    // The most-overlapping undetected GT of a *different* category. Strictly
    // greater, so ties keep the earliest, and a zero overlap never claims the
    // slot. Deliberately not spelled with the greedy-matcher's variable names —
    // this ranks candidate annotation defects, it does not assign detections.
    let mut best_fn: Option<(f64, &FnGt)> = None;
    for fg in cand.fn_gts.iter().filter(|fg| fg.cat_id != fp.cat_id) {
        let overlap = bbox_iou_plain(fp.bbox, fg.bbox);
        if overlap > best_fn.map_or(0.0, |(best, _)| best) {
            best_fn = Some((overlap, fg));
        }
    }

    if let Some((iou, fg)) = best_fn {
        if iou >= WRONG_LABEL_IOU {
            return Some(LabelError {
                image_id: img_id,
                dt_id: fp.dt_id,
                dt_score: fp.score,
                dt_category_id: fp.cat_id,
                gt_id: Some(fg.gt_id),
                gt_category_id: Some(fg.cat_id),
                iou,
                error_type: LabelErrorType::WrongLabel,
            });
        }
    }

    // No nearby GT at all — the detection is probably right and the annotation
    // absent.
    let max_iou_any_gt = cand
        .all_gt_bboxes
        .iter()
        .map(|&gt_bbox| bbox_iou_plain(fp.bbox, gt_bbox))
        .fold(0.0f64, f64::max);

    (max_iou_any_gt < MISSING_ANNOTATION_IOU).then_some(LabelError {
        image_id: img_id,
        dt_id: fp.dt_id,
        dt_score: fp.score,
        dt_category_id: fp.cat_id,
        gt_id: None,
        gt_category_id: None,
        iou: 0.0,
        error_type: LabelErrorType::MissingAnnotation,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coco::COCO;
    use crate::detection::COCOeval;
    use crate::params::IouType;
    use crate::types::{Annotation, Dataset};

    fn make_gt(json: serde_json::Value) -> COCO {
        let ds: Dataset = serde_json::from_value(json).unwrap();
        COCO::from_dataset(ds)
    }

    fn make_dt(gt: &COCO, anns_json: serde_json::Value) -> COCO {
        let anns: Vec<Annotation> = serde_json::from_value(anns_json).unwrap();
        gt.load_res_anns(anns).unwrap()
    }

    fn make_gt_dt() -> (COCO, COCO) {
        let gt = make_gt(serde_json::json!({
            "images": [
                {"id": 1, "width": 100, "height": 100},
                {"id": 2, "width": 100, "height": 100}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }));
        let dt = make_dt(
            &gt,
            serde_json::json!([
                {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
                {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.8},
                {"image_id": 2, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.7}
            ]),
        );
        (gt, dt)
    }

    #[test]
    fn test_diagnostics_perfect_detection() {
        let (gt, dt) = make_gt_dt();
        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.evaluate();

        let diag = ev.image_diagnostics(0.5, 0.5).unwrap();

        // All 3 detections should be TP
        assert_eq!(diag.annotations.dt_status.len(), 3);
        for status in diag.annotations.dt_status.values() {
            assert_eq!(*status, DtStatus::Tp);
        }

        // All 3 GTs should be matched
        assert_eq!(diag.annotations.gt_status.len(), 3);
        for status in diag.annotations.gt_status.values() {
            assert_eq!(*status, GtStatus::Matched);
        }

        // Image 1: 2 TP, 0 FP, 0 FN → F1 = 1.0
        let img1 = &diag.images[&1];
        assert_eq!(img1.tp, 2);
        assert_eq!(img1.fp, 0);
        assert_eq!(img1.fn_count, 0);
        assert!((img1.f1 - 1.0).abs() < 1e-9);
        assert_eq!(img1.error_profile, ErrorProfile::Perfect);

        // Image 2: 1 TP, 0 FP, 0 FN → F1 = 1.0
        let img2 = &diag.images[&2];
        assert_eq!(img2.tp, 1);
        assert!((img2.f1 - 1.0).abs() < 1e-9);

        // No label errors
        assert!(diag.label_errors.is_empty());
    }

    #[test]
    fn test_diagnostics_with_fp_and_fn() {
        let gt = make_gt(serde_json::json!({
            "images": [{"id": 1, "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [60, 60, 20, 20], "area": 400, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "cat"}]
        }));
        let dt = make_dt(
            &gt,
            serde_json::json!([
                {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
                {"image_id": 1, "category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.6}
            ]),
        );

        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.evaluate();

        let diag = ev.image_diagnostics(0.5, 0.5).unwrap();

        let img = &diag.images[&1];
        assert_eq!(img.tp, 1);
        assert_eq!(img.fp, 1);
        assert_eq!(img.fn_count, 1);
        // F1 = 2*1 / (2*1 + 1 + 1) = 0.5
        assert!((img.f1 - 0.5).abs() < 1e-9);
        assert_eq!(img.error_profile, ErrorProfile::Mixed);
    }

    #[test]
    fn test_diagnostics_wrong_label() {
        // DT predicts "dog" at a location where GT says "cat"
        let gt = make_gt(serde_json::json!({
            "images": [{"id": 1, "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }));
        let dt = make_dt(
            &gt,
            serde_json::json!([
                {"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "score": 0.95}
            ]),
        );

        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.evaluate();

        let diag = ev.image_diagnostics(0.5, 0.5).unwrap();

        // The "dog" detection is FP (no "dog" GT), the "cat" GT is FN (no "cat" DT)
        assert_eq!(diag.images[&1].fp, 1);
        assert_eq!(diag.images[&1].fn_count, 1);

        // Should detect a WrongLabel error
        assert_eq!(diag.label_errors.len(), 1);
        let err = &diag.label_errors[0];
        assert_eq!(err.error_type, LabelErrorType::WrongLabel);
        assert_eq!(err.dt_category_id, 2); // dog
        assert_eq!(err.gt_category_id, Some(1)); // cat
        assert!(err.iou > 0.9); // near-perfect overlap
    }

    #[test]
    fn test_diagnostics_missing_annotation() {
        // DT detects something where no GT exists at all
        let gt = make_gt(serde_json::json!({
            "images": [{"id": 1, "width": 200, "height": 200}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "cat"}]
        }));
        let dt = make_dt(
            &gt,
            serde_json::json!([
                {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
                {"image_id": 1, "category_id": 1, "bbox": [150, 150, 20, 20], "score": 0.85}
            ]),
        );

        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.evaluate();

        let diag = ev.image_diagnostics(0.5, 0.5).unwrap();

        // One TP (matched GT), one FP (far away)
        assert_eq!(diag.images[&1].tp, 1);
        assert_eq!(diag.images[&1].fp, 1);

        // The far-away FP should be flagged as MissingAnnotation
        assert_eq!(diag.label_errors.len(), 1);
        let err = &diag.label_errors[0];
        assert_eq!(err.error_type, LabelErrorType::MissingAnnotation);
        assert!(err.gt_id.is_none());
    }

    #[test]
    fn test_diagnostics_requires_evaluate() {
        let gt = make_gt(serde_json::json!({
            "images": [{"id": 1, "width": 100, "height": 100}],
            "annotations": [],
            "categories": [{"id": 1, "name": "cat"}]
        }));
        let dt = make_dt(&gt, serde_json::json!([]));
        let ev = COCOeval::new(gt, dt, IouType::Bbox);

        assert!(ev.image_diagnostics(0.5, 0.5).is_err());
    }

    #[test]
    fn test_bbox_iou_exact_overlap() {
        let a = [10.0, 10.0, 20.0, 20.0];
        assert!((bbox_iou_plain(a, a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [50.0, 50.0, 10.0, 10.0];
        assert_eq!(bbox_iou_plain(a, b), 0.0);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod image_ap_tests {
    use super::compute_image_ap;

    /// `ImageSummary.ap` had no assertion anywhere in the repo, despite surfacing
    /// in `coco eval --diagnostics`, the browse viewer, and the dashboard.
    ///
    /// The values below are closed-form on the 101-point grid, not recordings.
    /// With `n_gt` ground truths and detections in score order, recall after `k`
    /// true positives is `k / n_gt`, and the interpolated precision at a recall
    /// threshold is the best precision achieved at or beyond it. Averaging over
    /// the 101 thresholds gives a number that can be written down in advance.
    #[test]
    fn image_ap_matches_closed_form() {
        let rec_thrs = crate::params::default_rec_thrs();
        let n_thr = rec_thrs.len() as f64; // 101

        // Perfect: one GT, one matching detection. Recall reaches 1.0, so every
        // threshold is reachable, each at the lone-TP precision `coco_precision`
        // owns; the 101-point mean lands within an ulp of it.
        let lone_tp = crate::metrics::counts::coco_precision(1.0, 0.0);
        let perfect = compute_image_ap(&[0.9], &[true], 1, &rec_thrs);
        assert!((perfect - lone_tp).abs() <= f64::EPSILON, "{perfect}");

        // All false positives against one GT: recall never leaves 0, so only the
        // r=0 threshold is reachable and precision there is 0.
        assert_eq!(
            compute_image_ap(&[0.9, 0.8], &[false, false], 1, &rec_thrs),
            0.0
        );

        // One TP out of two GT, listed first. Recall tops out at 0.5, so the
        // reachable thresholds are r <= 0.5 — 51 of the 101 — each at the
        // lone-TP precision. AP = 51/101, to within the guard term.
        let ap = compute_image_ap(&[0.9, 0.8], &[true, false], 2, &rec_thrs);
        let reachable = rec_thrs.iter().filter(|&&t| t <= 0.5 + 1e-12).count() as f64;
        assert!(
            (ap - reachable / n_thr).abs() < 1e-12,
            "expected {}/{n_thr} = {}, got {ap}",
            reachable,
            reachable / n_thr
        );

        // FP ranked *above* the TP. Recall still tops out at 0.5, but precision at
        // that recall is 1/2 — VOC interpolation cannot rescue it, because there is
        // no higher-precision point further right.
        let ap_fp_first = compute_image_ap(&[0.9, 0.8], &[false, true], 2, &rec_thrs);
        assert!(
            (ap_fp_first - 0.5 * reachable / n_thr).abs() < 1e-12,
            "expected half the previous AP, got {ap_fp_first}"
        );

        // Ranking matters, and in the direction one would expect.
        assert!(
            ap_fp_first < ap,
            "a false positive ranked above the true positive must not score higher"
        );
    }

    /// The `n_gt == 0` convention is deliberately the *opposite* of TIDE's, which
    /// makes it exactly the kind of thing a copy-paste would silently invert.
    #[test]
    fn empty_image_is_perfect_only_when_nothing_was_predicted() {
        let rec_thrs = crate::params::default_rec_thrs();
        assert_eq!(compute_image_ap(&[], &[], 0, &rec_thrs), 1.0);
        assert_eq!(compute_image_ap(&[0.9], &[false], 0, &rec_thrs), 0.0);
        // No detections against real ground truth is a total miss, not a pass.
        assert_eq!(compute_image_ap(&[], &[], 3, &rec_thrs), 0.0);
    }
}
