use std::collections::HashMap;

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
    /// Suggests the GT label may be wrong.
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
/// Produced by [`COCOeval::image_diagnostics`]. Subsumes the per-annotation TP/FP/FN
/// classification previously done by `build_eval_index()` in Python, and adds per-image
/// F1/AP scores, error profiles, and label error detection.
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

/// Bbox IoU between two `[x, y, w, h]` boxes.
fn bbox_iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let x_overlap = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let y_overlap = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    if x_overlap <= 0.0 || y_overlap <= 0.0 {
        return 0.0;
    }
    let intersection = x_overlap * y_overlap;
    let union = a[2] * a[3] + b[2] * b[3] - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Compute AP from detections for a single image given the GT count.
///
/// Reuses the standard COCO 101-point interpolation with monotone precision correction
/// via [`super::accumulate::precision_recall_curve`].
///
/// `detections` is `(score, is_tp)` sorted by score descending.
/// `n_gt` is the total number of non-ignored GT annotations for this image.
fn compute_image_ap(detections: &[(f64, bool)], n_gt: u32) -> f64 {
    if n_gt == 0 {
        return if detections.is_empty() { 1.0 } else { 0.0 };
    }

    // Build cumulative TP/FP arrays
    let n = detections.len();
    let mut tp_cum = Vec::with_capacity(n);
    let mut fp_cum = Vec::with_capacity(n);
    let mut tp = 0.0f64;
    let mut fp = 0.0f64;
    for &(_score, is_tp) in detections {
        if is_tp {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        tp_cum.push(tp);
        fp_cum.push(fp);
    }

    let rec_thrs: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let (_final_recall, points) =
        super::accumulate::precision_recall_curve(&tp_cum, &fp_cum, n_gt as usize, &rec_thrs);

    // AP = mean precision at the 101 recall thresholds (unreached thresholds contribute 0)
    let sum: f64 = points.iter().map(|&(_, prec, _)| prec).sum();
    sum / rec_thrs.len() as f64
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
        if self.eval_imgs.is_empty() {
            return Err("image_diagnostics() requires evaluate() to be called first".into());
        }

        // Snap to nearest IoU threshold
        let t_idx = self
            .params
            .iou_thrs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - iou_thr).abs())
                    .partial_cmp(&((**b - iou_thr).abs()))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);
        let actual_iou_thr = self.params.iou_thrs[t_idx];

        // Use "all" area range
        let target_area_idx = self.params.area_range_idx("all").unwrap_or(0);
        let target_area = self.params.area_ranges[target_area_idx].range;

        // Last max_det (largest)
        let target_max_det = self.params.max_dets.iter().copied().max().unwrap_or(100);

        let mut dt_status: HashMap<u64, DtStatus> = HashMap::new();
        let mut gt_status: HashMap<u64, GtStatus> = HashMap::new();
        let mut dt_match_map: HashMap<u64, u64> = HashMap::new();
        let mut gt_match_map: HashMap<u64, u64> = HashMap::new();

        // Per-image: (score, is_tp) pairs for AP, and tp/fp/fn counts
        let mut img_detections: HashMap<u64, Vec<(f64, bool)>> = HashMap::new();
        let mut img_counts: HashMap<u64, (u32, u32, u32)> = HashMap::new(); // (tp, fp, fn)

        // Walk eval_imgs
        for eval_img in self.eval_imgs.iter().flatten() {
            if eval_img.area_rng != target_area || eval_img.max_det != target_max_det {
                continue;
            }
            if t_idx >= eval_img.dt_matched.len() {
                continue;
            }

            let img_id = eval_img.image_id;
            let matched = &eval_img.dt_matched[t_idx];
            let ignored = &eval_img.dt_ignore[t_idx];
            let matches = &eval_img.dt_matches[t_idx];
            debug_assert_eq!(matched.len(), matches.len());

            let counts = img_counts.entry(img_id).or_insert((0, 0, 0));

            // Classify detections
            for (d, &did) in eval_img.dt_ids.iter().enumerate() {
                if d >= ignored.len() || ignored[d] {
                    continue;
                }
                if dt_status.contains_key(&did) {
                    continue;
                }

                let is_tp = d < matched.len() && matched[d];
                if is_tp {
                    dt_status.insert(did, DtStatus::Tp);
                    let gt_id = matches[d];
                    dt_match_map.insert(did, gt_id);
                    gt_match_map.insert(gt_id, did);
                    counts.0 += 1;
                } else {
                    dt_status.insert(did, DtStatus::Fp);
                    counts.1 += 1;
                }

                img_detections
                    .entry(img_id)
                    .or_default()
                    .push((eval_img.dt_scores[d], is_tp));
            }

            // Classify ground truths
            let gt_matched_at_t = &eval_img.gt_matched[t_idx];
            for (g, &gid) in eval_img.gt_ids.iter().enumerate() {
                if gt_status.contains_key(&gid) {
                    continue;
                }
                if g < eval_img.gt_ignore.len() && eval_img.gt_ignore[g] {
                    continue;
                }
                let is_matched = g < gt_matched_at_t.len() && gt_matched_at_t[g];
                if is_matched {
                    gt_status.insert(gid, GtStatus::Matched);
                } else {
                    gt_status.insert(gid, GtStatus::Fn);
                    counts.2 += 1;
                }
            }
        }

        // Compute per-image F1, AP, and error profile
        let mut images: HashMap<u64, ImageSummary> = HashMap::new();

        for (&img_id, &(tp, fp, fn_count)) in &img_counts {
            let denom = 2 * tp + fp + fn_count;
            let f1 = if denom == 0 {
                1.0
            } else {
                (2 * tp) as f64 / denom as f64
            };

            // Sort detections by score descending for AP
            let mut dets = img_detections.remove(&img_id).unwrap_or_default();
            dets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let n_gt = tp + fn_count; // total non-ignored GT for this image
            let ap = compute_image_ap(&dets, n_gt);

            let error_profile = match (fp, fn_count) {
                (0, 0) => ErrorProfile::Perfect,
                (f, n) if f > 2 * n => ErrorProfile::FpHeavy,
                (f, n) if n > 2 * f => ErrorProfile::FnHeavy,
                _ => ErrorProfile::Mixed,
            };

            images.insert(
                img_id,
                ImageSummary {
                    tp,
                    fp,
                    fn_count,
                    f1,
                    ap,
                    error_profile,
                },
            );
        }

        // Label error detection
        let mut label_errors = Vec::new();

        // Group FP detections and FN GTs by image for efficient matching
        struct FpDt {
            dt_id: u64,
            score: f64,
            cat_id: u64,
            bbox: [f64; 4],
        }
        let mut img_fp_dts: HashMap<u64, Vec<FpDt>> = HashMap::new();
        struct FnGt {
            gt_id: u64,
            cat_id: u64,
            bbox: [f64; 4],
        }
        let mut img_fn_gts: HashMap<u64, Vec<FnGt>> = HashMap::new();
        let mut img_all_gts: HashMap<u64, Vec<[f64; 4]>> = HashMap::new(); // all GT bboxes for missing_annotation check

        // Collect FP detections above score threshold
        for (&dt_id, &status) in &dt_status {
            if status != DtStatus::Fp {
                continue;
            }
            if let Some(ann) = self.coco_dt.get_ann(dt_id) {
                if let Some(score) = ann.score {
                    if score < score_thr {
                        continue;
                    }
                    if let Some(bbox) = ann.bbox {
                        img_fp_dts.entry(ann.image_id).or_default().push(FpDt {
                            dt_id,
                            score,
                            cat_id: ann.category_id,
                            bbox,
                        });
                    }
                }
            }
        }

        // Collect all GTs per image (for missing_annotation IoU check)
        // and FN GTs per image (for wrong_label cross-category check)
        for (&gt_id, &status) in &gt_status {
            if let Some(ann) = self.coco_gt.get_ann(gt_id) {
                if let Some(bbox) = ann.bbox {
                    img_all_gts.entry(ann.image_id).or_default().push(bbox);
                    if status == GtStatus::Fn {
                        img_fn_gts.entry(ann.image_id).or_default().push(FnGt {
                            gt_id,
                            cat_id: ann.category_id,
                            bbox,
                        });
                    }
                }
            }
        }

        // For each image with high-confidence FPs, check for label errors
        for (&img_id, fp_dts) in &img_fp_dts {
            let fn_gts = img_fn_gts.get(&img_id);
            let all_gts = img_all_gts.get(&img_id);

            for fp in fp_dts {
                let (dt_id, dt_score, dt_cat, dt_bbox) = (fp.dt_id, fp.score, fp.cat_id, fp.bbox);
                // Check against FN GTs for wrong_label
                let mut best_fn_iou = 0.0f64;
                let mut best_fn_gt: Option<(u64, u64)> = None; // (gt_id, gt_cat_id)

                if let Some(fn_gts) = fn_gts {
                    for fg in fn_gts {
                        if fg.cat_id == dt_cat {
                            continue; // same category — not a label error
                        }
                        let iou = bbox_iou(dt_bbox, fg.bbox);
                        if iou > best_fn_iou {
                            best_fn_iou = iou;
                            best_fn_gt = Some((fg.gt_id, fg.cat_id));
                        }
                    }
                }

                if best_fn_iou >= 0.5 {
                    if let Some((gt_id, gt_cat)) = best_fn_gt {
                        label_errors.push(LabelError {
                            image_id: img_id,
                            dt_id,
                            dt_score,
                            dt_category_id: dt_cat,
                            gt_id: Some(gt_id),
                            gt_category_id: Some(gt_cat),
                            iou: best_fn_iou,
                            error_type: LabelErrorType::WrongLabel,
                        });
                        continue; // don't also flag as missing
                    }
                }

                // Check for missing_annotation: no nearby GT at all
                let max_iou_any_gt = all_gts.map_or(0.0, |gts| {
                    gts.iter()
                        .map(|&gt_bbox| bbox_iou(dt_bbox, gt_bbox))
                        .fold(0.0f64, f64::max)
                });

                if max_iou_any_gt < 0.1 {
                    label_errors.push(LabelError {
                        image_id: img_id,
                        dt_id,
                        dt_score,
                        dt_category_id: dt_cat,
                        gt_id: None,
                        gt_category_id: None,
                        iou: 0.0,
                        error_type: LabelErrorType::MissingAnnotation,
                    });
                }
            }
        }

        // Sort label errors by score descending
        label_errors.sort_by(|a, b| {
            b.dt_score
                .partial_cmp(&a.dt_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.dt_id.cmp(&b.dt_id))
        });

        Ok(ImageDiagnostics {
            annotations: AnnotationIndex {
                dt_status,
                gt_status,
                dt_match: dt_match_map,
                gt_match: gt_match_map,
            },
            images,
            label_errors,
            iou_thr: actual_iou_thr,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coco::COCO;
    use crate::eval::COCOeval;
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
        assert!((bbox_iou(a, a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [50.0, 50.0, 10.0, 10.0];
        assert_eq!(bbox_iou(a, b), 0.0);
    }
}
