use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use super::types::{EvalImg, TideErrors};
use super::COCOeval;

impl COCOeval {
    /// Compute average precision from per-detection matched/ignored flags.
    ///
    /// Uses the same 101-point interpolation as [`accumulate`](COCOeval::accumulate).
    /// Returns `0.0` when `num_gt == 0` or there are no detections.
    pub(super) fn compute_ap_from_matched(
        scores: &[f64],
        matched: &[bool],
        ignored: &[bool],
        num_gt: usize,
        rec_thrs: &[f64],
    ) -> f64 {
        if num_gt == 0 {
            return 0.0;
        }
        let nd = scores.len();
        if nd == 0 {
            return 0.0;
        }

        let mut inds: Vec<usize> = (0..nd).collect();
        inds.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut tp = vec![0.0f64; nd];
        let mut fp = vec![0.0f64; nd];
        for (out_idx, &src_idx) in inds.iter().enumerate() {
            if !ignored[src_idx] {
                if matched[src_idx] {
                    tp[out_idx] = 1.0;
                } else {
                    fp[out_idx] = 1.0;
                }
            }
        }

        for d in 1..nd {
            tp[d] += tp[d - 1];
            fp[d] += fp[d - 1];
        }

        let (_, curve) = super::accumulate::precision_recall_curve(&tp, &fp, num_gt, rec_thrs);
        curve.iter().map(|(_, pr, _)| pr).sum::<f64>() / 101.0
    }

    /// Decompose detection errors into TIDE error types.
    ///
    /// Requires [`evaluate`](COCOeval::evaluate) to have been called first.
    ///
    /// Returns a [`TideErrors`] with ΔAP values and counts for six error types:
    ///
    /// | Error | Meaning |
    /// |-------|---------|
    /// | `Cls`  | Wrong class, correct location (IoU ≥ `pos_thr` with other-class GT) |
    /// | `Loc`  | Right class, poor localization (`bg_thr` ≤ IoU < `pos_thr`) |
    /// | `Both` | Wrong class AND poor localization |
    /// | `Dupe` | Duplicate — correct class GT already claimed by higher-scoring TP |
    /// | `Bkg`  | Pure background (IoU < `bg_thr` with all GTs) |
    /// | `Miss` | Undetected GT (false negative) |
    pub fn tide_errors(&self, pos_thr: f64, bg_thr: f64) -> crate::error::Result<TideErrors> {
        if self.eval_imgs.is_empty() {
            return Err("tide_errors() requires evaluate() to be called first".into());
        }

        let cat_ids = &self.params.cat_ids;
        let iou_type = self.params.iou_type;
        let target_area_rng = self
            .params
            .area_range_idx("all")
            .map_or(self.params.area_ranges[0].range, |idx| {
                self.params.area_ranges[idx].range
            });
        let max_det = *self.params.max_dets.last().unwrap_or(&100);

        // Find t_idx for pos_thr (nearest threshold in params.iou_thrs)
        let t_idx = self
            .params
            .iou_thrs
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                (a - pos_thr)
                    .abs()
                    .partial_cmp(&(b - pos_thr).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i);

        let coco_gt = &self.coco_gt;
        let coco_dt = &self.coco_dt;

        // --- Cross-category IoU pass ---
        // For each image, compute max IoU between each DT annotation
        // and any GT annotation of a *different* category.
        let img_ids = &self.params.img_ids;

        // Returns: img_id → (dt_ann_id → (max_cross_iou, argmax_cross_gt_ann_id))
        // argmax_cross_gt_ann_id is u64::MAX when there are no cross-class GTs.
        let cross_iou_per_img: HashMap<u64, HashMap<u64, (f64, u64)>> = img_ids
            .par_iter()
            .map(|&img_id| {
                let mut dt_max_cross: HashMap<u64, (f64, u64)> = HashMap::new();

                // Collect all non-crowd GTs (cat_idx, ann_id)
                let gt_pairs: Vec<(usize, u64)> = cat_ids
                    .iter()
                    .enumerate()
                    .flat_map(|(cat_idx, &cat_id)| {
                        coco_gt
                            .get_ann_ids_for_img_cat(img_id, cat_id)
                            .iter()
                            .filter_map(move |&ann_id| {
                                let ann = coco_gt.get_ann(ann_id)?;
                                if ann.iscrowd {
                                    return None;
                                }
                                Some((cat_idx, ann_id))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();

                // Collect all DTs (cat_idx, ann_id)
                let dt_pairs: Vec<(usize, u64)> = cat_ids
                    .iter()
                    .enumerate()
                    .flat_map(|(cat_idx, &cat_id)| {
                        coco_dt
                            .get_ann_ids_for_img_cat(img_id, cat_id)
                            .iter()
                            .map(move |&ann_id| (cat_idx, ann_id))
                            .collect::<Vec<_>>()
                    })
                    .collect();

                if dt_pairs.is_empty() || gt_pairs.is_empty() {
                    for &(_, ann_id) in &dt_pairs {
                        dt_max_cross.insert(ann_id, (0.0, u64::MAX));
                    }
                    return (img_id, dt_max_cross);
                }

                // Compute cross-category IoU matrix [D × G]
                let dt_ids: Vec<u64> = dt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let gt_ids: Vec<u64> = gt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let iou_matrix =
                    Self::cross_category_iou(&dt_ids, &gt_ids, coco_dt, coco_gt, iou_type);

                // For each DT, find max IoU with any *other-category* GT and record that GT's id
                for (di, &(dt_cat_idx, dt_ann_id)) in dt_pairs.iter().enumerate() {
                    let mut max_cross = 0.0f64;
                    let mut argmax_cross_gt_ann_id = u64::MAX;
                    for (gi, &(gt_cat_idx, gt_ann_id)) in gt_pairs.iter().enumerate() {
                        if gt_cat_idx != dt_cat_idx && di < iou_matrix.len() {
                            let iou = iou_matrix[di][gi];
                            if iou > max_cross {
                                max_cross = iou;
                                argmax_cross_gt_ann_id = gt_ann_id;
                            }
                        }
                    }
                    dt_max_cross.insert(dt_ann_id, (max_cross, argmax_cross_gt_ann_id));
                }

                (img_id, dt_max_cross)
            })
            .collect();

        // --- Error type definition (local enum) ---
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum ErrType {
            Cls,
            Loc,
            Both,
            Dupe,
            Bkg,
        }

        // Per-category accumulated data for ΔAP computation
        struct CatData {
            scores: Vec<f64>,
            matched: Vec<bool>,
            ignored: Vec<bool>,
            // Error type for each FP DT (None = TP or ignored)
            fp_types: Vec<Option<ErrType>>,
            num_gt: usize,
        }

        let mut cat_data: HashMap<u64, CatData> = HashMap::new();
        let mut counts: HashMap<String, u64> = [
            ("Cls", 0u64),
            ("Loc", 0u64),
            ("Both", 0u64),
            ("Dupe", 0u64),
            ("Bkg", 0u64),
            ("Miss", 0u64),
        ]
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect();

        // GTs that have a Loc or Cls FP DT "targeting" them — these are not Miss errors.
        // A Loc DT targets the same-class GT with highest IoU in [bg_thr, pos_thr).
        // A Cls DT targets the cross-class GT with highest IoU >= pos_thr.
        // Collected across all categories so cross-category Cls coverage is captured.
        let mut covered_gt_ann_ids: HashSet<u64> = HashSet::new();

        // Pre-filter once; both passes below use the same (area_rng, max_det) predicate.
        let matching_eval_imgs: Vec<&EvalImg> = self
            .eval_imgs
            .iter()
            .flatten()
            .filter(|e| e.area_rng == target_area_rng && e.max_det == max_det)
            .collect();

        // --- Process each eval_img at (target_area_rng, max_det) ---
        for eval_img in &matching_eval_imgs {
            let img_id = eval_img.image_id;
            let cat_id = eval_img.category_id;
            let d = eval_img.dt_ids.len();
            let g = eval_img.gt_ids.len();

            // Build index: dt annotation ID → original position in coco_dt
            let dt_orig_ids = coco_dt.get_ann_ids_for_img_cat(img_id, cat_id);
            let gt_orig_ids = coco_gt.get_ann_ids_for_img_cat(img_id, cat_id);
            let dt_id_to_orig: HashMap<u64, usize> = dt_orig_ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect();
            let gt_id_to_orig: HashMap<u64, usize> = gt_orig_ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect();

            let same_iou_mat = self.ious.get(&(img_id, cat_id));
            let cross_map = cross_iou_per_img.get(&img_id);

            let entry = cat_data.entry(cat_id).or_insert_with(|| CatData {
                scores: Vec::new(),
                matched: Vec::new(),
                ignored: Vec::new(),
                fp_types: Vec::new(),
                num_gt: 0,
            });

            // Accumulate non-ignored GT count
            entry.num_gt += eval_img.gt_ignore.iter().filter(|&&x| !x).count();

            // Classify each DT
            for di in 0..d {
                let dt_ann_id = eval_img.dt_ids[di];
                let is_matched = eval_img.dt_matched[t_idx][di];
                let is_ignored = eval_img.dt_ignore[t_idx][di];

                let fp_type = if is_matched || is_ignored {
                    None
                } else {
                    // FP: classify by priority (matches tidecv: Loc > Cls > Dupe > Bkg > Both)
                    let (max_cross_iou, argmax_cross_gt_ann_id) = cross_map
                        .and_then(|m| m.get(&dt_ann_id))
                        .copied()
                        .unwrap_or((0.0, u64::MAX));

                    // Same-class IoU: max IoU to any same-class GT; track argmax GT and Dupe
                    let mut max_same_iou = 0.0f64;
                    let mut argmax_same_gt_ann_id = u64::MAX;
                    let mut best_same_gt_matched = false;
                    if let Some(iou_mat) = same_iou_mat {
                        if let Some(&di_orig) = dt_id_to_orig.get(&dt_ann_id) {
                            for gi_sorted in 0..g {
                                let gt_ann_id = eval_img.gt_ids[gi_sorted];
                                if let Some(&gi_orig) = gt_id_to_orig.get(&gt_ann_id) {
                                    let iou = if di_orig < iou_mat.len()
                                        && gi_orig < iou_mat[di_orig].len()
                                    {
                                        iou_mat[di_orig][gi_orig]
                                    } else {
                                        0.0
                                    };
                                    if iou > max_same_iou {
                                        max_same_iou = iou;
                                        argmax_same_gt_ann_id = gt_ann_id;
                                    }
                                    if iou >= pos_thr && eval_img.gt_matched[t_idx][gi_sorted] {
                                        best_same_gt_matched = true;
                                    }
                                }
                            }
                        }
                    }

                    // Priority order matches tidecv (BoxError > ClassError > DuplicateError > BackgroundError > OtherError):
                    //   Loc:  same-class max IoU ∈ [bg_thr, pos_thr]; upper bound excludes Dupe
                    //         (Dupe DTs have same-class IoU > pos_thr with an already-matched GT)
                    //   Cls:  cross-class max IoU ≥ pos_thr (Loc didn't fire)
                    //   Dupe: a same-class GT with IoU ≥ pos_thr is already matched by a higher TP
                    //   Bkg:  max IoU with any GT ≤ bg_thr (same-class < bg_thr already; check cross)
                    //   Both: fallthrough (cross-class IoU ∈ (bg_thr, pos_thr))
                    let err = if max_same_iou >= bg_thr && max_same_iou <= pos_thr {
                        ErrType::Loc
                    } else if max_cross_iou >= pos_thr {
                        ErrType::Cls
                    } else if best_same_gt_matched {
                        ErrType::Dupe
                    } else if max_cross_iou <= bg_thr {
                        ErrType::Bkg
                    } else {
                        ErrType::Both
                    };

                    // Track which GTs are "covered" (not Miss) by Loc or Cls FP DTs.
                    // Only Loc and Cls errors can be fixed to produce a TP for their target GT;
                    // Bkg/Both/Dupe fixes suppress the DT rather than turning it into a TP.
                    match err {
                        ErrType::Loc => {
                            if argmax_same_gt_ann_id != u64::MAX {
                                covered_gt_ann_ids.insert(argmax_same_gt_ann_id);
                            }
                        }
                        ErrType::Cls => {
                            if argmax_cross_gt_ann_id != u64::MAX {
                                covered_gt_ann_ids.insert(argmax_cross_gt_ann_id);
                            }
                        }
                        _ => {}
                    }

                    Some(err)
                };

                entry.scores.push(eval_img.dt_scores[di]);
                entry.matched.push(is_matched);
                entry.ignored.push(is_ignored);
                entry.fp_types.push(fp_type);
            }
        }

        // Aggregate FP error type counts
        for data in cat_data.values() {
            for err in data.fp_types.iter().flatten() {
                let key = match err {
                    ErrType::Cls => "Cls",
                    ErrType::Loc => "Loc",
                    ErrType::Both => "Both",
                    ErrType::Dupe => "Dupe",
                    ErrType::Bkg => "Bkg",
                };
                *counts.entry(key.to_string()).or_insert(0) += 1;
            }
        }

        // --- Count Miss errors (second pass, after all FP types are classified) ---
        // A GT is Miss only if it is unmatched, non-ignored, AND not covered by any Loc/Cls FP DT.
        // Cross-category Cls coverage requires the second pass (a dog DT may cover a cat GT).
        let mut cat_miss_counts: HashMap<u64, usize> = HashMap::new();
        for eval_img in &matching_eval_imgs {
            let g = eval_img.gt_ids.len();
            let n = (0..g)
                .filter(|&gi| {
                    !eval_img.gt_matched[t_idx][gi]
                        && !eval_img.gt_ignore[gi]
                        && !covered_gt_ann_ids.contains(&eval_img.gt_ids[gi])
                })
                .count();
            *counts.entry("Miss".to_string()).or_insert(0) += n as u64;
            *cat_miss_counts.entry(eval_img.category_id).or_insert(0) += n;
        }

        // --- ΔAP computation ---
        let mut baseline_aps: Vec<f64> = Vec::new();
        let mut d_cls: Vec<f64> = Vec::new();
        let mut d_loc: Vec<f64> = Vec::new();
        let mut d_both: Vec<f64> = Vec::new();
        let mut d_dupe: Vec<f64> = Vec::new();
        let mut d_bkg: Vec<f64> = Vec::new();
        let mut d_miss: Vec<f64> = Vec::new();
        let mut d_fp: Vec<f64> = Vec::new();

        let rec_thrs = &self.params.rec_thrs;

        for &cat_id in cat_ids {
            let data = match cat_data.get(&cat_id) {
                Some(d) if d.num_gt > 0 => d,
                _ => continue,
            };

            let baseline = Self::compute_ap_from_matched(
                &data.scores,
                &data.matched,
                &data.ignored,
                data.num_gt,
                rec_thrs,
            );
            baseline_aps.push(baseline);

            // Fix a set of FP error types.
            // Cls and Loc: flip FP → TP (the DT would have been correct if the error were fixed).
            // Bkg, Both, Dupe: suppress the DT (set ignored=true), matching tidecv's fix()→None
            // behaviour where these errors produce no corrected TP.
            let fix_fp = |fix_types: &[ErrType]| -> f64 {
                let mut fixed_matched = data.matched.clone();
                let mut fixed_ignored = data.ignored.clone();
                for (i, fp_type) in data.fp_types.iter().enumerate() {
                    if let Some(err) = fp_type {
                        if fix_types.contains(err) {
                            match err {
                                ErrType::Cls | ErrType::Loc => {
                                    fixed_matched[i] = true;
                                }
                                ErrType::Bkg | ErrType::Both | ErrType::Dupe => {
                                    fixed_ignored[i] = true;
                                }
                            }
                        }
                    }
                }
                Self::compute_ap_from_matched(
                    &data.scores,
                    &fixed_matched,
                    &fixed_ignored,
                    data.num_gt,
                    rec_thrs,
                )
            };

            d_cls.push(fix_fp(&[ErrType::Cls]) - baseline);
            d_loc.push(fix_fp(&[ErrType::Loc]) - baseline);
            d_both.push(fix_fp(&[ErrType::Both]) - baseline);
            d_dupe.push(fix_fp(&[ErrType::Dupe]) - baseline);
            d_bkg.push(fix_fp(&[ErrType::Bkg]) - baseline);
            d_fp.push(
                fix_fp(&[
                    ErrType::Cls,
                    ErrType::Loc,
                    ErrType::Both,
                    ErrType::Dupe,
                    ErrType::Bkg,
                ]) - baseline,
            );

            // Fix Miss: inject fake TPs for unmatched GTs
            let miss_count = cat_miss_counts.get(&cat_id).copied().unwrap_or(0);
            let miss_delta = if miss_count > 0 {
                let mut fixed_scores = Vec::with_capacity(data.scores.len() + miss_count);
                let mut fixed_matched = Vec::with_capacity(data.matched.len() + miss_count);
                let mut fixed_ignored = Vec::with_capacity(data.ignored.len() + miss_count);
                for _ in 0..miss_count {
                    fixed_scores.push(2.0);
                    fixed_matched.push(true);
                    fixed_ignored.push(false);
                }
                fixed_scores.extend_from_slice(&data.scores);
                fixed_matched.extend_from_slice(&data.matched);
                fixed_ignored.extend_from_slice(&data.ignored);
                Self::compute_ap_from_matched(
                    &fixed_scores,
                    &fixed_matched,
                    &fixed_ignored,
                    data.num_gt,
                    rec_thrs,
                ) - baseline
            } else {
                0.0
            };
            d_miss.push(miss_delta);
        }

        let mean_ap = |v: &[f64]| -> f64 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };

        let ap_base = mean_ap(&baseline_aps);
        let miss_mean = mean_ap(&d_miss);

        let mut delta_ap: HashMap<String, f64> = HashMap::new();
        delta_ap.insert("Cls".to_string(), mean_ap(&d_cls));
        delta_ap.insert("Loc".to_string(), mean_ap(&d_loc));
        delta_ap.insert("Both".to_string(), mean_ap(&d_both));
        delta_ap.insert("Dupe".to_string(), mean_ap(&d_dupe));
        delta_ap.insert("Bkg".to_string(), mean_ap(&d_bkg));
        delta_ap.insert("Miss".to_string(), miss_mean);
        delta_ap.insert("FP".to_string(), mean_ap(&d_fp));
        delta_ap.insert("FN".to_string(), miss_mean);

        Ok(TideErrors {
            delta_ap,
            counts,
            ap_base,
            pos_thr,
            bg_thr,
        })
    }
}
