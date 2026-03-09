use rayon::prelude::*;

use crate::coco::COCO;
use crate::mask;
use crate::params::IouType;
use crate::types::Rle;

use super::types::ConfusionMatrix;
use super::COCOeval;

impl COCOeval {
    /// Compute a cross-category IoU matrix between DT and GT annotations.
    ///
    /// Returns `Vec<Vec<f64>>` of shape `[D × G]`. Falls back to bbox IoU for segm mode
    /// when RLEs cannot be produced for all annotations.
    pub(super) fn cross_category_iou(
        dt_ann_ids: &[u64],
        gt_ann_ids: &[u64],
        coco_dt: &COCO,
        coco_gt: &COCO,
        iou_type: IouType,
    ) -> Vec<Vec<f64>> {
        let d = dt_ann_ids.len();
        let g = gt_ann_ids.len();
        if d == 0 || g == 0 {
            return vec![];
        }

        match iou_type {
            IouType::Bbox | IouType::Keypoints => {
                let dt_bbs: Vec<[f64; 4]> = dt_ann_ids
                    .iter()
                    .filter_map(|&id| coco_dt.get_ann(id)?.bbox)
                    .collect();
                let gt_bbs: Vec<[f64; 4]> = gt_ann_ids
                    .iter()
                    .filter_map(|&id| coco_gt.get_ann(id)?.bbox)
                    .collect();
                if dt_bbs.len() == d && gt_bbs.len() == g {
                    let iscrowd = vec![false; g];
                    mask::bbox_iou(&dt_bbs, &gt_bbs, &iscrowd)
                } else {
                    vec![vec![0.0; g]; d]
                }
            }
            IouType::Segm => {
                let dt_rles: Vec<Option<Rle>> = dt_ann_ids
                    .iter()
                    .map(|&id| coco_dt.get_ann(id).and_then(|a| coco_dt.ann_to_rle(a)))
                    .collect();
                let gt_rles: Vec<Option<Rle>> = gt_ann_ids
                    .iter()
                    .map(|&id| coco_gt.get_ann(id).and_then(|a| coco_gt.ann_to_rle(a)))
                    .collect();

                if dt_rles.iter().all(|r| r.is_some()) && gt_rles.iter().all(|r| r.is_some()) {
                    let dt_r: Vec<Rle> = dt_rles.into_iter().map(|r| r.unwrap()).collect();
                    let gt_r: Vec<Rle> = gt_rles.into_iter().map(|r| r.unwrap()).collect();
                    let iscrowd = vec![false; g];
                    mask::iou(&dt_r, &gt_r, &iscrowd)
                } else {
                    // Bbox fallback when any RLE is missing
                    Self::compute_bbox_iou_static(coco_gt, coco_dt, dt_ann_ids, gt_ann_ids)
                }
            }
        }
    }

    /// Compute a per-category confusion matrix across all images.
    ///
    /// Unlike `evaluate()`, this method compares **all** detections in an image against
    /// **all** ground truth boxes regardless of category. This enables cross-category
    /// confusion analysis ("the model keeps predicting `dog` on `cat` ground truth").
    ///
    /// This is a `&self` method — it does not call `evaluate()` and does not mutate state.
    /// It can be called standalone at any point after constructing `COCOeval`.
    ///
    /// # Matrix layout (rows = GT, cols = predicted)
    ///
    /// - `matrix[gt_cat_idx][dt_cat_idx]` — matched pair (true positive if same category)
    /// - `matrix[gt_cat_idx][num_cats]` — unmatched GT (false negative / missed detection)
    /// - `matrix[num_cats][dt_cat_idx]` — unmatched DT (false positive / spurious detection)
    ///
    /// # Arguments
    ///
    /// - `iou_thr` — IoU threshold for a DT↔GT match (default 0.5)
    /// - `max_det` — max detections per image after score sorting; `None` uses the last
    ///   value of `params.max_dets`
    /// - `min_score` — discard DTs below this confidence before the `max_det` truncation;
    ///   `None` keeps all detections
    pub fn confusion_matrix(
        &self,
        iou_thr: f64,
        max_det: Option<usize>,
        min_score: Option<f64>,
    ) -> ConfusionMatrix {
        // Resolve cat_ids / img_ids: respect user-set params filters but do not mutate.
        let cat_ids: Vec<u64> = if !self.params.cat_ids.is_empty() {
            self.params.cat_ids.clone()
        } else {
            let mut ids: Vec<u64> = self
                .coco_gt
                .dataset
                .categories
                .iter()
                .map(|c| c.id)
                .collect();
            ids.sort_unstable();
            ids
        };

        let img_ids: Vec<u64> = if !self.params.img_ids.is_empty() {
            self.params.img_ids.clone()
        } else {
            let mut ids: Vec<u64> = self.coco_gt.dataset.images.iter().map(|i| i.id).collect();
            ids.sort_unstable();
            ids
        };

        let num_cats = cat_ids.len();
        let k = num_cats + 1; // background index = num_cats
        let eff_max_det = max_det.unwrap_or_else(|| *self.params.max_dets.last().unwrap_or(&100));
        let iou_type = self.params.iou_type;

        let coco_gt = &self.coco_gt;
        let coco_dt = &self.coco_dt;

        // Compute a (k×k) local matrix for each image in parallel, then sum.
        let matrices: Vec<Vec<u64>> = img_ids
            .par_iter()
            .map(|&img_id| {
                let mut local = vec![0u64; k * k];

                // --- Collect non-crowd GTs: (cat_idx, ann_id) ---
                let gt_pairs: Vec<(usize, u64)> = cat_ids
                    .iter()
                    .enumerate()
                    .flat_map(|(cat_idx, &cat_id)| {
                        coco_gt
                            .get_ann_ids_for_img_cat(img_id, cat_id)
                            .iter()
                            .copied()
                            .filter_map(move |ann_id| {
                                let ann = coco_gt.get_ann(ann_id)?;
                                if ann.iscrowd {
                                    return None;
                                }
                                Some((cat_idx, ann_id))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();

                // --- Collect DTs: (cat_idx, score, ann_id), apply min_score ---
                let mut dt_pairs: Vec<(usize, f64, u64)> = cat_ids
                    .iter()
                    .enumerate()
                    .flat_map(|(cat_idx, &cat_id)| {
                        coco_dt
                            .get_ann_ids_for_img_cat(img_id, cat_id)
                            .iter()
                            .copied()
                            .filter_map(move |ann_id| {
                                let ann = coco_dt.get_ann(ann_id)?;
                                let score = ann.score.unwrap_or(0.0);
                                if min_score.is_some_and(|ms| score < ms) {
                                    return None;
                                }
                                Some((cat_idx, score, ann_id))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();

                // Sort DTs by score descending, then truncate to max_det.
                dt_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if dt_pairs.len() > eff_max_det {
                    dt_pairs.truncate(eff_max_det);
                }

                if gt_pairs.is_empty() && dt_pairs.is_empty() {
                    return local;
                }

                let d = dt_pairs.len();
                let g = gt_pairs.len();

                // --- Compute cross-category IoU matrix [D × G] ---
                let dt_ids: Vec<u64> = dt_pairs.iter().map(|&(_, _, ann_id)| ann_id).collect();
                let gt_ids: Vec<u64> = gt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let iou_matrix =
                    Self::cross_category_iou(&dt_ids, &gt_ids, coco_dt, coco_gt, iou_type);

                // --- Greedy matching at iou_thr (DTs already in score-sorted order) ---
                let mut gt_matched = vec![false; g];

                for di in 0..d {
                    let mut best_iou = iou_thr;
                    let mut best_gi: Option<usize> = None;

                    if !iou_matrix.is_empty() {
                        let row = &iou_matrix[di];
                        for (gi, (&is_matched, &iou)) in
                            gt_matched.iter().zip(row.iter()).enumerate()
                        {
                            if is_matched {
                                continue;
                            }
                            if iou >= best_iou {
                                best_iou = iou;
                                best_gi = Some(gi);
                            }
                        }
                    }

                    if let Some(gi) = best_gi {
                        gt_matched[gi] = true;
                        let dt_cat_idx = dt_pairs[di].0;
                        let gt_cat_idx = gt_pairs[gi].0;
                        local[gt_cat_idx * k + dt_cat_idx] += 1;
                    } else {
                        // Unmatched DT → false positive (background row)
                        let dt_cat_idx = dt_pairs[di].0;
                        local[num_cats * k + dt_cat_idx] += 1;
                    }
                }

                // Unmatched GTs → false negatives (background column)
                for (is_matched, &(gt_cat_idx, _)) in gt_matched.iter().zip(gt_pairs.iter()) {
                    if !is_matched {
                        local[gt_cat_idx * k + num_cats] += 1;
                    }
                }

                local
            })
            .collect();

        // Reduce: element-wise sum of per-image matrices.
        let mut matrix = vec![0u64; k * k];
        for local in matrices {
            for (i, &v) in local.iter().enumerate() {
                matrix[i] += v;
            }
        }

        ConfusionMatrix {
            matrix,
            num_cats,
            cat_ids,
            iou_thr,
        }
    }
}
