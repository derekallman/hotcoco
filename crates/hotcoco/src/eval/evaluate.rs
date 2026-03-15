use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::params::IouType;

use super::types::{EvalImg, EvalImgContext};
use super::{COCOeval, EvalMode};

impl COCOeval {
    /// Populate `params.img_ids` and `params.cat_ids` from the GT dataset if not already set.
    fn resolve_params(&mut self) {
        if self.params.img_ids.is_empty() {
            let mut ids: Vec<u64> = self.coco_gt.dataset.images.iter().map(|i| i.id).collect();
            ids.sort_unstable();
            self.params.img_ids = ids;
        }
        if self.params.cat_ids.is_empty() {
            let mut ids: Vec<u64> = self
                .coco_gt
                .dataset
                .categories
                .iter()
                .map(|c| c.id)
                .collect();
            ids.sort_unstable();
            self.params.cat_ids = ids;
        }
    }

    /// Build the sorted list of (img_id, cat_id) pairs to evaluate.
    ///
    /// Takes the union of non-empty GT and DT pairs, filters to the active img_ids/cat_ids
    /// from params, and returns them sorted for deterministic output order.
    ///
    /// In LVIS mode, DT-only pairs are dropped unless the category appears in `neg_cats`
    /// for that image (i.e. it was confirmed absent and unmatched DTs should count as FP).
    fn collect_sparse_pairs(
        &self,
        cat_ids: &[u64],
        neg_cats: &HashMap<u64, HashSet<u64>>,
    ) -> Vec<(u64, u64)> {
        let allowed_imgs: HashSet<u64> = self.params.img_ids.iter().copied().collect();
        let allowed_cats: HashSet<u64> = cat_ids.iter().copied().collect();

        // At large-scale (e.g. Objects365: 365 cats × 80K imgs = 29M pairs), ~96% of pairs
        // are empty. Driving evaluation from the index instead reduces pairs by ~35x.
        let mut sparse_set: HashSet<(u64, u64)> = HashSet::new();
        if self.params.use_cats {
            // Collect GT pairs first (needed for LVIS DT filtering).
            let mut gt_pairs: HashSet<(u64, u64)> = HashSet::new();
            for pair in self.coco_gt.nonempty_img_cat_pairs() {
                if allowed_imgs.contains(&pair.0) && allowed_cats.contains(&pair.1) {
                    gt_pairs.insert(pair);
                    sparse_set.insert(pair);
                }
            }
            for pair in self.coco_dt.nonempty_img_cat_pairs() {
                if allowed_imgs.contains(&pair.0) && allowed_cats.contains(&pair.1) {
                    if self.eval_mode == EvalMode::Lvis {
                        // Keep DT pair only if GT exists OR cat is explicitly neg for this image.
                        if gt_pairs.contains(&pair)
                            || neg_cats.get(&pair.0).is_some_and(|s| s.contains(&pair.1))
                        {
                            sparse_set.insert(pair);
                        }
                    } else {
                        sparse_set.insert(pair);
                    }
                }
            }
        } else {
            for img_id in self.coco_gt.nonempty_img_ids() {
                if allowed_imgs.contains(&img_id) {
                    sparse_set.insert((img_id, u64::MAX));
                }
            }
            for img_id in self.coco_dt.nonempty_img_ids() {
                if allowed_imgs.contains(&img_id) {
                    sparse_set.insert((img_id, u64::MAX));
                }
            }
        }

        let mut pairs: Vec<(u64, u64)> = sparse_set.into_iter().collect();
        pairs.sort_unstable();
        pairs
    }

    /// Run per-image evaluation.
    pub fn evaluate(&mut self) {
        self.resolve_params();

        let cat_ids = if self.params.use_cats {
            self.params.cat_ids.clone()
        } else {
            vec![u64::MAX] // dummy single category (avoids collision with real category_id=0)
        };

        // LVIS: scan GT image metadata to build per-image category sets.
        // Deferred from construction so the scan only happens when evaluate() is called.
        // neg_cats:       img_id → categories confirmed absent (unmatched DTs count as FP).
        // not_exhaustive: img_id → categories not fully checked (unmatched DTs are ignored).
        let (neg_cats, not_exhaustive) = if self.eval_mode == EvalMode::Lvis {
            let mut neg: HashMap<u64, HashSet<u64>> = HashMap::new();
            let mut not_ex: HashMap<u64, HashSet<u64>> = HashMap::new();
            for img in &self.coco_gt.dataset.images {
                if !img.neg_category_ids.is_empty() {
                    neg.insert(img.id, img.neg_category_ids.iter().copied().collect());
                }
                if !img.not_exhaustive_category_ids.is_empty() {
                    not_ex.insert(
                        img.id,
                        img.not_exhaustive_category_ids.iter().copied().collect(),
                    );
                }
            }
            (neg, not_ex)
        } else {
            (HashMap::new(), HashMap::new())
        };

        // LVIS: build freq_groups now that cat_ids are established.
        if self.eval_mode == EvalMode::Lvis {
            let cat_id_to_k_idx: HashMap<u64, usize> =
                cat_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let mut freq_groups = super::types::FreqGroups::default();
            for cat in &self.coco_gt.dataset.categories {
                if let Some(&k_idx) = cat_id_to_k_idx.get(&cat.id) {
                    match cat.frequency.as_deref() {
                        Some("r") => freq_groups.rare.push(k_idx),
                        Some("c") => freq_groups.common.push(k_idx),
                        Some("f") => freq_groups.frequent.push(k_idx),
                        _ => {}
                    }
                }
            }
            self.freq_groups = freq_groups;
        }

        let sparse_pairs = self.collect_sparse_pairs(&cat_ids, &neg_cats);

        // Compute IoUs only for pairs where both GT and DT are non-empty.
        // Pairs with only GT or only DT produce empty IoU matrices — skip storing them.
        #[allow(clippy::type_complexity)]
        let iou_results: Vec<((u64, u64), Vec<Vec<f64>>)> = sparse_pairs
            .par_iter()
            .filter_map(|&(img_id, cat_id)| {
                let iou_matrix = Self::compute_iou_static(
                    &self.coco_gt,
                    &self.coco_dt,
                    &self.params,
                    img_id,
                    cat_id,
                );
                if iou_matrix.is_empty() {
                    None
                } else {
                    Some(((img_id, cat_id), iou_matrix))
                }
            })
            .collect();

        self.ious.clear();
        self.ious.reserve(iou_results.len());
        for (key, val) in iou_results {
            self.ious.insert(key, val);
        }

        // Evaluate each (image, category, area_range) combination in parallel.
        // sparse_pairs × area_ranges replaces the old cat_ids × area_ranges × img_ids product.
        let max_det = *self.params.max_dets.last().unwrap_or(&100);

        // Build shared context (borrows self after self.ious is fully populated).
        let ctx = EvalImgContext {
            coco_gt: &self.coco_gt,
            coco_dt: &self.coco_dt,
            params: &self.params,
            ious: &self.ious,
        };

        // Tuple: (cat_id, area_rng, img_id, not_exhaustive_cat)
        let mut eval_tuples: Vec<(u64, [f64; 2], u64, bool)> =
            Vec::with_capacity(sparse_pairs.len() * self.params.area_ranges.len());
        for &(img_id, cat_id) in &sparse_pairs {
            let not_exhaustive_cat = self.eval_mode == EvalMode::Lvis
                && not_exhaustive
                    .get(&img_id)
                    .is_some_and(|s| s.contains(&cat_id));
            for ar in &self.params.area_ranges {
                eval_tuples.push((cat_id, ar.range, img_id, not_exhaustive_cat));
            }
        }

        self.eval_imgs = eval_tuples
            .par_iter()
            .map(|&(cat_id, area_rng, img_id, not_exhaustive_cat)| {
                Self::evaluate_img_static(
                    &ctx,
                    img_id,
                    cat_id,
                    area_rng,
                    max_det,
                    not_exhaustive_cat,
                )
            })
            .collect();
    }

    /// Evaluate a single image+category combination.
    ///
    /// `not_exhaustive_cat` — when true (LVIS mode), unmatched detections are
    /// ignored rather than counted as false positives.
    pub(super) fn evaluate_img_static(
        ctx: &EvalImgContext<'_>,
        img_id: u64,
        cat_id: u64,
        area_rng: [f64; 2],
        max_det: usize,
        not_exhaustive_cat: bool,
    ) -> Option<EvalImg> {
        let gt_ids = Self::get_anns_static(ctx.coco_gt, ctx.params, img_id, cat_id);
        let dt_ids = Self::get_anns_static(ctx.coco_dt, ctx.params, img_id, cat_id);

        if gt_ids.is_empty() && dt_ids.is_empty() {
            return None;
        }

        // Load GT annotations and track each annotation's original index in gt_ids,
        // which corresponds to its column in the IoU matrix from compute_iou_static.
        let gt_with_iou_idx: Vec<(usize, &crate::types::Annotation)> = gt_ids
            .iter()
            .enumerate()
            .filter_map(|(iou_idx, &id)| Some((iou_idx, ctx.coco_gt.get_ann(id)?)))
            .collect();
        let (gt_iou_indices, gt_anns): (Vec<usize>, Vec<&crate::types::Annotation>) =
            gt_with_iou_idx.iter().map(|&(idx, ann)| (idx, ann)).unzip();
        let is_kp = ctx.params.iou_type == IouType::Keypoints;
        let gt_ignore: Vec<bool> = gt_anns
            .iter()
            .map(|ann| {
                let a = ann.area.unwrap_or(0.0);
                let mut ignore = ann.iscrowd || a < area_rng[0] || a > area_rng[1];
                // For keypoints, also ignore GT annotations with num_keypoints == 0
                if is_kp {
                    ignore = ignore || ann.num_keypoints.unwrap_or(0) == 0;
                }
                ignore
            })
            .collect();

        // Sort GT: non-ignored first, then ignored
        let mut gt_order: Vec<usize> = (0..gt_anns.len()).collect();
        gt_order.sort_by_key(|&i| gt_ignore[i] as u8);
        let gt_ignore_sorted: Vec<bool> = gt_order.iter().map(|&i| gt_ignore[i]).collect();
        let gt_iscrowd_sorted: Vec<bool> = gt_order.iter().map(|&i| gt_anns[i].iscrowd).collect();
        let num_gt_not_ignored = gt_ignore_sorted.iter().filter(|&&x| !x).count();

        // Load DT annotations with their original IoU matrix row indices,
        // sort by score descending, then limit to max_det.
        let mut dt_with_iou_idx: Vec<(usize, &crate::types::Annotation)> = dt_ids
            .iter()
            .enumerate()
            .filter_map(|(iou_idx, &id)| Some((iou_idx, ctx.coco_dt.get_ann(id)?)))
            .collect();
        dt_with_iou_idx.sort_by(|a, b| {
            b.1.score
                .unwrap_or(0.0)
                .partial_cmp(&a.1.score.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if dt_with_iou_idx.len() > max_det {
            dt_with_iou_idx.truncate(max_det);
        }
        let (dt_iou_indices, dt_anns): (Vec<usize>, Vec<&crate::types::Annotation>) =
            dt_with_iou_idx.iter().map(|&(idx, ann)| (idx, ann)).unzip();
        // Extract scores and area-ignore flags in one pass over the sorted DT list.
        let (dt_scores, dt_area_ignore): (Vec<f64>, Vec<bool>) = dt_anns
            .iter()
            .map(|ann| {
                let a = ann.area.unwrap_or(0.0);
                (ann.score.unwrap_or(0.0), a < area_rng[0] || a > area_rng[1])
            })
            .unzip();

        // Get IoU matrix
        let iou_matrix = ctx.ious.get(&(img_id, cat_id));

        let num_iou_thrs = ctx.params.iou_thrs.len();
        let d = dt_anns.len();
        let g = gt_anns.len();

        let mut dt_matches = vec![vec![0u64; d]; num_iou_thrs];
        let mut gt_matches = vec![vec![0u64; g]; num_iou_thrs];
        let mut dt_matched = vec![vec![false; d]; num_iou_thrs];
        let mut gt_matched = vec![vec![false; g]; num_iou_thrs];
        // Initialize from dt_area_ignore so unmatched DTs have the correct ignore status
        // even when there are no GT annotations (iou_matrix is None).
        let mut dt_ignore_flags: Vec<Vec<bool>> =
            (0..num_iou_thrs).map(|_| dt_area_ignore.clone()).collect();

        if let Some(iou_mat) = iou_matrix {
            // Build a flat D×G IoU matrix in row-major order (one allocation).
            // dt_iou_indices[di] is the row in iou_mat for the di-th detection (score-sorted).
            // gt_iou_indices[gt_order[gi]] is the column for the gi-th GT (ignore-sorted).
            let mut iou_flat = vec![0.0_f64; d * g];
            for di in 0..d {
                let dt_row = dt_iou_indices[di];
                for (gi_sorted, &gi_orig) in gt_order.iter().enumerate() {
                    let gt_col = gt_iou_indices[gi_orig];
                    if dt_row < iou_mat.len() && gt_col < iou_mat[dt_row].len() {
                        iou_flat[di * g + gi_sorted] = iou_mat[dt_row][gt_col];
                    }
                }
            }

            // Greedy matching: for each IoU threshold, iterate detections in
            // score-descending order and greedily match each to the best available GT.
            //
            // Two-phase matching (matches pycocotools exactly):
            //   Phase 1: linear scan over non-ignored GTs for the highest-IoU match.
            //   Phase 2: only if phase 1 found no match, linear scan over ignored GTs.
            // For typical COCO images (g ≤ 5 GTs/cat), a linear scan over g elements
            // avoids pre-sorting d index vectors and eliminates d×2 Vec allocations.
            for (t_idx, &iou_thr) in ctx.params.iou_thrs.iter().enumerate() {
                for (di, dt_ann) in dt_anns.iter().enumerate() {
                    let mut best_iou = iou_thr;
                    let mut best_gi: Option<usize> = None;
                    let base = di * g;

                    // Phase 1: non-ignored GTs — linear scan for highest-IoU available match.
                    for gi in 0..num_gt_not_ignored {
                        if gt_matched[t_idx][gi] && !gt_iscrowd_sorted[gi] {
                            continue;
                        }
                        let iou_val = iou_flat[base + gi];
                        if iou_val >= best_iou {
                            best_iou = iou_val;
                            best_gi = Some(gi);
                        }
                    }

                    // Phase 2: ignored GTs — only if no non-ignored match found.
                    // This matches pycocotools' `if m>-1 and gtIg[m]==0: break`
                    // which stops at the first ignored GT when a non-ignored match exists.
                    if best_gi.is_none() {
                        for gi in num_gt_not_ignored..g {
                            if gt_matched[t_idx][gi] && !gt_iscrowd_sorted[gi] {
                                continue;
                            }
                            let iou_val = iou_flat[base + gi];
                            if iou_val >= best_iou {
                                best_iou = iou_val;
                                best_gi = Some(gi);
                            }
                        }
                    }

                    if let Some(gi) = best_gi {
                        dt_matches[t_idx][di] = gt_anns[gt_order[gi]].id;
                        gt_matches[t_idx][gi] = dt_ann.id;
                        dt_matched[t_idx][di] = true;
                        gt_matched[t_idx][gi] = true;

                        // DT is ignored if matched to ignored GT
                        dt_ignore_flags[t_idx][di] = gt_ignore_sorted[gi];
                    }
                    // Unmatched: dt_ignore_flags[t_idx][di] already set from dt_area_ignore
                }
            }
        }

        // LVIS: for not_exhaustive categories, mark all unmatched DTs as ignored
        // so they don't count as false positives.
        if not_exhaustive_cat {
            for t_idx in 0..num_iou_thrs {
                for di in 0..d {
                    if !dt_matched[t_idx][di] {
                        dt_ignore_flags[t_idx][di] = true;
                    }
                }
            }
        }

        // If there are no non-ignored GTs and no non-ignored DTs, skip
        let has_content = num_gt_not_ignored > 0
            || dt_anns
                .iter()
                .enumerate()
                .any(|(di, _)| !dt_area_ignore[di]);
        if !has_content && gt_ids.is_empty() {
            return None;
        }

        Some(EvalImg {
            image_id: img_id,
            category_id: cat_id,
            area_rng,
            max_det,
            dt_ids: dt_anns.iter().map(|a| a.id).collect(),
            gt_ids: gt_order.iter().map(|&i| gt_anns[i].id).collect(),
            dt_matches,
            gt_matches,
            dt_matched,
            gt_matched,
            dt_scores,
            gt_ignore: gt_ignore_sorted,
            dt_ignore: dt_ignore_flags,
        })
    }
}
