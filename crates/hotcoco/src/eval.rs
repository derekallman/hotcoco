//! COCO evaluation engine — faithful port of `pycocotools/cocoeval.py`.
//!
//! Implements evaluate, accumulate, and summarize for bbox, segm, and keypoint evaluation.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::coco::COCO;
use crate::mask;
use crate::params::{IouType, Params};
use crate::types::Rle;

/// Per-image, per-category evaluation result.
#[derive(Debug, Clone)]
pub struct EvalImg {
    pub image_id: u64,
    pub category_id: u64,
    pub area_rng: [f64; 2],
    pub max_det: usize,
    /// Detection annotation IDs (sorted by score descending, truncated to max_det)
    pub dt_ids: Vec<u64>,
    /// Ground truth annotation IDs (sorted: non-ignored first, then ignored)
    pub gt_ids: Vec<u64>,
    /// Detection matches for each IoU threshold: dt_matches[t][d] = matched gt_id or 0
    pub dt_matches: Vec<Vec<u64>>,
    /// Ground truth matches for each IoU threshold: gt_matches[t][g] = matched dt_id or 0
    pub gt_matches: Vec<Vec<u64>>,
    /// Whether each detection is matched per IoU threshold (reliable for id=0)
    pub dt_matched: Vec<Vec<bool>>,
    /// Whether each GT is matched per IoU threshold (reliable for id=0)
    pub gt_matched: Vec<Vec<bool>>,
    /// Detection scores
    pub dt_scores: Vec<f64>,
    /// Whether each GT is ignored
    pub gt_ignore: Vec<bool>,
    /// Whether each detection is ignored per IoU threshold
    pub dt_ignore: Vec<Vec<bool>>,
}

/// Accumulated evaluation results across all images.
///
/// Precision and scores are stored as flat 5-D arrays with shape `[T x R x K x A x M]`.
/// Recall is a flat 4-D array with shape `[T x K x A x M]`. Values of -1.0 indicate
/// that no data was available for that combination (e.g. a category with no GT instances).
#[derive(Debug, Clone)]
pub struct AccumulatedEval {
    /// Interpolated precision at each (iou_thr, recall_thr, category, area_range, max_det).
    pub precision: Vec<f64>,
    /// Maximum recall at each (iou_thr, category, area_range, max_det).
    pub recall: Vec<f64>,
    /// Detection score at each precision threshold, same shape as `precision`.
    pub scores: Vec<f64>,
    /// Number of IoU thresholds (T dimension).
    pub t: usize,
    /// Number of recall thresholds (R dimension).
    pub r: usize,
    /// Number of categories (K dimension).
    pub k: usize,
    /// Number of area ranges (A dimension).
    pub a: usize,
    /// Number of max-detection limits (M dimension).
    pub m: usize,
}

impl AccumulatedEval {
    /// Compute the flat index into `precision` (or `scores`) for the given 5-D coordinates.
    pub fn precision_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        ((((t * self.r + r) * self.k + k) * self.a + a) * self.m) + m
    }

    /// Compute the flat index into `recall` for the given 4-D coordinates.
    pub fn recall_idx(&self, t: usize, k: usize, a: usize, m: usize) -> usize {
        (((t * self.k + k) * self.a + a) * self.m) + m
    }

    /// Compute the flat index into `scores` (same layout as `precision`).
    #[allow(dead_code)]
    fn scores_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        self.precision_idx(t, r, k, a, m)
    }
}

/// COCO evaluation engine.
///
/// Computes AP and AR metrics for bbox, segmentation, and keypoint predictions.
/// Also supports LVIS federated evaluation via [`COCOeval::new_lvis`].
///
/// The standard workflow is three steps:
///
/// ```rust,ignore
/// let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
/// ev.evaluate();   // per-image IoU matching
/// ev.accumulate(); // aggregate into precision/recall curves
/// ev.summarize();  // print + store the summary metrics in ev.stats
/// ```
///
/// For LVIS, use [`run`](COCOeval::run) as a convenience:
///
/// ```rust,ignore
/// let mut ev = COCOeval::new_lvis(coco_gt, coco_dt, IouType::Segm);
/// ev.run();
/// let results = ev.get_results(); // HashMap<metric_name, f64>
/// ```
pub struct COCOeval {
    pub coco_gt: COCO,
    pub coco_dt: COCO,
    pub params: Params,
    pub eval_imgs: Vec<Option<EvalImg>>,
    ious: HashMap<(u64, u64), Vec<Vec<f64>>>,
    pub eval: Option<AccumulatedEval>,
    pub stats: Option<Vec<f64>>,
    /// LVIS federated evaluation mode.
    pub is_lvis: bool,
    /// LVIS: k_indices bucketed by category frequency: [rare, common, frequent].
    /// Populated during `evaluate()` when `is_lvis=true`.
    freq_groups: [Vec<usize>; 3],
    /// LVIS: img_id → set of neg_category_ids (unmatched DTs count as FP).
    neg_cats: HashMap<u64, HashSet<u64>>,
    /// LVIS: img_id → set of not_exhaustive_category_ids (unmatched DTs are ignored).
    not_exhaustive: HashMap<u64, HashSet<u64>>,
}

impl COCOeval {
    /// Create a new COCOeval from ground truth and detection COCO objects.
    pub fn new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        COCOeval {
            coco_gt,
            coco_dt,
            params: Params::new(iou_type),
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
            is_lvis: false,
            freq_groups: [Vec::new(), Vec::new(), Vec::new()],
            neg_cats: HashMap::new(),
            not_exhaustive: HashMap::new(),
        }
    }

    /// Create a new COCOeval configured for LVIS federated evaluation.
    ///
    /// LVIS uses federated annotation — each image is only exhaustively labeled
    /// for a subset of categories. This constructor sets `max_dets=300` and enables
    /// federated filtering so unmatched detections on unlabeled or unchecked categories
    /// are not penalized as false positives.
    ///
    /// Behaviour controlled by per-image GT fields:
    /// - `neg_category_ids`: categories confirmed absent → unmatched DTs count as FP.
    /// - `not_exhaustive_category_ids`: categories not fully checked → unmatched DTs ignored.
    ///
    /// Produces 13 metrics: AP, AP50, AP75, APs, APm, APl, APr (rare), APc (common),
    /// APf (frequent), AR@300, ARs@300, ARm@300, ARl@300.
    pub fn new_lvis(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        let mut params = Params::new(iou_type);
        params.max_dets = vec![300];

        let mut neg_cats: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut not_exhaustive: HashMap<u64, HashSet<u64>> = HashMap::new();
        for img in &coco_gt.dataset.images {
            if !img.neg_category_ids.is_empty() {
                neg_cats.insert(img.id, img.neg_category_ids.iter().copied().collect());
            }
            if !img.not_exhaustive_category_ids.is_empty() {
                not_exhaustive.insert(
                    img.id,
                    img.not_exhaustive_category_ids.iter().copied().collect(),
                );
            }
        }

        COCOeval {
            coco_gt,
            coco_dt,
            params,
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
            is_lvis: true,
            freq_groups: [Vec::new(), Vec::new(), Vec::new()],
            neg_cats,
            not_exhaustive,
        }
    }

    /// Run per-image evaluation.
    pub fn evaluate(&mut self) {
        // Set img_ids and cat_ids if not set
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

        let cat_ids = if self.params.use_cats {
            self.params.cat_ids.clone()
        } else {
            vec![u64::MAX] // dummy single category (avoids collision with real category_id=0)
        };

        let img_ids = self.params.img_ids.clone();

        // LVIS: build freq_groups now that cat_ids are established.
        if self.is_lvis {
            let cat_id_to_k_idx: HashMap<u64, usize> =
                cat_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let mut rare = Vec::new();
            let mut common = Vec::new();
            let mut frequent = Vec::new();
            for cat in &self.coco_gt.dataset.categories {
                if let Some(&k_idx) = cat_id_to_k_idx.get(&cat.id) {
                    match cat.frequency.as_deref() {
                        Some("r") => rare.push(k_idx),
                        Some("c") => common.push(k_idx),
                        Some("f") => frequent.push(k_idx),
                        _ => {}
                    }
                }
            }
            self.freq_groups = [rare, common, frequent];
        }

        // Build allowed sets from params (populated above, so always non-empty here).
        let allowed_imgs: HashSet<u64> = img_ids.iter().copied().collect();
        let allowed_cats: HashSet<u64> = cat_ids.iter().copied().collect();

        // Build sparse pairs: union of non-empty GT and DT (img, cat) pairs filtered to params.
        // At large-scale (e.g. Objects365: 365 cats × 80K imgs = 29M pairs), ~96% of pairs
        // are empty. Driving evaluation from the index instead reduces pairs by ~35x.
        //
        // LVIS federated filtering: DT pairs are only included if GT exists for that
        // (img, cat), or the category is in neg_category_ids for that image.
        // DT-only pairs for not-checked categories are silently dropped.
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
                    if self.is_lvis {
                        // Keep DT pair only if GT exists OR cat is explicitly neg for this image.
                        if gt_pairs.contains(&pair)
                            || self
                                .neg_cats
                                .get(&pair.0)
                                .is_some_and(|s| s.contains(&pair.1))
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

        // Sort for deterministic output order.
        let mut sparse_pairs: Vec<(u64, u64)> = sparse_set.into_iter().collect();
        sparse_pairs.sort_unstable();

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
        // sparse_pairs × area_rngs replaces the old cat_ids × area_rngs × img_ids product.
        let max_det = *self.params.max_dets.last().unwrap_or(&100);
        let area_rngs = self.params.area_rng.clone();

        // Tuple: (cat_id, area_rng, img_id, not_exhaustive_cat)
        let mut eval_tuples: Vec<(u64, [f64; 2], u64, bool)> =
            Vec::with_capacity(sparse_pairs.len() * area_rngs.len());
        for &(img_id, cat_id) in &sparse_pairs {
            let not_exhaustive_cat = self.is_lvis
                && self
                    .not_exhaustive
                    .get(&img_id)
                    .is_some_and(|s| s.contains(&cat_id));
            for &area_rng in &area_rngs {
                eval_tuples.push((cat_id, area_rng, img_id, not_exhaustive_cat));
            }
        }

        self.eval_imgs = eval_tuples
            .par_iter()
            .map(|&(cat_id, area_rng, img_id, not_exhaustive_cat)| {
                Self::evaluate_img_static(
                    &self.coco_gt,
                    &self.coco_dt,
                    &self.params,
                    &self.ious,
                    img_id,
                    cat_id,
                    area_rng,
                    max_det,
                    not_exhaustive_cat,
                )
            })
            .collect();
    }

    /// Compute the IoU/OKS matrix for a given image and category.
    fn compute_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        img_id: u64,
        cat_id: u64,
    ) -> Vec<Vec<f64>> {
        let gt_anns = Self::get_anns_static(coco_gt, params, img_id, cat_id);
        let dt_anns = Self::get_anns_static(coco_dt, params, img_id, cat_id);

        if gt_anns.is_empty() || dt_anns.is_empty() {
            return Vec::new();
        }

        match params.iou_type {
            IouType::Segm => Self::compute_segm_iou_static(coco_gt, coco_dt, dt_anns, gt_anns),
            IouType::Bbox => Self::compute_bbox_iou_static(coco_gt, coco_dt, dt_anns, gt_anns),
            IouType::Keypoints => {
                Self::compute_oks_static(coco_gt, coco_dt, params, dt_anns, gt_anns)
            }
        }
    }

    /// Get annotation IDs for an image, optionally filtered by category.
    fn get_anns_static<'a>(coco: &'a COCO, params: &Params, img_id: u64, cat_id: u64) -> &'a [u64] {
        if params.use_cats {
            coco.get_ann_ids_for_img_cat(img_id, cat_id)
        } else {
            coco.get_ann_ids_for_img(img_id)
        }
    }

    /// Compute segmentation mask IoU by converting annotations to RLE and calling `mask::iou`.
    fn compute_segm_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
    ) -> Vec<Vec<f64>> {
        let dt_rles: Vec<Rle> = dt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_dt.get_ann(id)?;
                coco_dt.ann_to_rle(ann)
            })
            .collect();
        let gt_rles: Vec<Rle> = gt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_gt.get_ann(id)?;
                coco_gt.ann_to_rle(ann)
            })
            .collect();

        let iscrowd: Vec<bool> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id).map(|a| a.iscrowd))
            .collect();

        mask::iou(&dt_rles, &gt_rles, &iscrowd)
    }

    /// Compute bounding box IoU by extracting bbox arrays and calling `mask::bbox_iou`.
    fn compute_bbox_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
    ) -> Vec<Vec<f64>> {
        let dt_bbs: Vec<[f64; 4]> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id)?.bbox)
            .collect();
        let gt_bbs: Vec<[f64; 4]> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id)?.bbox)
            .collect();
        let iscrowd: Vec<bool> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id).map(|a| a.iscrowd))
            .collect();

        mask::bbox_iou(&dt_bbs, &gt_bbs, &iscrowd)
    }

    /// Compute OKS (Object Keypoint Similarity) between detection and GT keypoints.
    ///
    /// OKS = mean_k[ exp( -d_k^2 / (2 * s_k^2 * area) ) ] where d_k is the Euclidean
    /// distance for keypoint k, s_k is the per-keypoint sigma, and area is the GT area.
    /// Only visible GT keypoints contribute. When no GT keypoints are visible, distance
    /// is measured to the GT bounding box boundary instead.
    fn compute_oks_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        dt_ids: &[u64],
        gt_ids: &[u64],
    ) -> Vec<Vec<f64>> {
        let sigmas = &params.kpt_oks_sigmas;
        let num_kpts = sigmas.len();
        // vars = (sigmas * 2)**2 = 4 * sigma^2  (matching pycocotools)
        let vars: Vec<f64> = sigmas.iter().map(|s| (2.0 * s).powi(2)).collect();

        let d = dt_ids.len();
        let g = gt_ids.len();
        let mut result = vec![vec![0.0f64; g]; d];

        let gt_anns: Vec<_> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id))
            .collect();
        let dt_anns: Vec<_> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id))
            .collect();

        for (j, gt_ann) in gt_anns.iter().enumerate() {
            let gt_kpts = match &gt_ann.keypoints {
                Some(k) => k,
                None => continue,
            };
            let gt_area = gt_ann.area.unwrap_or(0.0) + f64::EPSILON;
            let gt_bbox = gt_ann.bbox.unwrap_or([0.0; 4]);

            // Count visible GT keypoints
            let k1: usize = (0..num_kpts)
                .filter(|&ki| gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0)
                .count();

            // Compute ignore region bounds (double the GT bbox)
            let x0 = gt_bbox[0] - gt_bbox[2];
            let x1 = gt_bbox[0] + gt_bbox[2] * 2.0;
            let y0 = gt_bbox[1] - gt_bbox[3];
            let y1 = gt_bbox[1] + gt_bbox[3] * 2.0;

            for (i, dt_ann) in dt_anns.iter().enumerate() {
                let dt_kpts = match &dt_ann.keypoints {
                    Some(k) => k,
                    None => continue,
                };

                // Compute per-keypoint distances
                let mut e_vals: Vec<f64> = Vec::with_capacity(num_kpts);

                for (ki, &var_k) in vars.iter().enumerate().take(num_kpts) {
                    let gx = gt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                    let gy = gt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);
                    let xd = dt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                    let yd = dt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);

                    let (dx, dy) = if k1 > 0 {
                        (xd - gx, yd - gy)
                    } else {
                        // No visible GT keypoints: measure distance to bbox boundary
                        let dx = 0.0_f64.max(x0 - xd) + 0.0_f64.max(xd - x1);
                        let dy = 0.0_f64.max(y0 - yd) + 0.0_f64.max(yd - y1);
                        (dx, dy)
                    };

                    let e = (dx * dx + dy * dy) / var_k / gt_area / 2.0;
                    e_vals.push(e);
                }

                // Filter to visible keypoints if k1 > 0
                let filtered: Vec<f64> = if k1 > 0 {
                    e_vals
                        .iter()
                        .enumerate()
                        .filter(|&(ki, _)| gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0)
                        .map(|(_, &e)| e)
                        .collect()
                } else {
                    e_vals
                };

                if !filtered.is_empty() {
                    let oks: f64 =
                        filtered.iter().map(|&e| (-e).exp()).sum::<f64>() / filtered.len() as f64;
                    result[i][j] = oks;
                }
            }
        }

        result
    }

    /// Evaluate a single image+category combination.
    ///
    /// `not_exhaustive_cat` — when true (LVIS mode), unmatched detections are
    /// ignored rather than counted as false positives.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_img_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        ious: &HashMap<(u64, u64), Vec<Vec<f64>>>,
        img_id: u64,
        cat_id: u64,
        area_rng: [f64; 2],
        max_det: usize,
        not_exhaustive_cat: bool,
    ) -> Option<EvalImg> {
        let gt_ids = Self::get_anns_static(coco_gt, params, img_id, cat_id);
        let dt_ids = Self::get_anns_static(coco_dt, params, img_id, cat_id);

        if gt_ids.is_empty() && dt_ids.is_empty() {
            return None;
        }

        // Load GT annotations, determine ignore flags
        let gt_anns: Vec<_> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id))
            .collect();
        let is_kp = params.iou_type == IouType::Keypoints;
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

        // Load DT annotations, sort by score descending, limit to max_det
        let mut dt_anns: Vec<_> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id))
            .collect();
        dt_anns.sort_by(|a, b| {
            b.score
                .unwrap_or(0.0)
                .partial_cmp(&a.score.unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if dt_anns.len() > max_det {
            dt_anns.truncate(max_det);
        }

        let dt_scores: Vec<f64> = dt_anns.iter().map(|a| a.score.unwrap_or(0.0)).collect();

        // Determine which DT are ignored by area
        let dt_area_ignore: Vec<bool> = dt_anns
            .iter()
            .map(|ann| {
                let a = ann.area.unwrap_or(0.0);
                a < area_rng[0] || a > area_rng[1]
            })
            .collect();

        // Get IoU matrix
        let iou_matrix = ious.get(&(img_id, cat_id));

        let num_iou_thrs = params.iou_thrs.len();
        let d = dt_anns.len();
        let g = gt_anns.len();

        let mut dt_matches = vec![vec![0u64; d]; num_iou_thrs];
        let mut gt_matches = vec![vec![0u64; g]; num_iou_thrs];
        let mut dt_matched = vec![vec![false; d]; num_iou_thrs];
        let mut gt_matched = vec![vec![false; g]; num_iou_thrs];
        let mut dt_ignore_flags = vec![vec![false; d]; num_iou_thrs];

        if let Some(iou_mat) = iou_matrix {
            // Precompute remapped IoU matrix indexed by (dt_anns order, gt_order)
            // so we can use direct array indexing instead of HashMap lookups.
            let dt_id_to_iou_idx: HashMap<u64, usize> =
                dt_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            let gt_id_to_iou_idx: HashMap<u64, usize> =
                gt_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

            let iou_reordered: Vec<Vec<f64>> = dt_anns
                .iter()
                .map(|dt_ann| {
                    let dt_idx = dt_id_to_iou_idx.get(&dt_ann.id).copied();
                    gt_order
                        .iter()
                        .map(|&gi_orig| {
                            let gt_idx = gt_id_to_iou_idx.get(&gt_anns[gi_orig].id).copied();
                            match (dt_idx, gt_idx) {
                                (Some(di), Some(gi))
                                    if di < iou_mat.len() && gi < iou_mat[di].len() =>
                                {
                                    iou_mat[di][gi]
                                }
                                _ => 0.0,
                            }
                        })
                        .collect()
                })
                .collect();

            // Greedy matching: for each IoU threshold, iterate detections in
            // score-descending order and greedily match each to the best available GT.
            for (t_idx, &iou_thr) in params.iou_thrs.iter().enumerate() {
                for (di, dt_ann) in dt_anns.iter().enumerate() {
                    // Track the best GT match: iou must exceed the threshold
                    let mut best_iou = iou_thr;
                    let mut best_gi: Option<usize> = None;

                    let dt_row = &iou_reordered[di];

                    for (gi_sorted, &iou_val) in dt_row.iter().enumerate() {
                        // Skip already matched non-crowd GTs (pycocotools uses iscrowd,
                        // not the full ignore flag, so crowd GTs can be matched multiple times)
                        if gt_matched[t_idx][gi_sorted] && !gt_iscrowd_sorted[gi_sorted] {
                            continue;
                        }

                        // Match: iou must meet threshold, prefer non-ignored GT
                        if iou_val < best_iou {
                            continue;
                        }

                        // Prefer non-ignored GT over ignored GT
                        if let Some(prev_gi) = best_gi {
                            let best_ignored = gt_ignore_sorted[prev_gi];
                            let curr_ignored = gt_ignore_sorted[gi_sorted];
                            if !best_ignored && curr_ignored {
                                continue;
                            }
                        }

                        best_iou = iou_val;
                        best_gi = Some(gi_sorted);
                    }

                    if let Some(gi) = best_gi {
                        dt_matches[t_idx][di] = gt_anns[gt_order[gi]].id;
                        gt_matches[t_idx][gi] = dt_ann.id;
                        dt_matched[t_idx][di] = true;
                        gt_matched[t_idx][gi] = true;

                        // DT is ignored if matched to ignored GT
                        dt_ignore_flags[t_idx][di] = gt_ignore_sorted[gi];
                    } else {
                        // Unmatched DT: ignored if area out of range
                        dt_ignore_flags[t_idx][di] = dt_area_ignore[di];
                    }
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

    /// Accumulate per-image results into precision/recall arrays.
    pub fn accumulate(&mut self) {
        let t = self.params.iou_thrs.len();
        let r = self.params.rec_thrs.len();
        let k = if self.params.use_cats {
            self.params.cat_ids.len()
        } else {
            1
        };
        let a = self.params.area_rng.len();
        let m = self.params.max_dets.len();

        // Build category_id → k_idx mapping for grouping eval_imgs.
        let cat_id_to_k_idx: HashMap<u64, usize> = if self.params.use_cats {
            self.params
                .cat_ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect()
        } else {
            std::iter::once((u64::MAX, 0usize)).collect()
        };

        // Group eval_imgs by (k_idx, a_idx) — O(eval_imgs) once.
        // Replaces the old dense index formula k_actual * (a * N) + a_idx * N + img_idx,
        // which assumed a specific dense layout that no longer applies after the sparse refactor.
        let mut grouped: Vec<Vec<&EvalImg>> = vec![Vec::new(); k * a];
        for eval in self.eval_imgs.iter().flatten() {
            if let Some(&k_idx) = cat_id_to_k_idx.get(&eval.category_id) {
                let a_idx = self
                    .params
                    .area_rng
                    .iter()
                    .position(|&rng| rng == eval.area_rng)
                    .unwrap_or(0);
                grouped[k_idx * a + a_idx].push(eval);
            }
        }

        // Build flat list of (k_idx, a_idx, m_idx) work items
        let work_items: Vec<(usize, usize, usize)> = (0..k)
            .flat_map(|k_idx| {
                (0..a).flat_map(move |a_idx| (0..m).map(move |m_idx| (k_idx, a_idx, m_idx)))
            })
            .collect();

        // Each work item produces a set of (index, value) writes for precision, recall, scores
        /// Intermediate results from a single (category, area_range, max_det) work item.
        /// Each field is a list of (flat_index, value) pairs to write into the output arrays.
        struct AccResult {
            precision_writes: Vec<(usize, f64)>,
            recall_writes: Vec<(usize, f64)>,
            scores_writes: Vec<(usize, f64)>,
        }

        let acc_idx = AccumulatedEval {
            precision: vec![],
            recall: vec![],
            scores: vec![],
            t,
            r,
            k,
            a,
            m,
        };

        let results: Vec<AccResult> = work_items
            .par_iter()
            .map(|&(k_idx, a_idx, m_idx)| {
                let max_det = self.params.max_dets[m_idx];

                let mut all_dt_scores: Vec<f64> = Vec::new();
                let mut all_dt_matched: Vec<Vec<bool>> = vec![Vec::new(); t];
                let mut all_dt_ignore: Vec<Vec<bool>> = vec![Vec::new(); t];
                let mut num_gt = 0usize;

                for eval_img in &grouped[k_idx * a + a_idx] {
                    let nd = eval_img.dt_scores.len().min(max_det);

                    all_dt_scores.extend_from_slice(&eval_img.dt_scores[..nd]);
                    for t_idx in 0..t {
                        all_dt_matched[t_idx].extend_from_slice(&eval_img.dt_matched[t_idx][..nd]);
                        all_dt_ignore[t_idx].extend_from_slice(&eval_img.dt_ignore[t_idx][..nd]);
                    }

                    num_gt += eval_img.gt_ignore.iter().filter(|&&x| !x).count();
                }

                let mut precision_writes = Vec::new();
                let mut recall_writes = Vec::new();
                let mut scores_writes = Vec::new();

                if num_gt == 0 {
                    return AccResult {
                        precision_writes,
                        recall_writes,
                        scores_writes,
                    };
                }

                // Initialize precision and recall to 0.0 (distinct from -1.0 which means
                // "no data"). This ensures categories with GT but no matches show 0 AP,
                // not "missing".
                for t_idx in 0..t {
                    let recall_idx = ((t_idx * k + k_idx) * a + a_idx) * m + m_idx;
                    recall_writes.push((recall_idx, 0.0));
                    for r_idx in 0..r {
                        let p_idx = acc_idx.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        precision_writes.push((p_idx, 0.0));
                        scores_writes.push((p_idx, 0.0));
                    }
                }

                // Sort by score descending
                let mut inds: Vec<usize> = (0..all_dt_scores.len()).collect();
                inds.sort_by(|&a, &b| {
                    all_dt_scores[b]
                        .partial_cmp(&all_dt_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let nd = inds.len();

                // Hoist sorted_scores outside the threshold loop (identical across thresholds)
                let sorted_scores: Vec<f64> = inds.iter().map(|&i| all_dt_scores[i]).collect();

                // Pre-allocate buffers reused across thresholds
                let mut tp = vec![0.0f64; nd];
                let mut fp = vec![0.0f64; nd];
                let mut rc = vec![0.0f64; nd];
                let mut pr = vec![0.0f64; nd];

                let num_gt_f = num_gt as f64;

                for t_idx in 0..t {
                    // Classify each detection (in score-sorted order) as TP, FP, or ignored.
                    // Ignored detections contribute neither TP nor FP.
                    for (out_idx, &src_idx) in inds.iter().enumerate() {
                        if all_dt_ignore[t_idx][src_idx] {
                            tp[out_idx] = 0.0;
                            fp[out_idx] = 0.0;
                        } else if all_dt_matched[t_idx][src_idx] {
                            tp[out_idx] = 1.0;
                            fp[out_idx] = 0.0;
                        } else {
                            tp[out_idx] = 0.0;
                            fp[out_idx] = 1.0;
                        }
                    }

                    // Cumulative sum: tp[d] = total TPs up to detection d
                    for d in 1..nd {
                        tp[d] += tp[d - 1];
                        fp[d] += fp[d - 1];
                    }

                    // Compute recall and precision at each detection threshold
                    for d in 0..nd {
                        rc[d] = tp[d] / num_gt_f;
                        let total = tp[d] + fp[d];
                        pr[d] = if total > 0.0 { tp[d] / total } else { 0.0 };
                    }

                    let recall_idx = ((t_idx * k + k_idx) * a + a_idx) * m + m_idx;
                    if nd > 0 {
                        recall_writes.push((recall_idx, rc[nd - 1]));
                    }

                    // Make precision monotonically decreasing from right to left.
                    // This is the standard PASCAL VOC interpolation: at each recall level,
                    // precision is the maximum precision at any recall >= that level.
                    for d in (0..nd.saturating_sub(1)).rev() {
                        pr[d] = pr[d].max(pr[d + 1]);
                    }

                    // Map interpolated precision onto the 101 fixed recall thresholds.
                    // Both rc[] and rec_thrs are sorted ascending, so we use a two-pointer scan.
                    let mut rc_ptr = 0;
                    for (r_idx, &rec_thr) in self.params.rec_thrs.iter().enumerate() {
                        let p_idx = acc_idx.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        // Advance pointer to first rc >= rec_thr
                        while rc_ptr < nd && rc[rc_ptr] < rec_thr {
                            rc_ptr += 1;
                        }
                        if rc_ptr < nd {
                            precision_writes.push((p_idx, pr[rc_ptr]));
                            scores_writes.push((p_idx, sorted_scores[rc_ptr]));
                        }
                    }
                }

                AccResult {
                    precision_writes,
                    recall_writes,
                    scores_writes,
                }
            })
            .collect();

        // Merge results into output arrays
        let total = t * r * k * a * m;
        let mut precision = vec![-1.0f64; total];
        let mut scores = vec![-1.0f64; total];
        let total_recall = t * k * a * m;
        let mut recall = vec![-1.0f64; total_recall];

        for result in results {
            for (idx, val) in result.precision_writes {
                precision[idx] = val;
            }
            for (idx, val) in result.recall_writes {
                recall[idx] = val;
            }
            for (idx, val) in result.scores_writes {
                scores[idx] = val;
            }
        }

        self.eval = Some(AccumulatedEval {
            precision,
            recall,
            scores,
            t,
            r,
            k,
            a,
            m,
        });
    }

    /// Print the standard 12-line COCO evaluation summary.
    pub fn summarize(&mut self) {
        let eval = match &self.eval {
            Some(e) => e,
            None => {
                eprintln!("Please run evaluate() and accumulate() first.");
                return;
            }
        };

        let is_kp = self.params.iou_type == IouType::Keypoints;

        // Warn if parameters differ from what the hardcoded summary display expects.
        let defaults = Params::new(self.params.iou_type);
        let mut warnings = Vec::new();

        let default_iou: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * i as f64).collect();
        if self.params.iou_thrs != default_iou {
            warnings.push(
                "iou_thrs differ from default (0.50:0.05:0.95). AP50/AP75 lines may show -1.000."
                    .to_string(),
            );
        }
        let expected_max_dets = if self.is_lvis {
            vec![300usize]
        } else {
            defaults.max_dets.clone()
        };
        if self.params.max_dets != expected_max_dets {
            warnings.push(format!(
                "max_dets differ from expected ({:?}). AR lines may use unexpected max_dets values.",
                expected_max_dets
            ));
        }
        if self.params.area_rng_lbl != defaults.area_rng_lbl {
            warnings.push(format!(
                "area_rng_lbl differ from default ({:?}). Per-size metrics may fall back to index 0.",
                defaults.area_rng_lbl
            ));
        }

        for w in &warnings {
            eprintln!("Warning: {}", w);
        }

        // Compute a single summary statistic by averaging over the relevant slice
        // of the precision or recall array. Returns -1.0 if no valid data exists.
        let summarize_stat =
            |ap: bool, iou_thr: Option<f64>, area_lbl: &str, max_det: usize| -> f64 {
                let a_idx = self
                    .params
                    .area_rng_lbl
                    .iter()
                    .position(|l| l == area_lbl)
                    .unwrap_or(0);
                let m_idx = self
                    .params
                    .max_dets
                    .iter()
                    .position(|&d| d == max_det)
                    .unwrap_or(0);

                let t_indices: Vec<usize> = if let Some(thr) = iou_thr {
                    self.params
                        .iou_thrs
                        .iter()
                        .enumerate()
                        .filter(|(_, &t)| (t - thr).abs() < 1e-9)
                        .map(|(i, _)| i)
                        .collect()
                } else {
                    (0..eval.t).collect()
                };

                let mut vals = Vec::new();
                for &t_idx in &t_indices {
                    for k_idx in 0..eval.k {
                        if ap {
                            for r_idx in 0..eval.r {
                                let idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                                let v = eval.precision[idx];
                                if v >= 0.0 {
                                    vals.push(v);
                                }
                            }
                        } else {
                            let idx = eval.recall_idx(t_idx, k_idx, a_idx, m_idx);
                            let v = eval.recall[idx];
                            if v >= 0.0 {
                                vals.push(v);
                            }
                        }
                    }
                }

                if vals.is_empty() {
                    -1.0
                } else {
                    vals.iter().sum::<f64>() / vals.len() as f64
                }
            };

        let max_det_default = *self.params.max_dets.last().unwrap_or(&100);
        let max_det_small = if self.params.max_dets.len() >= 3 {
            self.params.max_dets[0]
        } else {
            max_det_default
        };
        let max_det_med = if self.params.max_dets.len() >= 3 {
            self.params.max_dets[1]
        } else {
            max_det_default
        };

        /// Definition of a single summary metric (one row in the COCO output table).
        struct MetricDef {
            /// true = Average Precision, false = Average Recall.
            ap: bool,
            /// Specific IoU threshold, or None to average over all thresholds.
            iou_thr: Option<f64>,
            /// Area range label to filter by (e.g. "all", "small", "medium", "large").
            area_lbl: &'static str,
            /// Maximum detections per image for this metric.
            max_det: usize,
        }

        let metrics_bbox_segm = |max_d: usize, max_d_s: usize, max_d_m: usize| -> Vec<MetricDef> {
            vec![
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: Some(0.5),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: Some(0.75),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "small",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "medium",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "large",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d_s,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d_m,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "small",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "medium",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "large",
                    max_det: max_d,
                },
            ]
        };

        let metrics_kp = |max_d: usize| -> Vec<MetricDef> {
            vec![
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: Some(0.5),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: Some(0.75),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "medium",
                    max_det: max_d,
                },
                MetricDef {
                    ap: true,
                    iou_thr: None,
                    area_lbl: "large",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: Some(0.5),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: Some(0.75),
                    area_lbl: "all",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "medium",
                    max_det: max_d,
                },
                MetricDef {
                    ap: false,
                    iou_thr: None,
                    area_lbl: "large",
                    max_det: max_d,
                },
            ]
        };

        if self.is_lvis {
            // LVIS summarize: 13 metrics with max_dets=300.
            // APr/APc/APf are computed as mean per-category AP within each freq group.
            let a_idx_all = self
                .params
                .area_rng_lbl
                .iter()
                .position(|l| l == "all")
                .unwrap_or(0);
            let m_idx_last = self.params.max_dets.len().saturating_sub(1);

            // Per-category AP for the freq-group metrics.
            let per_cat_ap: Vec<f64> = (0..eval.k)
                .map(|k_idx| {
                    let mut vals = Vec::new();
                    for t_idx in 0..eval.t {
                        for r_idx in 0..eval.r {
                            let idx =
                                eval.precision_idx(t_idx, r_idx, k_idx, a_idx_all, m_idx_last);
                            let v = eval.precision[idx];
                            if v >= 0.0 {
                                vals.push(v);
                            }
                        }
                    }
                    if vals.is_empty() {
                        -1.0
                    } else {
                        vals.iter().sum::<f64>() / vals.len() as f64
                    }
                })
                .collect();

            let freq_group_ap = |indices: &[usize]| -> f64 {
                if indices.is_empty() {
                    return -1.0;
                }
                let valid: Vec<f64> = indices
                    .iter()
                    .filter_map(|&k| {
                        let v = per_cat_ap[k];
                        if v >= 0.0 {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect();
                if valid.is_empty() {
                    -1.0
                } else {
                    valid.iter().sum::<f64>() / valid.len() as f64
                }
            };

            let ap = summarize_stat(true, None, "all", max_det_default);
            let ap50 = summarize_stat(true, Some(0.5), "all", max_det_default);
            let ap75 = summarize_stat(true, Some(0.75), "all", max_det_default);
            let aps = summarize_stat(true, None, "small", max_det_default);
            let apm = summarize_stat(true, None, "medium", max_det_default);
            let apl = summarize_stat(true, None, "large", max_det_default);
            let ap_r = freq_group_ap(&self.freq_groups[0]);
            let ap_c = freq_group_ap(&self.freq_groups[1]);
            let ap_f = freq_group_ap(&self.freq_groups[2]);
            let ar = summarize_stat(false, None, "all", max_det_default);
            let ar_s = summarize_stat(false, None, "small", max_det_default);
            let ar_m = summarize_stat(false, None, "medium", max_det_default);
            let ar_l = summarize_stat(false, None, "large", max_det_default);

            let lvis_metrics: &[(&str, f64)] = &[
                ("AP", ap),
                ("AP50", ap50),
                ("AP75", ap75),
                ("APs", aps),
                ("APm", apm),
                ("APl", apl),
                ("APr", ap_r),
                ("APc", ap_c),
                ("APf", ap_f),
                ("AR@300", ar),
                ("ARs@300", ar_s),
                ("ARm@300", ar_m),
                ("ARl@300", ar_l),
            ];

            let mut stats = Vec::with_capacity(lvis_metrics.len());
            for (name, val) in lvis_metrics {
                stats.push(*val);
                let val_str = if *val < 0.0 {
                    format!("{:0.3}", -1.0f64)
                } else {
                    format!("{:0.3}", val)
                };
                println!(" {:>10} = {}", name, val_str);
            }
            self.stats = Some(stats);
            return;
        }

        let metrics = if is_kp {
            metrics_kp(max_det_default)
        } else {
            metrics_bbox_segm(max_det_default, max_det_small, max_det_med)
        };

        let iou_type_str = match self.params.iou_type {
            IouType::Bbox => "bbox",
            IouType::Segm => "segm",
            IouType::Keypoints => "keypoints",
        };

        let mut stats = Vec::with_capacity(metrics.len());

        for m in &metrics {
            let val = summarize_stat(m.ap, m.iou_thr, m.area_lbl, m.max_det);
            stats.push(val);

            let metric_name = if m.ap {
                "Average Precision"
            } else {
                "Average Recall"
            };
            let metric_short = if m.ap { "AP" } else { "AR" };

            let iou_str = match m.iou_thr {
                Some(thr) => format!("{:.2}", thr),
                None => "0.50:0.95".to_string(),
            };

            let area_str = m.area_lbl;
            let det_str = m.max_det;

            let val_str = if val < 0.0 {
                format!("{:0.3}", -1.0)
            } else {
                format!("{:0.3}", val)
            };

            println!(
                " {:<18} @[ IoU={:<9} | area={:>6} | maxDets={:>3} ] = {}",
                format!("{} ({})", metric_name, metric_short),
                iou_str,
                area_str,
                det_str,
                val_str
            );
        }

        println!("Eval type: {}", iou_type_str);
        self.stats = Some(stats);
    }

    /// Run the full evaluation pipeline in one call: `evaluate` → `accumulate` → `summarize`.
    ///
    /// Equivalent to calling the three methods in sequence. Primarily used with LVIS
    /// pipelines (e.g. Detectron2 / MMDetection) that expect a single `run()` entry point.
    pub fn run(&mut self) {
        self.evaluate();
        self.accumulate();
        self.summarize();
    }

    /// Return summary metrics as a `HashMap<metric_name, value>`.
    ///
    /// Must be called after [`summarize`](COCOeval::summarize). Returns an empty map
    /// if `summarize` has not been run.
    ///
    /// For LVIS mode: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `APr`, `APc`, `APf`,
    /// `AR@300`, `ARs@300`, `ARm@300`, `ARl@300`.
    ///
    /// For standard COCO bbox/segm: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`,
    /// `AR1`, `AR10`, `AR100`, `ARs`, `ARm`, `ARl`.
    ///
    /// For keypoints: `AP`, `AP50`, `AP75`, `APm`, `APl`,
    /// `AR`, `AR50`, `AR75`, `ARm`, `ARl`.
    pub fn get_results(&self) -> HashMap<String, f64> {
        let stats = match &self.stats {
            Some(s) => s,
            None => return HashMap::new(),
        };

        let keys: &[&str] = if self.is_lvis {
            &[
                "AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf", "AR@300",
                "ARs@300", "ARm@300", "ARl@300",
            ]
        } else if self.params.iou_type == IouType::Keypoints {
            &[
                "AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl",
            ]
        } else {
            &[
                "AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm",
                "ARl",
            ]
        };

        keys.iter()
            .zip(stats.iter())
            .map(|(&k, &v)| (k.to_string(), v))
            .collect()
    }

    /// Print a formatted results table to stdout.
    ///
    /// For LVIS, matches the lvis-api `print_results()` style (metric name + value per line).
    /// For standard COCO, equivalent to the output already printed by `summarize()`.
    /// Must be called after `summarize()`.
    pub fn print_results(&self) {
        let results = self.get_results();
        if results.is_empty() {
            eprintln!("No results to print. Run evaluate(), accumulate(), and summarize() first.");
            return;
        }

        let keys: &[&str] = if self.is_lvis {
            &[
                "AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf", "AR@300",
                "ARs@300", "ARm@300", "ARl@300",
            ]
        } else if self.params.iou_type == IouType::Keypoints {
            &[
                "AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl",
            ]
        } else {
            &[
                "AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm",
                "ARl",
            ]
        };

        for key in keys {
            let val = results.get(*key).copied().unwrap_or(-1.0);
            let val_str = if val < 0.0 {
                format!("{:0.3}", -1.0f64)
            } else {
                format!("{:0.3}", val)
            };
            println!(" {:>10} = {}", key, val_str);
        }
    }
}
