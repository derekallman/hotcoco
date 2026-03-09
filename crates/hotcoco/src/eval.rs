//! COCO evaluation engine — faithful port of `pycocotools/cocoeval.py`.
//!
//! Implements evaluate, accumulate, and summarize for bbox, segm, and keypoint evaluation.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::coco::COCO;
use crate::mask;
use crate::params::{IouType, Params};
use crate::types::Rle;

/// Per-category confusion matrix for object detection.
///
/// Rows are ground truth categories, columns are predicted categories.
/// Index `num_cats` (the last row/column) represents "background" — unmatched GTs
/// (false negatives) land in the background column, unmatched DTs (false positives)
/// land in the background row.
///
/// Use [`COCOeval::confusion_matrix`] to compute this.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Raw counts, row-major, shape (num_cats+1) × (num_cats+1).
    /// Index `K = num_cats` is the background row/column.
    pub matrix: Vec<u64>,
    pub num_cats: usize,
    /// Category IDs corresponding to rows/cols 0..num_cats-1.
    pub cat_ids: Vec<u64>,
    pub iou_thr: f64,
}

impl ConfusionMatrix {
    /// Get the count at row `gt_idx`, column `pred_idx`.
    pub fn get(&self, gt_idx: usize, pred_idx: usize) -> u64 {
        let k = self.num_cats + 1;
        self.matrix[gt_idx * k + pred_idx]
    }

    /// Row-normalized matrix as flat `Vec<f64>` (same shape as `matrix`).
    ///
    /// Each row is divided by its sum so rows sum to 1.0.
    /// Zero rows remain all-zero.
    pub fn normalized(&self) -> Vec<f64> {
        let k = self.num_cats + 1;
        let mut norm = vec![0.0f64; k * k];
        for row in 0..k {
            let row_sum: u64 = (0..k).map(|col| self.matrix[row * k + col]).sum();
            if row_sum > 0 {
                let denom = row_sum as f64;
                for col in 0..k {
                    norm[row * k + col] = self.matrix[row * k + col] as f64 / denom;
                }
            }
        }
        norm
    }
}

/// TIDE error decomposition for object detection.
///
/// Produced by [`COCOeval::tide_errors`]. Each ΔAP value measures how much
/// average AP would improve if all errors of that type were fixed.
#[derive(Debug, Clone)]
pub struct TideErrors {
    /// ΔAP for each error type (fixing all errors of that type).
    /// Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`, `"FP"`, `"FN"`.
    pub delta_ap: HashMap<String, f64>,
    /// Count of each error type across all categories and images.
    /// Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`.
    pub counts: HashMap<String, u64>,
    /// Baseline AP at `pos_thr` (mean over categories with GT).
    pub ap_base: f64,
    /// IoU threshold for TP/FP classification.
    pub pos_thr: f64,
    /// Background IoU threshold for Loc/Both/Bkg discrimination.
    pub bg_thr: f64,
}

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
/// let results = ev.get_results(None, false); // HashMap<metric_name, f64>
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

                // Compute OKS in a single pass — sum exp(-e) over visible keypoints
                // (or all keypoints when k1 == 0) without intermediate allocations.
                let mut oks_sum = 0.0_f64;
                let mut oks_count = 0_usize;

                for (ki, &var_k) in vars.iter().enumerate().take(num_kpts) {
                    // When k1 > 0, only include visible GT keypoints
                    let visible = gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0;
                    if k1 > 0 && !visible {
                        continue;
                    }

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
                    oks_sum += (-e).exp();
                    oks_count += 1;
                }

                if oks_count > 0 {
                    result[i][j] = oks_sum / oks_count as f64;
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

        // Load GT annotations and track each annotation's original index in gt_ids,
        // which corresponds to its column in the IoU matrix from compute_iou_static.
        let gt_with_iou_idx: Vec<(usize, &crate::types::Annotation)> = gt_ids
            .iter()
            .enumerate()
            .filter_map(|(iou_idx, &id)| Some((iou_idx, coco_gt.get_ann(id)?)))
            .collect();
        let gt_anns: Vec<&crate::types::Annotation> =
            gt_with_iou_idx.iter().map(|&(_, ann)| ann).collect();
        let gt_iou_indices: Vec<usize> = gt_with_iou_idx.iter().map(|&(idx, _)| idx).collect();
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

        // Load DT annotations with their original IoU matrix row indices,
        // sort by score descending, then limit to max_det.
        let mut dt_with_iou_idx: Vec<(usize, &crate::types::Annotation)> = dt_ids
            .iter()
            .enumerate()
            .filter_map(|(iou_idx, &id)| Some((iou_idx, coco_dt.get_ann(id)?)))
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
        let dt_anns: Vec<&crate::types::Annotation> =
            dt_with_iou_idx.iter().map(|&(_, ann)| ann).collect();
        let dt_iou_indices: Vec<usize> = dt_with_iou_idx.iter().map(|&(idx, _)| idx).collect();

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

            // For each detection, build two sorted GT index lists (by descending IoU):
            // one for non-ignored GTs, one for ignored GTs. This enables early exit
            // within each group while preserving pycocotools' two-phase matching
            // semantics: non-ignored GTs are always preferred over ignored ones.
            let non_ignored_by_iou: Vec<Vec<usize>> = (0..d)
                .map(|di| {
                    let base = di * g;
                    let mut indices: Vec<usize> = (0..num_gt_not_ignored).collect();
                    indices.sort_by(|&a, &b| {
                        iou_flat[base + b]
                            .partial_cmp(&iou_flat[base + a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    indices
                })
                .collect();
            let ignored_by_iou: Vec<Vec<usize>> = (0..d)
                .map(|di| {
                    let base = di * g;
                    let mut indices: Vec<usize> = (num_gt_not_ignored..g).collect();
                    indices.sort_by(|&a, &b| {
                        iou_flat[base + b]
                            .partial_cmp(&iou_flat[base + a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    indices
                })
                .collect();

            // Greedy matching: for each IoU threshold, iterate detections in
            // score-descending order and greedily match each to the best available GT.
            //
            // Two-phase matching (matches pycocotools exactly):
            //   Phase 1: scan non-ignored GTs in descending IoU order.
            //   Phase 2: only if phase 1 found no match, scan ignored GTs.
            // Within each phase, early-exit once IoU drops below best_iou.
            for (t_idx, &iou_thr) in params.iou_thrs.iter().enumerate() {
                for (di, dt_ann) in dt_anns.iter().enumerate() {
                    let mut best_iou = iou_thr;
                    let mut best_gi: Option<usize> = None;
                    let base = di * g;

                    // Phase 1: non-ignored GTs (descending IoU, early exit)
                    for &gi in &non_ignored_by_iou[di] {
                        let iou_val = iou_flat[base + gi];
                        if iou_val < best_iou {
                            break;
                        }
                        if gt_matched[t_idx][gi] && !gt_iscrowd_sorted[gi] {
                            continue;
                        }
                        best_iou = iou_val;
                        best_gi = Some(gi);
                    }

                    // Phase 2: ignored GTs — only if no non-ignored match found.
                    // This matches pycocotools' `if m>-1 and gtIg[m]==0: break`
                    // which stops at the first ignored GT when a non-ignored match exists.
                    if best_gi.is_none() {
                        for &gi in &ignored_by_iou[di] {
                            let iou_val = iou_flat[base + gi];
                            if iou_val < best_iou {
                                break;
                            }
                            if gt_matched[t_idx][gi] && !gt_iscrowd_sorted[gi] {
                                continue;
                            }
                            best_iou = iou_val;
                            best_gi = Some(gi);
                        }
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

        // Build area_rng → index lookup using bit-exact f64 keys (avoids linear search).
        // area_rng values are copied verbatim from params, so bit-exact equality is safe.
        let area_rng_to_idx: HashMap<[u64; 2], usize> = self
            .params
            .area_rng
            .iter()
            .enumerate()
            .map(|(i, &rng)| ([rng[0].to_bits(), rng[1].to_bits()], i))
            .collect();

        // Group eval_imgs by (k_idx, a_idx) — O(eval_imgs) once.
        // Replaces the old dense index formula k_actual * (a * N) + a_idx * N + img_idx,
        // which assumed a specific dense layout that no longer applies after the sparse refactor.
        let mut grouped: Vec<Vec<&EvalImg>> = vec![Vec::new(); k * a];
        for eval in self.eval_imgs.iter().flatten() {
            if let Some(&k_idx) = cat_id_to_k_idx.get(&eval.category_id) {
                let a_key = [eval.area_rng[0].to_bits(), eval.area_rng[1].to_bits()];
                let a_idx = area_rng_to_idx.get(&a_key).copied().unwrap_or(0);
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
            let per_cat_ap = self.per_cat_ap(eval);

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
                let val_str = Self::format_metric(*val);
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

        let iou_type_str = self.params.iou_type.to_string();

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

            let val_str = Self::format_metric(val);

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

    /// Format a metric value: -1.0 sentinel stays as "-1.000", positive values use 3 decimal places.
    fn format_metric(val: f64) -> String {
        if val < 0.0 {
            format!("{:0.3}", -1.0f64)
        } else {
            format!("{:0.3}", val)
        }
    }

    /// Index of the "all" area range label, or 0 if not found.
    fn area_all_idx(&self) -> usize {
        self.params
            .area_rng_lbl
            .iter()
            .position(|l| l == "all")
            .unwrap_or(0)
    }

    /// Metric key names for the current evaluation mode.
    fn metric_keys(&self) -> &[&str] {
        if self.is_lvis {
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
        }
    }

    /// Per-category mean AP (averaged over all IoU thresholds and recall thresholds,
    /// at area="all" and the last max_dets setting). Returns one value per `params.cat_ids`
    /// entry; -1.0 for categories with no valid precision data.
    fn per_cat_ap(&self, eval: &AccumulatedEval) -> Vec<f64> {
        let a_idx = self.area_all_idx();
        let m_idx = eval.m - 1;
        (0..eval.k)
            .map(|k_idx| {
                let mut sum = 0.0;
                let mut count = 0_usize;
                for t_idx in 0..eval.t {
                    for r_idx in 0..eval.r {
                        let idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        let v = eval.precision[idx];
                        if v >= 0.0 {
                            sum += v;
                            count += 1;
                        }
                    }
                }
                if count == 0 {
                    -1.0
                } else {
                    sum / count as f64
                }
            })
            .collect()
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
    /// # Arguments
    ///
    /// * `prefix` — When `Some("val/bbox")`, keys become `"val/bbox/AP"` etc.
    ///   When `None`, keys are bare metric names (`"AP"`, `"AR100"`, …).
    /// * `per_class` — When `true` and [`accumulate`](COCOeval::accumulate) has been
    ///   run, adds per-category AP entries keyed as `"AP/{cat_name}"` (or
    ///   `"{prefix}/AP/{cat_name}"` with a prefix). Categories where all precision
    ///   values are −1 are skipped.
    ///
    /// # Metric keys
    ///
    /// For LVIS mode: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `APr`, `APc`, `APf`,
    /// `AR@300`, `ARs@300`, `ARm@300`, `ARl@300`.
    ///
    /// For standard COCO bbox/segm: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`,
    /// `AR1`, `AR10`, `AR100`, `ARs`, `ARm`, `ARl`.
    ///
    /// For keypoints: `AP`, `AP50`, `AP75`, `APm`, `APl`,
    /// `AR`, `AR50`, `AR75`, `ARm`, `ARl`.
    pub fn get_results(&self, prefix: Option<&str>, per_class: bool) -> HashMap<String, f64> {
        let stats = match &self.stats {
            Some(s) => s,
            None => return HashMap::new(),
        };

        let keys = self.metric_keys();

        let make_key = |metric: &str| -> String {
            match prefix {
                Some(p) => format!("{p}/{metric}"),
                None => metric.to_string(),
            }
        };

        let mut results: HashMap<String, f64> = keys
            .iter()
            .zip(stats.iter())
            .map(|(&k, &v)| (make_key(k), v))
            .collect();

        if per_class {
            if let Some(eval) = &self.eval {
                let per_cat = self.per_cat_ap(eval);
                for (ap, cat_id) in per_cat.iter().zip(self.params.cat_ids.iter()) {
                    if *ap >= 0.0 {
                        let cats = self.coco_gt.load_cats(&[*cat_id]);
                        if let Some(cat) = cats.first() {
                            results.insert(make_key(&format!("AP/{}", cat.name)), *ap);
                        }
                    }
                }
            }
        }

        results
    }

    /// Compute F-beta scores after `accumulate()`.
    ///
    /// Returns three metrics analogous to AP/AP50/AP75, but using max F-beta instead of
    /// mean precision. For each (IoU threshold, category), finds the recall operating point
    /// that maximises F-beta, then averages across categories.
    ///
    /// `beta` controls the precision/recall trade-off:
    /// - `beta = 1.0`  → F1 (harmonic mean, equal weight)
    /// - `beta < 1.0`  → weights precision more heavily
    /// - `beta > 1.0`  → weights recall more heavily
    ///
    /// Returns an empty map if `accumulate()` has not been run.
    pub fn f_scores(&self, beta: f64) -> HashMap<String, f64> {
        let eval = match &self.eval {
            Some(e) => e,
            None => return HashMap::new(),
        };

        let beta2 = beta * beta;
        let a_idx = self.area_all_idx();
        let m_idx = eval.m - 1;

        // Identify which IoU threshold indices correspond to 0.5 and 0.75.
        let mut is_t50 = vec![false; eval.t];
        let mut is_t75 = vec![false; eval.t];
        for (i, &thr) in self.params.iou_thrs.iter().enumerate() {
            if (thr - 0.5).abs() < 1e-9 {
                is_t50[i] = true;
            }
            if (thr - 0.75).abs() < 1e-9 {
                is_t75[i] = true;
            }
        }

        // Single pass: compute max-F-beta per (t_idx, k_idx), accumulate into three buckets.
        let mut sum_all = 0.0_f64;
        let mut count_all = 0_usize;
        let mut sum_50 = 0.0_f64;
        let mut count_50 = 0_usize;
        let mut sum_75 = 0.0_f64;
        let mut count_75 = 0_usize;

        for t_idx in 0..eval.t {
            for k_idx in 0..eval.k {
                let max_f = self
                    .params
                    .rec_thrs
                    .iter()
                    .enumerate()
                    .filter_map(|(r_idx, &r)| {
                        let p_idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        let p = eval.precision[p_idx];
                        if p < 0.0 {
                            return None;
                        }
                        let denom = beta2 * p + r;
                        if denom < f64::EPSILON {
                            return Some(0.0);
                        }
                        Some((1.0 + beta2) * p * r / denom)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);

                if max_f > f64::NEG_INFINITY {
                    sum_all += max_f;
                    count_all += 1;
                    if is_t50[t_idx] {
                        sum_50 += max_f;
                        count_50 += 1;
                    }
                    if is_t75[t_idx] {
                        sum_75 += max_f;
                        count_75 += 1;
                    }
                }
            }
        }

        let mean_or_neg1 = |sum: f64, count: usize| -> f64 {
            if count == 0 {
                -1.0
            } else {
                sum / count as f64
            }
        };

        let prefix = if (beta - 1.0).abs() < 1e-9 {
            "F1".to_string()
        } else {
            format!("F{beta}")
        };

        let mut results = HashMap::new();
        results.insert(prefix.clone(), mean_or_neg1(sum_all, count_all));
        results.insert(format!("{prefix}50"), mean_or_neg1(sum_50, count_50));
        results.insert(format!("{prefix}75"), mean_or_neg1(sum_75, count_75));
        results
    }

    /// Print a formatted results table to stdout.
    ///
    /// For LVIS, matches the lvis-api `print_results()` style (metric name + value per line).
    /// For standard COCO, equivalent to the output already printed by `summarize()`.
    /// Must be called after `summarize()`.
    pub fn print_results(&self) {
        let results = self.get_results(None, false);
        if results.is_empty() {
            eprintln!("No results to print. Run evaluate(), accumulate(), and summarize() first.");
            return;
        }

        let keys = self.metric_keys();

        for key in keys {
            let val = results.get(*key).copied().unwrap_or(-1.0);
            let val_str = Self::format_metric(val);
            println!(" {:>10} = {}", key, val_str);
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
    /// Compute a cross-category IoU matrix between DT and GT annotations.
    ///
    /// Returns `Vec<Vec<f64>>` of shape `[D × G]`. Falls back to bbox IoU for segm mode
    /// when RLEs cannot be produced for all annotations.
    fn cross_category_iou(
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
                    // Bbox fallback
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
            }
        }
    }

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
                        let ann_ids = coco_gt.get_ann_ids_for_img_cat(img_id, cat_id).to_vec();
                        ann_ids
                            .into_iter()
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
                        let ann_ids = coco_dt.get_ann_ids_for_img_cat(img_id, cat_id).to_vec();
                        ann_ids
                            .into_iter()
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

    /// Compute average precision from per-detection matched/ignored flags.
    ///
    /// Uses the same 101-point interpolation as [`accumulate`](COCOeval::accumulate).
    /// Returns `0.0` when `num_gt == 0` or there are no detections.
    fn compute_ap_from_matched(
        scores: &[f64],
        matched: &[bool],
        ignored: &[bool],
        num_gt: usize,
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

        let num_gt_f = num_gt as f64;
        let mut rc = vec![0.0f64; nd];
        let mut pr = vec![0.0f64; nd];
        for d in 0..nd {
            rc[d] = tp[d] / num_gt_f;
            let tot = tp[d] + fp[d];
            pr[d] = if tot > 0.0 { tp[d] / tot } else { 0.0 };
        }

        // Monotone decreasing from right to left (PASCAL VOC interpolation)
        for d in (0..nd.saturating_sub(1)).rev() {
            pr[d] = pr[d].max(pr[d + 1]);
        }

        // Average over 101 recall thresholds (unreachable thresholds contribute 0.0)
        let mut ap = 0.0f64;
        let mut rc_ptr = 0usize;
        for ri in 0..=100 {
            let rec_thr = ri as f64 / 100.0;
            while rc_ptr < nd && rc[rc_ptr] < rec_thr {
                rc_ptr += 1;
            }
            if rc_ptr < nd {
                ap += pr[rc_ptr];
            }
        }
        ap / 101.0
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
    pub fn tide_errors(&self, pos_thr: f64, bg_thr: f64) -> Result<TideErrors, String> {
        if self.eval_imgs.is_empty() {
            return Err("tide_errors() requires evaluate() to be called first".to_string());
        }

        let cat_ids = self.params.cat_ids.clone();
        let iou_type = self.params.iou_type;
        let target_area_rng = self.params.area_rng[0];
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
            .map(|(i, _)| i)
            .unwrap_or(0);

        let coco_gt = &self.coco_gt;
        let coco_dt = &self.coco_dt;

        // --- Cross-category IoU pass ---
        // For each image, compute max IoU between each DT annotation
        // and any GT annotation of a *different* category.
        let img_ids = self.params.img_ids.clone();

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

        // --- Process each eval_img at (target_area_rng, max_det) ---
        for eval_img_opt in &self.eval_imgs {
            let eval_img = match eval_img_opt {
                Some(e) => e,
                None => continue,
            };
            if eval_img.area_rng != target_area_rng || eval_img.max_det != max_det {
                continue;
            }

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
        for eval_img_opt in &self.eval_imgs {
            let eval_img = match eval_img_opt {
                Some(e) => e,
                None => continue,
            };
            if eval_img.area_rng != target_area_rng || eval_img.max_det != max_det {
                continue;
            }
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

        for &cat_id in &cat_ids {
            let data = match cat_data.get(&cat_id) {
                Some(d) if d.num_gt > 0 => d,
                _ => continue,
            };

            let baseline = Self::compute_ap_from_matched(
                &data.scores,
                &data.matched,
                &data.ignored,
                data.num_gt,
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
