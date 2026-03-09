use std::collections::HashMap;

use crate::coco::COCO;
use crate::params::Params;

/// Per-category confusion matrix for object detection.
///
/// Rows are ground truth categories, columns are predicted categories.
/// Index `num_cats` (the last row/column) represents "background" — unmatched GTs
/// (false negatives) land in the background column, unmatched DTs (false positives)
/// land in the background row.
///
/// Use [`super::COCOeval::confusion_matrix`] to compute this.
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
/// Produced by [`super::COCOeval::tide_errors`]. Each ΔAP value measures how much
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
    /// Detection matches for each IoU threshold: `dt_matches\[t\]\[d\]` = matched gt_id or 0
    pub dt_matches: Vec<Vec<u64>>,
    /// Ground truth matches for each IoU threshold: `gt_matches\[t\]\[g\]` = matched dt_id or 0
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

/// Array dimensions of an accumulated evaluation result.
///
/// Precision and scores have shape `[T x R x K x A x M]`;
/// recall has shape `[T x K x A x M]`.
#[derive(Debug, Clone, Copy)]
pub struct EvalShape {
    /// Number of IoU thresholds (T).
    pub t: usize,
    /// Number of recall thresholds (R).
    pub r: usize,
    /// Number of categories (K).
    pub k: usize,
    /// Number of area ranges (A).
    pub a: usize,
    /// Number of max-detection limits (M).
    pub m: usize,
}

impl EvalShape {
    /// Flat index into `precision` (or `scores`) for 5-D coordinates.
    pub fn precision_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        ((((t * self.r + r) * self.k + k) * self.a + a) * self.m) + m
    }

    /// Flat index into `recall` for 4-D coordinates.
    pub fn recall_idx(&self, t: usize, k: usize, a: usize, m: usize) -> usize {
        (((t * self.k + k) * self.a + a) * self.m) + m
    }
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
    /// Array dimensions — use to interpret the flat precision/recall/scores vectors.
    pub shape: EvalShape,
}

impl AccumulatedEval {
    /// Flat index into `precision` (or `scores`) for 5-D coordinates.
    pub fn precision_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        self.shape.precision_idx(t, r, k, a, m)
    }

    /// Flat index into `recall` for 4-D coordinates.
    pub fn recall_idx(&self, t: usize, k: usize, a: usize, m: usize) -> usize {
        self.shape.recall_idx(t, k, a, m)
    }
}

/// Read-only context shared across all [`COCOeval::evaluate_img_static`] calls
/// within a single [`COCOeval::evaluate`] invocation.
///
/// Grouping these four shared references avoids passing them individually to every
/// call and removes the `#[allow(clippy::too_many_arguments)]` suppressor.
pub(super) struct EvalImgContext<'a> {
    pub(super) coco_gt: &'a COCO,
    pub(super) coco_dt: &'a COCO,
    pub(super) params: &'a Params,
    pub(super) ious: &'a std::collections::HashMap<(u64, u64), Vec<Vec<f64>>>,
}
