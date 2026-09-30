use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use super::COCOeval;
use crate::metrics::calibration::{calibration_curve, calibration_error, scores_in_unit_interval};

// Private for the same reason as `BootstrapCI` in `compare.rs`.
use crate::metrics::calibration::CalibrationBin;

/// Confidence calibration analysis result.
///
/// Measures how well a model's predicted confidence scores align with actual
/// detection accuracy. A perfectly calibrated model produces detections at
/// confidence 0.8 that are correct 80% of the time.
///
/// Use [`COCOeval::calibration`] to compute. For the underlying math on arbitrary
/// `(score, correct)` arrays — no evaluator required — see
/// [`metrics::calibration`](crate::metrics::calibration).
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationResult {
    /// Expected Calibration Error — weighted mean of |accuracy - confidence| per bin.
    pub ece: f64,
    /// Maximum Calibration Error — worst per-bin |accuracy - confidence|.
    pub mce: f64,
    /// Per-bin breakdown.
    pub bins: Vec<CalibrationBin>,
    /// Per-category ECE, keyed by category ID.
    pub per_category: BTreeMap<u64, f64>,
    /// IoU threshold used to define "correct" (TP).
    pub iou_threshold: f64,
    /// Number of bins.
    pub n_bins: usize,
    /// Total number of (non-ignored) detections analyzed.
    pub num_detections: usize,
}

/// Confidences and outcomes as parallel arrays — the shape every metric function
/// in [`metrics`](crate::metrics) takes.
type ScoredOutcomes = (Vec<f64>, Vec<bool>);

impl COCOeval {
    /// Compute confidence calibration metrics.
    ///
    /// Requires [`evaluate`](COCOeval::evaluate) to have been called first.
    /// Iterates all per-image evaluation results and buckets detections by
    /// confidence score, computing accuracy (fraction of TPs) per bin.
    ///
    /// This is the detection adapter over
    /// [`metrics::calibration`](crate::metrics::calibration): it decides which
    /// detections count (the "all" area range, non-ignored, at `iou_threshold`)
    /// and the metric functions do the rest. To calibrate scores that did not come
    /// from a `COCOeval`, call those functions directly.
    ///
    /// # Arguments
    ///
    /// - `n_bins` — number of equal-width bins in [0, 1] (default 10)
    /// - `iou_threshold` — IoU threshold for TP/FP classification (default 0.5).
    ///   Must match one of `params.iou_thrs`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `evaluate()` has not been called, if `iou_threshold` is
    /// not found in `params.iou_thrs`, or if any detection score falls outside
    /// `[0, 1]`. The last is rejected rather than clamped: binning saturates an
    /// out-of-range score into an end bin while keeping its raw magnitude in the
    /// bin mean, so unnormalized scores would otherwise yield a calibration error
    /// above 1.0 with nothing to indicate why.
    pub fn calibration(
        &self,
        n_bins: usize,
        iou_threshold: f64,
    ) -> crate::error::Result<CalibrationResult> {
        if !self.evaluated() {
            return Err("calibration() requires evaluate() to be called first".into());
        }

        // Find the IoU threshold index
        let t_idx = self.params.iou_thr_idx(iou_threshold).ok_or_else(|| {
            format!(
                "iou_threshold={iou_threshold} not found in params.iou_thrs={:?}",
                self.params.iou_thrs
            )
        })?;

        // Collect detections globally and per-category
        let mut all: ScoredOutcomes = (Vec::new(), Vec::new());
        let mut per_cat: HashMap<u64, ScoredOutcomes> = HashMap::new();

        // `default_cells` owns the (area = "all", default max_det) predicate that
        // TIDE and per-image diagnostics also select on.
        for eval_img in self.default_cells() {
            let matched = eval_img.dt_matched.row(t_idx);
            let ignored = eval_img.dt_ignore.row(t_idx);
            debug_assert_eq!(matched.len(), eval_img.dt_scores.len());
            debug_assert_eq!(ignored.len(), eval_img.dt_scores.len());
            let n = matched
                .len()
                .min(ignored.len())
                .min(eval_img.dt_scores.len());

            // Resolved once per image rather than once per detection — the entry
            // is the same for every detection in an eval_img, and on COCO val that
            // is ~500K hash lookups collapsed to ~20K.
            let cat = per_cat.entry(eval_img.category_id).or_default();

            for d in 0..n {
                if ignored[d] {
                    continue;
                }
                let (score, correct) = (eval_img.dt_scores[d], matched[d]);
                cat.0.push(score);
                cat.1.push(correct);
                all.0.push(score);
                all.1.push(correct);
            }
        }

        // Scores must be confidences in [0, 1]. Detection scores arrive straight
        // from user JSON, so a model exporting logits lands here; failing loudly
        // beats a plausible-looking number nobody can interpret. Checking `all`
        // covers the per-category vectors too, since every detection is pushed to
        // both. The predicate itself belongs to `metrics::calibration` — it is a
        // property of the functions being called, not of detection.
        scores_in_unit_interval(&all.0).map_err(|e| format!("calibration(): {e}"))?;

        let bins = calibration_curve(&all.0, &all.1, n_bins);
        let (ece, mce) = calibration_error(&bins);

        let per_category: BTreeMap<u64, f64> = per_cat
            .iter()
            .map(|(&cat_id, (scores, matched))| {
                let cat_bins = calibration_curve(scores, matched, n_bins);
                (cat_id, calibration_error(&cat_bins).0)
            })
            .collect();

        Ok(CalibrationResult {
            ece,
            mce,
            bins,
            per_category,
            iou_threshold,
            n_bins,
            num_detections: all.0.len(),
        })
    }
}
