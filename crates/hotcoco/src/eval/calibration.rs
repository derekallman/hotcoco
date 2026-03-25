use std::collections::HashMap;

use serde::Serialize;

use super::COCOeval;

/// Single bin in a calibration analysis.
///
/// Each bin covers an equal-width interval of the `[0, 1]` confidence range.
/// After bucketing detections by confidence, `avg_confidence` and
/// `avg_accuracy` are the means within the bin.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationBin {
    /// Lower bound of the confidence interval (inclusive).
    pub bin_lower: f64,
    /// Upper bound of the confidence interval (exclusive, except last bin).
    pub bin_upper: f64,
    /// Mean predicted confidence of detections in this bin.
    pub avg_confidence: f64,
    /// Fraction of detections in this bin that are true positives.
    pub avg_accuracy: f64,
    /// Number of (non-ignored) detections in this bin.
    pub count: usize,
}

/// Confidence calibration analysis result.
///
/// Measures how well a model's predicted confidence scores align with actual
/// detection accuracy. A perfectly calibrated model produces detections at
/// confidence 0.8 that are correct 80% of the time.
///
/// Use [`COCOeval::calibration`] to compute.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationResult {
    /// Expected Calibration Error — weighted mean of |accuracy - confidence| per bin.
    pub ece: f64,
    /// Maximum Calibration Error — worst per-bin |accuracy - confidence|.
    pub mce: f64,
    /// Per-bin breakdown.
    pub bins: Vec<CalibrationBin>,
    /// Per-category ECE, keyed by category ID.
    pub per_category: HashMap<u64, f64>,
    /// IoU threshold used to define "correct" (TP).
    pub iou_threshold: f64,
    /// Number of bins.
    pub n_bins: usize,
    /// Total number of (non-ignored) detections analyzed.
    pub num_detections: usize,
}

/// A single detection's confidence and correctness for calibration.
#[derive(Clone, Copy)]
struct Detection {
    confidence: f64,
    correct: bool,
}

impl COCOeval {
    /// Compute confidence calibration metrics.
    ///
    /// Requires [`evaluate`](COCOeval::evaluate) to have been called first.
    /// Iterates all per-image evaluation results and buckets detections by
    /// confidence score, computing accuracy (fraction of TPs) per bin.
    ///
    /// # Arguments
    ///
    /// - `n_bins` — number of equal-width bins in [0, 1] (default 10)
    /// - `iou_threshold` — IoU threshold for TP/FP classification (default 0.5).
    ///   Must match one of `params.iou_thrs`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `evaluate()` has not been called or `iou_threshold`
    /// is not found in `params.iou_thrs`.
    pub fn calibration(
        &self,
        n_bins: usize,
        iou_threshold: f64,
    ) -> crate::error::Result<CalibrationResult> {
        if self.eval_imgs.is_empty() {
            return Err("calibration() requires evaluate() to be called first".into());
        }

        // Find the IoU threshold index
        let t_idx = self
            .params
            .iou_thrs
            .iter()
            .position(|&t| (t - iou_threshold).abs() < 1e-9)
            .ok_or_else(|| {
                format!(
                    "iou_threshold={iou_threshold} not found in params.iou_thrs={:?}",
                    self.params.iou_thrs
                )
            })?;

        // Use the "all" area range, matching standard COCO evaluation semantics.
        // Fallback to first area range if "all" label is absent (consistent with tide.rs).
        let target_area_rng = self.params.area_range_idx("all").unwrap_or(0);
        let target_area = self.params.area_ranges[target_area_rng].range;

        // Collect detections globally and per-category
        let mut all_dets: Vec<Detection> = Vec::new();
        let mut per_cat_dets: HashMap<u64, Vec<Detection>> = HashMap::new();

        for eval_img in self.eval_imgs.iter().flatten() {
            // Filter to "all" area range (evaluate() uses a single max_det for all entries)
            if eval_img.area_rng != target_area {
                continue;
            }

            let matched = &eval_img.dt_matched[t_idx];
            let ignored = &eval_img.dt_ignore[t_idx];
            debug_assert_eq!(matched.len(), eval_img.dt_scores.len());
            debug_assert_eq!(ignored.len(), eval_img.dt_scores.len());
            let n = matched
                .len()
                .min(ignored.len())
                .min(eval_img.dt_scores.len());

            for d in 0..n {
                if ignored[d] {
                    continue;
                }
                let det = Detection {
                    confidence: eval_img.dt_scores[d],
                    correct: matched[d],
                };
                per_cat_dets
                    .entry(eval_img.category_id)
                    .or_default()
                    .push(det);
                all_dets.push(det);
            }
        }

        let bins = compute_bins(&all_dets, n_bins);
        let (ece, mce) = compute_ece_mce(&bins, all_dets.len());
        let num_detections = all_dets.len();

        // Per-category ECE
        let per_category: HashMap<u64, f64> = per_cat_dets
            .iter()
            .map(|(&cat_id, dets)| {
                let cat_bins = compute_bins(dets, n_bins);
                let (cat_ece, _) = compute_ece_mce(&cat_bins, dets.len());
                (cat_id, cat_ece)
            })
            .collect();

        Ok(CalibrationResult {
            ece,
            mce,
            bins,
            per_category,
            iou_threshold,
            n_bins,
            num_detections,
        })
    }
}

fn compute_bins(dets: &[Detection], n_bins: usize) -> Vec<CalibrationBin> {
    let mut bins: Vec<CalibrationBin> = (0..n_bins)
        .map(|i| {
            let lower = i as f64 / n_bins as f64;
            let upper = (i + 1) as f64 / n_bins as f64;
            CalibrationBin {
                bin_lower: lower,
                bin_upper: upper,
                avg_confidence: 0.0,
                avg_accuracy: 0.0,
                count: 0,
            }
        })
        .collect();

    for det in dets {
        // Bin index: clamp to [0, n_bins-1]
        let idx = ((det.confidence * n_bins as f64) as usize).min(n_bins - 1);
        bins[idx].avg_confidence += det.confidence;
        bins[idx].avg_accuracy += if det.correct { 1.0 } else { 0.0 };
        bins[idx].count += 1;
    }

    for bin in &mut bins {
        if bin.count > 0 {
            let n = bin.count as f64;
            bin.avg_confidence /= n;
            bin.avg_accuracy /= n;
        }
    }

    bins
}

fn compute_ece_mce(bins: &[CalibrationBin], total: usize) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let mut ece = 0.0;
    let mut mce = 0.0f64;
    for bin in bins {
        if bin.count > 0 {
            let gap = (bin.avg_accuracy - bin.avg_confidence).abs();
            ece += (bin.count as f64 / total as f64) * gap;
            mce = mce.max(gap);
        }
    }
    (ece, mce)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bins_perfect_calibration() {
        // Detections where confidence == accuracy in each bin
        let dets: Vec<Detection> = (0..100)
            .map(|i| {
                let conf = (i as f64 + 0.5) / 100.0;
                Detection {
                    confidence: conf,
                    // For perfect calibration at bin level, each bin should have
                    // accuracy matching avg confidence. Use a simplified model.
                    correct: i % 2 == 0, // 50% correct overall
                }
            })
            .collect();

        let bins = compute_bins(&dets, 10);
        assert_eq!(bins.len(), 10);
        // Each bin should have 10 detections
        for bin in &bins {
            assert_eq!(bin.count, 10);
        }
    }

    #[test]
    fn test_ece_all_correct_high_confidence() {
        // All detections correct with confidence ~0.95
        let dets: Vec<Detection> = (0..100)
            .map(|_| Detection {
                confidence: 0.95,
                correct: true,
            })
            .collect();

        let bins = compute_bins(&dets, 10);
        let (ece, mce) = compute_ece_mce(&bins, dets.len());
        // accuracy = 1.0, confidence = 0.95, gap = 0.05
        assert!((ece - 0.05).abs() < 1e-9);
        assert!((mce - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_ece_empty() {
        let dets: Vec<Detection> = vec![];
        let bins = compute_bins(&dets, 10);
        let (ece, mce) = compute_ece_mce(&bins, 0);
        assert_eq!(ece, 0.0);
        assert_eq!(mce, 0.0);
    }

    #[test]
    fn test_ece_single_bin_overconfident() {
        // All detections at confidence 0.9 but only 50% correct
        let mut dets = Vec::new();
        for i in 0..100 {
            dets.push(Detection {
                confidence: 0.9,
                correct: i < 50,
            });
        }
        let bins = compute_bins(&dets, 10);
        let (ece, mce) = compute_ece_mce(&bins, dets.len());
        // gap = |0.5 - 0.9| = 0.4, weight = 1.0
        assert!((ece - 0.4).abs() < 1e-9);
        assert!((mce - 0.4).abs() < 1e-9);
    }
}
