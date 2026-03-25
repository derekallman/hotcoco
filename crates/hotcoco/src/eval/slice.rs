use std::collections::{HashMap, HashSet};

use super::accumulate::accumulate_impl;
use super::summarize::{build_metric_defs, summarize_impl};
use super::COCOeval;

/// Metrics for a single evaluation slice.
#[derive(Debug, Clone)]
pub struct SliceResult {
    /// Slice name (or `"_overall"` for the full-dataset baseline).
    pub name: String,
    /// Number of images in this slice.
    pub num_images: usize,
    /// All summary metrics (AP, AP50, AR100, etc.) for this slice.
    pub metrics: HashMap<String, f64>,
    /// Per-metric delta vs the overall baseline. Empty for the `_overall` entry.
    pub delta: HashMap<String, f64>,
}

/// Results across all slices plus the overall baseline.
#[derive(Debug, Clone)]
pub struct SlicedResults {
    /// Full-dataset metrics (used as the baseline for deltas).
    pub overall: SliceResult,
    /// One entry per user-provided slice.
    pub slices: Vec<SliceResult>,
}

impl COCOeval {
    /// Re-accumulate and summarize for each named image-ID subset.
    ///
    /// After calling [`evaluate`](COCOeval::evaluate), this method re-runs
    /// accumulation and summarization for each slice, computing all standard
    /// metrics and their deltas vs the full-dataset baseline. IoU computation
    /// is **not** repeated — only the lighter accumulate/summarize steps.
    ///
    /// The `"_overall"` key is reserved and must not appear in `slices`.
    pub fn slice_by(
        &self,
        slices: HashMap<String, Vec<u64>>,
    ) -> crate::error::Result<SlicedResults> {
        if self.eval_imgs.is_empty() {
            return Err("evaluate() must be called before slice_by()".into());
        }

        if slices.contains_key("_overall") {
            return Err("'_overall' is a reserved slice name".into());
        }

        let metrics = build_metric_defs(&self.params, self.eval_mode);
        let metric_keys: Vec<&str> = metrics.iter().map(|m| m.name).collect();

        // Compute overall (no filter)
        let overall_acc = accumulate_impl(&self.eval_imgs, &self.params, None);
        let overall_stats = summarize_impl(
            &overall_acc,
            &self.params,
            self.eval_mode,
            &self.freq_groups,
            &metrics,
        );

        let overall_metrics: HashMap<String, f64> = metric_keys
            .iter()
            .zip(overall_stats.iter())
            .map(|(&k, &v): (&&str, &f64)| (k.to_string(), v))
            .collect();

        let overall = SliceResult {
            name: "_overall".to_string(),
            num_images: self.params.img_ids.len(),
            metrics: overall_metrics.clone(),
            delta: HashMap::new(),
        };

        // Compute each slice
        let mut slice_results = Vec::with_capacity(slices.len());
        for (name, img_ids) in &slices {
            let filter: HashSet<u64> = img_ids.iter().copied().collect();
            let num_images = filter.len();

            let acc = accumulate_impl(&self.eval_imgs, &self.params, Some(&filter));
            let stats = summarize_impl(
                &acc,
                &self.params,
                self.eval_mode,
                &self.freq_groups,
                &metrics,
            );

            let metrics: HashMap<String, f64> = metric_keys
                .iter()
                .zip(stats.iter())
                .map(|(&k, &v): (&&str, &f64)| (k.to_string(), v))
                .collect();

            let delta: HashMap<String, f64> = metric_keys
                .iter()
                .map(|&k| {
                    let slice_val = metrics.get(k).copied().unwrap_or(-1.0);
                    let overall_val = overall_metrics.get(k).copied().unwrap_or(-1.0);
                    let d = if slice_val >= 0.0 && overall_val >= 0.0 {
                        slice_val - overall_val
                    } else {
                        0.0
                    };
                    (k.to_string(), d)
                })
                .collect();

            slice_results.push(SliceResult {
                name: name.clone(),
                num_images,
                metrics,
                delta,
            });
        }

        Ok(SlicedResults {
            overall,
            slices: slice_results,
        })
    }
}
