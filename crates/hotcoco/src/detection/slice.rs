use std::collections::{BTreeMap, HashMap, HashSet};

use super::COCOeval;
use super::accumulate::EvalGrouping;
use super::catalog::build_metric_defs;
use super::summarize::{accumulate_and_summarize, metric_delta, stats_to_map};

/// Metrics for a single evaluation slice.
#[derive(Debug, Clone)]
pub struct SliceResult {
    /// Slice name (or `"_overall"` for the full-dataset baseline).
    pub name: String,
    /// Number of images in this slice.
    pub num_images: usize,
    /// All summary metrics for this slice — AP, AP50, AR100, and the rest.
    pub metrics: BTreeMap<String, f64>,
    /// Per-metric delta vs the overall baseline. Empty for the `_overall` entry.
    pub delta: BTreeMap<String, f64>,
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
        if !self.evaluated() {
            return Err("evaluate() must be called before slice_by()".into());
        }

        if slices.contains_key("_overall") {
            return Err("'_overall' is a reserved slice name".into());
        }

        let metrics = build_metric_defs(&self.params, self.eval_mode);
        let metric_keys: Vec<&str> = metrics.iter().map(|m| m.name).collect();

        // One bucketing for the baseline and every slice — only the image filter
        // differs between them.
        let grouping = EvalGrouping::build(self);

        // Compute overall (no filter)
        let (_, overall_stats) = accumulate_and_summarize(&grouping, None, &metrics);
        let overall_metrics = stats_to_map(&metric_keys, &overall_stats);

        let overall = SliceResult {
            name: "_overall".to_string(),
            num_images: self.params.img_ids.len(),
            metrics: overall_metrics.clone(),
            delta: BTreeMap::new(),
        };

        // Compute each slice
        let mut slice_results = Vec::with_capacity(slices.len());
        for (name, img_ids) in &slices {
            let filter: HashSet<u64> = img_ids.iter().copied().collect();
            let num_images = filter.len();

            let (_, stats) = accumulate_and_summarize(&grouping, Some(&filter), &metrics);
            let slice_metrics = stats_to_map(&metric_keys, &stats);

            // Zipped against the two stat vectors, which are what `metric_keys`
            // is parallel to. Reading each key back out of the maps needed an
            // `unwrap_or(-1.0)` for a miss that cannot happen — and that
            // unreachable default is the crate's "not computed" sentinel, so a
            // lookup bug would have surfaced as a plausible-looking zero delta.
            let delta: BTreeMap<String, f64> = metric_keys
                .iter()
                .zip(overall_stats.iter().zip(stats.iter()))
                .map(|(&k, (&overall_val, &slice_val))| {
                    // Baseline first: the delta reads "slice minus overall".
                    (k.to_string(), metric_delta(overall_val, slice_val))
                })
                .collect();

            slice_results.push(SliceResult {
                name: name.clone(),
                num_images,
                metrics: slice_metrics,
                delta,
            });
        }

        Ok(SlicedResults {
            overall,
            slices: slice_results,
        })
    }
}
