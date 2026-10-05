//! The summary reduction: accumulated eval arrays + metric definitions -> numbers.
//!
//! This module computes and nothing else. The catalog of *which* metrics exist
//! is [`super::catalog`]; turning the resulting numbers into printed lines,
//! result maps, or DTOs is [`super::report`].

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

use crate::params::Params;

use super::EvalMode;
use super::accumulate::{AccumulatedEval, EvalGrouping, accumulate_impl};
use super::catalog::MetricDef;
use super::mode::FreqGroups;

/// Mean of `count` values summing to `sum`, or the `-1.0` "not computed" sentinel.
///
/// The sole producer of the sentinel documented on
/// [`metrics::is_computed`](crate::metrics::is_computed), which is how every
/// consumer reads it back.
pub(super) fn mean_or_missing(sum: f64, count: usize) -> f64 {
    if count == 0 { -1.0 } else { sum / count as f64 }
}

/// Mean of the values that were actually computed, or the `-1.0` sentinel.
///
/// [`mean_or_missing`]'s companion: that function owns what an empty mean *is*,
/// this one owns **which values are allowed into it** — averaging a `-1.0` in
/// treats "not computed" as a real score of minus one.
///
/// Sums with [`pairwise_sum`](crate::metrics::sum::pairwise_sum), numpy's order,
/// so a caller that visits values in the order numpy flattens the reference's
/// array gets the reference's mean to the last bit. The values are collected into
/// `scratch` because pairwise order depends on how many survive the filter;
/// callers that take many means pass one buffer for all of them.
pub(super) fn mean_of_valid(values: impl Iterator<Item = f64>, scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend(values.filter(|&v| crate::metrics::is_computed(v)));
    mean_or_missing(crate::metrics::sum::pairwise_sum(scratch), scratch.len())
}

/// B minus A, treating a metric missing from either side as no evidence.
///
/// The subtraction counterpart of [`mean_or_missing`]: `-1.0` is "not computed",
/// so subtracting it manufactures a swing of up to 1.0 out of missing data. The
/// comparison point estimate, its bootstrap CIs, and the per-slice deltas all
/// route through here so they agree on that.
#[inline]
pub(super) fn metric_delta(a: f64, b: f64) -> f64 {
    if a >= 0.0 && b >= 0.0 { b - a } else { 0.0 }
}

/// Metric names paired with their values, in catalog order.
///
/// `BTreeMap` rather than `HashMap`: these maps are serialized and iterated by
/// callers, and key order is part of what makes a saved comparison or slice
/// diffable.
pub(super) fn stats_to_map(metric_keys: &[&str], stats: &[f64]) -> BTreeMap<String, f64> {
    metric_keys
        .iter()
        .zip(stats.iter())
        .map(|(&k, &v)| (k.to_string(), v))
        .collect()
}

/// Re-accumulate an evaluated `COCOeval` over an image subset and summarize it.
///
/// The single entry point for re-summarization — `compare`, its bootstrap
/// statistic, and both halves of `slice_by`. `img_filter` of `None` means the
/// full dataset. Both halves of the result are returned because callers need
/// different parts: comparison reads the accumulated eval for per-category AP,
/// slicing and bootstrapping only the stats.
///
/// The evaluator arrives inside the [`EvalGrouping`] rather than beside it, so a
/// grouping bucketed under one evaluator's categories cannot be accumulated under
/// another's params.
pub(super) fn accumulate_and_summarize(
    grouping: &EvalGrouping<'_>,
    img_filter: Option<&HashSet<u64>>,
    metrics: &[MetricDef],
) -> (AccumulatedEval, Vec<f64>) {
    let ev = grouping.eval();
    let acc = accumulate_impl(grouping, img_filter);
    let stats = summarize_impl(
        &acc,
        &ev.params,
        ev.eval_mode,
        &ev.coco_gt.dataset.categories,
        metrics,
    );
    (acc, stats)
}

/// The AP samples over `t_indices × R × k_indices`, in the order numpy flattens
/// `precision[t, :, k, a, m]` — `k` fastest — with `-1.0` sentinels included.
///
/// Which integration applies is a property of the *mode*, not of the caller. COCO
/// and LVIS average the precision envelope over the 101 recall thresholds, so a
/// `(t, k)` cell yields `r` samples; Open Images takes the exact area under that
/// same envelope, so it yields one. Every AP path routes through here, so the two
/// integrations cannot split.
fn ap_samples<'a>(
    eval: &'a AccumulatedEval,
    eval_mode: EvalMode,
    t_indices: Range<usize>,
    k_indices: &'a [usize],
    a_idx: usize,
    m_idx: usize,
) -> impl Iterator<Item = f64> + 'a {
    let all_points = eval_mode == EvalMode::OpenImages;
    let n_r = if all_points { 1 } else { eval.shape.r };
    t_indices.flat_map(move |t_idx| {
        (0..n_r).flat_map(move |r_idx| {
            k_indices.iter().map(move |&k_idx| {
                if all_points {
                    eval.ap_all_points[eval.recall_idx(t_idx, k_idx, a_idx, m_idx)]
                } else {
                    eval.precision[eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx)]
                }
            })
        })
    })
}

/// Per-category mean AP — over every IoU and recall threshold, at area `"all"`
/// and the M slot holding the detection cap ([`Params::max_det_idx`], not the
/// last slot) — paired with its category id; `-1.0` for a category with no
/// valid precision.
///
/// Ids come from the K axis this accumulation used
/// ([`AccumulatedEval::cat_ids`]), paired here so no caller can zip the values
/// against another list. A `use_cats = false` run yields nothing: its one K slot
/// is the pool, not a category.
pub(super) fn per_cat_ap_static(
    eval: &AccumulatedEval,
    params: &Params,
    eval_mode: EvalMode,
) -> Vec<(u64, f64)> {
    let a_idx = params.all_area_idx();
    let m_idx = params.max_det_idx();
    let mut scratch = Vec::new();
    eval.cat_ids
        .iter()
        .enumerate()
        .map(|(k_idx, &cat_id)| {
            let k_one = [k_idx];
            let samples = ap_samples(eval, eval_mode, 0..eval.shape.t, &k_one, a_idx, m_idx);
            (cat_id, mean_of_valid(samples, &mut scratch))
        })
        .collect()
}

/// Pure computation of summary statistics from accumulated eval data.
///
/// Returns one `f64` per metric in the same order as the MetricDef vec for the
/// current evaluation mode.
///
/// `categories` are the ground truth's, read for their LVIS frequency tags and
/// bucketed over [`AccumulatedEval::cat_ids`] (see there for why not
/// `params.cat_ids`). Only an LVIS run builds the buckets; no other catalog has
/// a frequency-group metric, and this runs 2·n times inside a bootstrapped
/// `compare()`.
pub(super) fn summarize_impl(
    eval: &AccumulatedEval,
    params: &Params,
    eval_mode: EvalMode,
    categories: &[crate::types::Category],
    metrics: &[MetricDef],
) -> Vec<f64> {
    // One buffer for every mean below; see `mean_of_valid`. Sized for the
    // largest, AP over every IoU threshold and category.
    let mut scratch = Vec::with_capacity(eval.shape.t * eval.shape.r * eval.shape.k);
    let all_k: Vec<usize> = (0..eval.shape.k).collect();
    let freq_groups = if eval_mode == EvalMode::Lvis {
        FreqGroups::from_categories(categories, &eval.cat_ids)
    } else {
        FreqGroups::default()
    };
    // `k_indices` is every category, or for an LVIS frequency-group AP, the
    // categories in that bucket.
    let summarize_stat = |m: &MetricDef, k_indices: &[usize], scratch: &mut Vec<f64>| {
        // A missing area label or max-det setting degrades to the `-1.0` "not
        // computed" sentinel, like the missing-IoU-threshold branch below.
        // Falling back to index 0 would report the "all" slice under a per-size
        // metric's name — a plausible wrong number with nothing to flag it.
        let Some(a_idx) = params.area_range_idx(m.area_lbl) else {
            return -1.0;
        };
        let Some(m_idx) = params.max_dets.iter().position(|&d| d == m.max_det) else {
            return -1.0;
        };

        let t_indices = match m.iou_thr {
            // `Params::iou_thr_idx` owns this lookup: a single-threshold metric
            // like AP50 means exactly one slice of the IoU axis, never the
            // average of every threshold within tolerance.
            Some(thr) => params.iou_thr_idx(thr).map_or(0..0, |i| i..i + 1),
            None => 0..eval.shape.t,
        };

        // Visit order is the reference's flattening order: `s[s > -1]` over the
        // `[T, R, K]` precision slice or the `[T, K]` recall slice, `k` fastest,
        // with lvis-api's frequency-group slice `[T, R, K in bucket]` the same
        // way. With pairwise summation on top, every stat is `np.mean`'s to the
        // last bit.
        if m.ap {
            let samples = ap_samples(eval, eval_mode, t_indices, k_indices, a_idx, m_idx);
            mean_of_valid(samples, scratch)
        } else {
            let samples = t_indices.flat_map(|t_idx| {
                k_indices
                    .iter()
                    .map(move |&k_idx| eval.recall[eval.recall_idx(t_idx, k_idx, a_idx, m_idx)])
            });
            mean_of_valid(samples, scratch)
        }
    };

    metrics
        .iter()
        .map(|m| {
            let k_indices = m
                .freq_group
                .map_or(all_k.as_slice(), |fg| freq_groups.get(fg));
            summarize_stat(m, k_indices, &mut scratch)
        })
        .collect()
}
