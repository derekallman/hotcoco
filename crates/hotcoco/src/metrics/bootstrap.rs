//! Bootstrap confidence intervals over any resampled statistic.
//!
//! "Model B scores 1.2 AP higher" is not a result until you know whether it
//! survives resampling the evaluation set. [`bootstrap_ci`] answers that for any
//! statistic you can recompute on a subset of units — images for detection,
//! sequences for tracking — by taking the statistic itself as a closure.
//!
//! ```
//! use hotcoco::metrics::bootstrap::bootstrap_ci;
//!
//! let units: Vec<u32> = (0..100).collect();
//! // A toy statistic: the fraction of sampled units below 50.
//! let cis = bootstrap_ci(&units, 200, 0xC0C0, 0.95, |sample| {
//!     let hits = sample.iter().filter(|&&u| u < 50).count();
//!     vec![hits as f64 / sample.len() as f64]
//! });
//!
//! assert_eq!(cis.len(), 1);
//! assert!(cis[0].lower < 0.5 && cis[0].upper > 0.5);
//! ```
//!
//! Nothing here knows what a detection is. Detection's adapter is
//! [`compare`](crate::detection::compare), whose closure re-accumulates two
//! evaluators on the sampled images and returns their metric deltas.
//!
//! # Sampling convention
//!
//! Each sample draws `n_units` indices **with replacement**, then deduplicates
//! them into a set — so a sample holds roughly 63.2% of the units, not all of them.
//! That is deliberate and standard for detection: the accumulators treat a unit as
//! present or absent rather than weighted, so a repeated draw can't count twice.
//! Sampling is seeded per-sample (`seed + i`), which makes results reproducible and
//! order-independent under parallel execution.
//!
//! **This is therefore an m-of-n subsample, not the textbook bootstrap**, and the
//! intervals are not directly comparable to `scipy.stats.bootstrap` — which
//! resamples to full size with multiplicity. Two consequences worth stating:
//!
//! - Interval width is not the classical bootstrap's. Treat these as a spread
//!   estimate over which *images* were included, which is the question detection
//!   evaluation actually asks.
//! - "The CI contains the point estimate" is **not** guaranteed, and is not
//!   asserted as an invariant anywhere. Each sample sees ~63% of the data while the
//!   point estimate uses 100%, so a skewed statistic can legitimately fall outside.
//!   `lower <= upper` and `std_err >= 0` are the properties that do always hold.
//!
//! # Seed portability
//!
//! Reproducibility holds **within a build**. [`SmallRng`] is explicitly
//! non-portable: it is a different algorithm on 32-bit targets, and `rand`'s own
//! policy allows its output stream to change in any release. A seed does not pin a
//! result across a `rand` upgrade or a different pointer width — do not cite one in
//! a paper as if it did.

use std::collections::HashSet;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde::Serialize;

/// Bootstrap confidence interval for a single statistic.
#[derive(Debug, Clone, Serialize)]
pub struct BootstrapCI {
    /// Lower percentile bound.
    pub lower: f64,
    /// Upper percentile bound.
    pub upper: f64,
    /// Confidence level these bounds were computed at, for example 0.95.
    pub confidence: f64,
    /// Fraction of bootstrap samples in which the statistic was positive.
    ///
    /// For a delta between two models this reads as "how often B beat A" — often
    /// more useful than the interval, because it stays interpretable when the
    /// interval straddles zero.
    pub prob_positive: f64,
    /// Standard deviation across bootstrap samples.
    pub std_err: f64,
}

/// Percentile confidence intervals for a vector-valued statistic.
///
/// Draws `n_samples` bootstrap samples from `units` and calls `statistic` on each.
/// Every call must return a vector of the same length — one entry per quantity
/// being measured — and the result holds one [`BootstrapCI`] per entry, in the
/// same order.
///
/// `statistic` receives the sampled units themselves, deduplicated. Generic over
/// the unit type rather than handing back indices, which would make every caller
/// build a second set to map indices onto its own units.
///
/// Bounds are the `α/2` and `1 - α/2` quantiles of the samples, computed as
/// `numpy.quantile` computes them by default (`method="linear"`), so they match
/// it bit for bit on the same samples and sit symmetrically in the
/// distribution. No BCa correction — the intervals are readable as "the middle
/// 95% of what resampling produced", not as a bias-corrected estimator.
///
/// `statistic` is called from multiple threads, hence the `Sync` bound. Returns an
/// empty vector when `n_samples` is 0 or `units` is empty.
///
/// # Panics
///
/// If `confidence` is not strictly inside `(0, 1)`. Pass a fraction such as
/// `0.95`, not a percentage — `95` would otherwise clamp to the `[min, max]` of
/// the samples, a plausible-looking interval computed at the wrong level.
pub fn bootstrap_ci<T, F>(
    units: &[T],
    n_samples: usize,
    seed: u64,
    confidence: f64,
    statistic: F,
) -> Vec<BootstrapCI>
where
    T: Copy + Eq + std::hash::Hash + Sync,
    F: Fn(&HashSet<T>) -> Vec<f64> + Sync,
{
    assert!(
        confidence > 0.0 && confidence < 1.0,
        "bootstrap_ci: confidence must be in (0, 1), got {confidence} — \
         pass a fraction such as 0.95, not a percentage"
    );

    let n_units = units.len();
    if n_samples == 0 || n_units == 0 {
        return Vec::new();
    }

    let all_samples: Vec<Vec<f64>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            // Seeded per sample, so the draw does not depend on thread scheduling.
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            let sample: HashSet<T> = (0..n_units)
                .map(|_| units[rng.random_range(0..n_units)])
                .collect();
            statistic(&sample)
        })
        .collect();

    let num_stats = all_samples.first().map_or(0, Vec::len);
    let alpha = 1.0 - confidence;
    let nb = n_samples;

    (0..num_stats)
        .map(|m| {
            let mut samples: Vec<f64> = all_samples.iter().map(|s| s[m]).collect();
            // Total order: a NaN-producing statistic sorts deterministically
            // (positive NaN last) instead of handing std's sort a non-total
            // order, which may panic (Rust ≥ 1.81) or scramble the percentiles.
            samples.sort_by(f64::total_cmp);

            let lower = quantile_linear(&samples, alpha / 2.0);
            let upper = quantile_linear(&samples, 1.0 - alpha / 2.0);

            let pos_count = samples.iter().filter(|&&x| x > 0.0).count();
            let prob_positive = pos_count as f64 / nb as f64;

            let mean: f64 = samples.iter().sum::<f64>() / nb as f64;
            let variance = if nb > 1 {
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nb - 1) as f64
            } else {
                0.0
            };

            BootstrapCI {
                lower,
                upper,
                confidence,
                prob_positive,
                std_err: variance.sqrt(),
            }
        })
        .collect()
}

/// The `q`-quantile of non-empty `sorted`, as `numpy.quantile(..., method="linear")`
/// (numpy's default) computes it: virtual index `(n - 1) · q`, interpolated
/// between its two neighbors. The interpolation mirrors numpy's `_lerp`, which
/// works from the nearer endpoint, so results are bit-identical, not just close.
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len() - 1;
    let pos = last as f64 * q;
    if pos >= last as f64 {
        return sorted[last];
    }
    let i = pos.floor();
    let gamma = pos - i;
    let (a, b) = (sorted[i as usize], sorted[i as usize + 1]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Indices standing in for units, when the test only cares about the sampling.
    fn units(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    /// Sample mean of a fixed population — the interval must bracket the truth.
    #[test]
    fn interval_brackets_the_population_mean() {
        let values: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let truth = values.iter().sum::<f64>() / values.len() as f64;

        let cis = bootstrap_ci(&units(values.len()), 500, 42, 0.95, |sample| {
            let sum: f64 = sample.iter().map(|&i| values[i]).sum();
            vec![sum / sample.len() as f64]
        });

        assert_eq!(cis.len(), 1);
        assert!(
            cis[0].lower <= truth && truth <= cis[0].upper,
            "truth {truth} outside [{}, {}]",
            cis[0].lower,
            cis[0].upper
        );
    }

    /// The statistic sees the caller's own units, not positions into them.
    #[test]
    fn statistic_receives_the_units_themselves() {
        let image_ids: Vec<u64> = vec![101, 202, 303, 404];
        let cis = bootstrap_ci(&image_ids, 20, 1, 0.95, |sample| {
            assert!(sample.iter().all(|id| image_ids.contains(id)));
            vec![sample.len() as f64]
        });
        assert_eq!(cis.len(), 1);
    }

    #[test]
    fn same_seed_reproduces_the_same_interval() {
        let stat = |sample: &HashSet<usize>| vec![sample.len() as f64];
        let a = bootstrap_ci(&units(100), 50, 7, 0.9, stat);
        let b = bootstrap_ci(&units(100), 50, 7, 0.9, stat);
        assert_eq!(a[0].lower, b[0].lower);
        assert_eq!(a[0].upper, b[0].upper);
        assert_eq!(a[0].std_err, b[0].std_err);
    }

    #[test]
    fn different_seed_gives_a_different_draw() {
        let stat = |sample: &HashSet<usize>| vec![sample.len() as f64];
        let a = bootstrap_ci(&units(1000), 50, 1, 0.9, stat);
        let b = bootstrap_ci(&units(1000), 50, 2, 0.9, stat);
        // Asserting only that variance exists would pass with identical seeds.
        assert_ne!(a[0].std_err, b[0].std_err);
    }

    /// With-replacement draws deduplicated: ~1 - 1/e of units per sample.
    #[test]
    fn sample_holds_roughly_63_percent_of_units() {
        let cis = bootstrap_ci(&units(10_000), 20, 99, 0.95, |sample| {
            vec![sample.len() as f64 / 10_000.0]
        });
        assert!(
            cis[0].lower > 0.60 && cis[0].upper < 0.66,
            "expected ~0.632, got [{}, {}]",
            cis[0].lower,
            cis[0].upper
        );
    }

    #[test]
    fn always_positive_statistic_reports_probability_one() {
        let cis = bootstrap_ci(&units(50), 100, 3, 0.95, |_| vec![1.0]);
        assert_eq!(cis[0].prob_positive, 1.0);
        assert_eq!(cis[0].std_err, 0.0);
        assert_eq!(cis[0].lower, 1.0);
    }

    #[test]
    fn vector_statistic_yields_one_interval_per_entry() {
        let cis = bootstrap_ci(&units(50), 30, 5, 0.95, |s| {
            vec![s.len() as f64, -(s.len() as f64), 0.0]
        });
        assert_eq!(cis.len(), 3);
        assert_eq!(cis[1].prob_positive, 0.0);
        assert_eq!(cis[2].prob_positive, 0.0);
    }

    #[test]
    fn degenerate_inputs_return_empty_not_a_panic() {
        assert!(bootstrap_ci(&units(0), 10, 1, 0.95, |_| vec![1.0]).is_empty());
        assert!(bootstrap_ci(&units(10), 0, 1, 0.95, |_| vec![1.0]).is_empty());
    }

    /// `confidence = 95` (wrong unit) used to silently yield `[min, max]`.
    #[test]
    #[should_panic(expected = "confidence must be in (0, 1)")]
    fn percentage_confidence_panics_instead_of_min_max() {
        bootstrap_ci(&units(10), 10, 1, 95.0, |_| vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "confidence must be in (0, 1)")]
    fn confidence_bounds_are_exclusive() {
        bootstrap_ci(&units(10), 10, 1, 1.0, |_| vec![1.0]);
    }

    /// A statistic that produces NaN must not panic the percentile sort
    /// (std's sort panics on a non-total order since Rust 1.81).
    #[test]
    fn nan_statistic_does_not_panic_the_percentile_sort() {
        let cis = bootstrap_ci(&units(50), 40, 9, 0.95, |sample| {
            // NaN for some draws, finite for others.
            let n = sample.len() as f64;
            vec![if sample.len() % 3 == 0 { f64::NAN } else { n }]
        });
        assert_eq!(cis.len(), 1);
        // total_cmp puts positive NaN last, so the lower bound stays finite.
        assert!(cis[0].lower.is_finite());
    }

    /// Fixed sorted samples `sqrt(k + 0.5) * 1.37`, `k = 0..n`.
    fn fixed_sorted(n: usize) -> Vec<f64> {
        (0..n).map(|k| (k as f64 + 0.5).sqrt() * 1.37).collect()
    }

    /// Both bounds follow `numpy.quantile`'s default (`method="linear"`)
    /// exactly. The old indices, `floor(α/2·n)` and `ceil((1-α/2)·n)`, left
    /// one more sample below the lower bound than above the upper one. Expected
    /// values are `np.quantile(s, [(1-c)/2, 1-(1-c)/2])` from numpy 2.4.3.
    #[test]
    fn percentile_bounds_match_numpy_linear() {
        let cases: [(usize, f64, f64, f64); 4] = [
            (100, 0.95, 2.354675875759626, 13.494629160245948),
            (100, 0.9, 3.197598499238932, 13.321436123251699),
            (1000, 0.95, 6.914735711498051, 42.76781454282384),
            (1000, 0.9, 9.730835484433253, 42.21623350725108),
        ];
        for (n, confidence, lo, hi) in cases {
            let s = fixed_sorted(n);
            let alpha = 1.0 - confidence;
            assert_eq!(
                quantile_linear(&s, alpha / 2.0),
                lo,
                "n={n} c={confidence} lower"
            );
            assert_eq!(
                quantile_linear(&s, 1.0 - alpha / 2.0),
                hi,
                "n={n} c={confidence} upper"
            );
        }
    }

    /// Symmetric: reversing and negating the samples mirrors the interval.
    #[test]
    fn percentile_bounds_are_symmetric() {
        for n in [7, 100, 1000] {
            let s = fixed_sorted(n);
            let neg: Vec<f64> = s.iter().rev().map(|x| -x).collect();
            for q in [0.025, 0.05] {
                let lo = quantile_linear(&s, q);
                let hi_neg = quantile_linear(&neg, 1.0 - q);
                assert!((lo + hi_neg).abs() < 1e-12, "n={n} q={q}: {lo} vs {hi_neg}");
            }
        }
    }

    /// A single sample is its own interval (no out-of-bounds neighbor).
    #[test]
    fn percentile_of_one_sample() {
        assert_eq!(quantile_linear(&[3.0], 0.025), 3.0);
        assert_eq!(quantile_linear(&[3.0], 0.975), 3.0);
    }
}
