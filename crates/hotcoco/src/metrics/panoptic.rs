//! Panoptic quality — PQ, SQ, and RQ from per-category match counts.
//!
//! [`PqCounts`] is the accumulator: how many segments of a category matched,
//! were missed, or were spurious, plus the summed IoU of the matches. The
//! formulas from [Kirillov et al., CVPR 2019](https://arxiv.org/abs/1801.00868)
//! read straight off it:
//!
//! ```text
//! PQ = Σ IoU / (TP + ½ FP + ½ FN)        segmentation and recognition together
//! SQ = Σ IoU / TP                        mean IoU of the matched pairs
//! RQ = TP / (TP + ½ FP + ½ FN)           an F1 over segments
//! ```
//!
//! so `PQ = SQ × RQ`. Which segments count as matched, missed, or spurious is
//! [`primitives::panoptic`](crate::primitives::panoptic)'s decision; nothing
//! here looks at a pixel.
//!
//! The reference is panopticapi's `PQStat`, and the behavior that matters for
//! parity is which categories enter the average: one with no ground truth and
//! no predictions (`TP + FP + FN == 0`) is left out of the mean *and* the
//! count `n`, rather than averaged in as zero.

use serde::{Deserialize, Serialize};

use super::mean_or_missing;

/// Match counts for one category, summed over images.
///
/// The field names are panopticapi's, `fn` spelled `fn_` because it is a Rust
/// keyword.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PqCounts {
    /// Summed IoU of the matched pairs — the numerator of PQ and SQ.
    pub iou: f64,
    /// Matched ground-truth segments.
    pub tp: u64,
    /// Predicted segments that matched nothing and were not ignored.
    pub fp: u64,
    /// Ground-truth segments (not crowd) that nothing matched.
    pub fn_: u64,
}

impl PqCounts {
    /// Add another image's or category's counts into this one.
    pub fn add(&mut self, other: &PqCounts) {
        self.iou += other.iou;
        self.tp += other.tp;
        self.fp += other.fp;
        self.fn_ += other.fn_;
    }

    /// Whether anything was seen: panopticapi scores a category only when
    /// `tp + fp + fn > 0`, and leaves the rest out of every average.
    pub fn is_evaluable(&self) -> bool {
        self.tp + self.fp + self.fn_ > 0
    }

    /// PQ, SQ, RQ for this category, or the `-1.0` sentinel in all three
    /// when it is not evaluable — [`scores`](Self::scores) with the
    /// convention's spelling of "nothing to score" filled in.
    pub fn scores_or_missing(&self) -> PqScores {
        pq_average(std::iter::once(self)).0
    }

    /// PQ, SQ, RQ for this category, or `None` when it is not evaluable.
    pub fn scores(&self) -> Option<PqScores> {
        if !self.is_evaluable() {
            return None;
        }
        let denom = self.tp as f64 + 0.5 * self.fp as f64 + 0.5 * self.fn_ as f64;
        Some(PqScores {
            pq: self.iou / denom,
            sq: if self.tp == 0 {
                0.0
            } else {
                self.iou / self.tp as f64
            },
            rq: self.tp as f64 / denom,
        })
    }
}

/// The three panoptic numbers, each in `[0, 1]` — or all three the `-1.0`
/// "not computed" sentinel when an average had nothing to average.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PqScores {
    /// Panoptic quality.
    pub pq: f64,
    /// Segmentation quality.
    pub sq: f64,
    /// Recognition quality.
    pub rq: f64,
}

impl PqScores {
    /// Whether these are real numbers rather than the sentinel.
    pub fn is_computed(&self) -> bool {
        super::is_computed(self.pq)
    }
}

/// Mean PQ, SQ, and RQ over the evaluable categories, and how many there were.
///
/// panopticapi's `pq_average`: each of the three is averaged separately over
/// the categories with `tp + fp + fn > 0`, in the order given, and `n` is
/// their count. With no evaluable category the reference divides by zero;
/// this returns the `-1.0` sentinel in all three and `n == 0`, so a run with
/// no stuff categories reports "not computed" for the stuff split instead of
/// failing.
pub fn pq_average<'a>(counts: impl IntoIterator<Item = &'a PqCounts>) -> (PqScores, usize) {
    let (mut pq, mut sq, mut rq, mut n) = (0.0, 0.0, 0.0, 0usize);
    for scores in counts.into_iter().filter_map(PqCounts::scores) {
        pq += scores.pq;
        sq += scores.sq;
        rq += scores.rq;
        n += 1;
    }
    (
        PqScores {
            pq: mean_or_missing(pq, n),
            sq: mean_or_missing(sq, n),
            rq: mean_or_missing(rq, n),
        },
        n,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(iou: f64, tp: u64, fp: u64, fn_: u64) -> PqCounts {
        PqCounts { iou, tp, fp, fn_ }
    }

    #[test]
    fn formulas_match_the_paper() {
        let c = counts(1.6, 2, 1, 1);
        let s = c.scores().expect("evaluable");
        // denom = 2 + 0.5 + 0.5 = 3
        assert!((s.pq - 1.6 / 3.0).abs() < 1e-12);
        assert!((s.sq - 0.8).abs() < 1e-12);
        assert!((s.rq - 2.0 / 3.0).abs() < 1e-12);
        assert!((s.pq - s.sq * s.rq).abs() < 1e-12, "PQ = SQ × RQ");
    }

    #[test]
    fn no_true_positives_gives_zero_sq_not_nan() {
        let s = counts(0.0, 0, 2, 3).scores().expect("evaluable");
        assert_eq!(
            s,
            PqScores {
                pq: 0.0,
                sq: 0.0,
                rq: 0.0
            }
        );
    }

    #[test]
    fn empty_category_is_not_evaluable() {
        assert!(!PqCounts::default().is_evaluable());
        assert_eq!(PqCounts::default().scores(), None);
        assert!(!PqCounts::default().scores_or_missing().is_computed());
        let c = counts(1.0, 1, 0, 0);
        assert_eq!(Some(c.scores_or_missing()), c.scores());
    }

    #[test]
    fn average_skips_empty_categories_and_counts_the_rest() {
        let cats = [
            counts(1.0, 1, 0, 0),
            PqCounts::default(),
            counts(0.0, 0, 1, 0),
        ];
        let (s, n) = pq_average(&cats);
        assert_eq!(n, 2);
        assert!((s.pq - 0.5).abs() < 1e-12);
        assert!((s.sq - 0.5).abs() < 1e-12);
        assert!((s.rq - 0.5).abs() < 1e-12);
    }

    #[test]
    fn average_of_nothing_is_the_sentinel() {
        let (s, n) = pq_average(&[PqCounts::default()]);
        assert_eq!(n, 0);
        assert_eq!(
            s,
            PqScores {
                pq: -1.0,
                sq: -1.0,
                rq: -1.0
            }
        );
        assert!(!s.is_computed());
        let (s, n) = pq_average(&[]);
        assert_eq!((s.pq, n), (-1.0, 0));
    }

    #[test]
    fn add_sums_every_field() {
        let mut a = counts(0.5, 1, 2, 3);
        a.add(&counts(0.25, 4, 5, 6));
        assert_eq!(a, counts(0.75, 5, 7, 9));
    }
}
