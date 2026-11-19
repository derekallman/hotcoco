//! Tracking metrics — HOTA, CLEAR, and Identity from one sequence's counts,
//! and their combination across sequences.
//!
//! The formulas are TrackEval's (`trackeval/metrics/hota.py`, `clear.py`,
//! `identity.py`, and `count.py` at commit `12c8791`), and so are the field
//! names, so a number reads the same here as in a TrackEval printout:
//!
//! ```text
//! DetA = TP / (TP + FN + FP)         HOTA = √(DetA × AssA), per threshold α
//! MOTA = (TP − FP − IDSW) / (TP + FN)
//! IDF1 = IDTP / (IDTP + ½ IDFP + ½ IDFN)
//! ```
//!
//! Which boxes match is the tracking driver's decision, through
//! [`primitives::assign::lsap`](crate::primitives::assign::lsap); nothing
//! here looks at a box. A sequence's result keeps everything a combination
//! needs, so combining is a function of results alone, which is what lets
//! two evaluators over different sequences merge.
//!
//! # Rounding follows the reference
//!
//! Float addition is not associative, so a sum is bit-identical to
//! TrackEval's only when it adds in TrackEval's order. Where numpy sums an
//! array, the order is numpy 1.23's (`metrics/sum.rs`); a sum over sequences
//! goes in the order given, and TrackEval's is by sequence name.

use serde::{Deserialize, Serialize};

use super::sum::{grid_sum, pairwise_sum};
use crate::primitives::sim::iou_from_counts;

/// How many localization thresholds HOTA sweeps.
pub const NUM_ALPHAS: usize = 19;

/// HOTA's localization thresholds, `0.05, 0.10, …, 0.95`.
///
/// TrackEval builds them with `np.arange(0.05, 0.99, 0.05)`, which under
/// numpy 1.23 is `0.05 + i × 0.05` rounded twice. (numpy 2 on arm64 fuses the
/// two into one rounding and differs in the last bit at four thresholds.)
pub fn hota_alphas() -> [f64; NUM_ALPHAS] {
    std::array::from_fn(|i| 0.05 + i as f64 * 0.05)
}

/// The mean of a per-threshold array, as `np.mean` takes it: TrackEval's
/// summary value for an HOTA field.
pub fn mean_over_alphas(values: &[f64; NUM_ALPHAS]) -> f64 {
    pairwise_sum(values) / NUM_ALPHAS as f64
}

/// `AssA`, `AssRe`, and `AssPr` at one threshold.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AssociationScores {
    /// Association accuracy: each match weighted by the IoU of its two tracks.
    pub ass_a: f64,
    /// Association recall: weighted by the share of the ground-truth track matched.
    pub ass_re: f64,
    /// Association precision: weighted by the share of the predicted track matched.
    pub ass_pr: f64,
}

/// HOTA's association scores at one threshold, from how often each pair of
/// tracks matched.
///
/// `matches` holds `(ground-truth track, predicted track, frames matched)`,
/// ascending by ground-truth track and then predicted track, and only the
/// pairs that matched in some frame. `gt_track_dets[g]` and
/// `tracker_track_dets[t]` are how many frames each track is in, and `tp` is
/// the threshold's true positives. As TrackEval, each pair's share is summed
/// over the whole grid of track pairs with numpy 1.23's whole-array sum and
/// divided by `max(1, tp)`.
///
/// # Panics
///
/// If a track index is out of range or `matches` is not in that order.
pub fn association_scores(
    matches: &[(usize, usize, u64)],
    gt_track_dets: &[u64],
    tracker_track_dets: &[u64],
    tp: u64,
) -> AssociationScores {
    let tracks = tracker_track_dets.len();
    let cells = gt_track_dets.len() * tracks;
    let cell = |&(g, t, _): &(usize, usize, u64)| {
        assert!(
            g < gt_track_dets.len() && t < tracks,
            "association_scores: track pair ({g}, {t}) is outside a {} × {tracks} grid",
            gt_track_dets.len()
        );
        g * tracks + t
    };
    // Each match's share, laid out where the dense grid would hold it.
    let shares = |share: &dyn Fn(u64, u64, u64) -> f64| -> Vec<(usize, f64)> {
        matches
            .iter()
            .map(|m| {
                let &(g, t, count) = m;
                let c = count as f64;
                (
                    cell(m),
                    c * share(count, gt_track_dets[g], tracker_track_dets[t]),
                )
            })
            .collect()
    };
    let per_tp = (tp as f64).max(1.0);
    let ass_a = shares(&|count, g, t| iou_from_counts(count, g, t));
    let ass_re = shares(&|count, g, _| count as f64 / (g as f64).max(1.0));
    let ass_pr = shares(&|count, _, t| count as f64 / (t as f64).max(1.0));
    AssociationScores {
        ass_a: grid_sum(cells, &ass_a) / per_tp,
        ass_re: grid_sum(cells, &ass_re) / per_tp,
        ass_pr: grid_sum(cells, &ass_pr) / per_tp,
    }
}

/// HOTA for one sequence or a combination of them: every per-threshold
/// field, and the three single numbers TrackEval reads off the first
/// threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HotaResult {
    #[serde(rename = "HOTA")]
    pub hota: [f64; NUM_ALPHAS],
    #[serde(rename = "DetA")]
    pub det_a: [f64; NUM_ALPHAS],
    #[serde(rename = "AssA")]
    pub ass_a: [f64; NUM_ALPHAS],
    #[serde(rename = "DetRe")]
    pub det_re: [f64; NUM_ALPHAS],
    #[serde(rename = "DetPr")]
    pub det_pr: [f64; NUM_ALPHAS],
    #[serde(rename = "AssRe")]
    pub ass_re: [f64; NUM_ALPHAS],
    #[serde(rename = "AssPr")]
    pub ass_pr: [f64; NUM_ALPHAS],
    #[serde(rename = "LocA")]
    pub loc_a: [f64; NUM_ALPHAS],
    #[serde(rename = "OWTA")]
    pub owta: [f64; NUM_ALPHAS],
    #[serde(rename = "HOTA_TP")]
    pub tp: [u64; NUM_ALPHAS],
    #[serde(rename = "HOTA_FN")]
    pub fn_: [u64; NUM_ALPHAS],
    #[serde(rename = "HOTA_FP")]
    pub fp: [u64; NUM_ALPHAS],
    /// `HOTA` at the first threshold, α = 0.05.
    #[serde(rename = "HOTA(0)")]
    pub hota_0: f64,
    /// `LocA` at the first threshold.
    #[serde(rename = "LocA(0)")]
    pub loc_a_0: f64,
    /// `HOTA(0) × LocA(0)`.
    #[serde(rename = "HOTALocA(0)")]
    pub hota_loc_a_0: f64,
}

impl HotaResult {
    /// One sequence's HOTA from what its matching counted at each threshold:
    /// TP, FN, and FP, the association scores, and the summed similarity of
    /// the matches, which gives `LocA = max(1e-10, sum) / max(1e-10, TP)`.
    ///
    /// A sequence with no ground truth or no predictions is this with no
    /// true positives: TrackEval's early return, `LocA` 1 and every score 0.
    pub fn from_counts(
        tp: [u64; NUM_ALPHAS],
        fn_: [u64; NUM_ALPHAS],
        fp: [u64; NUM_ALPHAS],
        association: [AssociationScores; NUM_ALPHAS],
        similarity_sum: [f64; NUM_ALPHAS],
    ) -> Self {
        let mut r = Self::zeros(tp, fn_, fp);
        for a in 0..NUM_ALPHAS {
            r.ass_a[a] = association[a].ass_a;
            r.ass_re[a] = association[a].ass_re;
            r.ass_pr[a] = association[a].ass_pr;
            r.loc_a[a] = loc_a(similarity_sum[a], tp[a]);
        }
        r.finish()
    }

    /// Combine sequences, as TrackEval's `combine_sequences`: TP, FN, and FP
    /// summed; `AssA`, `AssRe`, `AssPr`, and `LocA` averaged weighted by each
    /// sequence's TP; the rest recomputed. Sequences add in the order given.
    pub fn combine<'a>(results: impl IntoIterator<Item = &'a HotaResult>) -> Self {
        let mut r = Self::zeros([0; NUM_ALPHAS], [0; NUM_ALPHAS], [0; NUM_ALPHAS]);
        // TrackEval forms each weighted sum with Python's `sum`, from 0, in
        // sequence order: the four weighted fields start at 0.0 and hold the
        // sums until the division below.
        for s in results {
            for a in 0..NUM_ALPHAS {
                let w = s.tp[a] as f64;
                r.tp[a] += s.tp[a];
                r.fn_[a] += s.fn_[a];
                r.fp[a] += s.fp[a];
                r.ass_a[a] += s.ass_a[a] * w;
                r.ass_re[a] += s.ass_re[a] * w;
                r.ass_pr[a] += s.ass_pr[a] * w;
                r.loc_a[a] += s.loc_a[a] * w;
            }
        }
        for a in 0..NUM_ALPHAS {
            let per_tp = (r.tp[a] as f64).max(1.0);
            r.ass_a[a] /= per_tp;
            r.ass_re[a] /= per_tp;
            r.ass_pr[a] /= per_tp;
            r.loc_a[a] = loc_a(r.loc_a[a], r.tp[a]);
        }
        r.finish()
    }

    /// The per-threshold fields as named curves, in TrackEval's field order,
    /// counts as floats: what a renderer draws against [`hota_alphas`].
    pub fn curves(&self) -> [(&'static str, [f64; NUM_ALPHAS]); 12] {
        let counts = |c: &[u64; NUM_ALPHAS]| c.map(|v| v as f64);
        [
            ("HOTA", self.hota),
            ("DetA", self.det_a),
            ("AssA", self.ass_a),
            ("DetRe", self.det_re),
            ("DetPr", self.det_pr),
            ("AssRe", self.ass_re),
            ("AssPr", self.ass_pr),
            ("LocA", self.loc_a),
            ("OWTA", self.owta),
            ("HOTA_TP", counts(&self.tp)),
            ("HOTA_FN", counts(&self.fn_)),
            ("HOTA_FP", counts(&self.fp)),
        ]
    }

    fn zeros(tp: [u64; NUM_ALPHAS], fn_: [u64; NUM_ALPHAS], fp: [u64; NUM_ALPHAS]) -> Self {
        let z = [0.0; NUM_ALPHAS];
        HotaResult {
            hota: z,
            det_a: z,
            ass_a: z,
            det_re: z,
            det_pr: z,
            ass_re: z,
            ass_pr: z,
            loc_a: z,
            owta: z,
            tp,
            fn_,
            fp,
            hota_0: 0.0,
            loc_a_0: 0.0,
            hota_loc_a_0: 0.0,
        }
    }

    /// TrackEval's `_compute_final_fields`: the fields that follow from the
    /// counts and the association scores.
    fn finish(mut self) -> Self {
        for a in 0..NUM_ALPHAS {
            let tp = self.tp[a] as f64;
            self.det_re[a] = tp / ((self.tp[a] + self.fn_[a]) as f64).max(1.0);
            self.det_pr[a] = tp / ((self.tp[a] + self.fp[a]) as f64).max(1.0);
            self.det_a[a] = tp / ((self.tp[a] + self.fn_[a] + self.fp[a]) as f64).max(1.0);
            self.hota[a] = (self.det_a[a] * self.ass_a[a]).sqrt();
            self.owta[a] = (self.det_re[a] * self.ass_a[a]).sqrt();
        }
        self.hota_0 = self.hota[0];
        self.loc_a_0 = self.loc_a[0];
        self.hota_loc_a_0 = self.hota_0 * self.loc_a_0;
        self
    }
}

/// `max(1e-10, similarity_sum) / max(1e-10, tp)`, TrackEval's `LocA`, which
/// is 1 where nothing matched.
fn loc_a(similarity_sum: f64, tp: u64) -> f64 {
    similarity_sum.max(1e-10) / (tp as f64).max(1e-10)
}

/// The counts behind CLEAR, summed as they are across sequences.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ClearCounts {
    #[serde(rename = "CLR_TP")]
    pub tp: u64,
    #[serde(rename = "CLR_FN")]
    pub fn_: u64,
    #[serde(rename = "CLR_FP")]
    pub fp: u64,
    /// Identity switches.
    #[serde(rename = "IDSW")]
    pub idsw: u64,
    /// Mostly tracked ground-truth tracks.
    #[serde(rename = "MT")]
    pub mt: u64,
    /// Partly tracked.
    #[serde(rename = "PT")]
    pub pt: u64,
    /// Mostly lost.
    #[serde(rename = "ML")]
    pub ml: u64,
    /// Fragmentations.
    #[serde(rename = "Frag")]
    pub frag: u64,
    /// Frames, as TrackEval counts them for CLEAR: 0 for a sequence with no
    /// ground truth or no predictions.
    #[serde(rename = "CLR_Frames")]
    pub frames: u64,
    /// Summed similarity of the matches, the numerator of MOTP.
    #[serde(rename = "MOTP_sum")]
    pub motp_sum: f64,
}

impl ClearCounts {
    /// Add another sequence's counts into these.
    pub fn add(&mut self, other: &ClearCounts) {
        self.tp += other.tp;
        self.fn_ += other.fn_;
        self.fp += other.fp;
        self.idsw += other.idsw;
        self.mt += other.mt;
        self.pt += other.pt;
        self.ml += other.ml;
        self.frag += other.frag;
        self.frames += other.frames;
        self.motp_sum += other.motp_sum;
    }

    /// The CLEAR scores these counts give: TrackEval's `_compute_final_fields`.
    pub fn scores(&self) -> ClearResult {
        let tp = self.tp as f64;
        let fp = self.fp as f64;
        let idsw = self.idsw as f64;
        let gt_tracks = ((self.mt + self.ml + self.pt) as f64).max(1.0);
        let gt_dets = ((self.tp + self.fn_) as f64).max(1.0);
        let tp_minus_fp = (self.tp as i64 - self.fp as i64) as f64;
        // `np.log10(IDSW) if IDSW > 0 else IDSW`.
        let log_idsw = if self.idsw > 0 { idsw.log10() } else { idsw };
        ClearResult {
            counts: *self,
            mota: (self.tp as i64 - self.fp as i64 - self.idsw as i64) as f64 / gt_dets,
            motp: self.motp_sum / tp.max(1.0),
            moda: tp_minus_fp / gt_dets,
            clr_re: tp / gt_dets,
            clr_pr: tp / ((self.tp + self.fp) as f64).max(1.0),
            mtr: self.mt as f64 / gt_tracks,
            ptr: self.pt as f64 / gt_tracks,
            mlr: self.ml as f64 / gt_tracks,
            smota: (self.motp_sum - fp - idsw) / gt_dets,
            clr_f1: tp / (tp + 0.5 * self.fn_ as f64 + 0.5 * fp).max(1.0),
            fp_per_frame: fp / (self.frames as f64).max(1.0),
            motal: (tp_minus_fp - log_idsw) / gt_dets,
        }
    }
}

/// CLEAR for one sequence or a combination of them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ClearResult {
    #[serde(flatten)]
    pub counts: ClearCounts,
    #[serde(rename = "MOTA")]
    pub mota: f64,
    #[serde(rename = "MOTP")]
    pub motp: f64,
    #[serde(rename = "MODA")]
    pub moda: f64,
    #[serde(rename = "CLR_Re")]
    pub clr_re: f64,
    #[serde(rename = "CLR_Pr")]
    pub clr_pr: f64,
    #[serde(rename = "MTR")]
    pub mtr: f64,
    #[serde(rename = "PTR")]
    pub ptr: f64,
    #[serde(rename = "MLR")]
    pub mlr: f64,
    #[serde(rename = "sMOTA")]
    pub smota: f64,
    #[serde(rename = "CLR_F1")]
    pub clr_f1: f64,
    #[serde(rename = "FP_per_frame")]
    pub fp_per_frame: f64,
    #[serde(rename = "MOTAL")]
    pub motal: f64,
}

impl ClearResult {
    /// TrackEval's result for a sequence with no ground truth or no
    /// predictions, which it returns before computing any score: every box
    /// is a miss or a false positive, every ground-truth track is mostly
    /// lost, `MLR` is 1 — even with no tracks — and every other score is 0,
    /// `MOTA` included. A combination recomputes the scores from the counts,
    /// so only the sequence's own row shows this.
    pub fn unmatched(gt_dets: u64, gt_tracks: u64, tracker_dets: u64) -> Self {
        let counts = ClearCounts {
            fn_: gt_dets,
            fp: tracker_dets,
            ml: gt_tracks,
            ..ClearCounts::default()
        };
        ClearResult {
            counts,
            mota: 0.0,
            motp: 0.0,
            moda: 0.0,
            clr_re: 0.0,
            clr_pr: 0.0,
            mtr: 0.0,
            ptr: 0.0,
            mlr: 1.0,
            smota: 0.0,
            clr_f1: 0.0,
            fp_per_frame: 0.0,
            motal: 0.0,
        }
    }

    /// Combine sequences: the counts summed in the order given, the scores
    /// recomputed.
    pub fn combine<'a>(results: impl IntoIterator<Item = &'a ClearResult>) -> Self {
        let mut counts = ClearCounts::default();
        for r in results {
            counts.add(&r.counts);
        }
        counts.scores()
    }
}

/// MT, PT, and ML from each ground-truth track's frames: `matched[g]` of
/// `present[g]`. A track matched in more than 80% of its frames is mostly
/// tracked, in at least 20% partly tracked, and otherwise mostly lost, a
/// track in no frame included.
///
/// # Panics
///
/// If the two slices differ in length.
pub fn track_coverage(matched: &[u64], present: &[u64]) -> (u64, u64, u64) {
    assert_eq!(
        matched.len(),
        present.len(),
        "track_coverage: {} matched counts for {} tracks",
        matched.len(),
        present.len()
    );
    let (mut mt, mut at_least_pt) = (0, 0);
    for (&m, &p) in matched.iter().zip(present) {
        if p == 0 {
            continue;
        }
        let ratio = m as f64 / p as f64;
        mt += u64::from(ratio > 0.8);
        at_least_pt += u64::from(ratio >= 0.2);
    }
    let pt = at_least_pt - mt;
    (mt, pt, matched.len() as u64 - mt - pt)
}

/// Fragmentations from how many times each ground-truth track went from
/// untracked to tracked: every start after a track's first.
pub fn fragmentations(starts: &[u64]) -> u64 {
    starts.iter().map(|&s| s.saturating_sub(1)).sum()
}

/// The counts behind the Identity scores.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityCounts {
    #[serde(rename = "IDTP")]
    pub idtp: u64,
    #[serde(rename = "IDFN")]
    pub idfn: u64,
    #[serde(rename = "IDFP")]
    pub idfp: u64,
}

impl IdentityCounts {
    /// Add another sequence's counts into these.
    pub fn add(&mut self, other: &IdentityCounts) {
        self.idtp += other.idtp;
        self.idfn += other.idfn;
        self.idfp += other.idfp;
    }

    /// The Identity scores these counts give.
    pub fn scores(&self) -> IdentityResult {
        let idtp = self.idtp as f64;
        let idfn = self.idfn as f64;
        let idfp = self.idfp as f64;
        IdentityResult {
            counts: *self,
            idf1: idtp / (idtp + 0.5 * idfp + 0.5 * idfn).max(1.0),
            idr: idtp / (idtp + idfn).max(1.0),
            idp: idtp / (idtp + idfp).max(1.0),
        }
    }
}

/// Identity for one sequence or a combination of them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IdentityResult {
    #[serde(flatten)]
    pub counts: IdentityCounts,
    #[serde(rename = "IDF1")]
    pub idf1: f64,
    #[serde(rename = "IDR")]
    pub idr: f64,
    #[serde(rename = "IDP")]
    pub idp: f64,
}

impl IdentityResult {
    /// Combine sequences: the counts summed, the scores recomputed.
    pub fn combine<'a>(results: impl IntoIterator<Item = &'a IdentityResult>) -> Self {
        let mut counts = IdentityCounts::default();
        for r in results {
            counts.add(&r.counts);
        }
        counts.scores()
    }
}

/// TrackEval's `Count`: boxes and tracks on each side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CountResult {
    /// Predicted boxes.
    #[serde(rename = "Dets")]
    pub dets: u64,
    #[serde(rename = "GT_Dets")]
    pub gt_dets: u64,
    /// Predicted tracks.
    #[serde(rename = "IDs")]
    pub ids: u64,
    #[serde(rename = "GT_IDs")]
    pub gt_ids: u64,
    /// Frames, for one sequence; `None` for a combination, which TrackEval's
    /// `Count` does not total.
    #[serde(rename = "Frames", default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u64>,
}

impl CountResult {
    /// One sequence's counts.
    pub fn new(dets: u64, gt_dets: u64, ids: u64, gt_ids: u64, frames: u64) -> Self {
        CountResult {
            dets,
            gt_dets,
            ids,
            gt_ids,
            frames: Some(frames),
        }
    }

    /// Combine sequences: every count but `frames` summed.
    pub fn combine<'a>(results: impl IntoIterator<Item = &'a CountResult>) -> Self {
        let mut c = CountResult::default();
        for r in results {
            c.dets += r.dets;
            c.gt_dets += r.gt_dets;
            c.ids += r.ids;
            c.gt_ids += r.gt_ids;
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `np.arange(0.05, 0.99, 0.05)` under numpy 1.23.5, as bit patterns.
    #[test]
    fn alphas_are_numpy_1_23_arange() {
        let bits: Vec<u64> = hota_alphas().iter().map(|a| a.to_bits()).collect();
        assert_eq!(bits, NUMPY_1_23_ALPHAS);
        assert_eq!(hota_alphas()[9], 0.5);
    }

    const NUMPY_1_23_ALPHAS: [u64; NUM_ALPHAS] = [
        0x3fa999999999999a,
        0x3fb999999999999a,
        0x3fc3333333333334,
        0x3fc999999999999a,
        0x3fd0000000000000,
        0x3fd3333333333333,
        0x3fd6666666666667,
        0x3fd999999999999a,
        0x3fdccccccccccccd,
        0x3fe0000000000000,
        0x3fe199999999999a,
        0x3fe3333333333334,
        0x3fe4ccccccccccce,
        0x3fe6666666666667,
        0x3fe8000000000001,
        0x3fe999999999999a,
        0x3feb333333333334,
        0x3fecccccccccccce,
        0x3fee666666666667,
    ];

    #[test]
    fn clear_scores_follow_trackeval() {
        let c = ClearCounts {
            tp: 8,
            fn_: 2,
            fp: 3,
            idsw: 1,
            mt: 1,
            pt: 1,
            ml: 0,
            frag: 0,
            frames: 4,
            motp_sum: 6.0,
        };
        let s = c.scores();
        assert_eq!(s.mota, (8.0 - 3.0 - 1.0) / 10.0);
        assert_eq!(s.moda, 0.5);
        assert_eq!(s.motp, 0.75);
        assert_eq!(s.clr_re, 0.8);
        assert_eq!(s.clr_pr, 8.0 / 11.0);
        assert_eq!(s.smota, (6.0 - 3.0 - 1.0) / 10.0);
        assert_eq!(s.clr_f1, 8.0 / 10.5);
        assert_eq!(s.fp_per_frame, 0.75);
        // log10(1) is 0, so MOTAL is MODA here.
        assert_eq!(s.motal, 0.5);
        assert_eq!((s.mtr, s.ptr, s.mlr), (0.5, 0.5, 0.0));
        // With no identity switches MOTAL subtracts 0, not log10(0).
        let none = ClearCounts { idsw: 0, ..c }.scores();
        assert_eq!(none.motal, none.moda);
    }

    #[test]
    fn unmatched_clear_is_the_early_return() {
        let r = ClearResult::unmatched(5, 2, 0);
        assert_eq!((r.counts.fn_, r.counts.ml, r.counts.frames), (5, 2, 0));
        assert_eq!((r.mlr, r.mota), (1.0, 0.0));
        // The combination of one such sequence recomputes: MOTA stays 0,
        // MLR is 1 because both tracks are mostly lost.
        let c = ClearResult::combine([&r]);
        assert_eq!((c.mota, c.mlr), (0.0, 1.0));
        // No ground truth: MOTA is 0 in the row but -FP in a combination.
        let fp_only = ClearResult::unmatched(0, 0, 4);
        assert_eq!((fp_only.mota, fp_only.mlr), (0.0, 1.0));
        assert_eq!(ClearResult::combine([&fp_only]).mota, -4.0);
    }

    #[test]
    fn coverage_thresholds_are_strict_above_80_and_inclusive_at_20() {
        // ratios 0.9, 0.8, 0.2, 0.1, and a track in no frame
        let (mt, pt, ml) = track_coverage(&[9, 4, 1, 1, 0], &[10, 5, 5, 10, 0]);
        assert_eq!((mt, pt, ml), (1, 2, 2));
        assert_eq!(fragmentations(&[0, 1, 3, 2]), 3);
    }

    #[test]
    fn identity_scores() {
        let s = IdentityCounts {
            idtp: 6,
            idfn: 2,
            idfp: 4,
        }
        .scores();
        assert_eq!(s.idf1, 6.0 / 9.0);
        assert_eq!(s.idr, 0.75);
        assert_eq!(s.idp, 0.6);
        assert_eq!(IdentityCounts::default().scores().idf1, 0.0);
    }

    #[test]
    fn hota_without_matches_is_the_early_return() {
        let none = [AssociationScores::default(); NUM_ALPHAS];
        let r = HotaResult::from_counts(
            [0; NUM_ALPHAS],
            [7; NUM_ALPHAS],
            [0; NUM_ALPHAS],
            none,
            [0.0; NUM_ALPHAS],
        );
        assert_eq!(r.loc_a, [1.0; NUM_ALPHAS]);
        assert_eq!(r.hota, [0.0; NUM_ALPHAS]);
        assert_eq!((r.hota_0, r.loc_a_0, r.hota_loc_a_0), (0.0, 1.0, 0.0));
    }

    #[test]
    fn hota_combines_weighted_by_true_positives() {
        let ass = |v: f64| {
            [AssociationScores {
                ass_a: v,
                ass_re: v,
                ass_pr: v,
            }; NUM_ALPHAS]
        };
        let a = HotaResult::from_counts(
            [3; NUM_ALPHAS],
            [1; NUM_ALPHAS],
            [0; NUM_ALPHAS],
            ass(1.0),
            [3.0; NUM_ALPHAS],
        );
        let b = HotaResult::from_counts(
            [1; NUM_ALPHAS],
            [0; NUM_ALPHAS],
            [2; NUM_ALPHAS],
            ass(0.5),
            [0.5; NUM_ALPHAS],
        );
        let c = HotaResult::combine([&a, &b]);
        assert_eq!(c.tp[0], 4);
        assert_eq!(c.ass_a[0], (3.0 + 0.5) / 4.0);
        assert_eq!(c.loc_a[0], (3.0 + 0.5) / 4.0);
        assert_eq!(c.det_a[0], 4.0 / 7.0);
        assert_eq!(c.hota[0], (c.det_a[0] * c.ass_a[0]).sqrt());
    }

    /// Two tracks matched for 2 of their 4 frames each: AssA is
    /// `2 × (2 / 6) / 2`.
    #[test]
    fn association_from_a_sparse_grid() {
        let s = association_scores(&[(0, 1, 2)], &[4, 1], &[1, 4], 2);
        assert_eq!(s.ass_a, 2.0 * (2.0 / 6.0) / 2.0);
        assert_eq!(s.ass_re, 2.0 * 0.5 / 2.0);
        assert_eq!(s.ass_pr, 2.0 * 0.5 / 2.0);
        let empty = association_scores(&[], &[1], &[1], 0);
        assert_eq!(empty, AssociationScores::default());
    }

    #[test]
    #[should_panic(expected = "outside a")]
    fn association_rejects_a_pair_off_the_grid() {
        association_scores(&[(2, 0, 1)], &[1, 1], &[1], 1);
    }
}
