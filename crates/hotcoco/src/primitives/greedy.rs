//! COCO greedy matching — the detection matcher as a first-class, documented stage.
//!
//! This is pycocotools' per-image detection→ground-truth assignment, lifted out
//! of the eval monolith so the detection driver and its LVIS/OID variants share
//! one matcher. The match *choices* are public contract (surfaced through
//! `evalImgs` `dtMatches`/`gtMatches`), so the algorithm — score-descending
//! iteration, best-IoU selection, the two ignore phases, crowd re-match, and tie
//! behavior — is reproduced exactly.
//!
//! The caller (driver) owns everything COCO-specific: computing the ignore/crowd
//! flags, ordering detections and ground-truths, building the IoU matrix, and
//! translating the returned indices back into annotation ids. This module owns
//! only the assignment algorithm.
//!
//! # What this matcher is *not* for
//!
//! It looks like a general IoU-threshold matcher; it is not. It is the
//! **detection lineage** matcher (COCO/LVIS/OID/TIDE), and its score-descending
//! greediness is a pycocotools compatibility requirement, not an optimality
//! claim — it does not maximize total similarity.
//!
//! - **Tracking-lineage metrics** (CLEAR, Identity, HOTA) must use
//!   [`crate::primitives::assign::lsap`]: TrackEval solves an optimal assignment
//!   per frame, and a greedy approximation silently changes IDSW/IDF1.
//! - **Panoptic quality** needs no solver at all — an IoU > 0.5 match between
//!   non-overlapping segments is provably unique.
//!
//! Reaching for this matcher outside detection is the likeliest way for the two
//! lineages to collide.
//!
//! # Threshold-epsilon policy: caller-owned, with one canonical clamp
//!
//! pycocotools starts each detection's search at `min(t, 1 - 1e-10)`
//! (`evaluateImg`: `iou = min([t, 1-1e-10])`). That clamp is **the caller's to
//! apply**: [`greedy_match_masked`] compares against exactly the `iou_thrs` it is
//! handed, so a family whose thresholds mean something other than COCO's is not
//! silently given COCO's fudge factor. The clamp is inert below `t == 1.0` and
//! observable only there. Callers that claim pycocotools matching semantics
//! apply [`coco_match_floor`]; callers implementing a different reference do not:
//!
//! | Caller | Clamped? | Why |
//! |---|---|---|
//! | the detection `evaluate()` path (incl. the OID group-of pass) | **yes** | pycocotools drop-in parity; both matching phases in one `evaluate()` must share a floor or `t == 1.0` is internally incoherent |
//! | TIDE (`pos_thr`/`bg_thr`) | no | parity contract is *tidecv*, not pycocotools — clamping would diverge from that reference |
//! | confusion matrix, per-image diagnostics, calibration | no | hotcoco-native analysis with a user-chosen threshold; COCO's fudge factor is not implied |
//!
//! Identical geometry yields IoU within a few ulp of — but **not** exactly —
//! `1.0`: the intersection width `(x + w) - x` does not round-trip to `w`, and
//! pycocotools computes it the same way. A *clamped* caller therefore still
//! matches exact duplicates at `t == 1.0`; an unclamped caller comparing against
//! a raw `1.0` might not. For sub-pixel geometry the drift can fall below even the
//! clamped floor. Both regimes are pinned by
//! `sim::tests::bbox_iou_algebraic_properties` and
//! `sim::tests::self_iou_degrades_for_subpixel_boxes`.

/// pycocotools' match floor for an IoU threshold: `min(t, 1 - 1e-10)`.
///
/// The canonical definition of the clamp described in the [module
/// docs][self#threshold-epsilon-policy-caller-owned-with-one-canonical-clamp].
/// It exists so the detection lineage has **one** spelling of the epsilon rather
/// than a literal repeated at each call site; the matcher deliberately does
/// not apply it for you.
///
/// ```
/// # use hotcoco::primitives::greedy::coco_match_floor;
/// assert_eq!(coco_match_floor(0.5), 0.5);       // inert below 1.0
/// assert_eq!(coco_match_floor(1.0), 1.0 - 1e-10);
/// ```
#[inline]
pub fn coco_match_floor(iou_thr: f64) -> f64 {
    iou_thr.min(1.0 - 1e-10)
}

/// Index of the best-scoring eligible candidate at or above `floor`.
///
/// Ties go to the **earliest** index. That is not a stylistic choice: it is what
/// `numpy.argmax` does, and therefore what every reference implementation built on
/// numpy does. Rust's [`Iterator::max_by`] returns the *last* maximum, so the
/// obvious one-liner silently disagrees with the reference wherever two candidates
/// tie — and similarity measures tie constantly. Intersection-over-area saturates
/// at 1.0 for *any* fully-contained box, so a detection inside two overlapping
/// regions ties by construction rather than by coincidence.
///
/// The tie-break is observable public contract through `evalImgs`, so it lives
/// in one function rather than at each call site. `tests/test_parity_oid.py` catches the
/// failure: picking the later of two tied group-of boxes leaves the earlier one
/// permanently unmatched, turning a true positive into a miss.
///
/// Unlike [`greedy_match_masked`] this performs no assignment and no exclusion — it
/// answers "which candidate is best for this one row", and repeated calls may
/// return the same index. `eligible` masks out candidates the caller does not want
/// considered; it must be at least as long as `sims`.
///
/// **This module now hosts two opposite tie-break rules, deliberately.**
/// [`greedy_match_masked`] compares with `>=`, so the *last* tied ground truth wins —
/// that is pycocotools, and it is observable through `evalImgs`. This function
/// compares with `>`, so the *first* wins, which is numpy. Neither can adopt the
/// other without breaking parity with its own reference. Do not "unify" them.
///
/// ```
/// # use hotcoco::primitives::greedy::best_above_floor;
/// let sims = [0.9, 1.0, 1.0, 0.2];
/// let all = [true; 4];
/// assert_eq!(best_above_floor(&sims, &all, 0.5), Some(1)); // first of the tied maxima
/// assert_eq!(best_above_floor(&sims, &all, 1.5), None);    // nothing clears the floor
/// let mask = [true, false, true, true];
/// assert_eq!(best_above_floor(&sims, &mask, 0.5), Some(2)); // index 1 masked out
/// ```
pub fn best_above_floor(sims: &[f64], eligible: &[bool], floor: f64) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_sim = f64::NEG_INFINITY;
    for (i, &s) in sims.iter().enumerate() {
        // Strictly greater, so the first of any tied run wins.
        if eligible[i] && s >= floor && s > best_sim {
            best = Some(i);
            best_sim = s;
        }
    }
    best
}

/// Dense per-threshold matrix: `rows` thresholds × `row_len` items, one allocation.
///
/// Everything the matcher reports is T parallel answers to the same question,
/// one row per IoU threshold. As `Vec<Vec<T>>` that costs T heap allocations per
/// field per evaluated cell — five such fields in every `EvalImg` over ~150k
/// val2017 cells, which the profiler attributed the majority of bbox evaluation
/// to. One strided buffer keeps the `[t][i]` indexing and drops the cost.
///
/// Rows are addressed as `m.row(t)` / `m.row_mut(t)`, single cells as
/// `m[(t, i)]`.
#[derive(Debug, Clone)]
pub struct ThreshMatrix<T> {
    rows: usize,
    row_len: usize,
    data: Vec<T>,
}

impl<T: Clone> ThreshMatrix<T> {
    /// A `rows × row_len` matrix with every cell set to `fill`.
    pub fn new(rows: usize, row_len: usize, fill: T) -> Self {
        Self {
            rows,
            row_len,
            data: vec![fill; rows * row_len],
        }
    }

    /// A matrix whose every row is a copy of `row` — one allocation instead of
    /// one clone per threshold.
    pub fn repeat_row(rows: usize, row: &[T]) -> Self {
        let mut data = Vec::with_capacity(rows * row.len());
        for _ in 0..rows {
            data.extend_from_slice(row);
        }
        Self {
            rows,
            row_len: row.len(),
            data,
        }
    }
}

impl<T> ThreshMatrix<T> {
    /// Number of threshold rows.
    pub fn num_rows(&self) -> usize {
        self.rows
    }

    /// Items per row.
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// The row for threshold `t`.
    pub fn row(&self, t: usize) -> &[T] {
        &self.data[t * self.row_len..(t + 1) * self.row_len]
    }

    /// Mutable row for threshold `t`.
    pub fn row_mut(&mut self, t: usize) -> &mut [T] {
        &mut self.data[t * self.row_len..(t + 1) * self.row_len]
    }

    /// All rows in threshold order. Yields `num_rows()` slices even when rows
    /// are empty, so consumers emitting one list per threshold stay correct
    /// for cells with no detections or no ground truths.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> + '_ {
        (0..self.rows).map(move |t| self.row(t))
    }
}

impl<T> std::ops::Index<(usize, usize)> for ThreshMatrix<T> {
    type Output = T;
    fn index(&self, (t, i): (usize, usize)) -> &T {
        debug_assert!(t < self.rows && i < self.row_len);
        &self.data[t * self.row_len + i]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for ThreshMatrix<T> {
    fn index_mut(&mut self, (t, i): (usize, usize)) -> &mut T {
        debug_assert!(t < self.rows && i < self.row_len);
        &mut self.data[t * self.row_len + i]
    }
}

/// Per-threshold greedy match results, indexed `[T]` over IoU thresholds.
pub struct GreedyMatches {
    /// `[T][D]`: for each threshold and detection (caller's score-descending
    /// order), the matched ground-truth index (caller's GT order) or `None`.
    pub dt_gt: ThreshMatrix<Option<usize>>,
    /// `[T][G]`: whether each ground-truth was matched at each threshold.
    pub gt_matched: ThreshMatrix<bool>,
}

/// The two per-GT policy masks [`greedy_match_masked`] takes, as named fields.
///
/// They are both `Option<&[bool]>` of the same length, and as adjacent
/// positional parameters nothing stopped a caller from transposing them — a
/// silent semantics swap, not an error. Construct with named fields (or
/// `GtMasks::default()` for the uniform case) so the mix-up cannot compile.
///
/// `None` is the **uniform** case: no GT is rematchable, and every GT is
/// phase-2 eligible. Every non-crowd, non-OID caller would otherwise materialize
/// two `vec![_; g]` per matched cell — ~800k allocations per COCO `evaluate()`
/// to restate the default.
#[derive(Debug, Clone, Copy, Default)]
pub struct GtMasks<'a> {
    /// Per-GT: may this ground truth be matched by more than one detection?
    /// (COCO crowd regions absorb any number of detections.) `None` = no.
    pub rematchable: Option<&'a [bool]>,
    /// Per-GT: may the phase-2 (ignored-GT) scan consider this ground truth?
    /// (OID holds group-of GTs out for a separate driver pass.) `None` = yes.
    pub phase2_eligible: Option<&'a [bool]>,
}

/// Greedy-match detections to ground-truths, pycocotools-exact.
///
/// The canonical entry point; every caller in the crate uses this one.
///
/// # Caller ordering contract
/// - Detections are ordered **score-descending** — matching iterates in this order.
/// - Ground-truths are partitioned **non-ignored first**: indices
///   `[0, num_gt_not_ignored)` are non-ignored, `[num_gt_not_ignored, g)` ignored.
/// - `iou_flat` is row-major `[D*G]` (`iou_flat[di * g + gi]`) in that ordering.
///
/// # The two per-GT policy masks
///
/// See [`GtMasks`] for what each mask means and why `None` is the uniform case.
/// The scans are `T×D×G`, so `None` is resolved to an empty slice once here and
/// the defaults are supplied by the `get`, rather than branching on an `Option`
/// per probe.
///
/// The matrix is flat (single allocation) rather than `sim`'s nested `[D][G]`
/// because the matching loop is `T×D×G` and benefits from contiguous access.
/// This doesn't conflict with the `sim` kernels' `[D][G]` output: a matcher
/// always sits behind a reorder step (detections score-descending, GTs
/// non-ignored-first), and that step is where the reordered flat matrix is
/// produced — sim's raw output is never fed in directly.
///
/// # Algorithm (per IoU threshold, per detection in score order)
/// Phase 1 scans non-ignored GTs for the highest-IoU available match `>= thr`.
/// Only if phase 1 finds nothing does phase 2 scan the ignored GTs. A GT already
/// matched is skipped unless `masks.rematchable[gi]` (crowd GTs, which multiple
/// detections may match). Phase 2 additionally skips any GT with
/// `masks.phase2_eligible[gi] == false` — OID group-of boxes, for instance, matched in a separate
/// driver pass). Among equal IoUs the later GT index wins — matching
/// pycocotools' `>=` update rule, which is observable through `evalImgs`.
///
/// # Panics
///
/// If `iou_flat.len() != d * g`, if `num_gt_not_ignored > g`, or if a supplied
/// mask is not exactly length `g`. The mask length is asserted rather than
/// tolerated: a short mask would otherwise degrade to the uniform default from
/// its end onward, giving a hybrid policy no caller intended.
pub fn greedy_match_masked(
    iou_flat: &[f64],
    d: usize,
    g: usize,
    num_gt_not_ignored: usize,
    masks: GtMasks<'_>,
    iou_thrs: &[f64],
) -> GreedyMatches {
    assert_eq!(
        iou_flat.len(),
        d * g,
        "greedy_match_masked: iou_flat must be d*g row-major ({d}x{g} = {}, got {})",
        d * g,
        iou_flat.len()
    );
    assert!(
        num_gt_not_ignored <= g,
        "greedy_match_masked: num_gt_not_ignored ({num_gt_not_ignored}) exceeds g ({g})"
    );
    for (name, mask) in [
        ("rematchable", masks.rematchable),
        ("phase2_eligible", masks.phase2_eligible),
    ] {
        if let Some(m) = mask {
            assert_eq!(
                m.len(),
                g,
                "greedy_match_masked: {name} mask must have one entry per GT \
                 (got {}, expected g = {g})",
                m.len()
            );
        }
    }

    let t = iou_thrs.len();
    let mut dt_gt = ThreshMatrix::new(t, d, None);
    let mut gt_matched = ThreshMatrix::new(t, g, false);

    // `None` becomes the empty slice; the `unwrap_or` defaults below carry its
    // documented meaning (a supplied mask is asserted to length `g` above, so
    // the `get` default is only ever reached through the `None` case).
    let rematchable = masks.rematchable.unwrap_or(&[]);
    let phase2_eligible = masks.phase2_eligible.unwrap_or(&[]);

    for (ti, &iou_thr) in iou_thrs.iter().enumerate() {
        // One row borrow per threshold keeps the T×D×G inner scans on plain
        // slice indexing instead of paying the strided-index arithmetic and
        // bounds check on every probe.
        let dt_row = dt_gt.row_mut(ti);
        let gt_row = gt_matched.row_mut(ti);
        for (di, dt_slot) in dt_row.iter_mut().enumerate() {
            let base = di * g;
            let mut best_iou = iou_thr;
            let mut best_gi: Option<usize> = None;

            // Phase 1: non-ignored GTs — highest-IoU available match.
            for gi in 0..num_gt_not_ignored {
                if gt_row[gi] && !rematchable.get(gi).copied().unwrap_or(false) {
                    continue;
                }
                let iou_val = iou_flat[base + gi];
                if iou_val >= best_iou {
                    best_iou = iou_val;
                    best_gi = Some(gi);
                }
            }

            // Phase 2: ignored GTs — only if phase 1 found no match.
            if best_gi.is_none() {
                for gi in num_gt_not_ignored..g {
                    if !phase2_eligible.get(gi).copied().unwrap_or(true) {
                        continue;
                    }
                    if gt_row[gi] && !rematchable.get(gi).copied().unwrap_or(false) {
                        continue;
                    }
                    let iou_val = iou_flat[base + gi];
                    if iou_val >= best_iou {
                        best_iou = iou_val;
                        best_gi = Some(gi);
                    }
                }
            }

            if let Some(gi) = best_gi {
                *dt_slot = Some(gi);
                gt_row[gi] = true;
            }
        }
    }

    GreedyMatches { dt_gt, gt_matched }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Every property the matcher's contract guarantees, over random inputs.
    ///
    /// Each detection family reaches its numbers through this function, so a
    /// violation here is a wrong metric everywhere at once — and the fixtures
    /// above are all 2x1 and 2x2. Randomizing the shape, the crowd flags, the
    /// phase-2 mask and the threshold list is what exercises the interactions
    /// between them.
    ///
    /// The IoU grid deliberately mixes continuous values with a coarse
    /// quantized set: exact ties are where the `>=` update rule (later GT index
    /// wins) is observable, and they essentially never occur under pure
    /// continuous sampling.
    #[test]
    fn greedy_match_contract_random() {
        let mut rng = StdRng::seed_from_u64(0x6DEED1);

        for case in 0..5000 {
            let d = rng.random_range(1..=6);
            let g = rng.random_range(1..=6);
            let num_ni = rng.random_range(0..=g);

            let quantized = rng.random_bool(0.5);
            let iou_flat: Vec<f64> = (0..d * g)
                .map(|_| {
                    if quantized {
                        // 0.0, 0.25, 0.5, 0.75, 1.0 — collides constantly.
                        rng.random_range(0..=4) as f64 / 4.0
                    } else {
                        rng.random_range(0.0..=1.0)
                    }
                })
                .collect();

            let rematchable: Vec<bool> = (0..g).map(|_| rng.random_bool(0.25)).collect();
            let phase2: Vec<bool> = (0..g).map(|_| rng.random_bool(0.75)).collect();

            let mut thrs: Vec<f64> = (0..rng.random_range(1..=4))
                .map(|_| rng.random_range(0.0..=1.0))
                .collect();
            thrs.sort_by(f64::total_cmp);

            // Exercise the struct-taking form here and the positional wrapper in
            // `simple()` below, so both entry points stay covered.
            let m = greedy_match_masked(
                &iou_flat,
                d,
                g,
                num_ni,
                GtMasks {
                    rematchable: Some(&rematchable),
                    phase2_eligible: Some(&phase2),
                },
                &thrs,
            );
            let ctx = format!("case {case}: d={d} g={g} num_ni={num_ni} thrs={thrs:?}");

            assert_eq!(m.dt_gt.num_rows(), thrs.len(), "{ctx}");
            assert_eq!(m.gt_matched.num_rows(), thrs.len(), "{ctx}");

            for (ti, &thr) in thrs.iter().enumerate() {
                assert_eq!(m.dt_gt.row_len(), d, "{ctx}");
                assert_eq!(m.gt_matched.row_len(), g, "{ctx}");

                let mut claimed = vec![0usize; g];
                for di in 0..d {
                    let Some(gi) = m.dt_gt[(ti, di)] else {
                        continue;
                    };

                    assert!(gi < g, "{ctx}: gt index {gi} out of range");

                    // A recorded match must clear the bar it was matched at.
                    assert!(
                        iou_flat[di * g + gi] >= thr,
                        "{ctx}: dt {di} matched gt {gi} at IoU {} < {thr}",
                        iou_flat[di * g + gi]
                    );

                    // Phase 2 is the only route to an ignored GT, and it honors
                    // the eligibility mask.
                    if gi >= num_ni {
                        assert!(
                            phase2[gi],
                            "{ctx}: dt {di} matched phase-2-ineligible gt {gi}"
                        );
                    }

                    claimed[gi] += 1;
                }

                // Injectivity: a GT is claimed once, unless it is rematchable —
                // crowd regions, which absorb any number of detections.
                for gi in 0..g {
                    if claimed[gi] > 1 {
                        assert!(
                            rematchable[gi],
                            "{ctx}: gt {gi} claimed {} times but is not rematchable",
                            claimed[gi]
                        );
                    }
                    // The two outputs are one fact in two shapes. `detection`'s
                    // confusion adapter reads both halves specifically to avoid a
                    // second source of truth, which makes this a contract.
                    assert_eq!(
                        m.gt_matched[(ti, gi)],
                        claimed[gi] > 0,
                        "{ctx}: gt_matched[{gi}] disagrees with dt_gt"
                    );
                }
            }
        }
    }

    /// Raising the IoU threshold cannot increase the number of **true-positive
    /// eligible** matches — those to non-ignored ground truths. This is what
    /// underwrites AP@0.5 >= AP@0.75 downstream.
    ///
    /// The obvious stronger claim — that the *total* match count is monotone —
    /// is **false**, and the counterexample is instructive rather than exotic.
    /// Phase 1 is preferred over phase 2, so raising the threshold can evict a
    /// detection out of phase 1 and into phase 2, freeing the non-ignored GT it
    /// was holding for a later detection. With `num_gt_not_ignored = 1` and
    ///
    /// ```text
    ///        G0     G1     G2          thresholds 0.42 and 0.70
    ///   D0  0.50   0.75   0.25         phase2 eligible: G1, G2
    ///   D1  0.75   0.50   0.50
    ///   D2  0.00   0.25   0.75
    /// ```
    ///
    /// the low threshold matches 2 (D0->G0 blocks D1, which takes G2 and blocks
    /// D2) while the high threshold matches 3 (D0 cannot reach G0, so it takes
    /// G1, leaving G0 for D1 and G2 for D2). Both give **one** TP-eligible match,
    /// which is why the property has to be stated over that subset.
    ///
    /// Established empirically over 200k random cases rather than proved; the
    /// loop here is smaller so the suite stays fast.
    #[test]
    fn tp_eligible_matches_are_monotone_in_threshold() {
        let mut rng = StdRng::seed_from_u64(0xA11CE);

        for case in 0..20000 {
            let d = rng.random_range(1..=5);
            let g = rng.random_range(1..=5);
            let num_ni = rng.random_range(0..=g);
            let quantized = rng.random_bool(0.5);
            let iou: Vec<f64> = (0..d * g)
                .map(|_| {
                    if quantized {
                        rng.random_range(0..=4) as f64 / 4.0
                    } else {
                        rng.random_range(0.0..=1.0)
                    }
                })
                .collect();
            let rematchable: Vec<bool> = (0..g).map(|_| rng.random_bool(0.2)).collect();
            let phase2: Vec<bool> = (0..g).map(|_| rng.random_bool(0.8)).collect();

            let mut thrs: Vec<f64> = (0..2).map(|_| rng.random_range(0.0..=1.0)).collect();
            thrs.sort_by(f64::total_cmp);

            let m = greedy_match_masked(
                &iou,
                d,
                g,
                num_ni,
                GtMasks {
                    rematchable: Some(&rematchable),
                    phase2_eligible: Some(&phase2),
                },
                &thrs,
            );
            let tp_at = |ti: usize| {
                m.dt_gt
                    .row(ti)
                    .iter()
                    .flatten()
                    .filter(|&&gi| gi < num_ni)
                    .count()
            };

            assert!(
                tp_at(1) <= tp_at(0),
                "case {case}: raising the threshold {:?} -> {:?} grew TP-eligible \
                 matches {} -> {} (d={d} g={g} num_ni={num_ni}) iou={iou:?}",
                thrs[0],
                thrs[1],
                tp_at(0),
                tp_at(1),
            );
        }
    }

    // No crowd, all GTs eligible for phase 2 — the `None`/`None` uniform case.
    fn simple(iou_flat: &[f64], d: usize, g: usize, num_ni: usize, thrs: &[f64]) -> GreedyMatches {
        greedy_match_masked(iou_flat, d, g, num_ni, GtMasks::default(), thrs)
    }

    /// The `None` encodings must be *exactly* the uniform masks, not merely close
    /// to them — every non-crowd caller now takes the `None` path, so a drift here
    /// is a silent change to every metric at once.
    #[test]
    fn none_masks_equal_their_explicit_uniform_forms() {
        // gi0/gi1 non-ignored, gi2 ignored. D0 claims gi1; D1's best is that same
        // gi1, so it discriminates the `rematchable` default. D2 clears nothing in
        // phase 1 and reaches gi2 only in phase 2, so it discriminates the
        // `phase2_eligible` default.
        #[rustfmt::skip]
        let iou = [
            0.9, 0.95, 0.2,
            0.5, 0.99, 0.3,
            0.1, 0.10, 0.9,
        ];
        let (d, g, num_ni) = (3, 3, 2);
        let thrs = [0.5, 0.85];

        let implicit = greedy_match_masked(&iou, d, g, num_ni, GtMasks::default(), &thrs);
        let explicit = greedy_match_masked(
            &iou,
            d,
            g,
            num_ni,
            GtMasks {
                rematchable: Some(&vec![false; g]),
                phase2_eligible: Some(&vec![true; g]),
            },
            &thrs,
        );

        for ti in 0..thrs.len() {
            assert_eq!(implicit.dt_gt.row(ti), explicit.dt_gt.row(ti));
            assert_eq!(implicit.gt_matched.row(ti), explicit.gt_matched.row(ti));
        }
    }

    #[test]
    fn matches_highest_iou_above_threshold() {
        // 1 DT, 2 non-ignored GTs; GT1 has higher IoU.
        let m = simple(&[0.6, 0.9], 1, 2, 2, &[0.5]);
        assert_eq!(m.dt_gt[(0, 0)], Some(1));
        assert_eq!(m.gt_matched.row(0), &[false, true]);
    }

    #[test]
    fn below_threshold_is_no_match() {
        let m = simple(&[0.4, 0.49], 1, 2, 2, &[0.5]);
        assert_eq!(m.dt_gt[(0, 0)], None);
    }

    #[test]
    fn score_order_gives_earlier_dt_first_pick() {
        // 2 DTs (score-desc), 1 GT. DT0 (first) takes it; DT1 gets nothing.
        let m = simple(&[0.9, 0.8], 2, 1, 1, &[0.5]);
        assert_eq!(m.dt_gt[(0, 0)], Some(0));
        assert_eq!(m.dt_gt[(0, 1)], None);
    }

    #[test]
    fn phase1_preferred_over_better_ignored_gt() {
        // g=2: gi0 non-ignored (IoU 0.6), gi1 ignored (IoU 0.99). Phase 1 finds
        // gi0, so phase 2 never runs even though gi1 has higher IoU.
        let m = simple(&[0.6, 0.99], 1, 2, 1, &[0.5]);
        assert_eq!(m.dt_gt[(0, 0)], Some(0));
    }

    #[test]
    fn falls_back_to_ignored_gt_when_no_phase1_match() {
        // gi0 non-ignored but below threshold (0.4); gi1 ignored at 0.8.
        let m = simple(&[0.4, 0.8], 1, 2, 1, &[0.5]);
        assert_eq!(m.dt_gt[(0, 0)], Some(1));
    }

    #[test]
    fn crowd_gt_rematched_by_multiple_dts() {
        // 2 DTs, 1 ignored crowd GT (rematchable). Both DTs match it.
        let m = greedy_match_masked(
            &[0.9, 0.8],
            2,
            1,
            0,
            GtMasks {
                rematchable: Some(&[true]),
                phase2_eligible: None,
            },
            &[0.5],
        );
        assert_eq!(m.dt_gt[(0, 0)], Some(0));
        assert_eq!(m.dt_gt[(0, 1)], Some(0));
    }

    #[test]
    fn non_rematchable_gt_taken_only_once() {
        let m = greedy_match_masked(
            &[0.9, 0.8],
            2,
            1,
            0,
            GtMasks {
                rematchable: Some(&[false]),
                phase2_eligible: None,
            },
            &[0.5],
        );
        assert_eq!(m.dt_gt[(0, 0)], Some(0));
        assert_eq!(m.dt_gt[(0, 1)], None);
    }

    #[test]
    fn phase2_ineligible_gt_is_skipped() {
        // gi0 non-ignored below threshold; gi1 ignored at 0.9 but phase2-ineligible.
        let m = greedy_match_masked(
            &[0.4, 0.9],
            1,
            2,
            1,
            GtMasks {
                rematchable: None,
                phase2_eligible: Some(&[true, false]),
            },
            &[0.5],
        );
        assert_eq!(m.dt_gt[(0, 0)], None);
    }

    #[test]
    fn equal_iou_later_index_wins() {
        // Two non-ignored GTs with identical IoU; pycocotools' `>=` picks the last.
        let m = simple(&[0.7, 0.7], 1, 2, 2, &[0.5]);
        assert_eq!(m.dt_gt[(0, 0)], Some(1));
    }

    #[test]
    #[should_panic(expected = "iou_flat must be d*g")]
    fn wrong_iou_matrix_length_panics() {
        // 2x2 declared, 3 values supplied — one misaligned row of plausible IoUs.
        greedy_match_masked(&[0.9, 0.8, 0.7], 2, 2, 2, GtMasks::default(), &[0.5]);
    }

    #[test]
    #[should_panic(expected = "num_gt_not_ignored")]
    fn num_not_ignored_beyond_g_panics() {
        greedy_match_masked(&[0.9], 1, 1, 2, GtMasks::default(), &[0.5]);
    }

    /// A short mask used to silently degrade to the uniform default from its end
    /// onward in release builds; it is now asserted to exactly `g`.
    #[test]
    #[should_panic(expected = "rematchable mask")]
    fn short_rematchable_mask_panics() {
        greedy_match_masked(
            &[0.9, 0.8],
            1,
            2,
            2,
            GtMasks {
                rematchable: Some(&[true]),
                phase2_eligible: None,
            },
            &[0.5],
        );
    }

    #[test]
    #[should_panic(expected = "phase2_eligible mask")]
    fn overlong_phase2_mask_panics() {
        greedy_match_masked(
            &[0.9, 0.8],
            1,
            2,
            1,
            GtMasks {
                rematchable: None,
                phase2_eligible: Some(&[true, true, false]),
            },
            &[0.5],
        );
    }
}
