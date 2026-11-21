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

use crate::error::Error;

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

/// The key the detection ranking sorts on: its integer order is the reverse of
/// `score`'s for every score but NaN, so keys sorted ascending put scores in
/// descending order, and a stable sort keeps tied scores in input order, as
/// pycocotools' mergesort does. `-0.0` and `0.0` compare equal as scores, so
/// they share a key. Otherwise the order is reversed [`f64::total_cmp`]:
/// positive `NaN` above every number, negative `NaN` below.
pub(crate) fn descending_score_key(score: f64) -> u64 {
    // `-0.0 + 0.0` is `0.0`.
    let bits = (score + 0.0).to_bits();
    // Ascending order: flip every bit of a negative, only the sign of the rest.
    let ascending = if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    };
    !ascending
}

/// The detection ranking: indices of `scores`, highest score first, tied
/// scores in input order — pycocotools' stable sort, and the order `COCOeval`
/// matches and accumulates in. [`greedy_match`] ranks its rows with this,
/// and a caller of [`greedy_match_masked`] ranks them with it first.
///
/// ```
/// # use hotcoco::primitives::greedy::score_order;
/// assert_eq!(score_order(&[0.5, 0.9, 0.5, -0.0, 0.0]), vec![1, 0, 2, 3, 4]);
/// ```
pub fn score_order(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by_key(|&i| descending_score_key(scores[i]));
    order
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

impl<T> Default for ThreshMatrix<T> {
    /// A 0 × 0 matrix with no allocation.
    fn default() -> Self {
        Self {
            rows: 0,
            row_len: 0,
            data: Vec::new(),
        }
    }
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
        let mut m = Self {
            rows: 0,
            row_len: 0,
            data: Vec::with_capacity(rows * row.len()),
        };
        m.reset_repeat_row(rows, row);
        m
    }

    /// [`new`](Self::new) in place, keeping the allocation.
    pub(crate) fn reset(&mut self, rows: usize, row_len: usize, fill: T) {
        self.rows = rows;
        self.row_len = row_len;
        self.data.clear();
        self.data.resize(rows * row_len, fill);
    }

    /// [`repeat_row`](Self::repeat_row) in place, keeping the allocation.
    pub(crate) fn reset_repeat_row(&mut self, rows: usize, row: &[T]) {
        self.rows = rows;
        self.row_len = row.len();
        self.data.clear();
        for _ in 0..rows {
            self.data.extend_from_slice(row);
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
#[derive(Debug, Clone, Default)]
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

/// Greedy-match detections to ground-truths, pycocotools-exact, on input
/// already in the matcher's order: the kernel behind [`greedy_match`], which
/// makes that order from input in any order. The detection driver calls the
/// same matcher through `greedy_match_into`, which reuses the result's
/// buffers.
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
    let mut out = GreedyMatches::default();
    greedy_match_into(
        iou_flat,
        d,
        g,
        num_gt_not_ignored,
        masks,
        iou_thrs,
        &mut out,
    );
    out
}

/// [`greedy_match_masked`] into `out`, reusing its buffers: the detection
/// driver matches one small cell after another, and allocating the result per
/// cell cost more than matching it.
pub(crate) fn greedy_match_into(
    iou_flat: &[f64],
    d: usize,
    g: usize,
    num_gt_not_ignored: usize,
    masks: GtMasks<'_>,
    iou_thrs: &[f64],
    out: &mut GreedyMatches,
) {
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
    let GreedyMatches { dt_gt, gt_matched } = out;
    dt_gt.reset(t, d, None);
    gt_matched.reset(t, g, false);

    // `None` becomes the empty slice; the `unwrap_or` defaults below carry its
    // documented meaning (a supplied mask is asserted to length `g` above, so
    // the `get` default is only ever reached through the `None` case).
    let rematchable = masks.rematchable.unwrap_or(&[]);
    let phase2_eligible = masks.phase2_eligible.unwrap_or(&[]);

    // How many ground truths a scan below can take: every non-ignored one,
    // and each ignored one phase 2 may consider. Once all are taken, no later
    // detection at that threshold can match, so the scan stops there. With
    // many detections and few ground truths, that skips most of the T×D×G
    // work. A rematchable ground truth is never used up, so one turns the
    // stop off (`None`).
    let takeable = (!rematchable.contains(&true)).then(|| {
        num_gt_not_ignored
            + (num_gt_not_ignored..g)
                .filter(|&gi| phase2_eligible.get(gi).copied().unwrap_or(true))
                .count()
    });

    for (ti, &iou_thr) in iou_thrs.iter().enumerate() {
        // One row borrow per threshold keeps the T×D×G inner scans on plain
        // slice indexing instead of paying the strided-index arithmetic and
        // bounds check on every probe.
        let dt_row = dt_gt.row_mut(ti);
        let gt_row = gt_matched.row_mut(ti);
        let mut open = takeable;
        for (di, dt_slot) in dt_row.iter_mut().enumerate() {
            if open == Some(0) {
                break;
            }
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
                // Without a rematchable ground truth, the scans skip a taken
                // one, so this match used up a ground truth that was open.
                if let Some(n) = &mut open {
                    *n -= 1;
                }
            }
        }
    }
}

/// The two per-GT masks [`match_group_of`] takes, as named fields: both are
/// `&[bool]` of length `g`, and as adjacent positional parameters nothing
/// would stop a caller from transposing them, as [`GtMasks`] says of its own.
#[derive(Debug, Clone, Copy)]
pub struct GroupOfBoxes<'a> {
    /// Per-GT: is this ground truth a group-of box?
    pub group_of: &'a [bool],
    /// Per-GT: does this ground truth count toward recall? A group-of box the
    /// area range ignores is taken but never credited.
    pub counted: &'a [bool],
}

/// Open Images' group-of pass, run after [`greedy_match_masked`] on the same
/// inputs with the group-of columns held out of both phases: in the ignored
/// tail, so phase 1 never sees them, and `phase2_eligible == false`.
///
/// The [Challenge protocol](https://storage.googleapis.com/openimages/web/evaluation.html):
///
/// > If at least one detection is inside group-of box a single True Positive
/// > is scored. ... Multiple correct detections inside the same group-of box
/// > is still count as a single True Positive. Otherwise, the group-of box is
/// > counted as a single False Negative.
///
/// So a group-of box is worth exactly one ground truth. Per threshold, each
/// detection `out` left unmatched takes the group-of column it overlaps most
/// at or above the threshold — [`best_above_floor`], the reference's
/// `argmax` then test, first of a tie. Detections are score-descending, so
/// the first to take a column is its highest-scoring detection, the one the
/// reference keeps with `scores_group_of[gt] = max(...)`: it is **credited**
/// (`credited[(t, di)]`), and the column becomes matched. A later taker is
/// matched but not credited — ignored, neither true nor false positive — and
/// a column nobody takes stays unmatched: a miss. This is TensorFlow's
/// `group_of_weight = 1.0`, the metric FiftyOne implements unconditionally.
///
/// "Inside" is IoA — intersection over the *detection's* area — the measure
/// COCO uses for a crowd region, so the caller computes a group-of column of
/// `iou_flat` the way it computes a crowd column. The hold-out from the
/// greedy phases is load-bearing: an IoA column saturates at 1.0 for any
/// detection inside the region, so a group-of box left in phase 1 would
/// outbid the real object a detection sits on and turn it into a miss.
///
/// `boxes.counted` says whether each ground truth counts toward recall. A
/// group-of box the area range ignores is still taken, so the detections
/// inside it are neither true nor false positives, but none is credited: a
/// true positive on a box the denominator leaves out would inflate the range,
/// as a detection matched to any other ignored ground truth is ignored under
/// COCO.
///
/// Columns with no group-of flag are never touched; a cell without one is a
/// reset of `credited` and nothing else, which is the common case.
///
/// # Panics
///
/// If `iou_flat.len() != d * g`, if either mask in `boxes` is not of length
/// `g`, or if `out` is not the result of matching these `d` detections at
/// these `iou_thrs`.
pub fn match_group_of(
    iou_flat: &[f64],
    d: usize,
    g: usize,
    boxes: GroupOfBoxes<'_>,
    iou_thrs: &[f64],
    out: &mut GreedyMatches,
    credited: &mut ThreshMatrix<bool>,
) {
    let GroupOfBoxes { group_of, counted } = boxes;
    assert_eq!(
        iou_flat.len(),
        d * g,
        "match_group_of: iou_flat must be d*g row-major ({d}x{g} = {}, got {})",
        d * g,
        iou_flat.len()
    );
    for (name, mask) in [("group_of", group_of), ("counted", counted)] {
        assert_eq!(
            mask.len(),
            g,
            "match_group_of: {name} must have one entry per GT (got {}, expected g = {g})",
            mask.len()
        );
    }
    let t = iou_thrs.len();
    assert!(
        out.dt_gt.num_rows() == t && out.dt_gt.row_len() == d && out.gt_matched.row_len() == g,
        "match_group_of: out must be the greedy result for these detections and thresholds"
    );
    credited.reset(t, d, false);
    if !group_of.contains(&true) {
        return;
    }
    for (ti, &floor) in iou_thrs.iter().enumerate() {
        let dt_row = out.dt_gt.row_mut(ti);
        let gt_row = out.gt_matched.row_mut(ti);
        let credit_row = credited.row_mut(ti);
        for (di, slot) in dt_row.iter_mut().enumerate() {
            if slot.is_some() {
                continue;
            }
            let row = &iou_flat[di * g..(di + 1) * g];
            let Some(gi) = best_above_floor(row, group_of, floor) else {
                continue;
            };
            *slot = Some(gi);
            // `gt_matched` is the "already credited" flag: the greedy phases
            // never match a group-of column, so only this pass sets it.
            credit_row[di] = counted[gi] && !gt_row[gi];
            gt_row[gi] = true;
        }
    }
}

/// The arrays [`greedy_match`] matches, each in the caller's own order.
#[derive(Debug, Clone, Copy)]
pub struct GreedyInput<'a> {
    /// The similarity of each detection to each ground truth, `num_dt ×
    /// num_gt`, row-major: `sim[d * num_gt + g]`.
    pub sim: &'a [f64],
    /// The detections: `sim`'s rows.
    pub num_dt: usize,
    /// The ground truths: `sim`'s columns.
    pub num_gt: usize,
    /// One per detection. Detections take turns from the highest score down,
    /// tied scores in row order, as [`score_order`] ranks them.
    pub scores: &'a [f64],
    /// The thresholds, one row of the result each, each matched at its
    /// [`coco_match_floor`].
    pub iou_thrs: &'a [f64],
    /// One per ground truth, all false when `None`. An ignored ground truth is
    /// a candidate only when no other one clears the threshold, and the
    /// detection that matches it is ignored too.
    pub gt_ignore: Option<&'a [bool]>,
    /// One per ground truth, all false when `None`: a crowd region, ignored,
    /// which any number of detections may match.
    pub iscrowd: Option<&'a [bool]>,
    /// One per ground truth, all false when `None`: an Open Images group-of
    /// box, held out of the greedy phases and matched in its own pass after
    /// them ([`match_group_of`]).
    pub group_of: Option<&'a [bool]>,
    /// One per detection, all false when `None`: whether a detection that
    /// matches nothing is ignored, as `COCOeval` seeds `dtIgnore` from the
    /// area range.
    pub dt_ignore: Option<&'a [bool]>,
}

/// [`greedy_match`]'s result: one row per threshold, and in each row one
/// entry per detection or ground truth in the caller's order.
#[derive(Debug, Clone)]
pub struct GreedyOutput {
    /// The ground truth each detection matched, or `None`.
    pub dt_matches: ThreshMatrix<Option<usize>>,
    /// The detection each ground truth kept, or `None`. A crowd region keeps
    /// the last detection to match it, and a group-of box the one credited
    /// with it.
    pub gt_matches: ThreshMatrix<Option<usize>>,
    /// Whether each detection is ignored: matched to an ignored ground truth
    /// (a group-of box's credited detection excepted), or unmatched and
    /// flagged in `dt_ignore`.
    pub dt_ignore: ThreshMatrix<bool>,
}

/// COCO's greedy matching on input in any order, pycocotools-exact: the
/// matcher behind `COCOeval.evaluate()` on arrays the caller supplies, and the
/// Python `primitives.greedy_match`.
///
/// It makes the order [`greedy_match_masked`] takes the way `COCOeval` makes
/// it: detections ranked by [`score_order`], and ground truths stably
/// partitioned with the ignored ones (`gt_ignore`, `iscrowd`, or `group_of`)
/// last. Then it runs that kernel, and [`match_group_of`] when any ground
/// truth is a group-of box, and maps the result back to the caller's order.
///
/// # Errors
///
/// `InvalidInput` when `sim` is not `num_dt × num_gt`, when `scores` or a flag
/// slice has the wrong length, or when `sim`, `scores`, or `iou_thrs` holds
/// NaN, which leaves the matching undefined.
pub fn greedy_match(input: &GreedyInput<'_>) -> crate::error::Result<GreedyOutput> {
    let GreedyInput {
        sim,
        num_dt: d,
        num_gt: g,
        scores,
        iou_thrs,
        ..
    } = *input;
    if d.checked_mul(g) != Some(sim.len()) {
        return Err(Error::InvalidInput(format!(
            "sim has {} entries, not {d} detections × {g} ground truths",
            sim.len()
        )));
    }
    check_no_nan(sim, "sim", "the matching")?;
    check_same_length(scores.len(), d, "scores", "sim's rows")?;
    check_no_nan(scores, "scores", "the matching order")?;
    check_no_nan(iou_thrs, "iou_thrs", "the matching")?;
    // A given column is borrowed; only a missing one is filled.
    fn flags<'a>(
        given: Option<&'a [bool]>,
        name: &str,
        n: usize,
        of: &str,
    ) -> crate::error::Result<std::borrow::Cow<'a, [bool]>> {
        match given {
            Some(flags) => check_same_length(flags.len(), n, name, of)
                .map(|()| std::borrow::Cow::Borrowed(flags)),
            None => Ok(std::borrow::Cow::Owned(vec![false; n])),
        }
    }
    let iscrowd = flags(input.iscrowd, "iscrowd", g, "sim's columns")?;
    let gt_ignore = flags(input.gt_ignore, "gt_ignore", g, "sim's columns")?;
    let group_of = flags(input.group_of, "group_of", g, "sim's columns")?;
    let dt_seed = flags(input.dt_ignore, "dt_ignore", d, "sim's rows")?;

    // A group-of box is ignored for the greedy phases and held out of the
    // second; its own pass is the one route to it, as in the detection driver.
    let ignored: Vec<bool> = (0..g)
        .map(|i| gt_ignore[i] || iscrowd[i] || group_of[i])
        .collect();
    let any_group_of = group_of.contains(&true);
    let t = iou_thrs.len();

    // Detections by score, ties in row order; ground truths stably
    // partitioned non-ignored first.
    let dt_order = score_order(scores);
    let gt_order: Vec<usize> = (0..g)
        .filter(|&i| !ignored[i])
        .chain((0..g).filter(|&i| ignored[i]))
        .collect();
    let num_not_ignored = ignored.iter().filter(|&&x| !x).count();
    let mut ranked = Vec::with_capacity(sim.len());
    for &di in &dt_order {
        let row = &sim[di * g..(di + 1) * g];
        ranked.extend(gt_order.iter().map(|&gi| row[gi]));
    }
    let crowd_ranked: Vec<bool> = gt_order.iter().map(|&gi| iscrowd[gi]).collect();
    let floors: Vec<f64> = iou_thrs.iter().map(|&t| coco_match_floor(t)).collect();
    // Open Images only: the group-of columns in rank order, their inverse for
    // the matcher's second phase, and which of them the count keeps.
    let group_of_ranked: Vec<bool> = gt_order.iter().map(|&gi| group_of[gi]).collect();
    let not_group_of: Vec<bool> = group_of_ranked.iter().map(|&x| !x).collect();
    let counted: Vec<bool> = gt_order.iter().map(|&gi| !gt_ignore[gi]).collect();
    let masks = GtMasks {
        rematchable: Some(&crowd_ranked),
        phase2_eligible: any_group_of.then_some(not_group_of.as_slice()),
    };
    let mut m = greedy_match_masked(&ranked, d, g, num_not_ignored, masks, &floors);
    let mut credited = ThreshMatrix::default();
    if any_group_of {
        let boxes = GroupOfBoxes {
            group_of: &group_of_ranked,
            counted: &counted,
        };
        match_group_of(&ranked, d, g, boxes, &floors, &mut m, &mut credited);
    }

    let mut out = GreedyOutput {
        dt_matches: ThreshMatrix::new(t, d, None),
        gt_matches: ThreshMatrix::new(t, g, None),
        // Seeded per threshold, as `COCOeval` seeds `dtIgnore`; a match
        // overwrites it with the ground truth's flag.
        dt_ignore: ThreshMatrix::repeat_row(t, &dt_seed),
    };
    for (ti, row) in m.dt_gt.iter_rows().enumerate() {
        for (rank, (&di, &matched)) in dt_order.iter().zip(row).enumerate() {
            let Some(pos) = matched else {
                continue;
            };
            let gi = gt_order[pos];
            out.dt_matches[(ti, di)] = Some(gi);
            let credited_here = any_group_of && credited[(ti, rank)];
            out.dt_ignore[(ti, di)] = pos >= num_not_ignored && !credited_here;
            // Rank order, so a crowd ground truth keeps the last detection;
            // a group-of box keeps the one credited with it.
            if credited_here || !group_of_ranked[pos] {
                out.gt_matches[(ti, gi)] = Some(di);
            }
        }
    }
    Ok(out)
}

/// The error for a kernel input holding NaN, which leaves `what` undefined:
/// [`greedy_match`]'s and [`try_lsap`](crate::primitives::assign::try_lsap)'s.
pub(crate) fn check_no_nan(values: &[f64], name: &str, what: &str) -> crate::error::Result<()> {
    if values.iter().any(|v| v.is_nan()) {
        return Err(Error::InvalidInput(format!(
            "{name} contains NaN; {what} is undefined"
        )));
    }
    Ok(())
}

/// The error for two inputs to [`greedy_match`] that must be parallel.
fn check_same_length(a: usize, b: usize, name_a: &str, name_b: &str) -> crate::error::Result<()> {
    if a != b {
        return Err(Error::InvalidInput(format!(
            "{name_a} and {name_b} must have the same length, got {a} and {b}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `greedy_match` on rows and columns in any order gives the kernel's
    /// answer on the same input sorted, mapped back: shuffling the rows and
    /// columns moves the matches with them and changes nothing else.
    #[test]
    fn greedy_match_takes_rows_and_columns_in_any_order() {
        // Rows score-descending, columns non-ignored first: the kernel's order.
        let (d, g) = (4, 3);
        let sim = [
            0.9, 0.2, 0.6, //
            0.8, 0.7, 0.0, //
            0.1, 0.55, 0.0, //
            0.0, 0.0, 0.95,
        ];
        let scores = [0.9, 0.8, 0.7, 0.6];
        let gt_ignore = [false, false, true];
        let thrs = [0.5, 0.75];
        fn input<'a>(sim: &'a [f64], scores: &'a [f64], gt_ignore: &'a [bool]) -> GreedyInput<'a> {
            GreedyInput {
                sim,
                num_dt: 4,
                num_gt: 3,
                scores,
                iou_thrs: &[0.5, 0.75],
                gt_ignore: Some(gt_ignore),
                iscrowd: None,
                group_of: None,
                dt_ignore: None,
            }
        }
        let sorted = greedy_match(&input(&sim, &scores, &gt_ignore)).expect("valid input");
        let kernel = greedy_match_masked(&sim, d, g, 2, GtMasks::default(), &thrs);
        assert_eq!(sorted.dt_matches.row(0), kernel.dt_gt.row(0));
        assert_eq!(sorted.dt_matches.row(1), kernel.dt_gt.row(1));
        assert_eq!(sorted.dt_ignore.row(0), [false, false, false, true]);

        // Rows reversed, columns rotated: row i of `shuffled` is row 3 - i,
        // column j is column (j + 1) % 3.
        let rows = [3, 2, 1, 0];
        let cols = [1, 2, 0];
        let mut shuffled = [0.0; 12];
        for (r, &from_r) in rows.iter().enumerate() {
            for (c, &from_c) in cols.iter().enumerate() {
                shuffled[r * g + c] = sim[from_r * g + from_c];
            }
        }
        let scores_s = rows.map(|r| scores[r]);
        let ignore_s = cols.map(|c| gt_ignore[c]);
        let got = greedy_match(&input(&shuffled, &scores_s, &ignore_s)).expect("valid input");
        let col_of = |orig: usize| cols.iter().position(|&c| c == orig).expect("a permutation");
        for t in 0..2 {
            for (r, &from_r) in rows.iter().enumerate() {
                assert_eq!(
                    got.dt_matches[(t, r)],
                    sorted.dt_matches[(t, from_r)].map(col_of),
                    "t{t} row {r}"
                );
                assert_eq!(
                    got.dt_ignore[(t, r)],
                    sorted.dt_ignore[(t, from_r)],
                    "t{t} row {r}"
                );
            }
            for (c, &from_c) in cols.iter().enumerate() {
                let want = sorted.gt_matches[(t, from_c)]
                    .map(|di| rows.iter().position(|&r| r == di).expect("a permutation"));
                assert_eq!(got.gt_matches[(t, c)], want, "t{t} column {c}");
            }
        }
    }

    /// A crowd region keeps the last detection to match it, and a detection
    /// that matches nothing keeps its `dt_ignore` seed.
    #[test]
    fn greedy_match_crowd_keeps_the_last_detection_and_seeds_dt_ignore() {
        let out = greedy_match(&GreedyInput {
            sim: &[0.9, 0.8, 0.0],
            num_dt: 3,
            num_gt: 1,
            scores: &[0.5, 0.9, 0.1],
            iou_thrs: &[0.5],
            gt_ignore: None,
            iscrowd: Some(&[true]),
            group_of: None,
            dt_ignore: Some(&[false, false, true]),
        })
        .expect("valid input");
        assert_eq!(out.dt_matches.row(0), [Some(0), Some(0), None]);
        // Score order is row 1, row 0: row 0 matches the crowd last.
        assert_eq!(out.gt_matches.row(0), [Some(0)]);
        assert_eq!(out.dt_ignore.row(0), [true, true, true]);
    }

    #[test]
    fn greedy_match_refuses_nan_and_lengths_that_disagree() {
        let ok = GreedyInput {
            sim: &[0.9, 0.1],
            num_dt: 1,
            num_gt: 2,
            scores: &[0.5],
            iou_thrs: &[0.5],
            gt_ignore: None,
            iscrowd: None,
            group_of: None,
            dt_ignore: None,
        };
        assert!(greedy_match(&ok).is_ok());
        let says = |input: GreedyInput<'_>, what: &str| match greedy_match(&input) {
            Err(Error::InvalidInput(m)) => assert!(m.contains(what), "{m}"),
            other => panic!("{what}: {other:?}"),
        };
        says(
            GreedyInput {
                sim: &[0.9, f64::NAN],
                ..ok
            },
            "sim contains NaN",
        );
        says(
            GreedyInput {
                scores: &[f64::NAN],
                ..ok
            },
            "scores contains NaN",
        );
        says(
            GreedyInput {
                iou_thrs: &[0.5, f64::NAN],
                ..ok
            },
            "iou_thrs contains NaN",
        );
        says(
            GreedyInput { sim: &[0.9], ..ok },
            "sim has 1 entries, not 1 detections × 2 ground truths",
        );
        says(
            GreedyInput {
                scores: &[0.5, 0.4],
                ..ok
            },
            "scores and sim's rows must have the same length, got 2 and 1",
        );
        says(
            GreedyInput {
                iscrowd: Some(&[false]),
                ..ok
            },
            "iscrowd and sim's columns",
        );
        says(
            GreedyInput {
                gt_ignore: Some(&[false; 3]),
                ..ok
            },
            "gt_ignore and sim's columns",
        );
        says(
            GreedyInput {
                group_of: Some(&[]),
                ..ok
            },
            "group_of and sim's columns",
        );
        says(
            GreedyInput {
                dt_ignore: Some(&[true, true]),
                ..ok
            },
            "dt_ignore and sim's rows",
        );
    }

    /// The group-of pass on one cell: two detections inside one group-of box,
    /// a third on a plain ground truth, a fourth inside a group-of box the
    /// area range ignores, and a group-of box nobody found. The first
    /// detection inside the counted box is credited, the second is matched
    /// but not, the plain match is untouched, the detection in the uncounted
    /// box is matched and not credited, and the empty box stays unmatched.
    #[test]
    fn group_of_pass_credits_the_first_detection_in_each_counted_box() {
        // Columns: [plain, group-of A, group-of B (nobody), group-of C
        // (uncounted)]; the group-of columns are ignored, so they sit in the
        // tail and `num_not_ignored == 1`. Rows score-descending; IoA
        // saturates at 1.0 inside a region.
        let iou = [
            0.0, 1.0, 0.0, 0.0, //
            0.9, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let group_of = [false, true, true, true];
        let counted = [true, true, true, false];
        let not_group_of = [true, false, false, false];
        let masks = GtMasks {
            rematchable: None,
            phase2_eligible: Some(&not_group_of),
        };
        let thrs = [0.5, 0.95];
        let mut m = greedy_match_masked(&iou, 4, 4, 1, masks, &thrs);
        assert_eq!(
            m.dt_gt.row(0),
            &[None, Some(0), None, None],
            "greedy phases skip group-of columns"
        );

        let mut credited = ThreshMatrix::default();
        let boxes = GroupOfBoxes {
            group_of: &group_of,
            counted: &counted,
        };
        match_group_of(&iou, 4, 4, boxes, &thrs, &mut m, &mut credited);
        // IoA inside a region is 1.0, so both thresholds take alike; the plain
        // match at 0.9 clears only the first.
        assert_eq!(m.dt_gt.row(0), &[Some(1), Some(0), Some(1), Some(3)]);
        assert_eq!(m.dt_gt.row(1), &[Some(1), None, Some(1), Some(3)]);
        assert_eq!(
            m.gt_matched.row(0),
            &[true, true, false, true],
            "box B is a miss"
        );
        assert_eq!(m.gt_matched.row(1), &[false, true, false, true]);
        for t in 0..2 {
            assert_eq!(credited.row(t), &[true, false, false, false]);
        }

        // No group-of column: a reset of `credited` and nothing else.
        let none = GroupOfBoxes {
            group_of: &[false; 4],
            counted: &counted,
        };
        let before = m.dt_gt.clone();
        match_group_of(&iou, 4, 4, none, &thrs, &mut m, &mut credited);
        assert_eq!(m.dt_gt.row(1), before.row(1));
        assert!(credited.iter_rows().all(|r| r.iter().all(|&c| !c)));
    }
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

    /// The matcher equals the plain scan it optimizes: every detection against
    /// every ground truth at every threshold, no early stop. The contract test
    /// above checks properties, which a matcher that stops too early still
    /// satisfies. The shapes run past the early stop: many detections, few
    /// ground truths, rematchable ground truths on some cases and none on
    /// others.
    #[test]
    fn greedy_match_equals_plain_scan() {
        fn plain_scan(
            iou_flat: &[f64],
            d: usize,
            g: usize,
            num_ni: usize,
            rematchable: &[bool],
            phase2: &[bool],
            thrs: &[f64],
        ) -> (Vec<Option<usize>>, Vec<bool>) {
            let mut dt_gt = vec![None; thrs.len() * d];
            let mut gt_matched = vec![false; thrs.len() * g];
            for (ti, &thr) in thrs.iter().enumerate() {
                for di in 0..d {
                    let open = |gi: usize| !gt_matched[ti * g + gi] || rematchable[gi];
                    let mut best = (thr, None);
                    // Phase 1: non-ignored ground truths.
                    for gi in 0..num_ni {
                        if open(gi) && iou_flat[di * g + gi] >= best.0 {
                            best = (iou_flat[di * g + gi], Some(gi));
                        }
                    }
                    // Phase 2: eligible ignored ones, only if phase 1 found none.
                    if best.1.is_none() {
                        for gi in num_ni..g {
                            if phase2[gi] && open(gi) && iou_flat[di * g + gi] >= best.0 {
                                best = (iou_flat[di * g + gi], Some(gi));
                            }
                        }
                    }
                    if let (_, Some(gi)) = best {
                        dt_gt[ti * d + di] = Some(gi);
                        gt_matched[ti * g + gi] = true;
                    }
                }
            }
            (dt_gt, gt_matched)
        }

        let mut rng = StdRng::seed_from_u64(0x5CA7);
        for case in 0..5000 {
            let d = rng.random_range(0..=40);
            let g = rng.random_range(0..=4);
            let num_ni = rng.random_range(0..=g);
            let iou_flat: Vec<f64> = (0..d * g)
                .map(|_| rng.random_range(0..=8) as f64 / 8.0)
                .collect();
            let crowd_rate = if rng.random_bool(0.5) { 0.0 } else { 0.3 };
            let rematchable: Vec<bool> = (0..g).map(|_| rng.random_bool(crowd_rate)).collect();
            let phase2: Vec<bool> = (0..g).map(|_| rng.random_bool(0.75)).collect();
            let thrs: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * i as f64).collect();

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
            let (dt_gt, gt_matched) =
                plain_scan(&iou_flat, d, g, num_ni, &rematchable, &phase2, &thrs);
            let ctx = format!("case {case}: d={d} g={g} num_ni={num_ni}");
            for ti in 0..thrs.len() {
                assert_eq!(
                    m.dt_gt.row(ti),
                    &dt_gt[ti * d..(ti + 1) * d],
                    "{ctx} t={ti}"
                );
                assert_eq!(
                    m.gt_matched.row(ti),
                    &gt_matched[ti * g..(ti + 1) * g],
                    "{ctx} t={ti}"
                );
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
