//! Per-image matching: one (image, category, area range, max_det) cell of the
//! evaluation.
//!
//! [`evaluate.rs`](super::evaluate) is the outer driver — it resolves parameters,
//! collects the sparse (image, category) pairs, and fans out across them. This
//! module is what each of those fan-out slots runs, and it owns no control flow
//! above the single pair.
//!
//! A fan-out slot is one *pair*, not one cell: [`gather_pair`] resolves the
//! annotations and orders the detections once, then [`evaluate_cell`] runs each
//! area range over that shared view. Only the ignore/area flags and the
//! partition they induce differ between ranges, so gathering per range meant
//! re-resolving every annotation id and re-sorting every detection four times
//! for the same answer.
//!
//! The assignment algorithm itself is not here either: that is
//! [`crate::primitives::greedy::greedy_match_masked`]. What lives here is everything
//! COCO-specific *around* it — deciding which ground truths are ignored, ordering
//! detections by score, reordering the IoU matrix to match, and translating the
//! matcher's indices back into annotation ids. The matcher's contract requires the
//! caller to own exactly this, so this module is that caller.
//!
//! Naming note: `matching`, not `match` — the latter is a reserved word and
//! `mod match;` does not compile without `r#` escaping.

use std::collections::HashMap;

use crate::coco::COCO;
use crate::params::{IouType, Params};
use crate::primitives::greedy::{GtMasks, ThreshMatrix};
use crate::types::Annotation;

use super::EvalMode;

/// Everything one (image, category) pair contributes that does **not** depend on
/// the area range: the resolved annotations, the detection score order, and the
/// pair's similarity matrix.
///
/// This is the range-invariant work the module docs describe, hoisted so the
/// four default area ranges share one resolution pass: resolving every
/// annotation id, sorting the detections by score, and the `(img, cat)` lookup
/// into the IoU cache. Everything that *does* vary by range lives in
/// [`GtView`]/[`DtView`] instead.
pub(super) struct PairCell<'a> {
    img_id: u64,
    cat_id: u64,
    max_det: usize,
    /// Ground truths in load order — the order `gt_ids` arrived in.
    gt_anns: Vec<&'a Annotation>,
    /// Index into `gt_anns` -> column in the pair's IoU matrix.
    gt_iou_indices: Vec<usize>,
    /// Number of GT ids returned before annotation lookup, which can drop
    /// entries. Only [`evaluate_cell`]'s final skip gate reads it — see the
    /// comment there for which counts gate the skip.
    gt_raw_count: usize,
    /// Detections score-descending and truncated to `max_det`.
    dt_anns: Vec<&'a Annotation>,
    /// Position in `dt_anns` -> row in the pair's IoU matrix.
    dt_iou_indices: Vec<usize>,
    dt_ids: Vec<u64>,
    dt_scores: Vec<f64>,
    /// The pair's similarity matrix, looked up once instead of per area range.
    iou_matrix: Option<&'a IouMatrix>,
}

/// Ground truths for one cell, partitioned non-ignored-first.
///
/// The matcher's contract requires that partition: indices
/// `[0, num_not_ignored)` are non-ignored and the rest are ignored. Fields
/// suffixed `_sorted` are in that partitioned order; the borrowed `anns` and
/// `iou_indices` stay in load order, and [`order`](Self::order) maps between them.
struct GtView<'a> {
    anns: &'a [&'a Annotation],
    /// Partitioned position -> index into `anns`.
    order: Vec<usize>,
    /// Index into `anns` -> column in the cell's IoU matrix.
    iou_indices: &'a [usize],
    ignore_sorted: Vec<bool>,
    /// Whether each GT counts toward the recall denominator — see
    /// [`EvalImg::gt_in_denominator`].
    in_denominator_sorted: Vec<bool>,
    iscrowd_sorted: Vec<bool>,
    /// Open Images only; empty otherwise. Guarded by `is_oid` at every use.
    is_group_of_sorted: Vec<bool>,
    num_not_ignored: usize,
}

impl GtView<'_> {
    fn len(&self) -> usize {
        self.anns.len()
    }

    /// Annotation id at a partitioned position.
    fn id_at(&self, sorted_idx: usize) -> u64 {
        self.anns[self.order[sorted_idx]].id
    }

    /// Annotation ids in partitioned order — the order `EvalImg` reports.
    fn sorted_ids(&self) -> Vec<u64> {
        (0..self.len()).map(|gi| self.id_at(gi)).collect()
    }
}

/// Detections for one cell: the pair's score-ordered list plus the one flag that
/// depends on the area range.
struct DtView<'a> {
    anns: &'a [&'a Annotation],
    /// Position -> row in the cell's IoU matrix.
    iou_indices: &'a [usize],
    /// Detection falls outside this cell's area range.
    area_ignore: Vec<bool>,
}

impl DtView<'_> {
    fn len(&self) -> usize {
        self.anns.len()
    }
}

/// Per-threshold match bookkeeping — the payload of an [`EvalImg`].
struct MatchOutcome {
    dt_matches: ThreshMatrix<u64>,
    gt_matches: ThreshMatrix<u64>,
    dt_matched: ThreshMatrix<bool>,
    gt_matched: ThreshMatrix<bool>,
    dt_ignore: ThreshMatrix<bool>,
}

/// Resolve one (image, category) pair's annotations, once for all area ranges.
///
/// Returns `None` for a pair with no ids on either side — the same skip
/// [`evaluate_cell`] would have made for every range.
///
/// The detection cap is applied *after* sorting, so it keeps the highest-scoring
/// detections rather than the first-loaded ones.
pub(super) fn gather_pair<'a>(
    ctx: &EvalImgContext<'a>,
    img_id: u64,
    cat_id: u64,
    max_det: usize,
) -> Option<PairCell<'a>> {
    use super::COCOeval;

    let gt_ids = COCOeval::get_anns_static(ctx.coco_gt, ctx.params, img_id, cat_id);
    let dt_ids = COCOeval::get_anns_static(ctx.coco_dt, ctx.params, img_id, cat_id);
    if gt_ids.is_empty() && dt_ids.is_empty() {
        return None;
    }

    let (gt_iou_indices, gt_anns): (Vec<usize>, Vec<&Annotation>) = gt_ids
        .iter()
        .enumerate()
        .filter_map(|(iou_idx, &id)| Some((iou_idx, ctx.coco_gt.get_ann(id)?)))
        .unzip();

    let mut with_iou_idx: Vec<(usize, &Annotation)> = dt_ids
        .iter()
        .enumerate()
        .filter_map(|(iou_idx, &id)| Some((iou_idx, ctx.coco_dt.get_ann(id)?)))
        .collect();
    with_iou_idx.sort_by(|a, b| {
        b.1.score
            .unwrap_or(0.0)
            .partial_cmp(&a.1.score.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    with_iou_idx.truncate(max_det);

    let (dt_iou_indices, dt_anns): (Vec<usize>, Vec<&Annotation>) =
        with_iou_idx.into_iter().unzip();
    let dt_ids: Vec<u64> = dt_anns.iter().map(|a| a.id).collect();
    let dt_scores: Vec<f64> = dt_anns.iter().map(|a| a.score.unwrap_or(0.0)).collect();

    Some(PairCell {
        img_id,
        cat_id,
        max_det,
        gt_anns,
        gt_iou_indices,
        gt_raw_count: gt_ids.len(),
        dt_anns,
        dt_iou_indices,
        dt_ids,
        dt_scores,
        iou_matrix: ctx.ious.get(&(img_id, cat_id)),
    })
}

/// Decide which of the pair's ground truths this area range ignores.
///
/// Ignore rules are mode-dependent: Open Images ignores group-of boxes and does
/// not care about `iscrowd`; COCO/LVIS ignore crowds, and keypoint evaluation
/// additionally ignores annotations with no labeled keypoints.
fn partition_gt<'a>(
    pair: &'a PairCell<'a>,
    area_rng: [f64; 2],
    is_kp: bool,
    is_oid: bool,
) -> GtView<'a> {
    let anns = pair.gt_anns.as_slice();

    // `ignore` governs *matching*; `in_denominator` governs the *recall
    // denominator*. COCO's single `gtIgnore` cannot express "not matchable here"
    // and "counted" at once, which Open Images group-of boxes need: they are
    // held out of matching (the second pass in `match_cell` absorbs them) yet
    // still count as one ground truth. See `EvalImg::gt_in_denominator`.
    let (ignore, in_denominator): (Vec<bool>, Vec<bool>) = anns
        .iter()
        .map(|ann| {
            let a = ann.area.unwrap_or(0.0);
            let area_ignore = a < area_rng[0] || a > area_rng[1];
            if is_oid {
                (
                    ann.is_group_of.unwrap_or(false) || area_ignore,
                    !area_ignore,
                )
            } else {
                let mut ignore = ann.iscrowd || area_ignore;
                if is_kp {
                    ignore = ignore || ann.num_visible_keypoints() == 0;
                }
                (ignore, !ignore)
            }
        })
        .unzip();

    // Stable sort on the ignore flag: non-ignored first, load order preserved
    // within each partition. Tie order is observable through `evalImgs`.
    let mut order: Vec<usize> = (0..anns.len()).collect();
    order.sort_by_key(|&i| ignore[i] as u8);

    let ignore_sorted: Vec<bool> = order.iter().map(|&i| ignore[i]).collect();
    let in_denominator_sorted: Vec<bool> = order.iter().map(|&i| in_denominator[i]).collect();
    let iscrowd_sorted: Vec<bool> = order.iter().map(|&i| anns[i].iscrowd).collect();
    let is_group_of_sorted: Vec<bool> = if is_oid {
        order
            .iter()
            .map(|&i| anns[i].is_group_of.unwrap_or(false))
            .collect()
    } else {
        Vec::new()
    };
    let num_not_ignored = ignore_sorted.iter().filter(|&&x| !x).count();

    GtView {
        anns,
        order,
        iou_indices: pair.gt_iou_indices.as_slice(),
        ignore_sorted,
        in_denominator_sorted,
        iscrowd_sorted,
        is_group_of_sorted,
        num_not_ignored,
    }
}

/// Flag the pair's detections that fall outside this area range.
fn area_filter_dt<'a>(pair: &'a PairCell<'a>, area_rng: [f64; 2]) -> DtView<'a> {
    let area_ignore: Vec<bool> = pair
        .dt_anns
        .iter()
        .map(|ann| {
            let a = ann.area.unwrap_or(0.0);
            a < area_rng[0] || a > area_rng[1]
        })
        .collect();

    DtView {
        anns: pair.dt_anns.as_slice(),
        iou_indices: pair.dt_iou_indices.as_slice(),
        area_ignore,
    }
}

/// Reorder the cell's IoU matrix into the flat row-major `[D*G]` layout the
/// matcher expects, with rows in score order and columns non-ignored-first.
fn reordered_iou(iou_mat: &IouMatrix, dt: &DtView<'_>, gt: &GtView<'_>) -> Vec<f64> {
    let (d, g) = (dt.len(), gt.len());
    let mut flat = vec![0.0_f64; d * g];
    for di in 0..d {
        // One row borrow per detection: the row index and its bounds test are
        // invariant across the whole GT scan below.
        let Some(row) = iou_mat.get(dt.iou_indices[di]) else {
            continue;
        };
        for (gi_sorted, &gi_orig) in gt.order.iter().enumerate() {
            if let Some(&v) = row.get(gt.iou_indices[gi_orig]) {
                flat[di * g + gi_sorted] = v;
            }
        }
    }
    flat
}

/// Run the matcher over one cell and translate its indices back to annotation ids.
fn match_cell(
    ctx: &EvalImgContext<'_>,
    gt: &GtView<'_>,
    dt: &DtView<'_>,
    iou_matrix: Option<&IouMatrix>,
    is_oid: bool,
) -> MatchOutcome {
    let (d, g) = (dt.len(), gt.len());
    let num_iou_thrs = ctx.params.iou_thrs.len();

    let mut dt_matches = ThreshMatrix::new(num_iou_thrs, d, 0u64);
    let mut gt_matches = ThreshMatrix::new(num_iou_thrs, g, 0u64);
    let mut dt_matched = ThreshMatrix::new(num_iou_thrs, d, false);
    // Seeded from the area-ignore flags so unmatched detections carry the right
    // ignore status even when there is no IoU data at all.
    let mut dt_ignore = ThreshMatrix::repeat_row(num_iou_thrs, &dt.area_ignore);

    let Some(iou_mat) = iou_matrix else {
        // No detections and/or no ground truths: nothing matched.
        return MatchOutcome {
            dt_matches,
            gt_matches,
            dt_matched,
            gt_matched: ThreshMatrix::new(num_iou_thrs, g, false),
            dt_ignore,
        };
    };

    let iou_flat = reordered_iou(iou_mat, dt, gt);

    // The per-GT policy flags encode the mode: crowd GTs are re-matchable
    // (COCO/LVIS only); under OID `iscrowd` is irrelevant and group-of GTs are
    // held out of the fallback phase, to be matched in the separate pass below.
    //
    // Each mode's *other* mask is uniform, and `None` says so without allocating
    // — this runs once per evaluated cell, so materializing both was two `Vec`s
    // per cell to restate the matcher's defaults. Only OID allocates, and only to
    // invert the group-of flags.
    let phase2_eligible: Option<Vec<bool>> =
        is_oid.then(|| gt.is_group_of_sorted.iter().map(|&x| !x).collect());

    // Both matching phases share `ctx.match_floors` — pycocotools' clamped
    // thresholds. See the policy table in `primitives::greedy`.
    let m = crate::primitives::greedy::greedy_match_masked(
        &iou_flat,
        d,
        g,
        gt.num_not_ignored,
        GtMasks {
            rematchable: (!is_oid).then_some(gt.iscrowd_sorted.as_slice()),
            phase2_eligible: phase2_eligible.as_deref(),
        },
        ctx.match_floors,
    );

    // Translate matched indices into annotation ids + ignore flags. Unmatched
    // detections keep the area-ignore status they were seeded with.
    for t_idx in 0..num_iou_thrs {
        for (di, dt_ann) in dt.anns.iter().enumerate() {
            if let Some(gi) = m.dt_gt[(t_idx, di)] {
                dt_matches[(t_idx, di)] = gt.id_at(gi);
                gt_matches[(t_idx, gi)] = dt_ann.id;
                dt_matched[(t_idx, di)] = true;
                // A detection matched to an ignored GT is itself ignored.
                dt_ignore[(t_idx, di)] = gt.ignore_sorted[gi];
            }
        }
    }
    let mut gt_matched = m.gt_matched;

    // Open Images second pass — group-of boxes.
    //
    // The protocol (https://storage.googleapis.com/openimages/web/evaluation.html):
    //
    //   "If at least one detection is inside group-of box a single True Positive
    //    is scored. ... Multiple correct detections inside the same group-of box
    //    is still count as a single True Positive. Otherwise, the group-of box is
    //    counted as a single False Negative."
    //
    // So a group-of box is worth exactly one ground truth: the best-scoring
    // detection inside it becomes a true positive, every other detection inside it
    // is ignored (neither TP nor FP), and if nothing is inside it the box is a
    // miss. This is the Open Images *Challenge* metric, equivalently TensorFlow's
    // `group_of_weight = 1.0`, and it is what FiftyOne implements unconditionally.
    //
    // "Inside" is IoA, not IoU — see the note in `iou.rs` where group-of GT
    // columns are flagged crowd so `sim` selects intersection-over-detection-area.
    //
    // Group-of GTs are excluded from both greedy phases (`phase2_eligible`),
    // so this is their only matching route. That exclusion is load-bearing: an IoA
    // column saturates at 1.0 for any detection inside the region, so a group-of
    // box left in phase 1 would outbid the real object a detection is sitting on
    // and turn that object into a false negative.
    //
    // Detections arrive score-descending, so the first one to claim a given box is
    // the highest-scoring one — the same choice TF makes with
    // `scores_group_of[gt_id] = max(scores_group_of[gt_id], scores[i])`.
    //
    // `is_group_of_sorted` doubles as the candidate mask: under OID `ignore` is
    // `is_group_of || area_ignore`, so every group-of box is ignored and therefore
    // already sorted into the `[num_not_ignored, g)` tail. A separate eligibility
    // vector would only restate that invariant.
    //
    // The guard skips the whole pass for cells with no group-of GT — the common
    // case, since group-of is a minority annotation — which otherwise costs a full
    // `d x g` scan per threshold for a guaranteed-empty result.
    if is_oid && gt.is_group_of_sorted.iter().any(|&x| x) {
        for (t_idx, &iou_thr) in ctx.match_floors.iter().enumerate() {
            for di in 0..d {
                if dt_matched[(t_idx, di)] {
                    continue;
                }
                // Best enclosing group-of box. The reference does
                // `np.argmax(ioa, axis=1)` then tests the threshold, which is the
                // same selection and the same first-wins tie-break that
                // `best_above_floor` owns — see its docs for why the tie matters.
                let row = &iou_flat[di * g..(di + 1) * g];
                let Some(gi) = crate::primitives::greedy::best_above_floor(
                    row,
                    &gt.is_group_of_sorted,
                    iou_thr,
                ) else {
                    continue;
                };

                dt_matches[(t_idx, di)] = gt.id_at(gi);
                dt_matched[(t_idx, di)] = true;
                // `gt_matched` *is* the "already credited" flag: group-of boxes are
                // excluded from both greedy phases, so it is false on entry here and
                // only this loop ever sets it — no separate `credited` vector.
                if gt_matched[(t_idx, gi)] {
                    // The box already has its true positive; absorb this one.
                    dt_ignore[(t_idx, di)] = true;
                } else {
                    // First (highest-scoring) detection inside this box scores it.
                    dt_ignore[(t_idx, di)] = false;
                    gt_matches[(t_idx, gi)] = dt.anns[di].id;
                    gt_matched[(t_idx, gi)] = true;
                }
            }
        }
    }

    MatchOutcome {
        dt_matches,
        gt_matches,
        dt_matched,
        gt_matched,
        dt_ignore,
    }
}

/// Evaluate one area range of an already-gathered image+category pair.
///
/// `not_exhaustive_cat` — when true (LVIS mode), unmatched detections are ignored
/// rather than counted as false positives.
///
/// Returns `None` for cells with nothing to report, which is what keeps
/// `evalImgs` sparse.
pub(super) fn evaluate_cell(
    ctx: &EvalImgContext<'_>,
    pair: &PairCell<'_>,
    area_rng: [f64; 2],
    not_exhaustive_cat: bool,
) -> Option<EvalImg> {
    let is_kp = ctx.params.iou_type == IouType::Keypoints;
    let is_oid = ctx.eval_mode == EvalMode::OpenImages;

    let gt = partition_gt(pair, area_rng, is_kp, is_oid);
    let dt = area_filter_dt(pair, area_rng);

    // Nothing non-ignored on either side means this cell contributes nothing —
    // but only skip it when there were no ground-truth ids at all, matching the
    // original condition. The two tests read different counts: `has_content`
    // reads the *resolved* views (non-ignored GTs, in-range detections after
    // the score sort and `max_det` cap), while the final gate is the *raw* GT
    // id count — the ids returned before annotation lookup, `gt_raw_count`.
    //
    // Both legs come from `partition_gt`/`area_filter_dt` alone, so the gate
    // runs before `match_cell`'s five `ThreshMatrix` allocations and the
    // greedy match — a discarded cell never pays for either.
    let has_content = gt.num_not_ignored > 0 || dt.area_ignore.iter().any(|&ignored| !ignored);
    if !has_content && pair.gt_raw_count == 0 {
        return None;
    }

    let mut outcome = match_cell(ctx, &gt, &dt, pair.iou_matrix, is_oid);

    // LVIS: on a not-exhaustively-labeled category, unmatched detections are
    // ignored instead of penalized as false positives.
    if not_exhaustive_cat {
        for t_idx in 0..ctx.params.iou_thrs.len() {
            for di in 0..dt.len() {
                if !outcome.dt_matched[(t_idx, di)] {
                    outcome.dt_ignore[(t_idx, di)] = true;
                }
            }
        }
    }

    Some(EvalImg {
        image_id: pair.img_id,
        category_id: pair.cat_id,
        area_rng,
        max_det: pair.max_det,
        dt_ids: pair.dt_ids.clone(),
        gt_ids: gt.sorted_ids(),
        dt_matches: outcome.dt_matches,
        gt_matches: outcome.gt_matches,
        dt_matched: outcome.dt_matched,
        gt_matched: outcome.gt_matched,
        dt_scores: pair.dt_scores.clone(),
        gt_ignore: gt.ignore_sorted,
        gt_in_denominator: gt.in_denominator_sorted,
        dt_ignore: outcome.dt_ignore,
    })
}

/// D×G IoU matrix (row-major: dt.len() rows, gt.len() columns).
pub(in crate::detection) type IouMatrix = Vec<Vec<f64>>;

/// Per-image, per-category evaluation result.
///
/// `#[non_exhaustive]`: evaluation families added later (panoptic, tracking) will
/// need fields here, and this keeps that additive rather than breaking. Construct
/// via evaluation, not by struct literal.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EvalImg {
    pub image_id: u64,
    pub category_id: u64,
    pub area_rng: [f64; 2],
    pub max_det: usize,
    /// Detection annotation IDs (sorted by score descending, truncated to max_det)
    pub dt_ids: Vec<u64>,
    /// Ground truth annotation IDs (sorted: non-ignored first, then ignored)
    pub gt_ids: Vec<u64>,
    /// Matched GT annotation id per IoU threshold: `dt_matches[(t, d)]` = GT id, or **0 as a
    /// sentinel for unmatched**. Do not use a non-zero check for presence — `Annotation.id`
    /// defaults to 0, so a real GT can have id=0. Use `dt_matched[(t, d)]` instead.
    pub dt_matches: ThreshMatrix<u64>,
    /// Matched DT annotation id per IoU threshold: `gt_matches[(t, g)]` = DT id, or **0 as a
    /// sentinel for unmatched**. Same caveat as `dt_matches`. Use `gt_matched[(t, g)]` instead.
    pub gt_matches: ThreshMatrix<u64>,
    /// Whether each detection was matched at each IoU threshold. Authoritative presence check;
    /// avoids the id=0 sentinel ambiguity in `dt_matches`.
    pub dt_matched: ThreshMatrix<bool>,
    /// Whether each GT was matched at each IoU threshold. Authoritative presence check;
    /// avoids the id=0 sentinel ambiguity in `gt_matches`.
    pub gt_matched: ThreshMatrix<bool>,
    /// Detection scores
    pub dt_scores: Vec<f64>,
    /// Whether each GT is ignored *for matching*
    pub gt_ignore: Vec<bool>,
    /// Whether each GT counts toward the recall denominator.
    ///
    /// Equal to `!gt_ignore` in every mode except Open Images, where a group-of
    /// box is held out of matching yet still counts as one ground truth — the
    /// protocol scores an undetected group-of box as a single false negative.
    /// Consumers computing `num_gt` must read this, not `gt_ignore`.
    pub gt_in_denominator: Vec<bool>,
    /// Whether each detection is ignored per IoU threshold
    pub dt_ignore: ThreshMatrix<bool>,
}

impl EvalImg {
    /// How many ground truths in this cell count toward recall.
    ///
    /// Use this rather than counting `!gt_ignore` — see
    /// [`gt_in_denominator`](Self::gt_in_denominator) for when the two differ.
    pub fn num_gt_in_denominator(&self) -> usize {
        self.gt_in_denominator.iter().filter(|&&x| x).count()
    }

    /// Whether ground truth `gi` is a *scored* miss when unmatched.
    ///
    /// The false-negative counterpart of
    /// [`num_gt_in_denominator`](Self::num_gt_in_denominator). Tallying false
    /// negatives from `!gt_ignore` instead disagrees with the recall the same
    /// evaluation reports.
    pub fn counts_as_miss(&self, gi: usize) -> bool {
        self.gt_in_denominator.get(gi).copied().unwrap_or(false)
    }
}

/// Read-only context shared across all [`gather_pair`]/[`evaluate_cell`] calls
/// within a single [`COCOeval::evaluate`](super::COCOeval::evaluate) invocation.
pub(super) struct EvalImgContext<'a> {
    pub(super) coco_gt: &'a COCO,
    pub(super) coco_dt: &'a COCO,
    pub(super) params: &'a Params,
    pub(super) ious: &'a HashMap<(u64, u64), IouMatrix>,
    pub(super) eval_mode: super::EvalMode,
    /// `params.iou_thrs` with pycocotools' match floor applied
    /// ([`crate::primitives::greedy::coco_match_floor`]). Resolved once per
    /// `evaluate()` rather than per image-category pair: this is read inside a
    /// rayon fan-out over every (category, area range, image) tuple, so deriving
    /// it at the call site would allocate a short `Vec` hundreds of thousands of
    /// times per evaluation.
    pub(super) match_floors: &'a [f64],
}
