//! Count aggregation and metric formulas.
//!
//! This slice provides the **detection rank-based PR accumulator**
//! ([`precision_recall_curve`]) — the pycocotools/PASCAL-VOC precision-at-fixed-
//! recall computation shared by detection's accumulate, TIDE, and diagnostics
//! paths — and [`average_precision`], the mechanical AP core layered on it
//! (sort → classify → cumsum → interpolate → mean).
//!
//! # Empty-set conventions stay at the call site
//!
//! What AP *means* when there is no ground truth is a per-metric decision, not a
//! mechanical one: TIDE reports `0.0` (a vacuous corpus AP), while per-image
//! diagnostics reports `1.0` (an empty image with nothing predicted is
//! legitimately perfect). These are different metrics, not drift, so the
//! primitive takes no policy flag — callers guard `num_gt == 0` themselves and
//! document why.
//!

use crate::primitives::greedy::score_order;

/// One sampled point on a precision-recall curve.
///
/// Named fields rather than a `(usize, f64, usize)` tuple: the two `usize` are
/// different indices, and nothing would stop a call site transposing them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrPoint {
    /// Index into the `rec_thrs` grid this point samples.
    pub rec_thr_idx: usize,
    /// Interpolated precision at that recall threshold.
    pub precision: f64,
    /// Detection rank (index into the score-descending ordering) at which the
    /// threshold is first met — lets the caller recover the sorted score there.
    pub detection_rank: usize,
}

/// Precision at one rank, as pycocotools computes it: `tp / (fp + tp + np.spacing(1))`.
///
/// **The one spelling of the guard term.** It changes the result only when
/// `tp + fp == 1` — any larger integer absorbs `2^-52` — and there it puts a lone
/// leading true positive at `1 - 2^-52` rather than `1.0`. Keeping it is what
/// makes the `precision` arrays bit-equal to the reference; `average_precision_all_points`
/// deliberately does not use it, because its reference (TensorFlow's
/// `compute_precision_recall`) has no such term.
#[inline]
pub(crate) fn coco_precision(tp: f64, fp: f64) -> f64 {
    tp / (tp + fp + f64::EPSILON)
}

/// Sort `(key, position)` pairs ascending by key, ties by position, as
/// `sort_unstable` on the pairs does — for pairs whose positions ascend in
/// slice order, which makes it the stable sort by key that ranks detections.
///
/// A least-significant-digit radix sort on the key's bytes, skipping the ones
/// every key shares: a pass per byte is stable, so position order survives
/// every tie. [`descending_score_key`](crate::primitives::greedy::descending_score_key) keys of scores in `[0, 1]` share their
/// top byte, and scores that came from `float32` their low three, so a
/// detector's ranking is about four passes over the pairs where a comparison
/// sort is `log2 n`. A short slice goes to the comparison sort. `scratch` is
/// working space the caller keeps across calls.
pub(crate) fn sort_score_keyed(items: &mut Vec<(u64, u32)>, scratch: &mut Vec<(u64, u32)>) {
    const RADIX_FROM: usize = 512;
    debug_assert!(
        items.windows(2).all(|w| w[0].1 < w[1].1),
        "positions ascend"
    );
    if items.len() < RADIX_FROM {
        items.sort_unstable();
        return;
    }
    let (any, all) = items
        .iter()
        .fold((0u64, u64::MAX), |(any, all), &(key, _)| {
            (any | key, all & key)
        });
    let varying = any ^ all;
    scratch.clear();
    scratch.resize(items.len(), (0, 0));
    for shift in (0..64).step_by(8) {
        if (varying >> shift).trailing_zeros() >= 8 {
            continue;
        }
        let mut at = [0usize; 256];
        for &(key, _) in items.iter() {
            at[((key >> shift) & 0xFF) as usize] += 1;
        }
        let mut sum = 0;
        for slot in &mut at {
            let count = *slot;
            *slot = sum;
            sum += count;
        }
        for &item in items.iter() {
            let digit = ((item.0 >> shift) & 0xFF) as usize;
            scratch[at[digit]] = item;
            at[digit] += 1;
        }
        std::mem::swap(items, scratch);
    }
}

/// Precision interpolated at fixed recall thresholds, from cumulative TP/FP.
///
/// `tp_cum` and `fp_cum` must already be cumulative (prefix-summed) over
/// detections sorted by score descending. Returns:
/// - the final recall achieved (`tp_cum[nd-1] / num_gt`);
/// - one [`PrPoint`] for each recall threshold that is reached.
///
/// Unreachable recall thresholds are omitted. Precision is `coco_precision`
/// made monotonically non-increasing right-to-left before sampling (PASCAL VOC
/// interpolation), matching pycocotools.
///
/// # Panics
///
/// If `tp_cum` and `fp_cum` have different lengths.
pub fn precision_recall_curve(
    tp_cum: &[f64],
    fp_cum: &[f64],
    num_gt: usize,
    rec_thrs: &[f64],
) -> (f64, Vec<PrPoint>) {
    let mut scratch = PrCurveScratch::default();
    let mut out = Vec::new();
    let final_recall =
        precision_recall_curve_into(tp_cum, fp_cum, num_gt, rec_thrs, &mut scratch, &mut out);
    (
        final_recall,
        out.iter()
            .map(|&(rec_thr_idx, precision, detection_rank)| PrPoint {
                rec_thr_idx,
                precision,
                detection_rank,
            })
            .collect(),
    )
}

/// Reusable working buffers for [`precision_recall_curve_into`] and
/// [`precision_recall_curve_of_order_into`].
///
/// Working vectors — two `nd`-long ones for the cumulative-array form, a
/// true-positive-long one plus a per-point index for the fused form — held by
/// the caller so a loop over IoU thresholds allocates once instead of once per
/// threshold. Opaque on purpose:
/// what is inside is an implementation detail of the accumulator, and the only
/// thing a caller may do with it is keep it alive.
#[derive(Debug, Default)]
pub struct PrCurveScratch {
    rc: Vec<f64>,
    pr: Vec<f64>,
    /// [`precision_recall_curve_of_order_into`] only: for each emitted point,
    /// which entry of `pr` (there, one per true positive) it reads.
    env_idx: Vec<usize>,
}

/// [`precision_recall_curve`] writing into caller-owned buffers.
///
/// Same computation, same values, same emission order — the only difference is
/// that the two `nd`-long working vectors and the output vector are supplied
/// rather than allocated. `out` is cleared first; each emitted tuple is
/// `(rec_thr_idx, precision, detection_rank)` — the fields of [`PrPoint`], as a
/// plain tuple so the hot accumulator's reusable buffer stays a flat `Vec`. The
/// return value is `final_recall`.
///
/// `detection::accumulate` uses this form only for Open Images, whose all-points
/// AP needs the cumulative arrays too; every other mode goes through
/// [`precision_recall_curve_of_order_into`], which computes the same curve
/// without materializing them.
///
/// # Panics
///
/// If `tp_cum` and `fp_cum` have different lengths.
pub fn precision_recall_curve_into(
    tp_cum: &[f64],
    fp_cum: &[f64],
    num_gt: usize,
    rec_thrs: &[f64],
    scratch: &mut PrCurveScratch,
    out: &mut Vec<(usize, f64, usize)>,
) -> f64 {
    out.clear();

    assert_eq!(
        tp_cum.len(),
        fp_cum.len(),
        "precision_recall_curve: tp_cum and fp_cum must be parallel arrays \
         (got {} vs {})",
        tp_cum.len(),
        fp_cum.len()
    );

    let nd = tp_cum.len();
    if nd == 0 || num_gt == 0 {
        return 0.0;
    }

    let num_gt_f = num_gt as f64;

    // Recall and precision at each detection rank.
    let (rc, pr) = (&mut scratch.rc, &mut scratch.pr);
    rc.clear();
    pr.clear();
    rc.reserve(nd);
    pr.reserve(nd);
    for d in 0..nd {
        rc.push(tp_cum[d] / num_gt_f);
        pr.push(coco_precision(tp_cum[d], fp_cum[d]));
    }

    let final_recall = rc[nd - 1];

    // Make precision monotonically non-increasing from right to left (VOC interp).
    for d in (0..nd.saturating_sub(1)).rev() {
        pr[d] = pr[d].max(pr[d + 1]);
    }

    // Two-pointer scan: map pr onto fixed recall thresholds.
    out.reserve(rec_thrs.len());
    let mut rc_ptr = 0;
    for (r_idx, &rec_thr) in rec_thrs.iter().enumerate() {
        while rc_ptr < nd && rc[rc_ptr] < rec_thr {
            rc_ptr += 1;
        }
        if rc_ptr < nd {
            out.push((r_idx, pr[rc_ptr], rc_ptr));
        }
    }

    final_recall
}

/// How one ranked detection moves the TP/FP counters.
enum Tally {
    TruePositive,
    FalsePositive,
    /// Contributes to neither counter but still occupies a rank, which is what
    /// keeps the cumulative counts lined up with the score ordering the curve is
    /// read at.
    Ignored,
}

/// **The one owner of TP/FP classification**, over bit lanes: lane `l` of
/// `matched` and `ignored` is one detection's flags at one IoU threshold, and
/// `lanes` masks the lanes in use. Returns the true-positive and
/// false-positive lanes. An ignored lane is neither; of the rest, a matched one
/// is a true positive and an unmatched one a false positive.
///
/// [`tally`] is this rule at a single lane, and
/// [`precision_recall_curves_of_lanes_into`] applies it to every threshold at
/// once, so the per-threshold and multi-threshold paths cannot classify a
/// detection differently.
#[inline]
fn tally_lanes(matched: u64, ignored: u64, lanes: u64) -> (u64, u64) {
    let counted = !ignored & lanes;
    (matched & counted, !matched & counted)
}

/// [`tally_lanes`] at one lane. `i` indexes the parallel `matched`/`ignored`
/// arrays; both [`cumulative_tp_fp`] and
/// [`precision_recall_curve_of_order_into`] decide TP versus FP versus ignored
/// through this and nowhere else.
#[inline]
fn tally(i: usize, matched: &[bool], ignored: Option<&[bool]>) -> Tally {
    let ignored = ignored.is_some_and(|ig| ig[i]);
    match tally_lanes(u64::from(matched[i]), u64::from(ignored), 1) {
        (1, _) => Tally::TruePositive,
        (_, 1) => Tally::FalsePositive,
        _ => Tally::Ignored,
    }
}

/// Cumulative TP and FP counts over detections visited in `order`.
///
/// `order` lists indices into the parallel `matched`/`ignored` arrays,
/// score-descending; classification is the private `tally`'s. `tp_cum` and `fp_cum` are
/// cleared and refilled to `order`'s length, so a caller sweeping IoU thresholds
/// reuses one pair of buffers.
pub fn cumulative_tp_fp(
    order: impl IntoIterator<Item = usize>,
    matched: &[bool],
    ignored: Option<&[bool]>,
    tp_cum: &mut Vec<f64>,
    fp_cum: &mut Vec<f64>,
) {
    tp_cum.clear();
    fp_cum.clear();

    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    for i in order {
        match tally(i, matched, ignored) {
            Tally::TruePositive => tp += 1.0,
            Tally::FalsePositive => fp += 1.0,
            Tally::Ignored => {}
        }
        tp_cum.push(tp);
        fp_cum.push(fp);
    }
}

/// [`precision_recall_curve_into`] straight from match flags, skipping the
/// cumulative arrays.
///
/// Same values, same emission order, same return as [`cumulative_tp_fp`]
/// followed by [`precision_recall_curve_into`] (the pair). `order` visits
/// indices into `matched`/`ignored` score-descending, as for
/// [`cumulative_tp_fp`]; classification is the private `tally`'s.
///
/// Where the pair writes four `nd`-long arrays (`tp_cum`, `fp_cum`, recall,
/// precision) and reads them back, this keeps integer counters, computes
/// precision at true-positive ranks only — the only ranks the VOC envelope can
/// take its maximum from, and the only ranks (rank 0 aside) a recall threshold
/// can first be met at — and samples recall on the fly: the two-pointer scan is
/// turned inside out to run over ranks, recording each threshold at the first
/// rank whose recall is not below it. That is the same rank the pair's scan
/// lands on for any `rec_thrs` — sorted or not, `NaN` included, because the
/// predicate is the negation of the pair's `recall < threshold` rather than a
/// rewrite of it. Recall is `tp / num_gt` as a fresh division at every change,
/// never a reciprocal multiply, so `19 / 20` still sits where the pair puts it
/// against a `0.95` threshold. The body's comment carries the envelope argument.
///
/// `detection::accumulate` runs this `T` times per (category, area range,
/// max_det) cell — ~10,000 calls per `accumulate()` on COCO val, several hundred
/// thousand across a bootstrap comparison — which is why it takes caller-owned
/// buffers.
pub fn precision_recall_curve_of_order_into(
    order: impl IntoIterator<Item = usize>,
    matched: &[bool],
    ignored: Option<&[bool]>,
    num_gt: usize,
    rec_thrs: &[f64],
    scratch: &mut PrCurveScratch,
    out: &mut Vec<(usize, f64, usize)>,
) -> f64 {
    out.clear();
    if num_gt == 0 {
        return 0.0;
    }
    let num_gt_f = num_gt as f64;

    // Forward pass. Only a true positive moves recall, so a recall threshold is
    // first met either at rank 0 or at a true-positive rank, and only a true
    // positive raises precision: at a false-positive rank precision is
    // `tp / (tp + fp)` with the same `tp` and a larger `fp`, so it is below the
    // precision at the true positive before it, and at an ignored rank it is
    // unchanged. The right-to-left envelope over every rank therefore equals the
    // envelope over the true-positive ranks alone, and precision is computed
    // there only — `#TP` divisions instead of `nd`. `pr` holds it, one entry per
    // true positive in rank order; `out` holds `(rec_thr_idx, <placeholder>,
    // rank)` plus, in `env_idx`, the index into `pr` of the true positive at
    // (or first after) that rank, until the envelope below fills the precision
    // in. A threshold met at rank 0 before any true positive gets index 0 — the
    // envelope from the first true positive on, which is what the pair's running
    // maximum over its leading zeros comes to as well.
    let pr = &mut scratch.pr;
    pr.clear();
    let env_idx = &mut scratch.env_idx;
    env_idx.clear();
    out.reserve(rec_thrs.len());
    env_idx.reserve(rec_thrs.len());
    let (mut tp, mut fp) = (0usize, 0usize);
    // `0 / num_gt`, exactly as the pair computes recall before the first true
    // positive; also the return for an empty `order`.
    let mut rc = 0.0f64;
    let mut r_ptr = 0;
    for (d, i) in order.into_iter().enumerate() {
        let is_tp = match tally(i, matched, ignored) {
            Tally::TruePositive => {
                tp += 1;
                rc = tp as f64 / num_gt_f;
                pr.push(coco_precision(tp as f64, fp as f64));
                true
            }
            Tally::FalsePositive => {
                fp += 1;
                false
            }
            Tally::Ignored => false,
        };
        if !is_tp && d != 0 {
            continue;
        }
        record_thresholds(rc, rec_thrs, &mut r_ptr, (d, pr.len()), out, env_idx);
    }
    fill_envelope(pr, env_idx, out);
    rc
}

/// At rank `d`, with `pr_len` precisions so far, the points of the recall
/// thresholds from `*r_ptr` on that recall `rc` is not below, each reading
/// the latest precision once [`fill_envelope`] fills it in.
///
/// "Not below", spelled as the negation of the pair's `rc < rec_thr` rather
/// than as `rc >= rec_thr`, so an incomparable (`NaN`) threshold records at
/// the current rank there and here alike.
#[inline]
fn record_thresholds(
    rc: f64,
    rec_thrs: &[f64],
    r_ptr: &mut usize,
    (d, pr_len): (usize, usize),
    points: &mut Vec<(usize, f64, usize)>,
    env_idx: &mut Vec<usize>,
) {
    while *r_ptr < rec_thrs.len()
        && rc.partial_cmp(&rec_thrs[*r_ptr]) != Some(std::cmp::Ordering::Less)
    {
        points.push((*r_ptr, 0.0, d));
        env_idx.push(pr_len.saturating_sub(1));
        *r_ptr += 1;
    }
}

/// Make `pr` monotonically non-increasing from right to left (VOC interp),
/// then give each of `points` the precision at its `env_idx` entry.
fn fill_envelope(pr: &mut [f64], env_idx: &[usize], points: &mut [(usize, f64, usize)]) {
    for j in (0..pr.len().saturating_sub(1)).rev() {
        pr[j] = pr[j].max(pr[j + 1]);
    }
    for (point, &j) in points.iter_mut().zip(env_idx.iter()) {
        // No true positive at all: every precision the pair computes is zero.
        point.1 = pr.get(j).copied().unwrap_or(0.0);
    }
}

/// The curves [`precision_recall_curves_of_lanes_into`] writes, one per lane,
/// and the working buffers it reuses. Held by the caller across calls, like
/// [`PrCurveScratch`].
#[derive(Debug, Default)]
pub(crate) struct LaneCurves {
    /// Per lane, the points [`precision_recall_curve_of_order_into`] emits.
    pub(crate) points: Vec<Vec<(usize, f64, usize)>>,
    /// Per lane, the final recall it returns.
    pub(crate) final_recall: Vec<f64>,
    /// Per group of lanes: its live lanes as a mask, and the ranks ignored at
    /// every one of them; per lane, its group.
    group_masks: Vec<u64>,
    ignored_group: Vec<usize>,
    group_of: Vec<usize>,
    /// Per lane: true positives so far, ranks ignored at that lane but not
    /// at its whole group, and the next recall threshold to record.
    tp: Vec<usize>,
    ignored: Vec<usize>,
    r_ptr: Vec<usize>,
    /// Per lane, as in [`PrCurveScratch`]: precision at each true positive,
    /// and the entry of it each point reads.
    pr: Vec<Vec<f64>>,
    env_idx: Vec<Vec<usize>>,
}

/// [`precision_recall_curve_of_order_into`] at up to 64 IoU thresholds in one
/// pass over `order`.
///
/// `matched[i]` and `ignored[i]` are detection `i`'s flags in every lane, lane
/// `l` in bit `l`; `num_gts[l]` is lane `l`'s ground-truth count, and there
/// are `num_gts.len()` lanes (1 to 64), bits above them not read. A lane is
/// one threshold, or one threshold of one area range when the caller packs
/// several ranges' flags side by side. For each lane `l`, `curves.points[l]`
/// and `curves.final_recall[l]` are what
/// [`precision_recall_curve_of_order_into`] produces from that lane's flags as
/// `bool` slices and its `num_gt`: the same values in the same order.
///
/// Lanes come in groups of `group` (which divides the lane count): one group
/// for one area range's thresholds. A rank is usually ignored at all of a
/// group's lanes or at none — a detection outside an area range is ignored at
/// every threshold of it — so a group ignored whole is counted once, and only
/// a group ignored at some of its lanes is counted lane by lane.
///
/// The single-lane kernel walks every rank once per threshold, and
/// `detection::accumulate` needs ten thresholds per (category, area range, cap)
/// cell. Here a rank costs a few word operations unless it is a true positive
/// at some lane, and the per-lane work — a precision, the recall thresholds it
/// meets — happens at true positives only, the ranks the single-lane kernel
/// does that work at. False positives are never counted: at rank `d`, a lane's
/// count is the `d + 1` ranks seen less its true positives and its ignored
/// ranks, each decided by [`tally_lanes`].
///
/// # Panics
///
/// If there are no lanes or more than 64.
pub(crate) fn precision_recall_curves_of_lanes_into(
    order: impl IntoIterator<Item = usize>,
    matched: &[u64],
    ignored: &[u64],
    num_gts: &[usize],
    group: usize,
    rec_thrs: &[f64],
    curves: &mut LaneCurves,
) {
    let lanes = num_gts.len();
    assert!(
        (1..=64).contains(&lanes),
        "precision_recall_curves_of_lanes_into: 1 to 64 lanes, got {lanes}"
    );
    let LaneCurves {
        points,
        final_recall,
        tp,
        group_masks,
        ignored_group,
        group_of,
        ignored: ignored_at,
        r_ptr,
        pr,
        env_idx,
    } = curves;
    points.resize_with(lanes, Vec::new);
    pr.resize_with(lanes, Vec::new);
    env_idx.resize_with(lanes, Vec::new);
    points[..lanes].iter_mut().for_each(Vec::clear);
    pr[..lanes].iter_mut().for_each(Vec::clear);
    env_idx[..lanes].iter_mut().for_each(Vec::clear);
    final_recall.clear();
    final_recall.resize(lanes, 0.0);
    for v in [&mut *tp, &mut *ignored_at, &mut *r_ptr] {
        v.clear();
        v.resize(lanes, 0);
    }
    // As the single-lane kernel, a lane with no ground truth has no points
    // and recall 0: it is left out of the walk.
    let lane_mask = num_gts
        .iter()
        .enumerate()
        .fold(0u64, |mask, (l, &n)| mask | (u64::from(n > 0) << l));
    if lane_mask == 0 {
        return;
    }
    assert!(
        group > 0 && lanes % group == 0,
        "precision_recall_curves_of_lanes_into: groups of {group} in {lanes} lanes"
    );
    let width_mask = u64::MAX >> (64 - group);
    group_masks.clear();
    group_masks.extend((0..lanes / group).map(|g| (width_mask << (g * group)) & lane_mask));
    ignored_group.clear();
    ignored_group.resize(group_masks.len(), 0);
    group_of.clear();
    group_of.extend((0..lanes).map(|l| l / group));
    for (d, i) in order.into_iter().enumerate() {
        let ig = ignored[i] & lane_mask;
        let (tp_lanes, _) = tally_lanes(matched[i], ig, lane_mask);
        // Branch-free on the common cases: in an area range other than
        // `all`, ranks inside and outside it interleave at random, so a branch
        // on "ignored at the whole group" mispredicts on about half of them.
        let mut bits = 0;
        for (count, &gm) in ignored_group.iter_mut().zip(group_masks.iter()) {
            let at = ig & gm;
            let whole = at == gm && gm != 0;
            *count += usize::from(whole);
            bits |= if whole { 0 } else { at };
        }
        while bits != 0 {
            ignored_at[bits.trailing_zeros() as usize] += 1;
            bits &= bits - 1;
        }
        // Rank 0 records at every lane, as the single-lane kernel does;
        // after it only a true positive moves recall.
        let mut recording = if d == 0 { lane_mask } else { tp_lanes };
        while recording != 0 {
            let l = recording.trailing_zeros() as usize;
            recording &= recording - 1;
            if tp_lanes >> l & 1 == 1 {
                tp[l] += 1;
                let fp = d + 1 - tp[l] - ignored_group[group_of[l]] - ignored_at[l];
                pr[l].push(coco_precision(tp[l] as f64, fp as f64));
            }
            // A fresh division, as the single-lane kernel's `rc`.
            let rc = tp[l] as f64 / num_gts[l] as f64;
            record_thresholds(
                rc,
                rec_thrs,
                &mut r_ptr[l],
                (d, pr[l].len()),
                &mut points[l],
                &mut env_idx[l],
            );
        }
    }
    for l in (0..lanes).filter(|&l| lane_mask >> l & 1 == 1) {
        final_recall[l] = tp[l] as f64 / num_gts[l] as f64;
        fill_envelope(&mut pr[l], &env_idx[l], &mut points[l]);
    }
}

/// Mean interpolated precision over `rec_thrs` — the tail every AP path
/// shares — writing into caller-owned buffers, via [`precision_recall_curve_into`].
/// Only the two intermediate `Vec`s (the [`PrCurveScratch`] pair and the tuple
/// output buffer) and the `Vec<PrPoint>` that [`precision_recall_curve`]
/// collects are skipped, in favor of reusing what `pr_curve`/`curve_out`
/// already hold.
fn mean_precision_into(
    tp_cum: &[f64],
    fp_cum: &[f64],
    num_gt: usize,
    rec_thrs: &[f64],
    pr_curve: &mut PrCurveScratch,
    curve_out: &mut Vec<(usize, f64, usize)>,
) -> f64 {
    precision_recall_curve_into(tp_cum, fp_cum, num_gt, rec_thrs, pr_curve, curve_out);
    curve_out
        .iter()
        .map(|&(_, precision, _)| precision)
        .sum::<f64>()
        / rec_thrs.len() as f64
}

/// Reusable working buffers for the ranked-AP `_into` variants
/// ([`average_precision_ranked_into`] and the internal
/// `average_precision_of_order_into` path it shares).
///
/// Bundles [`PrCurveScratch`] with the two cumulative TP/FP buffers and the
/// PR-curve tuple-output buffer that sit between it and the caller — the full
/// set `average_precision_of_order_into` otherwise allocates fresh every call.
/// TIDE's `category_deltas` calls the ranked AP ~8 times per category from
/// inside a rayon fan-out over categories; holding one `ApScratch` per work
/// item turns those ~8 × 4 per-call allocations into 4 for the whole category.
#[derive(Debug, Default)]
pub(crate) struct ApScratch {
    pr_curve: PrCurveScratch,
    tp_cum: Vec<f64>,
    fp_cum: Vec<f64>,
    curve_out: Vec<(usize, f64, usize)>,
}

/// AP of one explicit ranking, writing into caller-owned `scratch` — the
/// shared body of [`average_precision`] and [`average_precision_ranked_into`].
/// `order` visits indices into `matched`/`ignored` score-descending. See
/// [`ApScratch`] for what is reused.
fn average_precision_of_order_into(
    order: impl IntoIterator<Item = usize>,
    matched: &[bool],
    ignored: Option<&[bool]>,
    num_gt: usize,
    rec_thrs: &[f64],
    scratch: &mut ApScratch,
) -> f64 {
    cumulative_tp_fp(
        order,
        matched,
        ignored,
        &mut scratch.tp_cum,
        &mut scratch.fp_cum,
    );
    mean_precision_into(
        &scratch.tp_cum,
        &scratch.fp_cum,
        num_gt,
        rec_thrs,
        &mut scratch.pr_curve,
        &mut scratch.curve_out,
    )
}

/// Average precision over `rec_thrs`, from per-detection match flags.
///
/// The mechanical AP core: sort by score descending → classify each detection as
/// TP/FP (skipping ignored ones) → cumulative sum → interpolate onto `rec_thrs`
/// via [`precision_recall_curve`] → mean. Thresholds beyond the achieved recall
/// contribute zero, matching pycocotools' 101-point convention.
///
/// `scores`, `matched`, and `ignored` (when supplied) are parallel arrays over
/// detections in any order; `ignored = None` means no detection is ignored. The
/// sort is stable, so callers whose input is already score-descending keep their
/// tie order.
///
/// Returns `0.0` when there are no detections or no ground truth — but see the
/// [module note](self) on empty-set conventions: a caller that wants a different
/// answer for `num_gt == 0` must guard before calling.
///
/// The sort is total and stable, on the same key `COCOeval` ranks detections
/// by: `-0.0` ties `0.0`, positive `NaN` ranks above every number and negative
/// `NaN` below, instead of feeding std's sort a non-total order, which may
/// panic (Rust ≥ 1.81) or silently scramble the ranking.
///
/// # Panics
///
/// If `matched` (or `ignored`, when supplied) is not the same length as `scores`.
pub fn average_precision(
    scores: &[f64],
    matched: &[bool],
    ignored: Option<&[bool]>,
    num_gt: usize,
    rec_thrs: &[f64],
) -> f64 {
    assert_eq!(
        scores.len(),
        matched.len(),
        "average_precision: scores and matched must be parallel arrays (got {} vs {})",
        scores.len(),
        matched.len()
    );
    if let Some(ig) = ignored {
        assert_eq!(
            scores.len(),
            ig.len(),
            "average_precision: scores and ignored must be parallel arrays (got {} vs {})",
            scores.len(),
            ig.len()
        );
    }

    let nd = scores.len();
    if nd == 0 || num_gt == 0 || rec_thrs.is_empty() {
        return 0.0;
    }

    let order = score_order(scores);
    let mut scratch = ApScratch::default();
    average_precision_of_order_into(order, matched, ignored, num_gt, rec_thrs, &mut scratch)
}

/// [`average_precision`] for detections **already** in score-descending order.
///
/// Same metric, same value — it skips the sort, which is the only thing
/// `scores` was used for. A caller ranking one array of detections several ways
/// (TIDE runs eight AP evaluations per category over the same ranking, via
/// `average_precision_ranked_into`) sorts once and calls this family; sorting
/// stably twice and sorting stably once produce the same permutation, so the two
/// entry points are bit-identical on sorted input.
///
/// `matched[i]` and `ignored[i]` describe the detection at rank `i`. Passing an
/// unsorted ranking is not an error — it computes the AP of *that* ranking, which
/// is a different (and generally lower) number.
///
/// Returns `0.0` for no detections or no ground truth, matching
/// [`average_precision`]; see the [module note](self) on empty-set conventions.
///
/// # Panics
///
/// If `ignored` is supplied with a different length than `matched`.
pub fn average_precision_ranked(
    matched: &[bool],
    ignored: Option<&[bool]>,
    num_gt: usize,
    rec_thrs: &[f64],
) -> f64 {
    let mut scratch = ApScratch::default();
    average_precision_ranked_into(matched, ignored, num_gt, rec_thrs, &mut scratch)
}

/// [`average_precision_ranked`] writing into a caller-owned [`ApScratch`].
///
/// Same metric, same value, same panic contract — the only difference is that the
/// TP/FP cumulative buffers and the PR-curve working buffers are reused from
/// `scratch` instead of allocated per call. This is the form a caller ranking one
/// array of detections several ways wants: TIDE's `category_deltas` calls this
/// (not [`average_precision_ranked`]) eight times per category, so one `ApScratch`
/// per category turns what was up to 32 per-category allocations (four `Vec`s ×
/// eight calls) into four for the whole category.
///
/// # Panics
///
/// If `ignored` is supplied with a different length than `matched`.
pub(crate) fn average_precision_ranked_into(
    matched: &[bool],
    ignored: Option<&[bool]>,
    num_gt: usize,
    rec_thrs: &[f64],
    scratch: &mut ApScratch,
) -> f64 {
    if let Some(ig) = ignored {
        assert_eq!(
            matched.len(),
            ig.len(),
            "matched and ignored must be parallel arrays (got {} vs {})",
            matched.len(),
            ig.len()
        );
    }

    let nd = matched.len();
    if nd == 0 || num_gt == 0 || rec_thrs.is_empty() {
        return 0.0;
    }

    // Identity permutation: the caller's order *is* the ranking.
    average_precision_of_order_into(0..nd, matched, ignored, num_gt, rec_thrs, scratch)
}

/// Average precision by the VOC 2010 "all-points" rule — the exact area under the
/// interpolated precision-recall curve, with no recall grid.
///
/// This is the integration COCO does *not* use. COCO samples the same envelope at
/// [`crate::params::default_rec_thrs`]'s 101 points and averages, which quantizes
/// the result: a class with 2 ground truths and 1 true positive scores 0.504950 on
/// the grid against an exact 0.500000. The error is bounded by roughly `1/101` per
/// class, so it matters most where classes have few instances.
///
/// Open Images specifies this rule — "evaluated as in the PASCAL VOC 2010
/// protocol" — and both reference implementations follow it: TensorFlow's
/// `object_detection.utils.metrics.compute_average_precision` and FiftyOne's
/// `_compute_AP`. Notably FiftyOne uses the 101-point grid for its COCO evaluation
/// and this rule for Open Images, so the split is deliberate, not an oversight.
///
/// Takes cumulative counts because that is what the caller already has; deriving
/// them here would duplicate the score-ordering the accumulator has done.
/// `tp_cum` and `fp_cum` must be in score-descending order and the same length.
/// Returns `0.0` for an empty curve or `num_gt == 0`.
///
/// Runs in one reverse pass with no allocation. The reference builds padded
/// `recall`/`precision` arrays first, but both sentinels turn out to be inert: the
/// leading `precision = 0` is never a summation term, and the trailing
/// `(recall = 1, precision = 0)` contributes `(1 - max_recall) * 0`. They collapse
/// into the loop bounds and the initial envelope value. Sweeping right to left also
/// means the envelope is just the running maximum, so it needs no second pass.
pub fn average_precision_all_points(tp_cum: &[f64], fp_cum: &[f64], num_gt: usize) -> f64 {
    let nd = tp_cum.len();
    if nd == 0 || num_gt == 0 {
        return 0.0;
    }

    let n = num_gt as f64;
    let mut ap = 0.0;
    // Best precision at this recall or beyond. Starts at 0 — the reference's
    // trailing sentinel, which nothing to the right can beat.
    let mut envelope = 0.0f64;

    for i in (0..nd).rev() {
        let denom = tp_cum[i] + fp_cum[i];
        let precision = if denom > 0.0 { tp_cum[i] / denom } else { 0.0 };
        envelope = envelope.max(precision);

        // Divide before subtracting, as the reference does — it builds the recall
        // array first and differences it, so matching that order keeps the
        // arithmetic bit-comparable.
        let recall_prev = if i == 0 { 0.0 } else { tp_cum[i - 1] / n };
        // A step where recall does not move contributes exactly zero, so the
        // reference's explicit filter on that is unnecessary here.
        ap += (tp_cum[i] / n - recall_prev) * envelope;
    }

    ap
}

/// The F-beta score for one precision/recall pair.
///
/// `beta` weights recall relative to precision: `beta = 1` is the harmonic mean
/// (F1), `beta > 1` favors recall, `beta < 1` favors precision. Returns `0.0`
/// when both inputs are zero, where the formula is otherwise `0/0`.
pub fn f_beta(precision: f64, recall: f64, beta: f64) -> f64 {
    let beta2 = beta * beta;
    let denom = beta2 * precision + recall;
    if denom < f64::EPSILON {
        return 0.0;
    }
    (1.0 + beta2) * precision * recall / denom
}

/// The best F-beta achievable anywhere on a precision-recall curve.
///
/// `precisions[i]` is the precision at `recalls[i]`; the pair is the curve
/// [`precision_recall_curve`] produces. Sweeping it answers "how good could this
/// model be at its best operating point?", which is what an F-score reports —
/// unlike AP, which averages over the whole curve.
///
/// Entries where **either** the precision or the recall is negative are skipped:
/// `-1.0` is the crate's "not computed for this configuration" sentinel
/// ([`crate::metrics::is_missing`]), not a real low score, and a sentinel on
/// either axis makes the whole point meaningless. Returns `None` when no entry
/// is valid, so callers pick their own convention for an undefined score rather
/// than inheriting one.
///
/// # Panics
///
/// If `precisions` and `recalls` have different lengths.
pub fn max_f_beta(precisions: &[f64], recalls: &[f64], beta: f64) -> Option<f64> {
    assert_eq!(
        precisions.len(),
        recalls.len(),
        "max_f_beta: precisions and recalls must be parallel arrays (got {} vs {})",
        precisions.len(),
        recalls.len()
    );
    let mut best = f64::NEG_INFINITY;
    for (&p, &r) in precisions.iter().zip(recalls) {
        if crate::metrics::is_missing(p) || crate::metrics::is_missing(r) {
            continue;
        }
        best = best.max(f_beta(p, r, beta));
    }
    (best > f64::NEG_INFINITY).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::greedy::descending_score_key;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The shape guarantees `precision_recall_curve` makes to its callers.
    ///
    /// `report()`'s PR curves and [`max_f_beta`] both read this output directly,
    /// and both assume it is a well-formed curve rather than an arbitrary bag of
    /// points. VOC interpolation makes precision non-increasing in `r_idx`, and
    /// the two-pointer scan advances monotonically, so `detection_ptr` is
    /// non-decreasing and every emitted threshold is genuinely reached.
    #[test]
    fn precision_recall_curve_is_well_formed() {
        let mut rng = StdRng::seed_from_u64(0xC0_1174);
        let rec_thrs = crate::params::default_rec_thrs();

        for case in 0..5000 {
            let nd = rng.random_range(1..=40);
            let num_gt = rng.random_range(1..=25);

            // Cumulative TP/FP over score-descending detections: each step adds
            // one to exactly one of them, or to neither when ignored.
            let (mut tp_cum, mut fp_cum) = (Vec::with_capacity(nd), Vec::with_capacity(nd));
            let (mut tp, mut fp) = (0.0f64, 0.0f64);
            for _ in 0..nd {
                match rng.random_range(0..3) {
                    0 => tp += 1.0,
                    1 => fp += 1.0,
                    _ => {} // ignored: contributes to neither
                }
                tp_cum.push(tp);
                fp_cum.push(fp);
            }

            let (final_recall, curve) = precision_recall_curve(&tp_cum, &fp_cum, num_gt, &rec_thrs);
            let ctx = format!("case {case}: nd={nd} num_gt={num_gt}");

            // The recall a curve reports is the recall its last detection achieves.
            assert!(
                (final_recall - tp_cum[nd - 1] / num_gt as f64).abs() < 1e-12,
                "{ctx}: final_recall {final_recall} disagrees with tp_cum/num_gt"
            );

            let mut prev_r_idx: Option<usize> = None;
            let mut prev_precision = f64::INFINITY;
            let mut prev_ptr = 0usize;

            for &PrPoint {
                rec_thr_idx: r_idx,
                precision,
                detection_rank: ptr,
            } in &curve
            {
                assert!(r_idx < rec_thrs.len(), "{ctx}: r_idx {r_idx} out of range");
                assert!(ptr < nd, "{ctx}: detection_ptr {ptr} out of range");
                assert!(
                    (0.0..=1.0).contains(&precision),
                    "{ctx}: precision {precision} outside [0,1]"
                );

                if let Some(prev) = prev_r_idx {
                    assert!(r_idx > prev, "{ctx}: r_idx went {prev} -> {r_idx}");
                    assert!(
                        precision <= prev_precision + 1e-12,
                        "{ctx}: precision rose {prev_precision} -> {precision} at r_idx {r_idx}"
                    );
                    assert!(
                        ptr >= prev_ptr,
                        "{ctx}: detection_ptr went backwards {prev_ptr} -> {ptr}"
                    );
                }

                // An emitted threshold must actually be reached by that detection.
                assert!(
                    tp_cum[ptr] / num_gt as f64 >= rec_thrs[r_idx] - 1e-12,
                    "{ctx}: r_idx {r_idx} emitted at ptr {ptr} which does not reach it"
                );

                prev_r_idx = Some(r_idx);
                prev_precision = precision;
                prev_ptr = ptr;
            }

            // Thresholds are emitted exactly while they remain reachable, so the
            // curve is a prefix of the grid.
            let reachable = rec_thrs.iter().filter(|&&t| final_recall >= t).count();
            assert_eq!(
                curve.len(),
                reachable,
                "{ctx}: emitted {} points for {reachable} reachable thresholds \
                 (final_recall {final_recall})",
                curve.len()
            );
        }
    }

    /// All-points AP, derived by hand rather than recorded from this crate's output.
    ///
    /// Each case is small enough to integrate on paper, which is the point: the
    /// end-to-end check against TensorFlow lives in `tests/test_parity_oid.py`, and
    /// this pins the arithmetic so a failure there localizes to the reference
    /// rather than to this function.
    #[test]
    fn all_points_ap_matches_hand_derived_values() {
        // 2 GT, 1 found. Envelope is precision 1.0 over recall [0, 0.5], then 0.
        //   AP = (0.5 - 0.0) * 1.0 + (1.0 - 0.5) * 0.0 = 0.5
        assert_eq!(average_precision_all_points(&[1.0], &[0.0], 2), 0.5);

        // 2 GT, both found, no false positives — precision 1.0 across the board.
        //   AP = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert_eq!(
            average_precision_all_points(&[1.0, 2.0], &[0.0, 0.0], 2),
            1.0
        );

        // 1 GT, a false positive ranked above the true positive.
        //   raw:      recall [0.0, 1.0], precision [0.0, 0.5]
        //   envelope: precision 0.5 everywhere to the left of full recall
        //   AP = (1.0 - 0.0) * 0.5 = 0.5
        assert_eq!(
            average_precision_all_points(&[0.0, 1.0], &[1.0, 1.0], 1),
            0.5
        );

        // The quantization this function exists to avoid: the 101-point grid
        // reports 51/101 for the first case above, not 0.5.
        let grid = average_precision(&[0.9], &[true], None, 2, &crate::params::default_rec_thrs());
        assert!((grid - 51.0 / 101.0).abs() < 1e-12);
        assert!(
            (grid - 0.5).abs() > 1e-3,
            "the two integrations must actually differ"
        );

        // Degenerate inputs agree with the empty-set convention in the module note.
        assert_eq!(average_precision_all_points(&[], &[], 5), 0.0);
        assert_eq!(average_precision_all_points(&[1.0], &[0.0], 0), 0.0);
    }

    /// `f_beta` is a weighted harmonic mean, so it is bounded by its inputs and
    /// collapses to them when they agree.
    #[test]
    fn f_beta_algebraic_properties() {
        let mut rng = StdRng::seed_from_u64(0xFBE7A);

        for case in 0..20000 {
            let p: f64 = rng.random_range(0.0..=1.0);
            let r: f64 = rng.random_range(0.0..=1.0);
            let beta: f64 = rng.random_range(0.1..=5.0);

            let f = f_beta(p, r, beta);
            let ctx = format!("case {case}: p={p} r={r} beta={beta}");

            assert!((0.0..=1.0).contains(&f), "{ctx}: f_beta {f} outside [0,1]");
            // A mean cannot exceed its largest input nor fall below its smallest.
            assert!(f <= p.max(r) + 1e-12, "{ctx}: f_beta {f} above max(p,r)");
            assert!(f >= p.min(r) - 1e-12, "{ctx}: f_beta {f} below min(p,r)");

            // Equal inputs collapse to that value for every beta — the weighting
            // has nothing left to trade off.
            let equal = f_beta(p, p, beta);
            assert!(
                (equal - p).abs() < 1e-12,
                "{ctx}: f_beta(p, p, beta) = {equal}, expected {p}"
            );

            // max_f_beta is a maximum over the curve, so it dominates every point.
            if let Some(best) = max_f_beta(&[p], &[r], beta) {
                assert!(
                    (best - f).abs() < 1e-12,
                    "{ctx}: max over one point != that point"
                );
            }
        }
    }

    #[test]
    fn f_beta_at_one_is_the_harmonic_mean() {
        assert!((f_beta(0.5, 0.5, 1.0) - 0.5).abs() < 1e-12);
        // Harmonic mean of 1.0 and 0.5 is 2/3.
        assert!((f_beta(1.0, 0.5, 1.0) - 2.0 / 3.0).abs() < 1e-12);
        // Both zero would be 0/0; defined as 0.
        assert_eq!(f_beta(0.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn beta_shifts_the_weight_between_precision_and_recall() {
        // High precision, low recall. beta < 1 favors precision, so scores higher.
        let (p, r) = (0.9, 0.3);
        assert!(f_beta(p, r, 0.5) > f_beta(p, r, 1.0));
        assert!(f_beta(p, r, 2.0) < f_beta(p, r, 1.0));
    }

    #[test]
    fn max_f_beta_sweeps_the_curve_for_the_best_point() {
        // Best F1 is at the middle point: f_beta(0.6, 0.6) = 0.6.
        let precisions = [1.0, 0.6, 0.2];
        let recalls = [0.1, 0.6, 0.9];
        let best = max_f_beta(&precisions, &recalls, 1.0).expect("a valid point exists");
        assert!((best - 0.6).abs() < 1e-12);
    }

    #[test]
    fn max_f_beta_skips_the_missing_data_sentinel() {
        // -1.0 means "not computed", not "precision of -1".
        assert_eq!(max_f_beta(&[-1.0, -1.0], &[0.5, 0.5], 1.0), None);
        let best = max_f_beta(&[-1.0, 0.5], &[0.1, 0.5], 1.0).expect("one valid point");
        assert!((best - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empty_or_no_gt_is_zero() {
        assert_eq!(precision_recall_curve(&[], &[], 5, &[0.5]), (0.0, vec![]));
        assert_eq!(
            precision_recall_curve(&[1.0], &[0.0], 0, &[0.5]),
            (0.0, vec![])
        );
    }

    #[test]
    fn perfect_detections_precision_one() {
        // 4 TPs, no FPs, 4 GTs => recall reaches 1.0, precision 1.0 throughout.
        let tp = [1.0, 2.0, 3.0, 4.0];
        let fp = [0.0, 0.0, 0.0, 0.0];
        let (final_recall, curve) = precision_recall_curve(&tp, &fp, 4, &[0.0, 0.5, 1.0]);
        assert!((final_recall - 1.0).abs() < 1e-12);
        assert_eq!(curve.len(), 3);
        for p in &curve {
            assert!((p.precision - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn unreachable_recall_thresholds_omitted() {
        // 1 TP among 4 GTs => max recall 0.25; thresholds above are dropped.
        let tp = [1.0, 1.0];
        let fp = [0.0, 1.0];
        let (final_recall, curve) = precision_recall_curve(&tp, &fp, 4, &[0.1, 0.25, 0.5, 1.0]);
        assert!((final_recall - 0.25).abs() < 1e-12);
        // only the 0.1 and 0.25 thresholds are reachable
        assert_eq!(
            curve.iter().map(|c| c.rec_thr_idx).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn voc_interpolation_makes_precision_monotone() {
        // Raw precision dips then recovers; interpolation lifts the dip to the
        // later higher value. Ranks: tp=[1,1,2], fp=[0,1,1] => pr=[1, .5, .667],
        // recall=[.33,.33,.67]. After right-to-left max: [1, .667, .667].
        let tp = [1.0, 1.0, 2.0];
        let fp = [0.0, 1.0, 1.0];
        let (_, curve) = precision_recall_curve(&tp, &fp, 3, &[0.5]);
        // recall 0.5 first met at rank 2 (recall .667); interpolated precision .667
        assert_eq!(curve.len(), 1);
        let p = curve[0];
        assert_eq!(p.detection_rank, 2);
        assert!((p.precision - 2.0 / 3.0).abs() < 1e-12);
    }

    /// NaN scores must not scramble the ranking or panic the sort. `total_cmp`
    /// orders positive NaN above every number, so a NaN-scored detection ranks
    /// first — deterministically — and the AP is a hand-derivable value.
    #[test]
    fn nan_scores_rank_deterministically_instead_of_scrambling() {
        let rec_thrs = crate::params::default_rec_thrs();

        // NaN FP ranked above the real TP: tp_cum=[0,1], fp_cum=[1,1] =>
        // precision 0.5 at every reached threshold => AP = 0.5.
        let ap = average_precision(&[f64::NAN, 0.9], &[false, true], None, 1, &rec_thrs);
        assert!((ap - 0.5).abs() < 1e-12, "got {ap}");

        // Same arrays, NaN detection is the TP: precision 1.0 => AP = 1.0.
        let ap = average_precision(&[f64::NAN, 0.9], &[true, false], None, 1, &rec_thrs);
        assert!((ap - 1.0).abs() < 1e-12, "got {ap}");

        // Negative NaN sorts below every number — the TP at 0.9 stays first.
        let ap = average_precision(&[-f64::NAN, 0.9], &[false, true], None, 1, &rec_thrs);
        assert!((ap - 1.0).abs() < 1e-12, "got {ap}");

        // A larger NaN-laced array must not panic (std sort panics on a
        // non-total order since Rust 1.81).
        let scores: Vec<f64> = (0..50)
            .map(|i| {
                if i % 7 == 0 {
                    f64::NAN
                } else {
                    i as f64 / 50.0
                }
            })
            .collect();
        let matched: Vec<bool> = (0..50).map(|i| i % 2 == 0).collect();
        let ap = average_precision(&scores, &matched, None, 25, &rec_thrs);
        assert!(ap.is_finite());
    }

    /// The two AP entry points share one body: sorting first and delegating must
    /// equal calling the ranked form on pre-sorted input.
    #[test]
    fn sorted_and_ranked_entry_points_agree() {
        let mut rng = StdRng::seed_from_u64(0xAB5EED);
        let rec_thrs = crate::params::default_rec_thrs();

        for _ in 0..500 {
            let nd = rng.random_range(1..=20);
            let num_gt = rng.random_range(1..=10);
            let mut scores: Vec<f64> = (0..nd).map(|_| rng.random_range(0.0..=1.0)).collect();
            scores.sort_by(|a, b| b.total_cmp(a));
            let matched: Vec<bool> = (0..nd).map(|_| rng.random_bool(0.5)).collect();
            let ignored: Vec<bool> = (0..nd).map(|_| rng.random_bool(0.2)).collect();

            let a = average_precision(&scores, &matched, Some(&ignored), num_gt, &rec_thrs);
            let b = average_precision_ranked(&matched, Some(&ignored), num_gt, &rec_thrs);
            assert_eq!(a, b, "sorted input must make the two forms bit-identical");
        }
    }

    #[test]
    #[should_panic(expected = "parallel arrays")]
    fn average_precision_rejects_mismatched_lengths() {
        average_precision(&[0.9, 0.8], &[true], None, 1, &[0.5]);
    }

    #[test]
    #[should_panic(expected = "parallel arrays")]
    fn average_precision_rejects_mismatched_ignored() {
        average_precision(&[0.9], &[true], Some(&[false, true]), 1, &[0.5]);
    }

    #[test]
    #[should_panic(expected = "parallel arrays")]
    fn average_precision_ranked_rejects_mismatched_ignored() {
        average_precision_ranked(&[true, false], Some(&[false]), 1, &[0.5]);
    }

    /// A lone leading true positive reads [`coco_precision`]`(1, 0)`, one ulp
    /// under 1.0, on both curve paths — the only place the guard term shows.
    #[test]
    fn both_curve_paths_carry_the_pycocotools_guard_term() {
        let rec_thrs = [0.0, 0.5, 1.0];
        let lone_tp = coco_precision(1.0, 0.0);
        assert!(
            lone_tp < 1.0 && lone_tp > 1.0 - 2.0 * f64::EPSILON,
            "{lone_tp}"
        );

        // Rank 0 TP, rank 1 FP: the envelope has nothing above the lone TP.
        let (_, curve) = precision_recall_curve(&[1.0, 1.0], &[0.0, 1.0], 2, &rec_thrs);
        assert_eq!(curve[0].precision, lone_tp);

        let mut scratch = PrCurveScratch::default();
        let mut out = Vec::new();
        precision_recall_curve_of_order_into(
            [0, 1],
            &[true, false],
            None,
            2,
            &rec_thrs,
            &mut scratch,
            &mut out,
        );
        assert_eq!(out[0].1, lone_tp);
    }

    #[test]
    #[should_panic(expected = "parallel arrays")]
    fn precision_recall_curve_rejects_mismatched_lengths() {
        precision_recall_curve(&[1.0, 2.0], &[0.0], 2, &[0.5]);
    }

    #[test]
    #[should_panic(expected = "parallel arrays")]
    fn max_f_beta_rejects_mismatched_lengths() {
        max_f_beta(&[0.5, 0.6], &[0.5], 1.0);
    }

    /// The sentinel is skipped on the recall axis too — a `-1.0` recall with a
    /// valid precision must not produce a negative "best F-score".
    #[test]
    fn max_f_beta_skips_the_sentinel_in_recalls() {
        assert_eq!(max_f_beta(&[0.5, 0.5], &[-1.0, -1.0], 1.0), None);
        let best = max_f_beta(&[0.5, 0.8], &[-1.0, 0.8], 1.0).expect("one valid point");
        assert!((best - 0.8).abs() < 1e-12);
        // A sentinel on either axis alone invalidates that point, not the sweep.
        let best = max_f_beta(&[-1.0, 0.6, 0.9], &[0.4, -1.0, 0.9], 1.0).expect("one valid point");
        assert!((best - 0.9).abs() < 1e-12);
    }

    /// The fused kernel is bit-identical to `cumulative_tp_fp` followed by
    /// `precision_recall_curve_into` — same `final_recall`, same tuples, same
    /// order — everywhere `detection::accumulate` can take it.
    ///
    /// The cases that could tell the two apart are all here on purpose: no
    /// detections, no ground truth, a single rank, ignored detections at the
    /// front and back (`pr` at rank 0 is the `total == 0` zero), recall landing
    /// exactly on a threshold (`19 / 20` against `0.95`, and against the
    /// `0.9500000000000001` the reference grid does *not* contain), unsorted and
    /// duplicated `rec_thrs` (the two-pointer scan never rewinds), a `NaN`
    /// threshold (recorded at the current rank in both, because the predicate is
    /// the negation of `rc < thr`, not `rc >= thr`), empty `rec_thrs`, and a
    /// non-identity `order`.
    #[test]
    fn fused_curve_matches_cumulative_then_interpolate_bit_for_bit() {
        fn legacy(
            order: &[usize],
            matched: &[bool],
            ignored: Option<&[bool]>,
            num_gt: usize,
            rec_thrs: &[f64],
        ) -> (f64, Vec<(usize, f64, usize)>) {
            let (mut tp, mut fp) = (Vec::new(), Vec::new());
            cumulative_tp_fp(order.iter().copied(), matched, ignored, &mut tp, &mut fp);
            let mut scratch = PrCurveScratch::default();
            let mut out = Vec::new();
            let fr =
                precision_recall_curve_into(&tp, &fp, num_gt, rec_thrs, &mut scratch, &mut out);
            (fr, out)
        }
        fn fused(
            order: &[usize],
            matched: &[bool],
            ignored: Option<&[bool]>,
            num_gt: usize,
            rec_thrs: &[f64],
        ) -> (f64, Vec<(usize, f64, usize)>) {
            let mut scratch = PrCurveScratch::default();
            // Pre-seeded with garbage: the kernel must clear it.
            let mut out = vec![(usize::MAX, f64::NAN, usize::MAX)];
            let fr = precision_recall_curve_of_order_into(
                order.iter().copied(),
                matched,
                ignored,
                num_gt,
                rec_thrs,
                &mut scratch,
                &mut out,
            );
            (fr, out)
        }
        fn assert_same(
            label: &str,
            (fr_a, out_a): (f64, Vec<(usize, f64, usize)>),
            (fr_b, out_b): (f64, Vec<(usize, f64, usize)>),
        ) {
            assert_eq!(fr_a.to_bits(), fr_b.to_bits(), "{label}: final_recall");
            assert_eq!(out_a.len(), out_b.len(), "{label}: point count");
            for (i, (a, b)) in out_a.iter().zip(&out_b).enumerate() {
                assert_eq!(a.0, b.0, "{label}: rec_thr_idx at point {i}");
                assert_eq!(
                    a.1.to_bits(),
                    b.1.to_bits(),
                    "{label}: precision at point {i}"
                );
                assert_eq!(a.2, b.2, "{label}: detection_rank at point {i}");
            }
        }

        let grid = crate::params::default_rec_thrs();
        let odd_grids: [&[f64]; 5] = [
            &[],
            &[0.95, 0.9500000000000001, 0.95],
            &[0.5, 0.9, 0.3, 0.3, 1.0, 0.0],
            &[0.2, f64::NAN, 0.4, 0.6],
            &[1.5],
        ];

        // Hand-built boundary cases.
        let m19 = {
            // 19 TP among 20 ranks, one FP in the middle: recall lands on 19/20.
            let mut v = vec![true; 20];
            v[7] = false;
            v
        };
        let id20: Vec<usize> = (0..20).collect();
        let ig_ends = {
            let mut v = vec![false; 20];
            v[0] = true;
            v[19] = true;
            v
        };
        struct Case {
            label: &'static str,
            order: Vec<usize>,
            matched: Vec<bool>,
            ignored: Option<Vec<bool>>,
            num_gt: usize,
        }
        let case = |label, order, matched, ignored, num_gt| Case {
            label,
            order,
            matched,
            ignored,
            num_gt,
        };
        let cases = [
            case("empty order", vec![], vec![], None, 3),
            case("no ground truth", id20.clone(), m19.clone(), None, 0),
            case("single tp", vec![0], vec![true], None, 1),
            case("single fp", vec![0], vec![false], None, 1),
            case("single ignored", vec![0], vec![true], Some(vec![true]), 1),
            case("19 of 20", id20.clone(), m19.clone(), None, 20),
            case(
                "ignored at both ends",
                id20.clone(),
                m19.clone(),
                Some(ig_ends),
                20,
            ),
            case(
                "all ignored",
                id20.clone(),
                m19.clone(),
                Some(vec![true; 20]),
                5,
            ),
            case(
                "reversed order",
                (0..20).rev().collect(),
                m19.clone(),
                None,
                20,
            ),
        ];
        for Case {
            label,
            order,
            matched,
            ignored,
            num_gt,
        } in &cases
        {
            let ig = ignored.as_deref();
            assert_same(
                &format!("{label} / grid"),
                legacy(order, matched, ig, *num_gt, &grid),
                fused(order, matched, ig, *num_gt, &grid),
            );
            for (g, rec_thrs) in odd_grids.iter().enumerate() {
                assert_same(
                    &format!("{label} / odd grid {g}"),
                    legacy(order, matched, ig, *num_gt, rec_thrs),
                    fused(order, matched, ig, *num_gt, rec_thrs),
                );
            }
        }

        // Randomized: sizes the accumulator sees, ignore-heavy and ignore-free,
        // shuffled orders.
        let mut rng = StdRng::seed_from_u64(0xA2_F05E);
        for &nd in &[0usize, 1, 2, 17, 1000] {
            for &num_gt in &[0usize, 1, 7, 20, 1000] {
                for trial in 0..8 {
                    let p_ignored = [0.0, 0.1, 0.6][trial % 3];
                    let matched: Vec<bool> = (0..nd).map(|_| rng.random_bool(0.5)).collect();
                    let ignored: Option<Vec<bool>> = if trial % 2 == 0 {
                        Some((0..nd).map(|_| rng.random_bool(p_ignored)).collect())
                    } else {
                        None
                    };
                    let mut order: Vec<usize> = (0..nd).collect();
                    for i in (1..nd).rev() {
                        order.swap(i, rng.random_range(0..=i));
                    }
                    let ig = ignored.as_deref();
                    let label = format!("random nd={nd} num_gt={num_gt} trial={trial}");
                    assert_same(
                        &format!("{label} / grid"),
                        legacy(&order, &matched, ig, num_gt, &grid),
                        fused(&order, &matched, ig, num_gt, &grid),
                    );
                    assert_same(
                        &format!("{label} / odd grid 2"),
                        legacy(&order, &matched, ig, num_gt, odd_grids[2]),
                        fused(&order, &matched, ig, num_gt, odd_grids[2]),
                    );
                }
            }
        }
    }

    /// The radix ranking is `sort_unstable` on `(key, position)` exactly:
    /// keys with many ties, keys of `float32` scores (most bytes shared),
    /// keys over the whole range, NaN and signed-zero keys, scores a few ulps
    /// apart at several magnitudes (so keys differing only in their low byte,
    /// or only in a middle one, decide the order), and lengths on both sides
    /// of the comparison-sort cutoff.
    #[test]
    fn radix_ranking_is_the_comparison_sort() {
        let mut rng = StdRng::seed_from_u64(0x5_0E7);
        let mut scratch = Vec::new();
        for &n in &[0usize, 1, 2, 511, 512, 513, 5000, 40_000] {
            for kind in 0..5 {
                let scores: Vec<f64> = (0..n)
                    .map(|_| match kind {
                        0 => f64::from(rng.random_range(0..7u8)) / 6.0,
                        1 => f64::from(rng.random::<f32>()),
                        2 => f64::from_bits(rng.random::<u64>()),
                        3 => [0.0, -0.0, f64::NAN, 0.5, 1.0, -1.0, f64::INFINITY]
                            [rng.random_range(0..7)],
                        _ => {
                            let base = [0.3f64, 0.5, 0.9][rng.random_range(0..3)].to_bits();
                            let step = 1u64 << (8 * rng.random_range(0..4u32));
                            f64::from_bits(base + step * rng.random_range(0..200u64))
                        }
                    })
                    .collect();
                let mut want: Vec<(u64, u32)> = scores
                    .iter()
                    .zip(0u32..)
                    .map(|(&s, i)| (descending_score_key(s), i))
                    .collect();
                let mut got = want.clone();
                want.sort_unstable();
                sort_score_keyed(&mut got, &mut scratch);
                assert_eq!(got, want, "n={n} kind={kind}");
            }
        }
    }

    /// The multi-threshold kernel is, at every lane, bit-identical to the
    /// single-lane kernel run on that lane's flags — same `final_recall`, same
    /// tuples, same order.
    ///
    /// What could tell them apart is here on purpose: ranks ignored at every
    /// lane (counted once, not per lane) beside ranks ignored at some, set bits
    /// above `lanes` (which must not be read), one lane and the full 64, rank 0
    /// recording at lanes that are not true positives there, no detections, no
    /// ground truth, and the odd recall grids the single-lane test uses.
    #[test]
    fn lane_curves_match_the_single_lane_kernel_at_every_lane() {
        let grid = crate::params::default_rec_thrs();
        let odd_grids: [&[f64]; 5] = [
            &[],
            &[0.95, 0.9500000000000001, 0.95],
            &[0.5, 0.9, 0.3, 0.3, 1.0, 0.0],
            &[0.2, f64::NAN, 0.4, 0.6],
            &[1.5],
        ];
        let bit = |w: u64, l: usize| w >> l & 1 == 1;
        let check = |label: &str,
                     order: &[usize],
                     matched: &[u64],
                     ignored: &[u64],
                     num_gts: &[usize],
                     group: usize,
                     rec_thrs: &[f64],
                     curves: &mut LaneCurves| {
            // Garbage from a previous call must not survive.
            curves
                .points
                .iter_mut()
                .for_each(|p| p.push((7, f64::NAN, 7)));
            curves.final_recall.iter_mut().for_each(|r| *r = f64::NAN);
            precision_recall_curves_of_lanes_into(
                order.iter().copied(),
                matched,
                ignored,
                num_gts,
                group,
                rec_thrs,
                curves,
            );
            for (l, &num_gt) in num_gts.iter().enumerate() {
                let m: Vec<bool> = matched.iter().map(|&w| bit(w, l)).collect();
                let ig: Vec<bool> = ignored.iter().map(|&w| bit(w, l)).collect();
                let mut scratch = PrCurveScratch::default();
                let mut want = Vec::new();
                let fr = precision_recall_curve_of_order_into(
                    order.iter().copied(),
                    &m,
                    Some(&ig),
                    num_gt,
                    rec_thrs,
                    &mut scratch,
                    &mut want,
                );
                let got = &curves.points[l];
                assert_eq!(
                    curves.final_recall[l].to_bits(),
                    fr.to_bits(),
                    "{label} lane {l}: final_recall"
                );
                assert_eq!(got.len(), want.len(), "{label} lane {l}: point count");
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(a.0, b.0, "{label} lane {l}: rec_thr_idx at point {i}");
                    assert_eq!(
                        a.1.to_bits(),
                        b.1.to_bits(),
                        "{label} lane {l}: precision at point {i}"
                    );
                    assert_eq!(a.2, b.2, "{label} lane {l}: detection_rank at point {i}");
                }
            }
        };

        let mut curves = LaneCurves::default();
        let mut rng = StdRng::seed_from_u64(0x1A_9E5);
        for &lanes in &[1usize, 10, 63, 64] {
            let all = u64::MAX >> (64 - lanes);
            for &nd in &[0usize, 1, 2, 17, 600] {
                for &num_gt in &[0usize, 1, 7, 300] {
                    for trial in 0..6 {
                        // Matched mostly at low lanes, as IoU thresholds rise.
                        let matched: Vec<u64> = (0..nd)
                            .map(|_| {
                                let upto = rng.random_range(0..=lanes + 2);
                                let lanes_matched = if upto >= lanes {
                                    all
                                } else {
                                    (1u64 << upto) - 1
                                };
                                // Garbage above `lanes`, which must not be read.
                                lanes_matched | (rng.random::<u64>() & !all)
                            })
                            .collect();
                        // Lanes in groups — one per area range in the
                        // accumulator — every divisor of the lane count in
                        // turn, ignored a whole group at a time, at random
                        // lanes, or not at all.
                        let divisors: Vec<usize> = (1..=lanes).filter(|g| lanes % g == 0).collect();
                        let group = divisors[trial % divisors.len()];
                        let width_mask = u64::MAX >> (64 - group);
                        let p_whole = [0.0, 0.3, 0.8][trial % 3];
                        let ignored: Vec<u64> = (0..nd)
                            .map(|_| {
                                let mut lanes_ignored = 0u64;
                                for g in 0..lanes / group {
                                    let gm = width_mask << (g * group);
                                    lanes_ignored |= if rng.random_bool(p_whole) {
                                        gm
                                    } else if trial % 2 == 0 {
                                        rng.random::<u64>() & gm
                                    } else {
                                        0
                                    };
                                }
                                lanes_ignored | (rng.random::<u64>() & !all)
                            })
                            .collect();
                        let mut order: Vec<usize> = (0..nd).collect();
                        for i in (1..nd).rev() {
                            order.swap(i, rng.random_range(0..=i));
                        }
                        // One count for every lane, or one per lane, as area
                        // ranges packed side by side have, with lanes of no
                        // ground truth among them.
                        let num_gts: Vec<usize> = if trial < 3 {
                            vec![num_gt; lanes]
                        } else {
                            (0..lanes)
                                .map(|l| {
                                    if l % 5 == 2 {
                                        0
                                    } else {
                                        rng.random_range(0..=num_gt)
                                    }
                                })
                                .collect()
                        };
                        let label = format!(
                            "lanes={lanes} group={group} nd={nd} num_gt={num_gt} trial={trial}"
                        );
                        check(
                            &format!("{label} / grid"),
                            &order,
                            &matched,
                            &ignored,
                            &num_gts,
                            group,
                            &grid,
                            &mut curves,
                        );
                        for (g, rec_thrs) in odd_grids.iter().enumerate() {
                            check(
                                &format!("{label} / odd grid {g}"),
                                &order,
                                &matched,
                                &ignored,
                                &num_gts,
                                group,
                                rec_thrs,
                                &mut curves,
                            );
                        }
                    }
                }
            }
        }
    }
}
