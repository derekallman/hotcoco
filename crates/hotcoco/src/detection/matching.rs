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

use rayon::prelude::*;

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
    /// Count of `in_denominator_sorted` — [`EvalImg::num_gt_in_denominator`].
    num_in_denominator: usize,
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

/// What `accumulate()` reads from every (image, category) pair `evaluate()`
/// gathered, and nothing else: the score list once, and per area range the
/// ground-truth denominator plus a matched bit and an ignore bit per detection
/// per IoU threshold — in flat arenas, one header per pair pointing into them.
///
/// An [`EvalImg`] carries the same cell with every id, the ground-truth side of
/// the match, and its own copy of the scores, one per area range. This holds
/// no vector per pair and no record per area range. `evaluate()` builds it
/// through [`Cells::build`]; `EvalImg`s are built by
/// [`COCOeval::eval_imgs`](super::COCOeval::eval_imgs) on first access.
#[derive(Debug, Clone, Default)]
pub(super) struct Cells {
    /// One per pair, plus a sentinel whose offsets are the arenas' lengths,
    /// so a pair's score count is the gap to the next header.
    pairs: Vec<PairHeader>,
    /// IoU thresholds and area ranges at evaluate time: the row and block
    /// counts of every pair's bits.
    n_thr: usize,
    n_areas: usize,
    /// Every pair's scores, score-descending and cut at `max_det`, back to back.
    scores: Vec<f64>,
    /// Per pair, per area range: how many ground truths count toward recall.
    num_gt: Vec<u32>,
    /// Per pair, from a word boundary: `n_areas` blocks of `2 * n_thr` rows of
    /// `nd` bits — the matched rows, then the ignore rows.
    bits: Vec<u64>,
}

/// One pair's place in the [`Cells`] arenas.
#[derive(Debug, Clone, Copy, Default)]
struct PairHeader {
    image_id: u64,
    category_id: u64,
    scores_start: u32,
    /// Word index into `bits`.
    bits_start: u32,
}

/// One (image, category, area range) cell: a pair's index in [`Cells`] and
/// which of its area ranges, by evaluate-time position.
#[derive(Debug, Clone, Copy)]
pub(super) struct CellRef {
    pair: u32,
    area: u32,
}

impl CellRef {
    pub(super) fn new(pair: usize, area: usize) -> Self {
        CellRef {
            pair: arena_index(pair),
            area: arena_index(area),
        }
    }

    pub(super) fn pair(self) -> usize {
        self.pair as usize
    }

    pub(super) fn area(self) -> usize {
        self.area as usize
    }
}

/// Arena positions are `u32`, like annotation index positions.
fn arena_index(i: usize) -> u32 {
    u32::try_from(i).expect("cell arena positions are u32")
}

/// The bit words a pair of `nd` detections takes: `2 * n_thr` rows of `nd`
/// bits per area range, packed end to end.
fn words_for(n_thr: usize, n_areas: usize, nd: usize) -> usize {
    (2 * n_thr * n_areas * nd).div_ceil(64)
}

/// A pair with nothing to gather, in the layout pass of [`Cells::build`].
const NO_PAIR: u32 = u32::MAX;

impl Cells {
    /// The arena for `pairs`, sized exactly before anything is written: `nd`
    /// says how many scores a pair will push (`None` for one with nothing to
    /// gather), then `fill` writes each run of `run_len` pairs through its
    /// [`CellWriter`], in parallel, into that run's window of every arena.
    /// `fill` must push exactly the pairs `nd` admitted, with those counts,
    /// and `run_len` is at least one.
    pub(super) fn build(
        params: &Params,
        pairs: &[(u64, u64)],
        run_len: usize,
        nd: impl Fn(u64, u64) -> Option<usize> + Sync,
        fill: impl Fn(&[(u64, u64)], &mut CellWriter<'_>) + Sync,
    ) -> Self {
        let (n_thr, n_areas) = (params.iou_thrs.len(), params.area_ranges.len());
        let nds: Vec<u32> = pairs
            .par_iter()
            .map(|&(img_id, cat_id)| nd(img_id, cat_id).map_or(NO_PAIR, arena_index))
            .collect();
        // Per run: pairs kept, scores, and bit words.
        let runs: Vec<(usize, usize, usize)> = nds
            .par_chunks(run_len)
            .map(|run| {
                run.iter()
                    .filter(|&&nd| nd != NO_PAIR)
                    .fold((0, 0, 0), |(p, s, w), &nd| {
                        (
                            p + 1,
                            s + nd as usize,
                            w + words_for(n_thr, n_areas, nd as usize),
                        )
                    })
            })
            .collect();
        let (n_pairs, n_scores, n_words) =
            runs.iter().fold((0, 0, 0), |(p, s, w), &(rp, rs, rw)| {
                (p + rp, s + rs, w + rw)
            });
        let mut cells = Cells {
            pairs: vec![PairHeader::default(); n_pairs + 1],
            n_thr,
            n_areas,
            scores: vec![0.0; n_scores],
            num_gt: vec![0; n_pairs * n_areas],
            bits: vec![0; n_words],
        };
        cells.pairs[n_pairs] = PairHeader {
            scores_start: arena_index(n_scores),
            bits_start: arena_index(n_words),
            ..PairHeader::default()
        };
        // One writer per run, over disjoint windows of every arena.
        let (mut ph, mut sc, mut ng, mut bw) = (
            &mut cells.pairs[..n_pairs],
            &mut cells.scores[..],
            &mut cells.num_gt[..],
            &mut cells.bits[..],
        );
        let mut writers = Vec::with_capacity(runs.len());
        let (mut scores_base, mut bits_base) = (0, 0);
        for &(p, s, w) in &runs {
            let (pairs, rest) = std::mem::take(&mut ph).split_at_mut(p);
            ph = rest;
            let (scores, rest) = std::mem::take(&mut sc).split_at_mut(s);
            sc = rest;
            let (num_gt, rest) = std::mem::take(&mut ng).split_at_mut(p * n_areas);
            ng = rest;
            let (bits, rest) = std::mem::take(&mut bw).split_at_mut(w);
            bw = rest;
            writers.push(CellWriter {
                n_thr,
                n_areas,
                pairs,
                scores,
                num_gt,
                bits,
                scores_base,
                bits_base,
                n_pairs: 0,
                n_scores: 0,
                n_words: 0,
                nd: 0,
                areas_done: n_areas,
                bit: 0,
            });
            scores_base += s;
            bits_base += w;
        }
        writers
            .par_iter_mut()
            .zip(pairs.par_chunks(run_len))
            .for_each(|(writer, run)| {
                fill(run, writer);
                debug_assert!(writer.is_full(), "a run wrote what its layout pass counted");
            });
        cells
    }

    /// The number of pairs.
    pub(super) fn len(&self) -> usize {
        self.pairs.len().saturating_sub(1)
    }

    /// The pair's image and category ids.
    pub(super) fn ids(&self, pair: usize) -> (u64, u64) {
        let h = &self.pairs[pair];
        (h.image_id, h.category_id)
    }

    /// The pair's scores, descending, cut at evaluate time's `max_det`.
    pub(super) fn scores(&self, pair: usize) -> &[f64] {
        let (start, end) = (
            self.pairs[pair].scores_start as usize,
            self.pairs[pair + 1].scores_start as usize,
        );
        &self.scores[start..end]
    }

    /// [`EvalImg::num_gt_in_denominator`] for the cell.
    pub(super) fn num_gt(&self, cell: CellRef) -> u32 {
        debug_assert!(cell.area() < self.n_areas);
        self.num_gt[cell.pair() * self.n_areas + cell.area()]
    }

    /// The cell's matched and ignore rows.
    pub(super) fn block(&self, cell: CellRef) -> Block<'_> {
        debug_assert!(cell.area() < self.n_areas);
        let nd = self.scores(cell.pair()).len();
        Block {
            words: &self.bits,
            base: self.pairs[cell.pair()].bits_start as usize * 64
                + cell.area() * 2 * self.n_thr * nd,
            nd,
            n_thr: self.n_thr,
        }
    }
}

/// One cell's bits: `2 * n_thr` rows of `nd`, matched rows then ignore rows.
pub(super) struct Block<'a> {
    words: &'a [u64],
    /// Bit offset of the first row.
    base: usize,
    nd: usize,
    n_thr: usize,
}

impl Block<'_> {
    /// `EvalImg::dt_matched` row `t`, one bit per score.
    pub(super) fn matched(&self, t: usize) -> impl Iterator<Item = bool> + '_ {
        debug_assert!(t < self.n_thr);
        self.row(t)
    }

    /// `EvalImg::dt_ignore` row `t`.
    pub(super) fn ignore(&self, t: usize) -> impl Iterator<Item = bool> + '_ {
        debug_assert!(t < self.n_thr);
        self.row(self.n_thr + t)
    }

    fn row(&self, row: usize) -> impl Iterator<Item = bool> + '_ {
        let start = self.base + row * self.nd;
        (start..start + self.nd).map(|i| (self.words[i / 64] >> (i % 64)) & 1 == 1)
    }
}

/// One run's window of the [`Cells`] arenas, written pair by pair: a
/// [`begin_pair`](Self::begin_pair), then one [`push_area`](Self::push_area)
/// per area range in `params.area_ranges` order.
pub(super) struct CellWriter<'a> {
    n_thr: usize,
    n_areas: usize,
    pairs: &'a mut [PairHeader],
    scores: &'a mut [f64],
    num_gt: &'a mut [u32],
    bits: &'a mut [u64],
    /// Where the window starts in the whole arena: what headers record.
    scores_base: usize,
    bits_base: usize,
    /// How much of the window is written.
    n_pairs: usize,
    n_scores: usize,
    n_words: usize,
    /// The pair being written: its score count, how many of its area ranges
    /// are in, and the bit cursor within `bits`.
    nd: usize,
    areas_done: usize,
    bit: usize,
}

impl CellWriter<'_> {
    /// Start a pair with its scores; its bit words are reserved here.
    pub(super) fn begin_pair(&mut self, image_id: u64, category_id: u64, scores: &[f64]) {
        debug_assert_eq!(
            self.areas_done, self.n_areas,
            "the previous pair pushed every area range"
        );
        self.pairs[self.n_pairs] = PairHeader {
            image_id,
            category_id,
            scores_start: arena_index(self.scores_base + self.n_scores),
            bits_start: arena_index(self.bits_base + self.n_words),
        };
        self.scores[self.n_scores..self.n_scores + scores.len()].copy_from_slice(scores);
        self.nd = scores.len();
        self.areas_done = 0;
        self.bit = self.n_words * 64;
        self.n_pairs += 1;
        self.n_scores += scores.len();
        self.n_words += words_for(self.n_thr, self.n_areas, scores.len());
    }

    /// The current pair's next area range: its denominator, then its
    /// `2 * n_thr` rows of `nd` bits, matched rows first.
    pub(super) fn push_area<'r>(&mut self, num_gt: u32, rows: impl Iterator<Item = &'r [bool]>) {
        debug_assert!(self.areas_done < self.n_areas);
        self.num_gt[(self.n_pairs - 1) * self.n_areas + self.areas_done] = num_gt;
        let mut n_rows = 0;
        for row in rows {
            debug_assert_eq!(row.len(), self.nd);
            for &set in row {
                self.bits[self.bit / 64] |= (set as u64) << (self.bit % 64);
                self.bit += 1;
            }
            n_rows += 1;
        }
        debug_assert_eq!(n_rows, 2 * self.n_thr);
        self.areas_done += 1;
    }

    /// Whether every arena window is written to its end.
    fn is_full(&self) -> bool {
        self.areas_done == self.n_areas
            && self.n_pairs == self.pairs.len()
            && self.n_scores == self.scores.len()
            && self.n_words == self.bits.len()
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

/// The pair's ground-truth and detection ids, or `None` when both are empty.
///
/// pycocotools' `evaluateImg` skips a cell only when `len(gt) == 0 and
/// len(dt) == 0` on the *raw* per-(image, category) lists — before any area
/// range ignores anything and before the `max_det` cut — and that is the only
/// skip here too. Anything narrower is wrong in a way AP never shows: a cell
/// with detections but no ground truth, every one of them outside the area
/// range, has nothing to match and moves no counter, yet its detections still
/// occupy ranks in `accumulate()`'s score order, and the score sampled at a
/// recall threshold (`eval["scores"]`) is read off that order. Dropping such
/// cells shifted those samples onto later detections.
fn pair_ids<'a>(
    ctx: &EvalImgContext<'a>,
    img_id: u64,
    cat_id: u64,
) -> Option<(&'a [u64], &'a [u64])> {
    let gt_ids = super::COCOeval::get_anns_static(ctx.coco_gt, ctx.params, img_id, cat_id);
    let dt_ids = super::COCOeval::get_anns_static(ctx.coco_dt, ctx.params, img_id, cat_id);
    (!gt_ids.is_empty() || !dt_ids.is_empty()).then_some((gt_ids, dt_ids))
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
    let (gt_ids, dt_ids) = pair_ids(ctx, img_id, cat_id)?;

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
    // The index holds every annotation, so no id is dropped above:
    // `lean_scores_len` sizes the cell arenas on that.
    debug_assert_eq!(with_iou_idx.len(), dt_ids.len().min(max_det));

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
    let num_in_denominator = in_denominator_sorted.iter().filter(|&&x| x).count();

    GtView {
        anns,
        order,
        iou_indices: pair.gt_iou_indices.as_slice(),
        ignore_sorted,
        in_denominator_sorted,
        num_in_denominator,
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

/// Match one area range of a gathered pair: the partitioned ground truth and
/// the match outcome, with the LVIS not-exhaustive rule applied.
///
/// `not_exhaustive_cat` — when true (LVIS mode), unmatched detections are ignored
/// rather than counted as false positives.
///
/// Every gathered pair yields a cell for every area range; the one skip is in
/// [`gather_pair`].
fn match_area<'a>(
    ctx: &EvalImgContext<'_>,
    pair: &'a PairCell<'a>,
    area_rng: [f64; 2],
    not_exhaustive_cat: bool,
) -> (GtView<'a>, MatchOutcome) {
    let is_kp = ctx.params.iou_type == IouType::Keypoints;
    let is_oid = ctx.eval_mode == EvalMode::OpenImages;

    let gt = partition_gt(pair, area_rng, is_kp, is_oid);
    let dt = area_filter_dt(pair, area_rng);

    let mut outcome = match_cell(ctx, &gt, &dt, pair.iou_matrix, is_oid);

    if not_exhaustive_cat {
        for t_idx in 0..ctx.params.iou_thrs.len() {
            for di in 0..dt.len() {
                if !outcome.dt_matched[(t_idx, di)] {
                    outcome.dt_ignore[(t_idx, di)] = true;
                }
            }
        }
    }
    (gt, outcome)
}

/// How many scores [`push_pair_lean`] will push for a pair, or `None` when
/// [`gather_pair`] would find nothing: the layout pass of [`Cells::build`].
pub(super) fn lean_scores_len(
    ctx: &EvalImgContext<'_>,
    img_id: u64,
    cat_id: u64,
    max_det: usize,
) -> Option<usize> {
    pair_ids(ctx, img_id, cat_id).map(|(_, dt_ids)| dt_ids.len().min(max_det))
}

/// One pair under every area range, written to `cells` as the record
/// `accumulate()` reads; nothing when the pair has neither ground truth nor
/// detections.
pub(super) fn push_pair_lean(
    ctx: &EvalImgContext<'_>,
    img_id: u64,
    cat_id: u64,
    max_det: usize,
    not_exhaustive_cat: bool,
    cells: &mut CellWriter<'_>,
) {
    let Some(pair) = gather_pair(ctx, img_id, cat_id, max_det) else {
        return;
    };
    cells.begin_pair(img_id, cat_id, &pair.dt_scores);
    for ar in &ctx.params.area_ranges {
        let (gt, outcome) = match_area(ctx, &pair, ar.range, not_exhaustive_cat);
        cells.push_area(
            gt.num_in_denominator as u32,
            outcome
                .dt_matched
                .iter_rows()
                .chain(outcome.dt_ignore.iter_rows()),
        );
    }
}

/// One pair under the area ranges at `area_idxs` (indices into
/// `ctx.params.area_ranges`), as full [`EvalImg`]s written into `out` — one
/// slot per index, left `None` when the pair has neither ground truth nor
/// detections.
pub(super) fn evaluate_pair_full(
    ctx: &EvalImgContext<'_>,
    img_id: u64,
    cat_id: u64,
    max_det: usize,
    not_exhaustive_cat: bool,
    area_idxs: &[usize],
    out: &mut [Option<EvalImg>],
) {
    let Some(pair) = gather_pair(ctx, img_id, cat_id, max_det) else {
        return;
    };
    for (slot, &a_idx) in out.iter_mut().zip(area_idxs) {
        let area_rng = ctx.params.area_ranges[a_idx].range;
        let (gt, outcome) = match_area(ctx, &pair, area_rng, not_exhaustive_cat);
        *slot = Some(EvalImg {
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
        });
    }
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

/// Construct an [`EvalImgContext`] from its parts.
///
/// The one place outside `evaluate()` allowed to write the `ious` field:
/// `tests/architecture.rs` restricts the literal `ious:` struct-literal syntax
/// to this module, `mod.rs`, and `evaluate.rs`, because that field is normally
/// the whole-dataset similarity cache and must stay driver-private. Streaming
/// evaluation builds its own tiny, per-image similarity map — never the
/// whole-dataset one — but still needs a context to hand to
/// [`gather_pair`]/[`evaluate_cell`]. Routing through here keeps the one
/// allowed write site as-is instead of widening that allowlist for a scratch
/// map the check was never guarding against.
pub(super) fn build_context<'a>(
    coco_gt: &'a COCO,
    coco_dt: &'a COCO,
    params: &'a Params,
    ious: &'a HashMap<(u64, u64), IouMatrix>,
    eval_mode: EvalMode,
    match_floors: &'a [f64],
) -> EvalImgContext<'a> {
    EvalImgContext {
        coco_gt,
        coco_dt,
        params,
        ious,
        eval_mode,
        match_floors,
    }
}
