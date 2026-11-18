use std::collections::HashSet;
use std::sync::{Mutex, PoisonError};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::COCOeval;
use super::EvalMode;
use super::matching::{Block, CellRef, arena_index};
use crate::metrics::counts::descending_score_key;
use crate::params::Params;

/// The key two area ranges are compared on: bit equality of both bounds.
/// `[0, 1e5**2]` and `[0, 1e10]` are the same range only if they are the same
/// `f64`s — no tolerance, because `params.area_ranges` is copied, not computed.
pub(super) fn area_key(range: [f64; 2]) -> [u64; 2] {
    [range[0].to_bits(), range[1].to_bits()]
}

/// The evaluated pairs bucketed by K slot, read per (K slot, area slot).
///
/// The bucketing depends only on `params` and the evaluated cells — never on which
/// images an accumulation happens to cover — so it is built once and reused across
/// every filtered re-accumulation. `compare()` runs two per bootstrap resample and
/// `slice_by()` one per slice; rebuilding the two id→index maps and walking all
/// ~400k cells inside each of those was the same answer recomputed hundreds of
/// times. The image filter is applied where it actually varies: in the per-item
/// loop of [`accumulate_impl`].
///
/// It carries the evaluator it was built from rather than leaving the caller to
/// pass a matching `params`/`eval_mode` alongside it, so a grouping bucketed
/// under one evaluator's category list cannot be accumulated under another's.
pub(super) struct EvalGrouping<'a> {
    ev: &'a COCOeval,
    /// Every in-scope pair as `(pair index, image slot)`, grouped by K slot
    /// and in pair order within each: slot `k_idx` holds
    /// `pairs[k_starts[k_idx]..k_starts[k_idx + 1]]`. The image slot is
    /// what [`image_mask`](Self::image_mask) indexes.
    pairs: Vec<(u32, u32)>,
    k_starts: Vec<usize>,
    /// For each slot of `params.area_ranges`, the evaluate-time area indices
    /// whose cells land in it, in evaluate-time order.
    area_cells: Vec<Vec<usize>>,
    /// Distinct `image_id` -> dense slot, over every pair in `pairs`.
    img_slots: FxHashMap<u64, u32>,
    /// For each threshold of the T axis, its row in the grid the cells were
    /// matched on — see the axis-resolution block in [`build`](Self::build).
    t_rows: Vec<Option<usize>>,
}

impl<'a> EvalGrouping<'a> {
    /// Bucket an evaluated `COCOeval`'s cells.
    pub(super) fn build(ev: &'a COCOeval) -> Self {
        let params = &ev.params;
        let k = if params.use_cats {
            params.cat_ids.len()
        } else {
            1
        };

        // Build category_id → k_idx mapping for grouping eval_imgs.
        let cat_id_to_k_idx: FxHashMap<u64, usize> = if params.use_cats {
            params
                .cat_ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect()
        } else {
            std::iter::once((crate::detection::POOLED_CAT, 0usize)).collect()
        };

        // Area-range keys as bit-exact f64 pairs. The range values are copied verbatim
        // from `params`, so bit equality is the right test. A linear scan over the
        // handful of ranges beats hashing a 16-byte key; `rposition` makes a range
        // listed twice fill its last slot only.
        let area_keys: Vec<[u64; 2]> = params
            .area_ranges
            .iter()
            .map(|ar| area_key(ar.range))
            .collect();
        // Evaluate-time axis resolution. The cells were matched under the
        // params `evaluate()` saw; the axes `accumulate()` fills are `params` as
        // they stand now. Each axis is resolved by value, once per grouping —
        // not per cell, and not per `accumulate_impl` call, which bootstrap
        // resamples make 2·n times.
        //
        // Area ranges: where each evaluate-time range now sits in
        // `params.area_ranges`. The same range listed twice fills its last
        // slot only, and a range no longer listed is skipped. Every pair's
        // `areas` is indexed by the evaluate-time position.
        let mut area_cells: Vec<Vec<usize>> = vec![Vec::new(); area_keys.len()];
        if let Some(inputs) = ev.eval_inputs.as_ref() {
            for (at_eval, ar) in inputs.params.area_ranges.iter().enumerate() {
                if let Some(a_idx) = area_keys.iter().rposition(|key| *key == area_key(ar.range)) {
                    area_cells[a_idx].push(at_eval);
                }
            }
        }
        // IoU thresholds: the opposite direction — for each threshold of the T
        // axis, its row in the evaluate-time grid. The T axis is
        // `params.iou_thrs` as it stands, since `summarize()`, `report()`, and
        // the diagnostics name a row by its position there, but each cell holds
        // one row per evaluate-time threshold. `COCOeval::evaluated_iou_row`
        // owns the lookup, shared with every per-threshold analysis: a
        // reordered grid reads each row under its own label, and a threshold
        // `evaluate()` never matched at keeps the `-1.0` "not computed" fill
        // instead of reading a row it does not have.
        let t_rows: Vec<Option<usize>> = params
            .iou_thrs
            .iter()
            .map(|&thr| ev.evaluated_iou_row(thr))
            .collect();

        // Group the pairs by K slot with a counting sort: one pass finds each
        // pair's slot and counts them, the second writes every pair once into
        // its slot's span of one exactly sized array. A pair's area cells are
        // not stored at all — `accumulate_impl` reads them off `area_cells` — so the
        // grouping is one entry per pair, not one per cell per area range.
        // Growing a bucket per (category, area range) by pushing cost more
        // than the precision curves themselves at 10x COCO val2017.
        //
        // Pairs stay in `evaluate()` order within a slot, and that order feeds
        // the stable score sort and decides ties, so it is part of the output,
        // not an implementation detail. The lookups are keyed on the ids a pair
        // carries, never on its position, so a `params` reconfigured between
        // `evaluate()` and `accumulate()` still resolves every pair correctly.
        const OUT_OF_SCOPE: u32 = u32::MAX;
        let cells = &ev.cells;
        // With no area range in scope every bucket is empty, so no pair is.
        let n_pairs = if area_cells.iter().any(|at| !at.is_empty()) {
            cells.len()
        } else {
            0
        };
        let k_of_pair = |pair: usize| {
            cat_id_to_k_idx
                .get(&cells.ids(pair).1)
                .map_or(OUT_OF_SCOPE, |&k_idx| arena_index(k_idx))
        };
        let k_of: Vec<u32> = if crate::primitives::sim::worth_parallel(n_pairs) {
            (0..n_pairs).into_par_iter().map(k_of_pair).collect()
        } else {
            (0..n_pairs).map(k_of_pair).collect()
        };
        let mut k_starts = vec![0usize; k + 1];
        for &k_idx in k_of.iter().filter(|&&k_idx| k_idx != OUT_OF_SCOPE) {
            k_starts[k_idx as usize + 1] += 1;
        }
        for k_idx in 0..k {
            k_starts[k_idx + 1] += k_starts[k_idx];
        }
        // Image slots in first-seen order. `evaluate()` writes an image's pairs
        // together, so the memo turns one hash lookup per pair into one per image.
        let mut cursor = k_starts.clone();
        let mut pairs = vec![(0u32, 0u32); k_starts[k]];
        let mut img_slots: FxHashMap<u64, u32> = FxHashMap::default();
        let mut last_img: Option<(u64, u32)> = None;
        for (pair, &k_idx) in k_of.iter().enumerate() {
            if k_idx == OUT_OF_SCOPE {
                continue;
            }
            let image_id = cells.ids(pair).0;
            let slot = match last_img {
                Some((id, slot)) if id == image_id => slot,
                _ => {
                    let next = img_slots.len() as u32;
                    let slot = *img_slots.entry(image_id).or_insert(next);
                    last_img = Some((image_id, slot));
                    slot
                }
            };
            let at = &mut cursor[k_idx as usize];
            pairs[*at] = (arena_index(pair), slot);
            *at += 1;
        }

        EvalGrouping {
            ev,
            pairs,
            k_starts,
            area_cells,
            img_slots,
            t_rows,
        }
    }

    /// The evaluator this grouping was built from.
    pub(super) fn eval(&self) -> &'a COCOeval {
        self.ev
    }

    /// A dense "is this image in scope" bitmap, indexed by the slot each pair carries.
    ///
    /// Built once per accumulation instead of probing a `HashSet<u64>` once per
    /// cell per work item. The membership test runs `K × A × M × cells` times —
    /// on COCO val that is ~1.2M hash lookups per `accumulate`, and `compare()`
    /// pays it again for every one of its bootstrap resamples.
    ///
    /// `None` means the whole dataset, which is an all-true mask rather than a
    /// branch at every cell.
    pub(super) fn image_mask(&self, img_filter: Option<&HashSet<u64>>) -> Vec<bool> {
        let n = self.img_slots.len();
        let Some(filter) = img_filter else {
            return vec![true; n];
        };
        let mut mask = vec![false; n];
        for (img_id, &slot) in &self.img_slots {
            if filter.contains(img_id) {
                mask[slot as usize] = true;
            }
        }
        mask
    }

    /// The pairs of K slot `k_idx` whose image slot `in_scope` admits, in
    /// pair order, into `out`.
    fn pairs_into(&self, k_idx: usize, in_scope: impl Fn(u32) -> bool, out: &mut Vec<u32>) {
        out.clear();
        out.extend(
            self.pairs[self.k_starts[k_idx]..self.k_starts[k_idx + 1]]
                .iter()
                .filter(|&&(_, slot)| in_scope(slot))
                .map(|&(pair, _)| pair),
        );
    }
}

/// Empties `v` and makes room for exactly `n` elements, so a reused buffer
/// grows to its largest item without doubling past it.
fn clear_with_capacity<T>(v: &mut Vec<T>, n: usize) {
    v.clear();
    v.reserve_exact(n);
}

/// Accumulate per-image eval results into precision/recall arrays.
///
/// When `img_filter` is `Some`, only eval_imgs whose `image_id` is in the set
/// are included. Pass `None` to include all images (standard behavior).
///
/// `eval_mode` decides only whether [`AccumulatedEval::ap_all_points`] is filled.
/// Open Images is the one mode that reads it, and computing it costs about as much
/// again as the gridded curve beside it — measured at +32% on `accumulate` for COCO
/// val2017 bbox — so every other mode gets the `-1.0` "not computed" sentinel
/// rather than paying for a value it discards. `compare::bootstrap_ci` calls this
/// once per resample, which multiplies the saving.
pub(super) fn accumulate_impl(
    grouping: &EvalGrouping<'_>,
    img_filter: Option<&HashSet<u64>>,
) -> AccumulatedEval {
    let params = &grouping.eval().params;
    let cells = &grouping.eval().cells;
    let want_all_points = grouping.eval().eval_mode == EvalMode::OpenImages;
    let t = params.iou_thrs.len();
    let t_rows = &grouping.t_rows;
    let r = params.rec_thrs.len();
    let k = if params.use_cats {
        params.cat_ids.len()
    } else {
        1
    };
    let a = params.area_ranges.len();
    let m = params.max_dets.len();

    // One dense in-scope bitmap for the whole accumulation, replacing a
    // `HashSet<u64>` probe per cell per work item.
    let img_mask = grouping.image_mask(img_filter);

    // Work items are K slots, each with its area slots inside, in parallel,
    // and the M axis inside those. What an area range cannot change is done
    // once per K slot: which pairs are in scope, and the score ranking of
    // their detections — an area range decides which detections are ignored,
    // never which are gathered or how they rank. What the max-detection
    // settings share — the cells in scope, and `num_gt`, which does not depend
    // on the cap — is resolved once per area slot instead of once per `m`.
    //
    // Each area slot's output cells are disjoint from every other's, so no
    // floating-point sum is reassociated and the slots can write the output
    // arrays in place: each stages its writes on its thread, then takes the
    // lock once to apply them.

    /// One `(k, a)` slab of every output, staged on its thread until the
    /// slot is done.
    #[derive(Default)]
    struct Staged {
        /// `[M x T x R]`.
        precision: Vec<f64>,
        /// `[M x T x R]`.
        scores: Vec<f64>,
        /// `[M x T]` of `(max recall, all-points AP)`. The AP rides along with
        /// the recall it was computed from, so the two cannot be written at
        /// different indices or one forgotten on an early return.
        recall: Vec<(f64, f64)>,
        /// `T` and `R`, as [`reset`](Staged::reset) last sized the slab.
        t: usize,
        r: usize,
    }

    impl Staged {
        /// The slab as every sweep starts it, for `m` caps, the thresholds
        /// `t_rows`, and `r` recall points. Precision and scores start at 0.0
        /// (distinct from -1.0, "no data") wherever ground truth exists, so a
        /// category with GT but no matches shows 0 AP rather than "missing"
        /// and dropping out of the mean; only recall thresholds reached by
        /// detections are overwritten, and unreachable ones stay at 0.0. A
        /// threshold `evaluate()` never matched at is -1.0, "not computed".
        /// Per area slot, not per `m`: `num_gt` does not depend on the cap.
        fn reset(&mut self, t_rows: &[Option<usize>], m: usize, r: usize) {
            let t = t_rows.len();
            (self.t, self.r) = (t, r);
            self.precision.clear();
            self.precision.resize(m * t * r, 0.0);
            self.scores.clear();
            self.scores.resize(m * t * r, 0.0);
            self.recall.clear();
            self.recall.resize(m * t, (-1.0, -1.0));
            for (t_idx, row) in t_rows.iter().enumerate() {
                if row.is_none() {
                    for m_idx in 0..m {
                        let base = (m_idx * t + t_idx) * r;
                        self.precision[base..base + r].fill(-1.0);
                        self.scores[base..base + r].fill(-1.0);
                    }
                }
            }
        }

        /// Threshold `t_idx` at cap `m_idx`: its final recall and all-points
        /// AP, and the curve's `points` over the ranks `inds` of `ranked` —
        /// none for a cap that keeps no detection.
        fn record(
            &mut self,
            (m_idx, t_idx): (usize, usize),
            recall_ap: (f64, f64),
            points: &[(usize, f64, usize)],
            ranked: &Ranked,
            inds: &[u32],
        ) {
            let (t, r) = (self.t, self.r);
            self.recall[m_idx * t + t_idx] = recall_ap;
            for &(r_idx, pr_val, rc_ptr) in points {
                let i = (m_idx * t + t_idx) * r + r_idx;
                self.precision[i] = pr_val;
                let (_, at) = ranked.keyed[inds[rc_ptr] as usize];
                self.scores[i] = ranked.scores[at as usize];
            }
        }

        /// The slab into its `(k_idx, a_idx)` cells of `out`. The M slots of
        /// an output cell are contiguous, so it lands as `m`-runs of each
        /// `(t, r)`.
        fn write_out(&self, out: &mut AccumulatedEval, k_idx: usize, a_idx: usize) {
            let shape = out.shape;
            let (t, r, m) = (shape.t, shape.r, shape.m);
            for t_idx in 0..t {
                for r_idx in 0..r {
                    let base = shape.precision_idx(t_idx, r_idx, k_idx, a_idx, 0);
                    for m_idx in 0..m {
                        let i = (m_idx * t + t_idx) * r + r_idx;
                        out.precision[base + m_idx] = self.precision[i];
                        out.scores[base + m_idx] = self.scores[i];
                    }
                }
                let base = shape.recall_idx(t_idx, k_idx, a_idx, 0);
                for m_idx in 0..m {
                    let (recall, ap) = self.recall[m_idx * t + t_idx];
                    out.recall[base + m_idx] = recall;
                    out.ap_all_points[base + m_idx] = ap;
                }
            }
        }
    }

    /// `rows` as one run of consecutive evaluate-time rows, when it is one —
    /// always, unless `iou_thrs` changed after `evaluate()`. Then a
    /// detection's flags at them are one read.
    fn consecutive(rows: &[Option<usize>]) -> Option<std::ops::Range<usize>> {
        rows.first()
            .copied()
            .flatten()
            .map(|first| first..first + rows.len())
            .filter(|run| rows.iter().copied().eq(run.clone().map(Some)))
    }

    /// Detection `d`'s matched and ignore flags at the thresholds `rows`, at
    /// most 64, one bit each in `rows` order; `run` is
    /// [`consecutive`]`(rows)`. A threshold with no row reads as neither.
    #[inline]
    fn flags_at(
        block: &Block<'_>,
        d: usize,
        rows: &[Option<usize>],
        run: Option<&std::ops::Range<usize>>,
    ) -> (u64, u64) {
        match run {
            Some(run) => block.flags(d, run.clone()),
            None => rows.iter().enumerate().fold((0, 0), |(m, i), (bit, &row)| {
                let Some(row) = row else { return (m, i) };
                (
                    m | block.matched(d, row..row + 1) << bit,
                    i | block.ignore(d, row..row + 1) << bit,
                )
            }),
        }
    }

    /// A gather's detections ranked by score, and each cap's share of them.
    /// Sized exactly, so no buffer doubles with its old and new copies both
    /// live; the person category at 10x val2017 gathers ~130k detections.
    #[derive(Default)]
    struct Ranked {
        scores: Vec<f64>,
        /// Position of each gathered detection inside its cell's
        /// score-descending list — the per-cell truncation index that
        /// `dtScores[0:maxDet]` applies.
        rank_in_cell: Vec<u32>,
        /// `(descending_score_key(score), position)` per gathered detection,
        /// sorted: the score order, as gathered positions.
        keyed: Vec<(u64, u32)>,
        /// The radix sort's working space.
        sort_scratch: Vec<(u64, u32)>,
        /// Per gathered position, its rank in `keyed`.
        rank_of: Vec<u32>,
        /// `0..n`: every rank, the order a cap that keeps everything reads.
        all_ranks: Vec<u32>,
        /// Per `m`, the ranks its cap keeps; empty where the cap keeps all.
        filtered: Vec<Vec<u32>>,
    }

    impl Ranked {
        /// The ranks cap `m_idx` keeps, in rank order: every rank when the
        /// cap cuts no detection — a stable filter that keeps everything is
        /// the identity — which is what lets a sweep reuse the curves of the
        /// cap before it. With COCO's `[1, 10, 100]` on a detector's
        /// few-per-cell output, 10 usually cuts nothing.
        fn kept(&self, max_det: usize, cap: usize, m_idx: usize) -> &[u32] {
            if max_det >= cap || self.filtered[m_idx].len() == self.all_ranks.len() {
                &self.all_ranks
            } else {
                &self.filtered[m_idx]
            }
        }
    }

    /// Gather the detections of `pairs`, in order, each cut at `cap`, and rank
    /// them, once for every `m`.
    ///
    /// pycocotools concatenates each cell's `dtScores[0:maxDet]` and mergesorts
    /// (stably) per `m`. A stable sort of the per-cell-truncated concatenation
    /// equals the stable sort of the concatenation truncated at the *largest*
    /// cap, filtered to `rank_in_cell < maxDet`: filtering preserves relative
    /// order, and ties break on concatenation position, which the filter also
    /// preserves. So gather at `Params::max_det()` once, sort once, and derive
    /// each `m` by a stable filter. Cells may hold more detections than the
    /// *current* cap when `params.max_dets` shrank between `evaluate()` and
    /// `accumulate()`, so the gather truncates to the current cap and never
    /// trusts the stored length.
    ///
    /// The identity needs a strict weak order on scores, which every non-NaN
    /// score has (`-0.0` and `0.0` compare equal and keep input order). A NaN
    /// score has no place in it and ranks wherever its bits put it. `load_res`
    /// rejects NaN scores; `COCO::from_dataset` does not, so NaN ranking is
    /// undefined.
    fn rank_gather(
        cells: &super::matching::Cells,
        cap: usize,
        max_dets: &[usize],
        pairs: impl Iterator<Item = usize> + Clone,
        out: &mut Ranked,
    ) {
        let n: usize = pairs.clone().map(|p| cells.scores(p).len().min(cap)).sum();
        // The gather is a subset of the scores arena, so its positions, and
        // each cell's ranks below them, fit `u32`.
        let n_u32 = arena_index(n);
        clear_with_capacity(&mut out.scores, n);
        clear_with_capacity(&mut out.rank_in_cell, n);
        for pair in pairs {
            let scores = cells.scores(pair);
            let nd = scores.len().min(cap);
            out.scores.extend_from_slice(&scores[..nd]);
            out.rank_in_cell.extend(0..nd as u32);
        }
        // Sort by score descending, ties in concatenation order: by
        // `(descending_score_key, position)`, the integers in place rather
        // than two scores through an index. Positions are distinct and ascend
        // in concatenation order, so they break every tie the way a stable
        // sort on the key alone would. The position rides along to map ranks
        // back to gathered detections.
        clear_with_capacity(&mut out.keyed, n);
        out.keyed.extend(
            out.scores
                .iter()
                .zip(0..n_u32)
                .map(|(&score, i)| (descending_score_key(score), i)),
        );
        crate::metrics::counts::sort_score_keyed(&mut out.keyed, &mut out.sort_scratch);
        out.rank_of.clear();
        out.rank_of.resize(n, 0);
        for (rank, &(_, i)) in (0..n_u32).zip(&out.keyed) {
            out.rank_of[i as usize] = rank;
        }
        // Stable filter of the shared order == per-`m` sort (see above), as
        // ranks; a cap at or above the gather's keeps them all.
        clear_with_capacity(&mut out.all_ranks, n);
        out.all_ranks.extend(0..n_u32);
        out.filtered.resize_with(max_dets.len(), Vec::new);
        let rank_in_cell = &out.rank_in_cell;
        for (kept, &max_det) in out.filtered.iter_mut().zip(max_dets) {
            kept.clear();
            if max_det < cap {
                kept.extend(
                    out.keyed
                        .iter()
                        .zip(0..n_u32)
                        .filter_map(|(&(_, i), rank)| {
                            ((rank_in_cell[i as usize] as usize) < max_det).then_some(rank)
                        }),
                );
            }
        }
    }

    /// One area slot's cells and the buffers swept over them.
    #[derive(Default)]
    struct Swept {
        evals: Vec<CellRef>,
        /// The slot's own ranking, for a slot that reads more than one
        /// evaluate-time block and so gathers each pair more than once.
        own: Ranked,
        /// Per rank, the detection's matched and ignore flags at up to 64
        /// thresholds, one bit each.
        matched_bits: Vec<u64>,
        ignore_bits: Vec<u64>,
        /// Every threshold's curve at once, from the bit words, and each
        /// lane's `num_gt`.
        lanes: crate::metrics::counts::LaneCurves,
        lane_gts: Vec<usize>,
        /// Open Images only: one threshold's matched and ignore flags, in
        /// score order, and the cumulative arrays its all-points AP needs.
        matched: Vec<bool>,
        ignore: Vec<bool>,
        tp: Vec<f64>,
        fp: Vec<f64>,
        pr_scratch: crate::metrics::counts::PrCurveScratch,
        curve: Vec<(usize, f64, usize)>,
        staged: Staged,
    }

    /// A K slot's working buffers when every area range is swept at once —
    /// see the fused branch below.
    #[derive(Default)]
    struct Fused {
        /// The area slots with ground truth, and each lane's `num_gt`.
        active: Vec<usize>,
        lane_gts: Vec<usize>,
        /// Per rank, every active area range's matched and ignore flags at
        /// every threshold: area slot `active[s]` in lanes `s * t..(s + 1) * t`.
        matched: Vec<u64>,
        ignore: Vec<u64>,
        lanes: crate::metrics::counts::LaneCurves,
        /// One slab per active area slot.
        staged: Vec<Staged>,
    }

    let shape = EvalShape { t, r, k, a, m };
    let total = t * r * k * a * m;
    let total_recall = t * k * a * m;
    // The two `[T x R x K x A x M]` arrays are megabytes each at a
    // detector's category count; filled in parallel past the shared
    // threshold, since `-1.0` is not the zero a fresh page holds.
    let filled = |len: usize| -> Vec<f64> {
        if crate::primitives::sim::worth_parallel(len) {
            (0..len).into_par_iter().map(|_| -1.0).collect()
        } else {
            vec![-1.0; len]
        }
    };
    let outputs = Mutex::new(AccumulatedEval {
        precision: filled(total),
        recall: vec![-1.0; total_recall],
        ap_all_points: vec![-1.0; total_recall],
        scores: filled(total),
        shape,
        cat_ids: if params.use_cats {
            params.cat_ids.clone()
        } else {
            Vec::new()
        },
        params: params.clone(),
    });
    let cap = params.max_det();

    // Each K slot's working buffers: its pairs, its shared ranking, its
    // per-area `num_gt`, and one `Swept` per area slot. Taken from a pool and
    // returned after, so a set is reused by every K slot run after it rather
    // than built per rayon job split — at a few megabytes a set and close to
    // one split per category on eight threads, fresh sets were page faults on
    // every category, and faults taken in parallel contend in the kernel. At
    // most one set per thread is ever live.
    type Scratch = (Vec<u32>, Ranked, Vec<usize>, Vec<Swept>, Fused);
    let pool: Mutex<Vec<Scratch>> = Mutex::new(Vec::new());
    let work = |(pairs, shared, num_gts, swepts, fused): &mut Scratch, k_idx: usize| {
        grouping.pairs_into(k_idx, |slot| img_mask[slot as usize], pairs);
        // Independent of `max_det` — the cap truncates detections, never
        // ground truth — so it is summed once for the whole M axis.
        num_gts.clear();
        num_gts.extend(grouping.area_cells.iter().map(|at_evals| {
            pairs
                .iter()
                .flat_map(|&pair| {
                    at_evals
                        .iter()
                        .map(move |&at| cells.num_gt(CellRef::new(pair as usize, at)) as usize)
                })
                .sum::<usize>()
        }));
        if num_gts.iter().all(|&n| n == 0) {
            return;
        }
        // The ranking every slot reading one evaluate-time block shares.
        let shares = grouping
            .area_cells
            .iter()
            .zip(num_gts.iter())
            .any(|(at_evals, &n)| at_evals.len() == 1 && n > 0);
        if shares {
            rank_gather(
                cells,
                cap,
                &params.max_dets,
                pairs.iter().map(|&p| p as usize),
                shared,
            );
        }
        let (pairs, shared, num_gts) = (&*pairs, &*shared, &*num_gts);

        // Every area range in one walk of the cells and one curve pass per
        // cap, `t` lanes a range, rather than a walk per range and a pass per
        // range and cap — on COCO's four ranges, a quarter of the walks and
        // passes. It needs every range with ground truth to read one
        // evaluate-time block, so all share the ranking; curves with no
        // cumulative arrays, which only Open Images wants; and lanes that
        // fit a word. Otherwise each range is swept on its own, below. The
        // two produce the same slabs.
        fused.active.clear();
        fused
            .active
            .extend((0..a).filter(|&a_idx| num_gts[a_idx] > 0));
        if !want_all_points
            && t > 0
            && t * fused.active.len() <= 64
            && fused
                .active
                .iter()
                .all(|&a_idx| grouping.area_cells[a_idx].len() == 1)
        {
            let Fused {
                active,
                lane_gts,
                matched,
                ignore,
                lanes,
                staged,
            } = fused;
            let n_gathered = shared.scores.len();
            for flags in [&mut *matched, &mut *ignore] {
                flags.clear();
                flags.resize(n_gathered, 0);
            }
            let run = consecutive(t_rows);
            let mut blocks = Vec::with_capacity(active.len());
            let mut at = 0;
            for &pair in pairs {
                let pair = pair as usize;
                blocks.clear();
                blocks.extend(
                    active.iter().map(|&a_idx| {
                        cells.block(CellRef::new(pair, grouping.area_cells[a_idx][0]))
                    }),
                );
                for d in 0..cells.scores(pair).len().min(cap) {
                    let (mut m_word, mut i_word) = (0u64, 0u64);
                    for (s, block) in blocks.iter().enumerate() {
                        let (m_flags, i_flags) = flags_at(block, d, t_rows, run.as_ref());
                        m_word |= m_flags << (s * t);
                        i_word |= i_flags << (s * t);
                    }
                    let rank = shared.rank_of[at] as usize;
                    matched[rank] = m_word;
                    ignore[rank] = i_word;
                    at += 1;
                }
            }

            staged.resize_with(active.len(), Staged::default);
            for slab in staged.iter_mut() {
                slab.reset(t_rows, m, r);
            }
            lane_gts.clear();
            for &a_idx in active.iter() {
                lane_gts.extend(std::iter::repeat_n(num_gts[a_idx], t));
            }
            // The ranks the last curves were computed over: a cap that keeps
            // the same ranks as the cap before it has the same curves.
            let mut computed: Option<&[u32]> = None;
            for m_idx in 0..m {
                let inds = shared.kept(params.max_dets[m_idx], cap, m_idx);
                if !inds.is_empty() && !computed.is_some_and(|c| std::ptr::eq(c, inds)) {
                    crate::metrics::counts::precision_recall_curves_of_lanes_into(
                        inds.iter().map(|&i| i as usize),
                        matched,
                        ignore,
                        lane_gts,
                        t,
                        &params.rec_thrs,
                        lanes,
                    );
                    computed = Some(inds);
                }
                for (s, slab) in staged.iter_mut().enumerate() {
                    for (t_idx, row) in t_rows.iter().enumerate() {
                        if row.is_none() {
                            continue;
                        }
                        // GT exists but no detections: recall 0.0 and no curve.
                        let (recall, points) = if inds.is_empty() {
                            (0.0, &[][..])
                        } else {
                            let lane = s * t + t_idx;
                            (lanes.final_recall[lane], &lanes.points[lane][..])
                        };
                        slab.record((m_idx, t_idx), (recall, -1.0), points, shared, inds);
                    }
                }
            }

            // A poisoned lock means another item panicked, which rayon propagates.
            let mut out = outputs.lock().unwrap_or_else(PoisonError::into_inner);
            for (slab, &a_idx) in staged.iter().zip(active.iter()) {
                slab.write_out(&mut out, k_idx, a_idx);
            }
            return;
        }

        // One `Swept` per area slot, kept with the set across K slots: a
        // fresh one per (K, A) item was an allocation round trip per item
        // for buffers that only grow.
        swepts.resize_with(a, Swept::default);
        swepts.par_iter_mut().enumerate().for_each(|(a_idx, s)| {
            let num_gt = num_gts[a_idx];
            if num_gt == 0 {
                return;
            }
            let Swept {
                evals,
                own,
                matched_bits,
                ignore_bits,
                lanes,
                lane_gts,
                matched,
                ignore,
                tp,
                fp,
                pr_scratch,
                curve,
                staged,
            } = s;
            let at_evals = &grouping.area_cells[a_idx];
            evals.clear();
            evals.extend(pairs.iter().flat_map(|&pair| {
                at_evals
                    .iter()
                    .map(move |&at| CellRef::new(pair as usize, at))
            }));
            let ranked: &Ranked = if at_evals.len() == 1 {
                shared
            } else {
                rank_gather(
                    cells,
                    cap,
                    &params.max_dets,
                    evals.iter().map(|c| c.pair()),
                    own,
                );
                own
            };
            let n_gathered = ranked.scores.len();
            let inds_of =
                |m_idx: usize| -> &[u32] { ranked.kept(params.max_dets[m_idx], cap, m_idx) };

            staged.reset(t_rows, m, r);

            // Each detection's matched and ignore flags at up to 64 thresholds
            // at a time, as one bit per threshold, at its rank: one walk over
            // the cells per 64 thresholds, rather than one per threshold.
            for (chunk_idx, chunk) in t_rows.chunks(64).enumerate() {
                for bits in [&mut *matched_bits, &mut *ignore_bits] {
                    bits.clear();
                    bits.resize(n_gathered, 0);
                }
                let run = consecutive(chunk);
                let mut at = 0;
                for &cell in evals.iter() {
                    let block = cells.block(cell);
                    for d in 0..cells.scores(cell.pair()).len().min(cap) {
                        let (m_flags, i_flags) = flags_at(&block, d, chunk, run.as_ref());
                        let rank = ranked.rank_of[at] as usize;
                        matched_bits[rank] = m_flags;
                        ignore_bits[rank] = i_flags;
                        at += 1;
                    }
                }

                if !want_all_points {
                    // Caps outside, every threshold of the chunk in one pass
                    // over the cap's ranks. Same curves as the per-threshold
                    // branch below, which only Open Images takes.
                    let mut computed: Option<&[u32]> = None;
                    for m_idx in 0..m {
                        let inds = inds_of(m_idx);
                        if !inds.is_empty() && !computed.is_some_and(|c| std::ptr::eq(c, inds)) {
                            lane_gts.clear();
                            lane_gts.resize(chunk.len(), num_gt);
                            crate::metrics::counts::precision_recall_curves_of_lanes_into(
                                inds.iter().map(|&i| i as usize),
                                matched_bits,
                                ignore_bits,
                                lane_gts,
                                chunk.len(),
                                &params.rec_thrs,
                                lanes,
                            );
                            computed = Some(inds);
                        }
                        for (bit, row) in chunk.iter().enumerate() {
                            if row.is_none() {
                                continue;
                            }
                            // GT exists but no detections: recall 0.0, as below.
                            let (recall, points) = if inds.is_empty() {
                                (0.0, &[][..])
                            } else {
                                (lanes.final_recall[bit], &lanes.points[bit][..])
                            };
                            let t_idx = chunk_idx * 64 + bit;
                            staged.record((m_idx, t_idx), (recall, -1.0), points, ranked, inds);
                        }
                    }
                    continue;
                }

                // Open Images: thresholds outside, caps inside, one
                // threshold's flags unpacked at a time.
                for (bit, &row) in chunk.iter().enumerate() {
                    let t_idx = chunk_idx * 64 + bit;
                    if row.is_none() {
                        continue;
                    }
                    matched.clear();
                    matched.extend(matched_bits.iter().map(|&w| w >> bit & 1 == 1));
                    ignore.clear();
                    ignore.extend(ignore_bits.iter().map(|&w| w >> bit & 1 == 1));

                    for m_idx in 0..m {
                        let inds = inds_of(m_idx);
                        if inds.is_empty() {
                            // GT exists but no detections — recall and AP are 0.0, not -1.0
                            // "missing". The metric *is* computable here and the answer is that
                            // nothing was found; reporting "not computed" would drop the
                            // category from the mean and quietly raise mAP.
                            staged.record((m_idx, t_idx), (0.0, 0.0), &[], ranked, inds);
                            continue;
                        }

                        // `metrics::counts` owns the TP/FP classification and the
                        // curve. The all-points AP is the exact area under the same
                        // envelope the grid samples and needs the cumulative `tp`/`fp`
                        // arrays, score-ordered — it cannot be recovered from the 101
                        // samples afterwards — so Open Images materializes them and
                        // reads the curve from them.
                        crate::metrics::counts::cumulative_tp_fp(
                            inds.iter().map(|&i| i as usize),
                            matched,
                            Some(ignore),
                            tp,
                            fp,
                        );
                        let final_recall = crate::metrics::counts::precision_recall_curve_into(
                            tp,
                            fp,
                            num_gt,
                            &params.rec_thrs,
                            pr_scratch,
                            curve,
                        );
                        let all_points_ap =
                            crate::metrics::counts::average_precision_all_points(tp, fp, num_gt);
                        staged.record(
                            (m_idx, t_idx),
                            (final_recall, all_points_ap),
                            curve,
                            ranked,
                            inds,
                        );
                    }
                }
            }

            // A poisoned lock means another item panicked, which rayon propagates.
            let mut out = outputs.lock().unwrap_or_else(PoisonError::into_inner);
            staged.write_out(&mut out, k_idx, a_idx);
        });
    };
    (0..k).into_par_iter().for_each(|k_idx| {
        let taken = pool.lock().unwrap_or_else(PoisonError::into_inner).pop();
        let mut scratch = taken.unwrap_or_default();
        work(&mut scratch, k_idx);
        pool.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(scratch);
    });

    outputs.into_inner().unwrap_or_else(PoisonError::into_inner)
}

impl COCOeval {
    /// Accumulate per-image results into precision/recall arrays.
    ///
    /// The arrays follow `params` as they stand now, resolved against what
    /// `evaluate()` matched: an IoU threshold, area range, or category that
    /// `evaluate()` did not see is left at `-1.0` ("not computed") rather than
    /// filled from another slot's matches. Run `evaluate()` again after changing
    /// `params` to compute it.
    ///
    /// Retires the last [`summarize`](Self::summarize): [`report`](Self::report)
    /// and the readers beside it wait for the next one.
    pub fn accumulate(&mut self) {
        // Scoped so the grouping's borrow of `self` ends before `self.eval` is written.
        //
        // The image axis is resolved by value like the others: `img_ids`
        // narrowed since `evaluate()` accumulates those images only
        // (pycocotools reads the new list's positions in the old one, which
        // is another image set again). Empty means every evaluated
        // image, as it does before `evaluate()`.
        let narrowed: Option<HashSet<u64>> = self.eval_inputs.as_ref().and_then(|inputs| {
            let now: HashSet<u64> = self.params.img_ids.iter().copied().collect();
            let same = now.len() == inputs.params.img_ids.len()
                && inputs.params.img_ids.iter().all(|id| now.contains(id));
            let changed = !(now.is_empty() || same);
            changed.then_some(now)
        });
        let eval = accumulate_impl(&EvalGrouping::build(self), narrowed.as_ref());
        self.eval = Some(eval);
        self.advance_to(super::Stage::Accumulated);
    }
}

/// Array dimensions of an accumulated evaluation result.
///
/// Precision and scores have shape `[T x R x K x A x M]`;
/// recall has shape `[T x K x A x M]`.
#[derive(Debug, Clone, Copy)]
pub struct EvalShape {
    /// Number of IoU thresholds (T).
    pub t: usize,
    /// Number of recall thresholds (R).
    pub r: usize,
    /// Number of categories (K).
    pub k: usize,
    /// Number of area ranges (A).
    pub a: usize,
    /// Number of max-detection limits (M).
    pub m: usize,
}

impl EvalShape {
    /// Flat index into `precision` (or `scores`) for 5-D coordinates.
    pub fn precision_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        ((((t * self.r + r) * self.k + k) * self.a + a) * self.m) + m
    }

    /// Flat index into `recall` for 4-D coordinates.
    pub fn recall_idx(&self, t: usize, k: usize, a: usize, m: usize) -> usize {
        (((t * self.k + k) * self.a + a) * self.m) + m
    }
}

/// Accumulated evaluation results across all images.
///
/// Precision and scores are stored as flat 5-D arrays with shape `[T x R x K x A x M]`.
/// Recall is a flat 4-D array with shape `[T x K x A x M]`. Values of -1.0 indicate
/// that no data was available for that combination — a category with no GT instances, for example.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AccumulatedEval {
    /// Interpolated precision at each (iou_thr, recall_thr, category, area_range, max_det).
    pub precision: Vec<f64>,
    /// Maximum recall at each (iou_thr, category, area_range, max_det).
    pub recall: Vec<f64>,
    /// VOC 2010 all-points AP — same shape and indexing as
    /// [`recall`](Self::recall), so `recall_idx` works for both. The exact area
    /// under the precision envelope that [`precision`](Self::precision) holds
    /// sampled at 101 points; see
    /// [`average_precision_all_points`](crate::metrics::counts::average_precision_all_points)
    /// for why Open Images wants the former. `-1.0` in every other mode, which
    /// does not compute it.
    pub ap_all_points: Vec<f64>,
    /// Detection score at each precision threshold, same shape as `precision`.
    pub scores: Vec<f64>,
    /// Array dimensions — use to interpret the flat precision/recall/scores vectors.
    pub shape: EvalShape,
    /// The K axis: the category id at each `k` position, in the order
    /// `precision` and `recall` index them. Empty when `use_cats` was false and
    /// every category was pooled into the single K slot.
    ///
    /// Recorded here because it is a fact about *this* accumulation, not about
    /// `params` as they stand later: anything that names a K slot — a per-class
    /// table, an LVIS frequency-group mean — reads it from here, so a
    /// `params.cat_ids` edited after `accumulate()`, or a `use_cats = false`
    /// run, cannot put one category's precision under another's name.
    pub cat_ids: Vec<u64>,
    /// The `params` this accumulation laid its axes out under — pycocotools'
    /// `eval['params']`. Every lookup from a threshold, area label, or
    /// detection cap to a slot reads it, for the reason
    /// [`cat_ids`](Self::cat_ids) is recorded: `params` edited after
    /// `accumulate()` describe other arrays than these.
    pub params: Params,
}

impl AccumulatedEval {
    /// Flat index into `precision` (or `scores`) for 5-D coordinates.
    pub fn precision_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        self.shape.precision_idx(t, r, k, a, m)
    }

    /// Flat index into `recall` for 4-D coordinates.
    pub fn recall_idx(&self, t: usize, k: usize, a: usize, m: usize) -> usize {
        self.shape.recall_idx(t, k, a, m)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::COCO;
    use crate::params::{AreaRange, IouType};
    use crate::types::{Annotation, Category, Dataset, Image};

    /// A sequential, unmemoized grouping walk: three hash lookups per cell. The
    /// oracle the production build is checked against — the production code
    /// shares none of its lookups, so a memo keyed on the wrong id or a run
    /// concatenated out of order shows up as a difference.
    ///
    /// Returns, per `k_idx * a + a_idx` bucket, the cells in order as
    /// `(pair index, evaluate-time area index, image_id)`; slot numbers are not
    /// compared (they are internal), only that they are consistent — see
    /// `assert_slots_consistent`.
    fn reference_grouping(ev: &COCOeval) -> Vec<Vec<(usize, usize, u64)>> {
        let params = &ev.params;
        let k = if params.use_cats {
            params.cat_ids.len()
        } else {
            1
        };
        let a = params.area_ranges.len();
        let cat_id_to_k_idx: HashMap<u64, usize> = if params.use_cats {
            params
                .cat_ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect()
        } else {
            std::iter::once((crate::detection::POOLED_CAT, 0usize)).collect()
        };
        let area_rng_to_idx: HashMap<[u64; 2], usize> = params
            .area_ranges
            .iter()
            .enumerate()
            .map(|(i, ar)| (area_key(ar.range), i))
            .collect();
        let evaluated_ranges = &ev
            .eval_inputs
            .as_ref()
            .expect("fixture has been evaluated")
            .params
            .area_ranges;
        let mut grouped = vec![Vec::new(); k * a];
        for pair in 0..ev.cells.len() {
            let Some(&k_idx) = cat_id_to_k_idx.get(&ev.cells.ids(pair).1) else {
                continue;
            };
            for (at_eval, ar) in evaluated_ranges.iter().enumerate() {
                let Some(&a_idx) = area_rng_to_idx.get(&area_key(ar.range)) else {
                    continue;
                };
                grouped[k_idx * a + a_idx].push((pair, at_eval, ev.cells.ids(pair).0));
            }
        }
        grouped
    }

    fn area(label: &str, lo: f64, hi: f64) -> AreaRange {
        AreaRange {
            label: label.into(),
            range: [lo, hi],
        }
    }

    /// Every visited pair has a ground truth or a detection, so the fixture
    /// fills one cell per pair.
    fn assert_fixture_filled(ev: &COCOeval) {
        assert_eq!(
            ev.cells.len(),
            ev.eval_inputs
                .as_ref()
                .expect("evaluated")
                .sparse_pairs
                .len(),
            "fixture must fill every pair"
        );
    }

    /// Six images × three categories. Most (image, category) pairs carry both a
    /// ground truth and detections; a few carry only one side, so some cells are
    /// `None` and some pairs exist only through detections. Boxes step in size so
    /// the area ranges below split them.
    fn make_eval(use_cats: bool) -> COCOeval {
        let images: Vec<Image> = (1..=6)
            .map(|id| Image {
                id,
                file_name: format!("{id}.jpg"),
                width: 640,
                height: 640,
                ..Default::default()
            })
            .collect();
        let categories: Vec<Category> = [1u64, 2, 3]
            .iter()
            .map(|&id| Category {
                id,
                name: format!("c{id}"),
                ..Default::default()
            })
            .collect();
        let bbox = |img: u64, cat: u64| {
            let side = 10.0 * (1 + img + 2 * cat) as f64;
            [5.0 * img as f64, 5.0 * cat as f64, side, side]
        };
        let mut gt_anns = Vec::new();
        let mut dt_anns = Vec::new();
        let mut next = 1u64;
        for img in 1..=6u64 {
            for cat in 1..=3u64 {
                let b = bbox(img, cat);
                // Image 5 has no ground truth for category 2; image 6 has no
                // detections for category 3.
                if !(img == 5 && cat == 2) {
                    gt_anns.push(Annotation {
                        id: next,
                        image_id: img,
                        category_id: cat,
                        bbox: Some(b),
                        area: Some(b[2] * b[3]),
                        ..Default::default()
                    });
                    next += 1;
                }
                if !(img == 6 && cat == 3) {
                    for (j, score) in [0.9, 0.6].iter().enumerate() {
                        let shifted = [b[0] + 2.0 * j as f64, b[1], b[2], b[3]];
                        dt_anns.push(Annotation {
                            id: next,
                            image_id: img,
                            category_id: cat,
                            bbox: Some(shifted),
                            area: Some(shifted[2] * shifted[3]),
                            score: Some(*score),
                            ..Default::default()
                        });
                        next += 1;
                    }
                }
            }
        }
        let dataset = |annotations| Dataset {
            info: None,
            images: images.clone(),
            annotations,
            categories: categories.clone(),
            licenses: vec![],
        };
        let gt = COCO::from_dataset(dataset(gt_anns));
        let dt = COCO::from_dataset(dataset(dt_anns));
        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.params.use_cats = use_cats;
        ev.params.area_ranges = vec![
            area("all", 0.0, 1e10),
            area("small", 0.0, 2500.0),
            area("medium", 2500.0, 6400.0),
            area("large", 6400.0, 1e10),
        ];
        ev.evaluate().expect("evaluable inputs");
        ev
    }

    /// One bucket's cells, as `accumulate_impl` composes them from
    /// `pairs_into` and `area_cells`, with the image slot each carries.
    fn bucket(g: &EvalGrouping<'_>, k_idx: usize, a_idx: usize) -> Vec<(CellRef, u32)> {
        let mut pairs = Vec::new();
        g.pairs_into(k_idx, |_| true, &mut pairs);
        let slots = g.pairs[g.k_starts[k_idx]..g.k_starts[k_idx + 1]]
            .iter()
            .map(|&(_, slot)| slot);
        pairs
            .iter()
            .zip(slots)
            .flat_map(|(&pair, slot)| {
                g.area_cells[a_idx]
                    .iter()
                    .map(move |&at| (CellRef::new(pair as usize, at), slot))
            })
            .collect()
    }

    /// Every bucket, indexed `k_idx * a + a_idx` as the reference walk is.
    fn buckets(g: &EvalGrouping<'_>) -> Vec<Vec<(CellRef, u32)>> {
        let k = g.k_starts.len() - 1;
        let a = g.area_cells.len();
        (0..k * a).map(|b| bucket(g, b / a, b % a)).collect()
    }

    fn assert_same_grouping(ev: &COCOeval, label: &str) {
        let expected = reference_grouping(ev);
        let got = EvalGrouping::build(ev);
        let got_buckets = buckets(&got);
        assert_eq!(got_buckets.len(), expected.len(), "{label}: bucket count");
        for (b, (got_bucket, want_bucket)) in got_buckets.iter().zip(&expected).enumerate() {
            let got_cells: Vec<(usize, usize, u64)> = got_bucket
                .iter()
                .map(|&(cell, _)| (cell.pair(), cell.area(), ev.cells.ids(cell.pair()).0))
                .collect();
            assert_eq!(
                got_cells, *want_bucket,
                "{label}: bucket {b} differs from the sequential walk"
            );
        }
        assert_slots_consistent(&got, label);
    }

    /// Every cell of one image carries one slot, distinct images carry distinct
    /// slots, and `image_mask` selects exactly the cells of the requested images.
    fn assert_slots_consistent(g: &EvalGrouping<'_>, label: &str) {
        let mut slot_of: HashMap<u64, u32> = HashMap::new();
        let mut img_of: HashMap<u32, u64> = HashMap::new();
        let all = buckets(g);
        for &(cell, slot) in all.iter().flatten() {
            let image_id = g.eval().cells.ids(cell.pair()).0;
            assert_eq!(
                *slot_of.entry(image_id).or_insert(slot),
                slot,
                "{label}: image {image_id} carries two slots"
            );
            assert_eq!(
                *img_of.entry(slot).or_insert(image_id),
                image_id,
                "{label}: slot {slot} carries two images"
            );
        }
        let keep: HashSet<u64> = [2u64, 5].into_iter().collect();
        let mask = g.image_mask(Some(&keep));
        for &(cell, slot) in all.iter().flatten() {
            let image_id = g.eval().cells.ids(cell.pair()).0;
            assert_eq!(
                mask[slot as usize],
                keep.contains(&image_id),
                "{label}: image_mask disagrees with the filter for image {image_id}"
            );
        }
    }

    #[test]
    fn grouping_matches_sequential_walk_under_reconfigured_params() {
        let mut ev = make_eval(true);
        // `evaluate()` visits only (image, category) pairs that hold a ground
        // truth or a detection, and since 1.1 keeps every one of them (as
        // pycocotools does): one cell per visited pair, none empty.
        assert_fixture_filled(&ev);
        assert_same_grouping(&ev, "as evaluated");

        // Category subset, reordered: cells of category 2 must be skipped and the
        // remaining two land in swapped slots.
        ev.params.cat_ids = vec![3, 1];
        assert_same_grouping(&ev, "cat_ids = [3, 1]");

        // Area subset, reordered, one range listed twice: a repeated range fills its
        // last slot only, and the dropped ranges skip.
        ev.params.area_ranges = vec![
            area("large", 6400.0, 1e10),
            area("all", 0.0, 1e10),
            area("all again", 0.0, 1e10),
        ];
        assert_same_grouping(&ev, "areas reordered with a duplicate");
        let g = EvalGrouping::build(&ev);
        assert_eq!(
            buckets(&g).len(),
            ev.params.cat_ids.len() * ev.params.area_ranges.len()
        );
        for k_idx in 0..ev.params.cat_ids.len() {
            assert!(
                bucket(&g, k_idx, 1).is_empty() && !bucket(&g, k_idx, 2).is_empty(),
                "a repeated area range must fill its last slot only"
            );
        }

        // Nothing left in scope: every bucket empty, no slots.
        ev.params.area_ranges = vec![area("none", 1.0, 2.0)];
        let g = EvalGrouping::build(&ev);
        assert!(buckets(&g).iter().all(Vec::is_empty));
        assert!(g.img_slots.is_empty());
    }

    #[test]
    fn grouping_matches_sequential_walk_without_categories() {
        let ev = make_eval(false);
        assert_fixture_filled(&ev);
        assert_same_grouping(&ev, "use_cats = false");
        // The category axis collapses to one slot.
        let g = EvalGrouping::build(&ev);
        assert_eq!(buckets(&g).len(), ev.params.area_ranges.len());
        assert!(
            bucket(&g, 0, 0).len() >= 6,
            "every image lands in the 'all' bucket"
        );
    }
}
