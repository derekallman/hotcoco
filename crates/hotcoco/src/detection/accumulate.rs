use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, PoisonError};

use rayon::prelude::*;

use super::COCOeval;
use super::EvalMode;
use super::matching::CellRef;

/// The key two area ranges are compared on: bit equality of both bounds.
/// `[0, 1e5**2]` and `[0, 1e10]` are the same range only if they are the same
/// `f64`s — no tolerance, because `params.area_ranges` is copied, not computed.
fn area_key(range: [f64; 2]) -> [u64; 2] {
    [range[0].to_bits(), range[1].to_bits()]
}

/// `eval_imgs` bucketed by (category slot, area-range slot).
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
    /// Indexed `k_idx * a + a_idx`, each bucket in `eval_imgs` order. Each cell
    /// carries the dense image slot its `image_id` resolves to — see
    /// [`image_mask`](Self::image_mask).
    grouped: Vec<Vec<(CellRef, u32)>>,
    /// Number of area ranges — the stride of `grouped`.
    a: usize,
    /// Distinct `image_id` -> dense slot, over every cell in `grouped`.
    img_slots: HashMap<u64, u32>,
}

impl<'a> EvalGrouping<'a> {
    /// Bucket an evaluated `COCOeval`'s cells.
    pub(super) fn build(ev: &'a COCOeval) -> Self {
        // Tests pass their own run length to put run boundaries where they
        // want them.
        Self::build_chunked(ev, super::run_len(ev.cells.len()))
    }

    /// [`build`](Self::build) with the cells walked in runs of `chunk_len`.
    ///
    /// The result does not depend on `chunk_len` — bucket contents and order are
    /// the same for any value; only the internal image-slot numbering differs.
    fn build_chunked(ev: &'a COCOeval, chunk_len: usize) -> Self {
        let params = &ev.params;
        let k = if params.use_cats {
            params.cat_ids.len()
        } else {
            1
        };
        let a = params.area_ranges.len();

        // Build category_id → k_idx mapping for grouping eval_imgs.
        let cat_id_to_k_idx: HashMap<u64, usize> = if params.use_cats {
            params
                .cat_ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect()
        } else {
            std::iter::once((u64::MAX, 0usize)).collect()
        };

        // Area-range keys as bit-exact f64 pairs. The range values are copied verbatim
        // from `params`, so bit equality is the right test. A linear scan over the
        // handful of ranges beats hashing a 16-byte key per cell; `rposition` makes
        // a range listed twice fill its last slot only.
        let area_keys: Vec<[u64; 2]> = params
            .area_ranges
            .iter()
            .map(|ar| area_key(ar.range))
            .collect();
        // Where each evaluate-time area range now sits in `params.area_ranges`,
        // by value: the same range listed twice fills its last slot only, and a
        // range no longer listed is skipped. Every pair's `areas` is indexed by
        // the evaluate-time position, so this resolves once, not per cell.
        let remap: Vec<Option<usize>> = ev.eval_inputs.as_ref().map_or_else(Vec::new, |inputs| {
            inputs
                .params
                .area_ranges
                .iter()
                .map(|ar| area_keys.iter().rposition(|key| *key == area_key(ar.range)))
                .collect()
        });

        // Group the pairs by (k_idx, a_idx) — one pass over the cells, in parallel.
        //
        // The pairs are cut into a few contiguous runs per thread; each run keeps
        // its own buckets, in pair order, and the runs are concatenated in order
        // below — so every bucket ends up in `evaluate()` order exactly as a
        // sequential walk would leave it. That order feeds the stable score sort and
        // decides ties, so it is part of the output, not an implementation detail.
        // Explicit chunks rather than rayon's adaptive splitting: every run allocates
        // `k * a` buckets, so runs must stay few.
        //
        // `evaluate()` writes `area_ranges.len()` consecutive cells per (image,
        // category) pair, pairs sorted by image then category, so consecutive cells
        // almost always share their category and their image. The two memos turn one
        // hash lookup per cell into one per pair (category) and one per image. They
        // are keyed on the id a cell carries, never on its position, so a `params`
        // reconfigured between `evaluate()` and `accumulate()` still resolves every
        // cell correctly — the layout only makes the memo hit.
        //
        // Image slots are assigned per run first (an index into `imgs`), then
        // remapped to global slots once the runs are back together. Slot numbers are
        // internal — `image_mask` is the only reader — so which run saw an image
        // first does not matter, only that every cell of one image shares a slot.
        struct Run {
            /// `k_idx * a + a_idx` → cells in this run, in pair order; the `u32` is an
            /// index into `imgs`.
            buckets: Vec<Vec<(CellRef, u32)>>,
            /// Image ids in first-seen order. A repeat is only possible when an image's
            /// cells are not contiguous, and the remap below tolerates it.
            imgs: Vec<u64>,
        }
        let cells = &ev.cells;
        let runs: Vec<Run> = (0..cells.len().div_ceil(chunk_len))
            .into_par_iter()
            .map(|run| run * chunk_len)
            .map(|start| {
                let mut run = Run {
                    buckets: vec![Vec::new(); k * a],
                    imgs: Vec::new(),
                };
                let mut last_cat: Option<(u64, Option<usize>)> = None;
                let mut last_img: Option<(u64, u32)> = None;
                for pair in start..(start + chunk_len).min(cells.len()) {
                    let (image_id, category_id) = cells.ids(pair);
                    let k_idx = match last_cat {
                        Some((id, k_idx)) if id == category_id => k_idx,
                        _ => {
                            let k_idx = cat_id_to_k_idx.get(&category_id).copied();
                            last_cat = Some((category_id, k_idx));
                            k_idx
                        }
                    };
                    let Some(k_idx) = k_idx else {
                        continue;
                    };
                    for (at_eval, &a_idx) in remap.iter().enumerate() {
                        let Some(a_idx) = a_idx else {
                            continue;
                        };
                        // An image gets a slot only once a cell of it lands in a
                        // bucket, so a run with nothing in scope has no slots.
                        let local = match last_img {
                            Some((id, local)) if id == image_id => local,
                            _ => {
                                let local = run.imgs.len() as u32;
                                run.imgs.push(image_id);
                                last_img = Some((image_id, local));
                                local
                            }
                        };
                        run.buckets[k_idx * a + a_idx].push((CellRef::new(pair, at_eval), local));
                    }
                }
                run
            })
            .collect();

        // Global slots, and one run-local → global table per run.
        let mut img_slots: HashMap<u64, u32> = HashMap::new();
        let remaps: Vec<Vec<u32>> = runs
            .iter()
            .map(|run| {
                run.imgs
                    .iter()
                    .map(|&img_id| {
                        let next = img_slots.len() as u32;
                        *img_slots.entry(img_id).or_insert(next)
                    })
                    .collect()
            })
            .collect();

        // Concatenate the runs bucket by bucket, in run order.
        let grouped: Vec<Vec<(CellRef, u32)>> = (0..k * a)
            .into_par_iter()
            .map(|b| {
                let total: usize = runs.iter().map(|run| run.buckets[b].len()).sum();
                let mut bucket = Vec::with_capacity(total);
                for (run, remap) in runs.iter().zip(&remaps) {
                    bucket.extend(
                        run.buckets[b]
                            .iter()
                            .map(|&(eval, local)| (eval, remap[local as usize])),
                    );
                }
                bucket
            })
            .collect();

        EvalGrouping {
            ev,
            grouped,
            a,
            img_slots,
        }
    }

    /// The evaluator this grouping was built from.
    pub(super) fn eval(&self) -> &'a COCOeval {
        self.ev
    }

    /// A dense "is this image in scope" bitmap, indexed by the slot each cell carries.
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

    fn cell(&self, k_idx: usize, a_idx: usize) -> &[(CellRef, u32)] {
        &self.grouped[k_idx * self.a + a_idx]
    }
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

    // Work items are `(k_idx, a_idx)` with the M axis **inside**, not the full
    // `(k, a, m)` product. Everything the max-detection settings share — which
    // cells are in scope, and `num_gt`, which does not depend on the cap — is
    // resolved once per item instead of once per `m`. On COCO that is three
    // passes over the cell list collapsed into one, and the `num_gt == 0`
    // short-circuit now skips all three M slots together.
    //
    // Each item's output cells are disjoint from every other item's, so no
    // floating-point sum is reassociated and the items can write the output
    // arrays in place: each stages its writes on its thread, then takes the
    // lock once to apply them.

    /// One item's `(k, a)` slab of every output, staged on its thread until
    /// the item is done.
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
    }

    let shape = EvalShape { t, r, k, a, m };
    let total = t * r * k * a * m;
    let total_recall = t * k * a * m;
    let outputs = Mutex::new(AccumulatedEval {
        precision: vec![-1.0; total],
        recall: vec![-1.0; total_recall],
        ap_all_points: vec![-1.0; total_recall],
        scores: vec![-1.0; total],
        shape,
    });

    // Items are indexed as `grouped` is: `k_idx * a + a_idx`.
    (0..k * a)
        .into_par_iter()
        .for_each_init(Staged::default, |staged, item| {
            let (k_idx, a_idx) = (item / a, item % a);
            // Materialized once for every `m`, rather than filtered per `m`.
            let evals: Vec<CellRef> = grouping
                .cell(k_idx, a_idx)
                .iter()
                .filter(|&&(_, slot)| img_mask[slot as usize])
                .map(|&(e, _)| e)
                .collect();

            // Independent of `max_det` — the cap truncates detections, never
            // ground truth — so it is summed once for the whole M axis.
            let num_gt: usize = evals.iter().map(|&e| cells.num_gt(e) as usize).sum();
            if num_gt == 0 {
                return;
            }
            // Precision and scores start at 0.0 (distinct from -1.0, "no data")
            // wherever ground truth exists, so a category with GT but no matches
            // shows 0 AP rather than "missing" and dropping out of the mean. Only
            // recall thresholds reached by detections are overwritten below;
            // unreachable ones stay at 0.0. Per item, not per `m`: `num_gt` does
            // not depend on the max-det cap.
            staged.precision.clear();
            staged.precision.resize(m * t * r, 0.0);
            staged.scores.clear();
            staged.scores.resize(m * t * r, 0.0);
            staged.recall.clear();
            staged.recall.resize(m * t, (-1.0, -1.0));

            // One gather and one sort per work item, shared by every `m`.
            //
            // pycocotools concatenates each cell's `dtScores[0:maxDet]` and
            // mergesorts (stably) per `m`. A stable sort of the per-cell-truncated
            // concatenation equals the stable sort of the concatenation truncated
            // at the *largest* cap, filtered to `rank_in_cell < maxDet`: filtering
            // preserves relative order, and ties break on concatenation position,
            // which the filter also preserves. So gather at `Params::max_det()`
            // once, sort once, and derive each `m` by a stable filter. Cells may hold more detections than the
            // *current* cap when `params.max_dets` shrank between `evaluate()` and
            // `accumulate()`, so the gather truncates to the current cap and never
            // trusts the stored length.
            //
            // The identity needs a strict weak order on scores, which `partial_cmp`
            // gives for every finite value (and for `-0.0` versus `0.0`, which
            // compare equal and keep input order). A NaN score breaks it: the
            // comparator below treats NaN as equal to everything, so the sorted
            // order — and therefore which detections a filtered slot keeps —
            // depends on the input sequence. `load_res` rejects NaN scores;
            // `COCO::from_dataset` does not, so NaN ranking is undefined.
            let cap = params.max_det();
            let mut all_dt_scores: Vec<f64> = Vec::new();
            // Position of each gathered detection inside its cell's score-descending
            // list — the per-cell truncation index that `dtScores[0:maxDet]` applies.
            let mut rank_in_cell: Vec<usize> = Vec::new();
            let mut all_dt_matched: Vec<Vec<bool>> = vec![Vec::new(); t];
            let mut all_dt_ignore: Vec<Vec<bool>> = vec![Vec::new(); t];
            for &cell in &evals {
                let scores = cells.scores(cell.pair());
                let nd = scores.len().min(cap);
                all_dt_scores.extend_from_slice(&scores[..nd]);
                rank_in_cell.extend(0..nd);
                let block = cells.block(cell);
                for t_idx in 0..t {
                    all_dt_matched[t_idx].extend(block.matched(t_idx).take(nd));
                    all_dt_ignore[t_idx].extend(block.ignore(t_idx).take(nd));
                }
            }

            // Sort by score descending — stable, ties keep concatenation order.
            let mut order: Vec<usize> = (0..all_dt_scores.len()).collect();
            order.sort_by(|&a, &b| {
                all_dt_scores[b]
                    .partial_cmp(&all_dt_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Buffers reused across the M axis and the threshold sweep.
            let mut filtered: Vec<usize> = Vec::new();
            let (mut tp, mut fp) = (Vec::new(), Vec::new());
            let mut pr_scratch = crate::metrics::counts::PrCurveScratch::default();
            let mut curve: Vec<(usize, f64, usize)> = Vec::new();

            for m_idx in 0..m {
                let max_det = params.max_dets[m_idx];

                // Stable filter of the shared order == per-`m` sort (see above).
                let inds: &[usize] = if max_det >= cap {
                    &order
                } else {
                    filtered.clear();
                    filtered.extend(order.iter().copied().filter(|&i| rank_in_cell[i] < max_det));
                    &filtered
                };

                let nd = inds.len();

                if nd == 0 {
                    // GT exists but no detections — recall and AP are 0.0, not -1.0
                    // "missing". The metric *is* computable here and the answer is that
                    // nothing was found; reporting "not computed" would drop the
                    // category from the mean and quietly raise mAP.
                    let ap = if want_all_points { 0.0 } else { -1.0 };
                    for t_idx in 0..t {
                        staged.recall[m_idx * t + t_idx] = (0.0, ap);
                    }
                    continue;
                }

                for t_idx in 0..t {
                    // `metrics::counts` owns the TP/FP classification and the
                    // curve. The all-points AP is the exact area under the same
                    // envelope the grid samples and needs the cumulative `tp`/`fp`
                    // arrays, score-ordered — it cannot be recovered from the 101
                    // samples afterwards — so Open Images materializes them and
                    // reads the curve from them. Every other mode discards that
                    // AP and takes the fused kernel, which produces the same
                    // curve without the two arrays.
                    let (final_recall, all_points_ap) = if want_all_points {
                        crate::metrics::counts::cumulative_tp_fp(
                            inds.iter().copied(),
                            &all_dt_matched[t_idx],
                            Some(&all_dt_ignore[t_idx]),
                            &mut tp,
                            &mut fp,
                        );
                        let final_recall = crate::metrics::counts::precision_recall_curve_into(
                            &tp,
                            &fp,
                            num_gt,
                            &params.rec_thrs,
                            &mut pr_scratch,
                            &mut curve,
                        );
                        let ap =
                            crate::metrics::counts::average_precision_all_points(&tp, &fp, num_gt);
                        (final_recall, ap)
                    } else {
                        let final_recall =
                            crate::metrics::counts::precision_recall_curve_of_order_into(
                                inds.iter().copied(),
                                &all_dt_matched[t_idx],
                                Some(&all_dt_ignore[t_idx]),
                                num_gt,
                                &params.rec_thrs,
                                &mut pr_scratch,
                                &mut curve,
                            );
                        (final_recall, -1.0)
                    };
                    staged.recall[m_idx * t + t_idx] = (final_recall, all_points_ap);
                    for &(r_idx, pr_val, rc_ptr) in &curve {
                        let i = (m_idx * t + t_idx) * r + r_idx;
                        staged.precision[i] = pr_val;
                        staged.scores[i] = all_dt_scores[inds[rc_ptr]];
                    }
                }
            }

            // The M slots of an output cell are contiguous, so the slab lands
            // as `m`-runs of each `(t, r)`.
            // A poisoned lock means another item panicked, which rayon propagates.
            let mut out = outputs.lock().unwrap_or_else(PoisonError::into_inner);
            for t_idx in 0..t {
                for r_idx in 0..r {
                    let base = shape.precision_idx(t_idx, r_idx, k_idx, a_idx, 0);
                    for m_idx in 0..m {
                        let i = (m_idx * t + t_idx) * r + r_idx;
                        out.precision[base + m_idx] = staged.precision[i];
                        out.scores[base + m_idx] = staged.scores[i];
                    }
                }
                let base = shape.recall_idx(t_idx, k_idx, a_idx, 0);
                for m_idx in 0..m {
                    let (recall, ap) = staged.recall[m_idx * t + t_idx];
                    out.recall[base + m_idx] = recall;
                    out.ap_all_points[base + m_idx] = ap;
                }
            }
        });

    outputs.into_inner().unwrap_or_else(PoisonError::into_inner)
}

impl COCOeval {
    /// Accumulate per-image results into precision/recall arrays.
    pub fn accumulate(&mut self) {
        // Scoped so the grouping's borrow of `self` ends before `self.eval` is written.
        let eval = accumulate_impl(&EvalGrouping::build(self), None);
        self.eval = Some(eval);
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
            std::iter::once((u64::MAX, 0usize)).collect()
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
        ev.evaluate();
        ev
    }

    fn assert_same_grouping(ev: &COCOeval, chunk_len: usize, label: &str) {
        let expected = reference_grouping(ev);
        let got = EvalGrouping::build_chunked(ev, chunk_len);
        assert_eq!(got.grouped.len(), expected.len(), "{label}: bucket count");
        for (b, (got_bucket, want_bucket)) in got.grouped.iter().zip(&expected).enumerate() {
            let got_cells: Vec<(usize, usize, u64)> = got_bucket
                .iter()
                .map(|&(cell, _)| (cell.pair(), cell.area(), ev.cells.ids(cell.pair()).0))
                .collect();
            assert_eq!(
                got_cells, *want_bucket,
                "{label}, chunk_len {chunk_len}: bucket {b} differs from the sequential walk"
            );
        }
        assert_slots_consistent(&got, label);
    }

    /// Every cell of one image carries one slot, distinct images carry distinct
    /// slots, and `image_mask` selects exactly the cells of the requested images.
    fn assert_slots_consistent(g: &EvalGrouping<'_>, label: &str) {
        let mut slot_of: HashMap<u64, u32> = HashMap::new();
        let mut img_of: HashMap<u32, u64> = HashMap::new();
        for &(cell, slot) in g.grouped.iter().flatten() {
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
        for &(cell, slot) in g.grouped.iter().flatten() {
            let image_id = g.eval().cells.ids(cell.pair()).0;
            assert_eq!(
                mask[slot as usize],
                keep.contains(&image_id),
                "{label}: image_mask disagrees with the filter for image {image_id}"
            );
        }
    }

    /// Chunk lengths that put run boundaries in the middle of an image, in the
    /// middle of a pair's area-range block, at one cell, and nowhere.
    const CHUNKS: [usize; 5] = [1, 3, 7, 16, usize::MAX];

    #[test]
    fn grouping_matches_sequential_walk_under_reconfigured_params() {
        let mut ev = make_eval(true);
        // `evaluate()` visits only (image, category) pairs that hold a ground
        // truth or a detection, and since 1.1 keeps every one of them (as
        // pycocotools does): one cell per visited pair, none empty.
        assert_fixture_filled(&ev);
        for chunk_len in CHUNKS {
            assert_same_grouping(&ev, chunk_len, "as evaluated");
        }

        // Category subset, reordered: cells of category 2 must be skipped and the
        // remaining two land in swapped slots.
        ev.params.cat_ids = vec![3, 1];
        for chunk_len in CHUNKS {
            assert_same_grouping(&ev, chunk_len, "cat_ids = [3, 1]");
        }

        // Area subset, reordered, one range listed twice: a repeated range fills its
        // last slot only, and the dropped ranges skip.
        ev.params.area_ranges = vec![
            area("large", 6400.0, 1e10),
            area("all", 0.0, 1e10),
            area("all again", 0.0, 1e10),
        ];
        for chunk_len in CHUNKS {
            assert_same_grouping(&ev, chunk_len, "areas reordered with a duplicate");
        }
        let g = EvalGrouping::build(&ev);
        assert_eq!(
            g.grouped.len(),
            ev.params.cat_ids.len() * ev.params.area_ranges.len()
        );
        for k_idx in 0..ev.params.cat_ids.len() {
            assert!(
                g.cell(k_idx, 1).is_empty() && !g.cell(k_idx, 2).is_empty(),
                "a repeated area range must fill its last slot only"
            );
        }

        // Nothing left in scope: every bucket empty, no slots.
        ev.params.area_ranges = vec![area("none", 1.0, 2.0)];
        let g = EvalGrouping::build(&ev);
        assert!(g.grouped.iter().all(Vec::is_empty));
        assert!(g.img_slots.is_empty());
    }

    #[test]
    fn grouping_matches_sequential_walk_without_categories() {
        let ev = make_eval(false);
        assert_fixture_filled(&ev);
        for chunk_len in CHUNKS {
            assert_same_grouping(&ev, chunk_len, "use_cats = false");
        }
        // The category axis collapses to one slot.
        let g = EvalGrouping::build(&ev);
        assert_eq!(g.grouped.len(), ev.params.area_ranges.len());
        assert!(
            g.cell(0, 0).len() >= 6,
            "every image lands in the 'all' bucket"
        );
    }
}
