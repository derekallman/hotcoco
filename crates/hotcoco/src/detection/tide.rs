use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};

use rayon::prelude::*;
use serde::Serialize;

use crate::metrics::counts::{ApScratch, average_precision_ranked_into};

use super::COCOeval;
use super::matching::EvalImg;

/// TIDE false-positive error types, named as in tidecv.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ErrType {
    Cls,
    Loc,
    Both,
    Dupe,
    Bkg,
}

impl ErrType {
    /// The key this error type is reported under. Single source of the spelling —
    /// the aggregation must not re-enumerate the variants.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            ErrType::Cls => "Cls",
            ErrType::Loc => "Loc",
            ErrType::Both => "Both",
            ErrType::Dupe => "Dupe",
            ErrType::Bkg => "Bkg",
        }
    }
}

/// Every false-positive error type, in report order.
///
/// The one enumeration of the five. Counts, per-category ΔAP vectors, the
/// per-type oracle runs and the output map are all driven from this array, and
/// `err as usize` is the index into anything sized by it — which holds because
/// the array is in declaration order. `"Miss"` is not here: it is counted from
/// ground truths, not from detections, and is added to the output separately.
const FP_TYPES: [ErrType; 5] = [
    ErrType::Cls,
    ErrType::Loc,
    ErrType::Both,
    ErrType::Dupe,
    ErrType::Bkg,
];

/// What [`classify_fp`] needs to know about one false-positive detection.
///
/// Gathering this is the caller's job (it needs the per-image IoU views and the
/// cross-category scan); deciding what it *means* is the parity contract below.
pub(super) struct FpEvidence {
    /// Highest IoU with any same-class GT (0.0 if there are none).
    pub(super) max_same_iou: f64,
    /// Highest IoU with any *different*-class GT (0.0 if there are none).
    pub(super) max_cross_iou: f64,
    /// Some same-class GT with IoU >= `pos_thr` was already matched by a
    /// higher-scoring detection.
    pub(super) best_same_gt_matched: bool,
}

/// Classify a false positive into a TIDE error type.
///
/// **This is the tidecv parity contract**, extracted so it is readable and
/// testable on its own: the priority order is load-bearing, and reordering two
/// arms silently changes every published TIDE number.
///
/// Priority matches tidecv (`BoxError > ClassError > DuplicateError >
/// BackgroundError > OtherError`):
///
/// | Order | Type | Condition |
/// |---|---|---|
/// | 1 | `Loc`  | same-class max IoU in `[bg_thr, pos_thr]` — the upper bound is inclusive and is what excludes `Dupe`, whose same-class IoU exceeds `pos_thr` |
/// | 2 | `Cls`  | cross-class max IoU >= `pos_thr` (and `Loc` did not fire) |
/// | 3 | `Dupe` | a same-class GT at IoU >= `pos_thr` is already matched by a higher-scoring TP |
/// | 4 | `Bkg`  | max IoU with any GT <= `bg_thr` (same-class is already below `bg_thr` here, so only cross-class needs checking) |
/// | 5 | `Both` | fallthrough — cross-class IoU in `(bg_thr, pos_thr)` |
pub(super) fn classify_fp(ev: &FpEvidence, pos_thr: f64, bg_thr: f64) -> ErrType {
    if ev.max_same_iou >= bg_thr && ev.max_same_iou <= pos_thr {
        ErrType::Loc
    } else if ev.max_cross_iou >= pos_thr {
        ErrType::Cls
    } else if ev.best_same_gt_matched {
        ErrType::Dupe
    } else if ev.max_cross_iou <= bg_thr {
        ErrType::Bkg
    } else {
        ErrType::Both
    }
}

/// Highest-IoU *different*-category ground truth for every detection, by image.
///
/// `img_id → (dt_ann_id → (max_cross_iou, argmax_cross_gt_ann_id))`. The argmax is
/// `None` when the image holds no cross-category ground truth.
type CrossIouMap = HashMap<u64, HashMap<u64, (f64, Option<u64>)>>;

/// One detection's same-class evidence, scanned out of its cell's IoU matrix.
///
/// Separate from [`FpEvidence`]: that one is the parity contract's *input* and is
/// deliberately minimal, while `argmax_gt_ann_id` exists only to feed the
/// covered-GT set that narrows Miss.
#[derive(Default)]
struct SameClassScan {
    /// Highest IoU with any same-class GT in the cell.
    max_iou: f64,
    /// The GT achieving `max_iou`, if the cell had any same-class GT at all.
    argmax_gt_ann_id: Option<u64>,
    /// Some same-class GT at IoU >= `pos_thr` was already claimed by a
    /// higher-scoring detection.
    best_gt_matched: bool,
}

/// The ground truth a `Loc` or `Cls` error would become a TP for, if fixed.
///
/// Recorded only when that GT is *usable* — unmatched at `pos_thr` and in the
/// recall denominator. An error aimed at a GT some detection already claimed has
/// no target: tidecv's `BestGTMatch` suppresses it instead of promoting it.
#[derive(Clone, Copy)]
struct FixTarget {
    gt_id: u64,
    /// The GT's category. Differs from the detection's for `Cls`, and a fixed
    /// `Cls` error is a TP of *this* category, not its own.
    gt_cat: u64,
    /// Tie-break for [`resolve_fixes`]'s per-GT winner.
    dt_id: u64,
    /// Set by [`resolve_fixes`]: this error is its target GT's winner, so
    /// fixing it yields a TP. Every other `Loc`/`Cls` error is suppressed by
    /// the fix. `false` until resolved.
    won: bool,
}

/// Per-category detection records, accumulated across every in-scope cell.
struct CatData {
    scores: Vec<f64>,
    matched: Vec<bool>,
    ignored: Vec<bool>,
    /// Error type for each FP detection (`None` = TP or ignored).
    fp_types: Vec<Option<ErrType>>,
    /// Usable GT each `Loc`/`Cls` error targets; `None` everywhere else.
    fix_targets: Vec<Option<FixTarget>>,
    /// Set by [`resolve_fixes`]: scores of promoted `Cls` errors from *other*
    /// categories whose target GT is in this one, sorted descending. The `Cls`
    /// fix scores them here as TPs.
    incoming_cls: Vec<f64>,
    num_gt: usize,
}

impl CatData {
    fn new() -> Self {
        CatData {
            scores: Vec::new(),
            matched: Vec::new(),
            ignored: Vec::new(),
            fp_types: Vec::new(),
            fix_targets: Vec::new(),
            incoming_cls: Vec::new(),
            num_gt: 0,
        }
    }

    /// Permute every parallel array into score-descending order, once.
    ///
    /// Each category is scored up to nine ways in [`COCOeval::category_deltas`]
    /// (a baseline, five per-error-type fixes, Miss, and the FP/FN oracles) over
    /// the same detections. Ranking once here lets all of them use
    /// [`average_precision_ranked`](crate::metrics::counts::average_precision_ranked)
    /// instead of re-sorting — 3285 sorts of up to 22k elements on Objects365.
    ///
    /// Bit-identical because the comparator and the stability are the same: this
    /// is exactly the permutation `average_precision` computes, and stably sorting
    /// an already-sorted array is the identity.
    fn rank_by_score_desc(&mut self) {
        let mut order: Vec<usize> = (0..self.scores.len()).collect();
        order.sort_by_key(|&i| crate::metrics::counts::descending_score_key(self.scores[i]));
        self.scores = order.iter().map(|&i| self.scores[i]).collect();
        self.matched = order.iter().map(|&i| self.matched[i]).collect();
        self.ignored = order.iter().map(|&i| self.ignored[i]).collect();
        self.fp_types = order.iter().map(|&i| self.fp_types[i]).collect();
        self.fix_targets = order.iter().map(|&i| self.fix_targets[i]).collect();
    }

    /// Absorb another split's records for the same category, in order. Called
    /// only from [`Classified::merge`], which guarantees `self` is the earlier
    /// split — see that function's ordering note.
    fn extend(&mut self, mut other: CatData) {
        self.scores.append(&mut other.scores);
        self.matched.append(&mut other.matched);
        self.ignored.append(&mut other.ignored);
        self.fp_types.append(&mut other.fp_types);
        self.fix_targets.append(&mut other.fix_targets);
        self.num_gt += other.num_gt;
    }
}

/// Everything the false-positive classification pass produces.
#[derive(Default)]
struct Classified {
    /// Score-ranked, keyed by category id.
    cat_data: HashMap<u64, CatData>,
    /// Tallied by `err as usize`, so indexed like [`FP_TYPES`]. A fixed array
    /// rather than a keyed map: this runs over every detection, and a keyed map
    /// costs a `String` allocation per false positive just to reach a counter.
    fp_counts: [u64; FP_TYPES.len()],
}

impl Classified {
    /// Combine two splits of the parallel fold. `fp_counts` is elementwise
    /// integer addition, order-independent on its own; the per-category vectors inside
    /// `cat_data`, which `CatData::extend` appends in `self`-then-`other` order.
    /// That is safe only because rayon's `fold`/`reduce` always calls this with
    /// `self` covering the earlier contiguous range of `cells` and `other` the
    /// later one (the same left-then-right structure a sequential walk would
    /// produce), which is exactly what keeps `rank_by_score_desc`'s stable sort
    /// tie-breaking bit-identical to the sequential pass.
    fn merge(mut self, other: Self) -> Self {
        for (cat_id, data) in other.cat_data {
            // A vacant slot moves `data`'s `Vec`s in directly; only a category
            // that already has records from an earlier split pays for a copy
            // (`CatData::extend`'s `append`). `or_insert_with(CatData::new)`
            // would pay that copy on *every* first sighting of a category —
            // appending onto an empty `Vec` still allocates and memcpys, since
            // the empty `Vec` and `data`'s `Vec` are different allocations.
            match self.cat_data.entry(cat_id) {
                Entry::Vacant(v) => {
                    v.insert(data);
                }
                Entry::Occupied(o) => o.into_mut().extend(data),
            }
        }
        for (slot, n) in self.fp_counts.iter_mut().zip(other.fp_counts) {
            *slot += n;
        }
        self
    }
}

/// Undetected-ground-truth tallies, split two ways.
#[derive(Default)]
struct MissCounts {
    /// Miss errors across every category — the reported `"Miss"` count.
    total: u64,
    /// Miss errors per category: unmatched, in-denominator, and *not* covered by
    /// any `Loc`/`Cls` fix.
    per_cat: HashMap<u64, usize>,
    /// Every unmatched in-denominator GT per category, covered or not. This is
    /// what tidecv's FalseNeg oracle counts; the covered/uncovered split only
    /// narrows `per_cat`.
    fn_per_cat: HashMap<u64, usize>,
}

impl MissCounts {
    /// Combine two splits of the parallel fold. Every field is plain integer
    /// addition keyed by category id, so unlike [`Classified::merge`] this
    /// merge has no order dependency at all — which side is `self` vs `other`
    /// cannot change the result.
    fn merge(mut self, other: Self) -> Self {
        self.total += other.total;
        for (cat_id, n) in other.per_cat {
            *self.per_cat.entry(cat_id).or_insert(0) += n;
        }
        for (cat_id, n) in other.fn_per_cat {
            *self.fn_per_cat.entry(cat_id).or_insert(0) += n;
        }
        self
    }
}

/// Per-category buffers the error fixes in pass 4 build their modified copy
/// of a category's detections into, reused across the fixes instead of
/// allocated per call. Kept apart from [`ApScratch`] so a ranked-AP call can
/// read these while borrowing that one mutably.
#[derive(Debug, Default)]
struct FixScratch {
    matched: Vec<bool>,
    ignored: Vec<bool>,
}

/// One category's ΔAP contributions, in report order.
struct CatDeltas {
    baseline: f64,
    /// Indexed like [`FP_TYPES`].
    fp_types: [f64; FP_TYPES.len()],
    miss: f64,
    fp: f64,
    fn_oracle: f64,
}

impl COCOeval {
    /// Decompose detection errors into TIDE error types.
    ///
    /// Requires [`evaluate`](COCOeval::evaluate) to have been called first.
    ///
    /// Returns a [`TideErrors`] with ΔAP values and counts for six error types:
    ///
    /// | Error | Meaning |
    /// |-------|---------|
    /// | `Cls`  | Wrong class, correct location (IoU ≥ `pos_thr` with other-class GT) |
    /// | `Loc`  | Right class, poor localization (`bg_thr` ≤ IoU < `pos_thr`) |
    /// | `Both` | Wrong class AND poor localization |
    /// | `Dupe` | Duplicate — correct class GT already claimed by higher-scoring TP |
    /// | `Bkg`  | Pure background (IoU < `bg_thr` with all GTs) |
    /// | `Miss` | Undetected GT (false negative) |
    ///
    /// The five passes below run in this order because each consumes the one
    /// before it — in particular, Miss can only be narrowed once *every* category
    /// has been classified, since a dog detection can cover a cat ground truth.
    pub fn tide_errors(&self, pos_thr: f64, bg_thr: f64) -> crate::error::Result<TideErrors> {
        if !self.evaluated() {
            return Err("tide_errors() requires evaluate() to be called first".into());
        }

        // `pos_thr` is an analysis threshold, not a metric name, so it snaps to
        // the nearest grid point rather than requiring an exact match; `t_idx`
        // is that point's evaluate-time row.
        let (_, t_idx) = self.snapped_iou_row(pos_thr)?;

        // Filtered once; the classification and Miss passes walk the same cells.
        // `default_cells` owns the (area = "all", default max_det) predicate.
        let cells: Vec<&EvalImg> = self.default_cells().collect();

        let cross_iou_map = self.cross_category_ious();
        let mut classified =
            self.classify_detections(&cells, &cross_iou_map, t_idx, pos_thr, bg_thr);
        let covered_gts = resolve_fixes(&mut classified.cat_data);
        let misses = count_misses(&cells, &covered_gts, t_idx);
        let per_cat = self.category_deltas(&classified.cat_data, &misses);

        Ok(assemble(
            &per_cat,
            classified.fp_counts,
            misses.total,
            pos_thr,
            bg_thr,
        ))
    }

    /// Pass 1 — for every detection, the highest IoU against a ground truth of a
    /// *different* category, and which GT that was.
    ///
    /// Whole-run work, not per-image: `cat_slots` is built once here rather than
    /// inside the fan-out, per `cross_category_pairs`'s ordering note.
    fn cross_category_ious(&self) -> CrossIouMap {
        let cat_slots = Self::cat_slots(&self.params.cat_ids);
        let iou_type = self.params.iou_type;
        let coco_gt = &self.coco_gt;
        let coco_dt = &self.coco_dt;
        // `tide_errors` requires `evaluate()` first (checked by its caller), so on
        // a segm run this cache is populated and the matrices below skip
        // re-rasterizing every polygon.
        let segm_rles = self.segm_rles.as_ref();

        self.params
            .img_ids
            .par_iter()
            .map(|&img_id| {
                let mut dt_max_cross: HashMap<u64, (f64, Option<u64>)> = HashMap::new();

                // All non-crowd GTs and all DTs in the image, tagged with their
                // category slot. `None`: TIDE takes every detection in index
                // order — it scores each against its own `eval_imgs` entry, so it
                // needs neither a score floor nor a cap of its own.
                let (gt_pairs, dt_pairs) =
                    Self::cross_category_pairs(coco_gt, coco_dt, &cat_slots, img_id, None);

                if dt_pairs.is_empty() || gt_pairs.is_empty() {
                    for &(_, ann_id) in &dt_pairs {
                        dt_max_cross.insert(ann_id, (0.0, None));
                    }
                    return (img_id, dt_max_cross);
                }

                // Cross-category IoU matrix [D × G].
                let dt_ids: Vec<u64> = dt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let gt_ids: Vec<u64> = gt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let iou_matrix = Self::cross_category_iou(
                    &dt_ids, &gt_ids, coco_dt, coco_gt, iou_type, segm_rles,
                );

                // Strictly greater, so ties keep the earliest GT in `gt_pairs`
                // order. An IoU of exactly 0.0 never claims the argmax, which is
                // why "no overlap" and "no cross-category GT" both read as `None`.
                for (di, &(dt_cat_idx, dt_ann_id)) in dt_pairs.iter().enumerate() {
                    let row = &iou_matrix[di * gt_pairs.len()..(di + 1) * gt_pairs.len()];
                    let mut max_cross = 0.0f64;
                    let mut argmax_cross_gt = None;
                    for (gi, &(gt_cat_idx, gt_ann_id)) in gt_pairs.iter().enumerate() {
                        if gt_cat_idx != dt_cat_idx && row[gi] > max_cross {
                            max_cross = row[gi];
                            argmax_cross_gt = Some(gt_ann_id);
                        }
                    }
                    dt_max_cross.insert(dt_ann_id, (max_cross, argmax_cross_gt));
                }

                (img_id, dt_max_cross)
            })
            .collect()
    }

    /// Pass 2 — sort every detection into TP, ignored, or one of the five FP
    /// types, accumulating the per-category arrays the ΔAP pass scores.
    ///
    /// Also records the ground truth each `Loc` or `Cls` error targets;
    /// [`resolve_fixes`] then derives which ground truths a fix would recover,
    /// and pass 3 subtracts those from Miss.
    ///
    /// Fanned out with `fold`/`reduce`: each split walks a contiguous run of
    /// `cells` sequentially into its own `Classified`, and `Classified::merge`
    /// combines splits left-to-right.
    ///
    /// This looks like `confusion_matrix`'s fold/reduce accumulation but is not
    /// the same shape underneath, and copying it verbatim for another heavy
    /// payload would be a mistake: `confusion_matrix` merges by elementwise
    /// addition on a fixed k² matrix, so every split's accumulator is the same
    /// small constant size no matter how much work built it, and the merge
    /// order genuinely does not matter. `Classified::merge` instead moves
    /// per-detection `Vec`s that grow with the input, and the merge order *does*
    /// matter for one field. `fp_counts` is plainly order-independent
    /// (integer addition); the per-category
    /// vectors in `cat_data` are only order-independent because rayon's
    /// split/merge tree preserves `cells` order — see `Classified::merge`'s doc
    /// for why that matters for `rank_by_score_desc`'s tie-breaking.
    fn classify_detections(
        &self,
        cells: &[&EvalImg],
        cross_iou_map: &CrossIouMap,
        t_idx: usize,
        pos_thr: f64,
        bg_thr: f64,
    ) -> Classified {
        // GT id → category, for every GT a fix could still turn into a TP:
        // unmatched at this threshold and in the recall denominator. Built before
        // the fan-out because a `Cls` error's target lives in another category's
        // cell, which the error's own cell cannot see.
        let usable_gts: HashMap<u64, u64> = cells
            .par_iter()
            .flat_map_iter(|eval_img| {
                eval_img
                    .gt_ids
                    .iter()
                    .enumerate()
                    .filter(|&(gi, _)| {
                        !eval_img.gt_matched[(t_idx, gi)] && eval_img.counts_as_miss(gi)
                    })
                    .map(|(_, &gt_id)| (gt_id, eval_img.category_id))
            })
            .collect();

        let mut classified = cells
            .par_iter()
            .fold(Classified::default, |mut acc, eval_img| {
                let img_id = eval_img.image_id;
                let cat_id = eval_img.category_id;

                // Annotation id → its row/column in the cell's IoU matrix, which is
                // indexed by *original* (JSON-order) position within the cell.
                //
                // A linear scan, not a `HashMap`: `d` and `g` are single digits in
                // almost every cell, so building two hash tables per cell — hashing
                // every id, allocating twice — cost more than the handful of integer
                // compares it saved. Annotation ids are unique, so `position` and a
                // map lookup return the same answer.
                let dt_orig_ids = self.coco_dt.get_ann_ids_for_img_cat(img_id, cat_id);
                let gt_orig_ids = self.coco_gt.get_ann_ids_for_img_cat(img_id, cat_id);
                let orig_pos = |ids: &[u64], id: u64| ids.iter().position(|&x| x == id);
                // Sorted GT position → column, resolved once per cell. The scan in
                // `same_class_scan` is (detections × GTs), so resolving it there would
                // repeat the lookup once per pair.
                let gt_sorted_to_orig: Vec<Option<usize>> = eval_img
                    .gt_ids
                    .iter()
                    .map(|&id| orig_pos(gt_orig_ids, id))
                    .collect();

                let same_iou_mat = self.cell_ious(img_id, cat_id);
                let cross_map = cross_iou_map.get(&img_id);

                // Split borrows: `entry` pins `cat_data` for the whole loop while
                // the classification closure mutates the sibling fields, keeping the
                // entry lookup once per cell rather than once per detection.
                let entry = acc.cat_data.entry(cat_id).or_insert_with(CatData::new);
                entry.num_gt += eval_img.num_gt_in_denominator();
                let fp_counts = &mut acc.fp_counts;

                for (di, &dt_ann_id) in eval_img.dt_ids.iter().enumerate() {
                    let is_matched = eval_img.dt_matched[(t_idx, di)];
                    let is_ignored = eval_img.dt_ignore[(t_idx, di)];
                    let mut fix_target = None;

                    let fp_type = (!is_matched && !is_ignored).then(|| {
                        let (max_cross_iou, argmax_cross_gt) = cross_map
                            .and_then(|m| m.get(&dt_ann_id))
                            .copied()
                            .unwrap_or((0.0, None));

                        // One row borrow per detection. A detection or row that is out
                        // of range reads as all-zero, as the bounds tests it replaces
                        // did.
                        let row = same_iou_mat
                            .zip(orig_pos(dt_orig_ids, dt_ann_id))
                            .and_then(|(mat, di_orig)| mat.row(di_orig))
                            .unwrap_or(&[]);
                        let same =
                            same_class_scan(row, &gt_sorted_to_orig, eval_img, t_idx, pos_thr);

                        // The tidecv priority order lives in `classify_fp` — see its
                        // docs for the table. Everything above this line is evidence
                        // gathering; the decision itself is the parity contract.
                        let err = classify_fp(
                            &FpEvidence {
                                max_same_iou: same.max_iou,
                                max_cross_iou,
                                best_same_gt_matched: same.best_gt_matched,
                            },
                            pos_thr,
                            bg_thr,
                        );

                        // Only `Loc` and `Cls` can be fixed into a TP for their target
                        // GT, so only they cover it — and only if that GT is still
                        // usable; an error aimed at an already-matched GT is
                        // suppressed by its fix (tidecv's `BestGTMatch`).
                        // `Bkg`/`Both`/`Dupe` fixes suppress the detection outright,
                        // leaving the GT still missed.
                        let target = match err {
                            ErrType::Loc => same.argmax_gt_ann_id,
                            ErrType::Cls => argmax_cross_gt,
                            ErrType::Both | ErrType::Dupe | ErrType::Bkg => None,
                        };
                        if let Some((gt_id, &gt_cat)) =
                            target.and_then(|id| usable_gts.get_key_value(&id))
                        {
                            fix_target = Some(FixTarget {
                                gt_id: *gt_id,
                                gt_cat,
                                dt_id: dt_ann_id,
                                won: false,
                            });
                        }

                        fp_counts[err as usize] += 1;
                        err
                    });

                    entry.scores.push(eval_img.dt_scores[di]);
                    entry.matched.push(is_matched);
                    entry.ignored.push(is_ignored);
                    entry.fp_types.push(fp_type);
                    entry.fix_targets.push(fix_target);
                }

                acc
            })
            // `reduce_with` (not `reduce(Classified::default, ..)`): a `reduce`
            // with an identity seeds *every* leaf of the split/merge tree with
            // an empty `Classified`, so every leaf's `Classified::merge` pays
            // for a no-op merge against that empty seed — one extra memcpy per
            // leaf for data that was never going to change. `reduce_with` folds
            // pairwise instead and has no identity to seed, so a run with a
            // single rayon split (or `cells.is_empty()`) hits neither the
            // identity cost nor a "no splits happened" branch. Still
            // order-preserving left-to-right, so `Classified::merge`'s ordering
            // argument still holds.
            .reduce_with(Classified::merge)
            .unwrap_or_default();

        // Rank once per category, now that every cell has contributed; the nine
        // APs per category in `category_deltas` then read the presorted entry
        // point instead of re-sorting the same detections nine times.
        for data in classified.cat_data.values_mut() {
            data.rank_by_score_desc();
        }

        classified
    }

    /// Pass 4 — score each category up to nine ways and difference against its
    /// baseline AP.
    ///
    /// Fanned out over categories. `par_iter().map(..).collect()` is an *indexed*
    /// collect, so the results come back in `cat_ids` order and the means in
    /// [`assemble`] sum in exactly the sequence a sequential loop would — no float
    /// reordering.
    fn category_deltas(
        &self,
        cat_data: &HashMap<u64, CatData>,
        misses: &MissCounts,
    ) -> Vec<CatDeltas> {
        let rec_thrs = &self.params.rec_thrs;

        let per_cat: Vec<Option<CatDeltas>> = self
            .params
            .cat_ids
            .par_iter()
            .map(|&cat_id| {
                let data = match cat_data.get(&cat_id) {
                    Some(d) if d.num_gt > 0 => d,
                    _ => return None,
                };

                // One AP scratch per category, reused across the ranked-AP calls
                // below (baseline, one per `FP_TYPES` entry, FP oracle, FN oracle,
                // Miss) instead of each allocating its own TP/FP and PR-curve
                // buffers. `fix` holds the modified copies of `data` that `fix_fp`
                // and the FP oracle build.
                let mut ap_scratch = ApScratch::default();
                let mut fix = FixScratch::default();

                let baseline = average_precision_ranked_into(
                    &data.matched,
                    Some(&data.ignored),
                    data.num_gt,
                    rec_thrs,
                    &mut ap_scratch,
                );

                // Fix one FP error type, following tidecv's `fix_errors`. A `Loc`
                // or `Cls` error that [`resolve_fixes`] promoted becomes a TP —
                // here for `Loc`, in its target GT's category for `Cls` (scored
                // there via `incoming_cls`). Every other error of the fixed type,
                // and every `Bkg`/`Both`/`Dupe` error, is suppressed: tidecv's
                // `fix()→None` drops the detection, neither TP nor FP.
                //
                // `mut` and captures `ap_scratch` and `fix` by unique reference:
                // each call below reuses the same buffers rather than allocating.
                let mut fix_fp = |fix_type: ErrType| -> f64 {
                    let incoming: &[f64] = if fix_type == ErrType::Cls {
                        &data.incoming_cls
                    } else {
                        &[]
                    };
                    fix.matched.clear();
                    fix.ignored.clear();
                    // Merge the incoming TPs into the ranked detections. Strictly
                    // greater, so on a score tie the category's own detection
                    // ranks first — the order a stable sort of (own ++ incoming)
                    // would give.
                    let mut next_in = 0;
                    for (i, fp_type) in data.fp_types.iter().enumerate() {
                        while next_in < incoming.len() && incoming[next_in] > data.scores[i] {
                            fix.matched.push(true);
                            fix.ignored.push(false);
                            next_in += 1;
                        }
                        let (mut matched, mut ignored) = (data.matched[i], data.ignored[i]);
                        if *fp_type == Some(fix_type) {
                            if fix_type == ErrType::Loc
                                && data.fix_targets[i].is_some_and(|t| t.won)
                            {
                                matched = true;
                            } else {
                                ignored = true;
                            }
                        }
                        fix.matched.push(matched);
                        fix.ignored.push(ignored);
                    }
                    let rest = incoming.len() - next_in;
                    fix.matched.extend(std::iter::repeat_n(true, rest));
                    fix.ignored.extend(std::iter::repeat_n(false, rest));
                    average_precision_ranked_into(
                        &fix.matched,
                        Some(&fix.ignored),
                        data.num_gt,
                        rec_thrs,
                        &mut ap_scratch,
                    )
                };

                let mut fp_types = [0.0f64; FP_TYPES.len()];
                for (slot, &err) in fp_types.iter_mut().zip(FP_TYPES.iter()) {
                    *slot = fix_fp(err) - baseline;
                }

                // FP: tidecv's FalsePositiveError oracle — perfect precision
                // without affecting recall. Every false positive is scored out of
                // existence; nothing is converted into a TP, unlike the per-type
                // fixes above, so this is *not* the union of the five.
                let fp = {
                    fix.ignored.clear();
                    fix.ignored.extend(
                        data.ignored
                            .iter()
                            .zip(&data.fp_types)
                            .map(|(&ig, fp_type)| ig || fp_type.is_some()),
                    );
                    average_precision_ranked_into(
                        &data.matched,
                        Some(&fix.ignored),
                        data.num_gt,
                        rec_thrs,
                        &mut ap_scratch,
                    ) - baseline
                };

                // Score with `n` ground truths out of the denominator and the
                // detections untouched — the FN oracle and the Miss fix below.
                let mut drop_gts = |n: usize| {
                    average_precision_ranked_into(
                        &data.matched,
                        Some(&data.ignored),
                        data.num_gt.saturating_sub(n),
                        rec_thrs,
                        &mut ap_scratch,
                    ) - baseline
                };

                // FN: tidecv's FalseNegativeError oracle — perfect recall without
                // affecting precision. Every unmatched in-denominator GT leaves
                // the denominator. A superset of Miss, which drops only the GTs
                // no Loc/Cls fix could recover.
                let fn_count = misses.fn_per_cat.get(&cat_id).copied().unwrap_or(0);
                debug_assert!(
                    fn_count <= data.num_gt,
                    "FN count exceeds the GT denominator it was counted from"
                );
                let fn_oracle = drop_gts(fn_count);

                // Fix Miss: tidecv's `MissedError.fix()` returns `(class, -1)` —
                // each missed GT no Loc/Cls fix could recover leaves the
                // denominator.
                let miss = drop_gts(misses.per_cat.get(&cat_id).copied().unwrap_or(0));

                Some(CatDeltas {
                    baseline,
                    fp_types,
                    miss,
                    fp,
                    fn_oracle,
                })
            })
            .collect();

        per_cat.into_iter().flatten().collect()
    }
}

/// Scan one detection's row of its cell's same-class IoU matrix.
///
/// `gt_sorted_to_orig` maps each sorted GT position to its column, so this walks
/// GTs in `eval_img.gt_ids` order — which is what makes `argmax_gt_ann_id`
/// reproducible when two GTs tie.
fn same_class_scan(
    row: &[f64],
    gt_sorted_to_orig: &[Option<usize>],
    eval_img: &EvalImg,
    t_idx: usize,
    pos_thr: f64,
) -> SameClassScan {
    let mut scan = SameClassScan::default();
    for (gi_sorted, &gi_orig) in gt_sorted_to_orig.iter().enumerate() {
        let Some(gi_orig) = gi_orig else {
            continue;
        };
        let iou = row.get(gi_orig).copied().unwrap_or(0.0);
        if iou > scan.max_iou {
            scan.max_iou = iou;
            scan.argmax_gt_ann_id = Some(eval_img.gt_ids[gi_sorted]);
        }
        if iou >= pos_thr && eval_img.gt_matched[(t_idx, gi_sorted)] {
            scan.best_gt_matched = true;
        }
    }
    scan
}

/// Decide which `Loc`/`Cls` errors their fix turns into a TP — tidecv's
/// `BestGTMatch`.
///
/// A ground truth can be recovered once. Of all the `Loc` and `Cls` errors
/// targeting the same usable GT, only the highest-scoring is promoted; the rest
/// are suppressed by the fix. The contest spans both types (tidecv keeps
/// `best_score` on the GT), so fixing `Loc` alone still suppresses a `Loc`
/// error whose GT a higher-scoring `Cls` error claims.
///
/// On a score tie the lower detection id wins. tidecv keeps the first error it
/// sees, walking each image's detections in a stable score-descending sort of
/// their input order — the same order as ascending id whenever ids follow input
/// order, as `load_res` assigns them.
///
/// Runs after ranking, so the `incoming_cls` scores it gathers come from the
/// ranked arrays. Every comparison is a total order on `(score, dt_id)`, so the
/// `HashMap` iteration order cannot change the outcome.
///
/// Returns the GTs some `Loc`/`Cls` error targets — tidecv's `usable` flag.
/// Those are not Miss errors: a fix could recover them. A `Loc` detection
/// targets the same-class GT with highest IoU in the `Loc` window; a `Cls`
/// detection the cross-class GT with highest IoU >= `pos_thr`, which is why
/// this needs every category at once.
fn resolve_fixes(cat_data: &mut HashMap<u64, CatData>) -> HashSet<u64> {
    let wins = |(s, id): (f64, u64), (best_s, best_id): (f64, u64)| {
        s > best_s || (s == best_s && id < best_id)
    };

    let mut best: HashMap<u64, (f64, u64)> = HashMap::new();
    for data in cat_data.values() {
        for (t, &score) in data.fix_targets.iter().zip(&data.scores) {
            let Some(t) = t else { continue };
            let cand = (score, t.dt_id);
            best.entry(t.gt_id)
                .and_modify(|b| {
                    if wins(cand, *b) {
                        *b = cand;
                    }
                })
                .or_insert(cand);
        }
    }

    let mut incoming: HashMap<u64, Vec<f64>> = HashMap::new();
    for (&cat_id, data) in cat_data.iter_mut() {
        for (t, &score) in data.fix_targets.iter_mut().zip(&data.scores) {
            let Some(t) = t else { continue };
            t.won = best[&t.gt_id].1 == t.dt_id;
            // Only a `Cls` error targets another category's GT.
            if t.won && t.gt_cat != cat_id {
                incoming.entry(t.gt_cat).or_default().push(score);
            }
        }
    }
    for (cat_id, mut scores) in incoming {
        // Every incoming entry is a TP, so the order among equal scores is
        // immaterial; descending is what the merge in `fix_fp` needs.
        scores.sort_by_key(|&s| crate::metrics::counts::descending_score_key(s));
        cat_data
            .entry(cat_id)
            .or_insert_with(CatData::new)
            .incoming_cls = scores;
    }
    best.into_keys().collect()
}

/// Pass 3 — count undetected ground truths, after every category has been
/// classified.
///
/// A GT is a Miss only if it is unmatched, non-ignored, *and* not covered by a
/// `Loc`/`Cls` fix. The cross-category half of that coverage is why this cannot
/// be folded into [`COCOeval::classify_detections`]: a dog detection can cover a
/// cat ground truth, so no single category's pass has the whole answer.
///
/// Fanned out with `fold`/`reduce`, same shape as [`COCOeval::classify_detections`]
/// and `confusion_matrix`. Every field `MissCounts::merge` combines is additive
/// (a total, and two per-category maps of counts), so — unlike the classification
/// pass — this merge is order-independent outright; no tie-break anywhere here
/// depends on which order the cells were visited in.
fn count_misses(cells: &[&EvalImg], covered_gts: &HashSet<u64>, t_idx: usize) -> MissCounts {
    cells
        .par_iter()
        .fold(MissCounts::default, |mut counts, eval_img| {
            let (mut n_miss, mut n_fn) = (0usize, 0usize);
            for (gi, &gt_id) in eval_img.gt_ids.iter().enumerate() {
                // `counts_as_miss`, not `!gt_ignore`: `num_gt` in the classification
                // pass already counts an Open Images group-of box, so excluding it
                // here would make Miss disagree with the denominator it is a fraction
                // of.
                if eval_img.gt_matched[(t_idx, gi)] || !eval_img.counts_as_miss(gi) {
                    continue;
                }
                n_fn += 1;
                if !covered_gts.contains(&gt_id) {
                    n_miss += 1;
                }
            }
            counts.total += n_miss as u64;
            // Callers read both maps via `.get(&cat_id).copied().unwrap_or(0)`,
            // so a category this cell contributed nothing to is identical
            // whether it holds an explicit `0` entry or no entry at all — skip
            // the insertion for the common case (most cells miss nothing) and
            // save the `HashMap` a slot it will never be asked for.
            if n_miss > 0 {
                *counts.per_cat.entry(eval_img.category_id).or_insert(0) += n_miss;
            }
            if n_fn > 0 {
                *counts.fn_per_cat.entry(eval_img.category_id).or_insert(0) += n_fn;
            }
            counts
        })
        .reduce_with(MissCounts::merge)
        .unwrap_or_default()
}

/// Pass 5 — average the per-category deltas and name every key once.
fn assemble(
    per_cat: &[CatDeltas],
    fp_counts: [u64; FP_TYPES.len()],
    miss_total: u64,
    pos_thr: f64,
    bg_thr: f64,
) -> TideErrors {
    let mean = |f: &dyn Fn(&CatDeltas) -> f64| -> f64 {
        if per_cat.is_empty() {
            0.0
        } else {
            per_cat.iter().map(f).sum::<f64>() / per_cat.len() as f64
        }
    };

    let mut delta_ap: BTreeMap<String, f64> = FP_TYPES
        .iter()
        .enumerate()
        .map(|(i, err)| (err.as_str().to_string(), mean(&|c| c.fp_types[i])))
        .collect();
    delta_ap.insert("Miss".to_string(), mean(&|c| c.miss));
    delta_ap.insert("FP".to_string(), mean(&|c| c.fp));
    delta_ap.insert("FN".to_string(), mean(&|c| c.fn_oracle));

    // `"Miss"` is the extra key: it is a ground truth tally, not one of
    // `FP_TYPES`.
    let mut counts: BTreeMap<String, u64> = FP_TYPES
        .iter()
        .zip(fp_counts)
        .map(|(err, n)| (err.as_str().to_string(), n))
        .collect();
    counts.insert("Miss".to_string(), miss_total);

    TideErrors {
        delta_ap,
        counts,
        ap_base: mean(&|c| c.baseline),
        pos_thr,
        bg_thr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POS: f64 = 0.5;
    const BG: f64 = 0.1;

    fn ev(max_same_iou: f64, max_cross_iou: f64, best_same_gt_matched: bool) -> FpEvidence {
        FpEvidence {
            max_same_iou,
            max_cross_iou,
            best_same_gt_matched,
        }
    }

    fn classify(same: f64, cross: f64, dupe: bool) -> ErrType {
        classify_fp(&ev(same, cross, dupe), POS, BG)
    }

    #[test]
    fn each_error_type_is_reachable() {
        // Loc:  same-class IoU inside [bg, pos]
        assert_eq!(classify(0.3, 0.0, false), ErrType::Loc);
        // Cls:  no Loc, but a cross-class GT is well overlapped
        assert_eq!(classify(0.0, 0.9, false), ErrType::Cls);
        // Dupe: same-class GT above pos_thr already taken by a higher-scoring TP
        assert_eq!(classify(0.9, 0.0, true), ErrType::Dupe);
        // Bkg:  overlaps nothing
        assert_eq!(classify(0.0, 0.0, false), ErrType::Bkg);
        // Both: cross-class IoU strictly inside (bg, pos)
        assert_eq!(classify(0.0, 0.3, false), ErrType::Both);
    }

    #[test]
    fn loc_outranks_cls_and_dupe() {
        // A DT can satisfy Loc *and* Cls; tidecv reports Loc.
        assert_eq!(classify(0.3, 0.9, false), ErrType::Loc);
        // Loc also wins over a would-be Dupe.
        assert_eq!(classify(0.3, 0.0, true), ErrType::Loc);
    }

    #[test]
    fn cls_outranks_dupe_and_bkg() {
        assert_eq!(classify(0.9, 0.9, true), ErrType::Cls);
        // Cross-class overlap above pos_thr beats the "overlaps nothing" reading.
        assert_eq!(classify(0.0, 0.5, false), ErrType::Cls);
    }

    #[test]
    fn dupe_outranks_bkg() {
        // Same-class IoU above pos_thr, so Loc's upper bound excludes it; the
        // already-matched flag is then what distinguishes Dupe from Bkg.
        assert_eq!(classify(0.9, 0.0, true), ErrType::Dupe);
        assert_eq!(classify(0.9, 0.0, false), ErrType::Bkg);
    }

    /// The `Loc` window is closed at both ends. tidecv uses `>=` and `<=`, and
    /// the inclusive upper bound is specifically what keeps a detection sitting
    /// exactly on `pos_thr` out of `Dupe`.
    #[test]
    fn loc_window_is_inclusive_at_both_ends() {
        assert_eq!(classify(BG, 0.0, false), ErrType::Loc);
        assert_eq!(classify(POS, 0.0, true), ErrType::Loc);
        // Just outside the window on either side is not Loc.
        assert_ne!(classify(BG - 1e-9, 0.0, false), ErrType::Loc);
        assert_ne!(classify(POS + 1e-9, 0.0, true), ErrType::Loc);
    }

    /// `Bkg`'s bound is also inclusive: cross-class IoU exactly at `bg_thr` is
    /// background, and anything strictly above it falls through to `Both`.
    #[test]
    fn bkg_upper_bound_is_inclusive() {
        assert_eq!(classify(0.0, BG, false), ErrType::Bkg);
        assert_eq!(classify(0.0, BG + 1e-9, false), ErrType::Both);
    }

    #[test]
    fn as_str_covers_every_variant() {
        for (err, key) in [
            (ErrType::Cls, "Cls"),
            (ErrType::Loc, "Loc"),
            (ErrType::Both, "Both"),
            (ErrType::Dupe, "Dupe"),
            (ErrType::Bkg, "Bkg"),
        ] {
            assert_eq!(err.as_str(), key);
        }
    }

    /// `err as usize` indexes the count tally and the per-type delta vectors, so
    /// it must agree with the position in [`FP_TYPES`]. Reordering the enum
    /// declaration without reordering the array would silently file every `Loc`
    /// under `Cls`.
    #[test]
    fn fp_types_are_indexed_by_discriminant() {
        for (i, &err) in FP_TYPES.iter().enumerate() {
            assert_eq!(err as usize, i, "{} is out of order", err.as_str());
        }
    }
}

/// TIDE error decomposition for object detection.
///
/// Produced by [`super::COCOeval::tide_errors`]. Each ΔAP value measures how much
/// average AP would improve if all errors of that type were fixed.
#[derive(Debug, Clone, Serialize)]
pub struct TideErrors {
    /// ΔAP for each error type (fixing all errors of that type).
    /// Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`, `"FP"`, `"FN"`.
    ///
    /// `"FP"` and `"FN"` are tidecv's *special* oracles, not sums of the five
    /// types: `"FP"` suppresses every false positive (perfect precision,
    /// recall untouched); `"FN"` removes every unmatched in-denominator GT
    /// from the denominator (perfect recall, precision untouched). `"FN"`
    /// covers a superset of the GTs behind `"Miss"`, which drops only those
    /// no Loc/Cls fix could recover.
    pub delta_ap: BTreeMap<String, f64>,
    /// Count of each error type across all categories and images.
    /// Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`.
    pub counts: BTreeMap<String, u64>,
    /// Baseline AP at `pos_thr` (mean over categories with GT).
    pub ap_base: f64,
    /// IoU threshold for TP/FP classification.
    pub pos_thr: f64,
    /// Background IoU threshold for Loc/Both/Bkg discrimination.
    pub bg_thr: f64,
}
