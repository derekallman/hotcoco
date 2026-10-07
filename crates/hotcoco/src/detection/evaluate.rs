use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;

use super::matching::{
    Cells, EvalImgContext, IouMatrix, PairScratch, lean_scores_len, push_pair_lean,
};
use super::{COCOeval, EvalMode};
use crate::primitives::sim::worth_parallel;

impl COCOeval {
    /// Populate `params.img_ids` and `params.cat_ids` from the GT dataset if not already set.
    ///
    /// The writing half of [`COCOeval::resolved_ids`], which owns the derivation:
    /// "which ids does this dataset cover" is one question, and `confusion_matrix`
    /// asks it too without being allowed to mutate. Answering it separately here
    /// meant two places to keep agreeing about it.
    fn resolve_params(&mut self) {
        let (img_ids, cat_ids) = self.resolved_ids();
        let (img_ids, cat_ids) = (img_ids.into_owned(), cat_ids.into_owned());
        self.params.img_ids = img_ids;
        self.params.cat_ids = cat_ids;
    }

    /// Build the sorted list of (img_id, cat_id) pairs to evaluate.
    ///
    /// Takes the union of non-empty GT and DT pairs, filters to the active img_ids/cat_ids
    /// from params, and returns them sorted for deterministic output order.
    ///
    /// In LVIS mode, DT-only pairs are dropped unless the category appears in `neg_cats`
    /// for that image — that is, it was confirmed absent and unmatched DTs should count as FP.
    fn collect_sparse_pairs(
        &self,
        cat_ids: &[u64],
        neg_cats: &HashMap<u64, HashSet<u64>>,
    ) -> Vec<(u64, u64)> {
        // At large-scale (e.g. Objects365: 365 cats × 80K imgs = 29M pairs), ~96% of pairs
        // are empty. Driving evaluation from the index instead reduces pairs by ~35x.
        //
        // Images ascending, and within each image the two indexes' categories,
        // both ascending, merged: the pairs come out sorted and distinct with no
        // set of pairs to build and no sort, which at 10x COCO val2017 took a
        // third of `evaluate()`, on one thread.
        let mut img_ids = self.params.img_ids.clone();
        img_ids.sort_unstable();
        img_ids.dedup();
        if !self.params.use_cats {
            return img_ids
                .into_iter()
                .filter(|&img_id| {
                    !self.coco_gt.get_ann_ids_for_img(img_id).is_empty()
                        || !self.coco_dt.get_ann_ids_for_img(img_id).is_empty()
                })
                .map(|img_id| (img_id, u64::MAX))
                .collect();
        }
        let allowed_cats: rustc_hash::FxHashSet<u64> = cat_ids.iter().copied().collect();
        let lvis = self.eval_mode == EvalMode::Lvis;
        // At most every pair of both indexes, so the list never regrows.
        let mut pairs = Vec::with_capacity(
            self.coco_gt.nonempty_pair_count() + self.coco_dt.nonempty_pair_count(),
        );
        for img_id in img_ids {
            // LVIS keeps a category only the detections have where the image
            // lists it as confirmed absent.
            let neg = neg_cats.get(&img_id);
            let mut gt = self
                .coco_gt
                .cat_ids_of_img(img_id)
                .filter(|cat| allowed_cats.contains(cat))
                .peekable();
            let mut dt = self
                .coco_dt
                .cat_ids_of_img(img_id)
                .filter(|cat| allowed_cats.contains(cat))
                .peekable();
            loop {
                let cat = match (gt.peek().copied(), dt.peek().copied()) {
                    (None, None) => break,
                    // The ground truth's category, and the detections' too
                    // when they share it.
                    (Some(g), d) if d.is_none_or(|d| g <= d) => {
                        gt.next();
                        if d == Some(g) {
                            dt.next();
                        }
                        g
                    }
                    // A category only the detections have.
                    (_, Some(d)) => {
                        dt.next();
                        if lvis && !neg.is_some_and(|s| s.contains(&d)) {
                            continue;
                        }
                        d
                    }
                    (Some(_), None) => unreachable!("taken by the first arm"),
                };
                pairs.push((img_id, cat));
            }
        }
        // `EvalInputs` keeps the list for the evaluator's lifetime; shrinking
        // a large allocation usually happens in place.
        pairs.shrink_to_fit();
        pairs
    }

    /// Every GT/DT annotation id living in a `sparse_pairs` cell where *both*
    /// sides are non-empty — the only cells that ever reach the segm kernel
    /// inside [`Self::compute_iou_static`]; a cell with only GT or only DT
    /// returns early there without reading a mask. `sparse_pairs` itself is
    /// the union (GT-only ∪ DT-only ∪ both, with the LVIS `neg_cats`
    /// carve-out for DT-only) — narrower than "every id in scope", but still
    /// wider than what segm mask conversion needs, so this filters it once
    /// more down to the both-non-empty subset for [`super::iou::SegmRles::prepare`].
    fn segm_cell_ann_ids(&self, sparse_pairs: &[(u64, u64)]) -> (Vec<u64>, Vec<u64>) {
        let mut gt_ids = Vec::new();
        let mut dt_ids = Vec::new();
        for &(img_id, cat_id) in sparse_pairs {
            let gt = Self::get_anns_static(&self.coco_gt, &self.params, img_id, cat_id);
            let dt = Self::get_anns_static(&self.coco_dt, &self.params, img_id, cat_id);
            if !gt.is_empty() && !dt.is_empty() {
                gt_ids.extend_from_slice(&gt);
                dt_ids.extend_from_slice(&dt);
            }
        }
        (gt_ids, dt_ids)
    }

    /// Check that the inputs can be evaluated under the current `params`.
    ///
    /// [`evaluate`](Self::evaluate) runs it first, so a caller needs it only
    /// to validate inputs without evaluating them. Today it covers mask
    /// geometry (segm), where a polygon on an image without `height` and
    /// `width` rasterizes to an empty mask and AP comes out 0 with nothing to
    /// say why — see [`COCO::check_mask_dims`](crate::COCO::check_mask_dims).
    ///
    /// # Errors
    ///
    /// An error naming the offending images, from either the ground truth or
    /// the detections. Only the images and categories the evaluation covers
    /// are checked.
    pub fn check_inputs(&self) -> crate::error::Result<()> {
        use crate::primitives::sim::SimKind;
        if SimKind::from(self.params.iou_type) == SimKind::Mask {
            // Only the (image, category) pairs `evaluate()` will visit: an
            // annotation on an image outside `params.img_ids`, with no image
            // record, or in a category outside `params.cat_ids` is never
            // rasterized. `use_cats = false` pools every category.
            let (img_ids, cat_ids) = self.resolved_ids();
            let cat_ids = self.params.use_cats.then_some(&*cat_ids);
            self.coco_gt.check_mask_dims(&img_ids, cat_ids)?;
            self.coco_dt.check_mask_dims(&img_ids, cat_ids)?;
        }
        Ok(())
    }

    /// Run per-image evaluation.
    ///
    /// LVIS's per-image detection cap is not applied here: [`new_lvis`](Self::new_lvis)
    /// already replaced `coco_dt` with the capped copy, the same handle-replacing
    /// pattern as Open Images below.
    ///
    /// # Open Images replaces `coco_gt` (and possibly `coco_dt`)
    ///
    /// In [`EvalMode::OpenImages`], this method **overwrites the
    /// `coco_gt` handle** — and, when `params.expand_dt` is set, `coco_dt` —
    /// with hierarchy-expanded copies: every annotation is duplicated at each
    /// ancestor category and virtual categories are added for hierarchy-only
    /// nodes (see [`super::expand::expand_annotations`]). Any read through
    /// [`coco_gt`](Self::coco_gt) after `evaluate()` sees the expanded dataset,
    /// not the one the evaluator was constructed with; the caller's own handle
    /// to the original is untouched. The expansion deduplicates, so calling
    /// `evaluate()` again does not expand further. The eval paths must see the
    /// expanded data through the same handles the analysis surfaces read (TIDE,
    /// diagnostics, category names), which is why the handles are replaced
    /// rather than shadowed by private copies.
    ///
    /// # Errors
    ///
    /// What [`check_inputs`](Self::check_inputs) reports: inputs that cannot
    /// be evaluated under the current `params`. Nothing is evaluated then.
    pub fn evaluate(&mut self) -> crate::error::Result<()> {
        self.check_inputs()?;
        // OID: expand GT (and optionally DT) using hierarchy — this replaces
        // the `coco_gt`/`coco_dt` handles; see the method docs above.
        if self.eval_mode == EvalMode::OpenImages {
            let hierarchy = self.hierarchy.clone().unwrap_or_else(|| {
                crate::detection::hierarchy::Hierarchy::from_categories(
                    &self.coco_gt.dataset.categories,
                )
            });
            self.coco_gt = Arc::new(super::expand::expand_annotations(&self.coco_gt, &hierarchy));
            if self.params.expand_dt {
                self.coco_dt =
                    Arc::new(super::expand::expand_annotations(&self.coco_dt, &hierarchy));
            }
            self.hierarchy = Some(hierarchy);
        }

        self.resolve_params();

        let cat_ids = if self.params.use_cats {
            self.params.cat_ids.clone()
        } else {
            vec![u64::MAX] // placeholder single category (avoids collision with real category_id=0)
        };

        // LVIS: scan GT image metadata to build per-image category sets.
        // Deferred from construction so the scan only happens when evaluate() is called.
        // neg_cats:       img_id → categories confirmed absent (unmatched DTs count as FP).
        // not_exhaustive: img_id → categories not fully checked (unmatched DTs are ignored).
        let (neg_cats, not_exhaustive) = if self.eval_mode == EvalMode::Lvis {
            let mut neg: HashMap<u64, HashSet<u64>> = HashMap::new();
            let mut not_ex: HashMap<u64, HashSet<u64>> = HashMap::new();
            for img in &self.coco_gt.dataset.images {
                if !img.neg_category_ids.is_empty() {
                    neg.insert(img.id, img.neg_category_ids.iter().copied().collect());
                }
                if !img.not_exhaustive_category_ids.is_empty() {
                    not_ex.insert(
                        img.id,
                        img.not_exhaustive_category_ids.iter().copied().collect(),
                    );
                }
            }
            (neg, not_ex)
        } else {
            (HashMap::new(), HashMap::new())
        };

        let sparse_pairs = self.collect_sparse_pairs(&cat_ids, &neg_cats);

        // Segm only: convert every mask that a both-non-empty cell will
        // actually read, once, up front — pycocotools' `_prepare` step,
        // narrowed to the cells `evaluate()` itself will touch (see
        // `segm_cell_ann_ids`). The cross-category matrices in
        // `confusion_matrix`/`tide` still read through the same cache and
        // fall back to converting on the spot on a miss — see
        // `SegmRles::gt_rle_or_convert`/`dt_rle_or_convert`.
        use crate::primitives::sim::SimKind;
        self.segm_rles = (SimKind::from(self.params.iou_type) == SimKind::Mask).then(|| {
            let (gt_ids, dt_ids) = self.segm_cell_ann_ids(&sparse_pairs);
            super::iou::SegmRles::prepare(&self.coco_gt, &self.coco_dt, &gt_ids, &dt_ids)
        });

        // Compute IoUs only for pairs where both GT and DT are non-empty.
        // Pairs with only GT or only DT produce empty IoU matrices — skip storing them.
        let iou_of = |&(img_id, cat_id): &(u64, u64)| {
            Self::compute_iou_static(
                &self.coco_gt,
                &self.coco_dt,
                &self.params,
                img_id,
                cat_id,
                self.eval_mode,
                self.segm_rles.as_ref(),
            )
            .map(|iou_matrix| ((img_id, cat_id), iou_matrix))
        };
        let iou_results: Vec<((u64, u64), IouMatrix)> = if self.parallel_pairs() {
            sparse_pairs.par_iter().filter_map(iou_of).collect()
        } else {
            sparse_pairs.iter().filter_map(iou_of).collect()
        };

        // Replaces the cache wholesale, so `collect` sizes the map from the vec's
        // exact length rather than inheriting the previous run's capacity.
        self.ious = iou_results.into_iter().collect();

        // Evaluate each (image, category, area_range) combination in parallel,
        // over sparse_pairs × area_ranges rather than the full
        // cat_ids × area_ranges × img_ids product. The inputs are snapshotted so
        // `eval_imgs()` can rebuild the full records later from exactly what
        // this run saw, whatever `params` is set to in between.
        let inputs = EvalInputs {
            params: self.params.clone(),
            sparse_pairs,
            not_exhaustive,
        };
        self.cells = self.evaluate_pairs_lean(&inputs);
        self.eval_imgs = std::sync::OnceLock::new();
        self.default_eval_imgs = std::sync::OnceLock::new();
        self.eval_inputs = Some(inputs);
        Ok(())
    }

    /// Whether `evaluate()`'s per-pair passes fan out. A pair's IoU and
    /// matching work scales with its annotations, so the threshold is measured
    /// against both datasets' annotation counts. A small `StreamingEval` batch
    /// stays on the calling thread.
    fn parallel_pairs(&self) -> bool {
        worth_parallel(
            self.coco_gt.dataset.annotations.len() + self.coco_dt.dataset.annotations.len(),
        )
    }

    /// Run `f` with the per-cell context for `params`, reading the IoU cache
    /// `evaluate()` filled. The second argument is the detection cap.
    fn with_cell_context<R>(
        &self,
        params: &crate::params::Params,
        f: impl FnOnce(&EvalImgContext<'_>, usize) -> R,
    ) -> R {
        // Empty `max_dets` is degraded, not panicked on: `Params::max_det()`
        // owns the fallback cap (100), matching how every other degenerate
        // configuration on this path (missing area label, absent threshold)
        // degrades to the `-1.0` sentinel downstream instead of aborting —
        // `evaluate()` has no `Result` channel, and its siblings do not panic.
        let max_det = params.max_det();

        // pycocotools searches from `min(t, 1-1e-10)`, not from `t`. Inert below
        // 1.0, so the default 0.50:0.95 sweep is untouched; at t == 1.0 it admits
        // near-identical pairs, which is the drop-in behavior. Resolved once here
        // and shared — see `EvalImgContext::match_floors`.
        let match_floors: Vec<f64> = params
            .iou_thrs
            .iter()
            .map(|&t| crate::primitives::greedy::coco_match_floor(t))
            .collect();

        let ctx = EvalImgContext {
            coco_gt: &self.coco_gt,
            coco_dt: &self.coco_dt,
            params,
            ious: &self.ious,
            is_kp: params.iou_type == crate::params::IouType::Keypoints,
            is_oid: self.eval_mode == EvalMode::OpenImages,
            match_floors: &match_floors,
        };
        f(&ctx, max_det)
    }

    /// LVIS: whether `cat_id` is not exhaustively annotated on `img_id`.
    fn not_exhaustive_cat(&self, inputs: &EvalInputs, img_id: u64, cat_id: u64) -> bool {
        self.eval_mode == EvalMode::Lvis
            && inputs
                .not_exhaustive
                .get(&img_id)
                .is_some_and(|s| s.contains(&cat_id))
    }

    /// The per-pair walk of `evaluate()`: every gathered pair under every area
    /// range, as the lean cells `accumulate()` reads, in `sparse_pairs` order.
    /// Fans out over runs of pairs, not (pair, area range) cells, so
    /// `gather_pair` resolves what the ranges share once, and each run writes
    /// its window of arenas sized before the walk: the records never exist as
    /// one object per pair, and nothing is copied afterwards.
    fn evaluate_pairs_lean(&self, inputs: &EvalInputs) -> Cells {
        self.with_cell_context(&inputs.params, |ctx, max_det| {
            Cells::build(
                &inputs.params,
                &inputs.sparse_pairs,
                self.parallel_pairs()
                    .then(|| super::run_len(inputs.sparse_pairs.len())),
                |img_id, cat_id| lean_scores_len(ctx, img_id, cat_id, max_det),
                |run, cells| {
                    // One set of vectors, refilled for each pair of the run.
                    let mut scratch = PairScratch::default();
                    for &(img_id, cat_id) in run {
                        push_pair_lean(
                            ctx,
                            (img_id, cat_id),
                            max_det,
                            self.not_exhaustive_cat(inputs, img_id, cat_id),
                            &mut scratch,
                            cells,
                        );
                    }
                },
            )
        })
    }

    /// The same walk producing full [`EvalImg`](super::matching::EvalImg)s
    /// for the area ranges at `area_idxs` (indices into
    /// `inputs.params.area_ranges`): one `area_idxs.len()` chunk per pair,
    /// empty gathers included, so length, order, and `None` positions never
    /// depend on the data. Cells are written in place — collect-then-flatten
    /// moved ~800 MB of `EvalImg`s single-threaded on Objects365 (measured
    /// 2.0 s vs 1.5 s), and the parallel `None` fill spreads the first touch of
    /// that buffer across threads (270 ms sequential).
    pub(super) fn evaluate_pairs_full(
        &self,
        inputs: &EvalInputs,
        area_idxs: &[usize],
    ) -> Vec<Option<super::matching::EvalImg>> {
        let n = area_idxs.len();
        let mut out: Vec<Option<super::matching::EvalImg>> = (0..inputs.sparse_pairs.len() * n)
            .into_par_iter()
            .map(|_| None)
            .collect();
        if n == 0 {
            return out;
        }
        self.with_cell_context(&inputs.params, |ctx, max_det| {
            out.par_chunks_mut(n)
                .zip(inputs.sparse_pairs.par_iter())
                .for_each_init(
                    PairScratch::default,
                    |scratch, (chunk, &(img_id, cat_id))| {
                        super::matching::evaluate_pair_full(
                            ctx,
                            (img_id, cat_id),
                            max_det,
                            self.not_exhaustive_cat(inputs, img_id, cat_id),
                            area_idxs,
                            scratch,
                            chunk,
                        );
                    },
                );
        });
        out
    }
}

/// What one `evaluate()` run saw, kept so [`COCOeval::eval_imgs`] can rebuild
/// the full per-image records from the same inputs on demand.
pub(super) struct EvalInputs {
    /// `params` as resolved for the run — the copy `eval_imgs()` reads, so a
    /// `max_dets` or `img_ids` edited afterwards for `accumulate()` cannot
    /// change what the records describe.
    pub(super) params: crate::params::Params,
    /// The (image, category) pairs visited, in visit order.
    pub(super) sparse_pairs: Vec<(u64, u64)>,
    /// LVIS: image → categories not exhaustively annotated there.
    pub(super) not_exhaustive: HashMap<u64, HashSet<u64>>,
}
