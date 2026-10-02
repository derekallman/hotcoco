//! Incremental evaluation: one batch of images at a time, the batch code path
//! each time.
//!
//! [`COCOeval::evaluate`] is a whole-dataset batch. Every ground truth and
//! detection has to be loaded and indexed before the first (image, category)
//! cell is matched, so loading the results, building the evaluator, and
//! `evaluate()` itself all sit on the critical path at the end of a validation
//! epoch. Matching is per image, though: no cell reads another image's
//! annotations. [`StreamingEval`] uses that to match each detector batch as
//! soon as its detections exist and to leave only `accumulate()` and
//! `summarize()` for the end.
//!
//! Each `update()` runs a `COCOeval` over just that batch's images through the
//! same `evaluate()` the batch path runs, and keeps the lean cells it produces
//! — the record `accumulate()` reads — split back into one run per image. The
//! small datasets are dropped. `finalize()` lays every image's cells out in the
//! `(image, category)` order `evaluate()` uses, whatever order the batches
//! arrived in. Bit for bit the same `accumulate()` input as a batch run over the
//! same annotations, because it is the same code.
//!
//! This module is private; the published contract is on [`StreamingEval`].

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

use crate::coco::COCO;
use crate::params::Params;
use crate::types::{Annotation, Category, Dataset, Image};

use super::matching::Cells;
use super::{COCOeval, EvalMode};

/// Incremental evaluator: feed detector batches as they come, get an ordinary
/// [`COCOeval`] back.
///
/// [`update`](Self::update) matches a batch of images as soon as its
/// detections exist and keeps about 20 bytes per detection;
/// [`finalize`](Self::finalize) returns an evaluator whose numbers are
/// identical to a batch run over the same annotations, whatever order the
/// images arrived in or how they were batched.
///
/// # What the finalized evaluator supports
///
/// `accumulate()`, `summarize()`, `report()`, `results()`, `slice_by()`, and
/// [`compare`](super::compare) read the lean cells, `params`, and category
/// names, and work as they do on a batch evaluator.
///
/// [`COCOeval::eval_imgs`] is empty, and `confusion_matrix()`,
/// `tide_errors()`, `calibration()`, `f_scores()`, and `image_diagnostics()`
/// see no cells: they need the full per-image records, which are rebuilt from
/// the datasets, and a streamed evaluator has none. Build a batch [`COCOeval`]
/// for those.
///
/// # Restrictions
///
/// - **No Open Images.** [`new`](Self::new) returns an error for
///   [`EvalMode::OpenImages`]: hierarchy expansion needs the whole ground
///   truth before the first image is evaluated.
/// - **The category list is fixed at construction.** Batch `evaluate()` reads
///   the ground truth's categories before the first cell is matched, so a
///   category with no annotations still gets a `-1.0` slot on the K axis
///   instead of vanishing. Pass every category the run will see to `new()`.
///   An annotation naming a category outside the list is an error in
///   [`update`](Self::update), where a batch `COCOeval` drops it without a word:
///   the list cannot grow, so the label would be lost from every metric.
/// - **`params` is frozen at construction.** `iou_thrs`, `area_ranges`, and
///   `max_dets` shape every stored cell; `use_cats` and `cat_ids` fix the
///   K axis.
///
/// # Examples
///
/// ```no_run
/// use hotcoco::{COCO, EvalMode, Params, StreamingEval};
/// use hotcoco::params::IouType;
/// # fn main() -> hotcoco::error::Result<()> {
/// # let coco_gt = COCO::new(std::path::Path::new("gt.json"))?;
/// # let detections_for = |_ids: &[u64]| Vec::new();
/// let categories = coco_gt.dataset.categories.clone();
/// let mut se = StreamingEval::new(Params::new(IouType::Bbox), EvalMode::Coco, categories)?;
/// for batch in coco_gt.dataset.images.chunks(32) {
///     let ids: Vec<u64> = batch.iter().map(|img| img.id).collect();
///     let ann_ids = coco_gt.get_ann_ids(&ids, &[], None, None);
///     let gt_anns = coco_gt.load_anns(&ann_ids).into_iter().cloned().collect();
///     se.update(batch.to_vec(), gt_anns, detections_for(&ids))?;
/// }
/// let mut ev = se.finalize();
/// ev.accumulate();
/// ev.summarize();
/// # Ok(())
/// # }
/// ```
pub struct StreamingEval {
    params: Params,
    eval_mode: EvalMode,
    /// The categories and nothing else: what `new()` derives the K axis from
    /// and what the finalized evaluator reads category names off.
    categories: COCO,
    /// The ids in `categories`, which [`unknown_category_ids`](Self::unknown_category_ids)
    /// checks annotations against.
    known_categories: HashSet<u64>,
    /// One arena per `update()` call, kept whole.
    batches: Vec<Cells>,
    /// Every image seen, keyed by id so `finalize()` gathers image-ascending —
    /// the order `evaluate()` sorts its pairs into — mapped to its run of pairs
    /// in `batches`. An image seen again points at its newest run; the stale
    /// run stays in its arena until `finalize()` drops them all.
    images: BTreeMap<u64, (usize, Range<usize>)>,
}

impl StreamingEval {
    /// Start an incremental evaluation.
    ///
    /// `categories` should list every category the run will see, including
    /// ones with no annotations in any image, so each gets a K-axis slot. When
    /// `params.cat_ids` is empty it is filled from `categories` the way
    /// [`COCOeval::evaluate`] fills it from a ground-truth dataset; a
    /// non-empty `params.img_ids` is kept as the image filter, sorted.
    ///
    /// # Errors
    ///
    /// Returns an error for [`EvalMode::OpenImages`] — see the
    /// [restrictions](Self#restrictions).
    pub fn new(
        mut params: Params,
        eval_mode: EvalMode,
        categories: Vec<Category>,
    ) -> crate::error::Result<Self> {
        if eval_mode == EvalMode::OpenImages {
            return Err(crate::error::Error::Other(
                "StreamingEval does not support Open Images: hierarchy expansion needs the \
                 whole ground truth before the first image is evaluated. Use COCOeval::new_oid."
                    .to_string(),
            ));
        }

        let known_categories = categories.iter().map(|c| c.id).collect();
        let categories = COCO::from_dataset(Dataset {
            categories,
            ..Default::default()
        });
        if params.cat_ids.is_empty() {
            params.cat_ids = categories.get_cat_ids(&[], &[], &[]);
        }
        params.img_ids.sort_unstable();
        params.img_ids.dedup();

        Ok(StreamingEval {
            params,
            eval_mode,
            categories,
            known_categories,
            batches: Vec::new(),
            images: BTreeMap::new(),
        })
    }

    /// The ids of categories in `anns` that this evaluator was not constructed
    /// with, sorted and without duplicates.
    ///
    /// [`update`](Self::update) rejects a batch whose annotations name any of
    /// them. Exposed so a binding, or a caller validating data before streaming
    /// it, can ask the same question without sending a batch.
    ///
    /// Empty whenever `params.use_cats` is false: every annotation pools into
    /// one placeholder category then, so no id can be lost. The list checked is
    /// the one `new()` received, not `params.cat_ids`, which only narrows what
    /// is evaluated.
    pub fn unknown_category_ids<'a>(
        &self,
        anns: impl IntoIterator<Item = &'a Annotation>,
    ) -> Vec<u64> {
        if !self.params.use_cats {
            return Vec::new();
        }
        let mut ids: Vec<u64> = anns
            .into_iter()
            .map(|a| a.category_id)
            .filter(|id| !self.known_categories.contains(id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Match a batch of images' ground truth against their detections now.
    ///
    /// `images` are the batch's image records — every annotation's `image_id`
    /// must name one of them, and in LVIS mode they carry `neg_category_ids`
    /// and `not_exhaustive_category_ids`. A batch of one is fine; a detector's
    /// whole batch amortizes the per-call setup. Ground-truth ids need only be
    /// unique within the batch. Detections are loaded the way
    /// [`COCO::load_res_anns`] loads a results file: ids are assigned, `area`
    /// and the geometry the result kind implies are derived, and `iscrowd` is
    /// cleared — so raw predictions (`image_id`, `category_id`, `bbox`,
    /// `score`) are what to pass. Within an image, detections with tied scores
    /// rank in the order given, as they do in a results file.
    ///
    /// An image seen again in a later call replaces its earlier result. When
    /// `params.img_ids` is non-empty, images outside it are skipped.
    ///
    /// # Errors
    ///
    /// A ground truth or detection whose category is not in the list given to
    /// [`new`](Self::new) is an error naming every such id: an off-by-one class
    /// map or a background id would otherwise drop out of every metric without
    /// a trace. See [`unknown_category_ids`](Self::unknown_category_ids). The
    /// batch is rejected whole; the evaluator is as it was before the call.
    ///
    /// A detection with a NaN score is an error, as it is in `load_res`.
    pub fn update(
        &mut self,
        images: Vec<Image>,
        gt_anns: Vec<Annotation>,
        dt_anns: Vec<Annotation>,
    ) -> crate::error::Result<()> {
        let unknown = self.unknown_category_ids(gt_anns.iter().chain(&dt_anns));
        if !unknown.is_empty() {
            return Err(crate::error::Error::Other(format!(
                "category id(s) {unknown:?} are not in this StreamingEval's categories; pass \
                 every category the run will see to `categories`"
            )));
        }

        let mut ids: Vec<u64> = images.iter().map(|img| img.id).collect();
        ids.sort_unstable();
        ids.dedup();
        if !self.params.img_ids.is_empty() {
            ids.retain(|id| self.params.img_ids.binary_search(id).is_ok());
        }
        if ids.is_empty() {
            return Ok(());
        }

        // Datasets for just this batch. The ground truth carries the image
        // records (LVIS reads its category lists off them, and mask conversion
        // takes the raster size from them) and no categories: those are
        // resolved once, in `new()`. The detections go through
        // `load_res_anns` like a results file, which copies the images
        // across, assigns ids, and derives the geometry. The result kind is
        // read off this batch's first detection rather than the file's, the
        // same answer for a homogeneous run.
        let gt = COCO::from_dataset(Dataset {
            images,
            annotations: gt_anns,
            ..Default::default()
        });
        let dt = gt.load_res_anns(dt_anns)?;

        // This batch's ids are the run's scope: `evaluate()` keeps a non-empty
        // `img_ids` as given, so it filters its pairs through these few rather
        // than the whole run's list.
        let mut params = self.params.clone();
        params.img_ids.clone_from(&ids);
        let mut ev = COCOeval::with_mode(gt, dt, params, self.eval_mode, None);
        ev.evaluate();
        let cells = std::mem::take(&mut ev.cells);

        // `evaluate()` sorts its pairs by (image, category), so each image is
        // one contiguous run — empty for an image with nothing to match, which
        // still counts as seen.
        let batch = self.batches.len();
        let mut start = 0;
        for id in ids {
            let end = start
                + (start..cells.len())
                    .take_while(|&p| cells.ids(p).0 == id)
                    .count();
            self.images.insert(id, (batch, start..end));
            start = end;
        }
        debug_assert_eq!(start, cells.len(), "every pair belongs to a batch image");
        self.batches.push(cells);
        Ok(())
    }

    /// Assemble every image seen so far into a [`COCOeval`] ready for
    /// `accumulate()` → `summarize()` → `report()`.
    ///
    /// The cells are laid out image-ascending, category-ascending within an
    /// image, each pair with one slot per area range: the layout
    /// [`COCOeval::evaluate`] leaves. `params.img_ids`, when empty, becomes
    /// the ids seen, as `evaluate()` fills it from the ground truth. The
    /// evaluator's ground truth carries the categories and nothing else — see
    /// [what that supports](Self#what-the-finalized-evaluator-supports).
    pub fn finalize(self) -> COCOeval {
        let StreamingEval {
            mut params,
            eval_mode,
            categories,
            batches,
            images,
            known_categories: _,
        } = self;

        if params.img_ids.is_empty() {
            params.img_ids = images.keys().copied().collect();
        }
        let cells = Cells::gather(
            &params,
            images
                .values()
                .map(|(batch, range)| (&batches[*batch], range.clone())),
        );
        COCOeval::from_cells(categories, params, eval_mode, cells)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::params::IouType;

    fn ann(
        id: u64,
        image_id: u64,
        category_id: u64,
        bbox: [f64; 4],
        score: Option<f64>,
    ) -> Annotation {
        Annotation {
            id,
            image_id,
            category_id,
            bbox: Some(bbox),
            area: Some(bbox[2] * bbox[3]),
            score,
            ..Default::default()
        }
    }

    fn image(id: u64) -> Image {
        Image {
            id,
            width: 640,
            height: 480,
            ..Default::default()
        }
    }

    fn category(id: u64) -> Category {
        Category {
            id,
            name: format!("cat{id}"),
            ..Default::default()
        }
    }

    /// Three images, two categories, one category with no annotations at
    /// all, tied scores in one cell, and a detection-only cell.
    fn fixture() -> (Dataset, Dataset) {
        let categories = vec![category(1), category(2), category(3)];
        let images = vec![image(3), image(1), image(2)];
        let gts = vec![
            ann(1, 1, 1, [10.0, 10.0, 50.0, 50.0], None),
            ann(2, 1, 1, [100.0, 100.0, 50.0, 50.0], None),
            ann(3, 2, 2, [0.0, 0.0, 20.0, 20.0], None),
            ann(4, 3, 1, [200.0, 200.0, 120.0, 120.0], None),
        ];
        let dts = vec![
            ann(101, 1, 1, [12.0, 12.0, 50.0, 50.0], Some(0.9)),
            ann(102, 1, 1, [102.0, 102.0, 50.0, 50.0], Some(0.9)),
            ann(103, 1, 1, [300.0, 300.0, 10.0, 10.0], Some(0.9)),
            ann(104, 2, 2, [1.0, 1.0, 20.0, 20.0], Some(0.7)),
            // Detection-only cell: category 1 has no ground truth in image 2.
            ann(105, 2, 1, [50.0, 50.0, 20.0, 20.0], Some(0.6)),
            ann(106, 3, 1, [210.0, 210.0, 120.0, 120.0], Some(0.8)),
        ];
        let gt = Dataset {
            images: images.clone(),
            annotations: gts,
            categories: categories.clone(),
            ..Default::default()
        };
        let dt = Dataset {
            images,
            annotations: dts,
            categories,
            ..Default::default()
        };
        (gt, dt)
    }

    /// Stream `gt`/`dt_anns` into a `StreamingEval`, one `update()` per batch
    /// of image ids in `batches`, each batch carrying its images' annotations
    /// in dataset order.
    fn stream(gt: &Dataset, dt_anns: &[Annotation], batches: &[&[u64]]) -> COCOeval {
        let mut streaming = StreamingEval::new(
            Params::new(IouType::Bbox),
            EvalMode::Coco,
            gt.categories.clone(),
        )
        .expect("mode is supported");
        let of = |anns: &[Annotation], ids: &HashSet<u64>| -> Vec<Annotation> {
            anns.iter()
                .filter(|a| ids.contains(&a.image_id))
                .cloned()
                .collect()
        };
        for batch in batches {
            let ids: HashSet<u64> = batch.iter().copied().collect();
            streaming
                .update(
                    batch.iter().map(|&id| image(id)).collect(),
                    of(&gt.annotations, &ids),
                    of(dt_anns, &ids),
                )
                .expect("scores are finite");
        }
        streaming.finalize()
    }

    fn batch_eval(gt: &Dataset, dt: &Dataset) -> COCOeval {
        let mut batch = COCOeval::new(
            COCO::from_dataset(gt.clone()),
            COCO::from_dataset(dt.clone()),
            IouType::Bbox,
        );
        batch.evaluate();
        batch
    }

    /// The lean cells — what `accumulate()` reads — come out identical to the
    /// batch path's, including pair order, however the images are batched and
    /// in whatever order the batches arrive.
    #[test]
    fn cells_match_batch_for_any_batching_and_order() {
        let (gt, dt) = fixture();
        let batch = batch_eval(&gt, &dt);
        let expected = format!("{:?}", batch.cells);

        for batches in [
            vec![&[2u64][..], &[3][..], &[1][..]],
            vec![&[3u64, 1][..], &[2][..]],
            vec![&[2u64, 3, 1][..]],
        ] {
            let streamed = stream(&gt, &dt.annotations, &batches);
            assert_eq!(expected, format!("{:?}", streamed.cells), "{batches:?}");
            let (batch_inputs, streamed_inputs) = (
                batch.eval_inputs.as_ref().expect("evaluated"),
                streamed.eval_inputs.as_ref().expect("finalized"),
            );
            assert_eq!(batch_inputs.sparse_pairs, streamed_inputs.sparse_pairs);
            assert_eq!(batch.params.img_ids, streamed.params.img_ids);
            assert_eq!(batch.params.cat_ids, streamed.params.cat_ids);
            assert!(streamed.evaluated());
            assert!(
                streamed.eval_imgs().is_empty(),
                "no datasets to rebuild from"
            );
        }
    }

    /// Raw predictions — no `id`, no `area` — are what a training loop has.
    /// The batch path gives them ids and areas in `load_res`; the streaming
    /// path must do the same, or every detection in an image collides on
    /// id 0 and matching silently degenerates.
    #[test]
    fn raw_detections_match_batch_load_res() {
        let (gt, mut dt) = fixture();
        for d in &mut dt.annotations {
            d.id = 0;
            d.area = None;
        }
        let gt_coco = COCO::from_dataset(gt.clone());
        let dt_coco = gt_coco
            .load_res_anns(dt.annotations.clone())
            .expect("finite scores");
        let mut batch = COCOeval::new(gt_coco, dt_coco, IouType::Bbox);
        batch.evaluate();
        let streamed = stream(&gt, &dt.annotations, &[&[1, 2, 3]]);
        assert_eq!(
            format!("{:?}", batch.cells),
            format!("{:?}", streamed.cells)
        );
    }

    /// An image with nothing in it still counts as seen: it lands in
    /// `params.img_ids` the way an annotation-free ground-truth image does.
    #[test]
    fn empty_image_is_seen_but_contributes_no_cells() {
        let mut streaming = StreamingEval::new(
            Params::new(IouType::Bbox),
            EvalMode::Coco,
            vec![category(1)],
        )
        .expect("mode is supported");
        streaming
            .update(vec![image(7)], Vec::new(), Vec::new())
            .expect("nothing to reject");
        let ev = streaming.finalize();
        assert_eq!(ev.params.img_ids, vec![7]);
        assert_eq!(ev.cells.len(), 0);
    }

    #[test]
    fn images_outside_params_img_ids_are_skipped() {
        let mut params = Params::new(IouType::Bbox);
        params.img_ids = vec![1];
        let mut streaming = StreamingEval::new(params, EvalMode::Coco, vec![category(1)])
            .expect("mode is supported");
        streaming
            .update(
                vec![image(1), image(2)],
                vec![
                    ann(1, 1, 1, [0.0, 0.0, 10.0, 10.0], None),
                    ann(2, 2, 1, [0.0, 0.0, 10.0, 10.0], None),
                ],
                Vec::new(),
            )
            .expect("skipped, not rejected");
        let ev = streaming.finalize();
        assert_eq!(ev.params.img_ids, vec![1]);
        assert_eq!(ev.cells.len(), 1, "only image 1's pair survives");
        assert_eq!(ev.cells.ids(0), (1, 1));
    }

    /// An image sent twice contributes its newest run only, at its place in
    /// image order — not a second copy where the later batch landed.
    #[test]
    fn reseen_image_replaces_its_earlier_run() {
        let (gt, dt) = fixture();
        let batch = batch_eval(&gt, &dt);
        // First pass gives image 2 the wrong detections; the second corrects it.
        let mut wrong = dt.annotations.clone();
        for d in &mut wrong {
            if d.image_id == 2 {
                d.bbox = Some([600.0, 400.0, 10.0, 10.0]);
            }
        }
        let mut streaming = StreamingEval::new(
            Params::new(IouType::Bbox),
            EvalMode::Coco,
            gt.categories.clone(),
        )
        .expect("mode is supported");
        let anns_of = |anns: &[Annotation], ids: &[u64]| -> Vec<Annotation> {
            anns.iter()
                .filter(|a| ids.contains(&a.image_id))
                .cloned()
                .collect()
        };
        for (ids, dts) in [(vec![1u64, 2, 3], &wrong), (vec![2u64], &dt.annotations)] {
            streaming
                .update(
                    ids.iter().map(|&id| image(id)).collect(),
                    anns_of(&gt.annotations, &ids),
                    anns_of(dts, &ids),
                )
                .expect("scores are finite");
        }
        let streamed = streaming.finalize();
        assert_eq!(
            format!("{:?}", batch.cells),
            format!("{:?}", streamed.cells)
        );
    }

    #[test]
    fn nan_score_is_an_error() {
        let mut streaming = StreamingEval::new(
            Params::new(IouType::Bbox),
            EvalMode::Coco,
            vec![category(1)],
        )
        .expect("mode is supported");
        let err = streaming
            .update(
                vec![image(1)],
                Vec::new(),
                vec![ann(0, 1, 1, [0.0, 0.0, 10.0, 10.0], Some(f64::NAN))],
            )
            .expect_err("NaN score is rejected");
        assert!(err.to_string().contains("NaN"));
    }

    fn streaming_over(categories: Vec<Category>) -> StreamingEval {
        StreamingEval::new(Params::new(IouType::Bbox), EvalMode::Coco, categories)
            .expect("mode is supported")
    }

    #[test]
    fn unknown_category_ids_are_sorted_and_unique() {
        let se = streaming_over(vec![category(1), category(2)]);
        let anns = [
            ann(1, 1, 7, [0.0, 0.0, 5.0, 5.0], None),
            ann(2, 1, 3, [0.0, 0.0, 5.0, 5.0], None),
            ann(3, 1, 7, [0.0, 0.0, 5.0, 5.0], None),
            ann(4, 1, 1, [0.0, 0.0, 5.0, 5.0], None),
        ];
        assert_eq!(se.unknown_category_ids(&anns), vec![3, 7]);
        assert!(se.unknown_category_ids(&[]).is_empty());
    }

    /// An off-by-one class map or a background id used to vanish from every
    /// metric without a trace: `evaluate()` only visits the categories it was
    /// built with.
    #[test]
    fn detection_in_an_unlisted_category_is_an_error() {
        let mut se = streaming_over(vec![category(1)]);
        let err = se
            .update(
                vec![image(2)],
                Vec::new(),
                vec![ann(0, 2, 7, [0.0, 0.0, 5.0, 5.0], Some(0.5))],
            )
            .expect_err("category 7 is not in the list");
        let msg = err.to_string();
        assert!(msg.contains("[7]"), "names the id: {msg}");
        assert!(msg.contains("categories"), "says what to fix: {msg}");
    }

    /// A ground truth in an unlisted category is lost the same way, so both
    /// sides are checked — and every offender is named, not the first.
    #[test]
    fn ground_truth_in_an_unlisted_category_is_an_error() {
        let mut se = streaming_over(vec![category(1)]);
        let err = se
            .update(
                vec![image(2)],
                vec![
                    ann(1, 2, 9, [0.0, 0.0, 5.0, 5.0], None),
                    ann(2, 2, 4, [0.0, 0.0, 5.0, 5.0], None),
                ],
                Vec::new(),
            )
            .expect_err("categories 4 and 9 are not in the list");
        assert!(err.to_string().contains("[4, 9]"), "{err}");
    }

    /// A rejected batch must leave the run as if it had never been sent, so a
    /// caller that catches the error and carries on gets honest numbers.
    #[test]
    fn rejected_batch_leaves_the_evaluator_untouched() {
        let mut se = streaming_over(vec![category(1)]);
        se.update(
            vec![image(1)],
            vec![ann(1, 1, 1, [0.0, 0.0, 10.0, 10.0], None)],
            vec![ann(0, 1, 1, [0.0, 0.0, 10.0, 10.0], Some(0.9))],
        )
        .expect("known category");
        se.update(
            vec![image(2)],
            Vec::new(),
            vec![ann(0, 2, 7, [0.0, 0.0, 5.0, 5.0], Some(0.5))],
        )
        .expect_err("category 7 is not in the list");
        let ev = se.finalize();
        assert_eq!(ev.params.img_ids, vec![1], "image 2 was never recorded");
        assert_eq!(ev.cells.len(), 1);
    }

    /// Without `use_cats` every annotation pools into one placeholder
    /// category, so no id can vanish and none is checked.
    #[test]
    fn categories_play_no_role_without_use_cats() {
        let mut params = Params::new(IouType::Bbox);
        params.use_cats = false;
        let mut se = StreamingEval::new(params, EvalMode::Coco, vec![category(1)])
            .expect("mode is supported");
        let dt = [ann(0, 2, 7, [0.0, 0.0, 5.0, 5.0], Some(0.5))];
        assert!(se.unknown_category_ids(&dt).is_empty());
        se.update(vec![image(2)], Vec::new(), dt.to_vec())
            .expect("pooled, not rejected");
    }

    /// `cat_ids` narrows what is evaluated; `categories` is what is known. A
    /// deliberate subset must keep working, so a detection in a listed
    /// category the run excluded is skipped, not rejected.
    #[test]
    fn category_outside_cat_ids_but_listed_is_not_an_error() {
        let mut params = Params::new(IouType::Bbox);
        params.cat_ids = vec![1];
        let mut se = StreamingEval::new(params, EvalMode::Coco, vec![category(1), category(2)])
            .expect("mode is supported");
        se.update(
            vec![image(1)],
            Vec::new(),
            vec![ann(0, 1, 2, [0.0, 0.0, 5.0, 5.0], Some(0.5))],
        )
        .expect("category 2 is listed, just not evaluated");
        assert_eq!(se.finalize().cells.len(), 0);
    }

    #[test]
    fn rejects_open_images() {
        let Err(err) = StreamingEval::new(
            Params::new(IouType::Bbox),
            EvalMode::OpenImages,
            vec![category(1)],
        ) else {
            panic!("Open Images must be rejected");
        };
        assert!(err.to_string().contains("Open Images"));
    }
}
