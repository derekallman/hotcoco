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

use std::collections::BTreeMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::coco::COCO;
use crate::params::{AreaRange, IouType, Params};
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
#[derive(Clone)]
pub struct StreamingEval {
    params: Params,
    eval_mode: EvalMode,
    /// The categories and nothing else: what `new()` derives the K axis from,
    /// what the finalized evaluator reads category names off, and what
    /// [`unknown_category_ids`](Self::unknown_category_ids) checks ids against.
    /// Its `dataset.categories` is the list `new()` received, in that order:
    /// what [`merge`](Self::merge) compares and a saved state stores.
    categories: COCO,
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
            batches: Vec::new(),
            images: BTreeMap::new(),
        })
    }

    /// The ids of categories in `anns` that this evaluator was not constructed
    /// with, sorted and without duplicates.
    ///
    /// [`update`](Self::update) rejects a batch whose annotations name any of
    /// them, with [`Error::UnknownCategoryIds`](crate::Error::UnknownCategoryIds).
    /// Exposed so a caller validating data before streaming it can ask the
    /// same question without sending a batch.
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
            .filter(|&id| self.categories.get_cat(id).is_none())
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
    /// whole batch amortizes the per-call setup. Ground-truth `id`s are
    /// assigned here and `area` is derived where missing (the mask's pixel
    /// count, or the box's `w × h` for an annotation without a mask), so
    /// targets in the shape a data loader yields — `image_id`, `category_id`,
    /// `bbox`, `iscrowd` — are enough; an authored `area` is kept. Detections are loaded the way [`COCO::load_res_anns`]
    /// loads a results file: ids are assigned, `area` and the geometry the
    /// result kind implies are derived, and `iscrowd` is cleared — so raw
    /// predictions (`image_id`, `category_id`, `bbox`, `score`) are what to
    /// pass. Within an image, detections with tied scores rank in the order
    /// given, as they do in a results file.
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
    ///
    /// In segm mode, a polygon or box on an image record without `height` and
    /// `width` is an error — see [`COCOeval::check_inputs`].
    pub fn update(
        &mut self,
        images: Vec<Image>,
        mut gt_anns: Vec<Annotation>,
        dt_anns: Vec<Annotation>,
    ) -> crate::error::Result<()> {
        let unknown = self.unknown_category_ids(gt_anns.iter().chain(&dt_anns));
        if !unknown.is_empty() {
            return Err(crate::error::Error::UnknownCategoryIds(unknown));
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
        //
        // Ground truth gets ids, as a results file does, because every
        // `iscrowd`/area read goes through the id index and targets from a
        // data loader carry no `id`: they would all collide on 0 and resolve
        // to the batch's last annotation. Nothing after this call reads a
        // ground-truth id — the cells keep counts and bits — so assigning
        // them loses nothing. A missing `area` is filled by the evaluator
        // built below, as every `COCOeval` fills it.
        for (i, ann) in gt_anns.iter_mut().enumerate() {
            ann.id = (i + 1) as u64;
        }
        let mut gt = COCO::from_dataset(Dataset {
            images,
            annotations: gt_anns,
            ..Default::default()
        });
        if crate::primitives::sim::SimKind::from(self.params.iou_type)
            == crate::primitives::sim::SimKind::Mask
        {
            keep_rasterized_masks_of_arealess(&mut gt);
        }
        let dt = gt.load_res_anns(dt_anns)?;

        // This batch's ids are the run's scope: `evaluate()` keeps a non-empty
        // `img_ids` as given, so it filters its pairs through these few rather
        // than the whole run's list.
        let mut params = self.params.clone();
        params.img_ids.clone_from(&ids);
        let mut ev = COCOeval::with_mode(gt, dt, params, self.eval_mode, None);
        ev.check_inputs()?;
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

    /// Fold `other`'s images into this evaluator, as if its `update()` calls
    /// had been made here.
    ///
    /// The use is a run split across processes: each rank streams its shard,
    /// and one rank merges the rest. An image present in both keeps `other`'s
    /// result, the rule [`update`](Self::update) applies to an image seen
    /// again. No matching is redone and no cell is copied; `other`'s arenas
    /// are moved over whole.
    ///
    /// # Errors
    ///
    /// Both evaluators must have been built with the same evaluation mode,
    /// categories, and params — every field, `img_ids` included. Categories
    /// are compared by id, name, and LVIS frequency, in any order: the K axis
    /// is sorted by id, so list order changes no number. Float params are
    /// compared bit for bit. A mismatch is an error naming the first field
    /// that differs, and `self` is as it was.
    pub fn merge(&mut self, other: StreamingEval) -> crate::error::Result<()> {
        if self.eval_mode != other.eval_mode {
            return Err(mismatch("eval_mode"));
        }
        if category_key(&self.categories) != category_key(&other.categories) {
            return Err(mismatch("categories"));
        }
        if let Some(field) = params_mismatch(&self.params, &other.params) {
            return Err(mismatch(&format!("params.{field}")));
        }
        let offset = self.batches.len();
        self.batches.extend(other.batches);
        self.images.extend(
            other
                .images
                .into_iter()
                .map(|(id, (batch, range))| (id, (batch + offset, range))),
        );
        Ok(())
    }

    /// The evaluator's state as bytes, for `from_bytes` to restore in this or
    /// another process.
    ///
    /// Only what `finalize()` reads is written: images replaced by a later
    /// `update()` are dropped, so the size follows the images seen, not the
    /// calls made. The format starts with a version number; the bytes are
    /// not a stable interchange format across hotcoco versions that change it.
    pub fn to_bytes(&self) -> Vec<u8> {
        let cells = self.live_cells();
        let counts = cells.arena_sizes();
        let mut next = 0;
        let images = self
            .images
            .iter()
            .map(|(&id, (_, range))| {
                next += range.len();
                (id, range.len())
            })
            .collect();
        debug_assert_eq!(next, counts.0);
        let header = serde_json::to_vec(&StateHeader {
            eval_mode: match self.eval_mode {
                EvalMode::Coco => "coco",
                EvalMode::Lvis => "lvis",
                EvalMode::OpenImages => unreachable!("rejected by new()"),
            }
            .to_string(),
            params: ParamsState::from(&self.params),
            categories: self.categories.dataset.categories.clone(),
            images,
            n_pairs: counts.0,
            n_scores: counts.1,
            n_words: counts.2,
        })
        .expect("a header of plain data serializes");
        let (n_areas, (n_pairs, n_scores, n_words)) = (self.params.area_ranges.len(), counts);
        let mut out = Vec::with_capacity(
            16 + header.len()
                + (n_pairs + 1) * 24
                + n_scores * 8
                + n_pairs * n_areas * 4
                + n_words * 8,
        );
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        cells.write_le_bytes(&mut out);
        out
    }

    /// Restore an evaluator from [`to_bytes`](Self::to_bytes) output.
    ///
    /// The result finalizes to what the original would have, accepts further
    /// `update()` calls, and merges like any other.
    ///
    /// # Errors
    ///
    /// Bytes that are truncated, were not written by `to_bytes`, or carry a
    /// format version this build does not know are an error, never a panic.
    pub fn from_bytes(bytes: &[u8]) -> crate::error::Result<Self> {
        let bad =
            |why: String| crate::error::Error::Other(format!("invalid StreamingEval state: {why}"));
        if bytes.len() < 16 || &bytes[..4] != STATE_MAGIC {
            return Err(bad("not a StreamingEval state".into()));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
        if version != STATE_VERSION {
            return Err(bad(format!(
                "format version {version}, this build reads version {STATE_VERSION}"
            )));
        }
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
        let header_end = usize::try_from(header_len)
            .ok()
            .and_then(|n| n.checked_add(16))
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| bad("truncated header".into()))?;
        let header: StateHeader =
            serde_json::from_slice(&bytes[16..header_end]).map_err(|e| bad(e.to_string()))?;
        let eval_mode = match header.eval_mode.as_str() {
            "coco" => EvalMode::Coco,
            "lvis" => EvalMode::Lvis,
            other => return Err(bad(format!("unknown eval mode {other:?}"))),
        };
        let params = Params::from(header.params);
        // No working evaluator has either axis empty, and refusing them keeps a
        // crafted state off every summary path that indexes them.
        if params.iou_thrs.is_empty() || params.area_ranges.is_empty() {
            return Err(bad("no IoU thresholds or no area ranges".into()));
        }
        let dims = (params.iou_thrs.len(), params.area_ranges.len());
        let cells = Cells::from_le_bytes(
            dims,
            (header.n_pairs, header.n_scores, header.n_words),
            &bytes[header_end..],
        )
        .map_err(bad)?;

        let mut se = StreamingEval::new(params, eval_mode, header.categories)?;
        let mut start: usize = 0;
        let mut prev: Option<u64> = None;
        for (id, n) in header.images {
            // Strictly ascending, as `to_bytes` writes them: a repeated id
            // would orphan its first run, and the map would reorder the rest.
            if prev.is_some_and(|p| p >= id) {
                return Err(bad(format!("image {id} is out of order")));
            }
            prev = Some(id);
            let end = start
                .checked_add(n)
                .filter(|&end| end <= cells.len())
                .ok_or_else(|| bad(format!("image {id} claims more cells than exist")))?;
            if (start..end).any(|p| cells.ids(p).0 != id) {
                return Err(bad(format!("image {id} does not match its cells")));
            }
            se.images.insert(id, (0, start..end));
            start = end;
        }
        if start != cells.len() {
            return Err(bad("cells belong to no image".into()));
        }
        se.batches.push(cells);
        Ok(se)
    }

    /// Every live image's pairs gathered into one arena, image-ascending: the
    /// call `finalize()` makes, so what it feeds `accumulate()` is unchanged.
    fn live_cells(&self) -> Cells {
        Cells::gather(
            &self.params,
            self.images
                .values()
                .map(|(batch, range)| (&self.batches[*batch], range.clone())),
        )
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
        let cells = self.live_cells();
        let StreamingEval {
            mut params,
            eval_mode,
            categories,
            images,
            ..
        } = self;

        if params.img_ids.is_empty() {
            params.img_ids = images.keys().copied().collect();
        }
        COCOeval::from_cells(categories, params, eval_mode, cells)
    }
}

/// Swap each area-less polygon or rectangle ground truth's segmentation for
/// the RLE it rasterizes to, so it is rasterized once per batch instead of
/// twice: the evaluator's area fill needs the mask for the area, and
/// `evaluate()`'s `SegmRles::prepare` needs it again for the IoUs. Both read
/// the RLE back through `ann_to_rle` unchanged — the same mask, a copy rather
/// than a rasterization. An annotation that cannot be rasterized (no image
/// size) keeps its polygon, for `check_inputs` to report.
fn keep_rasterized_masks_of_arealess(gt: &mut COCO) {
    use crate::types::Segmentation;
    let rasterized: Vec<(usize, crate::types::Rle)> = gt
        .dataset
        .annotations
        .iter()
        .enumerate()
        .filter(|(_, ann)| {
            ann.area.is_none()
                && matches!(
                    ann.segmentation,
                    Some(Segmentation::Polygon(_) | Segmentation::Rect(_))
                )
        })
        .filter_map(|(i, ann)| Some((i, gt.ann_to_rle(ann)?)))
        .collect();
    for (i, rle) in rasterized {
        gt.dataset.annotations[i].segmentation = Some(Segmentation::UncompressedRle {
            size: [rle.h, rle.w],
            counts: rle.counts,
        });
    }
}

const STATE_MAGIC: &[u8; 4] = b"HCSE";
/// Bumped whenever the bytes `to_bytes` writes change meaning, so older bytes
/// are refused rather than misread; `state_bytes_are_pinned` fails until it is.
/// 2: each detection's matched and ignore flags are stored together.
const STATE_VERSION: u32 = 2;

/// What precedes the arenas in a saved state. Floats are stored as their bit
/// patterns: JSON has no spelling for an infinite area bound, and a round trip
/// must not move a threshold by even an ulp.
#[derive(Serialize, Deserialize)]
struct StateHeader {
    eval_mode: String,
    params: ParamsState,
    categories: Vec<Category>,
    /// `(image id, pair count)`, image-ascending.
    images: Vec<(u64, usize)>,
    n_pairs: usize,
    n_scores: usize,
    n_words: usize,
}

/// [`Params`] with every float as its bit pattern. Every conversion and
/// comparison that follows destructures without `..`, so a field added to `Params`
/// fails to compile here instead of silently dropping out of a saved state or
/// the merge check.
#[derive(Serialize, Deserialize, PartialEq)]
struct ParamsState {
    iou_type: IouType,
    img_ids: Vec<u64>,
    cat_ids: Vec<u64>,
    iou_thrs: Vec<u64>,
    rec_thrs: Vec<u64>,
    max_dets: Vec<usize>,
    area_ranges: Vec<(String, [u64; 2])>,
    use_cats: bool,
    kpt_oks_sigmas: Vec<u64>,
    expand_dt: bool,
}

fn to_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn from_bits(v: &[u64]) -> Vec<f64> {
    v.iter().map(|&b| f64::from_bits(b)).collect()
}

impl From<&Params> for ParamsState {
    fn from(p: &Params) -> Self {
        let Params {
            iou_type,
            img_ids,
            cat_ids,
            iou_thrs,
            rec_thrs,
            max_dets,
            area_ranges,
            use_cats,
            kpt_oks_sigmas,
            expand_dt,
        } = p;
        ParamsState {
            iou_type: *iou_type,
            img_ids: img_ids.clone(),
            cat_ids: cat_ids.clone(),
            iou_thrs: to_bits(iou_thrs),
            rec_thrs: to_bits(rec_thrs),
            max_dets: max_dets.clone(),
            area_ranges: area_ranges
                .iter()
                .map(|AreaRange { label, range }| {
                    (label.clone(), super::accumulate::area_key(*range))
                })
                .collect(),
            use_cats: *use_cats,
            kpt_oks_sigmas: to_bits(kpt_oks_sigmas),
            expand_dt: *expand_dt,
        }
    }
}

impl From<ParamsState> for Params {
    fn from(s: ParamsState) -> Self {
        let ParamsState {
            iou_type,
            img_ids,
            cat_ids,
            iou_thrs,
            rec_thrs,
            max_dets,
            area_ranges,
            use_cats,
            kpt_oks_sigmas,
            expand_dt,
        } = s;
        Params {
            iou_type,
            img_ids,
            cat_ids,
            iou_thrs: from_bits(&iou_thrs),
            rec_thrs: from_bits(&rec_thrs),
            max_dets,
            area_ranges: area_ranges
                .into_iter()
                .map(|(label, [lo, hi])| AreaRange {
                    label,
                    range: [f64::from_bits(lo), f64::from_bits(hi)],
                })
                .collect(),
            use_cats,
            kpt_oks_sigmas: from_bits(&kpt_oks_sigmas),
            expand_dt,
        }
    }
}

fn mismatch(what: &str) -> crate::error::Error {
    crate::error::Error::Other(format!(
        "cannot merge StreamingEvals built with different {what}: every rank must construct \
         its evaluator from the same categories and params"
    ))
}

/// What [`StreamingEval::merge`] compares of two category lists: the fields
/// a finalized evaluator reads, sorted by id, since the K axis is.
fn category_key(categories: &COCO) -> Vec<(u64, &str, Option<&str>)> {
    let mut key: Vec<_> = categories
        .dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str(), c.frequency.as_deref()))
        .collect();
    key.sort_unstable();
    key
}

/// The first [`Params`] field on which `a` and `b` differ, if any, with
/// floats compared bit for bit.
fn params_mismatch(a: &Params, b: &Params) -> Option<&'static str> {
    let (a, b) = (ParamsState::from(a), ParamsState::from(b));
    let ParamsState {
        iou_type,
        img_ids,
        cat_ids,
        iou_thrs,
        rec_thrs,
        max_dets,
        area_ranges,
        use_cats,
        kpt_oks_sigmas,
        expand_dt,
    } = &a;
    let checks = [
        ("iou_type", *iou_type == b.iou_type),
        ("img_ids", *img_ids == b.img_ids),
        ("cat_ids", *cat_ids == b.cat_ids),
        ("iou_thrs", *iou_thrs == b.iou_thrs),
        ("rec_thrs", *rec_thrs == b.rec_thrs),
        ("max_dets", *max_dets == b.max_dets),
        ("area_ranges", *area_ranges == b.area_ranges),
        ("use_cats", *use_cats == b.use_cats),
        ("kpt_oks_sigmas", *kpt_oks_sigmas == b.kpt_oks_sigmas),
        ("expand_dt", *expand_dt == b.expand_dt),
    ];
    debug_assert_eq!(checks.iter().all(|(_, same)| *same), a == b);
    checks.into_iter().find(|(_, same)| !same).map(|(f, _)| f)
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
    /// The exact bytes `to_bytes` writes for the fixture, as a hash. A
    /// layout change that keeps every length — as moving the flags to
    /// detection-major did — would otherwise load old bytes as wrong flags.
    /// The hash also moves when the fixture's results or the header's
    /// serialization change; only a change to what the bytes mean needs
    /// `STATE_VERSION` bumped.
    #[test]
    fn state_bytes_are_pinned() {
        let (gt, dt) = fixture();
        let bytes = shard(&gt, &dt.annotations, &[1, 2, 3]).to_bytes();
        let fnv1a = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
        assert_eq!(bytes[4..8], STATE_VERSION.to_le_bytes());
        assert_eq!(
            fnv1a, 0xb111_1c56_d118_cce4,
            "to_bytes changed: if the layout or meaning changed, bump STATE_VERSION; \
             then pin {fnv1a:#018x}"
        );
    }

    /// An unfinalized evaluator over the images `ids` of the fixture.
    fn shard(gt: &Dataset, dt_anns: &[Annotation], ids: &[u64]) -> StreamingEval {
        let mut se = StreamingEval::new(
            Params::new(IouType::Bbox),
            EvalMode::Coco,
            gt.categories.clone(),
        )
        .expect("mode is supported");
        let keep: HashSet<u64> = ids.iter().copied().collect();
        let of = |anns: &[Annotation]| -> Vec<Annotation> {
            anns.iter()
                .filter(|a| keep.contains(&a.image_id))
                .cloned()
                .collect()
        };
        se.update(
            ids.iter().map(|&id| image(id)).collect(),
            of(&gt.annotations),
            of(dt_anns),
        )
        .expect("scores are finite");
        se
    }

    /// Two ranks that each saw some images, merged, give the cells one stream
    /// over every image gives — in either merge order.
    #[test]
    fn merged_disjoint_shards_equal_one_stream() {
        let (gt, dt) = fixture();
        let expected = format!("{:?}", stream(&gt, &dt.annotations, &[&[1, 2, 3]]).cells);
        for (first, second) in [(&[1u64][..], &[2, 3][..]), (&[2, 3][..], &[1][..])] {
            let mut a = shard(&gt, &dt.annotations, first);
            a.merge(shard(&gt, &dt.annotations, second))
                .expect("shards are compatible");
            assert_eq!(expected, format!("{:?}", a.finalize().cells));
        }
    }

    /// An image on both ranks keeps `other`'s result, the rule `update`
    /// documents for an image seen again.
    #[test]
    fn merge_lets_the_other_evaluator_win_an_overlapping_image() {
        let (gt, dt) = fixture();
        let mut moved = dt.annotations.clone();
        for d in &mut moved {
            d.bbox = Some([500.0, 500.0, 5.0, 5.0]);
        }
        let mut a = shard(&gt, &dt.annotations, &[1, 2]);
        a.merge(shard(&gt, &moved, &[2, 3])).expect("compatible");
        let mixed: Vec<Annotation> = dt
            .annotations
            .iter()
            .filter(|d| d.image_id == 1)
            .chain(moved.iter().filter(|d| d.image_id != 1))
            .cloned()
            .collect();
        let expected = stream(&gt, &mixed, &[&[1, 2, 3]]);
        assert_eq!(
            format!("{:?}", expected.cells),
            format!("{:?}", a.finalize().cells)
        );
    }

    #[test]
    fn merge_rejects_mismatched_params_naming_the_field() {
        let (gt, dt) = fixture();
        let mut a = shard(&gt, &dt.annotations, &[1]);
        let mut other_params = Params::new(IouType::Bbox);
        other_params.max_dets = vec![1, 10];
        let mut b = StreamingEval::new(other_params, EvalMode::Coco, gt.categories.clone())
            .expect("mode is supported");
        b.update(vec![image(2)], Vec::new(), Vec::new())
            .expect("nothing to reject");
        let err = a.merge(b).expect_err("max_dets differ").to_string();
        assert!(err.contains("max_dets"), "{err}");
        assert_eq!(
            a.finalize().cells.len(),
            shard(&gt, &dt.annotations, &[1]).finalize().cells.len()
        );
    }

    #[test]
    fn merge_rejects_different_categories_and_modes() {
        let (gt, dt) = fixture();
        let mut a = shard(&gt, &dt.annotations, &[1]);
        let b = StreamingEval::new(
            Params::new(IouType::Bbox),
            EvalMode::Coco,
            vec![category(1), category(2)],
        )
        .expect("mode is supported");
        let err = a.merge(b).expect_err("category lists differ").to_string();
        assert!(err.contains("categories"), "{err}");
        let lvis = StreamingEval::new(
            Params::new(IouType::Bbox),
            EvalMode::Lvis,
            gt.categories.clone(),
        )
        .expect("mode is supported");
        let err = a.merge(lvis).expect_err("modes differ").to_string();
        assert!(err.contains("eval_mode"), "{err}");
    }
    /// A saved state restores to an evaluator that finalizes to the same
    /// cells and params, however many calls and replaced images built it.
    #[test]
    fn bytes_round_trip_finalizes_identically() {
        let (gt, dt) = fixture();
        let mut se = shard(&gt, &dt.annotations, &[1, 2]);
        se.merge(shard(&gt, &dt.annotations, &[2, 3]))
            .expect("compatible");
        let restored = StreamingEval::from_bytes(&se.to_bytes()).expect("own output");
        let (a, b) = (se.finalize(), restored.finalize());
        assert_eq!(format!("{:?}", a.cells), format!("{:?}", b.cells));
        assert_eq!(a.params.img_ids, b.params.img_ids);
        assert_eq!(a.params.cat_ids, b.params.cat_ids);
    }

    /// The restored evaluator is a working one: it takes more images and
    /// merges, and an infinite area bound survives the header.
    #[test]
    fn restored_evaluator_keeps_streaming_and_custom_params() {
        let (gt, dt) = fixture();
        let mut params = Params::new(IouType::Bbox);
        params.area_ranges.push(AreaRange {
            label: "huge".into(),
            range: [1e5, f64::INFINITY],
        });
        let mut se = StreamingEval::new(params, EvalMode::Coco, gt.categories.clone())
            .expect("mode is supported");
        se.update(
            vec![image(1)],
            gt.annotations.clone(),
            dt.annotations.clone(),
        )
        .expect("batch is valid");
        let mut restored = StreamingEval::from_bytes(&se.to_bytes()).expect("own output");
        restored
            .update(vec![image(9)], Vec::new(), Vec::new())
            .expect("accepts more images");
        let ev = restored.finalize();
        assert!(ev.params.area_ranges[4].range[1].is_infinite());
        assert_eq!(ev.params.img_ids, vec![1, 9]);
        assert!(
            se.merge(StreamingEval::from_bytes(&se.to_bytes()).expect("own output"))
                .is_ok()
        );
    }

    /// Run a restored evaluator through everything a caller runs next, so an
    /// out-of-bounds read that `from_bytes` let through would panic here.
    fn finalize_to_summary(se: StreamingEval) {
        let mut ev = se.finalize();
        ev.accumulate();
        let _ = ev.summarize_lines();
    }

    /// Hostile or damaged bytes are an error or a harmless state, never a
    /// panic: every prefix and every single-bit flip of a real state, each
    /// accepted one taken through `accumulate()` and the summary.
    #[test]
    fn corrupt_bytes_never_panic() {
        let (gt, dt) = fixture();
        let bytes = shard(&gt, &dt.annotations, &[1, 2, 3]).to_bytes();
        for n in 0..bytes.len() {
            assert!(
                StreamingEval::from_bytes(&bytes[..n]).is_err(),
                "prefix {n}"
            );
        }
        let mut accepted = 0;
        for i in 0..bytes.len() {
            for bit in 0..8 {
                let mut damaged = bytes.clone();
                damaged[i] ^= 1 << bit;
                if let Ok(se) = StreamingEval::from_bytes(&damaged) {
                    accepted += 1;
                    finalize_to_summary(se);
                }
            }
        }
        // Flips inside a score or an id are accepted, so the loop above did
        // reach `accumulate()`.
        assert!(accepted > 0);
        let mut wrong = bytes.clone();
        wrong[4] = 9;
        let err = StreamingEval::from_bytes(&wrong)
            .err()
            .expect("version 9")
            .to_string();
        assert!(err.contains("version"), "{err}");
    }

    /// `bytes` with its JSON header passed through `edit`, re-emitted with
    /// the new header length and the same cells.
    fn with_header(bytes: &[u8], edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")) as usize;
        let mut header: serde_json::Value =
            serde_json::from_slice(&bytes[16..16 + len]).expect("own header");
        edit(&mut header);
        let header = serde_json::to_vec(&header).expect("plain JSON");
        let mut out = bytes[..8].to_vec();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&bytes[16 + len..]);
        out
    }

    /// Counts in a well-formed header that overflow every size computation,
    /// or point past the cells, are errors and not panics — in debug, where
    /// the arithmetic would trap, and in release, where it would wrap.
    #[test]
    fn crafted_headers_are_errors_not_panics() {
        let (gt, dt) = fixture();
        let bytes = shard(&gt, &dt.annotations, &[1, 2, 3]).to_bytes();
        let cells_len = {
            let len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")) as usize;
            bytes.len() - 16 - len
        };
        let max = serde_json::json!(u64::MAX);
        // A header for no cells at all; the bytes after it must be the one
        // zeroed sentinel header.
        let mut sentinel_only = bytes[..bytes.len() - cells_len].to_vec();
        sentinel_only.extend_from_slice(&[0; 24]);
        let empty_arena = |h: &mut serde_json::Value| {
            h["n_pairs"] = 0.into();
            h["n_scores"] = 0.into();
            h["n_words"] = 0.into();
            h["images"] = serde_json::json!([]);
        };
        let cases: Vec<(&str, Vec<u8>)> = vec![
            (
                "n_pairs wraps to zero with nothing else",
                with_header(&bytes[..bytes.len() - cells_len], |h| {
                    h["n_pairs"] = max.clone();
                    h["n_scores"] = 0.into();
                    h["n_words"] = 0.into();
                    h["params"]["area_ranges"] = serde_json::json!([]);
                    h["images"] = serde_json::json!([]);
                }),
            ),
            (
                "n_pairs at the maximum",
                with_header(&bytes, |h| h["n_pairs"] = max.clone()),
            ),
            (
                "n_scores at the maximum",
                with_header(&bytes, |h| h["n_scores"] = max.clone()),
            ),
            (
                "n_words at the maximum",
                with_header(&bytes, |h| h["n_words"] = max.clone()),
            ),
            (
                "an image count at the maximum",
                with_header(&bytes, |h| h["images"][1][1] = max.clone()),
            ),
            (
                "image counts that wrap back to the total",
                with_header(&bytes, |h| {
                    h["images"][1][1] = max.clone();
                    h["images"][2][1] = 2.into();
                }),
            ),
            (
                "a repeated image id",
                with_header(&bytes, |h| {
                    let first = h["images"][0][0].clone();
                    h["images"][1][0] = first;
                }),
            ),
            (
                "no IoU thresholds",
                with_header(&bytes, |h| h["params"]["iou_thrs"] = serde_json::json!([])),
            ),
            (
                "no area ranges",
                with_header(&bytes, |h| {
                    h["params"]["area_ranges"] = serde_json::json!([]);
                }),
            ),
            (
                "no IoU thresholds over an empty arena",
                with_header(&sentinel_only, |h| {
                    empty_arena(h);
                    h["params"]["iou_thrs"] = serde_json::json!([]);
                }),
            ),
            (
                "no area ranges over an empty arena",
                with_header(&sentinel_only, |h| {
                    empty_arena(h);
                    h["params"]["area_ranges"] = serde_json::json!([]);
                }),
            ),
            (
                "an unknown eval mode",
                with_header(&bytes, |h| h["eval_mode"] = "oid".into()),
            ),
        ];
        for (what, crafted) in cases {
            match StreamingEval::from_bytes(&crafted) {
                Err(e) => assert!(
                    e.to_string().contains("invalid StreamingEval state"),
                    "{what}: {e}"
                ),
                Ok(_) => panic!("{what}: accepted"),
            }
        }
        // The rewrite itself is faithful: an unedited header round-trips.
        let same = with_header(&bytes, |_| {});
        finalize_to_summary(StreamingEval::from_bytes(&same).expect("unedited"));
    }

    /// LVIS mode survives the bytes: the mode, the frequency buckets, and the
    /// image-level label lists already folded into the cells.
    #[test]
    fn lvis_state_round_trips() {
        let (gt, dt) = fixture();
        let categories: Vec<Category> = gt
            .categories
            .iter()
            .zip(["r", "c", "f"])
            .map(|(c, f)| Category {
                frequency: Some(f.into()),
                ..c.clone()
            })
            .collect();
        let mut se = StreamingEval::new(
            EvalMode::Lvis.default_params(IouType::Bbox),
            EvalMode::Lvis,
            categories,
        )
        .expect("mode is supported");
        let mut img1 = image(1);
        img1.neg_category_ids = vec![2];
        se.update(
            vec![img1, image(2), image(3)],
            gt.annotations.clone(),
            dt.annotations.clone(),
        )
        .expect("batch is valid");
        let restored = StreamingEval::from_bytes(&se.to_bytes()).expect("own output");
        assert_eq!(restored.eval_mode, EvalMode::Lvis);
        let (mut a, mut b) = (se.finalize(), restored.finalize());
        assert_eq!(format!("{:?}", a.cells), format!("{:?}", b.cells));
        assert_eq!(a.summarize_lines(), b.summarize_lines());
        assert_eq!(a.stats, b.stats);
    }

    /// The list order a rank builds its categories in changes no number, so
    /// it does not block a merge; a different LVIS frequency does.
    #[test]
    fn merge_compares_categories_by_content_not_order() {
        let (gt, dt) = fixture();
        let mut a = shard(&gt, &dt.annotations, &[1]);
        let mut reversed = gt.categories.clone();
        reversed.reverse();
        let b = StreamingEval::new(Params::new(IouType::Bbox), EvalMode::Coco, reversed)
            .expect("mode is supported");
        a.merge(b).expect("same categories, other order");
        let mut rare = gt.categories.clone();
        rare[0].frequency = Some("r".into());
        let c = StreamingEval::new(Params::new(IouType::Bbox), EvalMode::Coco, rare)
            .expect("mode is supported");
        let err = a.merge(c).expect_err("frequency differs").to_string();
        assert!(err.contains("categories"), "{err}");
    }
}
