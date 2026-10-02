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
    /// The categories and nothing else: what `new()` derives the K axis from
    /// and what the finalized evaluator reads category names off.
    categories: COCO,
    /// The list `new()` received, kept so two evaluators can be compared and
    /// a saved state can rebuild this one.
    category_list: Vec<Category>,
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

        let category_list = categories.clone();
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
            category_list,
            batches: Vec::new(),
            images: BTreeMap::new(),
        })
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
    /// A detection with a NaN score is an error, as it is in `load_res`.
    pub fn update(
        &mut self,
        images: Vec<Image>,
        gt_anns: Vec<Annotation>,
        dt_anns: Vec<Annotation>,
    ) -> crate::error::Result<()> {
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
    /// categories, and params — every field, `img_ids` included. A mismatch is
    /// an error naming the first field that differs, and `self` is as it was.
    pub fn merge(&mut self, other: StreamingEval) -> crate::error::Result<()> {
        if self.eval_mode != other.eval_mode {
            return Err(mismatch("eval_mode"));
        }
        let ours = self.category_list.iter().map(|c| (c.id, &c.name));
        if !ours.eq(other.category_list.iter().map(|c| (c.id, &c.name))) {
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
            categories: self.category_list.clone(),
            images,
            n_pairs: counts.0,
            n_scores: counts.1,
            n_words: counts.2,
        })
        .expect("a header of plain data serializes");
        let mut out = Vec::with_capacity(16 + header.len() + counts.1 * 8 + counts.2 * 8);
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
        let dims = (params.iou_thrs.len(), params.area_ranges.len());
        let cells = Cells::from_le_bytes(
            dims,
            (header.n_pairs, header.n_scores, header.n_words),
            &bytes[header_end..],
        )
        .map_err(bad)?;

        let mut se = StreamingEval::new(params, eval_mode, header.categories)?;
        let mut start = 0;
        for (id, n) in header.images {
            let end = start + n;
            if end > cells.len() || (start..end).any(|p| cells.ids(p).0 != id) {
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
        let StreamingEval {
            mut params,
            eval_mode,
            categories,
            batches,
            images,
            category_list: _,
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

const STATE_MAGIC: &[u8; 4] = b"HCSE";
const STATE_VERSION: u32 = 1;

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

#[derive(Serialize, Deserialize)]
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
        ParamsState {
            iou_type: p.iou_type,
            img_ids: p.img_ids.clone(),
            cat_ids: p.cat_ids.clone(),
            iou_thrs: to_bits(&p.iou_thrs),
            rec_thrs: to_bits(&p.rec_thrs),
            max_dets: p.max_dets.clone(),
            area_ranges: p
                .area_ranges
                .iter()
                .map(|r| {
                    (
                        r.label.clone(),
                        [r.range[0].to_bits(), r.range[1].to_bits()],
                    )
                })
                .collect(),
            use_cats: p.use_cats,
            kpt_oks_sigmas: to_bits(&p.kpt_oks_sigmas),
            expand_dt: p.expand_dt,
        }
    }
}

impl From<ParamsState> for Params {
    fn from(s: ParamsState) -> Self {
        let mut p = Params::new(s.iou_type);
        p.img_ids = s.img_ids;
        p.cat_ids = s.cat_ids;
        p.iou_thrs = from_bits(&s.iou_thrs);
        p.rec_thrs = from_bits(&s.rec_thrs);
        p.max_dets = s.max_dets;
        p.area_ranges = s
            .area_ranges
            .into_iter()
            .map(|(label, [lo, hi])| AreaRange {
                label,
                range: [f64::from_bits(lo), f64::from_bits(hi)],
            })
            .collect();
        p.use_cats = s.use_cats;
        p.kpt_oks_sigmas = from_bits(&s.kpt_oks_sigmas);
        p.expand_dt = s.expand_dt;
        p
    }
}

fn mismatch(what: &str) -> crate::error::Error {
    crate::error::Error::Other(format!(
        "cannot merge StreamingEvals built with different {what}: every rank must construct \
         its evaluator from the same categories and params"
    ))
}

/// The first [`Params`] field on which `a` and `b` differ, if any.
fn params_mismatch(a: &Params, b: &Params) -> Option<&'static str> {
    let areas = |p: &Params| -> Vec<(String, [f64; 2])> {
        p.area_ranges
            .iter()
            .map(|r| (r.label.clone(), r.range))
            .collect()
    };
    let checks = [
        ("iou_type", a.iou_type == b.iou_type),
        ("img_ids", a.img_ids == b.img_ids),
        ("cat_ids", a.cat_ids == b.cat_ids),
        ("iou_thrs", a.iou_thrs == b.iou_thrs),
        ("rec_thrs", a.rec_thrs == b.rec_thrs),
        ("max_dets", a.max_dets == b.max_dets),
        ("area_ranges", areas(a) == areas(b)),
        ("use_cats", a.use_cats == b.use_cats),
        ("kpt_oks_sigmas", a.kpt_oks_sigmas == b.kpt_oks_sigmas),
        ("expand_dt", a.expand_dt == b.expand_dt),
    ];
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

    /// Hostile or damaged bytes are an error or a harmless state, never a
    /// panic: every prefix and every single-byte corruption of a real state.
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
        for i in 0..bytes.len() {
            let mut damaged = bytes.clone();
            damaged[i] ^= 0xFF;
            if let Ok(se) = StreamingEval::from_bytes(&damaged) {
                let _ = se.finalize();
            }
        }
        let mut wrong = bytes.clone();
        wrong[4] = 9;
        let err = StreamingEval::from_bytes(&wrong)
            .err()
            .expect("version 9")
            .to_string();
        assert!(err.contains("version"), "{err}");
    }
}
