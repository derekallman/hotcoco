//! The detection metric family.
//!
//! [`COCOeval`] is the driver: a faithful port of `pycocotools/cocoeval.py`'s
//! `evaluate` → `accumulate` → `summarize` lifecycle for bbox, segm, keypoint
//! and oriented-box geometry, plus the two protocol variants that share it —
//! LVIS federated evaluation ([`COCOeval::new_lvis`]) and Open Images
//! ([`COCOeval::new_oid`]).
//!
//! Layered on the same evaluated cells are the analysis methods, which are
//! adapters rather than metric implementations: they decide which detections
//! count, marshal them into flat arrays, and call
//! [`crate::metrics`]. TIDE's error taxonomy ([`TideErrors`]) and per-image
//! diagnostics ([`ImageDiagnostics`]) are the exceptions that stay here, being
//! genuinely detection-shaped.

mod accumulate;
mod calibration;
mod catalog;
mod compare;
mod confusion;
mod diagnostics;
mod evaluate;
pub mod expand;
pub mod hierarchy;
mod iou;
mod matching;
mod mode;
mod report;
mod results;
pub mod slice;
mod streaming;
mod summarize;
mod tide;

pub use accumulate::{AccumulatedEval, EvalShape};
pub use calibration::CalibrationResult;
pub use catalog::MetricDef;
pub use compare::{CategoryDelta, CompareOpts, ComparisonResult, compare};
pub use confusion::ConfusionMatrix;
pub use diagnostics::{
    AnnotationIndex, DtStatus, ErrorProfile, GtStatus, ImageDiagnostics, ImageSummary, LabelError,
    LabelErrorType,
};
pub use matching::EvalImg;
pub use mode::{EvalMode, FreqGroup};
pub use results::{EvalParams, EvalResults};
pub use slice::{SliceResult, SlicedResults};
pub use streaming::StreamingEval;
pub use tide::TideErrors;

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::coco::COCO;
use crate::detection::hierarchy::Hierarchy;
use crate::params::{IouType, Params};

/// How many of `n` items each parallel run takes, at least one.
fn run_len(n: usize) -> usize {
    n.div_ceil(crate::RUNS_PER_THREAD * rayon::current_num_threads())
        .max(1)
}

/// COCO evaluation engine.
///
/// Computes AP and AR metrics for bbox, segmentation, and keypoint predictions.
/// Also supports LVIS federated evaluation via [`COCOeval::new_lvis`].
///
/// The standard workflow is three steps:
///
/// ```no_run
/// # use hotcoco::{COCO, COCOeval, params::IouType};
/// # fn main() -> hotcoco::error::Result<()> {
/// # let coco_gt = COCO::new(std::path::Path::new("gt.json"))?;
/// # let coco_dt = coco_gt.load_res(std::path::Path::new("dt.json"))?;
/// let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
/// ev.evaluate();   // per-image IoU matching
/// ev.accumulate(); // aggregate into precision/recall curves
/// ev.summarize();  // print + store the summary metrics in ev.stats
/// # Ok(())
/// # }
/// ```
///
/// For LVIS, use [`run`](COCOeval::run) as a convenience:
///
/// ```no_run
/// # use hotcoco::{COCO, COCOeval, params::IouType};
/// # fn main() -> hotcoco::error::Result<()> {
/// # let coco_gt = COCO::new(std::path::Path::new("gt.json"))?;
/// # let coco_dt = coco_gt.load_res(std::path::Path::new("dt.json"))?;
/// let mut ev = COCOeval::new_lvis(coco_gt, coco_dt, IouType::Segm);
/// ev.run();
/// let results = ev.get_results(None, false); // BTreeMap<metric_name, f64>
/// # Ok(())
/// # }
/// ```
pub struct COCOeval {
    coco_gt: Arc<COCO>,
    coco_dt: Arc<COCO>,
    pub params: Params,
    /// The full per-image records, built on first access — see
    /// [`eval_imgs`](Self::eval_imgs). Reset by every `evaluate()`.
    eval_imgs: std::sync::OnceLock<Vec<Option<EvalImg>>>,
    /// The `"all"` area range's records alone, for the in-crate analyses that
    /// read only that range (see [`default_cells`](Self::default_cells)) —
    /// a quarter of [`eval_imgs`](Self::eval_imgs) on COCO's four ranges.
    default_eval_imgs: std::sync::OnceLock<Vec<Option<EvalImg>>>,
    /// Every gathered (image, category) pair, in the order `evaluate()` visits
    /// them — what `accumulate()` reads. Empty until `evaluate()` runs.
    cells: matching::Cells,
    /// What the last `evaluate()` saw; `None` until it runs.
    eval_inputs: Option<evaluate::EvalInputs>,
    ious: HashMap<(u64, u64), matching::IouMatrix>,
    /// Per-annotation RLEs for segm runs, rebuilt by each `evaluate()` (like
    /// `ious`) and `None` for every other geometry. See [`iou::SegmRles`].
    segm_rles: Option<iou::SegmRles>,
    pub(crate) eval: Option<AccumulatedEval>,
    pub(crate) stats: Option<Vec<f64>>,
    /// Evaluation mode (COCO, LVIS, or OpenImages).
    pub eval_mode: EvalMode,
    /// Open Images: category hierarchy for GT/DT expansion.
    pub hierarchy: Option<Hierarchy>,
}

impl COCOeval {
    /// The one struct literal behind all three public constructors. What they
    /// differ in is a parameter here; everything else is the same empty
    /// pre-`evaluate()` state, so a field added later is initialized once.
    ///
    /// LVIS replaces `coco_dt` here with its per-image-capped copy — see
    /// [`new_lvis`](Self::new_lvis) — so every constructor path, streaming
    /// batches included, applies the cap the way lvis-api's `LVISResults` does.
    /// A set [`COCO::cap_detections_per_image`] already capped is kept as is.
    fn with_mode(
        coco_gt: impl Into<Arc<COCO>>,
        coco_dt: impl Into<Arc<COCO>>,
        params: Params,
        eval_mode: EvalMode,
        hierarchy: Option<Hierarchy>,
    ) -> Self {
        let mut coco_dt: Arc<COCO> = coco_dt.into();
        if eval_mode == EvalMode::Lvis && !coco_dt.is_per_image_capped() {
            if let Some(capped) = coco_dt.capped_copy(params.max_det()) {
                coco_dt = Arc::new(capped);
            }
        }
        COCOeval {
            coco_gt: coco_gt.into(),
            coco_dt,
            params,
            eval_imgs: std::sync::OnceLock::new(),
            default_eval_imgs: std::sync::OnceLock::new(),
            cells: matching::Cells::default(),
            eval_inputs: None,
            ious: HashMap::new(),
            segm_rles: None,
            eval: None,
            stats: None,
            eval_mode,
            hierarchy,
        }
    }

    /// An evaluator in the state `evaluate()` leaves, from cells matched
    /// elsewhere: what [`StreamingEval::finalize`] returns. `coco_gt` carries
    /// what the summary surfaces read off it — category names — and no
    /// annotations, so the full per-image records cannot be rebuilt; the
    /// `eval_imgs` cache is seeded empty to say so, and `default_cells()`
    /// reads that same cache.
    fn from_cells(
        coco_gt: COCO,
        params: Params,
        eval_mode: EvalMode,
        cells: matching::Cells,
    ) -> Self {
        // Every pair `evaluate()` collects has a side, so the cells hold them all.
        let sparse_pairs = (0..cells.len()).map(|p| cells.ids(p)).collect();
        let inputs = evaluate::EvalInputs {
            params: params.clone(),
            sparse_pairs,
            // Read only by the rebuild, which the seeded cache makes unreachable.
            not_exhaustive: HashMap::new(),
        };
        let mut ev = Self::with_mode(
            coco_gt,
            COCO::from_dataset(crate::types::Dataset::default()),
            params,
            eval_mode,
            None,
        );
        ev.cells = cells;
        ev.eval_inputs = Some(inputs);
        ev.eval_imgs = std::sync::OnceLock::from(Vec::new());
        ev
    }

    /// Create a new COCOeval from ground truth and detection COCO objects.
    pub fn new(
        coco_gt: impl Into<Arc<COCO>>,
        coco_dt: impl Into<Arc<COCO>>,
        iou_type: IouType,
    ) -> Self {
        Self::with_mode(
            coco_gt,
            coco_dt,
            EvalMode::Coco.default_params(iou_type),
            EvalMode::Coco,
            None,
        )
    }

    /// The ground-truth dataset this evaluator reads. The evaluator never
    /// writes through it; Open Images [`evaluate`](Self::evaluate) replaces it
    /// with an expanded copy.
    pub fn coco_gt(&self) -> &Arc<COCO> {
        &self.coco_gt
    }

    /// The detection dataset this evaluator reads — see [`coco_gt`](Self::coco_gt).
    /// Under LVIS, the per-image-capped copy [`new_lvis`](Self::new_lvis) made.
    pub fn coco_dt(&self) -> &Arc<COCO> {
        &self.coco_dt
    }

    /// Per-image evaluation results (sparse — indexed by image position).
    ///
    /// Empty until [`evaluate`](Self::evaluate) runs. Built on first access and
    /// cached: `evaluate()` itself keeps only the lean per-cell bits that
    /// `accumulate()` reads, about 20 bytes per detection, and these full
    /// records — every id, both sides of the match, about 460 bytes per
    /// detection — are materialized from the same inputs when something asks
    /// for them. They describe what `evaluate()` produced, whatever `params`
    /// has been set to since.
    ///
    /// Also empty for an evaluator [`StreamingEval::finalize`] built: it holds
    /// the lean cells and no dataset to rebuild the records from.
    pub fn eval_imgs(&self) -> &[Option<EvalImg>] {
        match &self.eval_inputs {
            None => &[],
            Some(inputs) => self.eval_imgs.get_or_init(|| {
                let every_range: Vec<usize> = (0..inputs.params.area_ranges.len()).collect();
                self.evaluate_pairs_full(inputs, &every_range)
            }),
        }
    }

    /// Whether [`evaluate`](Self::evaluate) has run.
    pub fn evaluated(&self) -> bool {
        self.eval_inputs.is_some()
    }

    /// Accumulated precision/recall curves (set after `accumulate()`).
    //
    // The pyo3 binding caches the dict built from this value and invalidates
    // it in `evaluate()`/`accumulate()`/`run()`. A new mutation path that
    // replaces `self.eval` must also invalidate that cache.
    pub fn accumulated(&self) -> Option<&AccumulatedEval> {
        self.eval.as_ref()
    }

    /// Summary statistics (set after `summarize()`).
    pub fn stats(&self) -> Option<&[f64]> {
        self.stats.as_deref()
    }

    /// Cached similarity matrix for one (image, category) cell, if `evaluate()`
    /// computed one.
    ///
    /// # Deliberately one cell at a time
    ///
    /// `self.ious` is a **whole-dataset** similarity cache. Exposing it as one
    /// would foreclose the memory lever the tracking family depends on: HOTA's
    /// second pass must be free to *recompute* similarity rather than retain it,
    /// because at MOT20 scale retention costs hundreds of megabytes per sequence
    /// per thread.
    ///
    /// So this accessor hands out one cell, never the map. The detection driver may
    /// cache as much as it likes; nothing outside it may learn that a whole-dataset
    /// cache exists. **Keep this driver-private** — it must not gain a `pub` variant,
    /// and it must not return `&HashMap<..>`. `pub(in crate::detection)`, not
    /// `pub(super)`: the visibility is the enforcement. `tests/architecture.rs`
    /// separately bans direct `.ious` access outside this module.
    pub(in crate::detection) fn cell_ious(
        &self,
        img_id: u64,
        cat_id: u64,
    ) -> Option<&matching::IouMatrix> {
        self.ious.get(&(img_id, cat_id))
    }

    /// The evaluated cells every whole-dataset analysis reads: `area = "all"` at
    /// the default per-image detection cap.
    ///
    /// TIDE, calibration and per-image diagnostics all want exactly this subset
    /// of `eval_imgs`. The `max_det` leg is inert today — `evaluate()` stamps one
    /// cap on every cell, taken from [`Params::max_det`](crate::Params::max_det)
    /// — but it is what keeps a future second cap per cell from making these
    /// analyses double-count every detection.
    ///
    /// Both legs come from `params`, so an evaluator re-configured after
    /// `evaluate()` yields nothing rather than a partial mixture.
    pub(in crate::detection) fn default_cells(&self) -> impl Iterator<Item = &EvalImg> {
        let area_rng = self.params.all_area_range();
        let max_det = self.params.max_det();
        // The full set if something already built it; otherwise only the
        // `"all"` range, which is all these analyses read.
        let cells: &[Option<EvalImg>] = match (&self.eval_inputs, self.eval_imgs.get()) {
            (None, _) => &[],
            (Some(_), Some(all)) => all,
            (Some(inputs), None) => self
                .default_eval_imgs
                .get_or_init(|| self.evaluate_pairs_full(inputs, &[inputs.params.all_area_idx()])),
        };
        cells
            .iter()
            .flatten()
            .filter(move |e| e.area_rng == area_rng && e.max_det == max_det)
    }

    /// The image and category ids this evaluation covers, without mutating anything.
    ///
    /// User-set `params` filters win; otherwise the ids come from `COCO`'s
    /// unfiltered getters, which return them sorted — the order the whole
    /// evaluation is keyed on. Borrowed when `params` already holds them, so the
    /// common path allocates nothing.
    ///
    /// **Non-mutating** on purpose, because its two callers differ: `evaluate()`
    /// writes the answer back into `params`, while `confusion_matrix()` is a
    /// `&self` method that must cover the same ids without a prior `evaluate()`
    /// and without touching state.
    pub(in crate::detection) fn resolved_ids(&self) -> (Cow<'_, [u64]>, Cow<'_, [u64]>) {
        let img_ids = if self.params.img_ids.is_empty() {
            Cow::Owned(self.coco_gt.get_img_ids(&[], &[]))
        } else {
            Cow::Borrowed(self.params.img_ids.as_slice())
        };
        let cat_ids = if self.params.cat_ids.is_empty() {
            Cow::Owned(self.coco_gt.get_cat_ids(&[], &[], &[]))
        } else {
            Cow::Borrowed(self.params.cat_ids.as_slice())
        };
        (img_ids, cat_ids)
    }

    /// Create a new COCOeval configured for LVIS federated evaluation.
    ///
    /// LVIS uses federated annotation — each image is only exhaustively labeled
    /// for a subset of categories. This constructor sets `max_dets=300` and
    /// enables federated filtering so unmatched detections on unlabeled or
    /// unchecked categories are not penalized as false positives.
    ///
    /// # LVIS replaces `coco_dt`
    ///
    /// The 300 cap applies per image across every category, not per
    /// `(image, category)` cell: the evaluator holds the detections you pass
    /// capped to each image's 300 highest-scoring, as lvis-api's
    /// `LVISResults` does at load time, and [`coco_dt`](Self::coco_dt) returns
    /// that capped copy. Your own handle is untouched. The cap is the
    /// construction-time `params.max_det()`; editing `params.max_dets`
    /// afterwards changes the per-cell cap, not this one — lvis-api's
    /// `LVISResults(max_dets=)` and `LVISEval` params are likewise separate.
    ///
    /// Detections that [`COCO::cap_detections_per_image`] already capped —
    /// at any value, or `None` for none — are used as is, the way lvis-api's
    /// `LVISEval` takes an `LVISResults` unchanged. That is how to evaluate
    /// under a cap other than 300.
    ///
    /// Behavior controlled by per-image GT fields:
    /// - `neg_category_ids`: categories confirmed absent → unmatched DTs count as FP.
    /// - `not_exhaustive_category_ids`: categories not fully checked → unmatched DTs ignored.
    ///
    /// Produces 13 metrics: AP, AP50, AP75, APs, APm, APl, APr (rare), APc (common),
    /// APf (frequent), AR@300, ARs@300, ARm@300, ARl@300.
    pub fn new_lvis(
        coco_gt: impl Into<Arc<COCO>>,
        coco_dt: impl Into<Arc<COCO>>,
        iou_type: IouType,
    ) -> Self {
        Self::with_mode(
            coco_gt,
            coco_dt,
            EvalMode::Lvis.default_params(iou_type),
            EvalMode::Lvis,
            None,
        )
    }

    /// Run the full evaluation pipeline in one call: `evaluate` → `accumulate` → `summarize`.
    ///
    /// Equivalent to calling the three methods in sequence. Primarily used with LVIS
    /// pipelines such as Detectron2 and MMDetection that expect a single `run()` entry point.
    pub fn run(&mut self) {
        self.evaluate();
        self.accumulate();
        self.summarize();
    }

    /// Create a new COCOeval configured for Open Images detection evaluation.
    ///
    /// OID uses a single IoU threshold (0.5), one area range ("all"), and
    /// `max_dets=100`. If a [`Hierarchy`] is provided, GT annotations are expanded
    /// up the hierarchy during `evaluate()`. Set `params.expand_dt = true` to
    /// also expand detections.
    pub fn new_oid(
        coco_gt: impl Into<Arc<COCO>>,
        coco_dt: impl Into<Arc<COCO>>,
        hierarchy: Option<Hierarchy>,
    ) -> Self {
        Self::with_mode(
            coco_gt,
            coco_dt,
            EvalMode::OpenImages.default_params(IouType::Bbox),
            EvalMode::OpenImages,
            hierarchy,
        )
    }
}
