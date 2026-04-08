//! COCO evaluation engine — faithful port of `pycocotools/cocoeval.py`.
//!
//! Implements evaluate, accumulate, and summarize for bbox, segm, and keypoint evaluation.

pub(crate) mod types;

pub(super) mod accumulate;
mod calibration;
mod compare;
mod confusion;
mod diagnostics;
mod evaluate;
pub mod expand;
mod iou;
mod results;
pub mod slice;
mod summarize;
mod tide;
pub mod tracking;

pub use calibration::{CalibrationBin, CalibrationResult};
pub use compare::{BootstrapCI, CategoryDelta, CompareOpts, ComparisonResult, compare};
pub use diagnostics::{
    AnnotationIndex, DtStatus, ErrorProfile, GtStatus, ImageDiagnostics, ImageSummary, LabelError,
    LabelErrorType,
};
pub use results::EvalResults;
pub use slice::{SliceResult, SlicedResults};
pub use types::{AccumulatedEval, ConfusionMatrix, EvalImg, EvalShape, TideErrors};

use std::collections::HashMap;

use crate::coco::COCO;
use crate::hierarchy::Hierarchy;
use crate::params::{IouType, Params};
use types::FreqGroups;

/// Evaluation mode: determines matching semantics, metric sets, and output formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    /// Standard COCO evaluation (12 bbox/segm metrics or 10 keypoint metrics).
    Coco,
    /// LVIS federated evaluation (13 metrics including frequency-group AP).
    Lvis,
    /// Open Images detection evaluation (hierarchy-aware, group-of matching).
    OpenImages,
}

/// COCO evaluation engine.
///
/// Computes AP and AR metrics for bbox, segmentation, and keypoint predictions.
/// Also supports LVIS federated evaluation via [`COCOeval::new_lvis`].
///
/// The standard workflow is three steps:
///
/// ```rust,ignore
/// let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
/// ev.evaluate();   // per-image IoU matching
/// ev.accumulate(); // aggregate into precision/recall curves
/// ev.summarize();  // print + store the summary metrics in ev.stats
/// ```
///
/// For LVIS, use [`run`](COCOeval::run) as a convenience:
///
/// ```rust,ignore
/// let mut ev = COCOeval::new_lvis(coco_gt, coco_dt, IouType::Segm);
/// ev.run();
/// let results = ev.get_results(None, false); // HashMap<metric_name, f64>
/// ```
pub struct COCOeval {
    pub coco_gt: COCO,
    pub coco_dt: COCO,
    pub params: Params,
    pub(crate) eval_imgs: Vec<Option<EvalImg>>,
    ious: HashMap<(u64, u64), types::IouMatrix>,
    pub(crate) eval: Option<AccumulatedEval>,
    pub(crate) stats: Option<Vec<f64>>,
    /// Evaluation mode (COCO, LVIS, or OpenImages).
    pub eval_mode: EvalMode,
    /// LVIS: k_indices bucketed by category frequency.
    /// Populated during `evaluate()` when `eval_mode == Lvis`.
    freq_groups: FreqGroups,
    /// Open Images: category hierarchy for GT/DT expansion.
    pub hierarchy: Option<Hierarchy>,
}

impl COCOeval {
    /// Create a new COCOeval from ground truth and detection COCO objects.
    pub fn new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        COCOeval {
            coco_gt,
            coco_dt,
            params: Params::new(iou_type),
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
            eval_mode: EvalMode::Coco,
            freq_groups: FreqGroups::default(),
            hierarchy: None,
        }
    }

    /// Per-image evaluation results (sparse — indexed by image position).
    pub fn eval_imgs(&self) -> &[Option<EvalImg>] {
        &self.eval_imgs
    }

    /// Accumulated precision/recall curves (set after `accumulate()`).
    pub fn accumulated(&self) -> Option<&AccumulatedEval> {
        self.eval.as_ref()
    }

    /// Summary statistics (set after `summarize()`).
    pub fn stats(&self) -> Option<&[f64]> {
        self.stats.as_deref()
    }

    /// Create a new COCOeval configured for LVIS federated evaluation.
    ///
    /// LVIS uses federated annotation — each image is only exhaustively labeled
    /// for a subset of categories. This constructor sets `max_dets=300` and enables
    /// federated filtering so unmatched detections on unlabeled or unchecked categories
    /// are not penalized as false positives.
    ///
    /// Behavior controlled by per-image GT fields:
    /// - `neg_category_ids`: categories confirmed absent → unmatched DTs count as FP.
    /// - `not_exhaustive_category_ids`: categories not fully checked → unmatched DTs ignored.
    ///
    /// Produces 13 metrics: AP, AP50, AP75, APs, APm, APl, APr (rare), APc (common),
    /// APf (frequent), AR@300, ARs@300, ARm@300, ARl@300.
    pub fn new_lvis(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        let mut params = Params::new(iou_type);
        params.max_dets = vec![300];

        COCOeval {
            coco_gt,
            coco_dt,
            params,
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
            eval_mode: EvalMode::Lvis,
            freq_groups: FreqGroups::default(),
            hierarchy: None,
        }
    }

    /// Create a new COCOeval configured for Open Images detection evaluation.
    ///
    /// OID uses a single IoU threshold (0.5), one area range ("all"), and
    /// `max_dets=100`. If a [`Hierarchy`] is provided, GT annotations are expanded
    /// up the hierarchy during `evaluate()`. Set `params.expand_dt = true` to
    /// also expand detections.
    pub fn new_oid(coco_gt: COCO, coco_dt: COCO, hierarchy: Option<Hierarchy>) -> Self {
        let mut params = Params::new(IouType::Bbox);
        params.iou_thrs = vec![0.5];
        params.area_ranges = vec![crate::AreaRange {
            label: "all".to_string(),
            range: [0.0, 1e10],
        }];
        params.max_dets = vec![100];

        COCOeval {
            coco_gt,
            coco_dt,
            params,
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
            eval_mode: EvalMode::OpenImages,
            freq_groups: FreqGroups::default(),
            hierarchy,
        }
    }
}
