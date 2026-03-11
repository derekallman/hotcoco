//! COCO evaluation engine — faithful port of `pycocotools/cocoeval.py`.
//!
//! Implements evaluate, accumulate, and summarize for bbox, segm, and keypoint evaluation.

pub(crate) mod types;

pub(super) mod accumulate;
mod confusion;
mod evaluate;
mod iou;
mod results;
mod summarize;
mod tide;

pub use results::EvalResults;
pub use types::{AccumulatedEval, ConfusionMatrix, EvalImg, EvalShape, TideErrors};

use std::collections::HashMap;

use crate::coco::COCO;
use crate::params::{IouType, Params};
use types::FreqGroups;

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
    pub eval_imgs: Vec<Option<EvalImg>>,
    ious: HashMap<(u64, u64), Vec<Vec<f64>>>,
    pub eval: Option<AccumulatedEval>,
    pub stats: Option<Vec<f64>>,
    /// LVIS federated evaluation mode.
    pub is_lvis: bool,
    /// LVIS: k_indices bucketed by category frequency.
    /// Populated during `evaluate()` when `is_lvis=true`.
    freq_groups: FreqGroups,
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
            is_lvis: false,
            freq_groups: FreqGroups::default(),
        }
    }

    /// Create a new COCOeval configured for LVIS federated evaluation.
    ///
    /// LVIS uses federated annotation — each image is only exhaustively labeled
    /// for a subset of categories. This constructor sets `max_dets=300` and enables
    /// federated filtering so unmatched detections on unlabeled or unchecked categories
    /// are not penalized as false positives.
    ///
    /// Behaviour controlled by per-image GT fields:
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
            is_lvis: true,
            freq_groups: FreqGroups::default(),
        }
    }
}
