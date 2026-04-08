use crate::params::IouType;
use std::collections::HashMap;

/// Which tracking metric families to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingMetric {
    Hota,
    Clear,
    Identity,
}

/// Parameters for tracking evaluation.
#[derive(Debug, Clone)]
pub struct TrackingParams {
    /// IoU type used for matching (bbox or segm).
    pub iou_type: IouType,
    /// HOTA alpha thresholds (default: 0.05 to 0.95 step 0.05, 19 values).
    pub iou_thrs: Vec<f64>,
    /// Similarity threshold for CLEAR and Identity matching (default: 0.5).
    pub similarity_thr: f64,
    /// Which metric families to compute.
    pub metrics: Vec<TrackingMetric>,
    /// Maximum detections per frame (None = unlimited).
    pub max_detections: Option<usize>,
}

impl Default for TrackingParams {
    fn default() -> Self {
        Self {
            iou_type: IouType::Bbox,
            iou_thrs: (1..=19).map(|i| i as f64 * 0.05).collect(),
            similarity_thr: 0.5,
            metrics: vec![
                TrackingMetric::Hota,
                TrackingMetric::Clear,
                TrackingMetric::Identity,
            ],
            max_detections: None,
        }
    }
}

/// Contiguous index maps for GT and DT track IDs across a sequence.
///
/// Shared by HOTA and Identity metric computation — both need to assign
/// contiguous indices to track IDs for matrix operations.
#[derive(Debug)]
pub(crate) struct TrackIdMaps {
    /// Unique GT track IDs in insertion order.
    pub gt_ids: Vec<u64>,
    /// Unique DT track IDs in insertion order.
    pub dt_ids: Vec<u64>,
    /// GT track ID → contiguous index.
    pub gt_map: HashMap<u64, usize>,
    /// DT track ID → contiguous index.
    pub dt_map: HashMap<u64, usize>,
}

impl TrackIdMaps {
    /// Build ID maps from all frames in a sequence.
    pub fn build(frames: &[FrameData]) -> Self {
        let mut gt_ids = Vec::new();
        let mut dt_ids = Vec::new();
        let mut gt_map = HashMap::new();
        let mut dt_map = HashMap::new();

        for frame in frames {
            for &tid in &frame.gt_track_ids {
                gt_map.entry(tid).or_insert_with(|| {
                    let idx = gt_ids.len();
                    gt_ids.push(tid);
                    idx
                });
            }
            for &tid in &frame.dt_track_ids {
                dt_map.entry(tid).or_insert_with(|| {
                    let idx = dt_ids.len();
                    dt_ids.push(tid);
                    idx
                });
            }
        }

        Self {
            gt_ids,
            dt_ids,
            gt_map,
            dt_map,
        }
    }
}

/// Per-frame data for tracking evaluation.
#[derive(Debug, Clone)]
pub(crate) struct FrameData {
    /// Ground truth track IDs in this frame.
    pub gt_track_ids: Vec<u64>,
    /// Detection track IDs in this frame.
    pub dt_track_ids: Vec<u64>,
    /// Pre-computed IoU matrix (D x G).
    pub iou_matrix: Vec<Vec<f64>>,
}

/// Result of a per-frame matching operation.
#[derive(Debug, Clone)]
pub(crate) struct MatchResult {
    /// Matched pairs as (dt_index, gt_index).
    pub matches: Vec<(usize, usize)>,
    /// Indices of unmatched ground truths.
    pub unmatched_gt: Vec<usize>,
    /// Indices of unmatched detections.
    pub unmatched_dt: Vec<usize>,
}

/// HOTA metric results (per-sequence or combined).
#[derive(Debug, Clone)]
pub struct HotaResult {
    /// HOTA score per alpha threshold.
    pub hota: Vec<f64>,
    /// Detection accuracy per alpha threshold.
    pub det_a: Vec<f64>,
    /// Association accuracy per alpha threshold.
    pub ass_a: Vec<f64>,
    /// Localization accuracy per alpha threshold.
    pub loc_a: Vec<f64>,
    /// Detection recall per alpha threshold.
    pub det_re: Vec<f64>,
    /// Detection precision per alpha threshold.
    pub det_pr: Vec<f64>,
    /// Association recall per alpha threshold.
    pub ass_re: Vec<f64>,
    /// Association precision per alpha threshold.
    pub ass_pr: Vec<f64>,
    // Integer counts for cross-sequence aggregation.
    pub(crate) hota_tp: Vec<u64>,
    pub(crate) hota_fn: Vec<u64>,
    pub(crate) hota_fp: Vec<u64>,
    /// Weighted association sum per alpha (for aggregation: sum of A(c) per match).
    pub(crate) ass_sum: Vec<f64>,
    /// Sum of IoU for TPs per alpha (for LocA aggregation).
    pub(crate) loc_sum: Vec<f64>,
}

/// CLEAR/MOT metric results (per-sequence or combined).
#[derive(Debug, Clone)]
pub struct ClearResult {
    /// Multiple Object Tracking Accuracy.
    pub mota: f64,
    /// Multiple Object Tracking Precision (mean IoU of matched pairs).
    pub motp: f64,
    /// Number of identity switches.
    pub num_id_switches: u64,
    /// True positives (matched detections).
    pub clr_tp: u64,
    /// False negatives (missed ground truths).
    pub clr_fn: u64,
    /// False positives (unmatched detections).
    pub clr_fp: u64,
    /// Mostly tracked: GT tracks matched in >80% of frames.
    pub mt: u64,
    /// Partially tracked: GT tracks matched in 20%-80% of frames.
    pub pt: u64,
    /// Mostly lost: GT tracks matched in <20% of frames.
    pub ml: u64,
    /// Number of track fragmentations.
    pub frag: u64,
    /// Total number of GT track IDs.
    pub num_gt_ids: u64,
    /// Total number of DT track IDs.
    pub num_dt_ids: u64,
    // For aggregation:
    pub(crate) iou_sum: f64,
}

/// Identity (IDF1) metric results (per-sequence or combined).
#[derive(Debug, Clone)]
pub struct IdentityResult {
    /// Identity F1 score.
    pub idf1: f64,
    /// Identity precision.
    pub idp: f64,
    /// Identity recall.
    pub idr: f64,
    /// Identity true positives (matched frames from optimal assignment).
    pub idtp: u64,
    /// Identity false negatives.
    pub idfn: u64,
    /// Identity false positives.
    pub idfp: u64,
}

/// Per-sequence tracking evaluation results.
#[derive(Debug, Clone)]
pub struct SeqResult {
    /// Video/sequence ID.
    pub video_id: u64,
    /// HOTA results (if computed).
    pub hota: Option<HotaResult>,
    /// CLEAR results (if computed).
    pub clear: Option<ClearResult>,
    /// Identity results (if computed).
    pub identity: Option<IdentityResult>,
}

/// Combined tracking results across all sequences.
#[derive(Debug, Clone)]
pub struct CombinedResults {
    pub hota: Option<HotaResult>,
    pub clear: Option<ClearResult>,
    pub identity: Option<IdentityResult>,
}

impl CombinedResults {
    /// Flatten combined results into a metric name -> value map.
    pub fn to_map(&self, prefix: Option<&str>) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        let pfx = prefix.map(|p| format!("{p}/")).unwrap_or_default();

        if let Some(ref h) = self.hota {
            let n = h.hota.len() as f64;
            if n > 0.0 {
                map.insert(format!("{pfx}HOTA"), h.hota.iter().sum::<f64>() / n);
                map.insert(format!("{pfx}DetA"), h.det_a.iter().sum::<f64>() / n);
                map.insert(format!("{pfx}AssA"), h.ass_a.iter().sum::<f64>() / n);
                map.insert(format!("{pfx}LocA"), h.loc_a.iter().sum::<f64>() / n);
                map.insert(format!("{pfx}DetRe"), h.det_re.iter().sum::<f64>() / n);
                map.insert(format!("{pfx}DetPr"), h.det_pr.iter().sum::<f64>() / n);
                map.insert(format!("{pfx}AssRe"), h.ass_re.iter().sum::<f64>() / n);
                map.insert(format!("{pfx}AssPr"), h.ass_pr.iter().sum::<f64>() / n);
            }
        }

        if let Some(ref c) = self.clear {
            map.insert(format!("{pfx}MOTA"), c.mota);
            map.insert(format!("{pfx}MOTP"), c.motp);
            map.insert(format!("{pfx}IDSW"), c.num_id_switches as f64);
            map.insert(format!("{pfx}CLR_TP"), c.clr_tp as f64);
            map.insert(format!("{pfx}CLR_FN"), c.clr_fn as f64);
            map.insert(format!("{pfx}CLR_FP"), c.clr_fp as f64);
            map.insert(format!("{pfx}MT"), c.mt as f64);
            map.insert(format!("{pfx}PT"), c.pt as f64);
            map.insert(format!("{pfx}ML"), c.ml as f64);
            map.insert(format!("{pfx}Frag"), c.frag as f64);
        }

        if let Some(ref id) = self.identity {
            map.insert(format!("{pfx}IDF1"), id.idf1);
            map.insert(format!("{pfx}IDP"), id.idp);
            map.insert(format!("{pfx}IDR"), id.idr);
            map.insert(format!("{pfx}IDTP"), id.idtp as f64);
            map.insert(format!("{pfx}IDFN"), id.idfn as f64);
            map.insert(format!("{pfx}IDFP"), id.idfp as f64);
        }

        map
    }
}
