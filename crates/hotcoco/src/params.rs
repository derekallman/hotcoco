use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// A single area-range filter: a human-readable label paired with its `[min, max]` bounds.
///
/// Used in [`Params::area_ranges`] to keep labels and ranges in sync.
/// Standard COCO labels are `"all"`, `"small"`, `"medium"`, `"large"`.
#[derive(Debug, Clone)]
pub struct AreaRange {
    pub label: String,
    pub range: [f64; 2],
}

/// The type of IoU (intersection over union) computation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum IouType {
    /// Bounding box IoU.
    Bbox,
    /// Segmentation mask IoU (RLE-based).
    Segm,
    /// Keypoint OKS (object keypoint similarity).
    Keypoints,
}

impl fmt::Display for IouType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IouType::Bbox => write!(f, "bbox"),
            IouType::Segm => write!(f, "segm"),
            IouType::Keypoints => write!(f, "keypoints"),
        }
    }
}

impl FromStr for IouType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bbox" => Ok(IouType::Bbox),
            "segm" => Ok(IouType::Segm),
            "keypoints" => Ok(IouType::Keypoints),
            _ => Err(format!(
                "Unknown iou_type: '{}'. Expected 'bbox', 'segm', or 'keypoints'",
                s
            )),
        }
    }
}

/// Small-object area upper bound: 32² = 1024 px².
pub(crate) const AREA_SMALL: f64 = 32.0 * 32.0;

/// Medium/large-object area boundary: 96² = 9216 px².
pub(crate) const AREA_LARGE: f64 = 96.0 * 96.0;

/// Default OKS sigmas for the 17 COCO keypoints (nose, eyes, ears, shoulders, …, ankles).
pub(crate) const KPT_OKS_SIGMAS: [f64; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107,
    0.087, 0.087, 0.089, 0.089,
];

/// Generate the default COCO IoU threshold range: 0.50, 0.55, …, 0.95.
pub(crate) fn default_iou_thrs() -> Vec<f64> {
    (0..10).map(|i| 0.5 + 0.05 * i as f64).collect()
}

/// Evaluation parameters controlling IoU thresholds, area ranges, and detection limits.
///
/// Defaults match pycocotools: 10 IoU thresholds (0.50:0.05:0.95), 101 recall
/// thresholds, and standard COCO area ranges. Keypoint evaluation uses different
/// defaults (3 area ranges instead of 4, max 20 detections instead of 1/10/100).
#[derive(Debug, Clone)]
pub struct Params {
    /// IoU computation type (bbox, segm, or keypoints).
    pub iou_type: IouType,
    /// Image IDs to evaluate (empty = all images).
    pub img_ids: Vec<u64>,
    /// Category IDs to evaluate (empty = all categories).
    pub cat_ids: Vec<u64>,
    /// IoU thresholds for matching (default: 0.50, 0.55, ..., 0.95).
    pub iou_thrs: Vec<f64>,
    /// Recall thresholds for interpolated precision (default: 0.00, 0.01, ..., 1.00).
    pub rec_thrs: Vec<f64>,
    /// Maximum detections per image for each summary metric (default: [1, 10, 100]).
    pub max_dets: Vec<usize>,
    /// Area ranges for filtering, each with a label and `[min, max]` bounds.
    /// Default labels: `"all"`, `"small"`, `"medium"`, `"large"` (3 ranges for keypoints).
    pub area_ranges: Vec<AreaRange>,
    /// Whether to evaluate per-category (true) or pool all categories (false).
    pub use_cats: bool,
    /// Per-keypoint OKS sigmas (default: 17 COCO keypoint sigmas).
    pub kpt_oks_sigmas: Vec<f64>,
    /// Whether to expand detections up the category hierarchy (OID mode).
    /// Default: false (only GT is expanded).
    pub expand_dt: bool,
}

impl Params {
    /// Index of the area range with the given label, or `None` if not found.
    pub fn area_range_idx(&self, label: &str) -> Option<usize> {
        self.area_ranges.iter().position(|ar| ar.label == label)
    }

    /// Create default parameters for the given evaluation type.
    ///
    /// Keypoint evaluation uses 3 area ranges (all/medium/large) and a single
    /// max-detections value of 20. All other types use 4 area ranges
    /// (all/small/medium/large) and max-detections of [1, 10, 100].
    pub fn new(iou_type: IouType) -> Self {
        let (max_dets, area_ranges) = match iou_type {
            IouType::Keypoints => (
                vec![20],
                vec![
                    AreaRange {
                        label: "all".into(),
                        range: [0.0, 1e10],
                    },
                    AreaRange {
                        label: "medium".into(),
                        range: [AREA_SMALL, AREA_LARGE],
                    },
                    AreaRange {
                        label: "large".into(),
                        range: [AREA_LARGE, 1e10],
                    },
                ],
            ),
            _ => (
                vec![1, 10, 100],
                vec![
                    AreaRange {
                        label: "all".into(),
                        range: [0.0, 1e10],
                    },
                    AreaRange {
                        label: "small".into(),
                        range: [0.0, AREA_SMALL],
                    },
                    AreaRange {
                        label: "medium".into(),
                        range: [AREA_SMALL, AREA_LARGE],
                    },
                    AreaRange {
                        label: "large".into(),
                        range: [AREA_LARGE, 1e10],
                    },
                ],
            ),
        };

        let kpt_oks_sigmas = KPT_OKS_SIGMAS.to_vec();
        let iou_thrs = default_iou_thrs();
        let rec_thrs: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();

        Params {
            iou_type,
            img_ids: Vec::new(),
            cat_ids: Vec::new(),
            iou_thrs,
            rec_thrs,
            max_dets,
            area_ranges,
            use_cats: true,
            kpt_oks_sigmas,
            expand_dt: false,
        }
    }
}
