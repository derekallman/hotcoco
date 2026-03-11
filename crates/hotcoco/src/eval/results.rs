use std::collections::HashMap;
use std::io;
use std::path::Path;

use serde::Serialize;

use crate::params::{IouType, Params};

/// Serializable summary of evaluation parameters.
///
/// A lightweight projection of [`Params`] containing only the fields
/// relevant for understanding what configuration produced the metrics.
#[derive(Debug, Clone, Serialize)]
pub struct EvalParams {
    pub iou_type: IouType,
    pub iou_thresholds: Vec<f64>,
    /// Area ranges as a map from label to `[min, max]`.
    pub area_ranges: HashMap<String, [f64; 2]>,
    pub max_dets: Vec<usize>,
    pub is_lvis: bool,
}

/// Serializable evaluation results.
///
/// Returned by [`super::COCOeval::results`]. Contains summary metrics,
/// evaluation parameters, and optional per-class breakdown.
///
/// Use [`save`](EvalResults::save) to write JSON to a file, or
/// [`to_json`](EvalResults::to_json) to get a JSON string.
#[derive(Debug, Clone, Serialize)]
pub struct EvalResults {
    /// hotcoco version that produced these results.
    pub hotcoco_version: String,
    /// Evaluation parameters used to produce these metrics.
    pub params: EvalParams,
    /// Summary metrics (AP, AP50, AP75, AR1, AR10, AR100, etc.).
    pub metrics: HashMap<String, f64>,
    /// Per-class AP values, keyed by category name. `None` if not requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_class: Option<HashMap<String, f64>>,
}

impl EvalResults {
    /// Serialize results to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, io::Error> {
        serde_json::to_string_pretty(self).map_err(io::Error::other)
    }

    /// Write results as pretty-printed JSON to a file.
    pub fn save(&self, path: &Path) -> Result<(), io::Error> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(io::Error::other)
    }
}

impl EvalParams {
    /// Create from a [`Params`] struct and LVIS flag.
    pub(super) fn from_params(params: &Params, is_lvis: bool) -> Self {
        let area_ranges: HashMap<String, [f64; 2]> = params
            .area_ranges
            .iter()
            .map(|ar| (ar.label.clone(), ar.range))
            .collect();

        EvalParams {
            iou_type: params.iou_type,
            iou_thresholds: params.iou_thrs.clone(),
            area_ranges,
            max_dets: params.max_dets.clone(),
            is_lvis,
        }
    }
}
