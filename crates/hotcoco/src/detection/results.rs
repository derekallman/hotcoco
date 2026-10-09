use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use crate::params::{IouType, Params};
use crate::report::Provenance;

use super::EvalMode;

/// Serializable summary of evaluation parameters.
///
/// A projection of [`Params`] carrying every field that decides what the
/// metrics *mean* — the archive has to be self-explaining, because a saved
/// results file whose configuration must be reconstructed from memory is the
/// situation it exists to prevent. That is also why
/// [`reference_deviations`](Self::reference_deviations) rides along: a file
/// marked `Extension` should say *why* on its own.
#[derive(Debug, Clone, Serialize)]
pub struct EvalParams {
    pub iou_type: IouType,
    pub iou_thresholds: Vec<f64>,
    /// The recall grid AP is interpolated over — the x-axis that defines what
    /// AP means (101 points for the reference configuration).
    pub recall_thresholds: Vec<f64>,
    /// Area ranges as a map from label to `[min, max]`.
    pub area_ranges: BTreeMap<String, [f64; 2]>,
    pub max_dets: Vec<usize>,
    /// Whether evaluation was per-category (`true`) or pooled (`false`).
    pub use_cats: bool,
    /// Per-keypoint OKS sigmas. Only consulted by keypoint evaluation, but
    /// archived unconditionally so the file's shape does not depend on the run.
    pub kpt_oks_sigmas: Vec<f64>,
    /// Evaluation mode: "coco", "lvis", or "openimages".
    pub eval_mode: String,
    /// Ways this run departed from the reference configuration — the same
    /// strings [`COCOeval::reference_deviations`](super::COCOeval::reference_deviations)
    /// returns, and the reason whenever `provenance` says `extension`. Empty
    /// for a parity-verified run.
    pub reference_deviations: Vec<String>,
}

/// Serializable evaluation results.
///
/// Returned by [`super::COCOeval::results`]. Contains summary metrics,
/// evaluation parameters, and optional per-class breakdown.
///
/// Every map here is a `BTreeMap` so serialization is byte-stable — this is the
/// struct users archive, diff in CI, and check into git, so identical runs must
/// produce identical bytes. `report::EvalReport` makes the same choice.
///
/// Use [`save`](EvalResults::save) to write JSON to a file, or
/// [`to_json`](EvalResults::to_json) to get a JSON string.
///
/// `#[non_exhaustive]`: this is an output DTO that grows as families report more
/// about a run, and callers read it rather than construct it.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct EvalResults {
    /// hotcoco version that produced these results.
    pub hotcoco_version: String,
    /// Whether these numbers are comparable to a reference implementation's
    /// published output, or a hotcoco extension.
    ///
    /// Carried here, and not only on [`EvalReport`](crate::EvalReport), because
    /// this is the struct that gets archived: `save()` writes it, the CLI's
    /// `--json` emits it, and the PDF report renders from it. Provenance that
    /// exists only inside a live process cannot be audited afterwards.
    pub provenance: Provenance,
    /// Evaluation parameters used to produce these metrics.
    pub params: EvalParams,
    /// Summary metrics: AP, AP50, AP75, AR1, AR10, AR100, and the rest of the catalog.
    pub metrics: BTreeMap<String, f64>,
    /// Per-class AP values, keyed by category name. `None` if not requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_class: Option<BTreeMap<String, f64>>,
}

impl EvalResults {
    /// Serialize results to a pretty-printed JSON string.
    pub fn to_json(&self) -> crate::error::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Write results as pretty-printed JSON to a file.
    pub fn save(&self, path: &Path) -> crate::error::Result<()> {
        crate::report::save_json(self, path)
    }
}

impl EvalParams {
    /// Archive an evaluator's configuration.
    ///
    /// Takes the whole evaluator rather than a bare [`Params`] so the deviation
    /// strings come from
    /// [`reference_deviations`](super::COCOeval::reference_deviations), the same
    /// predicate that drives `Provenance`, instead of being re-derived here.
    pub(in crate::detection) fn from_eval(ev: &super::COCOeval) -> Self {
        let params: &Params = &ev.params;
        let area_ranges: BTreeMap<String, [f64; 2]> = params
            .area_ranges
            .iter()
            .map(|ar| (ar.label.clone(), ar.range))
            .collect();

        let mode_str = match ev.eval_mode {
            EvalMode::Coco => "coco",
            EvalMode::Lvis => "lvis",
            EvalMode::OpenImages => "openimages",
        };

        EvalParams {
            iou_type: params.iou_type,
            iou_thresholds: params.iou_thrs.clone(),
            recall_thresholds: params.rec_thrs.clone(),
            area_ranges,
            max_dets: params.max_dets.clone(),
            use_cats: params.use_cats,
            kpt_oks_sigmas: params.kpt_oks_sigmas.clone(),
            eval_mode: mode_str.to_string(),
            reference_deviations: ev.reference_deviations(),
        }
    }
}
