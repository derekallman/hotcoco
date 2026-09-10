//! Perception evaluation in pure Rust.
//!
//! Detection ships today — bbox, segmentation, keypoints, and oriented boxes
//! across the COCO, LVIS, and Open Images protocols — on a layered engine that
//! other metric families will share.
//!
//! ```no_run
//! use hotcoco::{COCO, COCOeval, params::IouType};
//! # fn main() -> hotcoco::error::Result<()> {
//! let gt = COCO::new(std::path::Path::new("instances_val2017.json"))?;
//! let dt = gt.load_res(std::path::Path::new("detections.json"))?;
//!
//! let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
//! ev.run();                       // evaluate -> accumulate -> summarize
//! let report = ev.report()?;      // metrics, per-class, curves, provenance
//! # Ok(())
//! # }
//! ```
//!
//! # How the crate is laid out
//!
//! | Module | What lives there |
//! |---|---|
//! | [`types`] | The COCO schema — `Dataset`, `Image`, `Annotation`, `Category`, `Rle`. |
//! | [`coco`] | The dataset object: load, index, query, filter, merge, split, sample. |
//! | [`mask`], [`geometry`] | RLE codec and rotated-rect mechanics. |
//! | [`primitives`] | Matching kernels — similarity, greedy assignment, LSAP. |
//! | [`metrics`] | Metric functions over flat arrays — AP, calibration, confusion, bootstrap. |
//! | [`report`] | [`EvalReport`] — the shape every metric family reports in. |
//! | [`detection`] | The detection metric family: AP/AR, LVIS, Open Images, TIDE. |
//! | [`quality`] | Dataset introspection: health checks and statistics. |
//! | [`convert`] | YOLO, Pascal VOC, CVAT, DOTA, and Open Images conversion. |
//!
//! # The functional layer
//!
//! [`primitives`] and [`metrics`] are free functions over flat arrays — no
//! evaluator required, the way `sklearn.metrics` and `torchmetrics.functional`
//! work:
//!
//! ```
//! use hotcoco::metrics::counts::average_precision;
//!
//! let ap = average_precision(&[0.9, 0.8, 0.3], &[true, false, true], None, 3, &[0.0, 0.5, 1.0]);
//! ```
//!
//! The two split by what a function *produces*: [`primitives`] produces matches,
//! [`metrics`] produces numbers from matches (see [`metrics`] for the full
//! split). Between them they hold *the* implementation of every similarity,
//! matching and accumulation rule in the crate — exactly one of each, with
//! `tests/architecture.rs` failing the build if a second appears. That is what
//! makes them the place for an auditor to look.
//!
//! [`COCOeval`] is the stateful driver on top — it owns the pycocotools-compatible
//! `evaluate`/`accumulate`/`summarize` lifecycle, and its analysis methods are
//! adapters that marshal `eval_imgs` into arrays and call the functions above.
//!
//! # Coming from 0.x
//!
//! 1.0 renamed several module paths (`eval` → [`detection`] and friends) with no
//! aliases; the crate-root re-exports resolve unchanged. The rename table is in
//! the [migration guide](https://derekallman.github.io/hotcoco/getting-started/migration/).

pub mod coco;
pub mod convert;
pub mod detection;
pub mod error;
pub mod geometry;
pub mod mask;
pub mod metrics;
pub mod params;
pub mod primitives;
pub mod quality;
pub mod report;
pub mod types;

pub use coco::COCO;
pub use convert::{
    ConvertError, CvatImportStats, CvatStats, DotaStats, OidStats, VocStats, YoloStats,
};
pub use detection::{
    AccumulatedEval, AnnotationIndex, COCOeval, CalibrationResult, CategoryDelta, CompareOpts,
    ComparisonResult, ConfusionMatrix, DtStatus, ErrorProfile, EvalImg, EvalMode, EvalParams,
    EvalResults, EvalShape, FreqGroup, GtStatus, ImageDiagnostics, ImageSummary, LabelError,
    LabelErrorType, MetricDef, SliceResult, SlicedResults, TideErrors, compare,
};
pub use error::{Error, UnknownAnnIds};
// Re-exported at the root because it is the shape of `EvalImg`'s per-threshold
// fields — a consumer holding an `EvalImg` needs the type nameable.
pub use primitives::greedy::ThreshMatrix;

pub use detection::hierarchy::Hierarchy;
// Re-exported from where they are defined, not through `detection`. Both are
// family-agnostic — any family that resamples gets a `BootstrapCI`, any family
// that bins confidences gets a `CalibrationBin` — so routing the crate-root path
// through the detection driver would make the next family import a detection path
// for a type detection does not own.
pub use metrics::bootstrap::BootstrapCI;
pub use metrics::calibration::CalibrationBin;
pub use params::{AreaRange, IouType, Params};
pub use quality::{
    CategoryStats, DatasetStats, DatasetSummary, Finding, HealthReport, Layer, SummaryStats,
};
pub use report::{EvalReport, Provenance};
pub use types::{Annotation, Category, Dataset, Image, Rle, Segmentation};
