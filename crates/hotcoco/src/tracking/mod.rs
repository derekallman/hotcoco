//! The tracking metric family: HOTA, CLEAR, and Identity on MOTChallenge
//! 2D boxes.
//!
//! [`TrackingEval`] is the driver — a port of TrackEval's `MotChallenge2DBox`
//! dataset and its `HOTA`, `CLEAR`, `Identity`, and `Count` metrics, checked
//! against TrackEval run on synthetic sequences (`tests/tracking.rs`,
//! `scripts/gen_tracking_fixtures.py`). It owns the control flow only:
//! reading MOTChallenge rows, applying a benchmark's clean-up, and matching
//! each sequence frame by frame. The formulas are
//! [`crate::metrics::tracking`]'s, the assignment
//! [`crate::primitives::assign::lsap`].
//!
//! ```no_run
//! use std::path::Path;
//! use hotcoco::tracking::{Benchmark, TrackingEval};
//! # fn main() -> hotcoco::error::Result<()> {
//! let mut ev = TrackingEval::new(Some(Benchmark::Mot17), 0.5, 0.5)?;
//! ev.update_folders(
//!     Path::new("data/gt/mot_challenge/MOT17-train"),
//!     Path::new("data/trackers/mot_challenge/MOT17-train/MyTracker/data"),
//!     None,
//! )?;
//! ev.summarize();
//! let hota = ev.results()?.stats()[0];
//! # Ok(())
//! # }
//! ```
//!
//! # Two ways in
//!
//! [`TrackingEval::update`] takes one sequence as MOTChallenge rows —
//! `frame, id, x, y, w, h, conf, class, visibility`, row-major — for each
//! side, with the sequence's frame count, which the rows cannot say: a frame
//! with no boxes still counts. [`TrackingEval::update_folders`] reads a
//! benchmark laid out as TrackEval expects it, and evaluates its sequences in
//! parallel. Each sequence is evaluated when it is added and only its result
//! is kept, so two evaluators over different sequences
//! [`merge`](TrackingEval::merge).
//!
//! # Benchmarks
//!
//! A [`Benchmark`] picks MOTChallenge's clean-up: only pedestrians are
//! scored, ground truth marked "do not evaluate" is dropped, and a prediction
//! on a distractor (a person on a vehicle, a static person, a reflection) is
//! taken out rather than counted as a false positive. `None` evaluates every
//! row as given, which is what a dataset of your own wants. The rules are in
//! `tracking/preprocess.rs`, one test each.
//!
//! # Where this differs from TrackEval on bad input
//!
//! TrackEval truncates a fractional frame or id, aliases a negative id
//! through numpy indexing, and accepts an unknown benchmark name as MOT17.
//! Each of those is an error here. It also needs 7 columns of predictions
//! and 8 of ground truth in every mode; here predictions need 6, and ground
//! truth 6 without a benchmark. Without a benchmark the class column is not
//! read at all, so a prediction class above 1 is not the error it is in
//! TrackEval.

mod evaluate;
mod folders;
mod preprocess;

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::metrics::tracking::{
    ClearResult, CountResult, HotaResult, IdentityResult, NUM_ALPHAS, hota_alphas, mean_over_alphas,
};
use crate::report::{EvalReport, Provenance};

use self::preprocess::Side;

/// The single numbers a run reports, in the order [`TrackingResult::stats`]
/// gives them, under TrackEval's names: HOTA's per-threshold fields as their
/// mean over the 19 thresholds, as TrackEval's summary takes them, then
/// CLEAR, Identity, and Count.
pub const METRIC_NAMES: [&str; 44] = [
    "HOTA",
    "DetA",
    "AssA",
    "DetRe",
    "DetPr",
    "AssRe",
    "AssPr",
    "LocA",
    "OWTA",
    "HOTA(0)",
    "LocA(0)",
    "HOTALocA(0)",
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "sMOTA",
    "CLR_F1",
    "FP_per_frame",
    "MOTAL",
    "MOTP_sum",
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
    "CLR_Frames",
    "IDF1",
    "IDR",
    "IDP",
    "IDTP",
    "IDFN",
    "IDFP",
    "Dets",
    "GT_Dets",
    "IDs",
    "GT_IDs",
];

/// Which MOTChallenge clean-up a run applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Benchmark {
    /// Ground truth marked "do not evaluate" is dropped; no class rule.
    #[serde(rename = "MOT15")]
    Mot15,
    /// Pedestrians only; predictions on distractors removed.
    #[serde(rename = "MOT16")]
    Mot16,
    /// The same rules as MOT16.
    #[serde(rename = "MOT17")]
    Mot17,
    /// MOT17's rules, with non-motorized vehicles as distractors too.
    #[serde(rename = "MOT20")]
    Mot20,
}

impl Benchmark {
    /// The name TrackEval's `BENCHMARK` option takes, such as `"MOT17"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Benchmark::Mot15 => "MOT15",
            Benchmark::Mot16 => "MOT16",
            Benchmark::Mot17 => "MOT17",
            Benchmark::Mot20 => "MOT20",
        }
    }
}

impl fmt::Display for Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Benchmark {
    type Err = Error;

    /// Parse a benchmark name, in any case: `"MOT15"`, `"MOT16"`, `"MOT17"`,
    /// or `"MOT20"`.
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_uppercase().as_str() {
            "MOT15" => Ok(Benchmark::Mot15),
            "MOT16" => Ok(Benchmark::Mot16),
            "MOT17" => Ok(Benchmark::Mot17),
            "MOT20" => Ok(Benchmark::Mot20),
            _ => Err(Error::InvalidInput(format!(
                "unknown benchmark {s:?}; expected MOT15, MOT16, MOT17, or MOT20"
            ))),
        }
    }
}

/// Every metric of one sequence, or of a combination of sequences, under
/// TrackEval's field names when serialized, flat as TrackEval reports them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TrackingResult {
    #[serde(flatten)]
    pub hota: HotaResult,
    #[serde(flatten)]
    pub clear: ClearResult,
    #[serde(flatten)]
    pub identity: IdentityResult,
    #[serde(flatten)]
    pub count: CountResult,
}

impl TrackingResult {
    /// Combine sequences as TrackEval's `COMBINED_SEQ`: counts summed and
    /// scores recomputed, in the order given. TrackEval combines in order of
    /// sequence name, as [`TrackingEval`] holds its sequences.
    pub fn combine<'a>(results: impl IntoIterator<Item = &'a TrackingResult> + Clone) -> Self {
        TrackingResult {
            hota: HotaResult::combine(results.clone().into_iter().map(|r| &r.hota)),
            clear: ClearResult::combine(results.clone().into_iter().map(|r| &r.clear)),
            identity: IdentityResult::combine(results.clone().into_iter().map(|r| &r.identity)),
            count: CountResult::combine(results.into_iter().map(|r| &r.count)),
        }
    }

    /// The values of [`METRIC_NAMES`], in its order.
    pub fn stats(&self) -> [f64; 44] {
        let h = &self.hota;
        let c = &self.clear;
        let n = &c.counts;
        let i = &self.identity;
        let k = &self.count;
        [
            mean_over_alphas(&h.hota),
            mean_over_alphas(&h.det_a),
            mean_over_alphas(&h.ass_a),
            mean_over_alphas(&h.det_re),
            mean_over_alphas(&h.det_pr),
            mean_over_alphas(&h.ass_re),
            mean_over_alphas(&h.ass_pr),
            mean_over_alphas(&h.loc_a),
            mean_over_alphas(&h.owta),
            h.hota_0,
            h.loc_a_0,
            h.hota_loc_a_0,
            c.mota,
            c.motp,
            c.moda,
            c.clr_re,
            c.clr_pr,
            c.mtr,
            c.ptr,
            c.mlr,
            c.smota,
            c.clr_f1,
            c.fp_per_frame,
            c.motal,
            n.motp_sum,
            n.tp as f64,
            n.fn_ as f64,
            n.fp as f64,
            n.idsw as f64,
            n.mt as f64,
            n.pt as f64,
            n.ml as f64,
            n.frag as f64,
            n.frames as f64,
            i.idf1,
            i.idr,
            i.idp,
            i.counts.idtp as f64,
            i.counts.idfn as f64,
            i.counts.idfp as f64,
            k.dets as f64,
            k.gt_dets as f64,
            k.ids as f64,
            k.gt_ids as f64,
        ]
    }

    /// The per-threshold fields as curves, beside their x-axis `"alphas"`:
    /// `HOTA`, `DetA`, `AssA`, `DetRe`, `DetPr`, `AssRe`, `AssPr`, `LocA`,
    /// `OWTA`, and the counts `HOTA_TP`, `HOTA_FN`, `HOTA_FP` as floats.
    pub fn curves(&self) -> BTreeMap<String, Vec<f64>> {
        let mut curves: BTreeMap<String, Vec<f64>> = self
            .hota
            .curves()
            .into_iter()
            .map(|(name, values)| (name.to_owned(), values.to_vec()))
            .collect();
        curves.insert("alphas".to_owned(), hota_alphas().to_vec());
        curves
    }
}

/// What a run is configured with; two evaluators merge only when theirs
/// agree.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Config {
    benchmark: Option<Benchmark>,
    clear_threshold: f64,
    identity_threshold: f64,
}

/// One sequence's MOTChallenge rows, both sides, row-major.
struct SequenceRows<'a> {
    name: &'a str,
    gt: &'a [f64],
    gt_ncols: usize,
    pred: &'a [f64],
    pred_ncols: usize,
    num_frames: usize,
}

impl Config {
    /// Read, check, clean, and evaluate one sequence. The flag says the
    /// sequence kept a ground-truth row TrackEval never evaluates: with no
    /// benchmark nothing is cleaned up, while TrackEval drops a row whose
    /// confidence is 0 even with its clean-up off.
    fn evaluate(&self, rows: &SequenceRows<'_>) -> Result<(TrackingResult, bool)> {
        let name = rows.name;
        let kept_zero_marked = self.benchmark.is_none()
            && rows.gt_ncols > 6
            && rows
                .gt
                .chunks_exact(rows.gt_ncols)
                .any(|row| row[6].trunc() == 0.0);
        let gt = preprocess::read_rows(
            name,
            Side::GroundTruth,
            rows.gt,
            rows.gt_ncols,
            rows.num_frames,
            self.benchmark,
        )?;
        let pred = preprocess::read_rows(
            name,
            Side::Prediction,
            rows.pred,
            rows.pred_ncols,
            rows.num_frames,
            self.benchmark,
        )?;
        preprocess::check_unique_ids(name, &gt, &pred)?;
        let (gt, pred) = preprocess::clean(name, self.benchmark, gt, pred)?;
        let result = evaluate::evaluate(&gt, &pred, self.clear_threshold, self.identity_threshold);
        Ok((result, kept_zero_marked))
    }
}

/// The tracking evaluator: the configuration, and every sequence's result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingEval {
    #[serde(flatten)]
    config: Config,
    sequences: BTreeMap<String, TrackingResult>,
    /// Whether any sequence kept a ground-truth row whose confidence is 0 —
    /// see [`reference_deviations`](Self::reference_deviations).
    #[serde(default)]
    kept_zero_marked: bool,
}

impl TrackingEval {
    /// An evaluator applying `benchmark`'s clean-up — `None` for none — with
    /// CLEAR's and Identity's IoU thresholds, TrackEval's `THRESHOLD`
    /// options; 0.5 for both is TrackEval's default.
    ///
    /// # Errors
    ///
    /// If a threshold is not a number from 0 to 1.
    pub fn new(
        benchmark: Option<Benchmark>,
        clear_threshold: f64,
        identity_threshold: f64,
    ) -> Result<Self> {
        for (name, t) in [
            ("clear_threshold", clear_threshold),
            ("identity_threshold", identity_threshold),
        ] {
            if !(0.0..=1.0).contains(&t) {
                return Err(Error::InvalidInput(format!(
                    "{name} is an IoU from 0 to 1, not {t}"
                )));
            }
        }
        Ok(TrackingEval {
            config: Config {
                benchmark,
                clear_threshold,
                identity_threshold,
            },
            sequences: BTreeMap::new(),
            kept_zero_marked: false,
        })
    }

    /// The benchmark whose clean-up this run applies, if any.
    pub fn benchmark(&self) -> Option<Benchmark> {
        self.config.benchmark
    }

    /// CLEAR's IoU threshold.
    pub fn clear_threshold(&self) -> f64 {
        self.config.clear_threshold
    }

    /// Identity's IoU threshold.
    pub fn identity_threshold(&self) -> f64 {
        self.config.identity_threshold
    }

    /// Evaluate one sequence and keep its result.
    ///
    /// `gt` and `pred` are MOTChallenge rows, row-major, `gt_ncols` and
    /// `pred_ncols` values to a row: `frame, id, x, y, w, h, conf, class,
    /// visibility`. Ground truth needs the first 8 with a benchmark and 6
    /// without; predictions need 6. A prediction row without a class column
    /// is a pedestrian. Frames count from 1 to `num_frames`. Empty rows are
    /// a side with no boxes, whatever the column count.
    ///
    /// # Errors
    ///
    /// If `name` is already evaluated, if the rows do not fill whole rows or
    /// have too few columns, if a frame is not a whole number in
    /// `1..=num_frames`, a track id not a whole number 0 or more, or a box,
    /// confidence, or class read not finite, if a track id repeats in a
    /// frame, or if a benchmark's class rules fail: a prediction class above
    /// 1, or a ground-truth class outside 1..13 in a frame with predictions.
    /// Each names the sequence. On an error nothing is kept.
    pub fn update(
        &mut self,
        name: &str,
        gt: &[f64],
        gt_ncols: usize,
        pred: &[f64],
        pred_ncols: usize,
        num_frames: usize,
    ) -> Result<()> {
        self.check_new(name)?;
        let (result, kept_zero_marked) = self.config.evaluate(&SequenceRows {
            name,
            gt,
            gt_ncols,
            pred,
            pred_ncols,
            num_frames,
        })?;
        self.sequences.insert(name.to_owned(), result);
        self.kept_zero_marked |= kept_zero_marked;
        Ok(())
    }

    /// Evaluate every sequence of a benchmark laid out as TrackEval expects
    /// it, in parallel, and return how many there were.
    ///
    /// Sequence `s` is `{gt_folder}/{s}/gt/gt.txt`, its frame count the
    /// `seqLength` of `{gt_folder}/{s}/seqinfo.ini`, and its predictions
    /// `{pred_folder}/{s}.txt`. The sequences are those named in `seqmap`
    /// (a TrackEval seqmap file: a header line, then one name per line) or,
    /// without one, every folder in `gt_folder` whose name does not start
    /// with a dot. A file is comma-separated, or whitespace-separated when a
    /// line has no comma; blank lines are skipped.
    ///
    /// # Errors
    ///
    /// A missing or unreadable file, as an I/O error naming it; a line that
    /// is not numbers, or that has a different number of fields from the
    /// first; a `seqinfo.ini` with no `seqLength`; a sequence already
    /// evaluated or named twice; no sequences at all; and every error
    /// [`update`](Self::update) raises. Nothing is kept on an error.
    pub fn update_folders(
        &mut self,
        gt_folder: &Path,
        pred_folder: &Path,
        seqmap: Option<&Path>,
    ) -> Result<usize> {
        let sequences = folders::sequences(gt_folder, pred_folder, seqmap)?;
        for s in &sequences {
            self.check_new(&s.name)?;
        }
        let config = self.config;
        let results = sequences
            .par_iter()
            .map(|s| {
                let gt = folders::read_rows_file(&s.gt, "ground-truth")?;
                let pred = folders::read_rows_file(&s.pred, "prediction")?;
                let num_frames = folders::seq_length(&s.seqinfo)?;
                config.evaluate(&SequenceRows {
                    name: &s.name,
                    gt: &gt.values,
                    gt_ncols: gt.ncols,
                    pred: &pred.values,
                    pred_ncols: pred.ncols,
                    num_frames,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let n = sequences.len();
        for (s, (result, kept_zero_marked)) in sequences.into_iter().zip(results) {
            self.sequences.insert(s.name, result);
            self.kept_zero_marked |= kept_zero_marked;
        }
        Ok(n)
    }

    /// An error if a sequence of this name is already evaluated.
    fn check_new(&self, name: &str) -> Result<()> {
        if self.sequences.contains_key(name) {
            return Err(Error::InvalidInput(format!(
                "sequence {name} was already evaluated; each sequence is added once"
            )));
        }
        Ok(())
    }

    /// Add another evaluator's sequences to this one — shards of a
    /// distributed run.
    ///
    /// # Errors
    ///
    /// If the two differ in benchmark or thresholds, or share a sequence
    /// name. Nothing is added on an error.
    pub fn merge(&mut self, other: &TrackingEval) -> Result<()> {
        if self.config != other.config {
            return Err(Error::InvalidInput(
                "cannot merge evaluators with different benchmark or threshold settings".into(),
            ));
        }
        if let Some(name) = other
            .sequences
            .keys()
            .find(|n| self.sequences.contains_key(*n))
        {
            return Err(Error::InvalidInput(format!(
                "cannot merge: both evaluators have sequence {name}"
            )));
        }
        self.sequences
            .extend(other.sequences.iter().map(|(k, v)| (k.clone(), v.clone())));
        self.kept_zero_marked |= other.kept_zero_marked;
        Ok(())
    }

    /// Every sequence's own result, by name. A sequence with no ground truth
    /// or no predictions left after clean-up has TrackEval's early-return
    /// row: CLEAR scores 0 but `MLR` 1, `CLR_Frames` 0, and `LocA` 1.
    pub fn sequence_results(&self) -> &BTreeMap<String, TrackingResult> {
        &self.sequences
    }

    /// The combined result, TrackEval's `COMBINED_SEQ`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidState`] before any sequence is added.
    pub fn results(&self) -> Result<TrackingResult> {
        if self.sequences.is_empty() {
            return Err(Error::InvalidState(
                "add a sequence with update() or update_folders() before reading results".into(),
            ));
        }
        Ok(TrackingResult::combine(self.sequences.values()))
    }

    /// The combined result's [`METRIC_NAMES`] values.
    ///
    /// # Errors
    ///
    /// As [`results`](Self::results).
    pub fn stats(&self) -> Result<[f64; 44]> {
        Ok(self.results()?.stats())
    }

    /// The combined result's per-threshold curves; see
    /// [`TrackingResult::curves`].
    ///
    /// # Errors
    ///
    /// As [`results`](Self::results).
    pub fn curves(&self) -> Result<BTreeMap<String, Vec<f64>>> {
        Ok(self.results()?.curves())
    }

    /// Ways this run departs from what TrackEval would compute.
    ///
    /// Each benchmark is TrackEval's clean-up for it, and the thresholds are
    /// TrackEval options. With no benchmark, the run is TrackEval with its
    /// clean-up off as long as no ground-truth row has confidence 0: TrackEval
    /// drops such a row in every mode, and no clean-up keeps it. That is the
    /// one entry.
    pub fn reference_deviations(&self) -> Vec<String> {
        if self.kept_zero_marked {
            vec![
                "benchmark is None, so ground-truth rows whose confidence is 0 were evaluated; \
                 TrackEval drops them even with its clean-up off"
                    .into(),
            ]
        } else {
            Vec::new()
        }
    }

    /// `ParityVerified` unless [`reference_deviations`](Self::reference_deviations)
    /// has an entry.
    pub fn provenance(&self) -> Provenance {
        if self.reference_deviations().is_empty() {
            Provenance::ParityVerified
        } else {
            Provenance::Extension
        }
    }

    /// The summary as lines: TrackEval's HOTA, CLEAR, Identity, and Count
    /// tables, a row per sequence and a `COMBINED` row, scores in percent.
    /// Empty before any sequence is added.
    pub fn summarize_lines(&self) -> Vec<String> {
        let Ok(combined) = self.results() else {
            return Vec::new();
        };
        let width = self
            .sequences
            .keys()
            .map(|n| n.chars().count())
            .max()
            .unwrap_or(0)
            .max(10)
            + 2;
        let rows: Vec<(&str, [f64; 44])> = self
            .sequences
            .iter()
            .map(|(n, r)| (n.as_str(), r.stats()))
            .chain(std::iter::once(("COMBINED", combined.stats())))
            .collect();
        let mut out = Vec::new();
        for (table, fields) in SUMMARY_TABLES {
            // Each column as wide as its name, and at least 9.
            let columns: Vec<(usize, usize)> = fields
                .iter()
                .map(|f| {
                    let k = METRIC_NAMES
                        .iter()
                        .position(|m| m == f)
                        .expect("a metric name");
                    (k, f.len().max(8) + 1)
                })
                .collect();
            let header: Vec<String> = fields
                .iter()
                .zip(&columns)
                .map(|(f, &(_, w))| format!("{f:>w$}"))
                .collect();
            out.push(format!("{table:<width$}{}", header.concat()));
            for (name, stats) in &rows {
                let cells: Vec<String> = fields
                    .iter()
                    .zip(&columns)
                    .map(|(f, &(k, w))| {
                        if is_count(f) {
                            format!("{:>w$}", stats[k] as u64)
                        } else {
                            format!("{:>w$.3}", 100.0 * stats[k])
                        }
                    })
                    .collect();
                out.push(format!("{name:<width$}{}", cells.concat()));
            }
            out.push(String::new());
        }
        out.pop();
        out
    }

    /// Print [`summarize_lines`](Self::summarize_lines).
    pub fn summarize(&self) {
        for line in self.summarize_lines() {
            println!("{line}");
        }
    }

    /// The run as an [`EvalReport`]: the combined [`METRIC_NAMES`] values,
    /// each sequence's under its name in `per_group`, and the per-threshold
    /// curves with their `"alphas"` axis.
    ///
    /// # Errors
    ///
    /// As [`results`](Self::results).
    pub fn report(&self) -> Result<EvalReport> {
        let combined = self.results()?;
        let named = |stats: [f64; 44]| METRIC_NAMES.into_iter().zip(stats);
        let mut report = EvalReport::new("tracking", self.provenance())
            .with_metrics(named(combined.stats()))
            .with_params(serde_json::json!({
                "benchmark": self.config.benchmark,
                "clear_threshold": self.config.clear_threshold,
                "identity_threshold": self.config.identity_threshold,
                "n_sequences": self.sequences.len(),
                "num_alphas": NUM_ALPHAS,
            }));
        for (name, values) in combined.curves() {
            report = report.with_curve(name, values);
        }
        for (seq, r) in &self.sequences {
            for (metric, v) in named(r.stats()) {
                report = report.with_group_metric(seq.clone(), metric, v);
            }
        }
        Ok(report)
    }
}

/// TrackEval's summary tables: each metric's `summary_fields`.
const SUMMARY_TABLES: [(&str, &[&str]); 4] = [
    (
        "HOTA",
        &[
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
            "HOTA(0)",
            "LocA(0)",
            "HOTALocA(0)",
        ],
    ),
    (
        "CLEAR",
        &[
            "MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR", "sMOTA", "CLR_TP",
            "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag",
        ],
    ),
    ("Identity", &["IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"]),
    ("Count", &["Dets", "GT_Dets", "IDs", "GT_IDs"]),
];

/// Whether a [`METRIC_NAMES`] entry is a count, printed whole, rather than a
/// score, printed in percent.
fn is_count(name: &str) -> bool {
    matches!(
        name,
        "CLR_TP"
            | "CLR_FN"
            | "CLR_FP"
            | "IDSW"
            | "MT"
            | "PT"
            | "ML"
            | "Frag"
            | "CLR_Frames"
            | "IDTP"
            | "IDFN"
            | "IDFP"
            | "Dets"
            | "GT_Dets"
            | "IDs"
            | "GT_IDs"
    )
}
