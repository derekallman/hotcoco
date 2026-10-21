//! The semantic segmentation metric family: mIoU, fwIoU, mAcc, and aAcc.
//!
//! [`SemanticEval`] is the driver — a port of mmsegmentation's `IoUMetric`,
//! checked against it, torchmetrics, and detectron2 on frozen label maps
//! (`tests/semantic.rs`, `scripts/gen_semantic_fixtures.py`). It owns the
//! control flow only: reading label maps, applying a dataset's label
//! convention, and summing one confusion matrix across images. The formulas
//! are [`crate::metrics::semantic`]'s.
//!
//! ```no_run
//! use std::path::Path;
//! use hotcoco::semantic::SemanticEval;
//! # fn main() -> hotcoco::error::Result<()> {
//! let mut ev = SemanticEval::new(150).reduce_zero_label(true);   // ADE20K
//! ev.update_folders(Path::new("annotations/validation"), Path::new("predictions"))?;
//! ev.summarize();                            // per-class IoU and Acc, then the four means
//! let miou = ev.scores().miou;
//! # Ok(())
//! # }
//! ```
//!
//! # Two ways in
//!
//! A training loop has the maps in memory: [`SemanticEval::update`] takes a
//! ground-truth map and a prediction of the same size, any integer type, as
//! often as there are batches. Saved predictions are PNG files beside the
//! ground truth's: [`SemanticEval::update_folders`] pairs the PNG files at
//! the top of the two folders by file name and reads both: 8- or 16-bit
//! grayscale, or a palette image whose index is the class. A dataset laid
//! out otherwise (Cityscapes keeps a folder per city and several PNG files
//! per frame) goes through [`SemanticEval::update_png`] on the pairs the
//! caller chooses.
//! Either way the state is one `num_classes × num_classes` matrix, so two
//! evaluators over disjoint shards [`merge`](SemanticEval::merge) into the
//! whole-split result.
//!
//! # Labels the reference ignores
//!
//! A ground-truth pixel equal to [`ignore_index`](SemanticEval::ignore_index)
//! — `255` by default, as in mmseg, Cityscapes, and VOC — is left out on both
//! sides. [`reduce_zero_label`](SemanticEval::reduce_zero_label) is mmseg's
//! `LoadAnnotations` rule for datasets whose files put "other" at `0`: `0`
//! becomes the ignore index and every other label shifts down by one, on the
//! ground truth only. Every other label must be below `num_classes`, on
//! either side; one that is not is an error naming the pixel.

mod png;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::metrics::is_computed;
use crate::metrics::semantic::{ClassCounts, Label, LabelPairOp, PixelConfusion, SemanticScores};
pub use crate::metrics::semantic::{LabelMap, METRIC_NAMES};
use crate::report::{EvalReport, Provenance};

use self::png::read_class_map;

/// One class in [`SemanticResults`]: its two scores — the sentinel where not
/// computed — beside the counts they came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ClassResult {
    /// The class name, or `class_<id>` when none was given.
    pub name: String,
    #[serde(rename = "IoU")]
    pub iou: f64,
    #[serde(rename = "Acc")]
    pub acc: f64,
    #[serde(flatten)]
    pub counts: ClassCounts,
}

/// A run's results: the four means, every class, and what makes them
/// auditable. `save()` writes it and the CLIs emit it.
///
/// `#[non_exhaustive]`: an output DTO that callers read, never construct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SemanticResults {
    /// hotcoco version that produced these results.
    pub hotcoco_version: String,
    pub provenance: Provenance,
    pub reference_deviations: Vec<String>,
    #[serde(rename = "mIoU")]
    pub miou: f64,
    #[serde(rename = "fwIoU")]
    pub fwiou: f64,
    #[serde(rename = "mAcc")]
    pub macc: f64,
    #[serde(rename = "aAcc")]
    pub aacc: f64,
    /// Every class, by class id.
    pub per_class: Vec<ClassResult>,
    /// Label-map pairs counted.
    pub n_images: usize,
    /// Pixels counted, after ignoring.
    pub n_pixels: u64,
}

impl SemanticResults {
    /// Serialize to a pretty-printed JSON string.
    pub fn to_json(&self) -> crate::error::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Write as pretty-printed JSON to a file.
    pub fn save(&self, path: &Path) -> crate::error::Result<()> {
        crate::report::save_json(self, path)
    }
}

/// The label rules apart from the matrix they fill, so a pair can count
/// into the evaluator's own matrix or into a parallel split's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct LabelRules {
    ignore_index: Option<u64>,
    reduce_zero_label: bool,
}

impl LabelRules {
    /// The one configuration that has no reference meaning. mmseg's
    /// `LoadAnnotations` relabel hardcodes 255 — `0 → 255`, shift down,
    /// `254 → 255` — and `IoUMetric` masks `ignore_index` on the shifted
    /// label. With any other ignore index the relabeled 255 is neither masked
    /// nor in range, and what mmseg does with it is an accident of
    /// `torch.histc`, so there is nothing to be faithful to.
    fn check(self) -> crate::error::Result<()> {
        if self.reduce_zero_label && self.ignore_index != Some(255) {
            return Err(Error::InvalidInput(format!(
                "reduce_zero_label follows mmseg's relabel, which marks unlabeled pixels \
                 255; it needs ignore_index 255 (the default), not {:?}",
                self.ignore_index
            )));
        }
        Ok(())
    }

    /// Count one pair into `counts` under these rules: the one funnel every
    /// input form reaches, so the length check lives here.
    fn count_into<G: Label, P: Label>(
        self,
        counts: &mut PixelConfusion,
        gt: &[G],
        pred: &[P],
    ) -> crate::error::Result<()> {
        if gt.len() != pred.len() {
            return Err(Error::InvalidInput(format!(
                "gt and pred must have the same number of pixels, got {} and {}",
                gt.len(),
                pred.len()
            )));
        }
        if self.reduce_zero_label {
            // mmseg's relabel: 0 and 255 are unlabeled, the rest shift down, and
            // the ignore index (255, see `check`) masks the shifted label.
            counts.update_with(gt, pred, |g| {
                if g == 0 || g == 255 {
                    return None;
                }
                let shifted = g - 1;
                (shifted != 255).then_some(shifted)
            })
        } else {
            counts.update(gt, pred, self.ignore_index)
        }
    }

    /// Count one PNG pair into `counts`, each error naming the file.
    fn count_png_into(
        self,
        counts: &mut PixelConfusion,
        gt: &Path,
        pred: &Path,
    ) -> crate::error::Result<()> {
        let g = read_class_map(gt)?;
        let p = read_class_map(pred)?;
        if (g.h, g.w) != (p.h, p.w) {
            return Err(Error::InvalidInput(format!(
                "{} is {}×{} but {} is {}×{}; a prediction must be the size of its ground truth",
                gt.display(),
                g.w,
                g.h,
                pred.display(),
                p.w,
                p.h
            )));
        }
        // A label error names its side, `gt pixel 3` or `pred pixel 3`, so
        // both files are named and the side says which one to fix.
        self.count_maps_into(counts, &g.labels, &p.labels)
            .map_err(|e| {
                Error::InvalidInput(format!("{} against {}: {e}", gt.display(), pred.display()))
            })
    }

    /// [`count_into`](Self::count_into) for maps of run-time width.
    fn count_maps_into(
        self,
        counts: &mut PixelConfusion,
        gt: &LabelMap<'_>,
        pred: &LabelMap<'_>,
    ) -> crate::error::Result<()> {
        struct Count<'c>(LabelRules, &'c mut PixelConfusion);
        impl LabelPairOp for Count<'_> {
            type Output = crate::error::Result<()>;
            fn run<G: Label, P: Label>(self, gt: &[G], pred: &[P]) -> Self::Output {
                self.0.count_into(self.1, gt, pred)
            }
        }
        LabelMap::with_pair(gt, pred, Count(self, counts))
    }
}

/// The semantic segmentation evaluator: one confusion matrix and the rules
/// that fill it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticEval {
    class_names: Vec<String>,
    #[serde(flatten)]
    rules: LabelRules,
    confusion: PixelConfusion,
    n_images: usize,
}

impl SemanticEval {
    /// An evaluator for classes `0..num_classes`, named `class_<id>`, with
    /// the ignore index `255`.
    pub fn new(num_classes: usize) -> Self {
        Self::with_class_names((0..num_classes).map(|c| format!("class_{c}")).collect())
    }

    /// An evaluator whose classes are `names`, in class-id order.
    pub fn with_class_names(names: Vec<String>) -> Self {
        let confusion = PixelConfusion::new(names.len());
        SemanticEval {
            class_names: names,
            rules: LabelRules {
                ignore_index: Some(255),
                reduce_zero_label: false,
            },
            confusion,
            n_images: 0,
        }
    }

    /// Set the ground-truth label that marks a pixel as unlabeled, or `None`
    /// to count every pixel. A value below `num_classes` is allowed and makes
    /// that class unscorable, as in the references.
    #[must_use]
    pub fn ignore_index(mut self, ignore_index: Option<u64>) -> Self {
        self.rules.ignore_index = ignore_index;
        self
    }

    /// Apply mmseg's `reduce_zero_label`: ground-truth `0` (and `255`, the
    /// value mmseg relabels it to) is unlabeled and every other ground-truth
    /// label shifts down by one, so a file with `num_classes + 1` labels scores
    /// `num_classes` classes. [`ignore_index`](Self::ignore_index) then applies
    /// to the shifted label, as `IoUMetric` applies it.
    #[must_use]
    pub fn reduce_zero_label(mut self, reduce: bool) -> Self {
        self.rules.reduce_zero_label = reduce;
        self
    }

    /// The class names, by class id.
    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }

    /// The side of the confusion matrix.
    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }

    /// Label-map pairs counted so far.
    pub fn n_images(&self) -> usize {
        self.n_images
    }

    /// The matrix behind every number: row the ground-truth class, column
    /// the predicted class.
    pub fn confusion(&self) -> &PixelConfusion {
        &self.confusion
    }

    /// The error the first `update` would raise for these options, if any:
    /// [`reduce_zero_label`](Self::reduce_zero_label) needs the ignore index
    /// `255`. The builder cannot fail, so a caller that wants the error
    /// before the first pair asks here.
    ///
    /// # Errors
    ///
    /// If `reduce_zero_label` is set with an ignore index other than `255`.
    pub fn check_options(&self) -> crate::error::Result<()> {
        self.rules.check()
    }

    /// Count one ground-truth map against one prediction of the same size,
    /// each flat in any pixel order so long as it is the same for both.
    ///
    /// # Errors
    ///
    /// If the two differ in length, or a label is negative or at or above
    /// `num_classes` (after the ignore and `reduce_zero_label` rules, for
    /// the ground truth). On an error nothing from this pair is counted.
    pub fn update<G: Label, P: Label>(&mut self, gt: &[G], pred: &[P]) -> crate::error::Result<()> {
        self.rules.check()?;
        // A rejected pair leaves the matrix as it was, so a pair counts
        // straight into the totals.
        self.rules.count_into(&mut self.confusion, gt, pred)?;
        self.n_images += 1;
        Ok(())
    }

    /// [`update`](Self::update) for two maps whose widths are known only at
    /// run time, such as numpy arrays.
    ///
    /// # Errors
    ///
    /// As [`update`](Self::update).
    pub fn update_maps(
        &mut self,
        gt: &LabelMap<'_>,
        pred: &LabelMap<'_>,
    ) -> crate::error::Result<()> {
        self.rules.check()?;
        self.rules.count_maps_into(&mut self.confusion, gt, pred)?;
        self.n_images += 1;
        Ok(())
    }

    /// Fold counts summed apart — a parallel run's, another evaluator's —
    /// into the totals.
    fn commit(&mut self, counts: &PixelConfusion, images: usize) {
        self.confusion.add(counts);
        self.n_images += images;
    }

    /// Count one PNG pair: 8- or 16-bit grayscale, or a palette image whose index
    /// is the class, as Cityscapes, ADE20K, and VOC store them. An RGB file is
    /// refused.
    ///
    /// # Errors
    ///
    /// A file that cannot be read or is not a label map, two maps of
    /// different sizes, or a label out of range, each naming the file.
    pub fn update_png(&mut self, gt: &Path, pred: &Path) -> crate::error::Result<()> {
        self.rules.check()?;
        self.rules.count_png_into(&mut self.confusion, gt, pred)?;
        self.n_images += 1;
        Ok(())
    }

    /// Count every PNG in `gt_folder` against the file of the same name in
    /// `pred_folder`, in parallel. Returns how many pairs were counted.
    ///
    /// # Errors
    ///
    /// If the ground-truth folder has no PNG files, if a ground-truth file
    /// has no prediction of the same name (every missing name is listed, up
    /// to five), or on the first pair [`update_png`](Self::update_png) would
    /// reject. Nothing is counted on an error.
    pub fn update_folders(
        &mut self,
        gt_folder: &Path,
        pred_folder: &Path,
    ) -> crate::error::Result<usize> {
        self.rules.check()?;
        let pairs = png_pairs(gt_folder, pred_folder)?;
        let n = self.num_classes();
        let rules = self.rules;
        // One matrix per rayon split rather than per image, so the reduction
        // is a handful of adds. Integer addition commutes, so the order the
        // splits meet in cannot change the counts.
        let total = pairs
            .par_iter()
            .try_fold(
                || PixelConfusion::new(n),
                |mut acc, (gt, pred)| {
                    rules.count_png_into(&mut acc, gt, pred)?;
                    Ok::<_, crate::error::Error>(acc)
                },
            )
            .try_reduce(
                || PixelConfusion::new(n),
                |mut a, b| {
                    a.add(&b);
                    Ok(a)
                },
            )?;
        self.commit(&total, pairs.len());
        Ok(pairs.len())
    }

    /// Add another evaluator's counts into this one — shards of a
    /// distributed run, or two halves of a split.
    ///
    /// # Errors
    ///
    /// If the two were built for different classes or label rules.
    pub fn merge(&mut self, other: &SemanticEval) -> crate::error::Result<()> {
        if self.num_classes() != other.num_classes() {
            return Err(Error::InvalidInput(format!(
                "cannot merge evaluators with different classes ({} and {} classes)",
                self.num_classes(),
                other.num_classes()
            )));
        }
        if let Some((i, (a, b))) = self
            .class_names
            .iter()
            .zip(&other.class_names)
            .enumerate()
            .find(|(_, (a, b))| a != b)
        {
            return Err(Error::InvalidInput(format!(
                "cannot merge evaluators with different class names: class {i} is {a:?} in one and {b:?} in the other"
            )));
        }
        if self.rules != other.rules {
            return Err(Error::InvalidInput(
                "cannot merge evaluators with different ignore_index or \
                 reduce_zero_label settings"
                    .into(),
            ));
        }
        self.commit(&other.confusion, other.n_images);
        Ok(())
    }

    /// The four means and the per-class pair, from the counts so far.
    pub fn scores(&self) -> SemanticScores {
        self.confusion.scores()
    }

    /// The four headline values, in [`METRIC_NAMES`] order.
    pub fn stats(&self) -> [f64; 4] {
        self.scores().stats()
    }

    /// Ways this run departs from what the reference would compute.
    ///
    /// Empty today, and kept as the one predicate [`provenance`](Self::provenance)
    /// reads so a future condition lands in both places at once. Every
    /// option here — the ignore index, `reduce_zero_label` — is a reference
    /// option with the reference's meaning, not a departure from it.
    pub fn reference_deviations(&self) -> Vec<String> {
        Vec::new()
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

    /// The run as a struct — see [`SemanticResults`].
    pub fn results(&self) -> SemanticResults {
        let scores = self.scores();
        let counts = self.confusion.all_class_counts();
        SemanticResults {
            hotcoco_version: env!("CARGO_PKG_VERSION").to_string(),
            provenance: self.provenance(),
            reference_deviations: self.reference_deviations(),
            miou: scores.miou,
            fwiou: scores.fwiou,
            macc: scores.macc,
            aacc: scores.aacc,
            per_class: self
                .class_names
                .iter()
                .zip(&scores.per_class)
                .zip(counts)
                .map(|((name, s), counts)| ClassResult {
                    name: name.clone(),
                    iou: s.iou,
                    acc: s.acc,
                    counts,
                })
                .collect(),
            n_images: self.n_images,
            n_pixels: self.confusion.total(),
        }
    }

    /// The summary as lines: one row per class with its IoU and Acc in
    /// percent (`-` where not computed), then the four means — mmseg's table.
    pub fn summarize_lines(&self) -> Vec<String> {
        let scores = self.scores();
        let pct = |v: f64| {
            if is_computed(v) {
                format!("{:6.2}", 100.0 * v)
            } else {
                format!("{:>6}", "-")
            }
        };
        let width = self
            .class_names
            .iter()
            .map(|n| n.chars().count())
            .max()
            .unwrap_or(5)
            .max(5);
        let mut out = vec![format!("{:<width$} |    IoU    Acc", "Class")];
        out.push("-".repeat(width + 17));
        for (name, s) in self.class_names.iter().zip(&scores.per_class) {
            out.push(format!("{name:<width$} | {} {}", pct(s.iou), pct(s.acc)));
        }
        out.push("-".repeat(width + 17));
        out.push(format!(
            "mIoU {}  fwIoU {}  mAcc {}  aAcc {}   ({} images, {} pixels)",
            pct(scores.miou).trim(),
            pct(scores.fwiou).trim(),
            pct(scores.macc).trim(),
            pct(scores.aacc).trim(),
            self.n_images,
            self.confusion.total()
        ));
        out
    }

    /// Print [`summarize_lines`](Self::summarize_lines).
    pub fn summarize(&self) {
        for line in self.summarize_lines() {
            println!("{line}");
        }
    }

    /// The run as an [`EvalReport`]: the four headline metrics, `IoU` and
    /// `Acc` per class name for the classes that have them, no groups, no
    /// curves.
    pub fn report(&self) -> EvalReport {
        let report =
            EvalReport::new("semantic", self.provenance()).with_params(serde_json::json!({
                "n_images": self.n_images,
                "n_pixels": self.confusion.total(),
                "num_classes": self.num_classes(),
                "ignore_index": self.rules.ignore_index,
                "reduce_zero_label": self.rules.reduce_zero_label,
            }));
        self.scores()
            .add_to_report(report, self.class_names.iter().cloned())
    }
}

/// The PNG files of `gt_folder`, sorted by name, each paired with the file
/// of the same name in `pred_folder`.
fn png_pairs(
    gt_folder: &Path,
    pred_folder: &Path,
) -> crate::error::Result<Vec<(PathBuf, PathBuf)>> {
    let is_png = |p: &Path| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"));
    let mut gts: Vec<PathBuf> = std::fs::read_dir(gt_folder)
        .map_err(|e| {
            Error::io(
                e,
                format_args!("cannot read ground-truth folder {}", gt_folder.display()),
            )
        })?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| is_png(p))
        .collect();
    gts.sort();
    if gts.is_empty() {
        return Err(Error::InvalidInput(format!(
            "no PNG files in ground-truth folder {}",
            gt_folder.display()
        )));
    }
    // Names compare case-insensitively on both sides, since the extension
    // filter above accepts `.PNG` as well as `.png`.
    let key = |p: &Path| p.file_name().map(|n| n.to_string_lossy().to_lowercase());
    let pred_names: BTreeMap<String, PathBuf> = std::fs::read_dir(pred_folder)
        .map_err(|e| {
            Error::io(
                e,
                format_args!("cannot read prediction folder {}", pred_folder.display()),
            )
        })?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter_map(|p| key(&p).map(|k| (k, p)))
        .collect();
    let mut pairs = Vec::with_capacity(gts.len());
    let mut missing = Vec::new();
    for gt in gts {
        let name = key(&gt).expect("a read_dir entry has a name");
        match pred_names.get(&name) {
            Some(pred) => pairs.push((gt, pred.clone())),
            None => missing.push(
                gt.file_name()
                    .expect("a name")
                    .to_string_lossy()
                    .into_owned(),
            ),
        }
    }
    if !missing.is_empty() {
        let shown = missing
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let more = if missing.len() > 5 {
            format!(" and {} more", missing.len() - 5)
        } else {
            String::new()
        };
        return Err(Error::InvalidInput(format!(
            "{} ground-truth file(s) have no prediction of the same name in {}: {shown}{more}",
            missing.len(),
            pred_folder.display()
        )));
    }
    Ok(pairs)
}
