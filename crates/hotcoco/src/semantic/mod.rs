//! The semantic segmentation metric family: mIoU, fwIoU, mAcc, aAcc, mDice,
//! and boundary IoU.
//!
//! [`SemanticEval`] is the driver — a port of mmsegmentation's `IoUMetric`,
//! checked against it, torchmetrics, and detectron2 on frozen label maps
//! (`tests/semantic.rs`, `scripts/gen_semantic_fixtures.py`), with boundary
//! IoU checked against the paper authors' kernel. It owns the control flow
//! only: reading label maps, applying a dataset's label convention, and
//! summing one confusion matrix and one set of boundary counts across
//! images. The formulas are [`crate::metrics::semantic`]'s.
//!
//! ```no_run
//! use std::path::Path;
//! use hotcoco::semantic::SemanticEval;
//! # fn main() -> hotcoco::error::Result<()> {
//! let mut ev = SemanticEval::new(150).reduce_zero_label(true);   // ADE20K
//! ev.update_folders(Path::new("annotations/validation"), Path::new("predictions"), &Default::default())?;
//! ev.summarize();                            // per-class scores, then the means
//! let miou = ev.scores().miou;
//! # Ok(())
//! # }
//! ```
//!
//! # Two ways in
//!
//! A training loop has the maps in memory: [`SemanticEval::update`] takes a
//! ground-truth map and a prediction of the same size, any integer type,
//! with the map's height and width, as often as there are batches. Saved
//! predictions are PNG files: [`SemanticEval::update_folders`] pairs the
//! ground-truth files with the predictions as a [`FolderPairing`] says — by
//! name at the top of the two folders by default, or by stem through
//! subfolders and suffixes for a layout such as Cityscapes' — and reads
//! both: 8- or 16-bit grayscale, or a palette image whose index is the
//! class. Any other pairing goes through [`SemanticEval::update_png`] on the
//! pairs the caller chooses.
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
//!
//! # Boundary IoU
//!
//! On by default, with the paper's band of 2% of the image diagonal:
//! [`SemanticEval::boundary_dilation`] sets the share or turns it off, which
//! leaves `mBIoU` the `-1.0` sentinel. It is the one metric here that is not
//! a function of the confusion matrix, so it needs each map's shape, which
//! `update` takes and a PNG file carries. [`crate::metrics::semantic::boundary`]
//! has the definition and the kernel.

mod png;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::metrics::is_computed;
pub use crate::metrics::semantic::LabelMap;
use crate::metrics::semantic::boundary::{BoundaryCounts, BoundaryScores, DEFAULT_DILATION_RATIO};
use crate::metrics::semantic::{ClassCounts, Label, LabelPairOp, PixelConfusion, SemanticScores};
use crate::report::{EvalReport, Provenance};

use self::png::read_class_map;

/// The headline metrics, in the order [`SemanticEval::stats`] reports them:
/// [`crate::metrics::semantic::METRIC_NAMES`] — the confusion-matrix means —
/// then `mBIoU`.
pub const METRIC_NAMES: [&str; 6] = ["mIoU", "fwIoU", "mAcc", "aAcc", "mDice", "mBIoU"];

/// One class in [`SemanticResults`]: its scores — the sentinel where not
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
    #[serde(rename = "Dice")]
    pub dice: f64,
    /// Boundary IoU, or the sentinel when it is off or the class has no
    /// boundary on either side.
    #[serde(rename = "BIoU")]
    pub biou: f64,
    #[serde(flatten)]
    pub counts: ClassCounts,
}

/// A run's results: the six means, every class, and what makes them
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
    #[serde(rename = "mDice")]
    pub mdice: f64,
    /// Mean boundary IoU, or the sentinel when boundary IoU is off.
    #[serde(rename = "mBIoU")]
    pub mbiou: f64,
    /// The boundary band as a share of the image diagonal; `None` when off.
    pub boundary_dilation: Option<f64>,
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

    /// The ground-truth reading these rules make: `None` skips the pixel.
    /// With `reduce_zero_label`, mmseg's relabel: 0 and 255 are unlabeled,
    /// the rest shift down, and the ignore index (255, see `check`) masks
    /// the shifted label.
    fn gt_label(self) -> impl Fn(u64) -> Option<u64> + Copy {
        move |g| {
            if self.reduce_zero_label {
                if g == 0 || g == 255 {
                    return None;
                }
                let shifted = g - 1;
                (shifted != 255).then_some(shifted)
            } else {
                (Some(g) != self.ignore_index).then_some(g)
            }
        }
    }

    /// Count one pair of `h × w` maps — or a batch of them, one after
    /// another — into `tally` under these rules: the one funnel every input
    /// form reaches, so the length checks live here.
    fn count_into<G: Label, P: Label>(
        self,
        tally: &mut Tally,
        gt: &[G],
        pred: &[P],
        h: usize,
        w: usize,
    ) -> crate::error::Result<()> {
        if gt.len() != pred.len() {
            return Err(Error::InvalidInput(format!(
                "gt and pred must have the same number of pixels, got {} and {}",
                gt.len(),
                pred.len()
            )));
        }
        let pixels = h * w;
        if gt.is_empty() {
            // An empty map counts as nothing, whatever shape it is called.
            return Ok(());
        }
        if pixels == 0 || gt.len() % pixels != 0 {
            return Err(Error::InvalidInput(format!(
                "{} pixels do not fill label maps of {h} × {w}",
                gt.len()
            )));
        }
        // The confusion matrix checks every label and counts nothing on an
        // error, so the boundary pass after it sees valid maps only.
        tally.confusion.update_with(gt, pred, self.gt_label())?;
        if let Some(boundary) = &mut tally.boundary {
            for (g, p) in gt.chunks_exact(pixels).zip(pred.chunks_exact(pixels)) {
                boundary.update(g, p, h, w, self.gt_label());
            }
        }
        Ok(())
    }

    /// Count one PNG pair into `tally`, each error naming the file.
    fn count_png_into(self, tally: &mut Tally, gt: &Path, pred: &Path) -> crate::error::Result<()> {
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
        self.count_maps_into(tally, &g.labels, &p.labels, g.h as usize, g.w as usize)
            .map_err(|e| {
                Error::InvalidInput(format!("{} against {}: {e}", gt.display(), pred.display()))
            })
    }

    /// [`count_into`](Self::count_into) for maps of run-time width.
    fn count_maps_into(
        self,
        tally: &mut Tally,
        gt: &LabelMap<'_>,
        pred: &LabelMap<'_>,
        h: usize,
        w: usize,
    ) -> crate::error::Result<()> {
        struct Count<'c>(LabelRules, &'c mut Tally, usize, usize);
        impl LabelPairOp for Count<'_> {
            type Output = crate::error::Result<()>;
            fn run<G: Label, P: Label>(self, gt: &[G], pred: &[P]) -> Self::Output {
                self.0.count_into(self.1, gt, pred, self.2, self.3)
            }
        }
        LabelMap::with_pair(gt, pred, Count(self, tally, h, w))
    }
}

/// What a pair counts into: the confusion matrix and, when boundary IoU is
/// on, the boundary counts. One per evaluator, and one per parallel split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Tally {
    confusion: PixelConfusion,
    boundary: Option<BoundaryCounts>,
}

impl Tally {
    fn new(num_classes: usize, boundary_dilation: Option<f64>) -> Self {
        Tally {
            confusion: PixelConfusion::new(num_classes),
            boundary: boundary_dilation.map(|ratio| BoundaryCounts::new(num_classes, ratio)),
        }
    }

    fn boundary_dilation(&self) -> Option<f64> {
        self.boundary.as_ref().map(BoundaryCounts::dilation_ratio)
    }

    fn add(&mut self, other: &Tally) {
        self.confusion.add(&other.confusion);
        if let (Some(a), Some(b)) = (&mut self.boundary, &other.boundary) {
            a.add(b);
        }
    }
}

/// The semantic segmentation evaluator: one confusion matrix, the boundary
/// counts beside it, and the rules that fill them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticEval {
    class_names: Vec<String>,
    #[serde(flatten)]
    rules: LabelRules,
    #[serde(flatten)]
    tally: Tally,
    n_images: usize,
}

impl SemanticEval {
    /// An evaluator for classes `0..num_classes`, named `class_<id>`, with
    /// the ignore index `255` and boundary IoU at the paper's band.
    pub fn new(num_classes: usize) -> Self {
        Self::with_class_names((0..num_classes).map(|c| format!("class_{c}")).collect())
    }

    /// An evaluator whose classes are `names`, in class-id order.
    pub fn with_class_names(names: Vec<String>) -> Self {
        let tally = Tally::new(names.len(), Some(DEFAULT_DILATION_RATIO));
        SemanticEval {
            class_names: names,
            rules: LabelRules {
                ignore_index: Some(255),
                reduce_zero_label: false,
            },
            tally,
            n_images: 0,
        }
    }

    /// Set the boundary IoU band as a share of the image diagonal — the
    /// paper's `0.02` by default — or `None` to leave boundary IoU out,
    /// which skips the pass and reports its sentinel. Resets the boundary
    /// counts so far.
    #[must_use]
    pub fn boundary_dilation(mut self, ratio: Option<f64>) -> Self {
        self.tally.boundary = ratio.map(|r| BoundaryCounts::new(self.class_names.len(), r));
        self
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

    /// The matrix behind every number but boundary IoU: row the ground-truth
    /// class, column the predicted class.
    pub fn confusion(&self) -> &PixelConfusion {
        &self.tally.confusion
    }

    /// The boundary counts behind `mBIoU`, or `None` when boundary IoU is off.
    pub fn boundary(&self) -> Option<&BoundaryCounts> {
        self.tally.boundary.as_ref()
    }

    /// The error the first `update` would raise for these options, if any:
    /// [`reduce_zero_label`](Self::reduce_zero_label) needs the ignore index
    /// `255`, and a boundary band must be a finite share of the diagonal, 0
    /// or more. The builder cannot fail, so a caller that wants the error
    /// before the first pair asks here.
    ///
    /// # Errors
    ///
    /// If `reduce_zero_label` is set with an ignore index other than `255`,
    /// or the boundary band is negative or not a number.
    pub fn check_options(&self) -> crate::error::Result<()> {
        self.rules.check()?;
        if let Some(ratio) = self.tally.boundary_dilation()
            && !(ratio >= 0.0 && ratio.is_finite())
        {
            return Err(Error::InvalidInput(format!(
                "boundary_dilation is a share of the image diagonal, 0 or more, not {ratio}"
            )));
        }
        Ok(())
    }

    /// Count one `h × w` ground-truth map against one prediction of the same
    /// size, both row-major — or a batch of such pairs, one map after
    /// another in each slice.
    ///
    /// # Errors
    ///
    /// If the two differ in length, if the length is not a multiple of
    /// `h * w`, or if a label is negative or at or above `num_classes`
    /// (after the ignore and `reduce_zero_label` rules, for the ground
    /// truth). On an error nothing from this pair is counted.
    pub fn update<G: Label, P: Label>(
        &mut self,
        gt: &[G],
        pred: &[P],
        h: usize,
        w: usize,
    ) -> crate::error::Result<()> {
        self.check_options()?;
        // A rejected pair leaves the counts as they were, so a pair counts
        // straight into the totals.
        self.rules.count_into(&mut self.tally, gt, pred, h, w)?;
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
        h: usize,
        w: usize,
    ) -> crate::error::Result<()> {
        self.check_options()?;
        self.rules
            .count_maps_into(&mut self.tally, gt, pred, h, w)?;
        self.n_images += 1;
        Ok(())
    }

    /// Fold counts summed apart — a parallel run's, another evaluator's —
    /// into the totals.
    fn commit(&mut self, tally: &Tally, images: usize) {
        self.tally.add(tally);
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
        self.check_options()?;
        self.rules.count_png_into(&mut self.tally, gt, pred)?;
        self.n_images += 1;
        Ok(())
    }

    /// Count every ground-truth file under `gt_folder` against its prediction
    /// under `pred_folder`, paired as `pairing` says, in parallel. Returns
    /// how many pairs were counted.
    ///
    /// # Errors
    ///
    /// If the ground-truth folder has no file ending in the suffix, if a
    /// ground-truth file has no prediction (every missing one is listed, up
    /// to five), if two predictions share a name, or on the first pair
    /// [`update_png`](Self::update_png) would reject. Nothing is counted on
    /// an error.
    pub fn update_folders(
        &mut self,
        gt_folder: &Path,
        pred_folder: &Path,
        pairing: &FolderPairing,
    ) -> crate::error::Result<usize> {
        self.check_options()?;
        let pairs = png_pairs(gt_folder, pred_folder, pairing)?;
        let fresh = || Tally::new(self.num_classes(), self.tally.boundary_dilation());
        let rules = self.rules;
        // One tally per rayon split rather than per image, so the reduction
        // is a handful of adds. Integer addition commutes, so the order the
        // splits meet in cannot change the counts.
        let total = pairs
            .par_iter()
            .try_fold(fresh, |mut acc, (gt, pred)| {
                rules.count_png_into(&mut acc, gt, pred)?;
                Ok::<_, crate::error::Error>(acc)
            })
            .try_reduce(fresh, |mut a, b| {
                a.add(&b);
                Ok(a)
            })?;
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
        if self.tally.boundary_dilation() != other.tally.boundary_dilation() {
            return Err(Error::InvalidInput(
                "cannot merge evaluators with different boundary_dilation settings".into(),
            ));
        }
        self.commit(&other.tally, other.n_images);
        Ok(())
    }

    /// The five confusion-matrix means and the per-class scores, from the
    /// counts so far.
    pub fn scores(&self) -> SemanticScores {
        self.tally.confusion.scores()
    }

    /// Boundary IoU per class and its mean, or `None` when boundary IoU is off.
    pub fn boundary_scores(&self) -> Option<BoundaryScores> {
        self.tally.boundary.as_ref().map(BoundaryCounts::scores)
    }

    /// The six headline values, in [`METRIC_NAMES`] order.
    pub fn stats(&self) -> [f64; 6] {
        let [a, b, c, d, e] = self.scores().stats();
        let mbiou = self.boundary_scores().map_or(-1.0, |b| b.mbiou);
        [a, b, c, d, e, mbiou]
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

    /// Boundary IoU per class, the sentinel throughout when it is off.
    fn biou_per_class(&self) -> Vec<f64> {
        self.boundary_scores()
            .map_or_else(|| vec![-1.0; self.num_classes()], |b| b.per_class)
    }

    /// The run as a struct — see [`SemanticResults`].
    pub fn results(&self) -> SemanticResults {
        let scores = self.scores();
        let counts = self.tally.confusion.all_class_counts();
        let [.., mbiou] = self.stats();
        SemanticResults {
            hotcoco_version: env!("CARGO_PKG_VERSION").to_string(),
            provenance: self.provenance(),
            reference_deviations: self.reference_deviations(),
            miou: scores.miou,
            fwiou: scores.fwiou,
            macc: scores.macc,
            aacc: scores.aacc,
            mdice: scores.mdice,
            mbiou,
            boundary_dilation: self.tally.boundary_dilation(),
            per_class: self
                .class_names
                .iter()
                .zip(&scores.per_class)
                .zip(self.biou_per_class())
                .zip(counts)
                .map(|(((name, s), biou), counts)| ClassResult {
                    name: name.clone(),
                    iou: s.iou,
                    acc: s.acc,
                    dice: s.dice,
                    biou,
                    counts,
                })
                .collect(),
            n_images: self.n_images,
            n_pixels: self.tally.confusion.total(),
        }
    }

    /// The summary as lines: one row per class with its IoU, Acc, Dice, and
    /// boundary IoU in percent (`-` where not computed), then the six means
    /// — mmseg's table, with two columns more.
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
        let mut out = vec![format!("{:<width$} |    IoU    Acc   Dice   BIoU", "Class")];
        out.push("-".repeat(width + 31));
        for ((name, s), biou) in self
            .class_names
            .iter()
            .zip(&scores.per_class)
            .zip(self.biou_per_class())
        {
            out.push(format!(
                "{name:<width$} | {} {} {} {}",
                pct(s.iou),
                pct(s.acc),
                pct(s.dice),
                pct(biou)
            ));
        }
        out.push("-".repeat(width + 31));
        let means = METRIC_NAMES
            .iter()
            .zip(self.stats())
            .map(|(name, v)| format!("{name} {}", pct(v).trim()))
            .collect::<Vec<_>>()
            .join("  ");
        out.push(format!(
            "{means}   ({} images, {} pixels)",
            self.n_images,
            self.tally.confusion.total()
        ));
        out
    }

    /// Print [`summarize_lines`](Self::summarize_lines).
    pub fn summarize(&self) {
        for line in self.summarize_lines() {
            println!("{line}");
        }
    }

    /// The run as an [`EvalReport`]: the headline metrics — `mBIoU` only when
    /// boundary IoU is on — `IoU`, `Acc`, `Dice`, and `BIoU` per class name
    /// for the classes that have them, no groups, no curves.
    pub fn report(&self) -> EvalReport {
        let report =
            EvalReport::new("semantic", self.provenance()).with_params(serde_json::json!({
                "n_images": self.n_images,
                "n_pixels": self.tally.confusion.total(),
                "num_classes": self.num_classes(),
                "ignore_index": self.rules.ignore_index,
                "reduce_zero_label": self.rules.reduce_zero_label,
                "boundary_dilation": self.tally.boundary_dilation(),
            }));
        let mut report = self
            .scores()
            .add_to_report(report, self.class_names.iter().cloned());
        if let Some(b) = self.boundary_scores() {
            report = report.with_metrics([("mBIoU", b.mbiou)]);
            for (name, biou) in self.class_names.iter().zip(b.per_class) {
                if is_computed(biou) {
                    report = report.with_class_metric(name.clone(), "BIoU", biou);
                }
            }
        }
        report
    }
}

/// How [`SemanticEval::update_folders`] pairs a ground-truth file with its
/// prediction. A ground-truth file is one under `gt_folder` — at its top, or
/// anywhere below it with `recursive` — whose name ends in `gt_suffix`; its
/// key is the name with the suffix removed, and its prediction is the file
/// named `key + pred_suffix` under `pred_folder`, found by the same walk,
/// wherever it sits. Suffixes compare without regard to case, so `.PNG`
/// pairs with `.png`. The default pairs the `.png` files at the top of each
/// folder by name.
///
/// Cityscapes keeps a folder per city and several files per frame, so its
/// ground truth is `recursive: true` with `gt_suffix:
/// "_gtFine_labelTrainIds.png"`, and `pred_suffix` is whatever the
/// predictions were saved under — mmsegmentation's `seg_map_suffix` and
/// `img_suffix`, the same rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FolderPairing {
    /// Walk the folders' subfolders too.
    pub recursive: bool,
    /// What a ground-truth file's name ends in.
    pub gt_suffix: String,
    /// What its prediction's name ends in.
    pub pred_suffix: String,
}

impl Default for FolderPairing {
    fn default() -> Self {
        FolderPairing {
            recursive: false,
            gt_suffix: ".png".into(),
            pred_suffix: ".png".into(),
        }
    }
}

/// The files of `folder` — at its top, or anywhere below it with `recursive`.
fn list_files(folder: &Path, recursive: bool, side: &str) -> crate::error::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut folders = vec![folder.to_path_buf()];
    while let Some(dir) = folders.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| {
            Error::io(
                e,
                format_args!("cannot read {side} folder {}", dir.display()),
            )
        })?;
        for path in entries.flatten().map(|entry| entry.path()) {
            if path.is_dir() {
                if recursive {
                    folders.push(path);
                }
            } else {
                files.push(path);
            }
        }
    }
    Ok(files)
}

/// The ground-truth files under `gt_folder`, each paired with its prediction
/// under `pred_folder` as `pairing` says, ordered by key.
fn png_pairs(
    gt_folder: &Path,
    pred_folder: &Path,
    pairing: &FolderPairing,
) -> crate::error::Result<Vec<(PathBuf, PathBuf)>> {
    let (gt_suffix, pred_suffix) = (
        pairing.gt_suffix.to_lowercase(),
        pairing.pred_suffix.to_lowercase(),
    );
    // A file's key: its name without the suffix, lowercased on both sides;
    // `None` for a name that does not end in the suffix.
    let key = |p: &Path, suffix: &str| -> Option<String> {
        let name = p.file_name()?.to_string_lossy().to_lowercase();
        name.strip_suffix(suffix).map(str::to_owned)
    };
    let mut gts: Vec<(String, PathBuf)> = list_files(gt_folder, pairing.recursive, "ground-truth")?
        .into_iter()
        .filter_map(|p| key(&p, &gt_suffix).map(|k| (k, p)))
        .collect();
    gts.sort();
    if gts.is_empty() {
        return Err(Error::InvalidInput(format!(
            "no files ending in {} in ground-truth folder {}{}",
            pairing.gt_suffix,
            gt_folder.display(),
            if pairing.recursive {
                ""
            } else {
                " (its subfolders are not read without `recursive`)"
            }
        )));
    }
    let mut preds: BTreeMap<String, PathBuf> = BTreeMap::new();
    for p in list_files(pred_folder, pairing.recursive, "prediction")? {
        let Some(k) = key(&p, &pred_suffix) else {
            continue;
        };
        if let Some(first) = preds.insert(k.clone(), p.clone()) {
            return Err(Error::InvalidInput(format!(
                "two prediction files have the name {k}{}: {} and {}",
                pairing.pred_suffix,
                first.display(),
                p.display()
            )));
        }
    }
    let mut pairs = Vec::with_capacity(gts.len());
    let mut missing = Vec::new();
    for (k, gt) in gts {
        match preds.get(&k) {
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
            "{} ground-truth file(s) have no prediction ending in {} in {}: {shown}{more}",
            missing.len(),
            pairing.pred_suffix,
            pred_folder.display()
        )));
    }
    Ok(pairs)
}
