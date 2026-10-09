//! Semantic segmentation — per-class IoU and accuracy from a pixel confusion
//! matrix, and the four means over them.
//!
//! The whole metric family is a function of one matrix: [`PixelConfusion`],
//! `num_classes × num_classes` counts of pixels with ground-truth class `g`
//! predicted as `p`, summed over every image. Pixels are matched by position,
//! so there is no matcher to call; this module builds the matrix and reads
//! the numbers off it:
//!
//! ```text
//! IoU_c  = TP_c / (TP_c + FP_c + FN_c)       per class, when the class is on either side
//! Acc_c  = TP_c / (TP_c + FN_c)              per class, when the class has ground truth
//! mIoU   = mean of IoU_c over the classes that have one
//! mAcc   = mean of Acc_c over the classes that have one
//! fwIoU  = Σ IoU_c · (TP_c + FN_c) / Σ (TP_c + FN_c)
//! aAcc   = Σ TP_c / Σ (TP_c + FN_c)          overall pixel accuracy
//! ```
//!
//! The reference is mmsegmentation's `IoUMetric`, whose conventions torchmetrics'
//! `multiclass_jaccard_index` shares: a pixel whose ground truth is the ignore
//! index is left out on both sides, and a class with no ground-truth pixel and
//! no predicted pixel has no IoU and is left out of the mean. One corner is a
//! choice between references, and this module follows those two rather than
//! detectron2's `SemSegEvaluator`: a class predicted somewhere but absent from
//! every ground-truth map has IoU `0` and counts toward `mIoU`, where detectron2
//! drops it. `fwIoU` is detectron2's; mmseg has none. The pinned sources and the
//! frozen comparison are `scripts/gen_semantic_fixtures.py`.
//!
//! ```
//! use hotcoco::metrics::semantic::PixelConfusion;
//!
//! let mut m = PixelConfusion::new(3);
//! m.update(&[0u8, 0, 1, 1], &[0u8, 0, 1, 2], None).unwrap();
//! let s = m.scores();
//! assert_eq!(s.per_class[0].iou, 1.0);
//! assert_eq!(s.per_class[1].iou, 0.5);   // one of two pixels, no false positive
//! assert_eq!(s.per_class[2].iou, 0.0);   // predicted once, never ground truth
//! assert_eq!(s.miou, 0.5);
//! ```

use serde::{Deserialize, Serialize};

use super::{is_computed, mean_or_missing};
use crate::primitives::sim::iou_from_counts;
use crate::report::EvalReport;

/// The headline metrics, in the order [`SemanticScores::stats`] reports them.
pub const METRIC_NAMES: [&str; 4] = ["mIoU", "fwIoU", "mAcc", "aAcc"];

/// An integer label type a label map can hold: every integer primitive numpy
/// stores a class-id map in.
pub trait Label: Copy {
    /// The label as an index, or `None` for a negative value.
    fn index(self) -> Option<u64>;
}

impl<T: Copy + TryInto<u64>> Label for T {
    #[inline]
    fn index(self) -> Option<u64> {
        self.try_into().ok()
    }
}

/// The pixel counts of one class: its diagonal cell, its row, and its column.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassCounts {
    /// Pixels of this class predicted as this class.
    pub tp: u64,
    /// Ground-truth pixels of this class, counted pixels only: `TP + FN`.
    pub gt_area: u64,
    /// Pixels predicted as this class: `TP + FP`.
    pub pred_area: u64,
}

impl ClassCounts {
    /// IoU, or `None` when the class appears on neither side.
    pub fn iou(&self) -> Option<f64> {
        (self.gt_area + self.pred_area > 0)
            .then(|| iou_from_counts(self.tp, self.gt_area, self.pred_area))
    }

    /// Per-class accuracy — recall over the class's pixels — or `None` when
    /// the class has no ground truth.
    pub fn acc(&self) -> Option<f64> {
        (self.gt_area > 0).then(|| self.tp as f64 / self.gt_area as f64)
    }
}

/// One class's two numbers, each `-1.0` where it is not computed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClassScores {
    /// IoU, or the sentinel when the class is on neither side.
    pub iou: f64,
    /// Accuracy, or the sentinel when the class has no ground truth.
    pub acc: f64,
}

/// Everything a confusion matrix yields: the four means and the per-class pair.
///
/// Each mean is the `-1.0` sentinel when nothing fed it — a split with every
/// pixel ignored, for instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticScores {
    /// Mean IoU over the classes that have one.
    pub miou: f64,
    /// IoU weighted by each class's share of the ground-truth pixels.
    pub fwiou: f64,
    /// Mean per-class accuracy over the classes with ground truth.
    pub macc: f64,
    /// Overall pixel accuracy.
    pub aacc: f64,
    /// `IoU` and `Acc` per class, indexed by class id.
    pub per_class: Vec<ClassScores>,
}

impl SemanticScores {
    /// Whether `mIoU` is a number rather than the sentinel.
    pub fn is_computed(&self) -> bool {
        is_computed(self.miou)
    }

    /// The four means, in [`METRIC_NAMES`] order.
    pub fn stats(&self) -> [f64; 4] {
        [self.miou, self.fwiou, self.macc, self.aacc]
    }

    /// Record these scores on a report: the four means as metrics, and `IoU`
    /// and `Acc` per class under `class_names` (in class order) where
    /// computed. The one place a semantic block is written, for every family
    /// that reports one.
    #[must_use]
    pub fn add_to_report<S: Into<String>>(
        &self,
        mut report: EvalReport,
        class_names: impl IntoIterator<Item = S>,
    ) -> EvalReport {
        report = report.with_metrics(METRIC_NAMES.iter().copied().zip(self.stats()));
        for (name, s) in class_names.into_iter().zip(&self.per_class) {
            let name = name.into();
            if is_computed(s.iou) {
                report = report.with_class_metric(name.clone(), "IoU", s.iou);
            }
            if is_computed(s.acc) {
                report = report.with_class_metric(name, "Acc", s.acc);
            }
        }
        report
    }
}

/// Pixel counts by `(ground-truth class, predicted class)`, summed over images.
///
/// Row-major, rows the ground truth. Built by [`update`](Self::update) one
/// label-map pair at a time, or taken whole through
/// [`from_counts`](Self::from_counts) when another tool already produced it.
/// Two matrices of the same shape add, which is how shards of a distributed
/// evaluation combine.
///
/// A matrix made with [`with_unlabeled_pred`](Self::with_unlabeled_pred) has
/// one column more than it has classes: a prediction that names no class —
/// the void of a panoptic output — which counts as a miss for its
/// ground-truth class and a false positive for none. That column is not a
/// class: it has no row, no scores, and no place in any mean.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelConfusion {
    num_classes: usize,
    unlabeled_pred: bool,
    counts: Vec<u64>,
}

impl PixelConfusion {
    /// An all-zero `num_classes × num_classes` matrix.
    pub fn new(num_classes: usize) -> Self {
        PixelConfusion {
            num_classes,
            unlabeled_pred: false,
            counts: vec![0; num_classes * num_classes],
        }
    }

    /// An all-zero `num_classes × (num_classes + 1)` matrix whose last column
    /// is a prediction that names no class — see the type docs.
    pub fn with_unlabeled_pred(num_classes: usize) -> Self {
        PixelConfusion {
            num_classes,
            unlabeled_pred: true,
            counts: vec![0; num_classes * (num_classes + 1)],
        }
    }

    /// A square matrix from its row-major counts.
    ///
    /// # Errors
    ///
    /// If `counts` is not `num_classes²` long.
    pub fn from_counts(num_classes: usize, counts: Vec<u64>) -> crate::error::Result<Self> {
        if counts.len() != num_classes * num_classes {
            return Err(format!(
                "a confusion matrix for {num_classes} classes has {} cells, got {}",
                num_classes * num_classes,
                counts.len()
            )
            .into());
        }
        Ok(PixelConfusion {
            num_classes,
            unlabeled_pred: false,
            counts,
        })
    }

    /// The number of classes: the rows, and the columns that are classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// The column of the prediction that names no class, when the matrix has
    /// one: `num_classes`.
    pub fn unlabeled_pred(&self) -> Option<usize> {
        self.unlabeled_pred.then_some(self.num_classes)
    }

    fn cols(&self) -> usize {
        self.num_classes + usize::from(self.unlabeled_pred)
    }

    /// The counts, row-major, rows the ground truth; `num_classes` rows of
    /// `num_classes` columns, plus one when there is an unlabeled column.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// The counts, moved out — see [`counts`](Self::counts).
    pub fn into_counts(self) -> Vec<u64> {
        self.counts
    }

    /// Pixels with ground truth `gt` predicted as `pred`; `pred` may be the
    /// unlabeled column.
    pub fn get(&self, gt: usize, pred: usize) -> u64 {
        self.counts[gt * self.cols() + pred]
    }

    /// Every pixel counted so far.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Count one label-map pair. Pixels whose ground truth is `ignore_index`
    /// are skipped on both sides; every other label must be below
    /// `num_classes`. A label map cannot name the unlabeled column.
    ///
    /// # Errors
    ///
    /// On the first label that is negative, or at or above `num_classes` —
    /// a prediction equal to the ignore index included — naming the side,
    /// the pixel, and the value. The matrix is left partly updated; a caller
    /// that wants all-or-nothing counts into a fresh matrix and adds it.
    ///
    /// # Panics
    ///
    /// If `gt` and `pred` differ in length.
    pub fn update<G: Label, P: Label>(
        &mut self,
        gt: &[G],
        pred: &[P],
        ignore_index: Option<u64>,
    ) -> crate::error::Result<()> {
        self.update_with(gt, pred, |g| (Some(g) != ignore_index).then_some(g))
    }

    /// [`update`](Self::update) with the ground truth read through `gt_label`:
    /// `None` skips the pixel, `Some(class)` counts it. The hook for a dataset
    /// convention such as mmseg's `reduce_zero_label`; the ignore rule is one
    /// such closure.
    ///
    /// # Errors and panics
    ///
    /// As [`update`](Self::update).
    pub fn update_with<G: Label, P: Label>(
        &mut self,
        gt: &[G],
        pred: &[P],
        gt_label: impl Fn(u64) -> Option<u64>,
    ) -> crate::error::Result<()> {
        assert_eq!(
            gt.len(),
            pred.len(),
            "gt and pred must have the same number of pixels (gt {}, pred {})",
            gt.len(),
            pred.len()
        );
        let n = self.num_classes as u64;
        let cols = self.cols() as u64;
        let bad = |side: &str, i: usize, value: Option<u64>| -> crate::error::Error {
            match value {
                Some(v) => format!(
                    "{side} pixel {i} has label {v}; labels must be below num_classes ({n}) \
                     or, for the ground truth, the ignore index"
                ),
                None => format!("{side} pixel {i} has a negative label"),
            }
            .into()
        };
        // Label maps are runs of one class, so consecutive pixels mostly hit
        // one cell: count a run, then add it once, rather than a dependent
        // read-modify-write per pixel.
        let mut run_cell = usize::MAX;
        let mut run_len = 0u64;
        for (i, (&g, &p)) in gt.iter().zip(pred).enumerate() {
            let Some(g) = g.index() else {
                return Err(bad("gt", i, None));
            };
            let Some(g) = gt_label(g) else {
                continue;
            };
            if g >= n {
                return Err(bad("gt", i, Some(g)));
            }
            let Some(p) = p.index().filter(|&p| p < n) else {
                return Err(bad("pred", i, pred[i].index()));
            };
            let cell = (g * cols + p) as usize;
            if cell == run_cell {
                run_len += 1;
            } else {
                if run_len > 0 {
                    self.counts[run_cell] += run_len;
                }
                run_cell = cell;
                run_len = 1;
            }
        }
        if run_len > 0 {
            self.counts[run_cell] += run_len;
        }
        Ok(())
    }

    /// Add `count` pixels with ground truth `gt` predicted as `pred` — for a
    /// caller that already holds a histogram, such as panoptic's segment
    /// overlaps collapsed onto categories. `pred` may be the unlabeled column.
    ///
    /// # Panics
    ///
    /// If `gt` is at or above `num_classes`, or `pred` is past the last column.
    pub fn add_count(&mut self, gt: usize, pred: usize, count: u64) {
        assert!(
            gt < self.num_classes && pred < self.cols(),
            "class index out of range: ({gt}, {pred}) in a {}-class matrix",
            self.num_classes
        );
        let cell = gt * self.cols() + pred;
        self.counts[cell] += count;
    }

    /// Add another matrix's counts into this one.
    ///
    /// # Panics
    ///
    /// If the two differ in shape.
    pub fn add(&mut self, other: &PixelConfusion) {
        assert_eq!(
            (self.num_classes, self.unlabeled_pred),
            (other.num_classes, other.unlabeled_pred),
            "cannot add a {}-class confusion matrix to a {}-class one",
            other.num_classes,
            self.num_classes
        );
        for (a, b) in self.counts.iter_mut().zip(&other.counts) {
            *a += b;
        }
    }

    /// The counts of class `c`: its row, including an unlabeled column, is
    /// its ground truth; its column over the class rows is its predictions.
    pub fn class_counts(&self, c: usize) -> ClassCounts {
        let cols = self.cols();
        ClassCounts {
            tp: self.counts[c * cols + c],
            gt_area: self.counts[c * cols..(c + 1) * cols].iter().sum(),
            pred_area: (0..self.num_classes)
                .map(|g| self.counts[g * cols + c])
                .sum(),
        }
    }

    /// The counts of every class, by class id.
    pub fn all_class_counts(&self) -> Vec<ClassCounts> {
        (0..self.num_classes)
            .map(|c| self.class_counts(c))
            .collect()
    }

    /// The four means and the per-class pair.
    pub fn scores(&self) -> SemanticScores {
        semantic_scores(&self.all_class_counts())
    }
}

/// [`SemanticScores`] from per-class counts, in class order.
///
/// The averaging rules are mmseg's `np.nanmean`: a class without an IoU
/// (nothing on either side) is left out of `mIoU`, one without ground truth
/// is left out of `mAcc`, and the sentinel stands in for either per class.
/// `fwIoU` weights by ground-truth share, so classes without ground truth
/// weigh nothing; `aAcc` is total correct over total counted.
pub fn semantic_scores(classes: &[ClassCounts]) -> SemanticScores {
    let (mut iou_sum, mut iou_n, mut acc_sum, mut acc_n) = (0.0, 0usize, 0.0, 0usize);
    let (mut tp_total, mut gt_total) = (0u64, 0u64);
    let mut weighted = 0.0;
    let per_class = classes
        .iter()
        .map(|c| {
            tp_total += c.tp;
            gt_total += c.gt_area;
            let iou = c.iou();
            let acc = c.acc();
            if let Some(v) = iou {
                iou_sum += v;
                iou_n += 1;
                weighted += v * c.gt_area as f64;
            }
            if let Some(v) = acc {
                acc_sum += v;
                acc_n += 1;
            }
            ClassScores {
                iou: iou.unwrap_or(-1.0),
                acc: acc.unwrap_or(-1.0),
            }
        })
        .collect();
    // `fwIoU` and `aAcc` are means over pixels, each ground-truth pixel
    // weighing one, so the count is the ground-truth total.
    let pixels = usize::try_from(gt_total).unwrap_or(usize::MAX);
    SemanticScores {
        miou: mean_or_missing(iou_sum, iou_n),
        macc: mean_or_missing(acc_sum, acc_n),
        fwiou: mean_or_missing(weighted, pixels),
        aacc: mean_or_missing(tp_total as f64, pixels),
        per_class,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_text<T>(r: crate::error::Result<T>) -> String {
        match r {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error"),
        }
    }

    fn worked() -> PixelConfusion {
        // Three classes. Ground truth: four of 0, two of 1, none of 2.
        // Predictions: 0 → 0,0,0,1; 1 → 1,2.
        let mut m = PixelConfusion::new(3);
        m.update(&[0u8, 0, 0, 0, 1, 1], &[0u8, 0, 0, 1, 1, 2], None)
            .expect("valid test input");
        m
    }

    #[test]
    fn counts_land_in_gt_rows_and_pred_columns() {
        let m = worked();
        assert_eq!(m.get(0, 0), 3);
        assert_eq!(m.get(0, 1), 1);
        assert_eq!(m.get(1, 1), 1);
        assert_eq!(m.get(1, 2), 1);
        assert_eq!(m.total(), 6);
        assert_eq!(
            m.class_counts(1),
            ClassCounts {
                tp: 1,
                gt_area: 2,
                pred_area: 2
            }
        );
    }

    #[test]
    fn formulas_by_hand() {
        let s = worked().scores();
        // Class 0: 3 / (4 + 3 - 3) = 0.75; class 1: 1 / (2 + 2 - 1) = 1/3;
        // class 2: 0 / (0 + 1 - 0) = 0, counted (predicted, never ground truth).
        assert_eq!(s.per_class[0].iou, 0.75);
        assert!((s.per_class[1].iou - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(s.per_class[2].iou, 0.0);
        assert!((s.miou - (0.75 + 1.0 / 3.0) / 3.0).abs() < 1e-15);
        // Acc: 3/4, 1/2, none for class 2.
        assert_eq!(s.per_class[0].acc, 0.75);
        assert_eq!(s.per_class[1].acc, 0.5);
        assert_eq!(s.per_class[2].acc, -1.0);
        assert_eq!(s.macc, 0.625);
        // fwIoU: 0.75 · 4/6 + (1/3) · 2/6 + 0 · 0.
        assert!((s.fwiou - (0.75 * 4.0 + 2.0 / 3.0) / 6.0).abs() < 1e-15);
        assert!((s.aacc - 4.0 / 6.0).abs() < 1e-15);
        assert_eq!(s.stats(), [s.miou, s.fwiou, s.macc, s.aacc]);
    }

    #[test]
    fn runs_and_single_pixels_count_the_same() {
        // Alternating labels defeat the run collapse; long runs exercise it.
        let gt: Vec<u8> = (0..1000).map(|i| (i / 100 % 3) as u8).collect();
        let pred: Vec<u8> = (0..1000).map(|i| (i % 2) as u8).collect();
        let mut runs = PixelConfusion::new(3);
        runs.update(&gt, &pred, None).expect("valid test input");
        let mut singles = PixelConfusion::new(3);
        for (g, p) in gt.iter().zip(&pred) {
            singles
                .update(&[*g], &[*p], None)
                .expect("valid test input");
        }
        assert_eq!(runs, singles);
        assert_eq!(runs.total(), 1000);
    }

    #[test]
    fn ignored_ground_truth_drops_the_pixel_on_both_sides() {
        let mut m = PixelConfusion::new(2);
        m.update(&[0u8, 255, 1], &[0u8, 1, 1], Some(255))
            .expect("valid test input");
        assert_eq!(m.total(), 2);
        assert_eq!(m.class_counts(1).pred_area, 1);
    }

    #[test]
    fn a_class_on_neither_side_has_no_scores_and_is_not_averaged() {
        let mut m = PixelConfusion::new(3);
        m.update(&[0u8, 1], &[0u8, 1], None)
            .expect("valid test input");
        let s = m.scores();
        assert_eq!(s.per_class[2].iou, -1.0);
        assert_eq!(s.per_class[2].acc, -1.0);
        assert_eq!(s.miou, 1.0);
        assert_eq!(s.macc, 1.0);
    }

    #[test]
    fn everything_ignored_is_the_sentinel_everywhere() {
        let mut m = PixelConfusion::new(2);
        m.update(&[255u8, 255], &[0u8, 1], Some(255))
            .expect("valid test input");
        let s = m.scores();
        assert!(!s.is_computed());
        assert_eq!(s.stats(), [-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn the_unlabeled_column_is_a_miss_and_not_a_class() {
        let mut m = PixelConfusion::with_unlabeled_pred(2);
        assert_eq!(m.unlabeled_pred(), Some(2));
        m.add_count(0, 0, 3);
        m.add_count(0, 2, 1); // one pixel of class 0 predicted as nothing
        m.add_count(1, 1, 4);
        assert_eq!(m.get(0, 2), 1);
        assert_eq!(m.total(), 8);
        assert_eq!(
            m.class_counts(0),
            ClassCounts {
                tp: 3,
                gt_area: 4,
                pred_area: 3
            }
        );
        let s = m.scores();
        assert_eq!(s.per_class.len(), 2);
        assert_eq!(s.per_class[0].iou, 0.75);
        assert_eq!(s.per_class[0].acc, 0.75);
        assert_eq!(s.miou, 0.875);
        // A label map cannot name that column.
        let err = err_text(m.update(&[0u8], &[2u8], None));
        assert!(err.contains("pred pixel 0 has label 2"), "{err}");
        assert_eq!(PixelConfusion::new(2).unlabeled_pred(), None);
    }

    #[test]
    fn out_of_range_labels_are_errors_naming_the_pixel() {
        let mut m = PixelConfusion::new(2);
        let err = err_text(m.update(&[0u8, 2], &[0u8, 0], None));
        assert!(err.contains("gt pixel 1 has label 2"), "{err}");
        let err = err_text(m.update(&[0u8, 0], &[0u8, 255], Some(255)));
        assert!(err.contains("pred pixel 1 has label 255"), "{err}");
        let err = err_text(m.update(&[0i32, -1], &[0i32, 0], None));
        assert!(err.contains("gt pixel 1 has a negative label"), "{err}");
    }

    #[test]
    fn adding_shards_equals_one_pass() {
        let gt = [0u8, 0, 1, 1, 2, 2];
        let pred = [0u8, 1, 1, 1, 2, 0];
        let mut whole = PixelConfusion::new(3);
        whole.update(&gt, &pred, None).expect("valid test input");
        let mut a = PixelConfusion::new(3);
        a.update(&gt[..3], &pred[..3], None)
            .expect("valid test input");
        let mut b = PixelConfusion::new(3);
        b.update(&gt[3..], &pred[3..], None)
            .expect("valid test input");
        a.add(&b);
        assert_eq!(a, whole);
    }

    #[test]
    fn from_counts_checks_the_size() {
        assert!(PixelConfusion::from_counts(2, vec![1, 2, 3]).is_err());
        let m = PixelConfusion::from_counts(2, vec![3, 1, 0, 2]).expect("square");
        assert_eq!(m.get(0, 1), 1);
        assert_eq!(m.class_counts(1).gt_area, 2);
        assert_eq!(m.clone().into_counts(), m.counts());
    }

    #[test]
    fn add_to_report_writes_the_semantic_block_once() {
        let s = worked().scores();
        let report = s.add_to_report(
            EvalReport::new("semantic", crate::report::Provenance::ParityVerified),
            ["a", "b", "c"],
        );
        assert_eq!(report.metric("mIoU"), Some(s.miou));
        assert_eq!(report.per_class["a"]["IoU"], 0.75);
        assert_eq!(report.per_class["c"]["IoU"], 0.0);
        assert!(
            !report.per_class["c"].contains_key("Acc"),
            "no ground truth, no Acc"
        );
    }

    #[test]
    #[should_panic(expected = "same number of pixels")]
    fn mismatched_lengths_panic() {
        let _ = PixelConfusion::new(2).update(&[0u8], &[0u8, 1], None);
    }
}
