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

    /// The label as an index clamped to `bad`, a negative value landing on
    /// `bad` too. One `min` for the counting loop, where the row or column
    /// `bad` collects everything out of range.
    fn clamped(self, bad: u64) -> u64;
}

/// Up to 64 bits, reinterpreting the bits as `u64` puts a negative value at
/// `2^63` or above, past any class count, so one `min` clamps both ways.
macro_rules! label_as_u64 {
    ($($t:ty),*) => {$(
        impl Label for $t {
            #[inline]
            fn index(self) -> Option<u64> {
                u64::try_from(self).ok()
            }
            #[inline]
            fn clamped(self, bad: u64) -> u64 {
                (self as i64 as u64).min(bad)
            }
        }
    )*};
}
label_as_u64!(i8, i16, i32, i64, isize);

macro_rules! label_unsigned {
    ($($t:ty),*) => {$(
        impl Label for $t {
            #[inline]
            fn index(self) -> Option<u64> {
                Some(self as u64)
            }
            #[inline]
            fn clamped(self, bad: u64) -> u64 {
                (self as u64).min(bad)
            }
        }
    )*};
}
label_unsigned!(bool, u8, u16, u32, u64, usize);

macro_rules! label_wide {
    ($($t:ty),*) => {$(
        impl Label for $t {
            #[inline]
            fn index(self) -> Option<u64> {
                u64::try_from(self).ok()
            }
            #[inline]
            fn clamped(self, bad: u64) -> u64 {
                self.index().map_or(bad, |v| v.min(bad))
            }
        }
    )*};
}
label_wide!(i128, u128);

/// The scratch geometry of one [`PixelConfusion::update_with`] call and the
/// one rule for which cell a pixel lands in, shared by the counting loop and
/// the walk that names the first out-of-range label.
///
/// Rows `0..n` are the classes; row `n` (`bad`) collects a ground-truth label
/// out of range and row `n + 1` (`sink`) a skipped pixel. Columns `0..n` are
/// the classes and column `n` (`bad`) a prediction out of range — which under
/// a skipped pixel is not an error: the prediction there is not checked.
struct Cells<'a, F> {
    n: usize,
    bad: usize,
    sink: usize,
    cols: usize,
    gt_label: &'a F,
}

impl<F: Fn(u64) -> Option<u64>> Cells<'_, F> {
    fn new(n: usize, gt_label: &F) -> Cells<'_, F> {
        Cells {
            n,
            bad: n,
            sink: n + 1,
            cols: n + 1,
            gt_label,
        }
    }

    fn len(&self) -> usize {
        (self.n + 2) * self.cols
    }

    #[inline]
    fn of<G: Label, P: Label>(&self, g: G, p: P) -> (usize, usize) {
        let bad = self.bad as u64;
        let row = match g.index() {
            None => self.bad,
            Some(g) => (self.gt_label)(g).map_or(self.sink, |class| class.min(bad) as usize),
        };
        (row, p.clamped(bad) as usize)
    }

    /// The error for the first pixel in the bad row, or in the bad column of
    /// a counted row; `None` when every pixel is in range.
    fn first_bad<G: Label, P: Label>(&self, gt: &[G], pred: &[P]) -> Option<crate::error::Error> {
        let n = self.n;
        let message = |side: &str, i: usize, value: Option<u64>| -> crate::error::Error {
            match value {
                Some(v) => format!(
                    "{side} pixel {i} has label {v}; labels must be below num_classes ({n}) \
                     or, for the ground truth, the ignore index"
                ),
                None => format!("{side} pixel {i} has a negative label"),
            }
            .into()
        };
        for (i, (&g, &p)) in gt.iter().zip(pred).enumerate() {
            let (row, col) = self.of(g, p);
            if row == self.bad {
                return Some(message("gt", i, g.index()));
            }
            if col == self.bad && row != self.sink {
                return Some(message("pred", i, p.index()));
            }
        }
        None
    }
}

/// Interleaved copies of the histogram the counting loop fills, so that
/// consecutive pixels of one class — a label map is mostly runs — do not
/// serialize on one cell's read-modify-write.
const HISTOGRAM_LANES: usize = 4;

/// Lanes are used when the map has at least this many pixels per lane cell,
/// so zeroing and merging the copies stays a few percent of the counting.
const LANE_PIXELS_PER_CELL: usize = 2;

/// Above this many cells one copy is used whatever the map's size: four of
/// an ADE20K-full (150 × 151) matrix are still small, four of a 1000-class
/// matrix would be 32 MB per call.
const MAX_LANED_CELLS: usize = 1 << 16;

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
    /// the pixel, and the value. Nothing from the pair is counted then.
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
        let n = self.num_classes;
        // The loop has no branch per pixel. It counts into a scratch matrix two
        // rows and one column larger than the classes: a skipped pixel lands in
        // the sink row, and a label at or past `num_classes`, or negative,
        // clamps into the bad row or the bad column. Those strips are inspected
        // once at the end; a count there sends the pair down the per-pixel walk
        // that names the first offender.
        let cells = Cells::new(n, &gt_label);
        let cols = cells.cols;
        let per_lane = cells.len();
        let stride = self.cols();
        if gt.len() < per_lane {
            // Fewer pixels than cells: zeroing and merging a scratch would
            // cost more than a second pass over the pixels does.
            if let Some(err) = cells.first_bad(gt, pred) {
                return Err(err);
            }
            for (&g, &p) in gt.iter().zip(pred) {
                let (row, col) = cells.of(g, p);
                if row < n {
                    self.counts[row * stride + col] += 1;
                }
            }
            return Ok(());
        }
        // Lanes as a stride: `0` folds every lane onto the first.
        let laned = per_lane <= MAX_LANED_CELLS
            && gt.len() >= LANE_PIXELS_PER_CELL * HISTOGRAM_LANES * per_lane;
        let lane_stride = if laned { per_lane } else { 0 };
        let mut scratch = vec![0u64; if laned { HISTOGRAM_LANES } else { 1 } * per_lane];
        let mut count = |lane: usize, g: G, p: P| {
            let (row, col) = cells.of(g, p);
            scratch[lane * lane_stride + row * cols + col] += 1;
        };
        // `slice::as_chunks` would do this directly, but it is past the MSRV;
        // the fixed-size view lets the lane loop unroll without a bounds check
        // per pixel.
        let gt_lanes = gt.chunks_exact(HISTOGRAM_LANES);
        let pred_lanes = pred.chunks_exact(HISTOGRAM_LANES);
        let (gt_rest, pred_rest) = (gt_lanes.remainder(), pred_lanes.remainder());
        for (g, p) in gt_lanes.zip(pred_lanes) {
            let g: &[G; HISTOGRAM_LANES] = g.try_into().expect("chunks_exact yields full chunks");
            let p: &[P; HISTOGRAM_LANES] = p.try_into().expect("chunks_exact yields full chunks");
            for lane in 0..HISTOGRAM_LANES {
                count(lane, g[lane], p[lane]);
            }
        }
        for (&g, &p) in gt_rest.iter().zip(pred_rest) {
            count(0, g, p);
        }
        let (first, rest) = scratch.split_at_mut(per_lane);
        for lane in rest.chunks_exact(per_lane) {
            for (total, &c) in first.iter_mut().zip(lane) {
                *total += c;
            }
        }
        let counted = &*first;
        let bad_row = &counted[cells.bad * cols..cells.sink * cols];
        let bad_col = counted
            .iter()
            .skip(cells.bad)
            .step_by(cols)
            .take(cells.sink);
        if bad_row.iter().chain(bad_col).any(|&c| c > 0) {
            // The walk applies the same rule to the same pixels, so it finds
            // the offender; the fallback only keeps a count from vanishing.
            return Err(cells
                .first_bad(gt, pred)
                .unwrap_or_else(|| "a label is out of range".into()));
        }
        for row in 0..n {
            let totals = &mut self.counts[row * stride..(row + 1) * stride];
            for (total, &c) in totals.iter_mut().zip(&counted[row * cols..]) {
                *total += c;
            }
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

    /// The matrix `update` builds from the pair, one pixel per call — every
    /// call then takes the two-pass path, whatever the batch takes.
    fn one_pixel_at_a_time<G: Label, P: Label>(
        n: usize,
        gt: &[G],
        pred: &[P],
        ignore: Option<u64>,
    ) -> PixelConfusion {
        let mut m = PixelConfusion::new(n);
        for (g, p) in gt.iter().zip(pred) {
            m.update(&[*g], &[*p], ignore).expect("valid test input");
        }
        m
    }

    #[test]
    fn every_counting_path_agrees_with_one_pixel_at_a_time() {
        // Lanes: alternating labels and long runs, many pixels per cell, at a
        // length that leaves a remainder after the chunks.
        let (n, len) = (3, 2001);
        assert!(len >= LANE_PIXELS_PER_CELL * HISTOGRAM_LANES * (n + 2) * (n + 1));
        let gt: Vec<u8> = (0..len).map(|i| (i / 100 % 3) as u8).collect();
        let pred: Vec<u8> = (0..len).map(|i| (i % 2) as u8).collect();
        let mut laned = PixelConfusion::new(n);
        laned.update(&gt, &pred, None).expect("valid test input");
        assert_eq!(laned, one_pixel_at_a_time(n, &gt, &pred, None));
        assert_eq!(laned.total(), len as u64);

        // One lane: enough pixels for a scratch, too few per cell for lanes.
        let (n, len) = (3, 100);
        assert!(len >= (n + 2) * (n + 1));
        assert!(len < LANE_PIXELS_PER_CELL * HISTOGRAM_LANES * (n + 2) * (n + 1));
        let mut single = PixelConfusion::new(n);
        single
            .update(&gt[..len], &pred[..len], None)
            .expect("valid test input");
        assert_eq!(
            single,
            one_pixel_at_a_time(n, &gt[..len], &pred[..len], None)
        );

        // One lane because the matrix is past MAX_LANED_CELLS; 16-bit and
        // 64-bit labels past 255 exercise the wider types.
        let (n, len) = (300, 100_000);
        assert!((n + 2) * (n + 1) > MAX_LANED_CELLS);
        let gt: Vec<u16> = (0..len).map(|i| (i % n) as u16).collect();
        let pred: Vec<i64> = (0..len).map(|i| ((i * 7) % n) as i64).collect();
        let mut wide = PixelConfusion::new(n);
        wide.update(&gt, &pred, Some(65535))
            .expect("valid test input");
        assert_eq!(wide, one_pixel_at_a_time(n, &gt, &pred, Some(65535)));
        assert_eq!(wide.total(), len as u64);
        assert_eq!(wide.get(299, (299 * 7) % n), 333);

        // Two passes: fewer pixels than cells.
        let mut small = PixelConfusion::new(50);
        small
            .update(&[0u8, 49, 3, 3, 3, 3, 3], &[0u8, 49, 3, 3, 3, 3, 4], None)
            .expect("valid test input");
        assert_eq!(small.get(3, 3), 4);
        assert_eq!(small.get(3, 4), 1);
        assert_eq!(small.total(), 7);
    }

    #[test]
    fn a_rejected_pair_leaves_the_matrix_as_it_was() {
        let mut m = worked();
        let before = m.clone();
        let err = err_text(m.update(&[0u8, 0, 0, 0, 0, 3], &[0u8, 0, 0, 0, 0, 0], None));
        assert!(err.contains("gt pixel 5 has label 3"), "{err}");
        assert_eq!(m, before);
        let err = err_text(m.update(&[0u8, 0], &[0i64, -5], None));
        assert!(err.contains("pred pixel 1 has a negative label"), "{err}");
        assert_eq!(m, before);
        // A negative label does not wrap into range however wide the type.
        let err = err_text(m.update(&[i64::MIN], &[0u8], None));
        assert!(err.contains("gt pixel 0 has a negative label"), "{err}");
        let err = err_text(m.update(&[0u8], &[u64::MAX], None));
        assert!(
            err.contains("pred pixel 0 has label 18446744073709551615"),
            "{err}"
        );
        assert_eq!(m, before);
    }

    #[test]
    fn a_skipped_pixel_does_not_check_its_prediction() {
        // Padding is often ignored on both sides; a negative or out-of-range
        // prediction under an ignored pixel is neither right nor wrong.
        let mut m = PixelConfusion::new(2);
        m.update(&[255u8, 0, 1], &[255u8, 0, 1], Some(255))
            .expect("the prediction under an ignored pixel is not checked");
        m.update(&[255u8, 1], &[-3i64, 1], Some(255))
            .expect("nor is a negative one");
        assert_eq!(m.total(), 3);
        assert_eq!(m.get(0, 0), 1);
        assert_eq!(m.get(1, 1), 2);
        // Under a counted pixel the same prediction is the usual error.
        let err = err_text(m.update(&[0u8], &[255u8], Some(255)));
        assert!(err.contains("pred pixel 0 has label 255"), "{err}");
        assert_eq!(m.total(), 3);
    }

    #[test]
    fn any_integer_primitive_is_a_label() {
        let mut m = PixelConfusion::new(2);
        m.update(&[true, false], &[1u128, 0], None)
            .expect("bool and u128 are labels");
        m.update(&[1i8, 0], &[1usize, 0], None)
            .expect("as are the narrow and pointer-sized ones");
        assert_eq!(m.get(1, 1), 2);
        assert_eq!(m.total(), 4);
    }

    #[test]
    fn zero_classes_counts_only_skipped_pixels() {
        let mut m = PixelConfusion::new(0);
        m.update(&[255u8, 255], &[0u8, 7], Some(255))
            .expect("every pixel skipped");
        assert_eq!(m.total(), 0);
        let err = err_text(m.update(&[0u8], &[0u8], Some(255)));
        assert!(err.contains("gt pixel 0 has label 0"), "{err}");
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
