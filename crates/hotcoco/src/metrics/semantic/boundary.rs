//! Boundary IoU — IoU restricted to a band along each class's contour.
//!
//! The definition is Cheng et al., *Boundary IoU: Improving Object-Centric
//! Image Segmentation Evaluation* (CVPR 2021), applied per class as the
//! paper describes for semantic segmentation. A class's boundary is the
//! part of its mask within `d` pixels of a pixel outside the mask or of the
//! image edge, with `d = round(ratio × image diagonal)` and at least 1
//! (`ratio` 0.02 in the paper). Per class, the intersection and union of the
//! two sides' boundaries are summed over the split, and the IoU of those
//! sums is the class's boundary IoU; `mBIoU` is the mean over the classes
//! whose union is not empty. A pixel whose ground truth is ignored is
//! dropped from both sides, as it is from the confusion matrix.
//!
//! The kernel the authors published, `mask_to_boundary` in
//! `boundary-iou-api`, erodes one binary mask with a 3 × 3 kernel `d`
//! times after padding it with a pixel of zeros, and subtracts. That is a
//! Chebyshev window: a pixel is boundary when the `(2d + 1) × (2d + 1)`
//! square around it holds a pixel of another label or reaches past the
//! image. [`boundary_mask`] computes that for every class at once, in three
//! linear passes over the label map, rather than `num_classes` erosions of
//! `d` steps each. `scripts/gen_semantic_fixtures.py` freezes the authors'
//! kernel on synthetic maps and `tests/semantic.rs` holds this module to it.
//!
//! detectron2's `SemSegEvaluator` also reports a `BoundaryIoU`, but erodes
//! the whole label map and keeps `label − window minimum`, so a class has a
//! boundary only beside a lower-numbered class and the number depends on
//! how the classes are numbered. This module does not follow it; the fixture
//! records detectron2's numbers so the difference is observed.

use serde::{Deserialize, Serialize};

use super::{ClassCounts, Label};
use crate::metrics::mean_or_missing;

/// The dilation the paper uses: 2% of the image diagonal.
pub const DEFAULT_DILATION_RATIO: f64 = 0.02;

/// The band width for an `h × w` map: `ratio` times the diagonal, rounded
/// half to even as Python's `round` does in the reference, and at least 1.
pub fn dilation(h: usize, w: usize, ratio: f64) -> usize {
    let diagonal = ((h * h + w * w) as f64).sqrt();
    let d = (ratio * diagonal).round_ties_even();
    if d < 1.0 { 1 } else { d as usize }
}

/// Buffers [`boundary_mask`] reuses between calls.
#[derive(Debug, Default)]
pub struct MaskScratch {
    rows: Vec<u8>,
    columns: Vec<u32>,
}

/// Mark the boundary pixels of an `h × w` label map: `out[i]` is 1 when the
/// `(2d + 1)`-square around pixel `i` holds a label other than `labels[i]`
/// or reaches past the map's edge, as the authors' `mask_to_boundary` marks
/// each class's mask.
///
/// # Panics
///
/// If `labels.len() != h * w`.
pub fn boundary_mask<L: Label>(
    labels: &[L],
    h: usize,
    w: usize,
    d: usize,
    out: &mut Vec<u8>,
    scratch: &mut MaskScratch,
) {
    assert_eq!(
        labels.len(),
        h * w,
        "a label map of {h} × {w} has {} pixels",
        h * w
    );
    out.clear();
    out.resize(h * w, 0);
    if h == 0 || w == 0 {
        return;
    }
    let MaskScratch { rows, columns } = scratch;
    rows.clear();
    rows.resize(h * w, 0);
    // Pass 1, each row on its own: a pixel is marked when its row segment
    // `c - d ..= c + d` holds another label or leaves the row.
    for (row, marks) in labels.chunks_exact(w).zip(rows.chunks_exact_mut(w)) {
        let mut run_start = 0;
        for c in 0..w {
            if c > 0 && row[c] != row[c - 1] {
                run_start = c;
            }
            let other_left = run_start > 0 && c - run_start < d;
            if other_left || c < d || c + d >= w {
                marks[c] = 1;
            }
        }
        let mut run_end = w;
        for c in (0..w).rev() {
            if c + 1 < w && row[c] != row[c + 1] {
                run_end = c + 1;
            }
            if run_end < w && run_end - c <= d {
                marks[c] = 1;
            }
        }
    }
    // Pass 2, down each column: a pixel is marked when a row within `d` of
    // its own holds another label in its column, or had a mark from pass 1
    // in its column — that row's segment then holds another label too — or
    // when the window leaves the map. `since_mark` counts rows since a
    // pass-1 mark in the column, `run_start` the row the column's current
    // label began at.
    columns.clear();
    columns.resize(2 * w, 0);
    let (since_mark, run_start) = columns.split_at_mut(w);
    let far = u32::MAX;
    since_mark.fill(far);
    run_start.fill(0);
    let d32 = u32::try_from(d).unwrap_or(u32::MAX);
    for r in 0..h {
        let row = &labels[r * w..(r + 1) * w];
        let marks = &rows[r * w..(r + 1) * w];
        let next = &mut out[r * w..(r + 1) * w];
        for c in 0..w {
            if r > 0 && row[c] != labels[(r - 1) * w + c] {
                run_start[c] = r as u32;
            }
            since_mark[c] = if marks[c] == 1 {
                0
            } else {
                since_mark[c].saturating_add(1)
            };
            let other_above = run_start[c] > 0 && (r as u32 - run_start[c]) < d32;
            if other_above || since_mark[c] <= d32 || r < d || r + d >= h {
                next[c] = 1;
            }
        }
    }
    // The same, bottom up, for a mark or another label below.
    since_mark.fill(far);
    let run_end = run_start;
    run_end.fill(h as u32);
    for r in (0..h).rev() {
        let row = &labels[r * w..(r + 1) * w];
        let marks = &rows[r * w..(r + 1) * w];
        let next = &mut out[r * w..(r + 1) * w];
        for c in 0..w {
            if r + 1 < h && row[c] != labels[(r + 1) * w + c] {
                run_end[c] = r as u32 + 1;
            }
            since_mark[c] = if marks[c] == 1 {
                0
            } else {
                since_mark[c].saturating_add(1)
            };
            let other_below = run_end[c] < h as u32 && (run_end[c] - r as u32) <= d32;
            if other_below || since_mark[c] <= d32 {
                next[c] = 1;
            }
        }
    }
}

/// Boundary pixel counts per class, summed over images: for class `c`,
/// `tp` the pixels on both boundaries with ground truth and prediction `c`,
/// `gt_area` the ground-truth boundary pixels of `c`, `pred_area` the
/// predicted boundary pixels of `c` — [`ClassCounts`] over boundary pixels,
/// so its `iou()` is the class's boundary IoU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryCounts {
    dilation_ratio: f64,
    classes: Vec<ClassCounts>,
    #[serde(skip)]
    scratch: Scratch,
}

/// The boundary masks and their passes' buffers, kept between images. Not
/// state: two counts compare equal whatever it holds, a clone starts empty,
/// and it is not serialized — as the confusion matrix's lane scratch.
#[derive(Default)]
struct Scratch {
    gt: Vec<u8>,
    pred: Vec<u8>,
    mask: MaskScratch,
}

impl Clone for Scratch {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for Scratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for Scratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scratch({} pixels)", self.gt.len())
    }
}

/// Boundary IoU per class and its mean, each the `-1.0` sentinel where the
/// union is empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryScores {
    /// Mean boundary IoU over the classes with a non-empty boundary union.
    pub mbiou: f64,
    /// Boundary IoU per class, by class id.
    pub per_class: Vec<f64>,
}

impl BoundaryCounts {
    /// Empty counts for `num_classes` classes at a band of `dilation_ratio`
    /// times the diagonal.
    pub fn new(num_classes: usize, dilation_ratio: f64) -> Self {
        BoundaryCounts {
            dilation_ratio,
            classes: vec![ClassCounts::default(); num_classes],
            scratch: Scratch::default(),
        }
    }

    /// The band width as a share of the image diagonal.
    pub fn dilation_ratio(&self) -> f64 {
        self.dilation_ratio
    }

    /// The counts of class `c`.
    pub fn class_counts(&self, c: usize) -> ClassCounts {
        self.classes[c]
    }

    /// Count one `h × w` pair. `gt_label` reads the ground truth as
    /// [`PixelConfusion::update_with`](super::PixelConfusion::update_with)
    /// does: `None` skips the pixel. Every prediction must be a class index
    /// below `num_classes` and every counted ground truth too; the caller
    /// has already run the confusion matrix, which rejects a pair that is
    /// not.
    ///
    /// # Panics
    ///
    /// If the maps are not `h * w` long, or a label is out of range.
    pub fn update<G: Label, P: Label>(
        &mut self,
        gt: &[G],
        pred: &[P],
        h: usize,
        w: usize,
        gt_label: impl Fn(u64) -> Option<u64>,
    ) {
        let d = dilation(h, w, self.dilation_ratio);
        let Scratch {
            gt: gb,
            pred: pb,
            mask,
        } = &mut self.scratch;
        boundary_mask(gt, h, w, d, gb, mask);
        boundary_mask(pred, h, w, d, pb, mask);
        for (i, (&g, &p)) in gt.iter().zip(pred).enumerate() {
            let Some(g) = g.index().and_then(&gt_label) else {
                continue;
            };
            let (g, p) = (g as usize, p.index().expect("a checked label") as usize);
            let (on_gt, on_pred) = (gb[i] == 1, pb[i] == 1);
            if on_gt {
                self.classes[g].gt_area += 1;
            }
            if on_pred {
                self.classes[p].pred_area += 1;
            }
            if on_gt && on_pred && g == p {
                self.classes[g].tp += 1;
            }
        }
    }

    /// Add another's counts into these.
    ///
    /// # Panics
    ///
    /// If the two differ in class count or dilation ratio.
    pub fn add(&mut self, other: &BoundaryCounts) {
        assert_eq!(
            (self.classes.len(), self.dilation_ratio),
            (other.classes.len(), other.dilation_ratio),
            "cannot add boundary counts of a different configuration"
        );
        for (a, b) in self.classes.iter_mut().zip(&other.classes) {
            a.tp += b.tp;
            a.gt_area += b.gt_area;
            a.pred_area += b.pred_area;
        }
    }

    /// The boundary IoU per class and its mean.
    pub fn scores(&self) -> BoundaryScores {
        let (mut sum, mut n) = (0.0, 0);
        let per_class = self
            .classes
            .iter()
            .map(|c| {
                let iou = c.iou();
                if let Some(v) = iou {
                    sum += v;
                    n += 1;
                }
                iou.unwrap_or(-1.0)
            })
            .collect();
        BoundaryScores {
            mbiou: mean_or_missing(sum, n),
            per_class,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(labels: &[u8], h: usize, w: usize, d: usize) -> Vec<u8> {
        let (mut out, mut scratch) = (Vec::new(), MaskScratch::default());
        boundary_mask(labels, h, w, d, &mut out, &mut scratch);
        out
    }

    #[test]
    fn dilation_rounds_half_to_even_like_the_reference() {
        assert_eq!(dilation(75, 100, 0.02), 2, "2.5 rounds to 2 in Python");
        assert_eq!(dilation(10, 10, 0.02), 1, "never below 1");
        assert_eq!(dilation(1024, 2048, 0.02), 46);
    }

    #[test]
    fn interior_of_a_wide_block_is_not_boundary() {
        // A 7 × 7 map of one class: at d = 1 only the outer ring is boundary;
        // at d = 3 everything but the center, whose square is the whole map
        // (the reference pads one pixel, so the center is four from it); at
        // d = 4 every pixel.
        let m = mask(&[0u8; 49], 7, 7, 1);
        let ring: Vec<u8> = (0..49)
            .map(|i| u8::from(i / 7 == 0 || i / 7 == 6 || i % 7 == 0 || i % 7 == 6))
            .collect();
        assert_eq!(m, ring);
        let mut all_but_center = vec![1u8; 49];
        all_but_center[24] = 0;
        assert_eq!(mask(&[0u8; 49], 7, 7, 3), all_but_center);
        assert_eq!(mask(&[0u8; 49], 7, 7, 4), vec![1u8; 49]);
    }

    #[test]
    fn a_label_change_marks_both_sides_within_d() {
        // One row, two labels, d = 2: the two pixels each side of the change
        // and the two at each end.
        let row = [0u8, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        assert_eq!(
            mask(&row, 1, 10, 2),
            vec![1u8; 10],
            "a 1-row map is all edge"
        );
        // Three rows the same: the middle row is boundary only near the
        // change and the ends.
        let mut rows = row.to_vec();
        rows.extend_from_slice(&row);
        rows.extend_from_slice(&row);
        let m = mask(&rows, 3, 10, 1);
        assert_eq!(&m[10..20], &[1, 0, 0, 0, 1, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn a_diagonal_neighbor_counts() {
        // The window is a square: (1, 1) is within d = 1 of (0, 0).
        let m = mask(&[1u8, 0, 0, 0, 0, 0, 0, 0, 0], 3, 3, 1);
        assert_eq!(m[4], 1);
        // At d = 1 in a 5 × 5 map the center is interior unless something
        // touches its square.
        let mut five = [0u8; 25];
        assert_eq!(mask(&five, 5, 5, 1)[12], 0);
        five[6] = 1; // (1, 1), diagonal to the center (2, 2)
        assert_eq!(mask(&five, 5, 5, 1)[12], 1);
    }

    #[test]
    fn counts_follow_the_definition() {
        // 1-row maps: everything is boundary, so BIoU is IoU with ignore applied.
        let mut b = BoundaryCounts::new(2, 0.02);
        b.update(&[0u8, 0, 1, 255], &[0u8, 1, 1, 1], 1, 4, |g| {
            (g != 255).then_some(g)
        });
        let s = b.scores();
        assert_eq!(s.per_class, vec![0.5, 0.5]);
        assert_eq!(s.mbiou, 0.5);
        let mut other = BoundaryCounts::new(2, 0.02);
        other.add(&b);
        other.add(&b);
        assert_eq!(other.scores(), s, "adding scales every count alike");
    }
}
