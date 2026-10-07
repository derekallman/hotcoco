//! Panoptic segment matching — the kernel underneath PQ.
//!
//! Panoptic quality pairs ground-truth and predicted *segments*, where a
//! segment is the set of pixels carrying one id in a per-image label map. Two
//! steps, both here:
//!
//! 1. [`Overlaps::compute`] counts, for every `(gt id, pred id)` pair, how many
//!    pixels carry both — the co-occurrence histogram panopticapi builds with
//!    `np.unique(gt * OFFSET + pred)`. This is the similarity step: everything
//!    PQ needs about geometry is in that table.
//! 2. [`match_segments`] applies the protocol's rules to it — same category,
//!    ground truth not crowd, [`pq_iou`] strictly above `0.5` — and classifies
//!    every segment as matched, missed, false positive, or ignored.
//!
//! # Why this is not `sim` + `greedy`
//!
//! PQ needs no solver. With the union computed the way panopticapi does, an
//! IoU above `0.5` is provably unique in both directions (each matched pair
//! covers more than half of both segments), so there is nothing to rank or
//! assign — the histogram *is* the matching. And its IoU formula is not the
//! one in [`sim`](super::sim): the denominator subtracts the prediction's
//! overlap with void, so a prediction is not penalized for covering unlabeled
//! pixels. That is why `tests/architecture.rs` lists this file as a second,
//! distinct formula home rather than a copy to consolidate.
//!
//! # Label-map conventions
//!
//! Both maps are flat `u32` slices of equal length, in the same pixel order
//! (which order does not matter — the histogram is order-free). Id `0` is
//! [`VOID`] in both. Ids are whatever the caller painted: PNG colors decoded
//! with `rgb2id` in the COCO panoptic format, or dense local labels assigned
//! while rasterizing RLEs. The kernel never sees category ids or annotation
//! ids; [`Segment`] carries those alongside the label.
//!
//! # Degenerate input
//!
//! Mismatched map lengths are a programmer error and panic, as every kernel
//! in `primitives` does. Empty maps produce an empty histogram, and a segment
//! list with no entries matches nothing — both legitimate states.

use rustc_hash::FxHashMap;

/// The label panopticapi reserves for unlabeled pixels.
pub const VOID: u32 = 0;

/// The pixel-count histogram of `(gt id, pred id)` co-occurrences for one image.
///
/// Sorted by `(gt, pred)`, so lookups are binary searches and iteration order
/// is panopticapi's `np.unique` order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Overlaps {
    pairs: Vec<(u32, u32, u64)>,
}

impl Overlaps {
    /// Count pixel co-occurrences between two label maps.
    ///
    /// # Panics
    ///
    /// If `gt` and `pred` have different lengths.
    pub fn compute(gt: &[u32], pred: &[u32]) -> Self {
        assert_eq!(
            gt.len(),
            pred.len(),
            "label maps differ in length: gt has {} pixels, pred has {}",
            gt.len(),
            pred.len()
        );
        Self::from_pixels(gt.iter().copied().zip(pred.iter().copied()))
    }

    /// [`compute`](Self::compute) over a stream of `(gt id, pred id)` pixel
    /// pairs. The caller guarantees the two sides are the same length and in
    /// the same pixel order. For a source that is not an iterator, feed an
    /// [`OverlapsBuilder`] directly.
    pub fn from_pixels(pixels: impl Iterator<Item = (u32, u32)>) -> Self {
        let mut builder = OverlapsBuilder::default();
        for (g, p) in pixels {
            builder.push(g, p);
        }
        builder.finish()
    }

    /// Pixels carrying `gt` in the ground-truth map and `pred` in the
    /// prediction map; `0` when the pair never co-occurs.
    pub fn get(&self, gt: u32, pred: u32) -> u64 {
        match self
            .pairs
            .binary_search_by(|&(g, p, _)| (g, p).cmp(&(gt, pred)))
        {
            Ok(i) => self.pairs[i].2,
            Err(_) => 0,
        }
    }

    /// Every co-occurring `(gt id, pred id, pixels)`, sorted by id pair.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32, u64)> + '_ {
        self.pairs.iter().copied()
    }

    /// Pixel count of every id present in the prediction map, sorted by id.
    ///
    /// panopticapi takes these from `np.unique(pan_pred, return_counts=True)`
    /// and uses them as the predicted areas — not the areas the JSON states.
    pub fn pred_areas(&self) -> Vec<(u32, u64)> {
        let mut by_pred: FxHashMap<u32, u64> = FxHashMap::default();
        for &(_, p, n) in &self.pairs {
            *by_pred.entry(p).or_insert(0) += n;
        }
        let mut out: Vec<(u32, u64)> = by_pred.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Pixel count of every id present in the ground-truth map, sorted by id.
    ///
    /// Already in `(gt, pred)` order, so this is one pass with no map.
    pub fn gt_areas(&self) -> Vec<(u32, u64)> {
        let mut out: Vec<(u32, u64)> = Vec::new();
        for &(g, _, n) in &self.pairs {
            match out.last_mut() {
                Some((last, total)) if *last == g => *total += n,
                _ => out.push((g, n)),
            }
        }
        out
    }
}

/// Accumulates an [`Overlaps`] one pixel pair at a time.
///
/// Neighboring pixels nearly always share a pair, so runs are counted and the
/// map is touched once per run rather than once per pixel. A caller whose
/// pixels come from nested buffers — rows of a decoded image — loops over
/// them and calls [`push`](Self::push), which keeps that loop tight.
#[derive(Debug, Default)]
pub struct OverlapsBuilder {
    counts: FxHashMap<u64, u64>,
    run_key: u64,
    run_len: u64,
}

impl OverlapsBuilder {
    /// One pixel carrying `gt` on the ground-truth side and `pred` on the
    /// prediction side.
    #[inline]
    pub fn push(&mut self, gt: u32, pred: u32) {
        let key = (u64::from(gt) << 32) | u64::from(pred);
        if key == self.run_key && self.run_len > 0 {
            self.run_len += 1;
        } else {
            self.flush();
            self.run_key = key;
            self.run_len = 1;
        }
    }

    fn flush(&mut self) {
        if self.run_len > 0 {
            *self.counts.entry(self.run_key).or_insert(0) += self.run_len;
            self.run_len = 0;
        }
    }

    /// The histogram of everything pushed so far.
    pub fn finish(mut self) -> Overlaps {
        self.flush();
        let mut pairs: Vec<(u32, u32, u64)> = self
            .counts
            .into_iter()
            .map(|(key, n)| ((key >> 32) as u32, key as u32, n))
            .collect();
        pairs.sort_unstable();
        Overlaps { pairs }
    }
}

/// panopticapi's intersection over union between one ground-truth segment and
/// one predicted segment.
///
/// The union is `pred_area + gt_area - intersection - void_intersection`: the
/// prediction's pixels that fall on [`VOID`] ground truth are taken out of the
/// denominator, so covering unlabeled pixels costs nothing. `gt_area` is
/// whatever the caller passes — panopticapi reads it from the JSON, not from
/// the PNG — and `pred_area` is the pixel count from the map.
///
/// A stated `gt_area` smaller than the pixels actually found can drive the
/// union to zero or below; the result is then not a finite number in `[0, 1]`
/// and the caller decides what that means. The division is left as is rather
/// than clamped so that case cannot pass for a real score.
#[inline]
pub fn pq_iou(intersection: u64, gt_area: u64, pred_area: u64, void_intersection: u64) -> f64 {
    let union =
        pred_area as i128 + gt_area as i128 - intersection as i128 - void_intersection as i128;
    intersection as f64 / union as f64
}

/// One segment as the matcher sees it: its label in the map, what the
/// annotation says about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    /// The id painted in the label map.
    pub label: u32,
    /// Category, compared for equality between the two sides.
    pub category_id: u64,
    /// Pixel count used in [`pq_iou`]. For a prediction this must be the
    /// count from the map; for ground truth, panopticapi uses the JSON's.
    pub area: u64,
    /// Ground truth only. A crowd segment is never matched, never a miss, and
    /// absorbs unmatched predictions of its category (see [`match_segments`]).
    pub iscrowd: bool,
}

/// What [`match_segments`] decided for one image, as indices into the segment
/// slices it was given.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SegmentMatches {
    /// `(gt index, pred index, iou)` for every pair with IoU above `0.5`.
    pub matched: Vec<(usize, usize, f64)>,
    /// Ground-truth indices that are neither matched nor crowd.
    pub missed: Vec<usize>,
    /// Prediction indices that are unmatched and not ignored.
    pub false_positives: Vec<usize>,
    /// Prediction indices that are unmatched but mostly cover void or a crowd
    /// region of their own category, and therefore count as nothing.
    pub ignored: Vec<usize>,
}

/// Apply the panoptic matching rules to one image's histogram.
///
/// Mirrors `pq_compute_single_core` in panopticapi, rule for rule:
///
/// - A `(gt, pred)` pair is a match when both labels are known segments, the
///   ground truth is not crowd, the categories agree, and [`pq_iou`] is
///   strictly greater than `0.5`. Every qualifying pair is recorded — the
///   reference does not deduplicate, and with consistent areas it never
///   needs to.
/// - An unmatched, non-crowd ground truth is a miss.
/// - An unmatched prediction is ignored when more than half of its pixels
///   lie on void plus the crowd region of its own category, if the image has
///   one — the **last** crowd segment of that category in `gt` order, as the
///   reference's dict overwrite picks it. Otherwise it is a false positive.
///
/// Labels in the maps that no [`Segment`] describes are skipped in matching,
/// and ground-truth pixels under such a label are not void: they neither
/// shrink a union nor help a prediction get ignored. Whether such labels are
/// an error is the caller's call (the reference rejects them on the
/// prediction side and accepts them on the ground-truth side).
///
/// # Panics
///
/// If a prediction has `area == 0`, since the ignore rule divides by it. A
/// caller that built `pred` from [`Overlaps::pred_areas`] cannot hit this.
pub fn match_segments(gt: &[Segment], pred: &[Segment], overlaps: &Overlaps) -> SegmentMatches {
    // Label -> index. A repeated label keeps the last entry, as a dict built
    // from the segment list would.
    let gt_index: FxHashMap<u32, usize> =
        gt.iter().enumerate().map(|(i, s)| (s.label, i)).collect();
    let pred_index: FxHashMap<u32, usize> =
        pred.iter().enumerate().map(|(i, s)| (s.label, i)).collect();

    let mut out = SegmentMatches::default();
    let mut gt_matched = vec![false; gt.len()];
    let mut pred_matched = vec![false; pred.len()];

    for (g_label, p_label, intersection) in overlaps.iter() {
        let (Some(&gi), Some(&pi)) = (gt_index.get(&g_label), pred_index.get(&p_label)) else {
            continue;
        };
        let (g, p) = (&gt[gi], &pred[pi]);
        if g.iscrowd || g.category_id != p.category_id {
            continue;
        }
        let iou = pq_iou(intersection, g.area, p.area, overlaps.get(VOID, p_label));
        if iou > 0.5 {
            out.matched.push((gi, pi, iou));
            gt_matched[gi] = true;
            pred_matched[pi] = true;
        }
    }

    // Category -> label of its crowd segment, last one in `gt` order winning.
    let mut crowd_by_category: FxHashMap<u64, u32> = FxHashMap::default();
    for (gi, g) in gt.iter().enumerate() {
        if gt_matched[gi] {
            continue;
        }
        if g.iscrowd {
            crowd_by_category.insert(g.category_id, g.label);
        } else {
            out.missed.push(gi);
        }
    }

    for (pi, p) in pred.iter().enumerate() {
        if pred_matched[pi] {
            continue;
        }
        assert!(p.area > 0, "prediction label {} has zero area", p.label);
        let mut unlabeled = overlaps.get(VOID, p.label);
        if let Some(&crowd_label) = crowd_by_category.get(&p.category_id) {
            unlabeled += overlaps.get(crowd_label, p.label);
        }
        if unlabeled as f64 / p.area as f64 > 0.5 {
            out.ignored.push(pi);
        } else {
            out.false_positives.push(pi);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(label: u32, category_id: u64, area: u64) -> Segment {
        Segment {
            label,
            category_id,
            area,
            iscrowd: false,
        }
    }

    #[test]
    fn overlaps_count_every_pair_once() {
        let gt = [1, 1, 2, 2, 0, 1];
        let pred = [5, 5, 5, 6, 6, 5];
        let o = Overlaps::compute(&gt, &pred);
        assert_eq!(o.get(1, 5), 3);
        assert_eq!(o.get(2, 5), 1);
        assert_eq!(o.get(2, 6), 1);
        assert_eq!(o.get(0, 6), 1);
        assert_eq!(o.get(1, 6), 0);
        assert_eq!(o.pred_areas(), vec![(5, 4), (6, 2)]);
        assert_eq!(o.gt_areas(), vec![(0, 1), (1, 3), (2, 2)]);
        let total: u64 = o.iter().map(|(_, _, n)| n).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn overlaps_are_sorted_by_pair() {
        let gt = [3, 1, 2, 1, 3];
        let pred = [1, 9, 4, 2, 1];
        let o = Overlaps::compute(&gt, &pred);
        let pairs: Vec<_> = o.iter().map(|(g, p, _)| (g, p)).collect();
        assert_eq!(pairs, vec![(1, 2), (1, 9), (2, 4), (3, 1)]);
    }

    #[test]
    #[should_panic(expected = "differ in length")]
    fn overlaps_reject_length_mismatch() {
        let _ = Overlaps::compute(&[1, 2], &[1]);
    }

    #[test]
    fn empty_maps_are_empty_histograms() {
        let o = Overlaps::compute(&[], &[]);
        assert_eq!(o.iter().count(), 0);
        assert!(o.pred_areas().is_empty());
        assert_eq!(match_segments(&[], &[], &o), SegmentMatches::default());
    }

    /// The void term is what makes this a different formula from `sim`'s:
    /// a prediction covering gt plus void pixels still scores 1.0.
    #[test]
    fn pq_iou_discounts_void_overlap() {
        assert_eq!(pq_iou(10, 10, 15, 5), 1.0);
        assert_eq!(pq_iou(10, 10, 15, 0), 10.0 / 15.0);
        assert_eq!(pq_iou(0, 10, 15, 0), 0.0);
    }

    #[test]
    fn pq_iou_with_an_understated_gt_area_is_not_a_score() {
        // 10 pixels found, 2 stated: union = 4 + 2 - 10 - 0 < 0.
        assert!(pq_iou(10, 2, 4, 0) < 0.0);
        // union exactly zero: not finite.
        assert!(!pq_iou(10, 2, 8, 0).is_finite());
    }

    #[test]
    fn exactly_half_is_not_a_match() {
        // gt 1 and pred 7 both cover 3 pixels and share 2, with no void
        // anywhere: iou = 2 / (3 + 3 - 2) = 0.5 exactly.
        let gt = [1, 1, 1, 2];
        let pred = [7, 7, 8, 7];
        let o = Overlaps::compute(&gt, &pred);
        let m = match_segments(
            &[seg(1, 5, 3), seg(2, 6, 1)],
            &[seg(7, 5, 3), seg(8, 6, 1)],
            &o,
        );
        assert!(m.matched.is_empty(), "0.5 must not match: {m:?}");
        assert_eq!(m.missed, vec![0, 1]);
        // Nothing lies on void, so both predictions are false positives.
        assert_eq!(m.false_positives, vec![0, 1]);
    }

    #[test]
    fn category_must_agree() {
        let gt = [1, 1, 1, 1];
        let pred = [7, 7, 7, 7];
        let o = Overlaps::compute(&gt, &pred);
        let m = match_segments(&[seg(1, 5, 4)], &[seg(7, 6, 4)], &o);
        assert!(m.matched.is_empty());
        assert_eq!(m.missed, vec![0]);
        assert_eq!(m.false_positives, vec![0]);
    }

    #[test]
    fn crowd_absorbs_a_prediction_of_its_category_and_is_never_a_miss() {
        // gt 1 is crowd of category 5; pred 7 lies 3/4 on it.
        let gt = [1, 1, 1, 0];
        let pred = [7, 7, 7, 7];
        let o = Overlaps::compute(&gt, &pred);
        let crowd = Segment {
            label: 1,
            category_id: 5,
            area: 3,
            iscrowd: true,
        };
        let m = match_segments(&[crowd], &[seg(7, 5, 4)], &o);
        assert!(m.matched.is_empty());
        assert!(m.missed.is_empty(), "crowd is not a miss");
        assert_eq!(m.ignored, vec![0]);
        assert!(m.false_positives.is_empty());

        // Same geometry, other category: nothing absorbs it, 1/4 void -> FP.
        let m = match_segments(&[crowd], &[seg(7, 6, 4)], &o);
        assert_eq!(m.false_positives, vec![0]);
    }

    #[test]
    fn last_crowd_segment_of_a_category_is_the_one_that_absorbs() {
        // Two crowd segments of category 5. Pred 7 lies on the first; the
        // reference's dict keeps the last, so pred 7 is not absorbed.
        let gt = [1, 1, 2, 2];
        let pred = [7, 7, 0, 0];
        let o = Overlaps::compute(&gt, &pred);
        let crowd = |label| Segment {
            label,
            category_id: 5,
            area: 2,
            iscrowd: true,
        };
        let m = match_segments(&[crowd(1), crowd(2)], &[seg(7, 5, 2)], &o);
        assert_eq!(m.false_positives, vec![0]);
        let m = match_segments(&[crowd(2), crowd(1)], &[seg(7, 5, 2)], &o);
        assert_eq!(m.ignored, vec![0]);
    }

    #[test]
    fn mostly_void_prediction_is_ignored() {
        let gt = [0, 0, 0, 1];
        let pred = [7, 7, 7, 7];
        let o = Overlaps::compute(&gt, &pred);
        let m = match_segments(&[seg(1, 5, 1)], &[seg(7, 5, 4)], &o);
        // iou = 1 / (4 + 1 - 1 - 3) = 1.0: matched despite 3/4 void.
        assert_eq!(m.matched.len(), 1);
        assert_eq!(m.matched[0].2, 1.0);

        // Different category: unmatched, 3/4 void -> ignored, not FP.
        let m = match_segments(&[seg(1, 5, 1)], &[seg(7, 6, 4)], &o);
        assert_eq!(m.ignored, vec![0]);
        assert_eq!(m.missed, vec![0]);
    }

    #[test]
    fn unknown_labels_are_skipped_and_gt_unknowns_are_not_void() {
        // gt label 9 has no segment; pred 7 lies half on it, half on gt 1.
        let gt = [9, 9, 1, 1];
        let pred = [7, 7, 7, 7];
        let o = Overlaps::compute(&gt, &pred);
        let m = match_segments(&[seg(1, 5, 2)], &[seg(7, 5, 4)], &o);
        // iou = 2 / (4 + 2 - 2 - 0) = 0.5: not a match, since 9 is not void.
        assert!(m.matched.is_empty());
        // And 0/4 void -> false positive, not ignored.
        assert_eq!(m.false_positives, vec![0]);
    }

    #[test]
    fn gt_area_comes_from_the_segment_not_the_map() {
        // The map has 4 gt pixels; the segment claims 8 (a JSON/PNG mismatch
        // panopticapi would also take at face value).
        let gt = [1, 1, 1, 1];
        let pred = [7, 7, 7, 7];
        let o = Overlaps::compute(&gt, &pred);
        let m = match_segments(&[seg(1, 5, 8)], &[seg(7, 5, 4)], &o);
        // iou = 4 / (4 + 8 - 4) = 0.5 -> no match.
        assert!(m.matched.is_empty());
        let m = match_segments(&[seg(1, 5, 4)], &[seg(7, 5, 4)], &o);
        assert_eq!(m.matched, vec![(0, 0, 1.0)]);
    }
}
