//! Similarity kernels and the `SimKind` geometry axis.
//!
//! # Similarity contract
//!
//! Every kernel returns a dense `[D][G]` matrix (detections × ground-truths) of
//! similarities that are:
//! - in `[0, 1]`, higher = better,
//! - directly threshold-comparable (a match is `sim >= threshold`),
//! - crowd/ignore-aware **for the IoU kernels**: the per-column `iscrowd` flag
//!   switches that column's formula (see [intersection-over-area](#crowd-columns-are-intersection-over-area)),
//!   exactly as pycocotools does. [`oks_matrix`] is the documented exception — it
//!   takes no flags, because pycocotools handles keypoint crowd/ignore in
//!   `evaluateImg` (via `gtIg`) rather than in `computeOks`. Adding a parameter
//!   later would be a breaking change, so the asymmetry is stated rather than
//!   papered over.
//!
//! # Which argument the flags index
//!
//! The flag slice indexes the **second** argument, whatever role it plays —
//! written `gt` here because detection passes ground truth there. IoU is symmetric
//! in the non-crowd branch, so a family may swap the arguments to get a
//! `[GT][tracker]`-oriented matrix (TrackEval's convention), but then the flags
//! describe *that* second argument. [`oks_matrix`] is asymmetric and cannot be
//! swapped at all.
//!
//! # Crowd columns are intersection-over-area
//!
//! For a flagged column the formula is `intersection / area(first argument)` —
//! that is, IoA rather than IoU. Detection calls this "crowd", but the quantity is general:
//! it is exactly what MOT preprocessing needs for distractor suppression
//! (TrackEval's `do_ioa=True`). A tracking driver wanting IoA against ignore
//! regions should call `bbox_iou(dets, ignore_regions, &vec![true; n])` rather
//! than write its own.
//!
//! Distance-based similarities (the 3D seam) enter later via a normalization
//! adapter (`1 - d / d_max`); they are not part of this slice.
//!
//! # BYO matrix
//!
//! A family that computes similarity by some other means supplies its own
//! `[D][G]` matrix directly (a `&[Vec<f64>]`) instead of a [`SimKind`]. Such a
//! matrix must already satisfy the contract above, including pre-encoding any
//! crowd/ignore formula changes — kernels bake that in, BYO matrices must too.
//!
//! # Kinds vs. IoU types
//!
//! [`SimKind`] is the *geometry* axis (how similarity is computed). It is
//! deliberately distinct from [`crate::params::IouType`] (the eval-config axis):
//! `IouType::Segm` is computed with `SimKind::Mask`, `IouType::Keypoints` with
//! `SimKind::Oks`. Parsers accept `"segm"` as an alias for `"mask"`.
//!
//! # Where the math lives
//!
//! This module **defines** every similarity formula; [`crate::mask`] and
//! [`crate::geometry`] own only the mechanics (RLE codec, polygon clipping) and
//! re-export these kernels under their historic paths as one-way path sugar.
//! Those historic paths are Tier-1 drop-in surface mirroring `pycocotools.mask`,
//! so **the signatures of [`mask_iou`], [`bbox_iou`], and [`obb_iou`] are
//! effectively frozen** — buffer-reusing or batched variants must be added under
//! new names, never by changing these.

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};

use rayon::iter::MinLen;
use rayon::prelude::*;

use crate::geometry::{obb_intersection_area, obb_to_corners};
use crate::mask::{MaskStats, columns_overlap, intersection_area, intersection_within, mask_stats};
use crate::types::Rle;

/// Minimum work before a fan-out switches from sequential to parallel (rayon):
/// the D×G product for a similarity kernel, or the annotation count for a
/// per-annotation pass. Below this threshold, thread dispatch overhead exceeds
/// the parallelism benefit. One constant for every site — the per-item work is
/// independent, so this only trades dispatch overhead, never results.
const MIN_PARALLEL_WORK: usize = 1024;

/// Whether `work` items are worth a rayon fan-out.
///
/// `pub(crate)` so every fan-out in the crate gates on the one
/// [`MIN_PARALLEL_WORK`] instead of redeclaring it, which the guard in
/// `tests/architecture.rs` rejects.
#[inline]
pub(crate) fn worth_parallel(work: usize) -> bool {
    work >= MIN_PARALLEL_WORK
}

/// The process that started rayon's thread pool, or 0 while none has.
static POOL_OWNER: AtomicU32 = AtomicU32::new(0);

/// Whether rayon's thread pool can run a fan-out in this process.
///
/// A child that `fork()` makes inherits the pool but none of its threads, so
/// a fan-out there queues work no thread takes and waits forever: a PyTorch
/// `DataLoader` worker on Linux, forked from a process that already
/// evaluated, decoding masks. The first fan-out records its process, before
/// the pool starts; a process whose id is not the recorded one is such a
/// child, and runs every fan-out on the calling thread. A child forked
/// before any fan-out records itself and starts a pool of its own.
///
/// No fork hook is needed, so a fork made from C or from Rust is covered as
/// well as `os.fork()`. Every touch of the pool goes through this: the
/// iterators through [`ForkSafe::fork_safe`], and [`join`],
/// [`drop_in_background`], and [`pool_threads`]. `tests/architecture.rs`
/// fails on a parallel iterator without `fork_safe` and on any other use of
/// rayon outside this file.
pub(crate) fn pool_usable() -> bool {
    let me = std::process::id();
    match POOL_OWNER.load(Ordering::Relaxed) {
        0 => match POOL_OWNER.compare_exchange(0, me, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => true,
            Err(owner) => owner == me,
        },
        owner => owner == me,
    }
}

/// How every parallel iterator in the crate reaches rayon's pool.
pub(crate) trait ForkSafe: IndexedParallelIterator {
    /// The iterator unchanged where the pool is usable, and run as one job on
    /// the calling thread where it is not ([`pool_usable`]). The result is
    /// the same either way: a fan-out already gives the same answer on any
    /// number of threads.
    fn fork_safe(self) -> MinLen<Self> {
        // rayon splits no job below its minimum length, and never splitting
        // runs the whole job on the calling thread without reaching a worker.
        self.with_min_len(if pool_usable() { 1 } else { usize::MAX })
    }
}

impl<I: IndexedParallelIterator> ForkSafe for I {}

/// [`rayon::join`] where the pool is usable ([`pool_usable`]), and `a` then
/// `b` on the calling thread where it is not.
pub(crate) fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    if pool_usable() {
        rayon::join(a, b)
    } else {
        (a(), b())
    }
}

/// Free `value` on the pool, off the caller's thread, or on the caller's
/// thread where the pool is not usable ([`pool_usable`]): a job spawned
/// there would never run, and the value would never be freed.
pub(crate) fn drop_in_background<T: Send + 'static>(value: T) {
    if pool_usable() {
        rayon::spawn(move || drop(value));
    } else {
        drop(value);
    }
}

/// How many threads a fan-out runs on: the pool's, or 1 where the pool is
/// not usable ([`pool_usable`]).
pub(crate) fn pool_threads() -> usize {
    if pool_usable() {
        rayon::current_num_threads()
    } else {
        1
    }
}

/// The shared IoU closing formula, identical across bbox/mask/OBB.
///
/// When the column is flagged, this is **intersection-over-area**: divide by
/// `dt_area` alone (a detection inside a crowd region is not penalized for the
/// region's extent). Otherwise it is standard intersection-over-union. See the
/// [module note](self#crowd-columns-are-intersection-over-area) — the IoA branch
/// is reusable beyond detection's crowd semantics.
#[inline]
fn iou_from_areas(inter: f64, dt_area: f64, gt_area: f64, gt_is_crowd: bool) -> f64 {
    if gt_is_crowd {
        if dt_area == 0.0 { 0.0 } else { inter / dt_area }
    } else {
        let union = dt_area + gt_area - inter;
        if union == 0.0 { 0.0 } else { inter / union }
    }
}

/// IoU of two sets from their sizes and their overlap, for a caller holding
/// counts rather than geometry: the pixels of one class on each side of a
/// confusion matrix, or two segment areas. The same closing formula as
/// [`bbox_iou_pair`] and the mask kernel, so a count-based IoU is never
/// written out again. `0.0` when both sets are empty.
#[inline]
pub fn iou_from_counts(intersection: u64, a: u64, b: u64) -> f64 {
    iou_from_areas(intersection as f64, a as f64, b as f64, false)
}

/// IoU of two weighted sets from their total weights and their overlap, or
/// `0.0` where the union is not above `floor`: [`iou_from_counts`] for
/// counts that are sums of similarities rather than whole numbers.
///
/// HOTA's association counts are such sums. TrackEval normalizes each frame's
/// similarities by `sim / (column sum + row sum − sim)` where that
/// denominator is above `f64::EPSILON`, and scores two tracks by
/// `overlap / (a + b − overlap)` over the whole sequence; both are this
/// formula, with the floor TrackEval's comparison sets.
#[inline]
pub(crate) fn iou_from_weights(overlap: f64, a: f64, b: f64, floor: f64) -> f64 {
    let union = a + b - overlap;
    if union > floor { overlap / union } else { 0.0 }
}

/// IoU of two `[x, y, w, h]` boxes as TrackEval's `_calculate_box_ious`
/// computes it: the box similarity of the tracking family.
///
/// It is not [`bbox_iou_pair`], and the difference moves last bits, which
/// moves a pair across a threshold when it sits on one:
///
/// - each box is turned into corners first, and its area is taken from them,
///   `((x + w) − x) × ((y + h) − y)`, which is not always `w × h`;
/// - a pair scores `0.0` when either area, or the union, is at most
///   `f64::EPSILON`, where [`bbox_iou_pair`] tests the union against zero.
#[inline]
pub(crate) fn bbox_iou_corners(a: [f64; 4], b: [f64; 4]) -> f64 {
    let (ax1, ay1) = (a[0] + a[2], a[1] + a[3]);
    let (bx1, by1) = (b[0] + b[2], b[1] + b[3]);
    let iw = (ax1.min(bx1) - a[0].max(b[0])).max(0.0);
    let ih = (ay1.min(by1) - a[1].max(b[1])).max(0.0);
    let intersection = iw * ih;
    let area_a = corner_area(a);
    let area_b = corner_area(b);
    let union = area_a + area_b - intersection;
    if area_a <= f64::EPSILON || area_b <= f64::EPSILON || union <= f64::EPSILON {
        0.0
    } else {
        intersection / union
    }
}

/// An `[x, y, w, h]` box's area as [`bbox_iou_corners`] takes it, from its
/// corners: `((x + w) − x) × ((y + h) − y)`. With `x` and `y` finite, a box
/// whose corner area is finite has finite corners, and no two such boxes
/// have a NaN IoU: the intersection is at most either area.
#[inline]
pub(crate) fn corner_area(b: [f64; 4]) -> f64 {
    ((b[0] + b[2]) - b[0]) * ((b[1] + b[3]) - b[1])
}

/// [`bbox_iou_corners`] of every box in `a` against every box in `b`, as one
/// row-major `a.len() × b.len()` buffer, empty when either side is.
pub(crate) fn bbox_iou_corners_flat(a: &[[f64; 4]], b: &[[f64; 4]]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    rows(a.len(), b.len(), |i, row| {
        for (out, &bj) in row.iter_mut().zip(b) {
            *out = bbox_iou_corners(a[i], bj);
        }
    })
}

/// Run a per-detection-row kernel into one row-major `d × g` buffer, going
/// parallel only past [`MIN_PARALLEL_WORK`]. `fill_row(i, row)` writes
/// detection `i`'s `g` values.
///
/// `pub(crate)` so a future kernel outside this module (panoptic, a
/// distance-derived kind) can obey the one-threshold rule that
/// `tests/architecture.rs` enforces, instead of being forced to redeclare the
/// constant and trip the guard.
#[inline]
pub(crate) fn rows<F>(d: usize, g: usize, fill_row: F) -> Vec<f64>
where
    F: Fn(usize, &mut [f64]) + Sync + Send,
{
    let mut out = vec![0.0; d * g];
    if g == 0 {
        return out;
    }
    if worth_parallel(d * g) {
        out.par_chunks_mut(g)
            .fork_safe()
            .enumerate()
            .for_each(|(i, row)| fill_row(i, row));
    } else {
        for (i, row) in out.chunks_mut(g).enumerate() {
            fill_row(i, row);
        }
    }
    out
}

/// A row-major `d × g` buffer as one vector per detection: the shape the
/// public kernels return.
fn nested(flat: Vec<f64>, d: usize, g: usize) -> Vec<Vec<f64>> {
    if g == 0 {
        return vec![Vec::new(); d];
    }
    flat.chunks_exact(g).map(<[f64]>::to_vec).collect()
}

/// Compute IoU between `dt` and `gt` RLE masks.
///
/// Returns a D×G matrix (row-major, `dt.len()` rows, `gt.len()` columns).
/// For `iscrowd[j] == true`, uses crowd IoU: intersection / area(dt) instead of
/// intersection / union.
///
/// A pair whose masks differ in `h` or `w` gets `-1.0`, as pycocotools
/// `rleIou` returns for it — the overlap of two masks on different canvases is
/// undefined, and a run-stream walk over them would report a plausible number.
pub fn mask_iou(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    nested(mask_iou_flat(dt, gt, iscrowd), dt.len(), gt.len())
}

/// [`mask_iou`] as one row-major `D×G` buffer, empty when either side is:
/// the shape a dense array wants, with no vector per detection.
pub fn mask_iou_flat(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<f64> {
    mask_iou_flat_of(dt, gt, iscrowd)
}

/// The IoU pycocotools' `rleIou` gives two masks of different sizes: `-1`
/// where their boxes overlap, and the boxes' IoU, `0`, where they do not —
/// its box IoU gates every pair before the size is looked at. The overlap is
/// its `bbIou`'s: a positive width and height, with crowd or without.
fn size_mismatch_iou(dt: &Rle, gt: &Rle) -> f64 {
    let (d, g) = (crate::mask::to_bbox(dt), crate::mask::to_bbox(gt));
    let w = (d[2] + d[0]).min(g[2] + g[0]) - d[0].max(g[0]);
    let h = (d[3] + d[1]).min(g[3] + g[1]) - d[1].max(g[1]);
    if w > 0.0 && h > 0.0 { -1.0 } else { 0.0 }
}

/// [`mask_iou_flat`] over masks held any way that lends an [`Rle`]: the
/// evaluator's, borrowed from its cache of converted masks rather than copied
/// out of it for every cell.
pub(crate) fn mask_iou_flat_of<R: std::borrow::Borrow<Rle> + Sync>(
    dt: &[R],
    gt: &[R],
    iscrowd: &[bool],
) -> Vec<f64> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return Vec::new();
    }

    // Each mask's area, and from the same walk the columns it spans and
    // whether its runs stay within the mask. A pair whose spans are apart
    // intersects in no pixel, so its intersection is 0 without a walk; any
    // other pair of in-bounds masks takes the foreground-interval merge, and
    // the IoU comes out of the same formula either way.
    let dt_meta: Vec<MaskStats> = dt.iter().map(|r| mask_stats(r.borrow())).collect();
    let gt_meta: Vec<MaskStats> = gt.iter().map(|r| mask_stats(r.borrow())).collect();

    rows(d, g, |i, row| {
        let (a, dt) = (&dt_meta[i], dt[i].borrow());
        for (j, out) in row.iter_mut().enumerate() {
            let gt = gt[j].borrow();
            *out = if dt.h != gt.h || dt.w != gt.w {
                size_mismatch_iou(dt, gt)
            } else {
                let b = &gt_meta[j];
                let inter = if !columns_overlap(a.columns, b.columns) {
                    0.0
                } else if a.within && b.within {
                    intersection_within(dt, gt) as f64
                } else {
                    intersection_area(dt, gt) as f64
                };
                iou_from_areas(inter, a.area as f64, b.area as f64, iscrowd[j])
            };
        }
    })
}

/// IoU between two axis-aligned bounding boxes, each `[x, y, w, h]`.
///
/// The scalar counterpart of [`bbox_iou`] — the one place a single-pair bbox IoU
/// is defined, so analysis code never hand-rolls it.
///
/// `#[inline]` is load-bearing, not decoration: [`bbox_iou`]'s inner loop only
/// auto-vectorizes because this body is inlined into it. Without inlining the
/// matrix kernel degrades to one call per pair, each copying two `[f64; 4]` by
/// value.
#[inline]
pub fn bbox_iou_pair(a: [f64; 4], b: [f64; 4], b_is_crowd: bool) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = (a[0] + a[2]).min(b[0] + b[2]);
    let y2 = (a[1] + a[3]).min(b[1] + b[3]);
    let iw = (x2 - x1).max(0.0);
    let ih = (y2 - y1).max(0.0);
    iou_from_areas(iw * ih, a[2] * a[3], b[2] * b[3], b_is_crowd)
}

/// Compute bbox IoU between sets of bounding boxes.
///
/// Each bbox is `[x, y, w, h]`. Returns D×G matrix.
pub fn bbox_iou(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    nested(bbox_iou_flat(dt, gt, iscrowd), dt.len(), gt.len())
}

/// [`bbox_iou`] as one row-major `D×G` buffer, empty when either side is:
/// the shape a dense array wants, with no vector per detection.
pub fn bbox_iou_flat(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[bool]) -> Vec<f64> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return Vec::new();
    }

    rows(d, g, |i, row| {
        for (j, out) in row.iter_mut().enumerate() {
            *out = bbox_iou_pair(dt[i], gt[j], iscrowd[j]);
        }
    })
}

/// IoU between two pre-computed rotated rectangle corner sets.
///
/// `area_a` and `area_b` are the rectangle areas (w × h).
/// Returns 0.0 for zero-area boxes or non-overlapping boxes.
pub(crate) fn obb_iou_pair(
    corners_a: &[(f64, f64); 4],
    area_a: f64,
    corners_b: &[(f64, f64); 4],
    area_b: f64,
    b_is_crowd: bool,
) -> f64 {
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }

    let inter_area = obb_intersection_area(corners_a, corners_b);
    if inter_area <= 0.0 {
        return 0.0;
    }

    iou_from_areas(inter_area, area_a, area_b, b_is_crowd)
}

/// Compute D×G IoU matrix for oriented bounding boxes.
///
/// `dt` contains detection OBBs `[cx, cy, w, h, angle]`, `gt` contains ground truth OBBs,
/// and `iscrowd` indicates whether each GT is a crowd annotation (one per GT).
///
/// When `iscrowd[j]` is true, IoU = intersection / dt_area (matching bbox crowd semantics).
pub fn obb_iou(dt: &[[f64; 5]], gt: &[[f64; 5]], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    nested(obb_iou_flat(dt, gt, iscrowd), dt.len(), gt.len())
}

/// [`obb_iou`] as one row-major `D×G` buffer, empty when either side is.
pub(crate) fn obb_iou_flat(dt: &[[f64; 5]], gt: &[[f64; 5]], iscrowd: &[bool]) -> Vec<f64> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return Vec::new();
    }

    // Pre-compute GT corners and areas (loop-invariant over DT rows).
    let gt_corners: Vec<[(f64, f64); 4]> = gt.iter().map(obb_to_corners).collect();
    let gt_areas: Vec<f64> = gt.iter().map(|b| b[2] * b[3]).collect();

    rows(d, g, |i, row| {
        let corners_a = obb_to_corners(&dt[i]);
        let area_a = dt[i][2] * dt[i][3];
        for (j, out) in row.iter_mut().enumerate() {
            *out = obb_iou_pair(&corners_a, area_a, &gt_corners[j], gt_areas[j], iscrowd[j]);
        }
    })
}

/// The geometry axis for similarity: which built-in kernel computes the matrix.
///
/// Each kind maps to a matrix kernel in this module: [`bbox_iou`], [`mask_iou`],
/// [`obb_iou`], [`oks_matrix`].
///
/// Marked `#[non_exhaustive]`: later families are expected to add kinds (a
/// panoptic kind, a distance-derived kind for 3D), and downstream code must not
/// be broken by that. Match with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SimKind {
    /// Axis-aligned bounding boxes — [`bbox_iou`].
    Bbox,
    /// Segmentation masks (RLE) — [`mask_iou`].
    Mask,
    /// Oriented bounding boxes — [`obb_iou`].
    Obb,
    /// Object keypoint similarity (pose) — [`oks_matrix`].
    Oks,
}

impl SimKind {
    /// The canonical lowercase name (round-trips through [`FromStr`]).
    pub fn as_str(self) -> &'static str {
        match self {
            SimKind::Bbox => "bbox",
            SimKind::Mask => "mask",
            SimKind::Obb => "obb",
            SimKind::Oks => "oks",
        }
    }
}

impl fmt::Display for SimKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SimKind {
    type Err = String;

    /// Parse a `SimKind`. Accepts `"segm"` as an alias for `"mask"` and
    /// `"keypoints"` as an alias for `"oks"`, so eval-config strings map cleanly
    /// onto the geometry axis.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bbox" => Ok(SimKind::Bbox),
            "mask" | "segm" => Ok(SimKind::Mask),
            "obb" => Ok(SimKind::Obb),
            "oks" | "keypoints" => Ok(SimKind::Oks),
            _ => Err(format!(
                "Unknown sim kind: '{s}'. Expected 'bbox', 'mask' (alias 'segm'), \
                 'obb', or 'oks' (alias 'keypoints')"
            )),
        }
    }
}

/// Object Keypoint Similarity (OKS) matrix, `[D][G]` (detections × ground-truths).
///
/// A ground-truth pose instance — the typed GT unit for [`oks_matrix`],
/// analogous to `[f64; 4]` for bboxes or `Rle` for masks. Keeps the keypoints,
/// area, and bbox of one instance together instead of in caller-aligned
/// parallel arrays.
#[derive(Debug, Clone, Copy)]
pub struct GtPose<'a> {
    /// Flat keypoints `[x, y, v, x, y, v, ...]` (length `3*K`); empty = skip.
    pub keypoints: &'a [f64],
    /// Object area — the OKS scale denominator.
    pub area: f64,
    /// Bounding box `[x, y, w, h]`, used only when no GT keypoints are visible.
    pub bbox: [f64; 4],
}

/// COCO-decoupled port of pycocotools' `computeOks`, matrix-shaped so pose
/// tracking (and any future keypoint family) can reuse it without a `COCO`
/// object. The geometric IoU kinds are pure functions already; OKS was the one
/// kernel still buried inside the eval monolith, so it is lifted here and the
/// monolith delegates to it.
///
/// Each detection is a flat keypoint slice (`[x, y, v, ...]`, length `3*K`) —
/// OKS is asymmetric, so detections need only keypoints while ground-truths
/// carry area and bbox too (a [`GtPose`]). `sigmas` has length `K`. Detection
/// visibility is ignored (only GT visibility gates the average). An empty
/// keypoint slice (GT or DT) leaves that column/row zero.
///
/// Definition (per pycocotools): `vars = (2σ)²`; per keypoint
/// `e = (dx² + dy²) / vars / (area + ε) / 2`, `OKS = mean(exp(-e))`. If the GT
/// has no visible keypoints (`k1 == 0`), distance is measured to the GT bbox
/// (doubled) instead, over all keypoints.
///
/// Similarity is in `[0, 1]`, higher = better — the [module contract](self).
pub fn oks_matrix(dt_keypoints: &[&[f64]], gt: &[GtPose<'_>], sigmas: &[f64]) -> Vec<Vec<f64>> {
    nested(
        oks_matrix_flat(dt_keypoints, gt, sigmas),
        dt_keypoints.len(),
        gt.len(),
    )
}

/// [`oks_matrix`] as one row-major `D×G` buffer, empty when either side is.
pub(crate) fn oks_matrix_flat(
    dt_keypoints: &[&[f64]],
    gt: &[GtPose<'_>],
    sigmas: &[f64],
) -> Vec<f64> {
    let num_kpts = sigmas.len();
    // vars = (sigmas * 2)**2 = 4 * sigma^2  (matching pycocotools)
    let vars: Vec<f64> = sigmas.iter().map(|s| (2.0 * s).powi(2)).collect();

    let d = dt_keypoints.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return Vec::new();
    }

    /// Per-GT quantities that are loop-invariant over detection rows, hoisted
    /// so the detection-major dispatch below does not recompute them `d` times.
    struct GtPrep<'a> {
        kpts: &'a [f64],
        /// `area + ε`, the OKS scale denominator.
        area: f64,
        /// Number of visible GT keypoints; 0 selects the bbox-distance branch.
        k1: usize,
        /// Ignore-region bounds (double the GT bbox), used only when `k1 == 0`.
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    }

    let prep: Vec<GtPrep<'_>> = gt
        .iter()
        .map(|gt_pose| {
            let gt_kpts = gt_pose.keypoints;
            let bb = gt_pose.bbox;
            let k1 = (0..num_kpts)
                .filter(|&ki| gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0)
                .count();
            GtPrep {
                kpts: gt_kpts,
                area: gt_pose.area + f64::EPSILON,
                k1,
                x0: bb[0] - bb[2],
                x1: bb[0] + bb[2] * 2.0,
                y0: bb[1] - bb[3],
                y1: bb[1] + bb[3] * 2.0,
            }
        })
        .collect();

    // Detection-major through the shared parallel dispatcher, like every
    // sibling kernel. The per-cell arithmetic is order-independent, so this
    // matches the historic GT-major loop bit for bit.
    rows(d, g, |i, row| {
        let dt_kpts = dt_keypoints[i];
        let oks = |p: &GtPrep<'_>| {
            if dt_kpts.is_empty() || p.kpts.is_empty() {
                return 0.0;
            }

            let mut oks_sum = 0.0_f64;
            let mut oks_count = 0_usize;

            for (ki, &var_k) in vars.iter().enumerate() {
                // When k1 > 0, only include visible GT keypoints.
                let visible = p.kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0;
                if p.k1 > 0 && !visible {
                    continue;
                }

                let gx = p.kpts.get(ki * 3).copied().unwrap_or(0.0);
                let gy = p.kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);
                let xd = dt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                let yd = dt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);

                let (dx, dy) = if p.k1 > 0 {
                    (xd - gx, yd - gy)
                } else {
                    // No visible GT keypoints: distance to bbox boundary.
                    let dx = 0.0_f64.max(p.x0 - xd) + 0.0_f64.max(xd - p.x1);
                    let dy = 0.0_f64.max(p.y0 - yd) + 0.0_f64.max(yd - p.y1);
                    (dx, dy)
                };

                let e = (dx * dx + dy * dy) / var_k / p.area / 2.0;
                oks_sum += (-e).exp();
                oks_count += 1;
            }

            if oks_count > 0 {
                oks_sum / oks_count as f64
            } else {
                0.0
            }
        };
        for (out, p) in row.iter_mut().zip(&prep) {
            *out = oks(p);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which pool threads ran a fan-out of 10 000 items: `None` for the
    /// calling thread.
    fn threads_that_ran() -> Vec<Option<usize>> {
        let mut ran: Vec<Option<usize>> = (0..10_000)
            .into_par_iter()
            .fork_safe()
            .map(|_| rayon::current_thread_index())
            .collect();
        ran.sort_unstable();
        ran.dedup();
        ran
    }

    /// A process that started the pool, or one forked before any fan-out,
    /// runs on the pool; a child forked after its parent started it runs on
    /// the calling thread, and so do `join`, `pool_threads`, and
    /// `drop_in_background`. A fork is simulated by recording another
    /// process as the pool's owner; the one test that does so restores it.
    #[test]
    fn a_child_forked_after_the_pool_started_runs_on_the_calling_thread() {
        let me = std::process::id();
        POOL_OWNER.store(0, Ordering::Relaxed);
        assert!(pool_usable(), "the first fan-out starts the pool");
        assert_eq!(POOL_OWNER.load(Ordering::Relaxed), me);
        assert!(threads_that_ran().iter().all(Option::is_some));

        POOL_OWNER.store(me.wrapping_add(1), Ordering::Relaxed);
        assert!(!pool_usable());
        assert_eq!(threads_that_ran(), vec![None]);
        assert_eq!(pool_threads(), 1);
        let on = join(rayon::current_thread_index, rayon::current_thread_index);
        assert_eq!(on, (None, None));
        drop_in_background(vec![0u8; 16]);

        POOL_OWNER.store(me, Ordering::Relaxed);
        assert!(pool_usable());
    }
    use crate::mask::area as rle_area;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The extent prefilter changes no mask IoU: every pair against the run
    /// walk on every pair, over masks that overlap, touch, sit apart, span
    /// columns (whose extents cover every row), are empty, are crowd regions,
    /// or are a different size.
    #[test]
    fn mask_iou_skips_only_pairs_that_share_no_pixel() {
        let mut rng = StdRng::seed_from_u64(0xE7_7E47);
        let (h, w) = (37u32, 41u32);
        let mask = |rng: &mut StdRng| -> crate::types::Rle {
            let mut m = vec![0u8; (h * w) as usize];
            match rng.random_range(0..5) {
                0 => {}
                1 => m.iter_mut().for_each(|p| *p = 1),
                2 => {
                    for p in &mut m {
                        *p = u8::from(rng.random_bool(0.03));
                    }
                }
                _ => {
                    let (x0, y0) = (rng.random_range(0..w), rng.random_range(0..h));
                    let (x1, y1) = (rng.random_range(x0..=w), rng.random_range(y0..=h));
                    for x in x0..x1 {
                        for y in y0..y1 {
                            m[(x * h + y) as usize] = 1;
                        }
                    }
                }
            }
            crate::mask::encode(&m, h, w).expect("a mask the canvas's size encodes")
        };
        for trial in 0..40 {
            let mut dt: Vec<_> = (0..rng.random_range(1..12))
                .map(|_| mask(&mut rng))
                .collect();
            let mut gt: Vec<_> = (0..rng.random_range(1..9))
                .map(|_| mask(&mut rng))
                .collect();
            if trial % 7 == 0 {
                gt.push(crate::mask::encode(&[1, 0, 1, 1], 2, 2).expect("a 2x2 mask encodes"));
            }
            // A mask of no pixels whose runs still claim some — no column
            // span when `h` is 0, runs past `h * w` when `w` is — against
            // its like: no intersection, as the walk counts none.
            if trial % 5 == 0 {
                for (h, w) in [(0, 5), (5, 0)] {
                    let empty = crate::types::Rle {
                        h,
                        w,
                        counts: vec![0, 3],
                    };
                    dt.push(empty.clone());
                    gt.push(empty);
                }
            }
            let iscrowd: Vec<bool> = gt.iter().map(|_| rng.random_bool(0.2)).collect();
            let got = mask_iou_flat(&dt, &gt, &iscrowd);
            // Two extents meet when both are positive and each starts before
            // the other ends.
            let meet = |a: f64, a_len: f64, b: f64, b_len: f64| {
                a_len > 0.0 && b_len > 0.0 && a < b + b_len && b < a + a_len
            };
            for (i, d) in dt.iter().enumerate() {
                for (j, g) in gt.iter().enumerate() {
                    let want = if d.h != g.h || d.w != g.w {
                        // pycocotools' box IoU gates its size check: `-1`
                        // only where the two boxes overlap.
                        let (db, gb) = (crate::mask::to_bbox(d), crate::mask::to_bbox(g));
                        if meet(db[0], db[2], gb[0], gb[2]) && meet(db[1], db[3], gb[1], gb[3]) {
                            -1.0
                        } else {
                            0.0
                        }
                    } else {
                        iou_from_areas(
                            intersection_area(d, g) as f64,
                            rle_area(d) as f64,
                            rle_area(g) as f64,
                            iscrowd[j],
                        )
                    };
                    assert_eq!(
                        got[i * gt.len() + j].to_bits(),
                        want.to_bits(),
                        "trial {trial}: dt {i} vs gt {j}"
                    );
                }
            }
        }
    }

    /// The foreground-interval merge counts what the run-by-run walk counts
    /// for runs that end within the mask — zero-length runs, runs that stop
    /// short, runs that end exactly at `h * w` — and the kernel keeps the
    /// walk for runs that overrun it, whose last step the walk counts whole.
    #[test]
    fn mask_iou_merges_foreground_runs_as_the_run_walk_counts() {
        use crate::mask::{intersection_within, mask_stats};
        let mut rng = StdRng::seed_from_u64(0x1_0E7);
        let (h, w) = (9u32, 7u32);
        let n = (h * w) as u64;
        let runs = |rng: &mut StdRng, overrun: bool| -> crate::types::Rle {
            let mut counts = Vec::new();
            let mut total = 0u64;
            let budget = if overrun {
                n + rng.random_range(1..20)
            } else {
                rng.random_range(0..=n)
            };
            while total < budget {
                let c = rng.random_range(0..=6u32).min((budget - total) as u32);
                counts.push(c);
                total += u64::from(c);
                if rng.random_bool(0.05) {
                    counts.push(0);
                }
            }
            crate::types::Rle { h, w, counts }
        };
        for trial in 0..2000 {
            let a = runs(&mut rng, trial % 3 == 0);
            let b = runs(&mut rng, trial % 4 == 0);
            if mask_stats(&a).within && mask_stats(&b).within {
                assert_eq!(
                    intersection_within(&a, &b),
                    intersection_area(&a, &b),
                    "trial {trial}: {:?} vs {:?}",
                    a.counts,
                    b.counts
                );
            }
            let crowd = [trial % 2 == 0];
            let got = mask_iou_flat(std::slice::from_ref(&a), std::slice::from_ref(&b), &crowd);
            let want = iou_from_areas(
                intersection_area(&a, &b) as f64,
                rle_area(&a) as f64,
                rle_area(&b) as f64,
                crowd[0],
            );
            assert_eq!(got[0].to_bits(), want.to_bits(), "trial {trial}");
        }
    }

    /// A box with realistic detection extents: at least one pixel on a side.
    ///
    /// Sub-pixel boxes are excluded deliberately — see
    /// [`self_iou_degrades_for_subpixel_boxes`], which covers them and documents
    /// the different guarantee that applies there.
    fn rand_box(rng: &mut StdRng) -> [f64; 4] {
        [
            rng.random_range(0.0..1000.0),
            rng.random_range(0.0..1000.0),
            rng.random_range(1.0..200.0),
            rng.random_range(1.0..200.0),
        ]
    }

    /// The algebraic properties every caller assumes of the bbox kernel.
    ///
    /// Symmetry is scoped to non-crowd columns on purpose: a crowd column is
    /// deliberately intersection-over-*area* (see [`iou_from_areas`]), which is
    /// asymmetric by construction. `oks_matrix` is asymmetric unconditionally and
    /// is excluded for the same reason — a symmetry property here would be
    /// asserting something the design explicitly rejects.
    #[test]
    fn bbox_iou_algebraic_properties() {
        let mut rng = StdRng::seed_from_u64(0x1005);

        for case in 0..20000 {
            let a = rand_box(&mut rng);
            let b = rand_box(&mut rng);

            let iou_fwd = bbox_iou_pair(a, b, false);
            let iou_rev = bbox_iou_pair(b, a, false);

            // Not `<= 1.0`: the intersection is computed from differences of
            // coordinates, so a near-identical pair can round marginally above
            // the union. See the self-IoU note below for the mechanism.
            assert!(
                (0.0..=1.0 + 1e-12).contains(&iou_fwd),
                "case {case}: IoU {iou_fwd} outside [0,1] for {a:?} vs {b:?}"
            );
            assert!(
                (iou_fwd - iou_rev).abs() < 1e-12,
                "case {case}: asymmetric, {iou_fwd} vs {iou_rev} for {a:?} vs {b:?}"
            );

            // Self-IoU is 1.0 to within a few ulp, but *not* exactly 1.0. The
            // intersection width is `(x + w) - x`, which does not round-trip to
            // `w`: [94.13, 88.47, 21.53, 46.14] against itself gives
            // 0.9999999999999993. pycocotools computes it the same way, so this
            // is the reference's arithmetic, not a defect to correct here.
            //
            // For boxes at least a pixel on a side the deviation is bounded
            // around 1e-13, which clears `coco_match_floor(1.0)` by three orders
            // of magnitude — that is what lets a clamped caller match exact
            // duplicates at `t == 1.0`.
            let self_iou = bbox_iou_pair(a, a, false);
            assert!(
                (self_iou - 1.0).abs() < 1e-12,
                "case {case}: self-IoU {self_iou} not within 1e-12 of 1.0 for {a:?}"
            );
            assert!(
                self_iou >= crate::primitives::greedy::coco_match_floor(1.0),
                "case {case}: self-IoU {self_iou} falls below the match floor for {a:?}"
            );
        }
    }

    /// Sub-pixel boxes lose enough precision that a self-match at `t == 1.0`
    /// fails *even with* [`coco_match_floor`] applied.
    ///
    /// Pinned so the limit is a known quantity rather than a surprise. The
    /// intersection extent is `(y + h) - y`; when `h` is ~1e-5 against a
    /// coordinate ~1e2 the subtraction keeps almost none of `h`'s significand, and
    /// the self-IoU drifts by up to ~1.5e-9 — past the `1 - 1e-10` floor.
    ///
    /// No detection dataset has boxes this small (COCO's smallest annotations are
    /// ~1 pixel), and the affected threshold is exactly 1.0, which no standard
    /// metric sweep reaches. It matters only for a caller passing hand-built
    /// degenerate geometry at `t == 1.0`.
    ///
    /// [`coco_match_floor`]: crate::primitives::greedy::coco_match_floor
    #[test]
    fn self_iou_degrades_for_subpixel_boxes() {
        let thin = [
            225.205_188_785_783_66,
            691.079_072_122_579_8,
            11.209,
            2.431e-05,
        ];
        let self_iou = bbox_iou_pair(thin, thin, false);

        assert!(
            (self_iou - 1.0).abs() > 1e-10,
            "expected measurable drift for a sub-pixel box, got {self_iou}"
        );
        assert!(
            self_iou < crate::primitives::greedy::coco_match_floor(1.0),
            "expected the drift to fall below the match floor, got {self_iou}"
        );
        // Still far too close to 1.0 to affect any real threshold.
        assert!((self_iou - 1.0).abs() < 1e-8);
    }

    /// Rotated-box IoU against frozen Shapely/GEOS values.
    ///
    /// Oriented boxes have no reference *evaluator* — which is why `report()`
    /// marks them `Provenance::Extension` — but the geometry underneath is a
    /// solved problem, and Shapely is an independent implementation of it. So the
    /// AP cannot be validated while the kernel it rests on can.
    ///
    /// Asserts IoU *values*, not threshold sides, against an oracle that derives
    /// corners from the OBB definition independently of `crate::geometry` —
    /// frozen so no Python runs at test time.
    ///
    /// Regenerate with `uv run python scripts/gen_obb_fixtures.py`.
    #[test]
    fn obb_iou_matches_shapely() {
        #[derive(serde::Deserialize)]
        struct Case {
            kind: String,
            a: [f64; 5],
            b: [f64; 5],
            iou: f64,
        }

        let data = include_str!("testdata/obb_iou_shapely.json");
        let cases: Vec<Case> = serde_json::from_str(data).expect("parse fixture");
        assert!(cases.len() > 500, "fixture looks truncated");

        let mut worst = 0.0f64;
        let mut worst_case = String::new();
        for (i, c) in cases.iter().enumerate() {
            let got = obb_iou(&[c.a], &[c.b], &[false])[0][0];
            let diff = (got - c.iou).abs();
            if diff > worst {
                worst = diff;
                worst_case = format!(
                    "case {i} ({}): a={:?} b={:?} shapely={} hotcoco={got}",
                    c.kind, c.a, c.b, c.iou
                );
            }
            assert!(
                diff < 1e-9,
                "case {i} ({}): IoU {got} vs Shapely {} (diff {diff:.3e})\n  a={:?}\n  b={:?}",
                c.kind,
                c.iou,
                c.a,
                c.b
            );
        }
        // Surfaced on success too: a silent creep toward the tolerance is worth
        // seeing before it becomes a failure.
        if worst > 0.0 {
            println!("obb_iou worst deviation from Shapely: {worst:.3e} — {worst_case}");
        }
    }

    /// `rows()` switches to rayon at `MIN_PARALLEL_WORK`; both branches must agree.
    ///
    /// Straddles the threshold rather than testing one side of it, because the
    /// bug this guards against is a kernel that is only correct in the branch the
    /// small fixtures happen to take.
    #[test]
    fn bbox_iou_parallel_and_sequential_agree() {
        let mut rng = StdRng::seed_from_u64(0x9E37_79B9);

        // Straddle the threshold rather than hardcoding its current value: raising
        // MIN_PARALLEL_WORK would otherwise leave every pair on the sequential
        // side and silently stop covering the rayon branch.
        let n = (MIN_PARALLEL_WORK as f64).sqrt().ceil() as usize;
        for &(d, g) in &[(2, 2), (n - 1, n - 1), (n, n - 1), (n, n), (n + 1, n)] {
            let dt: Vec<[f64; 4]> = (0..d).map(|_| rand_box(&mut rng)).collect();
            let gt: Vec<[f64; 4]> = (0..g).map(|_| rand_box(&mut rng)).collect();
            let iscrowd: Vec<bool> = (0..g).map(|_| rng.random_bool(0.2)).collect();

            let matrix = bbox_iou(&dt, &gt, &iscrowd);

            assert_eq!(matrix.len(), d);
            for (di, row) in matrix.iter().enumerate() {
                assert_eq!(row.len(), g);
                for (gi, &got) in row.iter().enumerate() {
                    let want = bbox_iou_pair(dt[di], gt[gi], iscrowd[gi]);
                    assert_eq!(
                        got,
                        want,
                        "d={d} g={g} (d*g={}) cell [{di}][{gi}] disagrees with the pair kernel",
                        d * g
                    );
                }
            }
        }
    }

    #[test]
    fn parse_roundtrips_and_aliases() {
        assert_eq!("bbox".parse(), Ok(SimKind::Bbox));
        assert_eq!("mask".parse(), Ok(SimKind::Mask));
        assert_eq!("obb".parse(), Ok(SimKind::Obb));
        assert_eq!("oks".parse(), Ok(SimKind::Oks));
        // aliases from the eval-config axis
        assert_eq!("segm".parse(), Ok(SimKind::Mask));
        assert_eq!("keypoints".parse(), Ok(SimKind::Oks));
        // canonical names round-trip
        for k in [SimKind::Bbox, SimKind::Mask, SimKind::Obb, SimKind::Oks] {
            assert_eq!(k.as_str().parse(), Ok(k));
        }
    }

    #[test]
    fn parse_rejects_unknown() {
        assert!("polygon".parse::<SimKind>().is_err());
    }

    // --- OKS ------------------------------------------------------------

    // One keypoint, sigma s, area A. e = (dx^2+dy^2)/(4 s^2)/(A+eps)/2.
    fn oks_1kpt(dx: f64, dy: f64, s: f64, area: f64) -> f64 {
        let e = (dx * dx + dy * dy) / (4.0 * s * s) / (area + f64::EPSILON) / 2.0;
        (-e).exp()
    }

    // A GT pose with a default bbox (only relevant to the k1 == 0 branch).
    fn gt(keypoints: &[f64], area: f64) -> GtPose<'_> {
        GtPose {
            keypoints,
            area,
            bbox: [0.0, 0.0, 30.0, 30.0],
        }
    }

    #[test]
    fn oks_identical_keypoints_is_one() {
        let sigmas = [0.05, 0.07];
        let kpts = [10.0, 10.0, 2.0, 20.0, 20.0, 2.0]; // both visible
        let dt = kpts; // identical
        let m = oks_matrix(&[&dt], &[gt(&kpts, 1000.0)], &sigmas);
        assert!((m[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn oks_single_displaced_keypoint_matches_formula() {
        let sigmas = [0.05];
        let area = 1000.0;
        let kpts = [10.0, 10.0, 2.0]; // visible
        let dt = [13.0, 14.0, 2.0]; // displaced by (3, 4)
        let m = oks_matrix(&[&dt], &[gt(&kpts, area)], &sigmas);
        assert!((m[0][0] - oks_1kpt(3.0, 4.0, 0.05, area)).abs() < 1e-12);
    }

    #[test]
    fn oks_averages_only_visible_gt_keypoints() {
        // kpt 0 visible & identical (contributes 1.0); kpt 1 not visible (v=0)
        // and far away — must be excluded, so OKS == 1.0.
        let sigmas = [0.05, 0.05];
        let kpts = [10.0, 10.0, 2.0, 0.0, 0.0, 0.0];
        let dt = [10.0, 10.0, 2.0, 999.0, 999.0, 2.0];
        let m = oks_matrix(&[&dt], &[gt(&kpts, 1000.0)], &sigmas);
        assert!((m[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn oks_no_visible_gt_uses_bbox_distance_branch() {
        // k1 == 0 (all GT keypoints invisible). A DT keypoint inside the doubled
        // GT bbox has zero boundary distance => e = 0 => OKS = 1.0.
        let sigmas = [0.05];
        let kpts = [0.0, 0.0, 0.0]; // invisible
        let pose = GtPose {
            keypoints: &kpts,
            area: 1000.0,
            bbox: [0.0, 0.0, 20.0, 20.0],
        };
        let inside = [10.0, 10.0, 2.0]; // within [x0,x1]x[y0,y1] = [-20,40]
        let m = oks_matrix(&[&inside], &[pose], &sigmas);
        assert!((m[0][0] - 1.0).abs() < 1e-12);
        // A DT keypoint far outside the bbox scores strictly less than 1.
        let outside = [1000.0, 1000.0, 2.0];
        let m2 = oks_matrix(&[&outside], &[pose], &sigmas);
        assert!(m2[0][0] < 1.0);
    }

    #[test]
    fn oks_empty_keypoints_leave_zero() {
        let sigmas = [0.05];
        let kpts = [10.0, 10.0, 2.0];
        let empty: &[f64] = &[];
        // empty DT row, present GT
        let m = oks_matrix(&[empty], &[gt(&kpts, 1000.0)], &sigmas);
        assert_eq!(m[0][0], 0.0);
        // present DT, empty GT column
        let m2 = oks_matrix(&[&kpts[..]], &[gt(empty, 1000.0)], &sigmas);
        assert_eq!(m2[0][0], 0.0);
    }

    /// `oks_matrix` now runs through the same `rows()` dispatcher as its
    /// siblings; straddle the parallel threshold and check every cell against a
    /// 1×1 call, so neither branch nor the hoisted per-GT prep can drift from
    /// the pairwise arithmetic.
    #[test]
    fn oks_parallel_and_sequential_agree_with_pairwise() {
        let mut rng = StdRng::seed_from_u64(0x0C50C5);
        let sigmas = [0.05, 0.07, 0.09];

        let n = (MIN_PARALLEL_WORK as f64).sqrt().ceil() as usize;
        for &(d, g) in &[(2, 3), (n, n - 1), (n, n + 1)] {
            let mut kpt_store: Vec<Vec<f64>> = Vec::new();
            for _ in 0..d + g {
                // Occasionally empty, sometimes with invisible keypoints.
                if rng.random_bool(0.05) {
                    kpt_store.push(Vec::new());
                } else {
                    kpt_store.push(
                        (0..3)
                            .flat_map(|_| {
                                [
                                    rng.random_range(0.0..100.0),
                                    rng.random_range(0.0..100.0),
                                    if rng.random_bool(0.7) { 2.0 } else { 0.0 },
                                ]
                            })
                            .collect(),
                    );
                }
            }
            let (dt_kpts, gt_kpts) = kpt_store.split_at(d);
            let dt: Vec<&[f64]> = dt_kpts.iter().map(Vec::as_slice).collect();
            let gts: Vec<GtPose<'_>> = gt_kpts
                .iter()
                .map(|k| GtPose {
                    keypoints: k,
                    area: rng.random_range(100.0..2000.0),
                    bbox: [10.0, 10.0, 30.0, 40.0],
                })
                .collect();

            let matrix = oks_matrix(&dt, &gts, &sigmas);
            assert_eq!(matrix.len(), d);
            for (di, row) in matrix.iter().enumerate() {
                assert_eq!(row.len(), g);
                for (gi, &got) in row.iter().enumerate() {
                    let want = oks_matrix(&[dt[di]], &[gts[gi]], &sigmas)[0][0];
                    assert_eq!(
                        got, want,
                        "d={d} g={g} cell [{di}][{gi}] disagrees with the 1x1 call"
                    );
                }
            }
        }
    }

    /// Bit patterns of TrackEval's `_calculate_box_ious(gt, tracker, 'xywh')`.
    /// The first pair is one where corner areas and `w × h` areas part: the
    /// tracking kernel must give the reference's bits, not `bbox_iou_pair`'s.
    #[test]
    fn corner_kernel_matches_trackeval_bits() {
        for &(a, b, bits) in TRACKEVAL_BOX_IOUS {
            assert_eq!(
                bbox_iou_corners(a, b).to_bits(),
                bits,
                "{a:?} vs {b:?}: got {}",
                bbox_iou_corners(a, b)
            );
        }
        let (a, b, bits) = TRACKEVAL_BOX_IOUS[0];
        assert_ne!(
            bbox_iou_pair(a, b, false).to_bits(),
            bits,
            "the first case no longer tells the two kernels apart"
        );
    }

    /// A zero-area box, or one under `f64::EPSILON`, scores 0 against
    /// anything, itself included.
    #[test]
    fn corner_kernel_scores_degenerate_boxes_zero() {
        let tiny = [5.0, 5.0, 1e-9, 1e-9];
        assert_eq!(bbox_iou_corners(tiny, tiny), 0.0);
        let flat = [0.0, 0.0, 10.0, 0.0];
        assert_eq!(bbox_iou_corners(flat, [0.0, 0.0, 10.0, 10.0]), 0.0);
        let negative = [0.0, 0.0, -3.0, 4.0];
        assert_eq!(bbox_iou_corners(negative, [0.0, 0.0, 10.0, 10.0]), 0.0);
        let m = bbox_iou_corners_flat(&[[0.0, 0.0, 10.0, 10.0]], &[[5.0, 0.0, 10.0, 10.0], tiny]);
        assert_eq!(m.len(), 2);
        assert_eq!(m[1], 0.0);
        assert_eq!(bbox_iou_corners_flat(&[], &[tiny]), Vec::<f64>::new());
    }

    /// Recorded under the pinned environment (`.venv-trackeval`, numpy
    /// 1.23.5) by running the lines of `_calculate_box_ious` on each pair as
    /// one-row arrays, `ground truth` first, and printing
    /// `hex(iou.view(np.uint64))`.
    const TRACKEVAL_BOX_IOUS: &[([f64; 4], [f64; 4], u64)] = &[
        (
            [2.49, 1.97, 1.47, 1.74],
            [1.29, 2.28, 1.86, 0.3],
            0x3fb15f3ae8272a7b,
        ),
        (
            [0.1, 0.2, 0.7, 0.3],
            [0.3, 0.1, 0.5, 0.35],
            0x3fdec4ec4ec4ec4a,
        ),
        (
            [10.0, 10.0, 50.0, 80.0],
            [11.0, 10.0, 50.0, 80.0],
            0x3feebebebebebebf,
        ),
        (
            [1.0, 1.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
            0x3fc2492492492492,
        ),
        ([0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 10.0, 10.0], 0x0),
        (
            [123.456, 78.9, 33.3, 91.7],
            [130.01, 80.2, 30.1, 88.8],
            0x3fe6ad5dc578399d,
        ),
    ];

    #[test]
    fn bbox_kernel_matches_and_obeys_contract() {
        // Re-exported kernel is the same math as crate::mask::bbox_iou, and the
        // similarity contract holds: [0,1], self-overlap == 1.
        let dt = [[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 10.0, 10.0]];
        let gt = [[0.0, 0.0, 10.0, 10.0]];
        let iscrowd = [false];
        let m = bbox_iou(&dt, &gt, &iscrowd);
        assert_eq!(m, crate::mask::bbox_iou(&dt, &gt, &iscrowd));
        assert!((m[0][0] - 1.0).abs() < 1e-12, "identical box => IoU 1");
        assert_eq!(m[1][0], 0.0, "disjoint box => IoU 0");
        for row in &m {
            for &v in row {
                assert!((0.0..=1.0).contains(&v), "similarity in [0,1]");
            }
        }
    }
}
