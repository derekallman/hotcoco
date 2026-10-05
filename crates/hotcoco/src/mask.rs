//! Pure Rust implementation of COCO mask operations: RLE encoding and decoding, IoU, merge, and area.
//!
//! This is a faithful port of the C `maskApi.c` from pycocotools/cocoapi.
//! The scan-line polygon rasterization and LEB128-like string encoding match
//! the original exactly to ensure metric parity.
//!
//! # Mechanics, not formulas
//!
//! This module owns the RLE *mechanics* — codec, area, `intersection_area`. The
//! IoU *formulas* live in [`crate::primitives::sim`]; [`iou`] and [`bbox_iou`]
//! below are re-exports of them, kept here so `hotcoco::mask::*` keeps mirroring
//! `pycocotools.mask.*`. The re-export is one-way path sugar, never the
//! definition.

use crate::types::Rle;

pub use crate::primitives::sim::{bbox_iou, mask_iou as iou};

/// Total pixel count `h * w`, or an error when it exceeds `u32::MAX`.
///
/// RLE run counts are 32-bit (matching the C `maskApi`), so a mask with more
/// than `u32::MAX` pixels is unrepresentable. Dimensions arrive here straight
/// from untrusted JSON, so the product is taken in `u64` and checked: `h * w` in
/// `u32` overflows to a debug panic and to wrapped garbage in release.
fn checked_hw(h: u32, w: u32) -> crate::error::Result<u32> {
    let hw = h as u64 * w as u64;
    u32::try_from(hw).map_err(|_| {
        crate::error::Error::from(format!(
            "image dimensions {h}x{w} = {hw} pixels exceed u32::MAX (RLE run counts are 32-bit)"
        ))
    })
}

/// Encode a column-major binary mask into RLE.
///
/// `mask` is stored in column-major order (Fortran order): pixel (x, y) is at index `y + h * x`.
///
/// Errors when `mask.len() != h * w`, or when `h * w` exceeds `u32::MAX`.
pub fn encode(mask: &[u8], h: u32, w: u32) -> crate::error::Result<Rle> {
    let n = checked_hw(h, w)? as usize;
    if mask.len() != n {
        return Err(format!("encode: mask length {} must equal h*w = {n}", mask.len()).into());
    }

    let mut counts = Vec::with_capacity(h.min(w) as usize * 2);
    let mut p: u8 = 0;
    let mut c: u32 = 0;

    for &v in mask.iter().take(n) {
        let v = (v != 0) as u8;
        if v != p {
            counts.push(c);
            c = 0;
            p = v;
        }
        c += 1;
    }
    counts.push(c);

    Ok(Rle { h, w, counts })
}

/// Decode an RLE to a column-major binary mask of size `h * w`.
pub fn decode(rle: &Rle) -> Vec<u8> {
    let n = (rle.h as usize) * (rle.w as usize);
    let mut mask = vec![0u8; n];
    let mut idx = 0usize;
    let mut v = 0u8;
    for &c in &rle.counts {
        let c = c as usize;
        let end = (idx.saturating_add(c)).min(n);
        mask[idx..end].fill(v);
        idx = end;
        v = 1 - v;
    }
    mask
}

/// Compute the area (number of foreground pixels) of an RLE mask.
///
/// Only sums the odd-indexed runs (which represent 1s).
#[inline]
pub fn area(rle: &Rle) -> u64 {
    rle.counts
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&c| c as u64)
        .sum()
}

/// Compute the bounding box `[x, y, w, h]` of an RLE mask.
#[inline]
pub fn to_bbox(rle: &Rle) -> [f64; 4] {
    let h = rle.h as usize;
    if h == 0 || rle.w == 0 || rle.counts.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let mut xs = rle.w as usize;
    let mut xe: usize = 0;
    let mut ys = rle.h as usize;
    let mut ye: usize = 0;
    let mut has_any = false;

    let mut cc = 0usize; // cumulative pixel count (column-major flat index)
    for (i, &c) in rle.counts.iter().enumerate() {
        let c = c as usize;
        // Skip zero-length foreground runs: they contribute no pixels, and
        // `cc + c - 1` below would underflow on one (untrusted RLE strings can
        // legally decode to zero-length runs).
        if i % 2 == 1 && c > 0 {
            // Foreground run: convert flat indices to (column, row) coordinates
            has_any = true;
            let x1 = cc / h; // start column
            let y1 = cc % h; // start row
            let end = cc + c - 1; // last pixel (inclusive)
            let x2 = end / h; // end column
            let y2 = end % h; // end row

            if x1 < xs {
                xs = x1;
            }
            if x2 >= xe {
                xe = x2 + 1;
            }
            if y1 < ys {
                ys = y1;
            }
            // If the run spans multiple columns, it covers all rows in between
            if x1 != x2 {
                ys = 0;
                ye = h;
            }
            if y2 >= ye {
                ye = y2 + 1;
            }
        }
        cc += c;
    }

    if !has_any {
        return [0.0, 0.0, 0.0, 0.0];
    }

    [xs as f64, ys as f64, (xe - xs) as f64, (ye - ys) as f64]
}

/// Merge multiple RLE masks with union (intersect=false) or intersection (intersect=true).
///
/// Errors when the masks do not all share the same dimensions (the result
/// would silently be stamped with the first mask's dims), or when `h * w`
/// exceeds `u32::MAX`.
pub fn merge(rles: &[Rle], intersect: bool) -> crate::error::Result<Rle> {
    if rles.is_empty() {
        return Ok(Rle {
            h: 0,
            w: 0,
            counts: vec![0],
        });
    }

    let h = rles[0].h;
    let w = rles[0].w;
    checked_hw(h, w)?;
    if let Some(bad) = rles[1..].iter().find(|r| r.h != h || r.w != w) {
        return Err(format!(
            "merge: mismatched RLE dimensions — first mask is {h}x{w}, another is {}x{}",
            bad.h, bad.w
        )
        .into());
    }

    if rles.len() == 1 {
        return Ok(rles[0].clone());
    }

    // Merge pairwise
    let mut result = rles[0].clone();
    for rle in &rles[1..] {
        result = merge_two(&result, rle, intersect);
    }
    // Ensure h/w stay correct
    result.h = h;
    result.w = w;
    Ok(result)
}

/// Merge two RLE masks using a two-pointer walk over both run streams.
///
/// At each step, consumes the shorter remaining run from either stream,
/// combining the current foreground/background values with AND (intersect=true)
/// or OR (intersect=false). Runs of equal output value are coalesced.
fn merge_two(a: &Rle, b: &Rle, intersect: bool) -> Rle {
    let h = a.h;
    let w = a.w;
    let n = (h as u64) * (w as u64);

    let mut counts = Vec::with_capacity(a.counts.len() + b.counts.len());
    let mut ca = 0u64; // remaining in current run of a
    let mut cb = 0u64; // remaining in current run of b
    let mut va = false; // current value of a
    let mut vb = false; // current value of b
    let mut ai = 0usize; // index in a.counts
    let mut bi = 0usize; // index in b.counts
    let mut total = 0u64;

    let mut v_prev: Option<bool> = None;

    while total < n {
        // Refill a (skip 0-length runs)
        while ca == 0 && ai < a.counts.len() {
            ca = a.counts[ai] as u64;
            va = ai % 2 == 1;
            ai += 1;
        }
        // Refill b (skip 0-length runs)
        while cb == 0 && bi < b.counts.len() {
            cb = b.counts[bi] as u64;
            vb = bi % 2 == 1;
            bi += 1;
        }
        // A stream whose counts sum to less than `h * w` is exhausted here; its
        // remaining pixels are background, exactly as `decode` zero-fills them.
        // Without this reset the last run's value would leak over the tail.
        if ca == 0 {
            va = false;
        }
        if cb == 0 {
            vb = false;
        }

        let step = if ca > 0 && cb > 0 {
            ca.min(cb)
        } else if ca > 0 {
            ca
        } else if cb > 0 {
            cb
        } else {
            break;
        };
        // The output represents exactly `n` pixels; cap the step so run counts
        // (including coalesced sums) stay within `n ≤ u32::MAX` even when an
        // unvalidated input's counts over-run its own `h * w`.
        let step = step.min(n - total);

        let v = if intersect { va && vb } else { va || vb };

        match v_prev {
            Some(prev) if prev == v => {
                // Extend the last run
                if let Some(last) = counts.last_mut() {
                    debug_assert!(u32::try_from(step).is_ok(), "RLE run exceeds u32::MAX");
                    *last += step as u32;
                }
            }
            _ => {
                // If we need to start with 1 but there's no leading 0 run, add a 0-length run
                if counts.is_empty() && v {
                    counts.push(0);
                }
                debug_assert!(u32::try_from(step).is_ok(), "RLE run exceeds u32::MAX");
                counts.push(step as u32);
            }
        }
        v_prev = Some(v);

        if ca > 0 {
            ca -= step;
        }
        if cb > 0 {
            cb -= step;
        }
        total += step;
    }

    if counts.is_empty() {
        debug_assert!(u32::try_from(n).is_ok(), "RLE total pixels exceed u32::MAX");
        counts.push(n as u32);
    }

    Rle { h, w, counts }
}

/// Compute the intersection area of two RLE masks without allocating.
///
/// Walks both RLE streams simultaneously (same logic as `merge_two` with intersect=true)
/// but only accumulates the count where both masks are foreground.
///
/// RLE *mechanics* — the IoU formula built on it lives in
/// [`crate::primitives::sim::mask_iou`].
#[inline]
pub(crate) fn intersection_area(a: &Rle, b: &Rle) -> u64 {
    let n = (a.h as u64) * (a.w as u64);
    let mut ca = 0u64;
    let mut cb = 0u64;
    let mut va = false;
    let mut vb = false;
    let mut ai = 0usize;
    let mut bi = 0usize;
    let mut total = 0u64;
    let mut count = 0u64;

    while total < n {
        // Advance past 0-length runs
        while ca == 0 && ai < a.counts.len() {
            ca = a.counts[ai] as u64;
            va = ai % 2 == 1;
            ai += 1;
        }
        while cb == 0 && bi < b.counts.len() {
            cb = b.counts[bi] as u64;
            vb = bi % 2 == 1;
            bi += 1;
        }
        // A stream whose counts sum to less than `h * w` is exhausted here, and
        // its tail is background (see `merge_two`), so nothing past this point
        // can be in the intersection.
        if ca == 0 || cb == 0 {
            break;
        }

        let step = ca.min(cb);
        if va && vb {
            count += step;
        }
        ca -= step;
        cb -= step;
        total += step;
    }

    count
}

/// `base + s * t`, rounded the way the pycocotools wheel for this architecture
/// rounds it: one fused multiply-add on arm64, two roundings everywhere else.
/// See the comment in [`fr_poly`] for why the choice is per-target.
#[inline]
fn interp(s: f64, t: f64, base: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        s.mul_add(t, base)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        s * t + base
    }
}

/// Convert a polygon (flat list of `[x0, y0, x1, y1, ...]`) to RLE.
///
/// Faithful port of `rleFrPoly` from maskApi.c.
/// Uses upsampling by 5x, Bresenham-like edge walking, y-boundary detection,
/// and differential RLE encoding — exactly matching the C implementation.
///
/// Boundary pixels round the way the pycocotools wheel for the same
/// architecture rounds them: the edge interpolation is one fused multiply-add on
/// arm64 and two roundings on x86-64, because that is what the C compiler emits
/// for each. A mask can therefore differ by a boundary pixel between an arm64
/// and an x86-64 machine — exactly as pycocotools' own output does.
///
/// Errors when `h * w` exceeds `u32::MAX`. Vertices far outside the image are
/// kept where they are, so edge slopes match the reference; the boundary walk
/// skips the stretches that cannot reach the image, so its cost is bounded by
/// the image size rather than the polygon's. Only coordinates beyond ±2²²
/// pixels are clamped, where maskApi.c's own integer arithmetic overflows.
pub fn fr_poly(xy: &[f64], h: u32, w: u32) -> crate::error::Result<Rle> {
    let hw = checked_hw(h, w)?;
    Ok(POLY_SCRATCH.with(|s| fr_poly_impl(&mut s.borrow_mut(), xy, h, w, hw)))
}

/// Reusable buffers for [`fr_poly`]'s three rasterization stages.
///
/// Every field is scratch — cleared on entry, meaningless after return. Only
/// the `counts` vector of the produced [`Rle`] is allocated per call, because
/// it is the value handed back. Rasterizing a val2017 segm run makes ~10⁵
/// `fr_poly` calls from rayon workers, and the five buffers below were five
/// mallocs each against one process-wide allocator lock; per-thread reuse
/// leaves each buffer sized for the largest polygon its thread has seen
/// (kilobytes) and takes the rasterizer off the lock entirely.
#[derive(Default)]
struct PolyScratch {
    x_int: Vec<i32>,
    y_int: Vec<i32>,
    u: Vec<i32>,
    v: Vec<i32>,
    a: Vec<u32>,
}

thread_local! {
    static POLY_SCRATCH: std::cell::RefCell<PolyScratch> =
        std::cell::RefCell::new(PolyScratch::default());
}

fn fr_poly_impl(scratch: &mut PolyScratch, xy: &[f64], h: u32, w: u32, hw: u32) -> Rle {
    let k = xy.len() / 2;
    if k < 3 {
        return Rle {
            h,
            w,
            counts: vec![hw],
        };
    }

    // Stage 1: upsample the vertices by `SCALE` and walk each edge.
    //
    // Untrusted coordinates like ±1e9 would overflow the i32 edge subtractions
    // in the walk (a debug panic), so vertices are clamped to ±2²² pixels — far
    // beyond any real annotation, and the bound that keeps every edge length
    // inside i32. pycocotools' `(int)` cast is undefined behavior past i32
    // anyway. Float-to-int casts saturate and NaN casts to 0, so no coordinate
    // value can panic here. The clamp is the only place a vertex moves: the walk
    // bounds its own length by the image (see `walk_edge`), not by bending
    // edges, so every polygon inside the ceiling rasterizes like the reference.
    const MAX_ABS_UPSAMPLED: f64 = SCALE * (1u32 << 22) as f64;
    let x_int = &mut scratch.x_int;
    let y_int = &mut scratch.y_int;
    x_int.clear();
    y_int.clear();
    for j in 0..k {
        let up = |c: f64| (SCALE * c + 0.5).clamp(-MAX_ABS_UPSAMPLED, MAX_ABS_UPSAMPLED) as i32;
        x_int.push(up(xy[j * 2]));
        y_int.push(up(xy[j * 2 + 1]));
    }
    // Close the polygon by repeating the first vertex
    x_int.push(x_int[0]);
    y_int.push(y_int[0]);

    let u = &mut scratch.u;
    let v = &mut scratch.v;
    u.clear();
    v.clear();
    let bands = Bands::new(h, w);
    for j in 0..k {
        walk_edge(
            (x_int[j], y_int[j]),
            (x_int[j + 1], y_int[j + 1]),
            &bands,
            u,
            v,
        );
    }

    boundary_to_rle(u, v, &mut scratch.a, h, w, hw)
}

/// `rleFrPoly`'s upsampling factor.
const SCALE: f64 = 5.0;

/// The upsampled coordinate ranges outside which a boundary point cannot change
/// [`boundary_to_rle`]'s output, inclusive.
///
/// Stage 2 emits a crossing for consecutive points `(u₀, v₀) → (u₁, v₁)` with
/// `u₀ ≠ u₁` only when the crossed boundary `min(u₀, u₁)` (or `u₁ − 1` when
/// rising) is `5·xd + 2` for a column `xd` in `0..w` — so only in
/// `[2, 5w − 3]`. A point outside `[−5, 5w + 5]` is more than two steps past
/// that range, and consecutive points are at most two apart in `u` (one along
/// an edge, one more where `(int)(x + 0.5)` truncates a negative vertex toward
/// zero), so no pair touching it crosses, and dropping a run of such points
/// joins two points that cross nothing either. `v` never gates a crossing — it
/// only places it, clamped to row `0` at or below `v = 2` and to row `h` at or above
/// `v = 5h + 2` — so points beyond `[−5, 5h + 5]` matter only through their
/// `u`: one point per distinct `u` reproduces every crossing they make.
struct Bands {
    u: (i64, i64),
    v: (i64, i64),
}

impl Bands {
    fn new(h: u32, w: u32) -> Self {
        let band = |extent: u32| (-(SCALE as i64), SCALE as i64 * (i64::from(extent) + 1));
        Bands {
            u: band(w),
            v: band(h),
        }
    }
}

/// The first `t` in `lo..hi` for which `pred` holds, or `hi` if none does.
/// `pred` must be monotone over the range: false, then true.
fn first_true(mut lo: i32, mut hi: i32, pred: impl Fn(i32) -> bool) -> i32 {
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Stage 1 for one edge: append its upsampled boundary points to `u`/`v`,
/// minus the points [`Bands`] proves cannot change the mask.
///
/// The points are maskApi.c's — `t` steps along the longer axis and the other
/// coordinate is interpolated with [`interp`] — and so are their values; only
/// which of them are materialized differs. A vertex at `x = 10⁶` on a 640 px
/// image is a five-million-point edge in maskApi.c and a few thousand here.
/// Both coordinates are monotone in `t` along one edge (a rounded linear
/// function), which is what lets the band edges and the `u`-steps be found by
/// bisection instead of by walking.
fn walk_edge(
    start: (i32, i32),
    end: (i32, i32),
    bands: &Bands,
    u: &mut Vec<i32>,
    v: &mut Vec<i32>,
) {
    let ((mut xs, mut ys), (mut xe, mut ye)) = (start, end);
    let dx = (xe - xs).unsigned_abs() as i32;
    let dy = (ys - ye).unsigned_abs() as i32;
    // If the edge runs "backwards" (right-to-left or bottom-to-top), flip the
    // direction so `t` always steps forward, then reverse the traversal order.
    let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
    if flip {
        std::mem::swap(&mut xs, &mut xe);
        std::mem::swap(&mut ys, &mut ye);
    }
    let x_major = dx >= dy;
    let len = if x_major { dx } else { dy };
    // Slope of the minor axis per step along the major axis
    let s: f64 = match (x_major, len) {
        (_, 0) => 0.0,
        (true, _) => (ye - ys) as f64 / dx as f64,
        (false, _) => (xe - xs) as f64 / dy as f64,
    };
    // `interp`, not a bare `base + s * t`, reproduces the reference's
    // arithmetic *per architecture*. maskApi.c writes `(int)(ys+s*t+.5)`;
    // whether the compiler fuses `s*t+ys` into one FMA (one rounding) or
    // leaves it as two roundings depends on the target: arm64 has an FMA
    // instruction and clang/gcc contract by default there, while the x86_64
    // wheels on PyPI are built for baseline x86-64, which has none. Rust never
    // contracts implicitly, so a single fixed choice matches one platform's
    // pycocotools and disagrees with the other's on boundary pixels (~2 of 400
    // random polygons — the first v1.0.0 tag failed CI on Linux for exactly
    // this after passing on an arm64 Mac). This path builds every segm GT
    // mask, so mirroring the platform is what keeps segmentation parity exact
    // wherever hotcoco and pycocotools are compared on the same machine.
    let at = |t: i32| -> (i32, i32) {
        if x_major {
            (t + xs, (interp(s, t as f64, ys as f64) + 0.5) as i32)
        } else {
            ((interp(s, t as f64, xs as f64) + 0.5) as i32, t + ys)
        }
    };

    // The `t` range whose `u` lies in the band, as a half-open `[lo, hi)`.
    // `key` turns a falling `u` into a rising one so one bisection serves both.
    let (u0, u1) = (at(0).0, at(len).0);
    let key = |t: i32| -> i64 {
        let ut = i64::from(at(t).0);
        if u1 >= u0 { ut } else { -ut }
    };
    let (band_lo, band_hi) = if u1 >= u0 {
        bands.u
    } else {
        (-bands.u.1, -bands.u.0)
    };
    let lo = first_true(0, len + 1, |t| key(t) >= band_lo);
    let hi = first_true(lo, len + 1, |t| key(t) > band_hi);

    // Inside `[v_in_lo, v_in_hi)` every point is kept; outside it, one point
    // per distinct `u`. An x-major edge keeps everything: `u = t + xs` is
    // already one point per `u`. A y-major edge has `v = t + ys`, so the
    // range is where `v` lies in its band.
    let (v_in_lo, v_in_hi) = if x_major {
        (lo, hi)
    } else {
        let clip = |t: i64| t.clamp(i64::from(lo), i64::from(hi)) as i32;
        (
            clip(bands.v.0 - i64::from(ys)),
            clip(bands.v.1 - i64::from(ys) + 1),
        )
    };
    let first = u.len();
    let mut t = lo;
    while t < hi {
        let (ut, vt) = at(t);
        u.push(ut);
        v.push(vt);
        t = if (v_in_lo..v_in_hi).contains(&t) {
            t + 1
        } else {
            let run_end = if t < v_in_lo { v_in_lo } else { hi };
            first_true(t + 1, run_end, |t2| at(t2).0 != ut)
        };
    }
    if flip {
        u[first..].reverse();
        v[first..].reverse();
    }
}

/// Stages 2 and 3 of `rleFrPoly`: turn the upsampled boundary `u`/`v` into the
/// RLE of the polygon's interior. `a` is scratch.
fn boundary_to_rle(u: &[i32], v: &[i32], a: &mut Vec<u32>, h: u32, w: u32, hw: u32) -> Rle {
    let h_s = h as i64;
    let w_s = w as i64;

    // Stage 2: Detect column transitions (x-boundary crossings) in the upsampled
    // boundary, downsample back to original resolution, and convert directly to
    // column-major flat indices (skipping intermediate bx/by storage).
    let m = u.len();
    a.clear();
    a.reserve(m);

    for j in 1..m {
        // Only process points where the x-coordinate changed (column crossing)
        if u[j] != u[j - 1] {
            // Determine which column boundary was crossed and downsample
            let xd_raw = if u[j] < u[j - 1] { u[j] } else { u[j] - 1 };
            let xd: f64 = (xd_raw as f64 + 0.5) / SCALE - 0.5;
            // Skip if this doesn't land on an integer column boundary within image bounds
            if xd != xd.floor() || xd < 0.0 || xd > (w_s - 1) as f64 {
                continue;
            }
            // Downsample the y-coordinate and clamp to image bounds
            let yd_raw = if v[j] < v[j - 1] { v[j] } else { v[j - 1] };
            let mut yd: f64 = (yd_raw as f64 + 0.5) / SCALE - 0.5;
            if yd < 0.0 {
                yd = 0.0;
            } else if yd > h_s as f64 {
                yd = h_s as f64;
            }
            yd = yd.ceil();
            // Convert (column, row) directly to column-major flat index
            a.push((xd as u32) * h + (yd as u32));
        }
    }

    // Stage 3: Sort flat indices, compute successive differences to get run lengths,
    // then merge any zero-length runs (which arise when two boundary points land on
    // the same pixel).
    // Sentinel: total pixel count marks the end of the mask
    a.push(hw);
    a.sort_unstable();

    // Convert sorted positions to run lengths via successive differences
    let mut prev: u32 = 0;
    for val in a.iter_mut() {
        let t = *val;
        *val = t - prev;
        prev = t;
    }

    // Merge zero-length runs (two boundary points at the same position cancel out)
    let mut counts: Vec<u32> = Vec::with_capacity(a.len());
    let mut i = 0;
    if !a.is_empty() {
        counts.push(a[0]);
        i = 1;
    }
    while i < a.len() {
        if a[i] > 0 {
            counts.push(a[i]);
            i += 1;
        } else {
            i += 1; // skip zero
            if i < a.len() {
                if let Some(last) = counts.last_mut() {
                    *last += a[i];
                }
                i += 1;
            }
        }
    }

    Rle { h, w, counts }
}

/// Convert a bounding box `[x, y, w, h]` to an RLE mask.
///
/// The box's four corners are rasterized as a polygon through [`fr_poly`], as
/// pycocotools' `rleFrBbox` does, so a box and the equivalent polygon produce
/// the same pixels. That matters for fractional boxes: filling
/// `floor(x)..ceil(x + w)` analytically rounds every edge outward and gives
/// `[0.5, 0.5, 2, 2]` an area of 9 where the reference gives 4 — and inside
/// an evaluation a box-only ground truth took that path while the detection
/// it was meant to match went through `fr_poly`, so identical fractional
/// boxes scored IoU 4/9 against each other.
///
/// Errors when `h * w` exceeds `u32::MAX`. Non-finite or far out-of-range
/// coordinates are handled as [`fr_poly`] handles them.
pub fn fr_bbox(bb: &[f64; 4], h: u32, w: u32) -> crate::error::Result<Rle> {
    fr_poly(&crate::types::Segmentation::rect_corners(bb), h, w)
}

/// Compress an RLE into the LEB128-like string format used by COCO.
///
/// This matches the `rleToString` function in maskApi.c exactly,
/// including delta encoding for indices > 2 (stride-2 differencing).
pub fn rle_to_string(rle: &Rle) -> String {
    let mut s = String::with_capacity(rle.counts.len() * 3);
    for (i, &cnt) in rle.counts.iter().enumerate() {
        // maskApi.c: x = (long) cnts[i]; if(i>2) x -= (long) cnts[i-2];
        let x = if i > 2 {
            (cnt as i64).wrapping_sub(rle.counts[i - 2] as i64)
        } else {
            cnt as i64
        };
        rle_encode_i64(&mut s, x);
    }
    s
}

/// Encode a single (possibly negative) value to the COCO LEB128-like format.
///
/// From maskApi.c `rleToString`:
/// ```c
/// c = x & 0x1f; x >>= 5;
/// more = (c & 0x10) ? x != -1 : x != 0;
/// if(more) c |= 0x20; c += 48; *s++ = c;
/// ```
fn rle_encode_i64(s: &mut String, mut x: i64) {
    loop {
        let c = (x & 0x1f) as u8;
        x >>= 5;
        let more = if c & 0x10 != 0 { x != -1 } else { x != 0 };
        let mut c = c;
        if more {
            c |= 0x20;
        }
        c += 48;
        s.push(c as char);
        if !more {
            break;
        }
    }
}

/// Decompress a COCO LEB128-like string back to an RLE.
///
/// Matches `rleFrString` from maskApi.c, including stride-2 delta
/// accumulation for indices > 2.
///
/// Returns an error if the decoded counts sum exceeds `h * w`.
pub fn rle_from_string(s: &str, h: u32, w: u32) -> crate::error::Result<Rle> {
    let bytes = s.as_bytes();
    let mut counts = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        let mut x: i64 = 0;
        let mut shift = 0;
        let mut more = true;
        while more && i < bytes.len() {
            if bytes[i] < 48 {
                return Err(format!(
                    "invalid RLE: byte value {} at position {} is below ASCII '0' (48)",
                    bytes[i], i
                )
                .into());
            }
            // Bound the LEB-style shift: any valid u32 run length — even
            // delta-encoded, hence possibly negative — fits well within 11
            // five-bit groups. Unbounded, an untrusted string of continuation
            // bits grows `shift` past 63 and overflows the `<<` below (a debug
            // panic, a masked shift in release).
            if shift > 55 {
                return Err(format!(
                    "invalid RLE: run length at byte {i} has too many continuation characters"
                )
                .into());
            }
            let c = (bytes[i] - 48) as i64;
            i += 1;
            x |= (c & 0x1f) << shift;
            more = (c & 0x20) != 0;
            shift += 5;
        }
        // Sign extend if the highest bit (bit 4 of the last group) is set
        if shift > 0 && (x & (1 << (shift - 1))) != 0 {
            x |= !0i64 << shift;
        }
        // maskApi.c rleFrString: if(m>2) x += (long) cnts[m-2];
        if counts.len() > 2 {
            x = x.wrapping_add(counts[counts.len() - 2] as i64);
        }
        if x < 0 {
            return Err(format!(
                "invalid RLE: negative count {x} at position {}",
                counts.len()
            )
            .into());
        }
        // Validate before narrowing: `as u32` would silently truncate, letting
        // an oversized run wrap and pass the total-vs-h*w check below.
        if x > u32::MAX as i64 {
            return Err(format!(
                "invalid RLE: count {x} at position {} exceeds u32::MAX",
                counts.len()
            )
            .into());
        }
        counts.push(x as u32);
    }

    // Validate total counts don't exceed h*w
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let hw = h as u64 * w as u64;
    if total > hw {
        return Err(format!("invalid RLE: total counts {total} exceed h*w={hw}").into());
    }

    Ok(Rle { h, w, counts })
}

/// Convert multiple polygons for a single object to a single merged RLE.
///
/// This corresponds to what pycocotools does when converting polygon segmentation:
/// rasterize each polygon separately, then merge all with union.
///
/// Errors when `h * w` exceeds `u32::MAX` (see [`fr_poly`]).
pub fn fr_polys(polygons: &[Vec<f64>], h: u32, w: u32) -> crate::error::Result<Rle> {
    let hw = checked_hw(h, w)?;
    if polygons.is_empty() {
        return Ok(Rle {
            h,
            w,
            counts: vec![hw],
        });
    }
    if let [poly] = polygons {
        return fr_poly(poly, h, w);
    }
    let rles: Vec<Rle> = polygons
        .iter()
        .map(|p| fr_poly(p, h, w))
        .collect::<crate::error::Result<_>>()?;
    merge(&rles, false)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let mask = vec![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0];
        let rle = encode(&mask, 3, 4).unwrap();
        let decoded = decode(&rle);
        assert_eq!(mask, decoded);
    }

    #[test]
    fn test_encode_all_zeros() {
        let mask = vec![0u8; 12];
        let rle = encode(&mask, 3, 4).unwrap();
        assert_eq!(rle.counts, vec![12]);
    }

    #[test]
    fn test_encode_all_ones() {
        let mask = vec![1u8; 12];
        let rle = encode(&mask, 3, 4).unwrap();
        assert_eq!(rle.counts, vec![0, 12]);
    }

    #[test]
    fn test_area() {
        let mask = vec![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0];
        let rle = encode(&mask, 3, 4).unwrap();
        assert_eq!(area(&rle), 5);
    }

    #[test]
    fn test_to_bbox() {
        // 3 rows x 4 cols, column-major
        // Col 0: [0,0,0], Col 1: [1,1,1], Col 2: [0,0,1], Col 3: [1,0,0]
        let mask = vec![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0];
        let rle = encode(&mask, 3, 4).unwrap();
        let bb = to_bbox(&rle);
        // x_min=1 (col 1), y_min=0 (row 0 in col 1), width=3, height=3
        assert_eq!(bb[0], 1.0);
        assert_eq!(bb[1], 0.0);
        assert_eq!(bb[2], 3.0);
        assert_eq!(bb[3], 3.0);
    }

    #[test]
    fn test_merge_union() {
        // Two masks
        let m1 = vec![0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        let m2 = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0];
        let r1 = encode(&m1, 3, 4).unwrap();
        let r2 = encode(&m2, 3, 4).unwrap();
        let merged = merge(&[r1, r2], false).unwrap();
        let decoded = decode(&merged);
        let expected = vec![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0];
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_merge_intersection() {
        let m1 = vec![0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        let m2 = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0];
        let r1 = encode(&m1, 3, 4).unwrap();
        let r2 = encode(&m2, 3, 4).unwrap();
        let merged = merge(&[r1, r2], true).unwrap();
        let decoded = decode(&merged);
        let expected = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0];
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_iou_basic() {
        let m1 = vec![0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        let m2 = vec![0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0];
        let r1 = encode(&m1, 3, 4).unwrap();
        let r2 = encode(&m2, 3, 4).unwrap();
        let ious = iou(&[r1], &[r2], &[false]);
        // intersection = 2, union = 3 + 3 - 2 = 4
        assert!((ious[0][0] - 0.5).abs() < 1e-10);
    }

    /// An RLE whose counts stop short of `h * w` ends in a foreground run.
    /// `decode` renders the omitted tail as background; the run-stream walkers
    /// must agree, or the tail leaks the last run's value into every pixel of
    /// the other mask and IoU exceeds 1.
    #[test]
    fn test_short_rle_tail_is_background() {
        let short = Rle {
            h: 10,
            w: 1,
            counts: vec![0, 2],
        };
        let full = Rle {
            h: 10,
            w: 1,
            counts: vec![0, 10],
        };
        assert_eq!(intersection_area(&short, &full), 2);
        assert_eq!(intersection_area(&full, &short), 2);
        let ious = iou(
            std::slice::from_ref(&short),
            std::slice::from_ref(&full),
            &[false],
        );
        assert!((ious[0][0] - 0.2).abs() < 1e-12, "got {}", ious[0][0]);

        // merge agrees with decode on both the union and the intersection.
        let union = merge(&[short.clone(), full.clone()], false).unwrap();
        assert_eq!(decode(&union), decode(&full));
        let inter = merge(&[full, short.clone()], true).unwrap();
        assert_eq!(decode(&inter), decode(&short));
    }

    /// Reference values from `pycocotools.mask.frPyObjects([[box]], 10, 10)`:
    /// the analytic floor/ceil fill gave 9 and 12.
    #[test]
    fn test_fr_bbox_fractional_matches_polygon_rasterization() {
        let a = fr_bbox(&[0.5, 0.5, 2.0, 2.0], 10, 10).unwrap();
        assert_eq!(area(&a), 4);
        assert_eq!(to_bbox(&a), [1.0, 1.0, 2.0, 2.0]);
        let b = fr_bbox(&[1.2, 1.7, 3.3, 2.1], 10, 10).unwrap();
        assert_eq!(area(&b), 8);

        // A box and its own corner polygon are the same mask, so a box-only
        // ground truth and a box detection cannot disagree about their overlap.
        let poly = fr_poly(&[0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 2.5, 0.5], 10, 10).unwrap();
        assert_eq!(a.counts, poly.counts);
        assert_eq!(iou(&[a], &[poly], &[false])[0][0], 1.0);
    }

    #[test]
    fn test_iou_mismatched_dims_is_minus_one() {
        // pycocotools `rleIou` returns -1 for masks on different canvases.
        let a = encode(&[1u8; 24], 4, 6).unwrap();
        let b = encode(&[1u8; 24], 6, 4).unwrap();
        let ious = iou(std::slice::from_ref(&a), &[b, a.clone()], &[false, false]);
        assert_eq!(ious[0][0], -1.0);
        assert_eq!(ious[0][1], 1.0);
    }

    #[test]
    fn test_bbox_iou() {
        let dt = [[0.0, 0.0, 10.0, 10.0]];
        let gt = [[5.0, 5.0, 10.0, 10.0]];
        let ious = bbox_iou(&dt, &gt, &[false]);
        // inter = 5*5 = 25, union = 100 + 100 - 25 = 175
        assert!((ious[0][0] - 25.0 / 175.0).abs() < 1e-10);
    }

    #[test]
    fn test_rle_string_roundtrip() {
        let rle = Rle {
            h: 10,
            w: 10,
            counts: vec![5, 3, 92],
        };
        let s = rle_to_string(&rle);
        let decoded = rle_from_string(&s, 10, 10).unwrap();
        assert_eq!(rle.counts, decoded.counts);
    }

    #[test]
    fn test_rle_string_large_counts() {
        let rle = Rle {
            h: 100,
            w: 100,
            counts: vec![100, 200, 9700],
        };
        let s = rle_to_string(&rle);
        let decoded = rle_from_string(&s, 100, 100).unwrap();
        assert_eq!(rle.counts, decoded.counts);
    }

    #[test]
    fn test_rle_string_zero_leading() {
        // Matches ppwwyyxx/cocoapi testZeroLeadingRLE:
        // An RLE starting with 0 (foreground at pixel 0) should roundtrip.
        let rle = Rle {
            h: 5,
            w: 5,
            counts: vec![0, 3, 22],
        };
        let s = rle_to_string(&rle);
        let decoded = rle_from_string(&s, 5, 5).unwrap();
        assert_eq!(rle.counts, decoded.counts);
        let mask = decode(&decoded);
        assert_eq!(mask[0], 1);
        assert_eq!(mask[1], 1);
        assert_eq!(mask[2], 1);
        assert_eq!(mask[3], 0);
    }

    #[test]
    fn test_rle_string_delta_encoding() {
        // Test that delta encoding works with many runs (i > 2 triggers delta).
        let rle = Rle {
            h: 100,
            w: 100,
            counts: vec![10, 20, 30, 40, 50, 60, 9790],
        };
        let s = rle_to_string(&rle);
        let decoded = rle_from_string(&s, 100, 100).unwrap();
        assert_eq!(rle.counts, decoded.counts);
    }

    #[test]
    fn test_rle_string_invalid_counts() {
        // Matches ppwwyyxx/cocoapi testInvalidRLECounts:
        // RLE counts exceeding h*w should return an error, not panic.
        let rle = Rle {
            h: 5,
            w: 5,
            counts: vec![10, 20], // sum=30 > 25
        };
        let s = rle_to_string(&rle);
        assert!(rle_from_string(&s, 5, 5).is_err());
    }

    #[test]
    fn test_decode_overflow_counts() {
        // Decode should not panic even if counts exceed h*w.
        let rle = Rle {
            h: 2,
            w: 2,
            counts: vec![3, 5], // sum=8 > 4
        };
        let mask = decode(&rle);
        assert_eq!(mask.len(), 4);
        // First 3 should be 0, but only 4 pixels total, so clamped
        assert_eq!(mask, vec![0, 0, 0, 1]);
    }

    #[test]
    fn test_fr_bbox() {
        let rle = fr_bbox(&[1.0, 1.0, 2.0, 2.0], 5, 5).unwrap();
        let mask = decode(&rle);
        // Column-major, 5x5
        // Col 0: [0,0,0,0,0], Col 1: [0,1,1,0,0], Col 2: [0,1,1,0,0], Col 3-4: zeros
        let expected = vec![
            0, 0, 0, 0, 0, // col 0
            0, 1, 1, 0, 0, // col 1
            0, 1, 1, 0, 0, // col 2
            0, 0, 0, 0, 0, // col 3
            0, 0, 0, 0, 0, // col 4
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_fr_poly_triangle() {
        // Simple triangle in a 10x10 image
        // Vertices: (2,2), (7,2), (4,7)
        let poly = vec![2.0, 2.0, 7.0, 2.0, 4.0, 7.0];
        let rle = fr_poly(&poly, 10, 10).unwrap();
        let a = area(&rle);
        // pycocotools gives area=12 for this triangle
        assert_eq!(a, 12, "Triangle area should match pycocotools");
    }

    #[test]
    fn test_fr_poly_large_area() {
        // Ann 2551 from COCO val2017: 96 vertices, 612x612 image
        // pycocotools mask area = 79002
        let poly = vec![
            147.76, 396.11, 158.48, 355.91, 153.12, 347.87, 137.04, 346.26, 125.25, 339.29, 124.71,
            301.77, 139.18, 262.64, 159.55, 232.63, 185.82, 209.04, 226.01, 196.72, 244.77, 196.18,
            251.74, 202.08, 275.33, 224.59, 283.9, 232.63, 295.16, 240.67, 315.53, 247.1, 327.85,
            249.78, 338.57, 253.0, 354.12, 263.72, 379.31, 276.04, 395.39, 286.23, 424.33, 304.99,
            454.95, 336.93, 479.62, 387.02, 491.58, 436.36, 494.57, 453.55, 497.56, 463.27, 493.08,
            511.86, 487.02, 532.62, 470.4, 552.99, 401.26, 552.99, 399.65, 547.63, 407.15, 535.3,
            389.46, 536.91, 374.46, 540.13, 356.23, 540.13, 354.09, 536.91, 341.23, 533.16, 340.15,
            526.19, 342.83, 518.69, 355.7, 512.26, 360.52, 510.65, 374.46, 510.11, 375.53, 494.03,
            369.1, 497.25, 361.06, 491.89, 361.59, 488.67, 354.63, 489.21, 346.05, 496.71, 343.37,
            492.42, 335.33, 495.64, 333.19, 489.21, 327.83, 488.67, 323.0, 499.39, 312.82, 520.83,
            304.24, 531.02, 291.91, 535.84, 273.69, 536.91, 269.4, 533.7, 261.36, 533.7, 256.0,
            531.02, 254.93, 524.58, 268.33, 509.58, 277.98, 505.82, 287.09, 505.29, 301.56, 481.7,
            302.1, 462.41, 294.06, 481.17, 289.77, 488.14, 277.98, 489.74, 261.36, 489.21, 254.93,
            488.67, 254.93, 484.38, 244.75, 482.24, 247.96, 473.66, 260.83, 467.23, 276.37, 464.02,
            283.34, 446.33, 285.48, 431.32, 287.63, 412.02, 277.98, 407.74, 260.29, 403.99, 257.61,
            401.31, 255.47, 391.12, 233.8, 389.37, 220.18, 393.91, 210.65, 393.91, 199.76, 406.61,
            187.51, 417.96, 178.43, 420.68, 167.99, 420.68, 163.45, 418.41, 158.01, 419.32, 148.47,
            418.41, 145.3, 413.88, 146.66, 402.53,
        ];
        let rle = fr_poly(&poly, 612, 612).unwrap();
        let a = area(&rle);
        assert!(
            (a as i64 - 79002).abs() <= 2,
            "Area {} should be within 2 of 79002",
            a
        );
    }

    /// Regression test: fr_bbox at origin produces counts=[0, ...] which has
    /// a 0-length initial run. intersection_area and merge_two must use `while`
    /// (not `if`) to skip these, otherwise IoU computes incorrectly.
    #[test]
    fn test_iou_bbox_at_origin() {
        let r1 = fr_bbox(&[0.0, 0.0, 10.0, 10.0], 20, 20).unwrap();
        let r2 = fr_bbox(&[0.0, 0.0, 10.0, 10.0], 20, 20).unwrap();
        // Identical masks → IoU = 1.0
        let ious = iou(
            std::slice::from_ref(&r1),
            std::slice::from_ref(&r2),
            &[false],
        );
        assert!(
            (ious[0][0] - 1.0).abs() < 1e-10,
            "Identical origin bboxes should have IoU=1.0, got {}",
            ious[0][0]
        );

        // Partially overlapping at origin
        let r3 = fr_bbox(&[0.0, 0.0, 5.0, 10.0], 20, 20).unwrap();
        let ious2 = iou(
            std::slice::from_ref(&r3),
            std::slice::from_ref(&r1),
            &[false],
        );
        // intersection = 5*10 = 50, union = 50 + 100 - 50 = 100
        assert!(
            (ious2[0][0] - 0.5).abs() < 1e-10,
            "Origin bbox IoU should be 0.5, got {}",
            ious2[0][0]
        );

        // Verify the RLE starts with a 0-count (the bug trigger)
        assert_eq!(
            r1.counts[0], 0,
            "fr_bbox at origin should produce counts starting with 0"
        );
    }

    /// Regression test: merge of masks at origin (0-length initial runs).
    #[test]
    fn test_merge_bbox_at_origin() {
        let r1 = fr_bbox(&[0.0, 0.0, 10.0, 10.0], 20, 20).unwrap();
        let r2 = fr_bbox(&[5.0, 0.0, 10.0, 10.0], 20, 20).unwrap();
        // Union area = 15*10 = 150
        let union = merge(&[r1.clone(), r2.clone()], false).unwrap();
        assert_eq!(area(&union), 150, "Union of overlapping origin masks");
        // Intersection area = 5*10 = 50
        let inter = merge(&[r1, r2], true).unwrap();
        assert_eq!(area(&inter), 50, "Intersection of overlapping origin masks");
    }

    #[test]
    fn test_fr_poly_rect_nonsquare() {
        // 40x40 rectangle in a 200h x 100w image
        let poly = vec![10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0];
        let rle = fr_poly(&poly, 200, 100).unwrap();
        let a = area(&rle);
        // pycocotools gives area=1600 for this rect
        assert_eq!(a, 1600, "Rect area should match pycocotools");
    }

    #[test]
    fn test_rle_from_string_byte_below_48() {
        // Byte value below ASCII '0' (48) should return an error, not panic
        let bad = "\x1f"; // byte 31
        assert!(rle_from_string(bad, 10, 10).is_err());

        let bad2 = "\x00"; // null byte
        assert!(rle_from_string(bad2, 10, 10).is_err());
    }

    #[test]
    fn test_rle_from_string_negative_count() {
        // A crafted string that decodes to a negative run count should error.
        // Encode a valid RLE, then corrupt the string to produce a negative delta.
        // The simplest negative encoding: a single group with sign bit set and value -1.
        // bits: value=0x1f (all 5 bits set), no-more flag (bit 5 clear) → byte = 0x1f + 48 = 79 = 'O'
        // This decodes to raw=0x1f=31, sign-extended → -1
        let negative_one = "O"; // byte 79 = 48 + 31, decodes to x=31, sign-extended to -1
        assert!(rle_from_string(negative_one, 10, 10).is_err());
    }

    /// maskApi.c's stage 1 verbatim: every boundary point of every edge, none
    /// skipped. The reference [`walk_edge`] must reproduce through
    /// [`boundary_to_rle`].
    fn fr_poly_full_walk(xy: &[f64], h: u32, w: u32) -> Rle {
        let k = xy.len() / 2;
        let up = |c: f64| (SCALE * c + 0.5) as i32;
        let x: Vec<i32> = (0..=k).map(|j| up(xy[(j % k) * 2])).collect();
        let y: Vec<i32> = (0..=k).map(|j| up(xy[(j % k) * 2 + 1])).collect();
        let (mut u, mut v) = (Vec::new(), Vec::new());
        for j in 0..k {
            let (mut xs, mut xe, mut ys, mut ye) = (x[j], x[j + 1], y[j], y[j + 1]);
            let dx = (xe - xs).abs();
            let dy = (ys - ye).abs();
            let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
            if flip {
                std::mem::swap(&mut xs, &mut xe);
                std::mem::swap(&mut ys, &mut ye);
            }
            if dx >= dy {
                let s = if dx == 0 {
                    0.0
                } else {
                    (ye - ys) as f64 / dx as f64
                };
                for d in 0..=dx {
                    let t = if flip { dx - d } else { d };
                    u.push(t + xs);
                    v.push((interp(s, t as f64, ys as f64) + 0.5) as i32);
                }
            } else {
                let s = (xe - xs) as f64 / dy as f64;
                for d in 0..=dy {
                    let t = if flip { dy - d } else { d };
                    v.push(t + ys);
                    u.push((interp(s, t as f64, xs as f64) + 0.5) as i32);
                }
            }
        }
        boundary_to_rle(&u, &v, &mut Vec::new(), h, w, h * w)
    }

    /// Skipping the points that cannot reach the image changes no mask.
    /// Random polygons whose vertices reach several image-extents past every
    /// edge, so every band boundary is crossed in both directions by x-major
    /// and y-major edges, against the full walk.
    #[test]
    fn test_fr_poly_skipped_walk_matches_full_walk() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..3000 {
            let h = 1 + (next() * 40.0) as u32;
            let w = 1 + (next() * 40.0) as u32;
            let k = 3 + (next() * 6.0) as usize;
            // Mostly near the image, sometimes far out, sometimes on a
            // half-pixel so upsampled vertices land exactly on band edges.
            let coord = |r: f64, extent: u32, far: f64| -> f64 {
                let span = extent as f64 * if far < 0.3 { 12.0 } else { 3.0 };
                let c = (r - 0.5) * span + extent as f64 / 2.0;
                if far > 0.8 {
                    (c * 2.0).round() / 2.0
                } else {
                    c
                }
            };
            let xy: Vec<f64> = (0..k)
                .flat_map(|_| {
                    let (rx, ry, fx, fy) = (next(), next(), next(), next());
                    [coord(rx, w, fx), coord(ry, h, fy)]
                })
                .collect();
            let fast = fr_poly(&xy, h, w).unwrap();
            let full = fr_poly_full_walk(&xy, h, w);
            assert_eq!(fast.counts, full.counts, "case {case}: {xy:?} on {h}x{w}");
        }
    }

    /// Vertices a million pixels out — past the image on the major axis, the
    /// minor axis, and both — rasterize like the full walk.
    #[test]
    fn test_fr_poly_far_vertex_matches_full_walk() {
        let (h, w) = (480, 640);
        for xy in [
            vec![10.0, 10.0, 1e6, 200.0, 300.0, 400.0],
            vec![10.0, 10.0, 200.0, 1e6, 300.0, 400.0],
            vec![-1e6, -1e6, 1e6, 5.0, 300.0, 1e6],
        ] {
            let fast = fr_poly(&xy, h, w).unwrap();
            assert_eq!(fast.counts, fr_poly_full_walk(&xy, h, w).counts, "{xy:?}");
        }
    }
}
