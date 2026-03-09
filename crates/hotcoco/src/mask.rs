//! Pure Rust implementation of COCO mask operations (RLE encoding/decoding, IoU, merge, etc.)
//!
//! This is a faithful port of the C `maskApi.c` from pycocotools/cocoapi.
//! The scan-line polygon rasterization and LEB128-like string encoding match
//! the original exactly to ensure metric parity.

use rayon::prelude::*;

use crate::types::Rle;

/// Minimum D×G product before IoU computation switches from sequential to parallel (rayon).
/// Below this threshold, thread dispatch overhead exceeds the parallelism benefit.
const MIN_PARALLEL_WORK: usize = 1024;

/// Encode a column-major binary mask into RLE.
///
/// `mask` is stored in column-major order (Fortran order): pixel (x, y) is at index `y + h * x`.
/// Length must be `h * w`.
pub fn encode(mask: &[u8], h: u32, w: u32) -> Rle {
    let n = (h as usize) * (w as usize);
    assert_eq!(mask.len(), n, "mask length must equal h*w");

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

    Rle { h, w, counts }
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
pub fn area(rle: &Rle) -> u64 {
    rle.counts
        .iter()
        .skip(1)
        .step_by(2)
        .map(|&c| c as u64)
        .sum()
}

/// Compute the bounding box `[x, y, w, h]` of an RLE mask.
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
        if i % 2 == 1 {
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
pub fn merge(rles: &[Rle], intersect: bool) -> Rle {
    if rles.is_empty() {
        return Rle {
            h: 0,
            w: 0,
            counts: vec![0],
        };
    }
    if rles.len() == 1 {
        return rles[0].clone();
    }

    let h = rles[0].h;
    let w = rles[0].w;

    // Merge pairwise
    let mut result = rles[0].clone();
    for rle in &rles[1..] {
        result = merge_two(&result, rle, intersect);
    }
    // Ensure h/w stay correct
    result.h = h;
    result.w = w;
    result
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

        let step = if ca > 0 && cb > 0 {
            ca.min(cb)
        } else if ca > 0 {
            ca
        } else if cb > 0 {
            cb
        } else {
            break;
        };

        let v = if intersect { va && vb } else { va || vb };

        match v_prev {
            Some(prev) if prev == v => {
                // Extend the last run
                if let Some(last) = counts.last_mut() {
                    *last += step as u32;
                }
            }
            _ => {
                // If we need to start with 1 but there's no leading 0 run, add a 0-length run
                if counts.is_empty() && v {
                    counts.push(0);
                }
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
        counts.push(n as u32);
    }

    Rle { h, w, counts }
}

/// Compute the intersection area of two RLE masks without allocating.
///
/// Walks both RLE streams simultaneously (same logic as `merge_two` with intersect=true)
/// but only accumulates the count where both masks are foreground.
fn intersection_area(a: &Rle, b: &Rle) -> u64 {
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
        if ca == 0 && cb == 0 {
            break;
        }

        let step = if ca > 0 && cb > 0 {
            ca.min(cb)
        } else if ca > 0 {
            ca
        } else {
            cb
        };

        if va && vb {
            count += step;
        }
        if ca > 0 {
            ca -= step;
        }
        if cb > 0 {
            cb -= step;
        }
        total += step;
    }

    count
}

/// Compute IoU between `dt` and `gt` RLE masks.
///
/// Returns a D×G matrix (row-major, `dt.len()` rows, `gt.len()` columns).
/// For `iscrowd[j] == true`, uses crowd IoU: intersection / area(dt) instead of intersection / union.
pub fn iou(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return vec![vec![]; d];
    }

    let dt_areas: Vec<u64> = dt.iter().map(area).collect();
    let gt_areas: Vec<u64> = gt.iter().map(area).collect();

    let compute_row = |i: usize| {
        let dt_a = dt_areas[i] as f64;
        let mut row = vec![0.0f64; g];
        for j in 0..g {
            let inter = intersection_area(&dt[i], &gt[j]);
            let gt_a = gt_areas[j] as f64;
            let inter_f = inter as f64;
            let iou_val = if iscrowd[j] {
                if dt_a == 0.0 {
                    0.0
                } else {
                    inter_f / dt_a
                }
            } else {
                let union = dt_a + gt_a - inter_f;
                if union == 0.0 {
                    0.0
                } else {
                    inter_f / union
                }
            };
            row[j] = iou_val;
        }
        row
    };

    // Use rayon only when D×G is large enough to offset thread dispatch overhead.
    if d * g >= MIN_PARALLEL_WORK {
        (0..d).into_par_iter().map(compute_row).collect()
    } else {
        (0..d).map(compute_row).collect()
    }
}

/// Compute bbox IoU between sets of bounding boxes.
///
/// Each bbox is `[x, y, w, h]`. Returns D×G matrix.
pub fn bbox_iou(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return vec![vec![]; d];
    }

    // Pre-compute GT areas and right/bottom coordinates (loop-invariant over DT rows).
    let gt_areas: Vec<f64> = gt.iter().map(|b| b[2] * b[3]).collect();
    let gt_x2: Vec<f64> = gt.iter().map(|b| b[0] + b[2]).collect();
    let gt_y2: Vec<f64> = gt.iter().map(|b| b[1] + b[3]).collect();

    let compute_row = |i: usize| {
        let da = dt[i][2] * dt[i][3]; // w * h
        let dt_x2 = dt[i][0] + dt[i][2];
        let dt_y2 = dt[i][1] + dt[i][3];
        let mut row = vec![0.0f64; g];
        for j in 0..g {
            let x1 = dt[i][0].max(gt[j][0]);
            let y1 = dt[i][1].max(gt[j][1]);
            let x2 = dt_x2.min(gt_x2[j]);
            let y2 = dt_y2.min(gt_y2[j]);
            let iw = (x2 - x1).max(0.0);
            let ih = (y2 - y1).max(0.0);
            let inter = iw * ih;

            let iou_val = if iscrowd[j] {
                if da == 0.0 {
                    0.0
                } else {
                    inter / da
                }
            } else {
                let union = da + gt_areas[j] - inter;
                if union == 0.0 {
                    0.0
                } else {
                    inter / union
                }
            };
            row[j] = iou_val;
        }
        row
    };

    // Use rayon only when D×G is large enough to offset thread dispatch overhead.
    if d * g >= MIN_PARALLEL_WORK {
        (0..d).into_par_iter().map(compute_row).collect()
    } else {
        (0..d).map(compute_row).collect()
    }
}

/// Convert a polygon (flat list of `[x0, y0, x1, y1, ...]`) to RLE.
///
/// Faithful port of `rleFrPoly` from maskApi.c.
/// Uses upsampling by 5x, Bresenham-like edge walking, y-boundary detection,
/// and differential RLE encoding — exactly matching the C implementation.
pub fn fr_poly(xy: &[f64], h: u32, w: u32) -> Rle {
    let k = xy.len() / 2;
    if k < 3 {
        return Rle {
            h,
            w,
            counts: vec![(h * w)],
        };
    }

    let scale: f64 = 5.0;
    let h_s = h as i64;
    let w_s = w as i64;

    // Stage 1: Upsample polygon vertices by 5x and walk each edge using a
    // Bresenham-like algorithm to produce dense boundary points (u, v).
    let mut x_int: Vec<i32> = Vec::with_capacity(k + 1);
    let mut y_int: Vec<i32> = Vec::with_capacity(k + 1);
    for j in 0..k {
        x_int.push((scale * xy[j * 2] + 0.5) as i32);
        y_int.push((scale * xy[j * 2 + 1] + 0.5) as i32);
    }
    // Close the polygon by repeating the first vertex
    x_int.push(x_int[0]);
    y_int.push(y_int[0]);

    // Pre-count total boundary points across all edges for allocation
    let mut m_total: usize = 0;
    for j in 0..k {
        m_total += (x_int[j] - x_int[j + 1])
            .unsigned_abs()
            .max((y_int[j] - y_int[j + 1]).unsigned_abs()) as usize
            + 1;
    }

    let mut u: Vec<i32> = Vec::with_capacity(m_total);
    let mut v: Vec<i32> = Vec::with_capacity(m_total);

    // Walk each edge, stepping along the longer axis (dx or dy).
    // If the edge runs "backwards" (right-to-left or bottom-to-top), flip the
    // direction so we always step forward, then reverse the traversal order.
    for j in 0..k {
        let mut xs = x_int[j];
        let mut xe = x_int[j + 1];
        let mut ys = y_int[j];
        let mut ye = y_int[j + 1];
        let dx = (xe - xs).unsigned_abs() as i32;
        let dy = (ys - ye).unsigned_abs() as i32;
        let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if flip {
            std::mem::swap(&mut xs, &mut xe);
            std::mem::swap(&mut ys, &mut ye);
        }
        // Slope of the minor axis per step along the major axis
        let s: f64 = if dx >= dy {
            if dx == 0 {
                0.0
            } else {
                (ye - ys) as f64 / dx as f64
            }
        } else if dy == 0 {
            0.0
        } else {
            (xe - xs) as f64 / dy as f64
        };
        if dx >= dy {
            // Step along x, interpolate y
            for d in 0..=dx {
                let t = if flip { dx - d } else { d };
                u.push(t + xs);
                v.push((ys as f64 + s * t as f64 + 0.5) as i32);
            }
        } else {
            // Step along y, interpolate x
            for d in 0..=dy {
                let t = if flip { dy - d } else { d };
                v.push(t + ys);
                u.push((xs as f64 + s * t as f64 + 0.5) as i32);
            }
        }
    }

    // Stage 2: Detect column transitions (x-boundary crossings) in the upsampled
    // boundary, downsample back to original resolution, and convert directly to
    // column-major flat indices (skipping intermediate bx/by storage).
    let m = u.len();
    let mut a: Vec<u32> = Vec::with_capacity(m);

    for j in 1..m {
        // Only process points where the x-coordinate changed (column crossing)
        if u[j] != u[j - 1] {
            // Determine which column boundary was crossed and downsample
            let xd_raw = if u[j] < u[j - 1] { u[j] } else { u[j] - 1 };
            let xd: f64 = (xd_raw as f64 + 0.5) / scale - 0.5;
            // Skip if this doesn't land on an integer column boundary within image bounds
            if xd != xd.floor() || xd < 0.0 || xd > (w_s - 1) as f64 {
                continue;
            }
            // Downsample the y-coordinate and clamp to image bounds
            let yd_raw = if v[j] < v[j - 1] { v[j] } else { v[j - 1] };
            let mut yd: f64 = (yd_raw as f64 + 0.5) / scale - 0.5;
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
    a.push(h * w);
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
/// Computes column-major RLE counts analytically from bbox coordinates
/// without allocating a full pixel mask.
pub fn fr_bbox(bb: &[f64; 4], h: u32, w: u32) -> Rle {
    let bx = bb[0];
    let by = bb[1];
    let bw = bb[2];
    let bh = bb[3];

    // Clamp to image bounds
    let xs = bx.max(0.0).floor() as u32;
    let ys = by.max(0.0).floor() as u32;
    let xe = ((bx + bw).ceil() as u32).min(w);
    let ye = ((by + bh).ceil() as u32).min(h);

    if xs >= xe || ys >= ye {
        return Rle {
            h,
            w,
            counts: vec![h * w],
        };
    }

    // In column-major order, each column within [xs, xe) has the pattern:
    //   ys zeros (from row 0 to ys), (ye - ys) ones, (h - ye) zeros
    // The first column starts at offset xs * h.
    // Between columns, the trailing zeros of one column merge with the leading zeros of the next.
    let col_ones = ye - ys;
    let num_cols = xe - xs;

    let mut counts = Vec::with_capacity((2 * num_cols + 2) as usize);

    // Leading zeros before first foreground pixel
    let leading = xs * h + ys;
    counts.push(leading);

    if num_cols == 1 {
        // Single column: ones, then trailing zeros
        counts.push(col_ones);
        let trailing = (w - xe) * h + (h - ye);
        if trailing > 0 {
            counts.push(trailing);
        }
    } else {
        // First column ones
        counts.push(col_ones);

        // For columns 1..num_cols-1, gap between columns = (h - ye) + ys
        let gap = h - col_ones; // = (h - ye) + ys
        for _ in 1..num_cols - 1 {
            counts.push(gap);
            counts.push(col_ones);
        }

        // Last column: gap, ones, trailing
        counts.push(gap);
        counts.push(col_ones);

        let trailing = (w - xe) * h + (h - ye);
        if trailing > 0 {
            counts.push(trailing);
        }
    }

    Rle { h, w, counts }
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
pub fn rle_from_string(s: &str, h: u32, w: u32) -> Result<Rle, String> {
    let bytes = s.as_bytes();
    let mut counts = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        let mut x: i64 = 0;
        let mut shift = 0;
        let mut more = true;
        while more && i < bytes.len() {
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
        counts.push(x as u32);
    }

    // Validate total counts don't exceed h*w
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let hw = h as u64 * w as u64;
    if total > hw {
        return Err(format!("invalid RLE: total counts {total} exceed h*w={hw}"));
    }

    Ok(Rle { h, w, counts })
}

/// Convert multiple polygons for a single object to a single merged RLE.
///
/// This corresponds to what pycocotools does when converting polygon segmentation:
/// rasterize each polygon separately, then merge all with union.
pub fn fr_polys(polygons: &[Vec<f64>], h: u32, w: u32) -> Rle {
    if polygons.is_empty() {
        return Rle {
            h,
            w,
            counts: vec![h * w],
        };
    }
    let rles: Vec<Rle> = polygons.iter().map(|p| fr_poly(p, h, w)).collect();
    merge(&rles, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let mask = vec![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0];
        let rle = encode(&mask, 3, 4);
        let decoded = decode(&rle);
        assert_eq!(mask, decoded);
    }

    #[test]
    fn test_encode_all_zeros() {
        let mask = vec![0u8; 12];
        let rle = encode(&mask, 3, 4);
        assert_eq!(rle.counts, vec![12]);
    }

    #[test]
    fn test_encode_all_ones() {
        let mask = vec![1u8; 12];
        let rle = encode(&mask, 3, 4);
        assert_eq!(rle.counts, vec![0, 12]);
    }

    #[test]
    fn test_area() {
        let mask = vec![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0];
        let rle = encode(&mask, 3, 4);
        assert_eq!(area(&rle), 5);
    }

    #[test]
    fn test_to_bbox() {
        // 3 rows x 4 cols, column-major
        // Col 0: [0,0,0], Col 1: [1,1,1], Col 2: [0,0,1], Col 3: [1,0,0]
        let mask = vec![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0];
        let rle = encode(&mask, 3, 4);
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
        let r1 = encode(&m1, 3, 4);
        let r2 = encode(&m2, 3, 4);
        let merged = merge(&[r1, r2], false);
        let decoded = decode(&merged);
        let expected = vec![0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0];
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_merge_intersection() {
        let m1 = vec![0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        let m2 = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0];
        let r1 = encode(&m1, 3, 4);
        let r2 = encode(&m2, 3, 4);
        let merged = merge(&[r1, r2], true);
        let decoded = decode(&merged);
        let expected = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0];
        assert_eq!(decoded, expected);
    }

    #[test]
    fn test_iou_basic() {
        let m1 = vec![0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        let m2 = vec![0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0];
        let r1 = encode(&m1, 3, 4);
        let r2 = encode(&m2, 3, 4);
        let ious = iou(&[r1], &[r2], &[false]);
        // intersection = 2, union = 3 + 3 - 2 = 4
        assert!((ious[0][0] - 0.5).abs() < 1e-10);
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
        let rle = fr_bbox(&[1.0, 1.0, 2.0, 2.0], 5, 5);
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
        let rle = fr_poly(&poly, 10, 10);
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
        let rle = fr_poly(&poly, 612, 612);
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
        let r1 = fr_bbox(&[0.0, 0.0, 10.0, 10.0], 20, 20);
        let r2 = fr_bbox(&[0.0, 0.0, 10.0, 10.0], 20, 20);
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
        let r3 = fr_bbox(&[0.0, 0.0, 5.0, 10.0], 20, 20);
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
        let r1 = fr_bbox(&[0.0, 0.0, 10.0, 10.0], 20, 20);
        let r2 = fr_bbox(&[5.0, 0.0, 10.0, 10.0], 20, 20);
        // Union area = 15*10 = 150
        let union = merge(&[r1.clone(), r2.clone()], false);
        assert_eq!(area(&union), 150, "Union of overlapping origin masks");
        // Intersection area = 5*10 = 50
        let inter = merge(&[r1, r2], true);
        assert_eq!(area(&inter), 50, "Intersection of overlapping origin masks");
    }

    #[test]
    fn test_fr_poly_rect_nonsquare() {
        // 40x40 rectangle in a 200h x 100w image
        let poly = vec![10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0];
        let rle = fr_poly(&poly, 200, 100);
        let a = area(&rle);
        // pycocotools gives area=1600 for this rect
        assert_eq!(a, 1600, "Rect area should match pycocotools");
    }
}
