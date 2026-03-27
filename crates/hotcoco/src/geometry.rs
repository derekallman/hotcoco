//! Computational geometry primitives for oriented bounding box (OBB) evaluation.
//!
//! Provides rotated IoU computation via Sutherland-Hodgman polygon clipping.
//! All angles are in radians, counter-clockwise positive.

use rayon::prelude::*;

/// Minimum D×G work before switching to parallel execution (same threshold as mask::bbox_iou).
const MIN_PARALLEL_WORK: usize = 1000;

/// Convert an OBB `[cx, cy, w, h, angle]` to its 4 corner points.
///
/// Returns corners in counter-clockwise order starting from the corner
/// that corresponds to (-w/2, -h/2) before rotation.
pub fn obb_to_corners(obb: &[f64; 5]) -> [(f64, f64); 4] {
    let [cx, cy, w, h, angle] = *obb;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let hw = w / 2.0;
    let hh = h / 2.0;

    // Half-extents along each local axis
    let dx_w = hw * cos_a;
    let dy_w = hw * sin_a;
    let dx_h = hh * sin_a;
    let dy_h = hh * cos_a;

    [
        (cx - dx_w + dx_h, cy - dy_w - dy_h), // bottom-left before rotation
        (cx + dx_w + dx_h, cy + dy_w - dy_h), // bottom-right
        (cx + dx_w - dx_h, cy + dy_w + dy_h), // top-right
        (cx - dx_w - dx_h, cy - dy_w + dy_h), // top-left
    ]
}

/// Signed area of a polygon using the shoelace formula.
///
/// Positive for counter-clockwise vertices, negative for clockwise.
fn signed_polygon_area(vertices: &[(f64, f64)]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].0 * vertices[j].1;
        area -= vertices[j].0 * vertices[i].1;
    }
    area * 0.5
}

/// Absolute area of a polygon.
fn polygon_area(vertices: &[(f64, f64)]) -> f64 {
    signed_polygon_area(vertices).abs()
}

/// Sutherland-Hodgman polygon clipping: clip `subject` against `clip`.
///
/// Both polygons must be convex. The clip polygon's edges define half-planes;
/// the subject polygon is clipped against each in turn. For two rectangles
/// (4 edges each), the result has at most 8 vertices.
fn sutherland_hodgman_clip(subject: &[(f64, f64)], clip: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if subject.is_empty() || clip.is_empty() {
        return Vec::new();
    }

    let mut output = subject.to_vec();

    let clip_len = clip.len();
    for i in 0..clip_len {
        if output.is_empty() {
            return output;
        }

        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % clip_len];

        let input = std::mem::take(&mut output);
        // Reserve capacity for worst case: each edge can add at most 1 vertex
        output.reserve(input.len() + 1);

        let input_len = input.len();
        for j in 0..input_len {
            let current = input[j];
            let previous = input[(j + input_len - 1) % input_len];

            let curr_inside = is_inside(current, edge_start, edge_end);
            let prev_inside = is_inside(previous, edge_start, edge_end);

            if curr_inside {
                if !prev_inside {
                    // Entering: add intersection then current
                    if let Some(pt) = line_intersection(previous, current, edge_start, edge_end) {
                        output.push(pt);
                    }
                }
                output.push(current);
            } else if prev_inside {
                // Leaving: add intersection only
                if let Some(pt) = line_intersection(previous, current, edge_start, edge_end) {
                    output.push(pt);
                }
            }
        }
    }

    output
}

/// Test if a point is on the left side (inside) of a directed edge.
///
/// Uses the cross product of the edge vector and the point vector.
/// Points exactly on the edge are considered inside.
#[inline]
fn is_inside(point: (f64, f64), edge_start: (f64, f64), edge_end: (f64, f64)) -> bool {
    let cross = (edge_end.0 - edge_start.0) * (point.1 - edge_start.1)
        - (edge_end.1 - edge_start.1) * (point.0 - edge_start.0);
    cross >= 0.0
}

/// Compute the intersection point of two line segments.
///
/// Uses the parametric form to find where lines (p1→p2) and (p3→p4) cross.
/// Returns `None` if the lines are parallel (denominator ≈ 0).
#[inline]
fn line_intersection(
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    p4: (f64, f64),
) -> Option<(f64, f64)> {
    let d1x = p2.0 - p1.0;
    let d1y = p2.1 - p1.1;
    let d2x = p4.0 - p3.0;
    let d2y = p4.1 - p3.1;

    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-12 {
        return None;
    }

    let t = ((p3.0 - p1.0) * d2y - (p3.1 - p1.1) * d2x) / denom;
    Some((p1.0 + t * d1x, p1.1 + t * d1y))
}

/// Compute the axis-aligned bounding box of an OBB.
///
/// Returns `[x, y, w, h]` where `(x, y)` is the top-left corner.
pub fn obb_to_aabb(obb: &[f64; 5]) -> [f64; 4] {
    let corners = obb_to_corners(obb);
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    [min_x, min_y, max_x - min_x, max_y - min_y]
}

/// Convert 4 corner points `[x1,y1,x2,y2,x3,y3,x4,y4]` to `[cx, cy, w, h, angle]`.
///
/// Center is the mean of the 4 corners. Width is the distance between corners 0-1,
/// height is the distance between corners 1-2, angle is atan2 of the first edge.
pub fn corners_to_obb(coords: &[f64]) -> [f64; 5] {
    let cx = (coords[0] + coords[2] + coords[4] + coords[6]) / 4.0;
    let cy = (coords[1] + coords[3] + coords[5] + coords[7]) / 4.0;

    let dx01 = coords[2] - coords[0];
    let dy01 = coords[3] - coords[1];
    let w = (dx01 * dx01 + dy01 * dy01).sqrt();

    let dx12 = coords[4] - coords[2];
    let dy12 = coords[5] - coords[3];
    let h = (dx12 * dx12 + dy12 * dy12).sqrt();

    let angle = dy01.atan2(dx01);

    [cx, cy, w, h, angle]
}

/// Compute IoU between two pre-computed rotated rectangle corner sets.
///
/// `area_a` and `area_b` are the rectangle areas (w × h).
/// Returns 0.0 for zero-area boxes or non-overlapping boxes.
fn obb_iou_pair(
    corners_a: &[(f64, f64); 4],
    area_a: f64,
    corners_b: &[(f64, f64); 4],
    area_b: f64,
    b_is_crowd: bool,
) -> f64 {
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }

    let intersection = sutherland_hodgman_clip(corners_a, corners_b);
    let inter_area = polygon_area(&intersection);

    if inter_area <= 0.0 {
        return 0.0;
    }

    if b_is_crowd {
        inter_area / area_a
    } else {
        let union_area = area_a + area_b - inter_area;
        if union_area <= 0.0 {
            0.0
        } else {
            inter_area / union_area
        }
    }
}

/// Convenience wrapper for computing IoU between two OBB parameter arrays.
#[cfg(test)]
fn obb_iou_single(a: &[f64; 5], b: &[f64; 5], b_is_crowd: bool) -> f64 {
    obb_iou_pair(
        &obb_to_corners(a),
        a[2] * a[3],
        &obb_to_corners(b),
        b[2] * b[3],
        b_is_crowd,
    )
}

/// Compute D×G IoU matrix for oriented bounding boxes.
///
/// `dt` contains detection OBBs `[cx, cy, w, h, angle]`, `gt` contains ground truth OBBs,
/// and `iscrowd` indicates whether each GT is a crowd annotation (one per GT).
///
/// When `iscrowd[j]` is true, IoU = intersection / dt_area (matching bbox crowd semantics).
pub fn obb_iou(dt: &[[f64; 5]], gt: &[[f64; 5]], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return vec![vec![]; d];
    }

    // Pre-compute GT corners and areas (loop-invariant over DT rows).
    let gt_corners: Vec<[(f64, f64); 4]> = gt.iter().map(obb_to_corners).collect();
    let gt_areas: Vec<f64> = gt.iter().map(|b| b[2] * b[3]).collect();

    let compute_row = |i: usize| {
        let corners_a = obb_to_corners(&dt[i]);
        let area_a = dt[i][2] * dt[i][3];
        let mut row = vec![0.0f64; g];
        for j in 0..g {
            row[j] = obb_iou_pair(&corners_a, area_a, &gt_corners[j], gt_areas[j], iscrowd[j]);
        }
        row
    };

    if d * g >= MIN_PARALLEL_WORK {
        (0..d).into_par_iter().map(compute_row).collect()
    } else {
        (0..d).map(compute_row).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const EPS: f64 = 1e-9;

    #[test]
    fn test_obb_to_corners_axis_aligned() {
        // 10×6 box centered at (5, 3), no rotation
        let corners = obb_to_corners(&[5.0, 3.0, 10.0, 6.0, 0.0]);
        // Corners should form a rectangle from (0, 0) to (10, 6)
        let mut xs: Vec<f64> = corners.iter().map(|c| c.0).collect();
        let mut ys: Vec<f64> = corners.iter().map(|c| c.1).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((xs[0] - 0.0).abs() < EPS);
        assert!((xs[3] - 10.0).abs() < EPS);
        assert!((ys[0] - 0.0).abs() < EPS);
        assert!((ys[3] - 6.0).abs() < EPS);
    }

    #[test]
    fn test_obb_to_corners_90deg() {
        // 10×4 box rotated 90° → should become 4×10
        let corners = obb_to_corners(&[0.0, 0.0, 10.0, 4.0, FRAC_PI_2]);
        let mut xs: Vec<f64> = corners.iter().map(|c| c.0).collect();
        let mut ys: Vec<f64> = corners.iter().map(|c| c.1).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // After 90° rotation, width and height swap
        assert!((xs[3] - xs[0] - 4.0).abs() < EPS);
        assert!((ys[3] - ys[0] - 10.0).abs() < EPS);
    }

    #[test]
    fn test_polygon_area_square() {
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!((polygon_area(&square) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_polygon_area_triangle() {
        let tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)];
        assert!((polygon_area(&tri) - 6.0).abs() < EPS);
    }

    #[test]
    fn test_identical_boxes_iou_1() {
        let a = [10.0, 10.0, 20.0, 30.0, 0.5];
        let iou = obb_iou_single(&a, &a, false);
        assert!(
            (iou - 1.0).abs() < EPS,
            "identical boxes should have IoU=1.0, got {iou}"
        );
    }

    #[test]
    fn test_non_overlapping_iou_0() {
        let a = [0.0, 0.0, 2.0, 2.0, 0.0];
        let b = [100.0, 100.0, 2.0, 2.0, 0.0];
        let iou = obb_iou_single(&a, &b, false);
        assert!(
            iou.abs() < EPS,
            "non-overlapping boxes should have IoU=0.0, got {iou}"
        );
    }

    #[test]
    fn test_axis_aligned_overlap() {
        // Two axis-aligned boxes: [0,0,4,4] and [2,0,4,4] → overlap 2×4 = 8
        // Union = 16 + 16 - 8 = 24, IoU = 8/24 = 1/3
        let a = [2.0, 2.0, 4.0, 4.0, 0.0];
        let b = [4.0, 2.0, 4.0, 4.0, 0.0];
        let iou = obb_iou_single(&a, &b, false);
        assert!(
            (iou - 1.0 / 3.0).abs() < 1e-6,
            "expected IoU ≈ 1/3, got {iou}"
        );
    }

    #[test]
    fn test_90deg_rotation_identical() {
        // A square rotated 90° is the same square
        let a = [5.0, 5.0, 4.0, 4.0, 0.0];
        let b = [5.0, 5.0, 4.0, 4.0, FRAC_PI_2];
        let iou = obb_iou_single(&a, &b, false);
        assert!(
            (iou - 1.0).abs() < 1e-6,
            "square rotated 90° should have IoU=1.0, got {iou}"
        );
    }

    #[test]
    fn test_45deg_rotation_partial_overlap() {
        // Square at origin, one rotated 45° — known overlap geometry
        let a = [0.0, 0.0, 2.0, 2.0, 0.0];
        let b = [0.0, 0.0, 2.0, 2.0, FRAC_PI_4];
        let iou = obb_iou_single(&a, &b, false);
        // Both have area 4. Intersection of two squares rotated 45° at same center:
        // the intersection is a regular octagon. IoU should be between 0 and 1.
        assert!(
            iou > 0.5,
            "45° rotated squares should have significant overlap, got {iou}"
        );
        assert!(
            iou < 1.0,
            "45° rotated squares should not fully overlap, got {iou}"
        );
    }

    #[test]
    fn test_iscrowd() {
        // Crowd mode: IoU = intersection / dt_area
        let a = [0.0, 0.0, 2.0, 2.0, 0.0]; // dt, area = 4
        let b = [0.0, 0.0, 4.0, 4.0, 0.0]; // gt crowd, area = 16
                                           // a is fully inside b, so intersection = 4, IoU_crowd = 4/4 = 1.0
        let iou = obb_iou_single(&a, &b, true);
        assert!(
            (iou - 1.0).abs() < EPS,
            "small box inside crowd should have IoU=1.0, got {iou}"
        );

        // Normal mode: IoU = 4 / (4 + 16 - 4) = 4/16 = 0.25
        let iou_normal = obb_iou_single(&a, &b, false);
        assert!(
            (iou_normal - 0.25).abs() < 1e-6,
            "expected IoU=0.25, got {iou_normal}"
        );
    }

    #[test]
    fn test_zero_area_box() {
        let a = [0.0, 0.0, 0.0, 2.0, 0.0]; // zero width
        let b = [0.0, 0.0, 2.0, 2.0, 0.0];
        assert!(obb_iou_single(&a, &b, false).abs() < EPS);
    }

    #[test]
    fn test_obb_iou_matrix() {
        let dt = vec![[0.0, 0.0, 2.0, 2.0, 0.0], [100.0, 100.0, 2.0, 2.0, 0.0]];
        let gt = vec![[0.0, 0.0, 2.0, 2.0, 0.0]];
        let iscrowd = vec![false];
        let result = obb_iou(&dt, &gt, &iscrowd);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 1);
        assert!((result[0][0] - 1.0).abs() < EPS);
        assert!(result[1][0].abs() < EPS);
    }

    #[test]
    fn test_obb_iou_empty() {
        let dt: Vec<[f64; 5]> = vec![];
        let gt = vec![[0.0, 0.0, 2.0, 2.0, 0.0]];
        let iscrowd = vec![false];
        let result = obb_iou(&dt, &gt, &iscrowd);
        assert!(result.is_empty());
    }

    #[test]
    fn test_180deg_rotation() {
        // 180° rotation of a rectangle should give identical corners (just reordered)
        let a = [5.0, 5.0, 6.0, 4.0, 0.0];
        let b = [5.0, 5.0, 6.0, 4.0, PI];
        let iou = obb_iou_single(&a, &b, false);
        assert!(
            (iou - 1.0).abs() < 1e-6,
            "180° rotation should give IoU=1.0, got {iou}"
        );
    }

    #[test]
    fn test_contained_box() {
        // Small box fully inside large box
        let big = [0.0, 0.0, 10.0, 10.0, 0.3];
        let small = [0.0, 0.0, 2.0, 2.0, 0.3]; // same center, same angle, smaller
        let iou = obb_iou_single(&small, &big, false);
        // intersection = 4, union = 100 + 4 - 4 = 100, IoU = 4/100 = 0.04
        assert!((iou - 0.04).abs() < 1e-6, "expected IoU=0.04, got {iou}");
    }
}
