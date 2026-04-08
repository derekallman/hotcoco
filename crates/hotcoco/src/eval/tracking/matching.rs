use crate::mask;

use super::types::MatchResult;

/// Compute IoU matrix between detection and ground truth bounding boxes.
///
/// Returns a D x G matrix where D = number of detections, G = number of ground truths.
/// All ground truths are treated as non-crowd (tracking doesn't use crowd semantics).
pub(crate) fn compute_iou_matrix(dt_bboxes: &[[f64; 4]], gt_bboxes: &[[f64; 4]]) -> Vec<Vec<f64>> {
    if dt_bboxes.is_empty() || gt_bboxes.is_empty() {
        return vec![vec![]; dt_bboxes.len()];
    }
    let iscrowd = vec![false; gt_bboxes.len()];
    mask::bbox_iou(dt_bboxes, gt_bboxes, &iscrowd)
}

/// Hungarian (optimal) matching using the LAPJV algorithm.
///
/// Finds the minimum-cost assignment between detections and ground truths,
/// then filters to pairs with IoU >= threshold. `num_gt` and `num_dt` are
/// passed explicitly because the IoU matrix may be empty when one side has
/// no objects (losing the other dimension's count).
pub(crate) fn hungarian_match(
    iou_matrix: &[Vec<f64>],
    threshold: f64,
    num_gt: usize,
    num_dt: usize,
) -> MatchResult {
    if num_dt == 0 || num_gt == 0 {
        return MatchResult {
            matches: vec![],
            unmatched_gt: (0..num_gt).collect(),
            unmatched_dt: (0..num_dt).collect(),
        };
    }

    // LAPJV requires a square matrix; pad to max(num_dt, num_gt).
    let n = num_dt.max(num_gt);
    let big_cost = 1e6;

    let mut cost_matrix = ndarray::Array2::from_elem((n, n), big_cost);
    for di in 0..num_dt {
        for gi in 0..num_gt {
            if iou_matrix[di][gi] >= threshold {
                cost_matrix[[di, gi]] = 1.0 - iou_matrix[di][gi];
            }
        }
    }

    let result = lapjv::lapjv(&cost_matrix);
    let (row_to_col, _) = match result {
        Ok(r) => r,
        Err(_) => {
            return MatchResult {
                matches: vec![],
                unmatched_gt: (0..num_gt).collect(),
                unmatched_dt: (0..num_dt).collect(),
            };
        }
    };

    let mut matches = Vec::new();
    let mut matched_gt = vec![false; num_gt];
    let mut matched_dt = vec![false; num_dt];

    for (di, &gi) in row_to_col.iter().enumerate() {
        if di < num_dt && gi < num_gt && iou_matrix[di][gi] >= threshold {
            matches.push((di, gi));
            matched_dt[di] = true;
            matched_gt[gi] = true;
        }
    }

    let unmatched_gt = (0..num_gt).filter(|&i| !matched_gt[i]).collect();
    let unmatched_dt = (0..num_dt).filter(|&i| !matched_dt[i]).collect();

    MatchResult {
        matches,
        unmatched_gt,
        unmatched_dt,
    }
}
