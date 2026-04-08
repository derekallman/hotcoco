use std::collections::HashMap;

use super::matching::hungarian_match;
use super::types::{FrameData, HotaResult, TrackIdMaps};

/// Compute HOTA metrics for a single sequence across multiple alpha thresholds.
///
/// Follows TrackEval's algorithm:
/// 1. Precompute `global_alignment_score` — a Jaccard-like cross-frame association
///    quality between each (GT_track, DT_track) pair.
/// 2. Per frame: run Hungarian matching on `global_alignment_score * IoU`, producing
///    one matching per frame (shared across all alpha thresholds).
/// 3. Per alpha: filter the matching to IoU >= alpha, count TP/FP/FN, compute
///    DetA, AssA, LocA, HOTA.
pub(crate) fn compute_hota(frames: &[FrameData], iou_thrs: &[f64]) -> HotaResult {
    let num_thrs = iou_thrs.len();

    // --- Pass 1: Compute global alignment score between track ID pairs ---

    let id_maps = TrackIdMaps::build(frames);
    let num_gt_ids = id_maps.gt_ids.len();
    let num_dt_ids = id_maps.dt_ids.len();

    // potential_matches_count[gi][di] = sum of sim_iou across frames
    // where sim_iou is a normalised version of the per-frame IoU
    let mut potential_matches_count = vec![vec![0.0f64; num_dt_ids]; num_gt_ids];
    let mut gt_id_count = vec![0u64; num_gt_ids];
    let mut dt_id_count = vec![0u64; num_dt_ids];

    // Precompute global track indices per frame (reused in Pass 2).
    let frame_globals: Vec<(Vec<usize>, Vec<usize>)> = frames
        .iter()
        .map(|frame| {
            let gt = frame
                .gt_track_ids
                .iter()
                .map(|tid| id_maps.gt_map[tid])
                .collect();
            let dt = frame
                .dt_track_ids
                .iter()
                .map(|tid| id_maps.dt_map[tid])
                .collect();
            (gt, dt)
        })
        .collect();

    for (frame, (gt_global, dt_global)) in frames.iter().zip(&frame_globals) {
        let num_gt = gt_global.len();
        let num_dt = dt_global.len();

        // Count appearances (deduplicate per frame — NumPy's fancy indexing
        // with duplicate indices only increments once per frame).
        {
            let mut seen = std::collections::HashSet::new();
            for &gi in gt_global {
                if seen.insert(gi) {
                    gt_id_count[gi] += 1;
                }
            }
            seen.clear();
            for &di in dt_global {
                if seen.insert(di) {
                    dt_id_count[di] += 1;
                }
            }
        }

        if num_gt == 0 || num_dt == 0 {
            continue;
        }

        // Compute sim_iou (normalised IoU) for this frame:
        // sim_iou_denom[g][d] = sum_col(sim[g,:]) + sum_row(sim[:,d]) - sim[g][d]
        // sim_iou[g][d] = sim[g][d] / sim_iou_denom[g][d]
        let sim = &frame.iou_matrix; // D x G
        let mut col_sums = vec![0.0; num_gt]; // sum over DTs for each GT
        let mut row_sums = vec![0.0; num_dt]; // sum over GTs for each DT
        for (di, row) in sim.iter().enumerate() {
            for (gi, &val) in row.iter().enumerate() {
                col_sums[gi] += val;
                row_sums[di] += val;
            }
        }

        // Accumulate sim_iou into potential_matches_count.
        // NumPy fancy indexing with duplicate (gt_id, dt_id) pairs keeps only
        // the last value (overwrite, not accumulate). We replicate this by
        // collecting per-pair values and keeping the last.
        let mut pair_values: HashMap<(usize, usize), f64> = HashMap::new();
        for (local_gi, &gi) in gt_global.iter().enumerate() {
            for (local_di, &di_global) in dt_global.iter().enumerate() {
                let s = sim[local_di][local_gi]; // iou_matrix is D x G
                let denom = col_sums[local_gi] + row_sums[local_di] - s;
                if denom > f64::EPSILON {
                    // Last write wins (matching NumPy fancy indexing)
                    pair_values.insert((gi, di_global), s / denom);
                }
            }
        }
        for (&(gi, di), &val) in &pair_values {
            potential_matches_count[gi][di] += val;
        }
    }

    // global_alignment_score[gi][di] = Jaccard of track-pair co-occurrence
    let mut global_alignment_score = vec![vec![0.0f64; num_dt_ids]; num_gt_ids];
    for gi in 0..num_gt_ids {
        for di in 0..num_dt_ids {
            let denom =
                gt_id_count[gi] as f64 + dt_id_count[di] as f64 - potential_matches_count[gi][di];
            if denom > f64::EPSILON {
                global_alignment_score[gi][di] = potential_matches_count[gi][di] / denom;
            }
        }
    }

    // --- Pass 2: Per-frame Hungarian matching using global_alignment_score * IoU ---
    // One matching per frame, then filter per alpha threshold.

    // Per-alpha accumulators
    let mut hota_tp = vec![0u64; num_thrs];
    let mut hota_fn = vec![0u64; num_thrs];
    let mut hota_fp = vec![0u64; num_thrs];
    let mut loc_sum = vec![0.0; num_thrs];

    // matches_counts[alpha][gi][di] = number of frames where gi and di are matched at this alpha
    let mut matches_counts: Vec<Vec<Vec<u64>>> =
        vec![vec![vec![0u64; num_dt_ids]; num_gt_ids]; num_thrs];

    for (frame, (gt_global, dt_global)) in frames.iter().zip(&frame_globals) {
        let num_gt = gt_global.len();
        let num_dt = dt_global.len();

        if num_gt == 0 {
            for fp in &mut hota_fp {
                *fp += num_dt as u64;
            }
            continue;
        }
        if num_dt == 0 {
            for r#fn in &mut hota_fn {
                *r#fn += num_gt as u64;
            }
            continue;
        }

        // Build score matrix: global_alignment_score * IoU
        let mut score_mat = vec![vec![0.0; num_gt]; num_dt];
        for local_di in 0..num_dt {
            for local_gi in 0..num_gt {
                let iou = frame.iou_matrix[local_di][local_gi];
                let gas = global_alignment_score[gt_global[local_gi]][dt_global[local_di]];
                score_mat[local_di][local_gi] = gas * iou;
            }
        }

        // Hungarian match on the combined score (threshold=0 to get all matches)
        let match_result = hungarian_match(&score_mat, 0.0, num_gt, num_dt);

        // Filter per alpha threshold
        let mut seen_pairs: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for (a, &alpha) in iou_thrs.iter().enumerate() {
            let mut num_matches = 0u64;
            // Deduplicate (gt_id, dt_id) pairs per frame — NumPy fancy indexing
            // with duplicate index pairs only increments once.
            seen_pairs.clear();
            for &(di, gi) in &match_result.matches {
                let iou = frame.iou_matrix[di][gi];
                if iou >= alpha - f64::EPSILON {
                    num_matches += 1;
                    loc_sum[a] += iou;
                    let pair = (gt_global[gi], dt_global[di]);
                    if seen_pairs.insert(pair) {
                        matches_counts[a][pair.0][pair.1] += 1;
                    }
                }
            }
            hota_tp[a] += num_matches;
            hota_fn[a] += num_gt as u64 - num_matches;
            hota_fp[a] += num_dt as u64 - num_matches;
        }
    }

    // --- Pass 3: Compute association metrics per alpha ---

    let mut hota = vec![0.0; num_thrs];
    let mut det_a = vec![0.0; num_thrs];
    let mut ass_a = vec![0.0; num_thrs];
    // LocA defaults to 1.0 (TrackEval convention: no matches → perfect localization).
    let mut loc_a = vec![1.0; num_thrs];
    let mut det_re = vec![0.0; num_thrs];
    let mut det_pr = vec![0.0; num_thrs];
    let mut ass_re = vec![0.0; num_thrs];
    let mut ass_pr = vec![0.0; num_thrs];
    let mut ass_sum = vec![0.0; num_thrs];

    for a in 0..num_thrs {
        let tp = hota_tp[a];
        let r#fn = hota_fn[a];
        let fp = hota_fp[a];
        let total = tp + r#fn + fp;

        if total > 0 {
            det_a[a] = tp as f64 / total as f64;
            let gt_total = tp + r#fn;
            let dt_total = tp + fp;
            det_re[a] = if gt_total > 0 {
                tp as f64 / gt_total as f64
            } else {
                0.0
            };
            det_pr[a] = if dt_total > 0 {
                tp as f64 / dt_total as f64
            } else {
                0.0
            };
        }

        if tp > 0 {
            loc_a[a] = loc_sum[a] / tp as f64;

            // AssA, AssRe, AssPr from matches_counts (following TrackEval):
            // For each (gi, di) pair with matches_count > 0:
            //   ass_a_pair = matches_count / (gt_id_count[gi] + dt_id_count[di] - matches_count)
            //   Weighted sum: AssA = sum(matches_count * ass_a_pair) / TP
            let mc = &matches_counts[a];
            let mut ass_a_sum = 0.0;
            let mut ass_re_sum_val = 0.0;
            let mut ass_pr_sum_val = 0.0;
            for gi in 0..num_gt_ids {
                for di in 0..num_dt_ids {
                    let c = mc[gi][di] as f64;
                    if c > 0.0 {
                        let gt_c = gt_id_count[gi] as f64;
                        let dt_c = dt_id_count[di] as f64;
                        let ass_a_pair = c / (gt_c + dt_c - c).max(1.0);
                        ass_a_sum += c * ass_a_pair;
                        let ass_re_pair = c / gt_c.max(1.0);
                        ass_re_sum_val += c * ass_re_pair;
                        let ass_pr_pair = c / dt_c.max(1.0);
                        ass_pr_sum_val += c * ass_pr_pair;
                    }
                }
            }
            ass_a[a] = ass_a_sum / tp as f64;
            ass_re[a] = ass_re_sum_val / tp as f64;
            ass_pr[a] = ass_pr_sum_val / tp as f64;
            ass_sum[a] = ass_a_sum;

            hota[a] = (det_a[a] * ass_a[a]).sqrt();
        }
    }

    HotaResult {
        hota,
        det_a,
        ass_a,
        loc_a,
        det_re,
        det_pr,
        ass_re,
        ass_pr,
        hota_tp,
        hota_fn,
        hota_fp,
        ass_sum,
        loc_sum,
    }
}
