use std::collections::{HashMap, HashSet};

use super::matching::hungarian_match;
use super::types::{ClearResult, FrameData};

/// Per-GT-track state accumulated during the single matching pass.
struct TrackState {
    frames_present: u64,
    frames_tracked: u64,
    /// Was this GT matched in the previous frame it appeared in?
    was_tracked: bool,
    /// Count of not-tracked → tracked transitions (includes first appearance).
    /// Final frag count = max(0, transitions - 1) per track.
    transitions_into_tracked: u64,
}

/// Compute CLEAR MOT metrics for a single sequence.
///
/// Follows TrackEval's matching: per-frame Hungarian on a score matrix that
/// gives a 1000x continuity bonus for matching to the same tracker ID as the
/// previous frame. All statistics are computed in a single pass.
pub(crate) fn compute_clear(frames: &[FrameData], similarity_thr: f64) -> ClearResult {
    let mut clr_tp: u64 = 0;
    let mut clr_fn: u64 = 0;
    let mut clr_fp: u64 = 0;
    let mut num_id_switches: u64 = 0;
    let mut iou_sum: f64 = 0.0;

    // prev_tracker_id[gt_tid] = last matched DT track ID (for IDSW detection)
    let mut prev_tracker_id: HashMap<u64, u64> = HashMap::new();
    // prev_timestep_tracker_id[gt_tid] = DT track ID matched in previous timestep
    // (for continuity bonus and frag counting; reset each frame)
    let mut prev_timestep: HashMap<u64, u64> = HashMap::new();
    let mut gt_stats: HashMap<u64, TrackState> = HashMap::new();

    let mut all_gt_ids: HashSet<u64> = HashSet::new();
    let mut all_dt_ids: HashSet<u64> = HashSet::new();

    for frame in frames {
        for &tid in &frame.gt_track_ids {
            all_gt_ids.insert(tid);
        }
        for &tid in &frame.dt_track_ids {
            all_dt_ids.insert(tid);
        }

        let num_gt = frame.gt_track_ids.len();
        let num_dt = frame.dt_track_ids.len();

        // TrackEval: skip frames with 0 GT (before frag reset)
        if num_gt == 0 {
            clr_fp += num_dt as u64;
            continue;
        }
        // TrackEval: skip frames with 0 DT (before frag reset, preserving prev_timestep)
        if num_dt == 0 {
            clr_fn += num_gt as u64;
            // Count GT appearances for MT/PT/ML but don't touch frag state
            for &gt_tid in &frame.gt_track_ids {
                let s = gt_stats.entry(gt_tid).or_insert(TrackState {
                    frames_present: 0,
                    frames_tracked: 0,
                    was_tracked: false,
                    transitions_into_tracked: 0,
                });
                s.frames_present += 1;
            }
            continue;
        }

        // Build score matrix matching TrackEval:
        // score = 1000 * (dt_id == prev_timestep_dt_id_for_this_gt) + iou
        // Zero out entries below threshold.
        let mut score_mat: Vec<Vec<f64>> = vec![vec![0.0; num_gt]; num_dt];
        for (di, score_row) in score_mat.iter_mut().enumerate() {
            for (gi, score_cell) in score_row.iter_mut().enumerate() {
                let iou = if di < frame.iou_matrix.len() && gi < frame.iou_matrix[di].len() {
                    frame.iou_matrix[di][gi]
                } else {
                    0.0
                };
                if iou >= similarity_thr - f64::EPSILON {
                    let gt_tid = frame.gt_track_ids[gi];
                    let dt_tid = frame.dt_track_ids[di];
                    let continuity_bonus = if prev_timestep.get(&gt_tid) == Some(&dt_tid) {
                        1000.0
                    } else {
                        0.0
                    };
                    *score_cell = continuity_bonus + iou;
                }
            }
        }

        // Hungarian match on negated score (minimize cost = maximize score).
        // Use threshold=EPSILON to match any nonzero entry.
        let match_result = hungarian_match(&score_mat, f64::EPSILON, num_gt, num_dt);

        // Build GT → DT mapping for matched pairs.
        // Deduplicate by GT track ID (NumPy indexing handles duplicates by
        // keeping the last assignment, and only counting IDSW once per ID).
        let mut current_matches: HashMap<u64, u64> = HashMap::new();
        let mut seen_gt_for_idsw: HashSet<u64> = HashSet::new();
        for &(di, gi) in &match_result.matches {
            let gt_tid = frame.gt_track_ids[gi];
            let dt_tid = frame.dt_track_ids[di];
            current_matches.insert(gt_tid, dt_tid);

            let iou = if di < frame.iou_matrix.len() && gi < frame.iou_matrix[di].len() {
                frame.iou_matrix[di][gi]
            } else {
                0.0
            };
            iou_sum += iou;

            // IDSW: only count once per unique GT track ID per frame
            if seen_gt_for_idsw.insert(gt_tid) {
                if let Some(&prev_dt) = prev_tracker_id.get(&gt_tid) {
                    if prev_dt != dt_tid {
                        num_id_switches += 1;
                    }
                }
            }
            prev_tracker_id.insert(gt_tid, dt_tid);
        }

        clr_tp += match_result.matches.len() as u64;
        clr_fn += match_result.unmatched_gt.len() as u64;
        clr_fp += match_result.unmatched_dt.len() as u64;

        // Frag counting (TrackEval convention):
        // 1. Read not_previously_tracked from prev_timestep (before reset)
        // 2. Reset prev_timestep for all GTs
        // 3. Set matched GTs
        // 4. Count not_previously_tracked AND currently_tracked transitions

        // Deduplicate GT track IDs (TrackEval uses per-ID boolean arrays)
        let mut seen_gt_ids: HashSet<u64> = HashSet::new();
        for &gt_tid in &frame.gt_track_ids {
            let s = gt_stats.entry(gt_tid).or_insert(TrackState {
                frames_present: 0,
                frames_tracked: 0,
                was_tracked: false,
                transitions_into_tracked: 0,
            });
            s.frames_present += 1;
            if current_matches.contains_key(&gt_tid) {
                s.frames_tracked += 1;
            }
            // Count frag transitions once per unique GT per frame
            if seen_gt_ids.insert(gt_tid) {
                let not_previously_tracked = !s.was_tracked;
                let is_tracked = current_matches.contains_key(&gt_tid);
                if is_tracked && not_previously_tracked {
                    s.transitions_into_tracked += 1;
                }
            }
        }

        // Reset prev_timestep state, then set only matched pairs
        // (TrackEval: prev_timestep_tracker_id[:] = np.nan, then set matched)
        prev_timestep.clear();
        for (&gt_tid, &dt_tid) in &current_matches {
            prev_timestep.insert(gt_tid, dt_tid);
        }

        // Update was_tracked for frag state
        for s in gt_stats.values_mut() {
            s.was_tracked = false;
        }
        for &gt_tid in current_matches.keys() {
            if let Some(s) = gt_stats.get_mut(&gt_tid) {
                s.was_tracked = true;
            }
        }
    }

    // Derive MT/PT/ML and total fragmentations.
    let mut mt: u64 = 0;
    let mut pt: u64 = 0;
    let mut ml: u64 = 0;
    let mut frag: u64 = 0;

    for s in gt_stats.values() {
        frag += s.transitions_into_tracked.saturating_sub(1);
        if s.frames_present > 0 {
            let ratio = s.frames_tracked as f64 / s.frames_present as f64;
            if ratio > 0.8 {
                mt += 1;
            } else if ratio < 0.2 {
                ml += 1;
            } else {
                pt += 1;
            }
        }
    }

    let num_gt_total = clr_tp + clr_fn;
    let mota = if num_gt_total > 0 {
        1.0 - (clr_fn + clr_fp + num_id_switches) as f64 / num_gt_total as f64
    } else {
        0.0
    };

    let motp = if clr_tp > 0 {
        iou_sum / clr_tp as f64
    } else {
        0.0
    };

    ClearResult {
        mota,
        motp,
        num_id_switches,
        clr_tp,
        clr_fn,
        clr_fp,
        mt,
        pt,
        ml,
        frag,
        num_gt_ids: all_gt_ids.len() as u64,
        num_dt_ids: all_dt_ids.len() as u64,
        iou_sum,
    }
}
