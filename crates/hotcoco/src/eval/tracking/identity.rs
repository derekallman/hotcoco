use super::types::{FrameData, IdentityResult, TrackIdMaps};

/// Compute Identity metrics (IDF1, IDP, IDR) for a single sequence.
///
/// Follows TrackEval's algorithm exactly:
/// 1. For each frame, find all (GT, DT) pairs with IoU >= threshold (no per-frame
///    Hungarian — just a mask of above-threshold pairs)
/// 2. Accumulate `potential_matches_count[gt_id][dt_id]` across frames
/// 3. Build augmented cost matrix `fn_mat + fp_mat` and solve one global Hungarian
/// 4. Derive IDTP, IDFN, IDFP from the optimal assignment
pub(crate) fn compute_identity(frames: &[FrameData], similarity_thr: f64) -> IdentityResult {
    let id_maps = TrackIdMaps::build(frames);
    let num_gt_ids = id_maps.gt_ids.len();
    let num_dt_ids = id_maps.dt_ids.len();

    // Count total detections per GT/DT track and accumulate potential matches.
    let mut gt_id_count = vec![0.0f64; num_gt_ids];
    let mut dt_id_count = vec![0.0f64; num_dt_ids];
    let mut potential_matches_count = vec![vec![0.0f64; num_dt_ids]; num_gt_ids];

    // Use raw counts for early-exit edge cases, but deduplicated counts
    // (gt_id_count.sum()) for IDTP computation — matching TrackEval.
    let raw_gt_dets: u64 = frames.iter().map(|f| f.gt_track_ids.len() as u64).sum();
    let raw_dt_dets: u64 = frames.iter().map(|f| f.dt_track_ids.len() as u64).sum();

    if raw_gt_dets == 0 {
        return IdentityResult {
            idf1: 0.0,
            idp: 0.0,
            idr: 0.0,
            idtp: 0,
            idfn: 0,
            idfp: raw_dt_dets,
        };
    }
    if raw_dt_dets == 0 {
        return IdentityResult {
            idf1: 0.0,
            idp: 0.0,
            idr: 0.0,
            idtp: 0,
            idfn: raw_gt_dets,
            idfp: 0,
        };
    }

    for frame in frames {
        let gt_global: Vec<usize> = frame
            .gt_track_ids
            .iter()
            .map(|tid| id_maps.gt_map[tid])
            .collect();
        let dt_global: Vec<usize> = frame
            .dt_track_ids
            .iter()
            .map(|tid| id_maps.dt_map[tid])
            .collect();

        // Count appearances per track ID (deduplicate per frame — NumPy's
        // fancy indexing with duplicate indices only increments once).
        {
            let mut seen = std::collections::HashSet::new();
            for &gi in &gt_global {
                if seen.insert(gi) {
                    gt_id_count[gi] += 1.0;
                }
            }
            seen.clear();
            for &di in &dt_global {
                if seen.insert(di) {
                    dt_id_count[di] += 1.0;
                }
            }
        }

        // Find all (gt, dt) pairs with IoU >= threshold (no Hungarian — just a mask).
        for (local_gi, &gi) in gt_global.iter().enumerate() {
            for (local_di, &di_global) in dt_global.iter().enumerate() {
                if frame.iou_matrix.len() > local_di
                    && frame.iou_matrix[local_di].len() > local_gi
                    && frame.iou_matrix[local_di][local_gi] >= similarity_thr
                {
                    potential_matches_count[gi][di_global] += 1.0;
                }
            }
        }
    }

    // Build augmented cost matrix following TrackEval:
    // Size: (num_gt_ids + num_dt_ids) x (num_gt_ids + num_dt_ids)
    // Upper-left (NxM): fn_count + fp_count for each (gt, dt) pair
    // Upper-right (NxN): fn_count for gt matched to "no tracker" (diagonal)
    // Lower-left (MxM): fp_count for dt matched to "no gt" (diagonal)
    // Lower-right: zero
    let n = num_gt_ids + num_dt_ids;
    let mut fn_mat = vec![vec![0.0f64; n]; n];
    let mut fp_mat = vec![vec![0.0f64; n]; n];

    // Large cost to prevent matching gt→unrelated-dt-padding or vice versa
    for row in &mut fp_mat[num_gt_ids..n] {
        for cell in &mut row[..num_dt_ids] {
            *cell = 1e10;
        }
    }
    for row in &mut fn_mat[..num_gt_ids] {
        for cell in &mut row[num_dt_ids..n] {
            *cell = 1e10;
        }
    }

    // Fill GT counts
    for (gt_id, gt_count) in gt_id_count.iter().enumerate() {
        for cell in &mut fn_mat[gt_id][..num_dt_ids] {
            *cell = *gt_count;
        }
        fn_mat[gt_id][num_dt_ids + gt_id] = *gt_count;
    }

    // Fill DT counts
    for (dt_id, dt_count) in dt_id_count.iter().enumerate() {
        for row in &mut fp_mat[..num_gt_ids] {
            row[dt_id] = *dt_count;
        }
        fp_mat[dt_id + num_gt_ids][dt_id] = *dt_count;
    }

    // Subtract potential matches from the upper-left block
    for (gt_id, pm_row) in potential_matches_count.iter().enumerate() {
        for (dt_id, &pm) in pm_row.iter().enumerate() {
            fn_mat[gt_id][dt_id] -= pm;
            fp_mat[gt_id][dt_id] -= pm;
        }
    }

    // Build combined cost matrix and solve with LAPJV
    let mut cost = ndarray::Array2::from_elem((n, n), 0.0);
    for i in 0..n {
        for j in 0..n {
            cost[[i, j]] = fn_mat[i][j] + fp_mat[i][j];
        }
    }

    let (idfn, idfp) = match lapjv::lapjv(&cost) {
        Ok((row_to_col, _)) => {
            let mut fn_sum = 0.0f64;
            let mut fp_sum = 0.0f64;
            for (i, &j) in row_to_col.iter().enumerate() {
                fn_sum += fn_mat[i][j];
                fp_sum += fp_mat[i][j];
            }
            (fn_sum.round() as u64, fp_sum.round() as u64)
        }
        Err(_) => (raw_gt_dets, raw_dt_dets),
    };

    // TrackEval: IDTP = gt_id_count.sum() - IDFN (deduplicated count, not raw dets)
    let gt_count_sum = gt_id_count.iter().sum::<f64>().round() as u64;
    let idtp = gt_count_sum.saturating_sub(idfn);

    let denom = (2 * idtp + idfn + idfp) as f64;
    let idf1 = if denom > 0.0 {
        (2 * idtp) as f64 / denom
    } else {
        0.0
    };
    let idp = if (idtp + idfp) > 0 {
        idtp as f64 / (idtp + idfp) as f64
    } else {
        0.0
    };
    let idr = if (idtp + idfn) > 0 {
        idtp as f64 / (idtp + idfn) as f64
    } else {
        0.0
    };

    IdentityResult {
        idf1,
        idp,
        idr,
        idtp,
        idfn,
        idfp,
    }
}
