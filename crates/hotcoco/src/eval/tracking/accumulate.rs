use super::types::{ClearResult, CombinedResults, HotaResult, IdentityResult, SeqResult};

/// Aggregate per-sequence tracking results into combined dataset-level metrics.
///
/// Integer counts are summed across sequences, then derived metrics (ratios)
/// are recomputed from the summed counts. For HOTA association metrics,
/// association sums are weighted by per-sequence TP count.
pub(crate) fn accumulate(seq_results: &[SeqResult]) -> CombinedResults {
    let hota = accumulate_hota(seq_results);
    let clear = accumulate_clear(seq_results);
    let identity = accumulate_identity(seq_results);

    CombinedResults {
        hota,
        clear,
        identity,
    }
}

fn accumulate_hota(seq_results: &[SeqResult]) -> Option<HotaResult> {
    let hota_results: Vec<&HotaResult> =
        seq_results.iter().filter_map(|s| s.hota.as_ref()).collect();

    if hota_results.is_empty() {
        return None;
    }

    let num_thrs = hota_results[0].hota.len();
    let mut tp = vec![0u64; num_thrs];
    let mut r#fn = vec![0u64; num_thrs];
    let mut fp = vec![0u64; num_thrs];
    let mut ass_sum_total = vec![0.0; num_thrs];
    let mut loc_sum_total = vec![0.0; num_thrs];

    for h in &hota_results {
        for t in 0..num_thrs {
            tp[t] += h.hota_tp[t];
            r#fn[t] += h.hota_fn[t];
            fp[t] += h.hota_fp[t];
            ass_sum_total[t] += h.ass_sum[t];
            loc_sum_total[t] += h.loc_sum[t];
        }
    }

    let mut hota = vec![0.0; num_thrs];
    let mut det_a = vec![0.0; num_thrs];
    let mut ass_a = vec![0.0; num_thrs];
    let mut loc_a = vec![1.0; num_thrs]; // TrackEval convention: 1.0 when no TPs
    let mut det_re = vec![0.0; num_thrs];
    let mut det_pr = vec![0.0; num_thrs];
    let mut ass_re = vec![0.0; num_thrs];
    let mut ass_pr = vec![0.0; num_thrs];

    for t in 0..num_thrs {
        let total = tp[t] + r#fn[t] + fp[t];
        if total > 0 {
            det_a[t] = tp[t] as f64 / total as f64;
            let gt_total = tp[t] + r#fn[t];
            let dt_total = tp[t] + fp[t];
            det_re[t] = if gt_total > 0 {
                tp[t] as f64 / gt_total as f64
            } else {
                0.0
            };
            det_pr[t] = if dt_total > 0 {
                tp[t] as f64 / dt_total as f64
            } else {
                0.0
            };
        }

        if tp[t] > 0 {
            ass_a[t] = ass_sum_total[t] / tp[t] as f64;
            loc_a[t] = loc_sum_total[t] / tp[t] as f64;
            hota[t] = (det_a[t] * ass_a[t]).sqrt();

            // AssRe and AssPr: recompute from summed per-sequence values.
            // For simplicity, use the same weighting as AssA.
            let mut re_sum = 0.0;
            let mut pr_sum = 0.0;
            for h in &hota_results {
                re_sum += h.ass_re[t] * h.hota_tp[t] as f64;
                pr_sum += h.ass_pr[t] * h.hota_tp[t] as f64;
            }
            ass_re[t] = re_sum / tp[t] as f64;
            ass_pr[t] = pr_sum / tp[t] as f64;
        }
    }

    Some(HotaResult {
        hota,
        det_a,
        ass_a,
        loc_a,
        det_re,
        det_pr,
        ass_re,
        ass_pr,
        hota_tp: tp,
        hota_fn: r#fn,
        hota_fp: fp,
        ass_sum: ass_sum_total,
        loc_sum: loc_sum_total,
    })
}

fn accumulate_clear(seq_results: &[SeqResult]) -> Option<ClearResult> {
    let clear_results: Vec<&ClearResult> = seq_results
        .iter()
        .filter_map(|s| s.clear.as_ref())
        .collect();

    if clear_results.is_empty() {
        return None;
    }

    let mut clr_tp: u64 = 0;
    let mut clr_fn: u64 = 0;
    let mut clr_fp: u64 = 0;
    let mut num_id_switches: u64 = 0;
    let mut iou_sum: f64 = 0.0;
    let mut mt: u64 = 0;
    let mut pt: u64 = 0;
    let mut ml: u64 = 0;
    let mut frag: u64 = 0;
    let mut num_gt_ids: u64 = 0;
    let mut num_dt_ids: u64 = 0;

    for c in &clear_results {
        clr_tp += c.clr_tp;
        clr_fn += c.clr_fn;
        clr_fp += c.clr_fp;
        num_id_switches += c.num_id_switches;
        iou_sum += c.iou_sum;
        mt += c.mt;
        pt += c.pt;
        ml += c.ml;
        frag += c.frag;
        num_gt_ids += c.num_gt_ids;
        num_dt_ids += c.num_dt_ids;
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

    Some(ClearResult {
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
        num_gt_ids,
        num_dt_ids,
        iou_sum,
    })
}

fn accumulate_identity(seq_results: &[SeqResult]) -> Option<IdentityResult> {
    let id_results: Vec<&IdentityResult> = seq_results
        .iter()
        .filter_map(|s| s.identity.as_ref())
        .collect();

    if id_results.is_empty() {
        return None;
    }

    let mut idtp: u64 = 0;
    let mut idfn: u64 = 0;
    let mut idfp: u64 = 0;

    for r in &id_results {
        idtp += r.idtp;
        idfn += r.idfn;
        idfp += r.idfp;
    }

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

    Some(IdentityResult {
        idf1,
        idp,
        idr,
        idtp,
        idfn,
        idfp,
    })
}
