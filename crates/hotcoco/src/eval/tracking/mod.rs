//! Multi-object tracking evaluation — HOTA, CLEAR/MOTA, Identity/IDF1.
//!
//! Provides `TrackingEval` for evaluating tracking results using TAO-style COCO JSON
//! with `video_id`, `track_id`, and `frame_index` fields.

mod accumulate;
mod clear;
mod hota;
mod identity;
pub(crate) mod matching;
pub(crate) mod types;

pub use types::{
    ClearResult, CombinedResults, HotaResult, IdentityResult, SeqResult, TrackingMetric,
    TrackingParams,
};

use std::collections::{BTreeMap, HashMap};

use rayon::prelude::*;

use crate::coco::COCO;
use crate::params::IouType;
use types::FrameData;

/// Multi-object tracking evaluator.
///
/// Follows the same `evaluate()` → `accumulate()` → `summarize()` pipeline as `COCOeval`,
/// but computes tracking-specific metrics (HOTA, MOTA, IDF1) instead of AP/AR.
///
/// # Example (Rust)
/// ```no_run
/// use std::path::Path;
/// use hotcoco::{COCO, IouType};
/// use hotcoco::eval::tracking::TrackingEval;
///
/// let coco_gt = COCO::new(Path::new("tracking_gt.json")).unwrap();
/// let coco_dt = coco_gt.load_res(Path::new("tracking_dt.json")).unwrap();
/// let mut ev = TrackingEval::new(coco_gt, coco_dt, IouType::Bbox);
/// ev.run();
/// let results = ev.get_results(None);
/// println!("HOTA: {:.4}", results["HOTA"]);
/// ```
pub struct TrackingEval {
    pub coco_gt: COCO,
    pub coco_dt: COCO,
    pub params: TrackingParams,
    seq_results: Vec<SeqResult>,
    combined: Option<CombinedResults>,
}

impl TrackingEval {
    /// Create a new tracking evaluator.
    pub fn new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        let params = TrackingParams {
            iou_type,
            ..TrackingParams::default()
        };
        Self {
            coco_gt,
            coco_dt,
            params,
            seq_results: Vec::new(),
            combined: None,
        }
    }

    /// Run per-sequence evaluation: matching + metric computation.
    ///
    /// Groups images by `video_id`, sorts by `frame_index`, then evaluates
    /// each sequence in parallel using rayon.
    pub fn evaluate(&mut self) {
        let sequences = self.build_sequences();

        let params = &self.params;
        self.seq_results = sequences
            .into_par_iter()
            .map(|(video_id, frames)| {
                let hota = if params.metrics.contains(&TrackingMetric::Hota) {
                    Some(hota::compute_hota(&frames, &params.iou_thrs))
                } else {
                    None
                };

                let clear = if params.metrics.contains(&TrackingMetric::Clear) {
                    Some(clear::compute_clear(&frames, params.similarity_thr))
                } else {
                    None
                };

                let identity = if params.metrics.contains(&TrackingMetric::Identity) {
                    Some(identity::compute_identity(&frames, params.similarity_thr))
                } else {
                    None
                };

                SeqResult {
                    video_id,
                    hota,
                    clear,
                    identity,
                }
            })
            .collect();
    }

    /// Aggregate per-sequence results into combined dataset-level metrics.
    pub fn accumulate(&mut self) {
        self.combined = Some(accumulate::accumulate(&self.seq_results));
    }

    /// Print a formatted summary of the tracking metrics.
    pub fn summarize(&self) {
        let Some(ref combined) = self.combined else {
            eprintln!("Warning: call accumulate() before summarize()");
            return;
        };

        println!();
        if let Some(ref h) = combined.hota {
            let n = h.hota.len() as f64;
            if n > 0.0 {
                println!(" HOTA  | DetA   | AssA   | LocA   | DetRe  | DetPr  | AssRe  | AssPr ");
                println!(
                    " {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4}",
                    h.hota.iter().sum::<f64>() / n,
                    h.det_a.iter().sum::<f64>() / n,
                    h.ass_a.iter().sum::<f64>() / n,
                    h.loc_a.iter().sum::<f64>() / n,
                    h.det_re.iter().sum::<f64>() / n,
                    h.det_pr.iter().sum::<f64>() / n,
                    h.ass_re.iter().sum::<f64>() / n,
                    h.ass_pr.iter().sum::<f64>() / n,
                );
            }
        }

        if let Some(ref c) = combined.clear {
            println!(" MOTA  | MOTP   | IDSW  | CLR_TP | CLR_FN | CLR_FP | MT  | PT  | ML  | Frag");
            println!(
                " {:.4} | {:.4} | {:5} | {:6} | {:6} | {:6} | {:3} | {:3} | {:3} | {:4}",
                c.mota,
                c.motp,
                c.num_id_switches,
                c.clr_tp,
                c.clr_fn,
                c.clr_fp,
                c.mt,
                c.pt,
                c.ml,
                c.frag,
            );
        }

        if let Some(ref id) = combined.identity {
            println!(" IDF1  | IDP    | IDR    | IDTP  | IDFN  | IDFP ");
            println!(
                " {:.4} | {:.4} | {:.4} | {:5} | {:5} | {:5}",
                id.idf1, id.idp, id.idr, id.idtp, id.idfn, id.idfp,
            );
        }
        println!();
    }

    /// Convenience: evaluate + accumulate + summarize.
    pub fn run(&mut self) {
        self.evaluate();
        self.accumulate();
        self.summarize();
    }

    /// Get results as a flat metric-name → value map.
    pub fn get_results(&self, prefix: Option<&str>) -> HashMap<String, f64> {
        self.combined
            .as_ref()
            .map(|c| c.to_map(prefix))
            .unwrap_or_default()
    }

    /// Access per-sequence results (available after `evaluate()`).
    pub fn seq_results(&self) -> &[SeqResult] {
        &self.seq_results
    }

    /// Access combined results (available after `accumulate()`).
    pub fn combined(&self) -> Option<&CombinedResults> {
        self.combined.as_ref()
    }

    /// Build per-sequence frame data from the GT and DT COCO datasets.
    ///
    /// Groups images by `video_id` (or treats all images as one sequence if no video_id),
    /// sorts frames by `frame_index` (or `id`), then collects GT/DT annotations per frame.
    fn build_sequences(&self) -> Vec<(u64, Vec<FrameData>)> {
        // Group GT images by video_id.
        let mut video_images: BTreeMap<u64, Vec<&crate::types::Image>> = BTreeMap::new();
        for img in &self.coco_gt.dataset.images {
            let vid = img.video_id.unwrap_or(0);
            video_images.entry(vid).or_default().push(img);
        }

        // Sort each video's images by frame_index, then by id.
        for images in video_images.values_mut() {
            images.sort_by_key(|img| (img.frame_index.unwrap_or(0), img.id));
        }

        // Build GT and DT annotation indices by image_id (reuses convert utility).
        let gt_by_img = crate::convert::anns_by_image(&self.coco_gt.dataset);
        let dt_by_img = crate::convert::anns_by_image(&self.coco_dt.dataset);

        let max_det = self.params.max_detections;

        video_images
            .into_iter()
            .map(|(video_id, images)| {
                let frames: Vec<FrameData> = images
                    .iter()
                    .map(|img| {
                        let gt_anns: Vec<_> = gt_by_img.get(&img.id).cloned().unwrap_or_default();
                        let mut dt_anns: Vec<_> =
                            dt_by_img.get(&img.id).cloned().unwrap_or_default();

                        // Sort DT by score descending and truncate to max_detections.
                        dt_anns.sort_by(|a, b| {
                            b.score
                                .unwrap_or(0.0)
                                .partial_cmp(&a.score.unwrap_or(0.0))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if let Some(max) = max_det {
                            dt_anns.truncate(max);
                        }

                        // Filter out crowd GTs (tracking ignores crowd annotations).
                        let gt_anns: Vec<_> = gt_anns.into_iter().filter(|a| !a.iscrowd).collect();

                        let gt_bboxes: Vec<[f64; 4]> =
                            gt_anns.iter().filter_map(|a| a.bbox).collect();
                        let dt_bboxes: Vec<[f64; 4]> =
                            dt_anns.iter().filter_map(|a| a.bbox).collect();

                        let gt_track_ids: Vec<u64> =
                            gt_anns.iter().map(|a| a.track_id.unwrap_or(a.id)).collect();
                        let dt_track_ids: Vec<u64> =
                            dt_anns.iter().map(|a| a.track_id.unwrap_or(a.id)).collect();

                        let iou_matrix = matching::compute_iou_matrix(&dt_bboxes, &gt_bboxes);

                        FrameData {
                            gt_track_ids,
                            dt_track_ids,
                            iou_matrix,
                        }
                    })
                    .collect();

                (video_id, frames)
            })
            .collect()
    }
}
