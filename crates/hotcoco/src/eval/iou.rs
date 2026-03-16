use crate::coco::COCO;
use crate::mask;
use crate::params::{IouType, Params};
use crate::types::Rle;

use super::{COCOeval, EvalMode};

impl COCOeval {
    /// Compute the IoU/OKS matrix for a given image and category.
    pub(super) fn compute_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        img_id: u64,
        cat_id: u64,
        eval_mode: EvalMode,
    ) -> Vec<Vec<f64>> {
        let gt_anns = Self::get_anns_static(coco_gt, params, img_id, cat_id);
        let dt_anns = Self::get_anns_static(coco_dt, params, img_id, cat_id);

        if gt_anns.is_empty() || dt_anns.is_empty() {
            return Vec::new();
        }

        match params.iou_type {
            IouType::Segm => {
                Self::compute_segm_iou_static(coco_gt, coco_dt, dt_anns, gt_anns, eval_mode)
            }
            IouType::Bbox => {
                Self::compute_bbox_iou_static(coco_gt, coco_dt, dt_anns, gt_anns, eval_mode)
            }
            IouType::Keypoints => {
                Self::compute_oks_static(coco_gt, coco_dt, params, dt_anns, gt_anns)
            }
        }
    }

    /// Get annotation IDs for an image, optionally filtered by category.
    pub(super) fn get_anns_static<'a>(
        coco: &'a COCO,
        params: &Params,
        img_id: u64,
        cat_id: u64,
    ) -> &'a [u64] {
        if params.use_cats {
            coco.get_ann_ids_for_img_cat(img_id, cat_id)
        } else {
            coco.get_ann_ids_for_img(img_id)
        }
    }

    /// Compute segmentation mask IoU by converting annotations to RLE and calling `mask::iou`.
    pub(super) fn compute_segm_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
    ) -> Vec<Vec<f64>> {
        let dt_rles: Vec<Rle> = dt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_dt.get_ann(id)?;
                coco_dt.ann_to_rle(ann)
            })
            .collect();
        let (gt_rles, iscrowd): (Vec<Rle>, Vec<bool>) = gt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_gt.get_ann(id)?;
                // OID: iscrowd is irrelevant — always use standard IoU
                let crowd = if eval_mode == EvalMode::OpenImages {
                    false
                } else {
                    ann.iscrowd
                };
                Some((coco_gt.ann_to_rle(ann)?, crowd))
            })
            .unzip();

        mask::iou(&dt_rles, &gt_rles, &iscrowd)
    }

    /// Compute bounding box IoU by extracting bbox arrays and calling `mask::bbox_iou`.
    pub(super) fn compute_bbox_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
    ) -> Vec<Vec<f64>> {
        let dt_bbs: Vec<[f64; 4]> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id)?.bbox)
            .collect();
        let (gt_bbs, iscrowd): (Vec<[f64; 4]>, Vec<bool>) = gt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_gt.get_ann(id)?;
                // OID: iscrowd is irrelevant — always use standard IoU
                let crowd = if eval_mode == EvalMode::OpenImages {
                    false
                } else {
                    ann.iscrowd
                };
                Some((ann.bbox?, crowd))
            })
            .unzip();

        mask::bbox_iou(&dt_bbs, &gt_bbs, &iscrowd)
    }

    /// Compute OKS (Object Keypoint Similarity) between detection and GT keypoints.
    ///
    /// OKS = mean_k[ exp( -d_k^2 / (2 * s_k^2 * area) ) ] where d_k is the Euclidean
    /// distance for keypoint k, s_k is the per-keypoint sigma, and area is the GT area.
    /// Only visible GT keypoints contribute. When no GT keypoints are visible, distance
    /// is measured to the GT bounding box boundary instead.
    pub(super) fn compute_oks_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        dt_ids: &[u64],
        gt_ids: &[u64],
    ) -> Vec<Vec<f64>> {
        let sigmas = &params.kpt_oks_sigmas;
        let num_kpts = sigmas.len();
        // vars = (sigmas * 2)**2 = 4 * sigma^2  (matching pycocotools)
        let vars: Vec<f64> = sigmas.iter().map(|s| (2.0 * s).powi(2)).collect();

        let d = dt_ids.len();
        let g = gt_ids.len();
        let mut result = vec![vec![0.0f64; g]; d];

        let gt_anns: Vec<_> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id))
            .collect();
        let dt_anns: Vec<_> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id))
            .collect();

        for (j, gt_ann) in gt_anns.iter().enumerate() {
            let gt_kpts = match &gt_ann.keypoints {
                Some(k) => k,
                None => continue,
            };
            let gt_area = gt_ann.area.unwrap_or(0.0) + f64::EPSILON;
            let gt_bbox = gt_ann.bbox.unwrap_or([0.0; 4]);

            // Count visible GT keypoints
            let k1: usize = (0..num_kpts)
                .filter(|&ki| gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0)
                .count();

            // Compute ignore region bounds (double the GT bbox)
            let x0 = gt_bbox[0] - gt_bbox[2];
            let x1 = gt_bbox[0] + gt_bbox[2] * 2.0;
            let y0 = gt_bbox[1] - gt_bbox[3];
            let y1 = gt_bbox[1] + gt_bbox[3] * 2.0;

            for (i, dt_ann) in dt_anns.iter().enumerate() {
                let dt_kpts = match &dt_ann.keypoints {
                    Some(k) => k,
                    None => continue,
                };

                // Compute OKS in a single pass — sum exp(-e) over visible keypoints
                // (or all keypoints when k1 == 0) without intermediate allocations.
                let mut oks_sum = 0.0_f64;
                let mut oks_count = 0_usize;

                for (ki, &var_k) in vars.iter().enumerate().take(num_kpts) {
                    // When k1 > 0, only include visible GT keypoints
                    let visible = gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0;
                    if k1 > 0 && !visible {
                        continue;
                    }

                    let gx = gt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                    let gy = gt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);
                    let xd = dt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                    let yd = dt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);

                    let (dx, dy) = if k1 > 0 {
                        (xd - gx, yd - gy)
                    } else {
                        // No visible GT keypoints: measure distance to bbox boundary
                        let dx = 0.0_f64.max(x0 - xd) + 0.0_f64.max(xd - x1);
                        let dy = 0.0_f64.max(y0 - yd) + 0.0_f64.max(yd - y1);
                        (dx, dy)
                    };

                    let e = (dx * dx + dy * dy) / var_k / gt_area / 2.0;
                    oks_sum += (-e).exp();
                    oks_count += 1;
                }

                if oks_count > 0 {
                    result[i][j] = oks_sum / oks_count as f64;
                }
            }
        }

        result
    }
}
