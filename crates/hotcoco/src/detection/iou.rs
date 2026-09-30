use std::collections::HashMap;

use rayon::prelude::*;

use crate::coco::COCO;
use crate::params::Params;
use crate::primitives::sim::{self, SimKind};
use crate::types::Rle;

use super::{COCOeval, EvalMode};

/// Every annotation's mask that a both-non-empty `(img, cat)` cell will
/// actually read, converted to RLE once per `evaluate()`.
///
/// pycocotools converts every in-scope mask in `_prepare`, before the IoU
/// loop; converting inside the per-cell IoU computation instead put
/// `fr_polys` — the 5×-upsampled polygon rasterizer — at ~48% of all samples
/// in a val2017 segm profile. hotcoco's `_prepare` twin narrows that further:
/// [`COCOeval::compute_iou_static`] never reads a mask from a cell with only
/// GT or only DT, so this only converts the ids [`COCOeval::segm_cell_ann_ids`]
/// finds in a cell with both — a DETR-shaped result set puts most detections
/// in a cell with no matching GT. Rebuilt on each `evaluate()` call, exactly
/// like the `ious` cache, so it can never go stale relative to the datasets
/// it was drawn from.
///
/// A GT-less-cell DT id — excluded from this cache on purpose — still gets a
/// correct answer from `confusion_matrix`/`tide`'s cross-category matrices:
/// they fall back to converting it on the spot (see
/// [`gt_rle_or_convert`](Self::gt_rle_or_convert)) once per image per call,
/// the same per-id cost `_prepare` used to pay upfront for every id in scope.
/// That fallback isn't itself cached, so calling `confusion_matrix()`/`tide()`
/// more than once repeats it each time — a trade `_prepare`'s original
/// "re-paid it in every … build" comment was written to avoid, now accepted
/// for the ids this cache no longer holds.
///
/// Annotations without a convertible mask (no segmentation *and* no bbox) are
/// absent; readers fall back to [`COCO::ann_to_rle`].
pub(super) struct SegmRles {
    // Private on purpose, mirroring the `ious` cache discipline ("the
    // visibility is the enforcement"): consumers get one RLE at a time via the
    // `*_rle_or_convert` helpers and can never iterate or retain the map.
    gt: HashMap<u64, Rle>,
    dt: HashMap<u64, Rle>,
}

impl SegmRles {
    /// The cache-miss policy, in one place: a hit clones the prepared RLE, a
    /// miss — or no cache at all — converts on the spot. Correctness never
    /// depends on what the cache happens to hold; a miss only costs speed.
    pub(super) fn gt_rle_or_convert(cache: Option<&Self>, coco_gt: &COCO, id: u64) -> Option<Rle> {
        if let Some(rle) = cache.and_then(|c| c.gt.get(&id)) {
            return Some(rle.clone());
        }
        coco_gt.ann_to_rle(coco_gt.get_ann(id)?)
    }

    /// Detection twin of [`gt_rle_or_convert`](Self::gt_rle_or_convert).
    pub(super) fn dt_rle_or_convert(cache: Option<&Self>, coco_dt: &COCO, id: u64) -> Option<Rle> {
        if let Some(rle) = cache.and_then(|c| c.dt.get(&id)) {
            return Some(rle.clone());
        }
        coco_dt.ann_to_rle(coco_dt.get_ann(id)?)
    }

    /// Convert exactly the annotations that [`COCOeval::compute_segm_iou_static`]
    /// will actually read: the GT/DT ids living in `(img, cat)` cells where
    /// *both* sides are non-empty.
    ///
    /// [`COCOeval::compute_iou_static`] returns early — no RLE ever read —
    /// for a cell with only GT or only DT, so rasterizing that cell's masks
    /// here bought nothing; `gt_ids`/`dt_ids` come from
    /// [`COCOeval::segm_cell_ann_ids`], the one place that walks the
    /// `(img, cat)` index to find the both-non-empty cells. A caller outside
    /// that shape (`confusion_matrix`, `tide`, a cross-category read) still
    /// gets a correct answer on a cache miss — see
    /// [`gt_rle_or_convert`](Self::gt_rle_or_convert) — just paid for on the
    /// spot instead of upfront.
    pub(super) fn prepare(coco_gt: &COCO, coco_dt: &COCO, gt_ids: &[u64], dt_ids: &[u64]) -> Self {
        let convert = |coco: &COCO, ids: &[u64]| -> HashMap<u64, Rle> {
            ids.par_iter()
                .filter_map(|&id| Some((id, coco.ann_to_rle(coco.get_ann(id)?)?)))
                .collect()
        };

        SegmRles {
            gt: convert(coco_gt, gt_ids),
            dt: convert(coco_dt, dt_ids),
        }
    }
}

/// Whether this ground truth's similarity column uses intersection-over-area.
///
/// IoA — intersection divided by the *detection's* area — is how both COCO and
/// Open Images express "this ground truth is a region, not an instance". They just
/// flag it differently: COCO with `iscrowd`, Open Images with `is_group_of`. The
/// protocol wording is "a detection is inside a group-of box if the area of
/// intersection of the detection and the box divided by the area of the detection
/// is greater than 0.5".
///
/// One function rather than the same branch in each of the three similarity
/// kernels: hardcoding `false` for Open Images made the group-of pass unreachable
/// for any detection smaller than the group box — the normal case — and it was
/// wrong in all three places at once, because there were three places.
fn uses_ioa(ann: &crate::types::Annotation, eval_mode: EvalMode) -> bool {
    if eval_mode == EvalMode::OpenImages {
        ann.is_group_of.unwrap_or(false)
    } else {
        ann.iscrowd
    }
}

/// Scatter a kernel's valid-only matrix back to the full `d × g` shape, with
/// zero rows/columns for annotations that contributed no geometry.
///
/// `dt_rows[i]` / `gt_cols[j]` are the original positions (index into the raw id
/// slice) of the kernel's row `i` / column `j`. This is what keeps every matrix
/// builder aligned with `matching::gather_pair`, which assigns `iou_indices` by
/// enumerating the same raw id slices: a bbox-less annotation between two valid
/// ones must occupy a zero row/column, not vanish and shift every later
/// annotation onto its neighbor's IoU row. A zero never matches — the match
/// floors are strictly positive — so a geometry-less annotation scores as
/// unmatched rather than borrowing someone else's overlap.
fn scatter_full(
    valid: Vec<Vec<f64>>,
    dt_rows: &[usize],
    gt_cols: &[usize],
    d: usize,
    g: usize,
) -> Vec<Vec<f64>> {
    if dt_rows.len() == d && gt_cols.len() == g {
        // Every annotation had geometry: the index lists are strictly increasing
        // subsequences of 0..d / 0..g, so full length means identity.
        return valid;
    }
    let mut full = vec![vec![0.0_f64; g]; d];
    for (vi, &di) in dt_rows.iter().enumerate() {
        for (vj, &gj) in gt_cols.iter().enumerate() {
            full[di][gj] = valid[vi][vj];
        }
    }
    full
}

/// Shared scaffold behind `compute_{bbox,segm,obb}_iou_static`.
///
/// Each of those three does the same four things: pull a detection's geometry
/// by id (dropping the ones with none, remembering which rows survived),
/// pull a ground truth's annotation *and* geometry by id (same drop-and-remember,
/// plus [`uses_ioa`] since that's a property of the annotation, not the
/// geometry), hand the two dense geometry lists to a `sim::*_iou` kernel, and
/// [`scatter_full`] the kernel's valid-only output back to `d × g`. Only the
/// geometry type and the extraction closures differ per family, so those are
/// the type parameters and the arguments; the marshaling itself lives here once.
fn iou_scaffold<D, G>(
    coco_gt: &COCO,
    dt_ids: &[u64],
    gt_ids: &[u64],
    eval_mode: EvalMode,
    dt_geom: impl Fn(u64) -> Option<D>,
    gt_geom: impl Fn(&crate::types::Annotation, u64) -> Option<G>,
    kernel: impl FnOnce(&[D], &[G], &[bool]) -> Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let (dt_rows, dt_geoms): (Vec<usize>, Vec<D>) = dt_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, &id)| Some((idx, dt_geom(id)?)))
        .unzip();
    let mut gt_cols = Vec::with_capacity(gt_ids.len());
    let mut gt_geoms = Vec::with_capacity(gt_ids.len());
    let mut iscrowd = Vec::with_capacity(gt_ids.len());
    for (idx, &id) in gt_ids.iter().enumerate() {
        let Some(ann) = coco_gt.get_ann(id) else {
            continue;
        };
        let Some(geom) = gt_geom(ann, id) else {
            continue;
        };
        gt_cols.push(idx);
        gt_geoms.push(geom);
        iscrowd.push(uses_ioa(ann, eval_mode));
    }

    let valid = kernel(&dt_geoms, &gt_geoms, &iscrowd);
    scatter_full(valid, &dt_rows, &gt_cols, dt_ids.len(), gt_ids.len())
}

impl COCOeval {
    /// Compute the IoU/OKS matrix for a given image and category.
    ///
    /// **Shape contract:** the result is either empty (no ids on one side) or
    /// exactly `dt_ids.len() × gt_ids.len()`, with row `i` / column `j`
    /// corresponding to the `i`-th detection id / `j`-th ground-truth id —
    /// including annotations whose geometry is missing, which occupy all-zero
    /// rows/columns via [`scatter_full`]. `matching::gather_pair` indexes this
    /// matrix by position in the same id slices, so a builder that dropped a
    /// geometry-less annotation would silently shift every later annotation
    /// onto its neighbor's IoU row.
    pub(super) fn compute_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        img_id: u64,
        cat_id: u64,
        eval_mode: EvalMode,
        segm_rles: Option<&SegmRles>,
    ) -> Vec<Vec<f64>> {
        let gt_anns = Self::get_anns_static(coco_gt, params, img_id, cat_id);
        let dt_anns = Self::get_anns_static(coco_dt, params, img_id, cat_id);

        if gt_anns.is_empty() || dt_anns.is_empty() {
            return Vec::new();
        }

        // Dispatch on the geometry axis, not the eval-config one: `SimKind` is
        // what selects a kernel, and every family (detection here, tracking and
        // panoptic later) branches on the same four values. The helpers below do
        // marshaling only — reshaping annotations into the kernel's input types.
        match SimKind::from(params.iou_type) {
            SimKind::Mask => Self::compute_segm_iou_static(
                coco_gt, coco_dt, dt_anns, gt_anns, eval_mode, segm_rles,
            ),
            SimKind::Bbox => {
                Self::compute_bbox_iou_static(coco_gt, coco_dt, dt_anns, gt_anns, eval_mode)
            }
            SimKind::Oks => Self::compute_oks_static(coco_gt, coco_dt, params, dt_anns, gt_anns),
            SimKind::Obb => {
                Self::compute_obb_iou_static(coco_gt, coco_dt, dt_anns, gt_anns, eval_mode)
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

    /// Compute segmentation mask IoU by converting annotations to RLE and calling `sim::mask_iou`.
    ///
    /// RLEs come through [`SegmRles::dt_rle_or_convert`]/[`SegmRles::gt_rle_or_convert`],
    /// which own the cache-or-convert policy.
    pub(super) fn compute_segm_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
        segm_rles: Option<&SegmRles>,
    ) -> Vec<Vec<f64>> {
        iou_scaffold(
            coco_gt,
            dt_ids,
            gt_ids,
            eval_mode,
            |id| SegmRles::dt_rle_or_convert(segm_rles, coco_dt, id),
            |_ann, id| SegmRles::gt_rle_or_convert(segm_rles, coco_gt, id),
            sim::mask_iou,
        )
    }

    /// Compute bounding box IoU by extracting bbox arrays and calling `sim::bbox_iou`.
    pub(super) fn compute_bbox_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
    ) -> Vec<Vec<f64>> {
        iou_scaffold(
            coco_gt,
            dt_ids,
            gt_ids,
            eval_mode,
            |id| coco_dt.get_ann(id)?.bbox,
            |ann, _id| ann.bbox,
            sim::bbox_iou,
        )
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
        // The OKS math lives in the shared `primitives::sim::oks_matrix` kernel
        // (COCO-decoupled, matrix-shaped). Here we only marshal the annotations
        // into the kernel's flat-slice form. A missing `keypoints` field maps to
        // an empty slice, which the kernel skips, leaving that row/column zero.
        // Only an id with no annotation record at all goes through
        // `scatter_full`'s zero-fill.
        let (gt_cols, gt_anns): (Vec<usize>, Vec<_>) = gt_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &id)| Some((idx, coco_gt.get_ann(id)?)))
            .unzip();
        let (dt_rows, dt_anns): (Vec<usize>, Vec<_>) = dt_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &id)| Some((idx, coco_dt.get_ann(id)?)))
            .unzip();

        let gt: Vec<crate::primitives::sim::GtPose<'_>> = gt_anns
            .iter()
            .map(|a| crate::primitives::sim::GtPose {
                keypoints: a.keypoints.as_deref().unwrap_or(&[]),
                area: a.area.unwrap_or(0.0),
                bbox: a.bbox.unwrap_or([0.0; 4]),
            })
            .collect();
        let dt_keypoints: Vec<&[f64]> = dt_anns
            .iter()
            .map(|a| a.keypoints.as_deref().unwrap_or(&[]))
            .collect();

        let valid = crate::primitives::sim::oks_matrix(&dt_keypoints, &gt, &params.kpt_oks_sigmas);
        scatter_full(valid, &dt_rows, &gt_cols, dt_ids.len(), gt_ids.len())
    }

    /// Compute oriented bounding box IoU by extracting OBB arrays and calling `sim::obb_iou`.
    pub(super) fn compute_obb_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
    ) -> Vec<Vec<f64>> {
        iou_scaffold(
            coco_gt,
            dt_ids,
            gt_ids,
            eval_mode,
            |id| coco_dt.get_ann(id)?.obb.as_deref().copied(),
            |ann, _id| ann.obb.as_deref().copied(),
            sim::obb_iou,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::coco::COCO;
    use crate::params::IouType;
    use crate::types::{Annotation, Category, Dataset, Image, Segmentation};

    use super::COCOeval;

    /// A whole-image, all-background RLE — pixel content is irrelevant here,
    /// only that `ann_to_rle` succeeds.
    fn blank_rle(h: u32, w: u32) -> Segmentation {
        Segmentation::UncompressedRle {
            size: [h, w],
            counts: vec![h * w],
        }
    }

    fn segm_ann(id: u64, category_id: u64, score: Option<f64>) -> Annotation {
        Annotation {
            id,
            image_id: 1,
            category_id,
            bbox: Some([0.0, 0.0, 4.0, 4.0]),
            area: Some(16.0),
            segmentation: Some(blank_rle(10, 10)),
            score,
            ..Default::default()
        }
    }

    /// Regression for candidate F: the segm RLE cache must hold ids from
    /// `(img, cat)` cells with both a GT and a DT, and skip ids from a
    /// DT-only cell — `compute_iou_static` never reads that cell's mask, so
    /// converting it was pure waste. Reverting `SegmRles::prepare` to convert
    /// every in-scope id regardless of pairing makes `cat 2`'s DT id 102
    /// reappear in the cache, and this test catches it.
    #[test]
    fn segm_rle_cache_skips_dt_only_cells() {
        let img = Image {
            id: 1,
            height: 10,
            width: 10,
            ..Default::default()
        };
        let gt = Dataset {
            images: vec![img.clone()],
            categories: vec![
                Category {
                    id: 1,
                    name: "matched".into(),
                    ..Default::default()
                },
                Category {
                    id: 2,
                    name: "dt_only".into(),
                    ..Default::default()
                },
            ],
            annotations: vec![segm_ann(1, 1, None)],
            ..Default::default()
        };
        let dt = Dataset {
            images: vec![img],
            categories: gt.categories.clone(),
            annotations: vec![
                segm_ann(101, 1, Some(0.9)), // cat 1: GT and DT both present
                segm_ann(102, 2, Some(0.8)), // cat 2: DT only, no GT
            ],
            ..Default::default()
        };

        let mut ev = COCOeval::new(
            COCO::from_dataset(gt),
            COCO::from_dataset(dt),
            IouType::Segm,
        );
        ev.evaluate();

        let cache = ev.segm_rles.expect("segm run always builds the RLE cache");
        assert!(
            cache.gt.contains_key(&1),
            "GT id in a both-non-empty cell must be cached"
        );
        assert!(
            cache.dt.contains_key(&101),
            "DT id in a both-non-empty cell must be cached"
        );
        assert!(
            !cache.dt.contains_key(&102),
            "DT id in a DT-only cell (cat 2 has no GT) must not be cached — \
             compute_iou_static never reads it"
        );
    }
}
