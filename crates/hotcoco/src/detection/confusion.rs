use std::collections::HashMap;

use rayon::prelude::*;

use crate::coco::COCO;
use crate::metrics::confusion;
use crate::params::IouType;
use crate::primitives;
use crate::primitives::sim::SimKind;

use super::COCOeval;

/// An annotation paired with the `cat_ids` slot its category occupies.
///
/// The category *index*, not the id: both callers use it to address a confusion
/// row or to test "is this ground truth a different class than that detection",
/// and re-deriving the index from the id at those sites would be a second lookup
/// of something the collection walk already knows.
pub(super) type CatPair = (usize, u64);

/// How [`COCOeval::cross_category_pairs`] should rank an image's detections.
#[derive(Debug, Clone, Copy)]
pub(super) struct DtRank {
    /// Discard detections scoring below this before the cap. `None` keeps all.
    pub(super) min_score: Option<f64>,
    /// Keep at most this many, highest-scoring first.
    pub(super) max_det: usize,
}

impl COCOeval {
    /// Compute a cross-category IoU matrix between DT and GT annotations.
    ///
    /// Returns a **flat** row-major `[D × G]` buffer (`iou[di * g + gi]`), empty
    /// when either side is empty. Flat is what both callers want — the confusion
    /// matrix hands it straight to `greedy_match_masked`, which takes flat, and TIDE
    /// scans one detection's row at a time — so producing it here saves each of
    /// them re-flattening a D·G buffer of their own.
    ///
    /// The length is exactly `d * g` regardless of what the underlying kernel
    /// returned, which is why callers need no shape checks. The `iou.rs`
    /// builders now guarantee the full `d × g` shape themselves (geometry-less
    /// annotations occupy zero rows/columns), so the short-matrix guards below
    /// are belt-and-braces against a future builder regression, and the segm
    /// bbox fallback fires only if that guarantee is ever broken.
    ///
    /// Dispatch is on [`SimKind`] — the sanctioned projection from an `IouType`
    /// to a geometry kernel — through the same marshaling helpers `evaluate()`
    /// uses, which `detection/iou.rs` owns.
    ///
    /// `SimKind::Oks` routes to the bbox arm on purpose: there is no
    /// cross-category OKS, because the ground truths in a cross-category matrix
    /// belong to *other* categories and their keypoint schemas do not line up.
    /// Both callers therefore compare boxes on a keypoint run.
    ///
    /// `EvalMode::Coco` in every call, regardless of the evaluator's own mode:
    /// the helpers derive each ground truth's crowd flag from its annotation,
    /// and both callers drop `iscrowd` ground truths before getting here, so the
    /// flags come out uniformly false. Passing the real mode would *not* be
    /// equivalent, because under Open Images `uses_ioa` reads `is_group_of`,
    /// which nothing here filters.
    pub(super) fn cross_category_iou(
        dt_ann_ids: &[u64],
        gt_ann_ids: &[u64],
        coco_dt: &COCO,
        coco_gt: &COCO,
        iou_type: IouType,
        segm_rles: Option<&super::iou::SegmRles>,
    ) -> Vec<f64> {
        let d = dt_ann_ids.len();
        let g = gt_ann_ids.len();
        if d == 0 || g == 0 {
            return vec![];
        }

        let bbox = || {
            Self::compute_bbox_iou_static(
                coco_gt,
                coco_dt,
                dt_ann_ids,
                gt_ann_ids,
                super::EvalMode::Coco,
            )
        };

        // A short matrix would mean some row or column lands under the wrong
        // index once flattened. The builders guarantee full shape today; if a
        // regression ever reintroduces a short matrix, zeros are the honest
        // answer for a matrix we cannot align.
        let full = |nested: &[Vec<f64>]| nested.len() == d && nested[0].len() == g;
        let zero_if_short = |nested: Vec<Vec<f64>>| {
            if full(&nested) {
                nested
            } else {
                vec![vec![0.0; g]; d]
            }
        };

        let nested = match SimKind::from(iou_type) {
            SimKind::Bbox | SimKind::Oks => zero_if_short(bbox()),
            SimKind::Mask => {
                // The last `evaluate()` already rasterized these; the
                // cache-or-convert policy lives on `SegmRles`.
                let masks = Self::compute_segm_iou_static(
                    coco_gt,
                    coco_dt,
                    dt_ann_ids,
                    gt_ann_ids,
                    super::EvalMode::Coco,
                    segm_rles,
                );
                // Bbox fallback when any RLE is missing.
                if full(&masks) { masks } else { bbox() }
            }
            SimKind::Obb => zero_if_short(Self::compute_obb_iou_static(
                coco_gt,
                coco_dt,
                dt_ann_ids,
                gt_ann_ids,
                super::EvalMode::Coco,
            )),
        };

        flatten_iou(nested, d, g)
    }

    /// Collect one image's ground truths and detections across **all**
    /// categories, tagged with the category slot each came from — the input to
    /// [`cross_category_iou`](Self::cross_category_iou) for both the confusion
    /// matrix and TIDE's cross-category pass. Crowd ground truths are dropped: a
    /// crowd column would let one region absorb several detections and
    /// double-count.
    ///
    /// `dt_rank` is what differs between the callers. `Some` — the confusion
    /// matrix — drops detections below `min_score`, orders the rest
    /// score-descending, and caps them at `max_det`, ready for the matcher.
    /// `None` — TIDE — keeps every detection in index order.
    ///
    /// Walks `get_ann_ids_for_img` rather than sweeping `cat_ids`, keeping the
    /// cost `O(annotations in this image)` instead of `O(categories)` hash probes
    /// per image.
    ///
    /// Order is contract — both callers are tie-order sensitive. The per-image
    /// and per-pair id lists keep `dataset.annotations` order
    /// (`COCO::create_index`), so a **stable** sort by slot yields category-major
    /// order with JSON order within each category.
    pub(super) fn cross_category_pairs(
        coco_gt: &COCO,
        coco_dt: &COCO,
        cat_slots: &HashMap<u64, usize>,
        img_id: u64,
        dt_rank: Option<DtRank>,
    ) -> (Vec<CatPair>, Vec<CatPair>) {
        let mut gt_pairs: Vec<CatPair> = coco_gt
            .get_ann_ids_for_img(img_id)
            .iter()
            .filter_map(|&ann_id| {
                let ann = coco_gt.get_ann(ann_id)?;
                if ann.iscrowd {
                    return None;
                }
                Some((*cat_slots.get(&ann.category_id)?, ann_id))
            })
            .collect();
        // Stable: see the ordering note above.
        gt_pairs.sort_by_key(|&(cat_idx, _)| cat_idx);

        let Some(rank) = dt_rank else {
            let mut dt_pairs: Vec<CatPair> = coco_dt
                .get_ann_ids_for_img(img_id)
                .iter()
                .filter_map(|&ann_id| {
                    let ann = coco_dt.get_ann(ann_id)?;
                    Some((*cat_slots.get(&ann.category_id)?, ann_id))
                })
                .collect();
            dt_pairs.sort_by_key(|&(cat_idx, _)| cat_idx);
            return (gt_pairs, dt_pairs);
        };

        let mut scored: Vec<(usize, f64, u64)> = coco_dt
            .get_ann_ids_for_img(img_id)
            .iter()
            .filter_map(|&ann_id| {
                let ann = coco_dt.get_ann(ann_id)?;
                let score = ann.score.unwrap_or(0.0);
                if rank.min_score.is_some_and(|ms| score < ms) {
                    return None;
                }
                Some((*cat_slots.get(&ann.category_id)?, score, ann_id))
            })
            .collect();

        // Category-major first, then a stable sort by score descending — equal
        // scores keep category-major order.
        scored.sort_by_key(|&(cat_idx, _, _)| cat_idx);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(rank.max_det);

        let dt_pairs = scored
            .into_iter()
            .map(|(cat_idx, _, ann_id)| (cat_idx, ann_id))
            .collect();
        (gt_pairs, dt_pairs)
    }

    /// `cat_id -> slot` for a category list, built once per analysis call and
    /// hoisted out of the per-image loop.
    pub(super) fn cat_slots(cat_ids: &[u64]) -> HashMap<u64, usize> {
        cat_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect()
    }

    /// Compute a per-category confusion matrix across all images.
    ///
    /// Unlike `evaluate()`, this method compares **all** detections in an image against
    /// **all** ground truth boxes regardless of category. This enables cross-category
    /// confusion analysis ("the model keeps predicting `dog` on `cat` ground truth").
    ///
    /// This is a `&self` method — it does not call `evaluate()` and does not mutate state.
    /// It can be called standalone at any point after constructing `COCOeval`.
    ///
    /// # Matrix layout (rows = GT, cols = predicted)
    ///
    /// - `matrix[gt_cat_idx][dt_cat_idx]` — matched pair (true positive if same category)
    /// - `matrix[gt_cat_idx][num_cats]` — unmatched GT (false negative / missed detection)
    /// - `matrix[num_cats][dt_cat_idx]` — unmatched DT (false positive / spurious detection)
    ///
    /// # Arguments
    ///
    /// - `iou_thr` — IoU threshold for a DT↔GT match (default 0.5)
    /// - `max_det` — max detections per image after score sorting; `None` uses the last
    ///   value of `params.max_dets`
    /// - `min_score` — discard DTs below this confidence before the `max_det` truncation;
    ///   `None` keeps all detections
    pub fn confusion_matrix(
        &self,
        iou_thr: f64,
        max_det: Option<usize>,
        min_score: Option<f64>,
    ) -> ConfusionMatrix {
        // Respect user-set params filters but do not mutate. `resolved_ids` is the
        // owner — the same derivation `evaluate()`'s `resolve_params` writes back
        // — so a standalone `confusion_matrix()` covers exactly the ids an
        // `evaluate()` would.
        let (img_ids, cat_ids) = self.resolved_ids();
        let cat_slots = Self::cat_slots(&cat_ids);

        let num_cats = cat_ids.len();
        let k = num_cats + 1; // background index = num_cats
        let eff_max_det = max_det.unwrap_or_else(|| self.params.max_det());
        let iou_type = self.params.iou_type;

        let coco_gt = &self.coco_gt;
        let coco_dt = &self.coco_dt;
        // Populated iff the last `evaluate()` was a segm run; per-id fallback
        // inside `cross_category_iou` covers the standalone-call case.
        let segm_rles = self.segm_rles.as_ref();

        // Per-worker accumulation is **sparse**: an image touches at most `d + g`
        // of the `(K+1)²` cells, so each split collects that many `(row, col)`
        // pairs and one dense matrix is filled at the end. Counts are integer
        // addition and every pair lands in exactly one cell, so the order pairs
        // are appended and summed in cannot change the result.
        let (gt_labels, dt_labels) = img_ids
            .par_iter()
            .fold(LabelPairs::default, |mut acc: LabelPairs, &img_id| {
                // Detections arrive score-descending and capped, which is
                // what the matcher below assumes of its row order.
                let (gt_pairs, dt_pairs) = Self::cross_category_pairs(
                    coco_gt,
                    coco_dt,
                    &cat_slots,
                    img_id,
                    Some(DtRank {
                        min_score,
                        max_det: eff_max_det,
                    }),
                );

                if gt_pairs.is_empty() && dt_pairs.is_empty() {
                    return acc;
                }

                let d = dt_pairs.len();
                let g = gt_pairs.len();

                // --- Compute cross-category IoU matrix [D × G] ---
                let dt_ids: Vec<u64> = dt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let gt_ids: Vec<u64> = gt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let iou_flat = Self::cross_category_iou(
                    &dt_ids, &gt_ids, coco_dt, coco_gt, iou_type, segm_rles,
                );

                // --- Greedy matching at iou_thr (DTs already in score-sorted order) ---
                //
                // The shared matcher at its simplest setting: one threshold, every
                // GT non-ignored (phase 2 never runs), no crowd (a crowd GT would
                // double-count in a cross-category matrix). `iou_thr` is passed
                // through **unclamped** — hotcoco-native analysis does not inherit
                // pycocotools' `min(t, 1-1e-10)` match floor; see the policy table
                // in `primitives::greedy`.
                let matches = primitives::greedy::greedy_match_masked(
                    &iou_flat,
                    d,
                    g,
                    g, // all GTs non-ignored
                    primitives::greedy::GtMasks::default(),
                    &[iou_thr],
                );
                // Both halves of the matcher's result. Deriving `gt_matched`
                // from `dt_gt` here would be a second answer to a question the
                // matcher already answered, free to drift from the one `EvalImg`
                // reports.
                let matched = matches.dt_gt.row(0);
                let gt_matched = matches.gt_matched.row(0);

                // Turn the matching into one record per decision: every detection
                // (paired with a GT category or with background), then every GT
                // that nothing claimed. Translating matches into labeled pairs is
                // the detection-specific step — the counting itself belongs to
                // `metrics::confusion` and is shared with every other family.
                acc.gt.reserve(d + g);
                acc.dt.reserve(d + g);

                for (di, &gi_opt) in matched.iter().enumerate() {
                    acc.gt.push(gi_opt.map(|gi| gt_pairs[gi].0));
                    acc.dt.push(Some(dt_pairs[di].0));
                }
                for (is_matched, &(gt_cat_idx, _)) in gt_matched.iter().zip(gt_pairs.iter()) {
                    if !is_matched {
                        acc.gt.push(Some(gt_cat_idx));
                        acc.dt.push(None);
                    }
                }

                acc
            })
            .reduce(LabelPairs::default, LabelPairs::append)
            .into_parts();

        // One dense matrix, filled once. `accumulate_confusion` stays the owner of
        // the label-pair → cell mapping, background lane included.
        let mut matrix = vec![0u64; k * k];
        confusion::accumulate_confusion(&mut matrix, &gt_labels, &dt_labels, num_cats);

        // `COCO::cat_name` owns the unnamed-category fallback.
        let cat_names: Vec<String> = cat_ids
            .iter()
            .map(|&id| self.coco_gt.cat_name(id))
            .collect();

        ConfusionMatrix {
            matrix,
            num_cats,
            cat_ids: cat_ids.into_owned(),
            cat_names,
            iou_thr,
        }
    }
}

/// Row-major flattening of a `sim` kernel's nested `[D][G]` output.
///
/// The result is always exactly `d * g` long: a short row, or fewer rows than
/// `d`, leaves zeros rather than shifting every later entry into the wrong cell.
/// That is what lets callers index `flat[di * g + gi]` with no shape test.
fn flatten_iou(nested: Vec<Vec<f64>>, d: usize, g: usize) -> Vec<f64> {
    let mut flat = vec![0.0_f64; d * g];
    for (di, row) in nested.into_iter().take(d).enumerate() {
        let base = di * g;
        for (gi, v) in row.into_iter().take(g).enumerate() {
            flat[base + gi] = v;
        }
    }
    flat
}

/// The parallel fold's per-split accumulator: match records, not counts.
///
/// One entry per matching decision, in the aligned form
/// [`metrics::confusion::accumulate_confusion`](crate::metrics::confusion::accumulate_confusion)
/// consumes. Keeping the two label vectors together is what makes the merge a
/// single operation that cannot append to one and forget the other — they are
/// index-parallel, and a split pair is silently wrong rather than a length error.
#[derive(Default)]
struct LabelPairs {
    gt: Vec<Option<usize>>,
    dt: Vec<Option<usize>>,
}

impl LabelPairs {
    /// Concatenate two splits' records. Order is irrelevant to the counts — every
    /// record lands in exactly one cell and the cells are integer counters — but
    /// the two vectors must stay aligned, which is why this is one function.
    fn append(mut self, mut other: Self) -> Self {
        self.gt.append(&mut other.gt);
        self.dt.append(&mut other.dt);
        self
    }

    fn into_parts(self) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
        (self.gt, self.dt)
    }
}

/// Per-category confusion matrix for object detection.
///
/// Rows are ground truth categories, columns are predicted categories.
/// Index `num_cats` (the last row/column) represents "background" — unmatched GTs
/// (false negatives) land in the background column, unmatched DTs (false positives)
/// land in the background row.
///
/// Use [`super::COCOeval::confusion_matrix`] to compute this.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Raw counts, row-major, shape (num_cats+1) × (num_cats+1).
    /// Index `K = num_cats` is the background row/column.
    pub matrix: Vec<u64>,
    pub num_cats: usize,
    /// Category IDs corresponding to rows/cols 0..num_cats-1.
    pub cat_ids: Vec<u64>,
    /// Category names corresponding to rows/cols 0..num_cats-1.
    pub cat_names: Vec<String>,
    pub iou_thr: f64,
}

impl ConfusionMatrix {
    /// Get the count at row `gt_idx`, column `pred_idx`.
    pub fn get(&self, gt_idx: usize, pred_idx: usize) -> u64 {
        let k = self.num_cats + 1;
        self.matrix[gt_idx * k + pred_idx]
    }

    /// Row-normalized matrix as flat `Vec<f64>` (same shape as `matrix`).
    ///
    /// Each row is divided by its sum so rows sum to 1.0.
    /// Zero rows remain all-zero.
    pub fn normalized(&self) -> Vec<f64> {
        confusion::row_normalize(&self.matrix, self.num_cats)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::coco::COCO;
    use crate::types::Dataset;

    /// The category-major derivation `cross_category_pairs` replaced, written out
    /// in full. Kept in the test rather than in the source because it is the
    /// *old* implementation: its only job is to say what the new one must equal.
    fn category_major_reference(
        coco_gt: &COCO,
        coco_dt: &COCO,
        cat_ids: &[u64],
        img_id: u64,
        dt_rank: Option<DtRank>,
    ) -> (Vec<CatPair>, Vec<CatPair>) {
        let gt_pairs: Vec<CatPair> = cat_ids
            .iter()
            .enumerate()
            .flat_map(|(cat_idx, &cat_id)| {
                coco_gt
                    .get_ann_ids_for_img_cat(img_id, cat_id)
                    .iter()
                    .filter_map(move |&ann_id| {
                        let ann = coco_gt.get_ann(ann_id)?;
                        if ann.iscrowd {
                            return None;
                        }
                        Some((cat_idx, ann_id))
                    })
            })
            .collect();

        let Some(rank) = dt_rank else {
            let dt_pairs = cat_ids
                .iter()
                .enumerate()
                .flat_map(|(cat_idx, &cat_id)| {
                    coco_dt
                        .get_ann_ids_for_img_cat(img_id, cat_id)
                        .iter()
                        .map(move |&ann_id| (cat_idx, ann_id))
                })
                .collect();
            return (gt_pairs, dt_pairs);
        };

        let mut scored: Vec<(usize, f64, u64)> = cat_ids
            .iter()
            .enumerate()
            .flat_map(|(cat_idx, &cat_id)| {
                coco_dt
                    .get_ann_ids_for_img_cat(img_id, cat_id)
                    .iter()
                    .filter_map(move |&ann_id| {
                        let ann = coco_dt.get_ann(ann_id)?;
                        let score = ann.score.unwrap_or(0.0);
                        if rank.min_score.is_some_and(|ms| score < ms) {
                            return None;
                        }
                        Some((cat_idx, score, ann_id))
                    })
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(rank.max_det);

        let dt_pairs = scored
            .into_iter()
            .map(|(cat_idx, _, ann_id)| (cat_idx, ann_id))
            .collect();
        (gt_pairs, dt_pairs)
    }

    /// Ground truth deliberately **interleaved** across categories, in an order no
    /// category-major walk would produce: 3, 1, 2, 1, 3, 2. Plus one crowd box
    /// (dropped) and one annotation in a category outside `cat_ids` (also dropped).
    /// Detections carry repeated scores so tie-breaking is exercised, and their
    /// insertion order is interleaved too.
    fn interleaved_fixture() -> (COCO, COCO, Vec<u64>) {
        let gt_json = serde_json::json!({
            "images": [{"id": 1, "width": 200, "height": 200}],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 11, "image_id": 1, "category_id": 1, "bbox": [10, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 12, "image_id": 1, "category_id": 2, "bbox": [20, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 13, "image_id": 1, "category_id": 1, "bbox": [30, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 14, "image_id": 1, "category_id": 3, "bbox": [40, 0, 10, 10], "area": 100, "iscrowd": 1},
                {"id": 15, "image_id": 1, "category_id": 2, "bbox": [50, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 16, "image_id": 1, "category_id": 9, "bbox": [60, 0, 10, 10], "area": 100, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "a"}, {"id": 2, "name": "b"},
                {"id": 3, "name": "c"}, {"id": 9, "name": "outside"}
            ]
        });
        let ds: Dataset = serde_json::from_value(gt_json).unwrap();
        let gt = COCO::from_dataset(ds);

        let dt = gt
            .load_res_anns(
                serde_json::from_value(serde_json::json!([
                    {"image_id": 1, "category_id": 2, "bbox": [20, 0, 10, 10], "score": 0.5},
                    {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
                    {"image_id": 1, "category_id": 1, "bbox": [10, 0, 10, 10], "score": 0.5},
                    {"image_id": 1, "category_id": 2, "bbox": [50, 0, 10, 10], "score": 0.9},
                    {"image_id": 1, "category_id": 1, "bbox": [30, 0, 10, 10], "score": 0.1},
                    {"image_id": 1, "category_id": 9, "bbox": [60, 0, 10, 10], "score": 0.7}
                ]))
                .unwrap(),
            )
            .unwrap();

        (gt, dt, vec![1, 2, 3])
    }

    /// The walk-the-image rewrite must be **byte-identical** to the
    /// category-major sweep it replaced, ordering included: both callers feed the
    /// result to a stable sort, so a reordering of equal-scoring detections would
    /// silently change which ground truth each one claims.
    #[test]
    fn pairs_match_the_category_major_derivation() {
        let (gt, dt, cat_ids) = interleaved_fixture();
        let slots = COCOeval::cat_slots(&cat_ids);

        for rank in [
            None,
            Some(DtRank {
                min_score: None,
                max_det: 100,
            }),
            Some(DtRank {
                min_score: Some(0.4),
                max_det: 100,
            }),
            Some(DtRank {
                min_score: None,
                max_det: 2,
            }),
        ] {
            let got = COCOeval::cross_category_pairs(&gt, &dt, &slots, 1, rank);
            let want = category_major_reference(&gt, &dt, &cat_ids, 1, rank);
            assert_eq!(got, want, "diverged for dt_rank = {rank:?}");
        }
    }

    /// The fixture has to actually exercise the thing: if the annotations came
    /// out category-major already, the test above would pass against any
    /// implementation.
    #[test]
    fn fixture_insertion_order_is_not_already_category_major() {
        let (gt, _, cat_ids) = interleaved_fixture();
        let slots = COCOeval::cat_slots(&cat_ids);
        let raw: Vec<usize> = gt
            .get_ann_ids_for_img(1)
            .iter()
            .filter_map(|&id| slots.get(&gt.get_ann(id)?.category_id).copied())
            .collect();
        assert!(
            raw.windows(2).any(|w| w[0] > w[1]),
            "fixture is already sorted by category slot: {raw:?}"
        );
    }
}
