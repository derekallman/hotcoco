//! Where an evaluation reads its detections: a results [`COCO`]'s records, or
//! the rows a [`DetectionColumns`] keeps for an array of boxes.
//!
//! Both are the same results set — [`DetectionColumns::to_coco`] is the
//! `COCO` — with the same ids (`1..=n` in row order) and the same index, so
//! everything below answers alike for either. The matcher reads three fields
//! of a detection and the box IoU a fourth; those are what [`Dets`] serves.

use std::sync::Arc;

use crate::ann_index::AnnIndex;
use crate::coco::{COCO, DetectionColumns, box_area};

/// The fields of a detection the matcher reads.
#[derive(Debug, Clone, Copy)]
pub(super) struct DtRef {
    pub(super) id: u64,
    /// `0.0` for a detection without a score, as the ranking reads one.
    pub(super) score: f64,
    /// `0.0` for a detection without an area, as the area ranges read one.
    pub(super) area: f64,
}

/// The detections of one evaluation, borrowed.
#[derive(Clone, Copy)]
pub(super) enum Dets<'a> {
    Records(&'a COCO),
    Columns(&'a DetectionColumns),
}

impl<'a> Dets<'a> {
    /// The ids by image and by (image, category), in array order.
    pub(super) fn index(self) -> &'a AnnIndex {
        match self {
            Dets::Records(coco) => coco.ann_index(),
            Dets::Columns(columns) => &columns.index,
        }
    }

    /// The detection with `id`.
    pub(super) fn get(self, id: u64) -> Option<DtRef> {
        match self {
            Dets::Records(coco) => coco.get_ann(id).map(|ann| DtRef {
                id: ann.id,
                score: ann.score.unwrap_or(0.0),
                area: ann.area.unwrap_or(0.0),
            }),
            // The area `load_res` gives a box result, which its record holds.
            Dets::Columns(columns) => columns.position(id).map(|i| {
                let row = &columns.rows[i];
                DtRef {
                    id,
                    score: row.score,
                    area: box_area(&row.bbox),
                }
            }),
        }
    }

    /// The box of the detection with `id`.
    pub(super) fn bbox(self, id: u64) -> Option<[f64; 4]> {
        match self {
            Dets::Records(coco) => coco.get_ann(id)?.bbox,
            Dets::Columns(columns) => columns.position(id).map(|i| columns.rows[i].bbox),
        }
    }

    /// The records, which every geometry but a box needs. `None` for columns,
    /// which an evaluator reads only for a box evaluation.
    pub(super) fn records(self) -> Option<&'a COCO> {
        match self {
            Dets::Records(coco) => Some(coco),
            Dets::Columns(_) => None,
        }
    }
}

/// An evaluator's detections, owned: a results `COCO`, or the columns
/// `load_res` kept for an array of boxes, whose `COCO` is built the first
/// time something other than a box `evaluate()` reads the detections —
/// `report()`, TIDE, the per-image records, a geometry other than boxes.
pub(super) enum DtHandle {
    Records(Arc<COCO>),
    Columns(Arc<DetectionColumns>),
}

impl DtHandle {
    /// The results `COCO`, built from the columns on first call.
    pub(super) fn coco(&self) -> &Arc<COCO> {
        match self {
            DtHandle::Records(coco) => coco,
            DtHandle::Columns(columns) => columns.coco(),
        }
    }

    /// The columns, when the detections came as an array of boxes.
    pub(super) fn columns(&self) -> Option<&DetectionColumns> {
        match self {
            DtHandle::Records(_) => None,
            DtHandle::Columns(columns) => Some(columns),
        }
    }

    /// How many detections, without building the `COCO`.
    pub(super) fn len(&self) -> usize {
        match self {
            DtHandle::Records(coco) => coco.dataset.annotations.len(),
            DtHandle::Columns(columns) => columns.len(),
        }
    }
}
