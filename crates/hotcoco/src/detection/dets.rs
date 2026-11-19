//! Where an evaluation reads its detections: always a
//! [`DetectionTable`], with the grouping index and the id lookup of the
//! results set it was filled from.
//!
//! The table comes from a results [`COCO`]'s records, or is the one a
//! [`DetectionColumns`] kept for an array of boxes. Both are the same results
//! set — [`DetectionColumns::coco`] is the `COCO` — with the same ids and the
//! same index, so the matcher cannot tell which it was handed.

use std::sync::Arc;

use crate::ann_index::{AnnIndex, IdIndex};
use crate::coco::{COCO, DetectionColumns};
use crate::table::{DetectionTable, GroundTruthTable, GtFlags};

/// The fields of a detection the matcher reads.
#[derive(Debug, Clone, Copy)]
pub(super) struct DtRef {
    pub(super) id: u64,
    pub(super) score: f64,
    pub(super) area: f64,
}

/// The fields of a ground truth the matcher reads.
#[derive(Debug, Clone, Copy)]
pub(super) struct GtRef {
    pub(super) id: u64,
    pub(super) area: f64,
    pub(super) flags: GtFlags,
}

/// The ground truth of one evaluation, borrowed: its table, with the grouping
/// index and id lookup of the dataset it was filled from.
#[derive(Clone, Copy)]
pub(super) struct Gts<'a> {
    coco: &'a COCO,
    table: &'a GroundTruthTable,
}

impl<'a> Gts<'a> {
    /// `table` must be [`GroundTruthTable::from_records`] of `coco`'s
    /// annotations as they are now.
    pub(super) fn new(coco: &'a COCO, table: &'a GroundTruthTable) -> Self {
        Gts { coco, table }
    }

    /// The ids by image and by (image, category), in array order.
    pub(super) fn index(self) -> &'a AnnIndex {
        self.coco.ann_index()
    }

    /// The records, which every geometry but a box is still read from.
    pub(super) fn records(self) -> &'a COCO {
        self.coco
    }

    /// The ground truth with `id`.
    pub(super) fn get(self, id: u64) -> Option<GtRef> {
        let i = self.coco.id_index().position(id)?;
        Some(GtRef {
            id,
            area: self.table.area(i),
            flags: self.table.flags(i),
        })
    }

    /// The box of the ground truth with `id`, and its flags.
    pub(super) fn bbox(self, id: u64) -> Option<([f64; 4], GtFlags)> {
        let i = self.coco.id_index().position(id)?;
        Some((self.table.bbox(i)?, self.table.flags(i)))
    }
}

/// The detections of one evaluation, borrowed.
#[derive(Clone, Copy)]
pub(super) struct Dets<'a> {
    index: &'a AnnIndex,
    ids: &'a IdIndex,
    table: &'a DetectionTable,
    source: &'a DtHandle,
}

impl<'a> Dets<'a> {
    /// The ids by image and by (image, category), in array order.
    pub(super) fn index(self) -> &'a AnnIndex {
        self.index
    }

    /// How many detections.
    pub(super) fn len(self) -> usize {
        self.table.len()
    }

    /// The detection with `id`.
    pub(super) fn get(self, id: u64) -> Option<DtRef> {
        let i = self.ids.position(id)?;
        Some(DtRef {
            id,
            score: self.table.score(i),
            area: self.table.area(i),
        })
    }

    /// The box of the detection with `id`.
    pub(super) fn bbox(self, id: u64) -> Option<[f64; 4]> {
        self.table.bbox(self.ids.position(id)?)
    }

    /// The records, which every geometry but a box is still read from. For
    /// detections that came as an array, this call is what builds them.
    pub(super) fn records(self) -> &'a COCO {
        self.source.coco()
    }
}

/// What a constructor hands `COCOeval::with_mode`: records not yet capped or
/// area-filled, or columns. No table is built until `with_mode` knows which
/// records the evaluation reads.
pub(super) enum DtSource {
    Records(Arc<COCO>),
    Columns(Arc<DetectionColumns>),
}

/// An evaluator's detections, owned: a results `COCO` and the table filled
/// from it, or the columns `load_res` kept for an array of boxes, whose `COCO`
/// is built the first time something reads the records — `report()`, TIDE,
/// the per-image records, a geometry other than boxes.
pub(super) enum DtHandle {
    Records {
        coco: Arc<COCO>,
        table: DetectionTable,
    },
    Columns(Arc<DetectionColumns>),
}

impl DtHandle {
    /// The handle of a results `COCO`: its table is filled here, so the
    /// records must be the ones the evaluation is to read — capped, expanded,
    /// and with their areas filled.
    pub(super) fn records(coco: Arc<COCO>) -> Self {
        let table = DetectionTable::from_records(&coco.dataset.annotations);
        DtHandle::Records { coco, table }
    }

    /// Fill the table's box column, before a box evaluation reads it. Columns
    /// hold theirs from the start.
    pub(super) fn fill_boxes(&mut self) {
        if let DtHandle::Records { coco, table } = self {
            table.fill_boxes(&coco.dataset.annotations);
        }
    }

    /// What the matcher reads.
    pub(super) fn dets(&self) -> Dets<'_> {
        match self {
            DtHandle::Records { coco, table } => Dets {
                index: coco.ann_index(),
                ids: coco.id_index(),
                table,
                source: self,
            },
            DtHandle::Columns(columns) => Dets {
                index: &columns.index,
                ids: &columns.ids,
                table: &columns.table,
                source: self,
            },
        }
    }

    /// The results `COCO`, built from the columns on first call.
    pub(super) fn coco(&self) -> &Arc<COCO> {
        match self {
            DtHandle::Records { coco, .. } => coco,
            DtHandle::Columns(columns) => columns.coco(),
        }
    }

    /// Whether the detections came as an array of boxes.
    pub(super) fn is_columns(&self) -> bool {
        matches!(self, DtHandle::Columns(_))
    }
}
