//! The compact tables an evaluator reads, in place of annotation records.
//!
//! A record ([`Annotation`]) is about 200 bytes and holds everything a COCO
//! file can say. An evaluator reads a handful of numbers per annotation, so it
//! reads them from columns: one value per annotation, addressed by the
//! annotation's position. A table is filled once, from records or straight
//! from an array, and never edited; a changed dataset gets a new table.
//!
//! Geometry columns are their own types ([`BoxColumn`]) rather than fields of
//! a row, so another family's table — tracking's, grouped by sequence and
//! frame — holds the same column beside different per-row values.

use rayon::prelude::*;

use crate::coco::box_area;
use crate::primitives::sim::worth_parallel;
use crate::types::Annotation;

/// One `[x, y, w, h]` box per row, or none for a row whose annotation has no
/// box.
#[derive(Debug, Default)]
pub(crate) struct BoxColumn {
    xywh: Vec<[f64; 4]>,
    /// Rows without a box, when there are any: a detector's results have one
    /// on every row, and pay nothing for the rows that could lack one.
    missing: Option<Vec<bool>>,
}

impl BoxColumn {
    /// A column with a box on every row.
    pub(crate) fn full(xywh: Vec<[f64; 4]>) -> Self {
        BoxColumn {
            xywh,
            missing: None,
        }
    }

    /// The column of boxes read off records, some of which may have none.
    fn from_options(boxes: Vec<Option<[f64; 4]>>) -> Self {
        let missing = boxes
            .iter()
            .any(Option::is_none)
            .then(|| boxes.iter().map(Option::is_none).collect());
        let xywh = boxes.into_iter().map(Option::unwrap_or_default).collect();
        BoxColumn { xywh, missing }
    }

    /// The box of row `i`.
    pub(crate) fn get(&self, i: usize) -> Option<[f64; 4]> {
        match &self.missing {
            Some(missing) if missing[i] => None,
            _ => Some(self.xywh[i]),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.xywh.len()
    }
}

/// What a detection evaluation reads of each detection, one row per
/// detection in the order of the results set.
///
/// Rows are addressed by position. A detection's image and category are not
/// columns: they are the grouping index's, which lists a group's detections.
#[derive(Debug, Default)]
pub(crate) struct DetectionTable {
    /// `0.0` for a detection without a score, as the ranking reads one.
    score: Vec<f64>,
    /// `0.0` for a detection without an area, as the area ranges read one.
    area: Vec<f64>,
    boxes: BoxColumn,
}

impl DetectionTable {
    /// The table of a results set's records, row `i` for `anns[i]`.
    pub(crate) fn from_records(anns: &[Annotation]) -> Self {
        // One pass over the records: they are the expensive thing to read.
        let (score, (area, boxes)) = collect(anns, |ann| {
            (
                ann.score.unwrap_or(0.0),
                (ann.area.unwrap_or(0.0), ann.bbox),
            )
        });
        DetectionTable {
            score,
            area,
            boxes: BoxColumn::from_options(boxes),
        }
    }

    /// The table of box results that never were records: each row's area is
    /// its box's, the area `load_res` gives a box result.
    pub(crate) fn from_boxes(xywh: Vec<[f64; 4]>, score: Vec<f64>) -> Self {
        debug_assert_eq!(xywh.len(), score.len());
        let area = if worth_parallel(xywh.len()) {
            xywh.par_iter().map(box_area).collect()
        } else {
            xywh.iter().map(box_area).collect()
        };
        DetectionTable {
            score,
            area,
            boxes: BoxColumn::full(xywh),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.boxes.len()
    }

    pub(crate) fn score(&self, i: usize) -> f64 {
        self.score[i]
    }

    pub(crate) fn area(&self, i: usize) -> f64 {
        self.area[i]
    }

    pub(crate) fn bbox(&self, i: usize) -> Option<[f64; 4]> {
        self.boxes.get(i)
    }
}

/// The flags a protocol reads off a ground truth, read from its record once.
///
/// Each protocol reads its own — see `GtRule` in the matcher — so all are
/// kept and the rule picks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GtFlags(u8);

impl GtFlags {
    const CROWD: u8 = 1;
    const GROUP_OF: u8 = 2;
    const LVIS_IGNORE: u8 = 4;
    const NO_LABELED_KEYPOINTS: u8 = 8;

    /// The flags of `ann`.
    pub(crate) fn of(ann: &Annotation) -> Self {
        let bit = |on: bool, bit: u8| if on { bit } else { 0 };
        GtFlags(
            bit(ann.iscrowd, Self::CROWD)
                | bit(ann.is_group_of.unwrap_or(false), Self::GROUP_OF)
                | bit(lvis_ignore(ann), Self::LVIS_IGNORE)
                | bit(ann.num_visible_keypoints() == 0, Self::NO_LABELED_KEYPOINTS),
        )
    }

    /// COCO's `iscrowd`.
    pub(crate) fn crowd(self) -> bool {
        self.0 & Self::CROWD != 0
    }

    /// Open Images' `IsGroupOf`.
    pub(crate) fn group_of(self) -> bool {
        self.0 & Self::GROUP_OF != 0
    }

    /// lvis-api's `ignore`.
    pub(crate) fn lvis_ignore(self) -> bool {
        self.0 & Self::LVIS_IGNORE != 0
    }

    /// No labeled keypoint, which COCO's keypoint evaluation ignores.
    pub(crate) fn no_labeled_keypoints(self) -> bool {
        self.0 & Self::NO_LABELED_KEYPOINTS != 0
    }
}

/// lvis-api's `if g["ignore"]`: the annotation's `ignore` key, by Python
/// truthiness. `ignore` is not a COCO schema field, so it rides in `extra`;
/// real LVIS files do not carry it and this is one lookup in an empty map.
fn lvis_ignore(ann: &Annotation) -> bool {
    use serde_json::Value;
    match ann.extra.get("ignore") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// What a detection evaluation reads of each ground truth, one row per
/// annotation in the order of the dataset. Rows are addressed by position.
#[derive(Debug, Default)]
pub(crate) struct GroundTruthTable {
    /// `0.0` for a ground truth without an area, as the area ranges read one.
    area: Vec<f64>,
    flags: Vec<GtFlags>,
    boxes: BoxColumn,
}

impl GroundTruthTable {
    /// The table of a dataset's records, row `i` for `anns[i]`.
    pub(crate) fn from_records(anns: &[Annotation]) -> Self {
        let (area, (flags, boxes)) = collect(anns, |ann| {
            (ann.area.unwrap_or(0.0), (GtFlags::of(ann), ann.bbox))
        });
        GroundTruthTable {
            area,
            flags,
            boxes: BoxColumn::from_options(boxes),
        }
    }

    pub(crate) fn area(&self, i: usize) -> f64 {
        self.area[i]
    }

    pub(crate) fn flags(&self, i: usize) -> GtFlags {
        self.flags[i]
    }

    pub(crate) fn bbox(&self, i: usize) -> Option<[f64; 4]> {
        self.boxes.get(i)
    }
}

/// Three columns off the records in one pass, in parallel past the shared
/// threshold.
fn collect<A: Send, B: Send, C: Send>(
    anns: &[Annotation],
    fields: impl Fn(&Annotation) -> (A, (B, C)) + Sync + Send,
) -> (Vec<A>, (Vec<B>, Vec<C>)) {
    if worth_parallel(anns.len()) {
        anns.par_iter().map(fields).unzip()
    } else {
        anns.iter().map(fields).unzip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_boxes_fill_the_same_table() {
        let boxes = vec![[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 10.0, 0.5]];
        let scores = vec![0.9, 0.1];
        let anns: Vec<Annotation> = boxes
            .iter()
            .zip(&scores)
            .map(|(b, &s)| Annotation {
                bbox: Some(*b),
                score: Some(s),
                area: Some(box_area(b)),
                ..Default::default()
            })
            .collect();
        let (a, b) = (
            DetectionTable::from_records(&anns),
            DetectionTable::from_boxes(boxes, scores),
        );
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a.score(i), b.score(i));
            assert_eq!(a.area(i), b.area(i));
            assert_eq!(a.bbox(i), b.bbox(i));
        }
    }

    #[test]
    fn a_record_without_a_field_reads_as_the_matcher_expects() {
        let anns = vec![
            Annotation {
                bbox: Some([1.0, 1.0, 2.0, 2.0]),
                ..Default::default()
            },
            Annotation::default(),
        ];
        let t = DetectionTable::from_records(&anns);
        assert_eq!(t.bbox(0), Some([1.0, 1.0, 2.0, 2.0]));
        // No box is no box, not a zero-sized one at the origin.
        assert_eq!(t.bbox(1), None);
        assert_eq!((t.score(1), t.area(1)), (0.0, 0.0));
    }
}
