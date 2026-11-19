//! The compact tables an evaluator reads, in place of annotation records.
//!
//! A record ([`Annotation`]) is about 200 bytes and holds everything a COCO
//! file can say. An evaluator reads a handful of numbers per annotation, so it
//! reads them from columns: one value per annotation, addressed by the
//! annotation's position. A table is filled once, from records or straight
//! from an array, and never edited; a changed dataset gets a new table. The
//! one column that waits is the boxes of a table filled from records, which
//! the first box evaluation fills ([`DetectionTable::fill_boxes`]).
//!
//! Geometry columns are their own types ([`BoxColumn`]) rather than fields of
//! a row, so another family's table — tracking's [`TrackTable`], grouped by
//! frame — holds the same column beside different per-row values.

use std::ops::Range;

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

    /// The boxes of `anns`, row `i` for `anns[i]`.
    fn from_records(anns: &[Annotation]) -> Self {
        // Both columns in the one pass over the records, with no
        // `Option<[f64; 4]>` per row held in between.
        let (xywh, missing): (Vec<[f64; 4]>, Vec<bool>) = collect(anns, |ann| {
            (ann.bbox.unwrap_or_default(), ann.bbox.is_none())
        });
        let missing = missing.contains(&true).then_some(missing);
        BoxColumn { xywh, missing }
    }

    /// The box of row `i`.
    pub(crate) fn get(&self, i: usize) -> Option<[f64; 4]> {
        match &self.missing {
            Some(missing) if missing[i] => None,
            _ => Some(self.xywh[i]),
        }
    }

    /// The boxes of a run of rows, of a column with a box on every row.
    pub(crate) fn slice(&self, rows: Range<usize>) -> &[[f64; 4]] {
        assert!(
            self.missing.is_none(),
            "a column with rows that lack a box has no slice of boxes"
        );
        &self.xywh[rows]
    }
}

/// One row of a [`TrackTable`] as a loader reads it, before the rows are
/// grouped by frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrackRow {
    /// The frame, counted from 0.
    pub(crate) frame: usize,
    /// The track id as given.
    pub(crate) id: u64,
    pub(crate) xywh: [f64; 4],
    pub(crate) class: i64,
    pub(crate) conf: f64,
}

/// One side of one tracking sequence — its ground truth or its predictions:
/// a box per row, grouped by frame, with each frame's rows in one contiguous
/// run in the order they were given.
///
/// Track ids are renumbered `0..num_tracks` once, when the table is built,
/// in ascending order of the ids given, as TrackEval renumbers them: HOTA and
/// Identity count into a grid of every ground-truth track against every
/// predicted track, indexed by the new numbers. Clean-up is a smaller table
/// ([`filter`](Self::filter)), renumbered again, so an evaluation reads a
/// raw table and a cleaned one the same way.
#[derive(Debug)]
pub(crate) struct TrackTable {
    boxes: BoxColumn,
    /// Each row's track, renumbered.
    track: Vec<usize>,
    /// The id each renumbered track was given: `ids[track[i]]` is row `i`'s.
    ids: Vec<u64>,
    class: Vec<i64>,
    conf: Vec<f64>,
    /// Frame `f`'s rows are `frame_start[f]..frame_start[f + 1]`.
    frame_start: Vec<usize>,
}

impl TrackTable {
    /// The table of `rows` over `num_frames` frames, every row's frame below
    /// `num_frames`. A stable grouping: rows of one frame keep their order.
    pub(crate) fn new(num_frames: usize, rows: &[TrackRow]) -> Self {
        let mut frame_start = vec![0usize; num_frames + 1];
        for r in rows {
            frame_start[r.frame + 1] += 1;
        }
        for f in 0..num_frames {
            frame_start[f + 1] += frame_start[f];
        }
        let mut next = frame_start.clone();
        let mut order = vec![0usize; rows.len()];
        for (i, r) in rows.iter().enumerate() {
            order[next[r.frame]] = i;
            next[r.frame] += 1;
        }
        Self::from_order(rows.len(), frame_start, |k| rows[order[k]])
    }

    /// The rows where `keep` is true, still grouped by frame, renumbered
    /// over the tracks that keep a row.
    pub(crate) fn filter(&self, keep: &[bool]) -> Self {
        assert_eq!(keep.len(), self.len(), "one keep flag per row");
        let mut frame_start = Vec::with_capacity(self.frame_start.len());
        frame_start.push(0);
        let mut kept = Vec::with_capacity(keep.iter().filter(|&&k| k).count());
        for f in 0..self.num_frames() {
            kept.extend(self.frame(f).filter(|&i| keep[i]));
            frame_start.push(kept.len());
        }
        let xywh = self.boxes.slice(0..self.len());
        Self::from_order(kept.len(), frame_start, |k| {
            let i = kept[k];
            TrackRow {
                frame: 0,
                id: self.ids[self.track[i]],
                xywh: xywh[i],
                class: self.class[i],
                conf: self.conf[i],
            }
        })
    }

    /// The columns of `n` rows already in frame order, `row(k)` the `k`-th,
    /// with the ids renumbered. A row's `frame` is not read: `frame_start`
    /// says where each frame's run is.
    fn from_order(n: usize, frame_start: Vec<usize>, row: impl Fn(usize) -> TrackRow) -> Self {
        let mut ids: Vec<u64> = (0..n).map(|k| row(k).id).collect();
        ids.sort_unstable();
        ids.dedup();
        let (mut track, mut xywh, mut class, mut conf) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for k in 0..n {
            let r = row(k);
            track.push(ids.binary_search(&r.id).expect("every id was collected"));
            xywh.push(r.xywh);
            class.push(r.class);
            conf.push(r.conf);
        }
        TrackTable {
            boxes: BoxColumn::full(xywh),
            track,
            ids,
            class,
            conf,
            frame_start,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.track.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.track.is_empty()
    }

    pub(crate) fn num_frames(&self) -> usize {
        self.frame_start.len() - 1
    }

    pub(crate) fn num_tracks(&self) -> usize {
        self.ids.len()
    }

    /// The rows of frame `f`, counted from 0.
    pub(crate) fn frame(&self, f: usize) -> Range<usize> {
        self.frame_start[f]..self.frame_start[f + 1]
    }

    pub(crate) fn boxes(&self, rows: Range<usize>) -> &[[f64; 4]] {
        self.boxes.slice(rows)
    }

    /// The renumbered tracks of a run of rows.
    pub(crate) fn tracks(&self, rows: Range<usize>) -> &[usize] {
        &self.track[rows]
    }

    /// The id renumbered track `t` was given.
    pub(crate) fn id_of(&self, t: usize) -> u64 {
        self.ids[t]
    }

    pub(crate) fn class(&self, i: usize) -> i64 {
        self.class[i]
    }

    pub(crate) fn conf(&self, i: usize) -> f64 {
        self.conf[i]
    }

    /// How many rows each renumbered track has: the frames it is in, once
    /// ids are unique within a frame.
    pub(crate) fn track_lengths(&self) -> Vec<u64> {
        let mut lengths = vec![0u64; self.num_tracks()];
        for &t in &self.track {
            lengths[t] += 1;
        }
        lengths
    }
}

/// What a table says when a box is read before [`fill_boxes`] ran: a bug in
/// the evaluator, never a property of the data.
///
/// [`fill_boxes`]: DetectionTable::fill_boxes
const BOXES_NOT_FILLED: &str = "a box evaluation fills the box column before reading it";

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
    /// `None` until a box evaluation asks: a mask or keypoint evaluation
    /// never reads a box, and the column is 32 bytes a detection.
    boxes: Option<BoxColumn>,
}

impl DetectionTable {
    /// The table of a results set's records, row `i` for `anns[i]`, without
    /// its boxes — see [`fill_boxes`](Self::fill_boxes).
    pub(crate) fn from_records(anns: &[Annotation]) -> Self {
        // One pass over the records: they are the expensive thing to read.
        let (score, area) = collect(anns, |ann| {
            (ann.score.unwrap_or(0.0), ann.area.unwrap_or(0.0))
        });
        DetectionTable {
            score,
            area,
            boxes: None,
        }
    }

    /// Fill the box column from the records the table was filled from, if it
    /// is not filled yet. A box evaluation calls this before it reads a box.
    pub(crate) fn fill_boxes(&mut self, anns: &[Annotation]) {
        debug_assert_eq!(anns.len(), self.score.len());
        self.boxes
            .get_or_insert_with(|| BoxColumn::from_records(anns));
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
            boxes: Some(BoxColumn::full(xywh)),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.score.len()
    }

    pub(crate) fn score(&self, i: usize) -> f64 {
        self.score[i]
    }

    pub(crate) fn area(&self, i: usize) -> f64 {
        self.area[i]
    }

    pub(crate) fn bbox(&self, i: usize) -> Option<[f64; 4]> {
        self.boxes.as_ref().expect(BOXES_NOT_FILLED).get(i)
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
    /// `None` until a box evaluation asks, as [`DetectionTable`]'s.
    boxes: Option<BoxColumn>,
}

impl GroundTruthTable {
    /// The table of a dataset's records, row `i` for `anns[i]`, without its
    /// boxes — see [`fill_boxes`](Self::fill_boxes).
    pub(crate) fn from_records(anns: &[Annotation]) -> Self {
        let (area, flags) = collect(anns, |ann| (ann.area.unwrap_or(0.0), GtFlags::of(ann)));
        GroundTruthTable {
            area,
            flags,
            boxes: None,
        }
    }

    /// Fill the box column from the records the table was filled from, if it
    /// is not filled yet. A box evaluation calls this before it reads a box.
    pub(crate) fn fill_boxes(&mut self, anns: &[Annotation]) {
        debug_assert_eq!(anns.len(), self.area.len());
        self.boxes
            .get_or_insert_with(|| BoxColumn::from_records(anns));
    }

    pub(crate) fn area(&self, i: usize) -> f64 {
        self.area[i]
    }

    pub(crate) fn flags(&self, i: usize) -> GtFlags {
        self.flags[i]
    }

    pub(crate) fn bbox(&self, i: usize) -> Option<[f64; 4]> {
        self.boxes.as_ref().expect(BOXES_NOT_FILLED).get(i)
    }
}

/// Two columns off the records in one pass, in parallel past the shared
/// threshold.
fn collect<A: Send, B: Send>(
    anns: &[Annotation],
    fields: impl Fn(&Annotation) -> (A, B) + Sync + Send,
) -> (Vec<A>, Vec<B>) {
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
        let (mut a, b) = (
            DetectionTable::from_records(&anns),
            DetectionTable::from_boxes(boxes, scores),
        );
        a.fill_boxes(&anns);
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
        let mut t = DetectionTable::from_records(&anns);
        t.fill_boxes(&anns);
        assert_eq!(t.bbox(0), Some([1.0, 1.0, 2.0, 2.0]));
        // No box is no box, not a zero-sized one at the origin.
        assert_eq!(t.bbox(1), None);
        assert_eq!((t.score(1), t.area(1)), (0.0, 0.0));
    }

    fn boxed() -> Vec<Annotation> {
        vec![Annotation {
            bbox: Some([1.0, 1.0, 2.0, 2.0]),
            ..Default::default()
        }]
    }

    /// A box read before `fill_boxes` panics. Read as "no box" instead, it
    /// would score every pair zero and the run would look like a bad model.
    #[test]
    #[should_panic(expected = "fills the box column before reading it")]
    fn a_detection_box_read_before_the_fill_panics() {
        DetectionTable::from_records(&boxed()).bbox(0);
    }

    #[test]
    #[should_panic(expected = "fills the box column before reading it")]
    fn a_ground_truth_box_read_before_the_fill_panics() {
        GroundTruthTable::from_records(&boxed()).bbox(0);
    }

    /// Rows group by frame in the order given, ids renumber by rank, and a
    /// filter keeps each frame's run and renumbers over what is left.
    #[test]
    fn a_track_table_groups_stably_and_filters_into_a_renumbered_table() {
        let row = |frame, id, x| TrackRow {
            frame,
            id,
            xywh: [x, 0.0, 1.0, 1.0],
            class: 1,
            conf: 1.0,
        };
        let t = TrackTable::new(3, &[row(2, 40, 0.0), row(0, 7, 1.0), row(2, 7, 2.0)]);
        assert_eq!((t.len(), t.num_frames(), t.num_tracks()), (3, 3, 2));
        assert_eq!((t.frame(0), t.frame(1), t.frame(2)), (0..1, 1..1, 1..3));
        assert_eq!(t.tracks(0..3), &[0, 1, 0]);
        assert_eq!(t.boxes(1..3)[0][0], 0.0, "frame 2 keeps the order given");
        assert_eq!(t.track_lengths(), vec![2, 1]);

        let f = t.filter(&[true, false, true]);
        assert_eq!((f.frame(0), f.frame(2)), (0..1, 1..2));
        assert_eq!((f.num_tracks(), f.id_of(0)), (1, 7));
        assert_eq!(f.boxes(1..2)[0][0], 2.0);
    }
}
