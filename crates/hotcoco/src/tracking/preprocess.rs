//! MOTChallenge rows in, cleaned tables out: the checks TrackEval's loader
//! makes, and the clean-up of `MotChallenge2DBox.get_preprocessed_seq_data`.
//!
//! Each rule is one block below, with one test of its own.
//!
//! - **Rows** ([`read_rows`]): `frame, id, x, y, w, h[, conf[, class, …]]`. A
//!   frame is a whole number in `1..=num_frames` and an id a whole number 0
//!   or more, both by [`id_from_f64`]; a box, a confidence, and a class read
//!   with a benchmark are finite. Ground truth needs 8 columns with a
//!   benchmark, 6 without; predictions need 6. A prediction row with fewer
//!   than 8 columns has class 1. A class is truncated toward zero, as
//!   TrackEval's `astype(int)` reads it.
//! - **Unique ids** ([`check_unique_ids`]): a track id at most once per frame
//!   on each side, checked on the raw rows. TrackEval checks again after
//!   clean-up, which only removes rows, so that check cannot fire and is not
//!   repeated.
//! - **Clean-up** ([`clean`]), per frame, with a benchmark only:
//!   1. a prediction class above 1 is an error, MOT15 included;
//!   2. not MOT15, and both sides have rows in the frame: a ground-truth
//!      class outside 1..13 is an error, and a prediction matched to a
//!      distractor ground truth is removed — matched by an assignment on
//!      the raw IoU, every class included, with IoU below `0.5 − ε` zeroed and
//!      pairs above `ε` kept;
//!   3. a ground-truth box is kept when its confidence, truncated toward
//!      zero, is not 0 (TrackEval's `zero_marked`) and, unless MOT15, its
//!      class is 1.

use crate::error::{Error, Result};
use crate::primitives::assign::lsap;
use crate::primitives::sim::{bbox_iou_corners_flat, corner_area};
use crate::table::{TrackRow, TrackTable};
use crate::types::id_from_f64;

use super::Benchmark;

/// `np.finfo(float).eps`, every one of TrackEval's tolerances.
pub(super) const EPS: f64 = f64::EPSILON;

/// The pedestrian class, the one MOTChallenge evaluates.
const PEDESTRIAN: i64 = 1;

/// The ground-truth classes MOTChallenge defines: 1 (pedestrian) to 13 (crowd).
const VALID_GT_CLASSES: std::ops::RangeInclusive<i64> = 1..=13;

/// Person on vehicle, static person, distractor, reflection.
const DISTRACTORS: [i64; 4] = [2, 7, 8, 12];

/// Non-motorized vehicle: a distractor in MOT20 only.
const NON_MOT_VEHICLE: i64 = 6;

/// Which side of a sequence a table holds, as messages name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Side {
    GroundTruth,
    Prediction,
}

impl Side {
    fn name(self) -> &'static str {
        match self {
            Side::GroundTruth => "ground truth",
            Side::Prediction => "prediction",
        }
    }
}

/// The table of one side's MOTChallenge rows: `values` row-major, `ncols`
/// to a row.
pub(super) fn read_rows(
    seq: &str,
    side: Side,
    values: &[f64],
    ncols: usize,
    num_frames: usize,
    benchmark: Option<Benchmark>,
) -> Result<TrackTable> {
    let bad = |msg: String| Error::InvalidInput(format!("sequence {seq}: {} {msg}", side.name()));
    if values.is_empty() {
        return table(seq, num_frames, &[]);
    }
    let min = match (side, benchmark) {
        (Side::GroundTruth, Some(_)) => 8,
        _ => 6,
    };
    if ncols < min {
        let columns = if min == 8 {
            "frame, id, x, y, w, h, conf, class"
        } else {
            "frame, id, x, y, w, h"
        };
        return Err(bad(format!(
            "rows need at least {min} columns ({columns}), got {ncols}"
        )));
    }
    if values.len() % ncols != 0 {
        return Err(bad(format!(
            "has {} values, which is not a whole number of {ncols}-column rows",
            values.len()
        )));
    }
    let reads_class = benchmark.is_some() && ncols > 7;
    let mut rows = Vec::with_capacity(values.len() / ncols);
    for (i, row) in values.chunks_exact(ncols).enumerate() {
        let at = |msg: String| bad(format!("row {i}: {msg}"));
        let frame = id_from_f64(row[0])
            .filter(|&f| f >= 1 && f <= num_frames as u64)
            .ok_or_else(|| {
                at(format!(
                    "frame {} is not a whole number from 1 to {num_frames}",
                    row[0]
                ))
            })?;
        let id = id_from_f64(row[1]).ok_or_else(|| {
            at(format!(
                "track id {} is not a whole number 0 or more",
                row[1]
            ))
        })?;
        let xywh = [row[2], row[3], row[4], row[5]];
        if let Some(v) = xywh.iter().find(|v| !v.is_finite()) {
            return Err(at(format!("box value {v} is not a finite number")));
        }
        // Finite values whose far edge or area is not, the area taken from
        // the corners as the IoU kernel takes it: two such boxes make an IoU
        // of inf / inf, NaN, which the assignment refuses. Refused here, as a
        // box that is not finite is; TrackEval raises on the NaN only.
        if !corner_area(xywh).is_finite() {
            let [x, y, w, h] = xywh;
            return Err(at(format!(
                "box [{x:?}, {y:?}, {w:?}, {h:?}] reaches past the largest float, so its IoU is not a number"
            )));
        }
        let conf = match row.get(6) {
            Some(&c) if !c.is_finite() => {
                return Err(at(format!("confidence {c} is not a finite number")));
            }
            Some(&c) => c,
            None => 1.0,
        };
        let class = if reads_class {
            let c = row[7];
            if !c.is_finite() {
                return Err(at(format!("class {c} is not a finite number")));
            }
            c.trunc() as i64
        } else {
            PEDESTRIAN
        };
        rows.push(TrackRow {
            frame: frame as usize - 1,
            id,
            xywh,
            class,
            conf,
        });
    }
    table(seq, num_frames, &rows)
}

/// [`TrackTable::new`], or the error for a frame count no memory holds: one
/// past `usize`, or one whose frame offsets the system will not allocate.
fn table(seq: &str, num_frames: usize, rows: &[TrackRow]) -> Result<TrackTable> {
    TrackTable::new(num_frames, rows).ok_or_else(|| {
        Error::InvalidInput(format!(
            "sequence {seq}: {num_frames} frames are more than memory can index"
        ))
    })
}

/// TrackEval's `_check_unique_ids`: an error naming the sequence, the frame,
/// and the ids, at the first frame where a track id repeats — predictions
/// checked before ground truth, as the reference checks them.
pub(super) fn check_unique_ids(seq: &str, gt: &TrackTable, pred: &TrackTable) -> Result<()> {
    let mut seen_gt = vec![usize::MAX; gt.num_tracks()];
    let mut seen_pred = vec![usize::MAX; pred.num_tracks()];
    for f in 0..gt.num_frames() {
        for (side, table, seen) in [
            (Side::Prediction, pred, &mut seen_pred),
            (Side::GroundTruth, gt, &mut seen_gt),
        ] {
            let mut repeated: Vec<u64> = Vec::new();
            for &t in table.tracks(table.frame(f)) {
                if seen[t] == f {
                    repeated.push(table.id_of(t));
                }
                seen[t] = f;
            }
            if !repeated.is_empty() {
                repeated.sort_unstable();
                repeated.dedup();
                let ids: Vec<String> = repeated.iter().map(u64::to_string).collect();
                return Err(Error::InvalidInput(format!(
                    "sequence {seq}, frame {}: {} track id(s) {} appear more than once",
                    f + 1,
                    side.name(),
                    ids.join(", ")
                )));
            }
        }
    }
    Ok(())
}

/// The tables TrackEval evaluates: `benchmark`'s clean-up applied to the raw
/// ones, or the raw ones unchanged without a benchmark.
pub(super) fn clean(
    seq: &str,
    benchmark: Option<Benchmark>,
    gt: TrackTable,
    pred: TrackTable,
) -> Result<(TrackTable, TrackTable)> {
    let Some(benchmark) = benchmark else {
        return Ok((gt, pred));
    };
    let mut keep_gt = vec![true; gt.len()];
    let mut keep_pred = vec![true; pred.len()];
    for f in 0..gt.num_frames() {
        let (g, p) = (gt.frame(f), pred.frame(f));
        let at =
            |msg: String| Error::InvalidInput(format!("sequence {seq}, frame {}: {msg}", f + 1));

        // Rule 1: MOTChallenge evaluates pedestrians only.
        if let Some(c) = p
            .clone()
            .map(|i| pred.class(i))
            .filter(|&c| c > PEDESTRIAN)
            .max()
        {
            return Err(at(format!(
                "prediction class {c}; MOTChallenge evaluates class 1 (pedestrian) only"
            )));
        }

        // Rule 2: predictions on distractors are removed.
        if benchmark != Benchmark::Mot15 && !g.is_empty() && !p.is_empty() {
            if let Some(c) = g
                .clone()
                .map(|i| gt.class(i))
                .find(|c| !VALID_GT_CLASSES.contains(c))
            {
                return Err(at(format!(
                    "ground-truth class {c} is not a MOTChallenge class (1 to 13)"
                )));
            }
            let (nr, nc) = (g.len(), p.len());
            let mut score = bbox_iou_corners_flat(gt.boxes(g.clone()), pred.boxes(p.clone()));
            for s in &mut score {
                if *s < 0.5 - EPS {
                    *s = 0.0;
                }
            }
            let (rows, cols) = lsap(&score, nr, nc, true);
            for (r, c) in rows.into_iter().zip(cols) {
                if score[r * nc + c] > EPS && benchmark.is_distractor(gt.class(g.start + r)) {
                    keep_pred[p.start + c] = false;
                }
            }
        }

        // Rule 3: ground truth that is not evaluated.
        for i in g {
            let zero_marked = gt.conf(i).trunc() == 0.0;
            keep_gt[i] =
                !zero_marked && (benchmark == Benchmark::Mot15 || gt.class(i) == PEDESTRIAN);
        }
    }
    Ok((gt.filter(&keep_gt), pred.filter(&keep_pred)))
}

impl Benchmark {
    /// Whether a ground truth of `class` takes the prediction matched to it
    /// out of the evaluation.
    fn is_distractor(self, class: i64) -> bool {
        DISTRACTORS.contains(&class) || (self == Benchmark::Mot20 && class == NON_MOT_VEHICLE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `frame, id, x, y, w, h, conf, class, visibility` rows.
    fn rows(r: &[[f64; 9]]) -> Vec<f64> {
        r.iter().flatten().copied().collect()
    }

    fn table(side: Side, r: &[[f64; 9]], frames: usize, b: Option<Benchmark>) -> TrackTable {
        read_rows("s", side, &rows(r), 9, frames, b).expect("valid rows")
    }

    fn err(r: Result<impl std::fmt::Debug>) -> String {
        match r {
            Err(Error::InvalidInput(m)) => m,
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    /// The ids left in frame `f`, in row order.
    fn ids(t: &TrackTable, f: usize) -> Vec<u64> {
        t.tracks(t.frame(f)).iter().map(|&k| t.id_of(k)).collect()
    }

    const B: [f64; 4] = [10.0, 10.0, 20.0, 40.0];

    fn row(frame: f64, id: f64, b: [f64; 4], conf: f64, class: f64) -> [f64; 9] {
        [frame, id, b[0], b[1], b[2], b[3], conf, class, 1.0]
    }

    #[test]
    fn rows_group_by_frame_and_keep_their_order() {
        let t = table(
            Side::Prediction,
            &[
                row(2.0, 9.0, B, 1.0, -1.0),
                row(1.0, 4.0, B, 1.0, -1.0),
                row(2.0, 3.0, B, 1.0, -1.0),
            ],
            3,
            None,
        );
        assert_eq!(
            (ids(&t, 0), ids(&t, 1), ids(&t, 2)),
            (vec![4], vec![9, 3], vec![])
        );
        // Renumbered in ascending id order: 3 -> 0, 4 -> 1, 9 -> 2.
        assert_eq!(t.tracks(t.frame(1)), &[2, 0]);
    }

    #[test]
    fn a_frame_outside_the_sequence_is_an_error() {
        for frame in [0.0, 4.0, 1.5, -1.0, f64::NAN] {
            let m = err(read_rows(
                "s",
                Side::GroundTruth,
                &rows(&[row(frame, 1.0, B, 1.0, 1.0)]),
                9,
                3,
                None,
            ));
            assert!(m.contains("sequence s: ground truth row 0: frame"), "{m}");
        }
    }

    #[test]
    fn a_fractional_or_negative_id_is_an_error() {
        for id in [1.5, -1.0] {
            let m = err(read_rows(
                "s",
                Side::Prediction,
                &rows(&[row(1.0, id, B, 1.0, 1.0)]),
                9,
                3,
                None,
            ));
            assert!(m.contains("track id"), "{m}");
        }
    }

    #[test]
    fn a_box_or_confidence_that_is_not_finite_is_an_error() {
        let nan_box = row(1.0, 1.0, [f64::NAN, 0.0, 1.0, 1.0], 1.0, 1.0);
        assert!(
            err(read_rows(
                "s",
                Side::Prediction,
                &rows(&[nan_box]),
                9,
                1,
                None
            ))
            .contains("box value NaN")
        );
        let nan_conf = row(1.0, 1.0, B, f64::NAN, 1.0);
        assert!(
            err(read_rows(
                "s",
                Side::Prediction,
                &rows(&[nan_conf]),
                9,
                1,
                None
            ))
            .contains("confidence NaN")
        );
    }

    #[test]
    fn a_box_past_the_largest_float_is_an_error() {
        for b in [
            [1e308, 0.0, 1e308, 1.0],
            [0.0, -1e308, 1.0, -1e308],
            [0.0, 0.0, 1e200, 1e200],
            // `x + w` and `w * h` are finite, but `(x + w) - x` rounds up an
            // ulp and the corner area the IoU kernel divides by is not: the
            // NaN reached the assignment, which panicked.
            [-2.040552280408471e307, 0.0, 8.988465674311579e307, 2.0],
            [-3.0 * 2f64.powi(970), 0.0, f64::MAX, 1.0],
        ] {
            let m = err(read_rows(
                "s",
                Side::GroundTruth,
                &rows(&[row(1.0, 1.0, b, 1.0, 1.0)]),
                9,
                1,
                None,
            ));
            assert!(m.contains("past the largest float"), "{b:?}: {m}");
        }
        // `w * h` overflows, but `x + w` rounds back to `x`: the corner area
        // is 0, an IoU of 0 with any box, as TrackEval scores it.
        for big in [[1e300, 1e300, 1e7, 1e7], [1e300, 0.0, 1e200, 1e200]] {
            assert!(
                read_rows(
                    "s",
                    Side::GroundTruth,
                    &rows(&[row(1.0, 1.0, big, 1.0, 1.0)]),
                    9,
                    1,
                    None
                )
                .is_ok(),
                "{big:?}"
            );
        }
    }

    #[test]
    fn ground_truth_needs_eight_columns_with_a_benchmark_and_six_without() {
        let six = [1.0, 1.0, 0.0, 0.0, 5.0, 5.0];
        let m = err(read_rows(
            "s",
            Side::GroundTruth,
            &six,
            6,
            1,
            Some(Benchmark::Mot17),
        ));
        assert!(m.contains("at least 8 columns"), "{m}");
        assert!(read_rows("s", Side::GroundTruth, &six, 6, 1, None).is_ok());
        assert!(read_rows("s", Side::Prediction, &six, 6, 1, Some(Benchmark::Mot17)).is_ok());
        assert!(
            err(read_rows("s", Side::Prediction, &six[..5], 5, 1, None)).contains("at least 6")
        );
        assert!(
            err(read_rows("s", Side::Prediction, &six[..5], 6, 1, None))
                .contains("not a whole number")
        );
    }

    /// TrackEval gives a prediction file of fewer than 8 columns class 1, so
    /// a benchmark's class check passes it; column 7 of a wider one is read.
    #[test]
    fn a_prediction_without_a_class_column_is_a_pedestrian() {
        let seven = [1.0, 1.0, 0.0, 0.0, 5.0, 5.0, 0.9];
        let t =
            read_rows("s", Side::Prediction, &seven, 7, 1, Some(Benchmark::Mot17)).expect("valid");
        assert_eq!(t.class(0), PEDESTRIAN);
        let gt = table(Side::GroundTruth, &[], 1, Some(Benchmark::Mot17));
        assert!(clean("s", Some(Benchmark::Mot17), gt, t).is_ok());
    }

    #[test]
    fn a_repeated_id_in_a_frame_is_an_error_naming_sequence_frame_and_ids() {
        let gt = table(
            Side::GroundTruth,
            &[row(2.0, 5.0, B, 1.0, 1.0), row(2.0, 5.0, B, 1.0, 1.0)],
            2,
            None,
        );
        let pred = table(
            Side::Prediction,
            &[row(1.0, 5.0, B, 1.0, 1.0), row(2.0, 5.0, B, 1.0, 1.0)],
            2,
            None,
        );
        let m = err(check_unique_ids("seq-a", &gt, &pred));
        assert_eq!(
            m,
            "sequence seq-a, frame 2: ground truth track id(s) 5 appear more than once"
        );
        // The same id in two frames is a track, not a repeat.
        let fine = table(Side::GroundTruth, &[row(1.0, 5.0, B, 1.0, 1.0)], 2, None);
        assert!(check_unique_ids("s", &fine, &pred).is_ok());
    }

    /// Rule 1, for MOT15 too, and only with a benchmark. `-1`, what tracker
    /// files write, is fine.
    #[test]
    fn rule_1_a_prediction_class_above_1_is_an_error() {
        for b in [Benchmark::Mot15, Benchmark::Mot17] {
            let gt = table(Side::GroundTruth, &[], 1, Some(b));
            let pred = table(Side::Prediction, &[row(1.0, 1.0, B, 1.0, 2.0)], 1, Some(b));
            assert!(err(clean("s", Some(b), gt, pred)).contains("prediction class 2"));
            let gt = table(Side::GroundTruth, &[], 1, Some(b));
            let ok = table(Side::Prediction, &[row(1.0, 1.0, B, 1.0, -1.0)], 1, Some(b));
            assert!(clean("s", Some(b), gt, ok).is_ok());
        }
        let gt = table(Side::GroundTruth, &[], 1, None);
        let pred = table(Side::Prediction, &[row(1.0, 1.0, B, 1.0, 2.0)], 1, None);
        assert!(clean("s", None, gt, pred).is_ok());
    }

    /// Rule 2a: only in a frame both sides have rows in, and not for MOT15.
    #[test]
    fn rule_2_a_ground_truth_class_outside_1_to_13_is_an_error_where_both_sides_have_rows() {
        let bad_gt = [row(1.0, 1.0, B, 1.0, 14.0), row(2.0, 1.0, B, 1.0, 14.0)];
        let pred = [row(2.0, 1.0, B, 1.0, 1.0)];
        let m = err(clean(
            "s",
            Some(Benchmark::Mot17),
            table(Side::GroundTruth, &bad_gt, 2, Some(Benchmark::Mot17)),
            table(Side::Prediction, &pred, 2, Some(Benchmark::Mot17)),
        ));
        // Frame 1 has no prediction, so the class is first checked in frame 2.
        assert!(
            m.starts_with("sequence s, frame 2: ground-truth class 14"),
            "{m}"
        );
        assert!(
            clean(
                "s",
                Some(Benchmark::Mot15),
                table(Side::GroundTruth, &bad_gt, 2, Some(Benchmark::Mot15)),
                table(Side::Prediction, &pred, 2, Some(Benchmark::Mot15)),
            )
            .is_ok()
        );
    }

    /// Rule 2b: a prediction matched (IoU at least 0.5) to a distractor is
    /// removed; one under 0.5 stays, and class 6 is a distractor in MOT20 only.
    #[test]
    fn rule_2_a_prediction_matched_to_a_distractor_is_removed() {
        let far = [200.0, 200.0, 20.0, 40.0];
        let near_miss = [10.0, 30.0, 20.0, 40.0]; // IoU 1/3 with B
        let cleaned = |b: Benchmark, class: f64, pbox: [f64; 4]| {
            let gt = table(
                Side::GroundTruth,
                &[row(1.0, 1.0, B, 1.0, class), row(1.0, 2.0, far, 1.0, 1.0)],
                1,
                Some(b),
            );
            let pred = table(
                Side::Prediction,
                &[row(1.0, 7.0, pbox, 0.9, -1.0)],
                1,
                Some(b),
            );
            let (_, pred) = clean("s", Some(b), gt, pred).expect("clean");
            pred.len()
        };
        for distractor in [2.0, 7.0, 8.0, 12.0] {
            assert_eq!(
                cleaned(Benchmark::Mot17, distractor, B),
                0,
                "class {distractor}"
            );
            assert_eq!(
                cleaned(Benchmark::Mot17, distractor, near_miss),
                1,
                "class {distractor}"
            );
            assert_eq!(
                cleaned(Benchmark::Mot15, distractor, B),
                1,
                "MOT15, class {distractor}"
            );
        }
        assert_eq!(cleaned(Benchmark::Mot17, 1.0, B), 1);
        assert_eq!(cleaned(Benchmark::Mot17, 6.0, B), 1);
        assert_eq!(cleaned(Benchmark::Mot20, 6.0, B), 0);
    }

    /// Rule 2b matches before the ground truth is filtered: a zero-marked
    /// pedestrian still takes the prediction it overlaps, so a distractor
    /// behind it keeps its prediction instead of removing it.
    #[test]
    fn rule_2_matching_sees_every_ground_truth_row() {
        let b = Benchmark::Mot17;
        let gt = table(
            Side::GroundTruth,
            &[
                row(1.0, 1.0, B, 0.0, 1.0),
                row(1.0, 2.0, [12.0, 10.0, 20.0, 40.0], 1.0, 8.0),
            ],
            1,
            Some(b),
        );
        let pred = table(Side::Prediction, &[row(1.0, 7.0, B, 0.9, -1.0)], 1, Some(b));
        let (gt, pred) = clean("s", Some(b), gt, pred).expect("clean");
        assert_eq!((gt.len(), pred.len()), (0, 1));
    }

    /// Rule 3: a ground truth is evaluated when its confidence truncates to
    /// a non-zero value and, unless MOT15, it is a pedestrian.
    #[test]
    fn rule_3_ground_truth_kept_when_not_zero_marked_and_a_pedestrian() {
        let gt_rows = [
            row(1.0, 1.0, B, 1.0, 1.0),
            row(1.0, 2.0, B, 0.0, 1.0),
            row(1.0, 3.0, B, 0.5, 1.0), // truncates to 0
            row(1.0, 4.0, B, -1.0, 1.0),
            row(1.0, 5.0, B, 1.0, 7.0),
        ];
        let kept = |b: Option<Benchmark>| {
            let gt = table(Side::GroundTruth, &gt_rows, 1, b);
            let pred = table(Side::Prediction, &[], 1, b);
            let (gt, _) = clean("s", b, gt, pred).expect("clean");
            ids(&gt, 0)
        };
        assert_eq!(kept(Some(Benchmark::Mot17)), vec![1, 4]);
        assert_eq!(kept(Some(Benchmark::Mot15)), vec![1, 4, 5]);
        assert_eq!(kept(None), vec![1, 2, 3, 4, 5]);
    }

    /// Clean-up renumbers over the tracks that are left.
    #[test]
    fn a_cleaned_table_is_renumbered() {
        let b = Some(Benchmark::Mot17);
        let gt = table(
            Side::GroundTruth,
            &[row(1.0, 3.0, B, 0.0, 1.0), row(1.0, 8.0, B, 1.0, 1.0)],
            1,
            b,
        );
        let pred = table(Side::Prediction, &[], 1, b);
        let (gt, _) = clean("s", b, gt, pred).expect("clean");
        assert_eq!(
            (gt.num_tracks(), gt.id_of(0), gt.tracks(gt.frame(0))),
            (1, 8, &[0][..])
        );
    }
}
