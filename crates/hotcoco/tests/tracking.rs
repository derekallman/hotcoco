//! Tracking parity: hotcoco vs TrackEval on frozen synthetic sequences.
//!
//! `tests/fixtures/trackeval_expected.json` (repo root) holds MOTChallenge
//! rows and what TrackEval `12c8791` computed for them under numpy 1.23.5 and
//! scipy 1.15.3; `scripts/gen_tracking_fixtures.py` regenerates it. Every
//! field TrackEval reports — HOTA, CLEAR, Identity, and Count, per sequence
//! and combined — is compared: integers exactly, floats to the bit.
//!
//! The fixture's `error_cases` are inputs TrackEval refuses; each must be an
//! `InvalidInput` here, except the two where hotcoco accepts on purpose
//! (see [`DELIBERATELY_ACCEPTED`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hotcoco::Provenance;
use hotcoco::error::Error;
use hotcoco::metrics::tracking::hota_alphas;
use hotcoco::tracking::{Benchmark, METRIC_NAMES, TrackingEval, TrackingResult};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
struct Fixture {
    trackeval_commit: String,
    alphas: Vec<f64>,
    eps: f64,
    cases: Vec<Case>,
    error_cases: Vec<ErrorCase>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    benchmark: Option<String>,
    sequences: Vec<Sequence>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Sequence {
    name: String,
    num_frames: usize,
    gt: Vec<Vec<f64>>,
    pred: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Expected {
    combined: Map<String, Value>,
    per_sequence: BTreeMap<String, Map<String, Value>>,
}

#[derive(Deserialize)]
struct ErrorCase {
    name: String,
    benchmark: Option<String>,
    sequences: Vec<Sequence>,
    trackeval_error: TrackEvalError,
}

#[derive(Deserialize)]
struct TrackEvalError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

/// TrackEval raises on these and hotcoco does not, by decision: without a
/// benchmark the class column is not read, and a prediction's confidence
/// column is never read, so 6 columns are enough.
const DELIBERATELY_ACCEPTED: &[&str] = &["tracker_class_2_no_preproc", "tracker_six_columns"];

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/trackeval_expected.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture file"))
        .expect("fixture JSON")
}

fn benchmark(name: Option<&str>) -> Option<Benchmark> {
    name.map(|b| b.parse().expect("a fixture benchmark"))
}

/// One side's rows, flat, with their column count.
fn flat(rows: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let ncols = rows.first().map_or(0, Vec::len);
    assert!(rows.iter().all(|r| r.len() == ncols), "ragged fixture rows");
    (rows.iter().flatten().copied().collect(), ncols)
}

fn add(ev: &mut TrackingEval, s: &Sequence) -> hotcoco::error::Result<()> {
    let (gt, gt_ncols) = flat(&s.gt);
    let (pred, pred_ncols) = flat(&s.pred);
    ev.update(&s.name, &gt, gt_ncols, &pred, pred_ncols, s.num_frames)
}

fn evaluator(benchmark_name: Option<&str>, sequences: &[Sequence]) -> TrackingEval {
    let mut ev =
        TrackingEval::new(benchmark(benchmark_name), 0.5, 0.5).expect("default thresholds");
    for s in sequences {
        add(&mut ev, s).unwrap_or_else(|e| panic!("sequence {}: {e}", s.name));
    }
    ev
}

/// TrackEval's integer fields; every other field is a float.
fn is_integer_field(name: &str) -> bool {
    matches!(
        name,
        "HOTA_TP"
            | "HOTA_FN"
            | "HOTA_FP"
            | "CLR_TP"
            | "CLR_FN"
            | "CLR_FP"
            | "IDSW"
            | "MT"
            | "PT"
            | "ML"
            | "Frag"
            | "CLR_Frames"
            | "IDTP"
            | "IDFN"
            | "IDFP"
            | "Dets"
            | "GT_Dets"
            | "IDs"
            | "GT_IDs"
            | "Frames"
    )
}

/// What one comparison found: the floats that were not bit-identical, with
/// their largest deviation, and the failures.
#[derive(Default)]
struct Tally {
    floats: usize,
    not_bit_exact: Vec<String>,
    failures: Vec<String>,
}

impl Tally {
    fn compare(&mut self, at: &str, field: &str, got: &Value, want: &Value) {
        match (got, want) {
            (Value::Array(g), Value::Array(w)) => {
                if g.len() != w.len() {
                    self.failures.push(format!(
                        "{at} {field}: {} values, expected {}",
                        g.len(),
                        w.len()
                    ));
                    return;
                }
                for (k, (g, w)) in g.iter().zip(w).enumerate() {
                    self.compare(at, &format!("{field}[{k}]"), g, w);
                }
            }
            _ if is_integer_field(field.split('[').next().unwrap_or(field)) => {
                if got.as_u64().is_none() || got.as_u64() != want.as_u64() {
                    self.failures
                        .push(format!("{at} {field}: {got}, expected {want} (integer)"));
                }
            }
            _ => {
                let (Some(g), Some(w)) = (got.as_f64(), want.as_f64()) else {
                    self.failures.push(format!("{at} {field}: {got} vs {want}"));
                    return;
                };
                self.floats += 1;
                if g.to_bits() == w.to_bits() {
                    return;
                }
                let diff = (g - w).abs();
                if diff <= 1e-9 {
                    self.not_bit_exact
                        .push(format!("{at} {field}: {g:e} vs {w:e} (diff {diff:.1e})"));
                } else {
                    self.failures
                        .push(format!("{at} {field}: {g}, expected {w} (diff {diff:.3e})"));
                }
            }
        }
    }

    /// Every expected field against ours, and no field of ours left over.
    fn compare_result(&mut self, at: &str, got: &TrackingResult, want: &Map<String, Value>) {
        let got = serde_json::to_value(got).expect("serialize");
        let got = got.as_object().expect("a flat object");
        for (field, w) in want {
            match got.get(field) {
                Some(g) => self.compare(at, field, g, w),
                None => self.failures.push(format!("{at}: no field {field}")),
            }
        }
        for field in got.keys().filter(|f| !want.contains_key(*f)) {
            self.failures
                .push(format!("{at}: field {field} is not one TrackEval reports"));
        }
    }
}

/// Every case, every sequence, every field: integers exactly and floats to
/// the bit, against TrackEval.
#[test]
fn matches_trackeval_on_every_field() {
    let fixture = fixture();
    assert!(fixture.trackeval_commit.starts_with("12c8791"));
    assert!(fixture.cases.len() >= 30, "fixture looks truncated");
    let mut tally = Tally::default();
    for case in &fixture.cases {
        let ev = evaluator(case.benchmark.as_deref(), &case.sequences);
        tally.compare_result(
            &format!("{} combined", case.name),
            &ev.results().expect("sequences added"),
            &case.expected.combined,
        );
        assert_eq!(
            ev.sequence_results().len(),
            case.expected.per_sequence.len(),
            "{}",
            case.name
        );
        for (seq, want) in &case.expected.per_sequence {
            let got = &ev.sequence_results()[seq];
            tally.compare_result(&format!("{} {seq}", case.name), got, want);
        }
    }
    assert!(
        tally.failures.is_empty(),
        "{} mismatches against TrackEval:\n  {}",
        tally.failures.len(),
        tally.failures.join("\n  ")
    );
    assert!(
        tally.not_bit_exact.is_empty(),
        "{} of {} floats are within 1e-9 but not bit-identical:\n  {}",
        tally.not_bit_exact.len(),
        tally.floats,
        tally.not_bit_exact.join("\n  ")
    );
    println!("{} floats bit-identical to TrackEval", tally.floats);
}

#[test]
fn alphas_and_tolerance_are_trackevals() {
    let fixture = fixture();
    let ours: Vec<u64> = hota_alphas().iter().map(|a| a.to_bits()).collect();
    let theirs: Vec<u64> = fixture.alphas.iter().map(|a| a.to_bits()).collect();
    assert_eq!(ours, theirs);
    assert_eq!(fixture.eps, f64::EPSILON);
}

/// Every input TrackEval refuses is refused here, as bad input; the two
/// hotcoco accepts on purpose are accepted.
#[test]
fn refuses_what_trackeval_refuses() {
    let fixture = fixture();
    assert!(!fixture.error_cases.is_empty());
    for case in &fixture.error_cases {
        let mut ev = TrackingEval::new(benchmark(case.benchmark.as_deref()), 0.5, 0.5)
            .expect("default thresholds");
        let outcome = case.sequences.iter().try_for_each(|s| add(&mut ev, s));
        let te = &case.trackeval_error;
        if DELIBERATELY_ACCEPTED.contains(&case.name.as_str()) {
            assert!(
                outcome.is_ok(),
                "{}: accepted on purpose, got {outcome:?} (TrackEval: {} {})",
                case.name,
                te.kind,
                te.message
            );
            continue;
        }
        match outcome {
            Err(Error::InvalidInput(msg)) => {
                assert!(
                    msg.contains("seq-a"),
                    "{}: {msg} does not name the sequence",
                    case.name
                );
            }
            other => panic!(
                "{}: expected InvalidInput as TrackEval raises {} ({}), got {other:?}",
                case.name, te.kind, te.message
            ),
        }
    }
    for name in DELIBERATELY_ACCEPTED {
        assert!(
            fixture.error_cases.iter().any(|c| c.name == *name),
            "{name} is no longer in the fixture; drop it from DELIBERATELY_ACCEPTED"
        );
    }
}

fn case<'a>(fixture: &'a Fixture, name: &str) -> &'a Case {
    fixture
        .cases
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no fixture case {name}"))
}

/// Two evaluators over halves of the sequences merge into the whole run,
/// to the bit.
#[test]
fn shards_merge_to_the_whole() {
    let fixture = fixture();
    let c = case(&fixture, "multi_sequence");
    assert!(c.sequences.len() >= 2);
    let whole = evaluator(c.benchmark.as_deref(), &c.sequences);
    let (a, b) = c.sequences.split_at(c.sequences.len() / 2);
    let mut left = evaluator(c.benchmark.as_deref(), b);
    left.merge(&evaluator(c.benchmark.as_deref(), a))
        .expect("disjoint shards");
    assert_eq!(left, whole);
    assert_eq!(left.results().expect("run"), whole.results().expect("run"));
}

#[test]
fn merge_refuses_another_configuration_or_a_shared_sequence() {
    let fixture = fixture();
    let c = case(&fixture, "multi_sequence");
    let mut a = evaluator(c.benchmark.as_deref(), &c.sequences[..1]);
    let other = evaluator(Some("MOT20"), &c.sequences[1..2]);
    assert!(matches!(a.merge(&other), Err(Error::InvalidInput(_))));
    let same = evaluator(c.benchmark.as_deref(), &c.sequences[..1]);
    let Err(Error::InvalidInput(msg)) = a.merge(&same) else {
        panic!("a shared sequence must be refused");
    };
    assert!(msg.contains(&c.sequences[0].name), "{msg}");
    assert_eq!(a.sequence_results().len(), 1, "nothing added on an error");
}

/// An evaluator survives serialization, as pickling needs.
#[test]
fn an_evaluator_round_trips_through_json() {
    let fixture = fixture();
    let c = case(&fixture, "random_mot17");
    let ev = evaluator(c.benchmark.as_deref(), &c.sequences);
    let back: TrackingEval =
        serde_json::from_str(&serde_json::to_string(&ev).expect("serialize")).expect("deserialize");
    assert_eq!(back, ev);
    assert_eq!(back.stats().expect("run"), ev.stats().expect("run"));
}

#[test]
fn report_has_the_contract_shape() {
    let fixture = fixture();
    let c = case(&fixture, "multi_sequence");
    let ev = evaluator(c.benchmark.as_deref(), &c.sequences);
    let report = ev.report().expect("run");
    assert_eq!(report.task, "tracking");
    assert_eq!(report.provenance, Provenance::ParityVerified);
    assert_eq!(report.metrics.len(), METRIC_NAMES.len());
    // The headline HOTA is the mean of the 19 per-threshold values.
    let combined = ev.results().expect("run");
    let mean = combined.hota.hota.iter().sum::<f64>() / 19.0;
    assert!((report.metric("HOTA").expect("HOTA") - mean).abs() < 1e-12);
    assert_eq!(
        report.metric("IDSW"),
        Some(combined.clear.counts.idsw as f64)
    );
    assert_eq!(report.curves["alphas"].len(), 19);
    for (name, values) in &report.curves {
        assert_eq!(values.len(), 19, "curve {name}");
    }
    assert_eq!(report.per_group.len(), c.sequences.len());
    assert_eq!(
        report.params["benchmark"],
        c.benchmark.as_deref().map_or(Value::Null, Value::from)
    );
    let lines = ev.summarize_lines();
    assert!(lines.iter().any(|l| l.starts_with("COMBINED")));
    for s in &c.sequences {
        assert!(
            lines.iter().any(|l| l.starts_with(&s.name)),
            "no row for {}",
            s.name
        );
    }
}

/// The folder layout TrackEval reads gives what the arrays give.
#[test]
fn folders_score_as_the_arrays_do() {
    let fixture = fixture();
    let c = case(&fixture, "multi_sequence");
    let dir = tempfile::tempdir().expect("temp dir");
    let (gt_root, pred_root) = (dir.path().join("gt"), dir.path().join("pred"));
    std::fs::create_dir_all(&pred_root).expect("pred dir");
    let text = |rows: &[Vec<f64>]| -> String {
        rows.iter()
            .map(|r| r.iter().map(f64::to_string).collect::<Vec<_>>().join(",") + "\n")
            .collect()
    };
    for s in &c.sequences {
        let seq = gt_root.join(&s.name);
        std::fs::create_dir_all(seq.join("gt")).expect("seq dir");
        std::fs::write(seq.join("gt").join("gt.txt"), text(&s.gt)).expect("gt.txt");
        std::fs::write(
            seq.join("seqinfo.ini"),
            format!(
                "[Sequence]\nname={}\nseqLength = {}\nframeRate=30\n",
                s.name, s.num_frames
            ),
        )
        .expect("seqinfo");
        std::fs::write(pred_root.join(format!("{}.txt", s.name)), text(&s.pred)).expect("pred");
    }
    let arrays = evaluator(c.benchmark.as_deref(), &c.sequences);
    let mut files = TrackingEval::new(benchmark(c.benchmark.as_deref()), 0.5, 0.5).expect("new");
    assert_eq!(
        files
            .update_folders(&gt_root, &pred_root, None)
            .expect("folders"),
        c.sequences.len()
    );
    assert_eq!(files, arrays);

    // A seqmap picks sequences by name, after a header line.
    let seqmap = dir.path().join("seqmap.txt");
    std::fs::write(&seqmap, format!("name\n{}\n\n", c.sequences[0].name)).expect("seqmap");
    let mut one = TrackingEval::new(benchmark(c.benchmark.as_deref()), 0.5, 0.5).expect("new");
    assert_eq!(
        one.update_folders(&gt_root, &pred_root, Some(&seqmap))
            .expect("seqmap"),
        1
    );

    // A sequence already evaluated is refused, and nothing is added.
    assert!(matches!(
        one.update_folders(&gt_root, &pred_root, None),
        Err(Error::InvalidInput(_))
    ));
    assert_eq!(one.sequence_results().len(), 1);
}

/// A missing file is an I/O error that keeps its kind and names the path.
#[test]
fn a_missing_folder_or_file_is_not_found_naming_the_path() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut ev = TrackingEval::new(None, 0.5, 0.5).expect("new");
    let missing = dir.path().join("nowhere");
    let expect_not_found = |r: hotcoco::error::Result<usize>, path: &Path| match r {
        Err(Error::Io(e)) => {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
            assert!(e.to_string().contains(&path.display().to_string()), "{e}");
        }
        other => panic!("expected NotFound for {}, got {other:?}", path.display()),
    };
    expect_not_found(ev.update_folders(&missing, dir.path(), None), &missing);

    // A sequence folder whose prediction file is missing.
    let seq = dir.path().join("gt").join("s1");
    std::fs::create_dir_all(seq.join("gt")).expect("dirs");
    std::fs::write(seq.join("gt").join("gt.txt"), "1,1,0,0,5,5,1,1,1\n").expect("gt");
    std::fs::write(seq.join("seqinfo.ini"), "[Sequence]\nseqLength=1\n").expect("ini");
    let pred_root = dir.path().join("pred");
    std::fs::create_dir_all(&pred_root).expect("pred dir");
    expect_not_found(
        ev.update_folders(&dir.path().join("gt"), &pred_root, None),
        &pred_root.join("s1.txt"),
    );
    let seqmap = dir.path().join("missing-seqmap.txt");
    expect_not_found(
        ev.update_folders(&dir.path().join("gt"), &pred_root, Some(&seqmap)),
        &seqmap,
    );
    assert!(ev.sequence_results().is_empty());
}

#[test]
fn a_malformed_text_file_is_bad_input_naming_the_line() {
    let dir = tempfile::tempdir().expect("temp dir");
    let seq = dir.path().join("gt").join("s1");
    std::fs::create_dir_all(seq.join("gt")).expect("dirs");
    std::fs::write(seq.join("seqinfo.ini"), "[Sequence]\nseqLength=2\n").expect("ini");
    std::fs::write(seq.join("gt").join("gt.txt"), "1,1,0,0,5,5,1,1,1\n").expect("gt");
    let pred_root = dir.path().join("pred");
    std::fs::create_dir_all(&pred_root).expect("pred dir");
    for (body, needle) in [
        ("1,1,0,0,5,5,1\n2,1,0,0,5\n", "line 2: 5 fields"),
        ("1,1,0,0,five,5,1\n", "\"five\" is not a number"),
    ] {
        std::fs::write(pred_root.join("s1.txt"), body).expect("pred");
        let mut ev = TrackingEval::new(Some(Benchmark::Mot17), 0.5, 0.5).expect("new");
        match ev.update_folders(&dir.path().join("gt"), &pred_root, None) {
            Err(Error::InvalidInput(m)) => assert!(m.contains(needle), "{m}"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
    // Spaces after commas, a trailing comma, and blank lines are read.
    std::fs::write(
        pred_root.join("s1.txt"),
        "1, 1, 0, 0, 5, 5, 0.9,\n\n2 1 0 0 5 5 0.9\n",
    )
    .expect("pred");
    let mut ev = TrackingEval::new(Some(Benchmark::Mot17), 0.5, 0.5).expect("new");
    ev.update_folders(&dir.path().join("gt"), &pred_root, None)
        .expect("lenient layout");
    assert_eq!(ev.results().expect("run").count.dets, 2);
    // No seqLength.
    std::fs::write(seq.join("seqinfo.ini"), "[Sequence]\nname=s1\n").expect("ini");
    let mut ev = TrackingEval::new(None, 0.5, 0.5).expect("new");
    assert!(matches!(
        ev.update_folders(&dir.path().join("gt"), &pred_root, None),
        Err(Error::InvalidInput(m)) if m.contains("seqLength")
    ));
}

// --- The proposal's error table, row by row, as far as the core raises it.

/// `frame, id, x, y, w, h, conf, class, visibility`.
fn row(frame: f64, id: f64, conf: f64, class: f64) -> [f64; 9] {
    [frame, id, 10.0, 10.0, 20.0, 40.0, conf, class, 1.0]
}

fn one(
    benchmark: Option<Benchmark>,
    gt: &[[f64; 9]],
    pred: &[[f64; 9]],
) -> hotcoco::error::Result<()> {
    let mut ev = TrackingEval::new(benchmark, 0.5, 0.5).expect("new");
    let g: Vec<f64> = gt.iter().flatten().copied().collect();
    let p: Vec<f64> = pred.iter().flatten().copied().collect();
    ev.update("seq-x", &g, 9, &p, 9, 3)
}

fn invalid(r: hotcoco::error::Result<()>) -> String {
    match r {
        Err(Error::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn the_same_id_twice_in_a_frame_names_sequence_frame_and_ids() {
    let m = invalid(one(
        None,
        &[row(1.0, 1.0, 1.0, 1.0)],
        &[row(2.0, 4.0, 1.0, 1.0), row(2.0, 4.0, 1.0, 1.0)],
    ));
    assert_eq!(
        m,
        "sequence seq-x, frame 2: prediction track id(s) 4 appear more than once"
    );
}

#[test]
fn a_frame_outside_the_sequence_is_bad_input() {
    for frame in [0.0, 4.0] {
        let m = invalid(one(None, &[row(frame, 1.0, 1.0, 1.0)], &[]));
        assert!(m.contains("seq-x") && m.contains("frame"), "{m}");
    }
}

#[test]
fn a_class_outside_the_benchmark_is_bad_input() {
    let b = Some(Benchmark::Mot17);
    assert!(
        invalid(one(
            b,
            &[row(1.0, 1.0, 1.0, 14.0)],
            &[row(1.0, 1.0, 1.0, 1.0)]
        ))
        .contains("class 14")
    );
    assert!(invalid(one(b, &[], &[row(1.0, 1.0, 1.0, 2.0)])).contains("prediction class 2"));
    assert!(one(None, &[], &[row(1.0, 1.0, 1.0, 2.0)]).is_ok());
}

#[test]
fn a_fractional_or_negative_id_or_frame_is_bad_input() {
    for bad in [
        row(1.5, 1.0, 1.0, 1.0),
        row(1.0, 2.5, 1.0, 1.0),
        row(1.0, -1.0, 1.0, 1.0),
        row(-1.0, 1.0, 1.0, 1.0),
    ] {
        invalid(one(None, &[bad], &[]));
    }
}

#[test]
fn nan_in_a_box_or_confidence_is_bad_input() {
    let mut nan_box = row(1.0, 1.0, 1.0, 1.0);
    nan_box[3] = f64::NAN;
    assert!(invalid(one(None, &[], &[nan_box])).contains("box value NaN"));
    assert!(invalid(one(None, &[row(1.0, 1.0, f64::NAN, 1.0)], &[])).contains("confidence NaN"));
}

#[test]
fn too_few_columns_or_a_partial_row_is_bad_input() {
    let mut ev = TrackingEval::new(Some(Benchmark::Mot17), 0.5, 0.5).expect("new");
    let seven = [1.0, 1.0, 0.0, 0.0, 5.0, 5.0, 1.0];
    assert!(invalid(ev.update("s", &seven, 7, &[], 6, 1)).contains("at least 8 columns"));
    assert!(invalid(ev.update("s", &[], 9, &seven[..5], 5, 1)).contains("at least 6 columns"));
    assert!(invalid(ev.update("s", &[], 9, &seven, 6, 1)).contains("not a whole number"));
    assert!(ev.sequence_results().is_empty(), "nothing kept on an error");
}

#[test]
fn an_unknown_benchmark_is_bad_input() {
    for name in ["MOT99", "", "mot17x"] {
        assert!(
            matches!(name.parse::<Benchmark>(), Err(Error::InvalidInput(_))),
            "{name}"
        );
    }
    assert_eq!(
        "mot17".parse::<Benchmark>().expect("any case"),
        Benchmark::Mot17
    );
    for b in [
        Benchmark::Mot15,
        Benchmark::Mot16,
        Benchmark::Mot17,
        Benchmark::Mot20,
    ] {
        assert_eq!(b.as_str().parse::<Benchmark>().expect("round trip"), b);
    }
}

#[test]
fn the_same_sequence_twice_is_bad_input() {
    let mut ev = TrackingEval::new(None, 0.5, 0.5).expect("new");
    let g = row(1.0, 1.0, 1.0, 1.0);
    ev.update("s", &g, 9, &g, 9, 1).expect("first");
    let m = invalid(ev.update("s", &g, 9, &g, 9, 1));
    assert!(m.contains("sequence s"), "{m}");
}

#[test]
fn results_before_any_update_is_invalid_state() {
    let ev = TrackingEval::new(None, 0.5, 0.5).expect("new");
    assert!(matches!(ev.results(), Err(Error::InvalidState(_))));
    assert!(matches!(ev.stats(), Err(Error::InvalidState(_))));
    assert!(matches!(ev.curves(), Err(Error::InvalidState(_))));
    assert!(matches!(ev.report(), Err(Error::InvalidState(_))));
    assert_eq!(ev.summarize_lines(), Vec::<String>::new());
}

#[test]
fn a_threshold_outside_0_to_1_is_bad_input() {
    for (clear, identity) in [(1.5, 0.5), (0.5, -0.1), (f64::NAN, 0.5)] {
        assert!(matches!(
            TrackingEval::new(None, clear, identity),
            Err(Error::InvalidInput(_))
        ));
    }
}

/// With no benchmark, a ground-truth row whose confidence is 0 is evaluated,
/// where TrackEval drops it in every mode: the one way a run departs from the
/// reference, and it says so.
#[test]
fn keeping_a_zero_confidence_row_is_a_stated_deviation() -> Result<(), Error> {
    let pred = [1.0, 5.0, 10.0, 10.0, 20.0, 20.0, 0.9];
    for (conf, deviates) in [(1.0, false), (0.0, true), (0.5, true)] {
        let gt = [1.0, 1.0, 10.0, 10.0, 20.0, 20.0, conf, 1.0];
        let mut ev = TrackingEval::new(None, 0.5, 0.5)?;
        ev.update("s", &gt, 8, &pred, 7, 1)?;
        assert_eq!(
            !ev.reference_deviations().is_empty(),
            deviates,
            "conf {conf}"
        );
        assert_eq!(ev.results()?.count.gt_dets, 1, "the row is kept");

        // A benchmark drops the row, as TrackEval does: nothing to state.
        let mut ev = TrackingEval::new(Some(Benchmark::Mot17), 0.5, 0.5)?;
        ev.update("s", &gt, 8, &pred, 7, 1)?;
        assert!(ev.reference_deviations().is_empty(), "conf {conf}");
    }
    // Six-column ground truth has no confidence to read.
    let mut ev = TrackingEval::new(None, 0.5, 0.5)?;
    ev.update("s", &[1.0, 1.0, 10.0, 10.0, 20.0, 20.0], 6, &pred, 7, 1)?;
    assert_eq!(ev.reference_deviations(), Vec::<String>::new());

    // And a shard that has one carries it through a merge.
    let mut clean = TrackingEval::new(None, 0.5, 0.5)?;
    clean.update(
        "a",
        &[1.0, 1.0, 10.0, 10.0, 20.0, 20.0, 1.0, 1.0],
        8,
        &pred,
        7,
        1,
    )?;
    let mut marked = TrackingEval::new(None, 0.5, 0.5)?;
    marked.update(
        "b",
        &[1.0, 1.0, 10.0, 10.0, 20.0, 20.0, 0.0, 1.0],
        8,
        &pred,
        7,
        1,
    )?;
    clean.merge(&marked)?;
    assert_eq!(clean.reference_deviations().len(), 1);
    Ok(())
}
