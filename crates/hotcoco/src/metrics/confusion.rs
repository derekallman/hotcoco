//! Confusion counts over matched prediction/ground-truth pairs.
//!
//! `sklearn.metrics.confusion_matrix(y_true, y_pred)` assumes every sample has
//! both a true and a predicted label. Detection, tracking, and panoptic don't:
//! a prediction can match nothing (a false positive) and a ground truth can go
//! unpredicted (a false negative). So these functions take **`Option` labels**,
//! and reserve one extra row and column — index `num_classes` — for background.
//!
//! ```
//! use hotcoco::metrics::confusion::confusion_matrix;
//!
//! // Three classes. One correct `0`, one `1` predicted as `2`, one missed `1`,
//! // and one spurious `0`.
//! let gt  = [Some(0), Some(1), Some(1), None];
//! let dt  = [Some(0), Some(2), None,    Some(0)];
//! let m = confusion_matrix(&gt, &dt, 3);
//!
//! let at = |gt_class: usize, dt_class: usize| m[gt_class * 4 + dt_class]; // side = 3 + 1
//! assert_eq!(at(0, 0), 1);   // class 0 predicted correctly
//! assert_eq!(at(1, 2), 1);   // class 1 confused for class 2
//! assert_eq!(at(1, 3), 1);   // class 1 missed    -> background column
//! assert_eq!(at(3, 0), 1);   // spurious class 0  -> background row
//! ```
//!
//! Producing the pairs is the caller's job, and it is the only family-specific
//! step: detection matches boxes by IoU, tracking matches tracks by identity.
//! Once matched, the counting is the same operation.

/// Accumulate a confusion matrix from aligned label pairs.
///
/// `gt_labels` and `dt_labels` are parallel arrays over **match records**, not
/// over predictions. Each entry `i` says what a single matching decision produced:
///
/// | `gt_labels[i]` | `dt_labels[i]` | Meaning | Lands at |
/// |---|---|---|---|
/// | `Some(g)` | `Some(d)` | matched pair (correct when `g == d`) | `[g][d]` |
/// | `Some(g)` | `None` | ground truth with no prediction | `[g][num_classes]` |
/// | `None` | `Some(d)` | prediction matching no ground truth | `[num_classes][d]` |
/// | `None` | `None` | nothing happened | ignored |
///
/// Returns a flat row-major matrix of side `num_classes + 1`, rows indexed by
/// ground truth and columns by prediction.
///
/// A label at or beyond `num_classes` names a class the matrix has no lane for,
/// so **that side is treated as `None`** and the table above applies to the
/// result. The record's in-range partner still counts: a valid ground truth
/// matched by an out-of-range prediction stays a miss (`[g][num_classes]`)
/// rather than vanishing from FN accounting, and vice versa. Both sides
/// out-of-range degrades to `(None, None)` — ignored.
///
/// Counts are `u64` addition, so accumulating per-image and summing gives the same
/// answer as one whole-dataset call — which is what lets callers parallelize over
/// images and reduce afterwards.
///
/// # Panics
///
/// If `gt_labels` and `dt_labels` have different lengths, or on a class count
/// [`check_num_classes`] refuses.
pub fn confusion_matrix(
    gt_labels: &[Option<usize>],
    dt_labels: &[Option<usize>],
    num_classes: usize,
) -> Vec<u64> {
    assert_num_classes(num_classes, "confusion_matrix");
    let mut matrix = vec![0u64; (num_classes + 1) * (num_classes + 1)];
    accumulate_confusion(&mut matrix, gt_labels, dt_labels, num_classes);
    matrix
}

/// Whether a confusion matrix for `num_classes` classes can be built: its
/// largest form, `(num_classes + 1)²` `u64` cells with a background row and
/// column (or the semantic family's unlabeled column), must fit in one
/// allocation the system grants. Past `isize::MAX` bytes the cell count wraps
/// in release builds, and the matrix built is too small for the first count
/// written to it; below that, an allocation the system refuses aborts the
/// process when the matrix is built, so the check asks for it first.
///
/// Every constructor of a confusion matrix checks this and panics on a count
/// it refuses. A caller that sizes something else by the class count first,
/// as `SemanticEval::new` names each class, or one that wants the error
/// instead, checks up front.
///
/// # Errors
///
/// `InvalidInput` naming the count, when the matrix would not fit.
pub fn check_num_classes(num_classes: usize) -> crate::error::Result<()> {
    let too_many = || {
        crate::error::Error::InvalidInput(format!(
            "{num_classes} is too many classes for a confusion matrix"
        ))
    };
    let cells = num_classes
        .checked_add(1)
        .and_then(|side| side.checked_mul(side))
        .filter(|cells| {
            cells
                .checked_mul(size_of::<u64>())
                .is_some_and(|bytes| isize::try_from(bytes).is_ok())
        })
        .ok_or_else(too_many)?;
    // Reserved and released at once: no page is touched, and a refusal here
    // is an error where the same refusal in `vec!` would be an abort.
    Vec::<u64>::new()
        .try_reserve_exact(cells)
        .map_err(|_| too_many())
}

/// [`check_num_classes`] as the panic a constructor documents, naming `who`.
pub(crate) fn assert_num_classes(num_classes: usize, who: &str) {
    if let Err(e) = check_num_classes(num_classes) {
        panic!("{who}: {e}");
    }
}

/// Add label pairs into an existing confusion matrix.
///
/// The accumulating form of [`confusion_matrix`], for callers folding many batches
/// into one result. Allocating a fresh `k²` matrix per batch and adding it in costs
/// a memset plus a read-modify-write of the whole matrix to record a handful of
/// records — on Objects365 (80k images, 365 classes) that is ~86 GB of memory
/// traffic for ~1.1M records, and it scales with `num_classes²`, not with the data.
///
/// Label semantics — including the out-of-range-label fallback to background —
/// are documented on [`confusion_matrix`].
///
/// # Panics
///
/// If `matrix` is not exactly `(num_classes + 1)²` elements, or if `gt_labels`
/// and `dt_labels` have different lengths. Both are asserted rather than
/// degraded: an undersized matrix would otherwise come back all-zero with no
/// error.
pub fn accumulate_confusion(
    matrix: &mut [u64],
    gt_labels: &[Option<usize>],
    dt_labels: &[Option<usize>],
    num_classes: usize,
) {
    let k = num_classes + 1;
    assert_eq!(
        matrix.len(),
        k * k,
        "accumulate_confusion: matrix must be (num_classes + 1)² = {} elements \
         for num_classes = {num_classes} (got {})",
        k * k,
        matrix.len()
    );
    assert_eq!(
        gt_labels.len(),
        dt_labels.len(),
        "accumulate_confusion: gt_labels and dt_labels must be parallel arrays \
         (got {} vs {})",
        gt_labels.len(),
        dt_labels.len()
    );

    for (gl, dl) in gt_labels.iter().zip(dt_labels) {
        // Background index is `num_classes`. An out-of-range label has no lane,
        // so that *side* degrades to `None` — the in-range partner still counts
        // (see the table on `confusion_matrix`).
        let row = match gl.filter(|&g| g < num_classes) {
            Some(g) => g,
            None => num_classes,
        };
        let col = match dl.filter(|&d| d < num_classes) {
            Some(d) => d,
            None => num_classes,
        };
        if row == num_classes && col == num_classes {
            continue; // neither a countable prediction nor a countable ground truth
        }
        matrix[row * k + col] += 1;
    }
}

/// Divide each row of a confusion matrix by its own sum.
///
/// Turns raw counts into "of the ground truths of this class, what fraction were
/// predicted as each class" — the form a confusion heatmap usually plots, because
/// it stays readable when class frequencies differ by orders of magnitude.
///
/// Takes `num_classes`, the same argument [`confusion_matrix`] takes, and adds the
/// background lane itself. Taking the matrix *side* instead would put two
/// conventions for one dimension forty lines apart in the same module, and passing
/// the wrong one would not panic — it would normalize the top-left sub-block and
/// return a short vector of plausible-looking garbage.
///
/// Rows summing to zero stay all-zero rather than producing `NaN`.
pub fn row_normalize(matrix: &[u64], num_classes: usize) -> Vec<f64> {
    let k = num_classes + 1;
    let mut norm = vec![0.0f64; k * k];
    for row in 0..k {
        let row_sum: u64 = (0..k).map(|col| matrix[row * k + col]).sum();
        if row_sum > 0 {
            let denom = row_sum as f64;
            for col in 0..k {
                norm[row * k + col] = matrix[row * k + col] as f64 / denom;
            }
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The matched-pair block must equal `sklearn.metrics.confusion_matrix`.
    ///
    /// The in-crate tests are hand-derived, which verifies the author's arithmetic
    /// rather than agreement with the field — a matrix that transposed rows and
    /// columns, say, would reproduce a symmetric fixture perfectly.
    ///
    /// Scoped to records where both labels are present, which is sklearn's whole
    /// domain: it has no way to express "this detection matched nothing". The
    /// background row and column are the part sklearn cannot check, and are covered
    /// by [`confusion_marginals_account_for_every_record`].
    ///
    /// Regenerate with
    /// `uv run --with scikit-learn --with netcal python scripts/gen_metrics_fixtures.py`.
    #[test]
    fn matched_pairs_match_sklearn() {
        #[derive(serde::Deserialize)]
        struct Case {
            num_classes: usize,
            gt: Vec<usize>,
            dt: Vec<usize>,
            matrix: Vec<Vec<u64>>,
        }

        let data = include_str!("testdata/confusion_sklearn.json");
        let cases: Vec<Case> = serde_json::from_str(data).expect("parse fixture");
        assert!(cases.len() > 150, "fixture looks truncated");

        for (i, c) in cases.iter().enumerate() {
            let gt: Vec<Option<usize>> = c.gt.iter().map(|&v| Some(v)).collect();
            let dt: Vec<Option<usize>> = c.dt.iter().map(|&v| Some(v)).collect();
            let ours = confusion_matrix(&gt, &dt, c.num_classes);
            let side = c.num_classes + 1;

            for g in 0..c.num_classes {
                for d in 0..c.num_classes {
                    assert_eq!(
                        ours[g * side + d],
                        c.matrix[g][d],
                        "case {i}: cell [{g}][{d}] is {} but sklearn says {}",
                        ours[g * side + d],
                        c.matrix[g][d]
                    );
                }
            }

            // Every record had both labels, so nothing may land in the background
            // lane. This is what catches a case landing in the wrong lane entirely
            // rather than the wrong cell.
            for k in 0..side {
                assert_eq!(
                    ours[c.num_classes * side + k],
                    0,
                    "case {i}: background row"
                );
                assert_eq!(
                    ours[k * side + c.num_classes],
                    0,
                    "case {i}: background col"
                );
            }
        }
    }

    /// The marginals — every other test here asserts individual cells, and a cell
    /// test cannot see a record that was dropped, double-counted, or landed in the
    /// wrong lane.
    ///
    /// Two facts follow from the construction: every in-range record increments
    /// exactly one cell, and which cell is fixed by the `(gt, dt)` pair. So the
    /// grand total is the number of in-range records, each row sums to the ground
    /// truths of that class, and each column to the predictions of that class. A
    /// flipped rematchable flag upstream, or a length drift between the two label
    /// arrays, shows up here and nowhere else.
    #[test]
    fn confusion_marginals_account_for_every_record() {
        let mut rng = StdRng::seed_from_u64(0xC0F5);

        for case in 0..5000 {
            let num_classes = rng.random_range(1..=6);
            let n = rng.random_range(0..=40);
            let side = num_classes + 1;

            // Deliberately draw labels past `num_classes` sometimes: out-of-range
            // records are documented as skipped, so they must not land anywhere.
            let label = |rng: &mut StdRng| -> Option<usize> {
                match rng.random_range(0..4) {
                    0 => None,
                    1 => Some(rng.random_range(num_classes..num_classes + 3)),
                    _ => Some(rng.random_range(0..num_classes)),
                }
            };
            let gt: Vec<Option<usize>> = (0..n).map(|_| label(&mut rng)).collect();
            let dt: Vec<Option<usize>> = (0..n).map(|_| label(&mut rng)).collect();

            let m = confusion_matrix(&gt, &dt, num_classes);
            let ctx = format!("case {case}: num_classes={num_classes} n={n}");
            assert_eq!(m.len(), side * side, "{ctx}");

            // An out-of-range label is documented to degrade to `None`
            // (background), so marginals are stated over the *effective* labels.
            let eff = |l: &Option<usize>| l.filter(|&v| v < num_classes);
            let counted = gt
                .iter()
                .zip(&dt)
                .filter(|(g, d)| {
                    // `(None, None)` — original or by degradation — is
                    // documented as "nothing happened".
                    !(eff(g).is_none() && eff(d).is_none())
                })
                .count() as u64;

            assert_eq!(
                m.iter().sum::<u64>(),
                counted,
                "{ctx}: grand total disagrees with the number of countable records"
            );

            for g in 0..side {
                let row: u64 = m[g * side..(g + 1) * side].iter().sum();
                let want = gt
                    .iter()
                    .zip(&dt)
                    .filter(|(gl, dl)| match eff(gl) {
                        Some(v) => v == g,
                        // Unmatched predictions land in the background row.
                        None => g == num_classes && eff(dl).is_some(),
                    })
                    .count() as u64;
                assert_eq!(row, want, "{ctx}: row {g} sum {row} != {want}");
            }

            for d in 0..side {
                let col: u64 = (0..side).map(|g| m[g * side + d]).sum();
                let want = gt
                    .iter()
                    .zip(&dt)
                    .filter(|(gl, dl)| match eff(dl) {
                        Some(v) => v == d,
                        // Undetected ground truths land in the background column.
                        None => d == num_classes && eff(gl).is_some(),
                    })
                    .count() as u64;
                assert_eq!(col, want, "{ctx}: column {d} sum {col} != {want}");
            }

            // Batching must equal one whole call — this is what lets callers
            // parallelize over images and reduce afterwards.
            if n >= 2 {
                let split = rng.random_range(1..n);
                let mut batched = vec![0u64; side * side];
                accumulate_confusion(&mut batched, &gt[..split], &dt[..split], num_classes);
                accumulate_confusion(&mut batched, &gt[split..], &dt[split..], num_classes);
                assert_eq!(batched, m, "{ctx}: batched at {split} != whole");
            }
        }
    }

    #[test]
    fn diagonal_counts_correct_predictions() {
        let gt = [Some(0), Some(1), Some(2)];
        let dt = [Some(0), Some(1), Some(2)];
        let m = confusion_matrix(&gt, &dt, 3);
        let k = 4;
        for c in 0..3 {
            assert_eq!(m[c * k + c], 1);
        }
        assert_eq!(m.iter().sum::<u64>(), 3);
    }

    #[test]
    fn unmatched_gt_and_dt_land_in_the_background_lane() {
        let gt = [Some(1), None];
        let dt = [None, Some(2)];
        let m = confusion_matrix(&gt, &dt, 3);
        let at = |gt_class: usize, dt_class: usize| m[gt_class * 4 + dt_class];
        assert_eq!(at(1, 3), 1, "missed GT -> background column");
        assert_eq!(at(3, 2), 1, "spurious DT -> background row");
    }

    #[test]
    fn background_to_background_is_not_counted() {
        let m = confusion_matrix(&[None, None], &[None, None], 3);
        assert_eq!(m.iter().sum::<u64>(), 0);
    }

    /// An out-of-range label degrades to background instead of deleting the
    /// whole record — the in-range partner must stay in FN/FP accounting.
    #[test]
    fn out_of_range_partner_does_not_delete_the_record() {
        let m = confusion_matrix(&[Some(99), Some(0)], &[Some(0), Some(99)], 3);
        let at = |gt_class: usize, dt_class: usize| m[gt_class * 4 + dt_class];
        // Out-of-range GT, valid DT: the prediction counts as spurious.
        assert_eq!(at(3, 0), 1, "valid DT must survive an invalid GT label");
        // Valid GT, out-of-range DT: the ground truth counts as missed.
        assert_eq!(at(0, 3), 1, "valid GT must survive an invalid DT label");
        assert_eq!(m.iter().sum::<u64>(), 2);

        // Both sides out-of-range degrades to (None, None): nothing counted.
        let m = confusion_matrix(&[Some(99)], &[Some(42)], 3);
        assert_eq!(m.iter().sum::<u64>(), 0);
        // Out-of-range paired with None likewise counts nothing.
        let m = confusion_matrix(&[Some(99), None], &[None, Some(42)], 3);
        assert_eq!(m.iter().sum::<u64>(), 0);
    }

    #[test]
    fn per_batch_sums_equal_one_whole_call() {
        let gt = [Some(0), Some(1), None, Some(2)];
        let dt = [Some(1), None, Some(0), Some(2)];

        let whole = confusion_matrix(&gt, &dt, 3);
        let a = confusion_matrix(&gt[..2], &dt[..2], 3);
        let b = confusion_matrix(&gt[2..], &dt[2..], 3);
        let summed: Vec<u64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        assert_eq!(whole, summed, "batching must not change the counts");
    }

    #[test]
    fn row_normalize_makes_rows_sum_to_one() {
        // One real class plus background => side 2. Row 0: 3 and 1 => 0.75 / 0.25.
        // Row 1 empty => stays zero.
        let m = vec![3, 1, 0, 0];
        let n = row_normalize(&m, 1);
        assert!((n[0] - 0.75).abs() < 1e-12);
        assert!((n[1] - 0.25).abs() < 1e-12);
        assert_eq!(n[2], 0.0);
        assert_eq!(n[3], 0.0);
    }

    #[test]
    #[should_panic(expected = "parallel arrays")]
    fn mismatched_lengths_panic_instead_of_truncating() {
        confusion_matrix(&[Some(0), Some(1), Some(2)], &[Some(0)], 3);
    }

    /// An undersized matrix used to be a silent no-op (all-zero output, no
    /// error). It is now a descriptive assert, like its sibling `row_normalize`.
    #[test]
    #[should_panic(expected = "(num_classes + 1)²")]
    fn undersized_matrix_panics_instead_of_no_op() {
        let mut too_small = vec![0u64; 4]; // needs (3+1)² = 16
        accumulate_confusion(&mut too_small, &[Some(0)], &[Some(0)], 3);
    }

    #[test]
    #[should_panic(expected = "(num_classes + 1)²")]
    fn oversized_matrix_panics_too() {
        let mut too_big = vec![0u64; 25]; // needs (3+1)² = 16
        accumulate_confusion(&mut too_big, &[Some(0)], &[Some(0)], 3);
    }

    #[test]
    fn check_num_classes_refuses_a_matrix_past_one_allocation() {
        for fine in [0, 1, 80, 1000] {
            assert!(check_num_classes(fine).is_ok(), "{fine}");
        }
        // `(n + 1)²` wraps, or its bytes pass `isize::MAX`, or (`1 << 28`,
        // about 2⁵⁹ bytes) no system grants the allocation, which `vec!`
        // would answer with an abort.
        for too_many in [1 << 28, 1 << 30, 1 << 32, usize::MAX - 1, usize::MAX] {
            match check_num_classes(too_many) {
                Err(crate::error::Error::InvalidInput(m)) => {
                    assert!(
                        m.contains("too many classes") && m.contains(&too_many.to_string()),
                        "{m}"
                    );
                }
                other => panic!("{too_many}: {other:?}"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "too many classes")]
    fn confusion_matrix_panics_on_too_many_classes() {
        confusion_matrix(&[], &[], 1 << 32);
    }

    #[test]
    #[should_panic(expected = "too many classes")]
    fn pixel_confusion_panics_on_too_many_classes() {
        crate::metrics::semantic::PixelConfusion::new(1 << 32);
    }
}
