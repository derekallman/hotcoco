//! Rectangular linear sum assignment (LSAP) — a semantic port of scipy's
//! `rectangular_lsap.cpp` (Crouse's shortest-augmenting-path algorithm).
//!
//! This is the tracking-lineage matcher: CLEAR, IDF1, and HOTA all call scipy's
//! `linear_sum_assignment`. It is deliberately **not** a textbook Jonker-Volgenant
//! solver — the point is to reproduce scipy's *exact* assignment vector, including
//! its tie-breaking among co-optimal solutions, because a different (equally
//! optimal) assignment changes downstream tracking counts like IDSW/TPA. It is
//! validated against scipy for identical `(row_ind, col_ind)`, not merely equal
//! total cost.
//!
//! The algorithm and its tie behavior are transcribed from scipy's source:
//! - work on a matrix with `rows <= cols`, transposing if necessary;
//! - for each unassigned row, grow a shortest augmenting path (Dijkstra on
//!   reduced costs with dual variables `u`, `v`);
//! - among equal shortest-path costs, prefer an as-yet-unassigned column
//!   (`row4col[j] == NONE`), and otherwise keep the first `remaining` slot found;
//! - the `remaining` scan order (`nc-1 .. 0`, with swap-removal) is part of the
//!   contract, so it is reproduced exactly.

const NONE: usize = usize::MAX;

/// Scratch state for one solve. Holding the working buffers on a struct keeps the
/// augmenting-path routine to a normal argument count (scipy passes ~11 pointers
/// because it is C) and lets a caller reuse the allocation across solves later.
struct Solver<'a> {
    /// Cost matrix, row-major `[nr * nc]`, already transposed to `nr <= nc` and
    /// negated if maximizing.
    cost: &'a [f64],
    nr: usize,
    nc: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    shortest: Vec<f64>,
    path: Vec<usize>,
    col4row: Vec<usize>,
    row4col: Vec<usize>,
    sr: Vec<bool>,
    sc: Vec<bool>,
    remaining: Vec<usize>,
}

impl<'a> Solver<'a> {
    fn new(cost: &'a [f64], nr: usize, nc: usize) -> Self {
        Solver {
            cost,
            nr,
            nc,
            u: vec![0.0; nr],
            v: vec![0.0; nc],
            shortest: vec![f64::INFINITY; nc],
            path: vec![NONE; nc],
            col4row: vec![NONE; nr],
            row4col: vec![NONE; nc],
            sr: vec![false; nr],
            sc: vec![false; nc],
            remaining: vec![0; nc],
        }
    }

    /// Grow a shortest augmenting path from `start`. Returns `(sink, min_val)`,
    /// or `(NONE, INFINITY)` if infeasible (only possible with non-finite costs).
    fn augmenting_path(&mut self, start: usize) -> (usize, f64) {
        let nc = self.nc;
        let mut min_val = 0.0f64;
        let mut num_remaining = nc;
        for it in 0..nc {
            self.remaining[it] = nc - it - 1;
        }
        self.sr.fill(false);
        self.sc.fill(false);
        self.shortest.fill(f64::INFINITY);

        let mut i = start;
        let mut sink = NONE;
        while sink == NONE {
            let mut index = NONE;
            let mut lowest = f64::INFINITY;
            self.sr[i] = true;

            for it in 0..num_remaining {
                let j = self.remaining[it];
                let r = min_val + self.cost[i * nc + j] - self.u[i] - self.v[j];
                if r < self.shortest[j] {
                    self.path[j] = i;
                    self.shortest[j] = r;
                }
                // Strict `<` keeps the first slot at a given cost; the equal-cost
                // clause steers ties toward an unassigned column (a valid sink).
                if self.shortest[j] < lowest
                    || (self.shortest[j] == lowest && self.row4col[j] == NONE)
                {
                    lowest = self.shortest[j];
                    index = it;
                }
            }

            min_val = lowest;
            if min_val.is_infinite() {
                return (NONE, min_val);
            }

            let j = self.remaining[index];
            if self.row4col[j] == NONE {
                sink = j;
            } else {
                i = self.row4col[j];
            }
            self.sc[j] = true;
            num_remaining -= 1;
            self.remaining[index] = self.remaining[num_remaining];
            self.remaining[num_remaining] = j;
        }

        (sink, min_val)
    }

    /// `false` when some row has no augmenting path: the matrix is infeasible,
    /// which for this solver means non-finite entries. Without the check the
    /// shortest-path scan never improves on `INFINITY`, no sink is found, and
    /// the augmentation below would index `path[usize::MAX]`.
    #[must_use]
    fn solve(&mut self) -> bool {
        for cur_row in 0..self.nr {
            let (sink, min_val) = self.augmenting_path(cur_row);
            if sink == NONE {
                return false;
            }

            // Update dual variables along the scanned sets.
            self.u[cur_row] += min_val;
            for i in 0..self.nr {
                if self.sr[i] && i != cur_row {
                    self.u[i] += min_val - self.shortest[self.col4row[i]];
                }
            }
            for j in 0..self.nc {
                if self.sc[j] {
                    self.v[j] -= min_val - self.shortest[j];
                }
            }

            // Augment the matching along the path back to `cur_row`.
            let mut j = sink;
            loop {
                let i = self.path[j];
                self.row4col[j] = i;
                std::mem::swap(&mut self.col4row[i], &mut j);
                if i == cur_row {
                    break;
                }
            }
        }
        true
    }
}

/// Solve the rectangular linear sum assignment problem, matching
/// `scipy.optimize.linear_sum_assignment`.
///
/// `cost` is row-major `[nr * nc]`. Returns `(row_ind, col_ind)`, each of length
/// `min(nr, nc)`, with `row_ind` ascending — identical layout to scipy. When
/// `maximize` is true the total assigned cost is maximized instead of minimized.
///
/// Costs must be finite; for finite costs a full `min(nr, nc)` assignment always
/// exists, so this always returns a complete matching.
///
/// # Panics
///
/// If `cost.len() != nr * nc`, or if the matrix is infeasible (which for this
/// solver means non-finite entries). Both are asserted rather than left to
/// produce a downstream index panic. [`try_lsap`] returns the second as an
/// error instead.
pub fn lsap(cost: &[f64], nr: usize, nc: usize, maximize: bool) -> (Vec<usize>, Vec<usize>) {
    match try_lsap(cost, nr, nc, maximize) {
        Some(assignment) => assignment,
        None => {
            panic!("lsap: cost matrix is infeasible — it likely contains NaN or infinite entries")
        }
    }
}

/// [`lsap`] for costs that may not be finite: `None` for an infeasible
/// matrix — no complete assignment of finite total cost — which is a
/// `ValueError` from scipy and a panic from `lsap`. A matrix with some
/// infinite entries is still solved when a finite assignment exists.
///
/// # Panics
///
/// If `cost.len() != nr * nc`.
pub fn try_lsap(
    cost: &[f64],
    nr: usize,
    nc: usize,
    maximize: bool,
) -> Option<(Vec<usize>, Vec<usize>)> {
    if nr == 0 || nc == 0 {
        return Some((Vec::new(), Vec::new()));
    }
    assert_eq!(
        cost.len(),
        nr * nc,
        "lsap: cost must be nr*nc row-major ({nr}x{nc})"
    );

    // Work on rows <= cols; transpose the cost matrix if the input is tall.
    let transpose = nc < nr;
    let (rn, cn) = if transpose { (nc, nr) } else { (nr, nc) };
    // Only allocate a working copy when the input must be transformed. Transpose
    // needs a reordered buffer (fold the negation into the same pass); a plain
    // maximize negates in one pass. The common minimize path borrows `cost`
    // directly — no per-call nr*nc allocation (this runs at MOT scale).
    let owned: Option<Vec<f64>> = if transpose {
        let mut c = vec![0.0f64; rn * cn];
        for i in 0..nr {
            for j in 0..nc {
                let v = cost[i * nc + j];
                c[j * nr + i] = if maximize { -v } else { v };
            }
        }
        Some(c)
    } else if maximize {
        Some(cost.iter().map(|x| -x).collect())
    } else {
        None
    };
    let c: &[f64] = owned.as_deref().unwrap_or(cost);

    let mut solver = Solver::new(c, rn, cn);
    if !solver.solve() {
        return None;
    }
    let col4row = solver.col4row;

    let mut row_ind = Vec::with_capacity(rn);
    let mut col_ind = Vec::with_capacity(rn);
    if transpose {
        // Transposed "rows" are original columns; sort the pairs by original row
        // so row_ind comes out ascending, exactly as scipy does via argsort.
        let mut order: Vec<usize> = (0..rn).collect();
        order.sort_by_key(|&k| col4row[k]);
        for k in order {
            row_ind.push(col4row[k]);
            col_ind.push(k);
        }
    } else {
        for (i, &c4r) in col4row.iter().enumerate() {
            row_ind.push(i);
            col_ind.push(c4r);
        }
    }
    Some((row_ind, col_ind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn cost_of(cost: &[f64], nc: usize, r: &[usize], c: &[usize]) -> f64 {
        r.iter().zip(c).map(|(&i, &j)| cost[i * nc + j]).sum()
    }

    // Brute-force minimum assignment cost over all injective maps of the smaller
    // dimension onto the larger (a size-min(nr,nc) matching).
    fn brute_min(cost: &[f64], nr: usize, nc: usize) -> f64 {
        if nr > nc {
            // Transpose so the recursion assigns every row (rows <= cols).
            let mut t = vec![0.0; nr * nc];
            for i in 0..nr {
                for j in 0..nc {
                    t[j * nr + i] = cost[i * nc + j];
                }
            }
            return brute_min(&t, nc, nr);
        }
        fn rec(cost: &[f64], nr: usize, nc: usize, row: usize, used: &mut Vec<bool>) -> f64 {
            if row == nr {
                return 0.0;
            }
            let mut best = f64::INFINITY;
            for j in 0..nc {
                if !used[j] {
                    used[j] = true;
                    let sub = cost[row * nc + j] + rec(cost, nr, nc, row + 1, used);
                    used[j] = false;
                    best = best.min(sub);
                }
            }
            best
        }
        rec(cost, nr, nc, 0, &mut vec![false; nc])
    }

    #[test]
    fn known_square_assignment() {
        // Optimal: (0->1, 1->0, 2->2) cost 1+2+3 = 6 beats the diagonal 4+... .
        let cost = [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let (r, c) = lsap(&cost, 3, 3, false);
        assert_eq!(r, vec![0, 1, 2]);
        assert!((cost_of(&cost, 3, &r, &c) - brute_min(&cost, 3, 3)).abs() < 1e-12);
    }

    #[test]
    fn wide_matrix_assigns_all_rows() {
        // 2 rows, 3 cols: every row assigned, row_ind ascending.
        let cost = [1.0, 2.0, 3.0, 4.0, 1.0, 5.0];
        let (r, c) = lsap(&cost, 2, 3, false);
        assert_eq!(r, vec![0, 1]);
        assert_eq!(c.len(), 2);
        assert!((cost_of(&cost, 3, &r, &c) - brute_min(&cost, 2, 3)).abs() < 1e-12);
    }

    #[test]
    fn tall_matrix_triggers_transpose_and_keeps_row_ind_sorted() {
        // 3 rows, 2 cols: only 2 rows can be assigned; row_ind must be ascending.
        let cost = [1.0, 4.0, 2.0, 1.0, 5.0, 3.0];
        let (r, c) = lsap(&cost, 3, 2, false);
        assert_eq!(r.len(), 2);
        assert!(r.windows(2).all(|w| w[0] < w[1]), "row_ind ascending");
        assert!(r.iter().all(|&i| i < 3) && c.iter().all(|&j| j < 2));
        assert!((cost_of(&cost, 2, &r, &c) - brute_min(&cost, 3, 2)).abs() < 1e-12);
    }

    #[test]
    fn maximize_picks_largest() {
        let cost = [1.0, 2.0, 3.0, 4.0];
        let (r, c) = lsap(&cost, 2, 2, true);
        // max total = 2 + 3 = 5 via (0->1, 1->0)
        assert_eq!((r, c), (vec![0, 1], vec![1, 0]));
    }

    #[test]
    fn trivial_sizes() {
        assert_eq!(lsap(&[], 0, 0, false), (vec![], vec![]));
        assert_eq!(lsap(&[], 0, 3, false), (vec![], vec![]));
        assert_eq!(lsap(&[7.0], 1, 1, false), (vec![0], vec![0]));
    }

    #[test]
    fn matches_scipy_assignment_vectors() {
        // Frozen scipy.optimize.linear_sum_assignment outputs (see
        // scripts/gen_assign_fixtures.py). Weighted toward small-integer matrices
        // where co-optima exist, so this checks tie-breaking, not just optimality.
        let data = include_str!("testdata/lsap_scipy.json");
        let cases: serde_json::Value = serde_json::from_str(data).expect("parse fixture");
        let u = |x: &serde_json::Value| x.as_u64().expect("u64") as usize;
        let uv = |x: &serde_json::Value| {
            x.as_array()
                .expect("array")
                .iter()
                .map(u)
                .collect::<Vec<_>>()
        };
        for (idx, case) in cases.as_array().expect("array").iter().enumerate() {
            let (nr, nc) = (u(&case["nr"]), u(&case["nc"]));
            let maximize = case["maximize"].as_bool().expect("bool");
            let cost: Vec<f64> = case["cost"]
                .as_array()
                .expect("array")
                .iter()
                .map(|x| x.as_f64().expect("f64"))
                .collect();
            let (r, c) = lsap(&cost, nr, nc, maximize);
            let (exp_r, exp_c) = (uv(&case["row_ind"]), uv(&case["col_ind"]));
            assert_eq!(r, exp_r, "row_ind case {idx} ({nr}x{nc} max={maximize})");
            assert_eq!(c, exp_c, "col_ind case {idx} ({nr}x{nc} max={maximize})");
        }
    }

    #[test]
    fn brute_force_optimality_random() {
        let mut rng = StdRng::seed_from_u64(0xC0C0);
        for _ in 0..2000 {
            let nr = rng.random_range(1..=5);
            let nc = rng.random_range(1..=5);
            // Mix of continuous and small-integer costs to exercise ties.
            let integer = rng.random_bool(0.5);
            let cost: Vec<f64> = (0..nr * nc)
                .map(|_| {
                    if integer {
                        rng.random_range(0..4) as f64
                    } else {
                        rng.random_range(0.0..10.0)
                    }
                })
                .collect();
            let maximize = rng.random_bool(0.5);
            let (r, c) = lsap(&cost, nr, nc, maximize);
            assert_eq!(r.len(), nr.min(nc));
            // valid assignment: distinct rows, distinct cols, in range
            let mut rows = r.clone();
            rows.sort_unstable();
            rows.dedup();
            assert_eq!(rows.len(), r.len());
            let mut cols = c.clone();
            cols.sort_unstable();
            cols.dedup();
            assert_eq!(cols.len(), c.len());
            // optimal cost
            let got = cost_of(&cost, nc, &r, &c);
            if maximize {
                let neg: Vec<f64> = cost.iter().map(|x| -x).collect();
                let opt = -brute_min(&neg, nr, nc);
                assert!((got - opt).abs() < 1e-9, "maximize not optimal");
            } else {
                let opt = brute_min(&cost, nr, nc);
                assert!((got - opt).abs() < 1e-9, "minimize not optimal");
            }
        }
    }

    #[test]
    #[should_panic(expected = "infeasible")]
    fn all_non_finite_row_reports_infeasible_not_index_oob() {
        // A row with no finite entry leaves the shortest-path scan stuck at
        // INFINITY, so no sink is found. Before this was a real assert, the
        // augmentation then indexed `path[usize::MAX]` and panicked out of bounds
        // with no clue as to the cause. A single NaN is *not* enough — any finite
        // entry in the row still yields a reachable column.
        lsap(&[1.0, 2.0, f64::NAN, f64::NAN], 2, 2, false);
    }

    /// `try_lsap` reports the same matrices as `None`, either direction, and
    /// still solves one whose infinite entries leave a finite assignment.
    #[test]
    fn try_lsap_returns_none_for_an_infeasible_matrix() {
        let inf = f64::INFINITY;
        for (cost, maximize) in [
            ([inf, inf, inf, inf], false),
            ([-inf, 1.0, 1.0, 2.0], false),
            ([inf, 1.0, 1.0, 2.0], true),
        ] {
            assert_eq!(
                try_lsap(&cost, 2, 2, maximize),
                None,
                "{cost:?} maximize={maximize}"
            );
        }
        let solved = try_lsap(&[inf, 1.0, 1.0, inf], 2, 2, false);
        assert_eq!(solved, Some((vec![0, 1], vec![1, 0])));
    }

    #[test]
    #[should_panic(expected = "row-major")]
    fn wrong_length_cost_is_rejected() {
        lsap(&[1.0, 2.0, 3.0], 2, 2, false);
    }
}
