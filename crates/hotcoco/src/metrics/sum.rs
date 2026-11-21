//! Floating-point summation in numpy's order.
//!
//! Float addition is not associative, so two sums of the same values agree to the
//! last bit only when they add in the same order. pycocotools forms every summary
//! statistic with `np.mean`, which sums by pairwise reduction rather than left to
//! right; [`pairwise_sum`] reproduces that order so hotcoco's means are
//! bit-identical to the reference, not merely within rounding of it.

/// Below this length numpy adds serially.
const PW_UNROLL: usize = 8;

/// Up to this length numpy adds with [`PW_UNROLL`] interleaved accumulators;
/// above it, it splits the slice in two and recurses (numpy's `PW_BLOCKSIZE`).
const PW_BLOCKSIZE: usize = 128;

/// Sum `values` in the order numpy's `np.add.reduce` uses for a contiguous `f64`
/// array, and therefore the order `np.sum` and `np.mean` use.
///
/// A port of `pairwise_sum` in numpy's `loops_utils.h.src`:
///
/// - Fewer than 8 values: a plain left-to-right loop.
/// - Up to 128 values: eight accumulators over stride-8 lanes, combined as
///   `((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))`, then the tail of fewer
///   than 8 values added serially.
/// - More than 128 values: split at half the length rounded down to a multiple
///   of 8, and sum each half the same way.
///
/// To match `np.mean(x)` bit for bit, pass the elements in the order numpy
/// flattens `x` — C order — and divide by the length.
///
/// An empty slice sums to `0.0`.
pub(crate) fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < PW_UNROLL {
        values.iter().fold(0.0, |s, &v| s + v)
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; PW_UNROLL];
        r.copy_from_slice(&values[..PW_UNROLL]);
        let blocks_end = n - n % PW_UNROLL;
        for block in values[PW_UNROLL..blocks_end].chunks_exact(PW_UNROLL) {
            for (acc, &v) in r.iter_mut().zip(block) {
                *acc += v;
            }
        }
        let head = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        values[blocks_end..].iter().fold(head, |s, &v| s + v)
    } else {
        let half = n / 2;
        let split = half - half % PW_UNROLL;
        pairwise_sum(&values[..split]) + pairwise_sum(&values[split..])
    }
}

/// numpy's default buffer size in elements, `np.getbufsize()`.
const NUMPY_BUFSIZE: usize = 8192;

/// `np.sum` of a contiguous 1-D `f64` array as numpy 1.23 reduces it: its
/// buffered iterator hands the reduction runs of [`NUMPY_BUFSIZE`] values,
/// each summed by [`pairwise_sum`], and adds the runs left to right. The
/// dense form of [`grid_sum`]; up to 8192 values it is `pairwise_sum`.
pub(crate) fn buffered_pairwise_sum(values: &[f64]) -> f64 {
    values
        .chunks(NUMPY_BUFSIZE)
        .fold(0.0, |total, run| total + pairwise_sum(run))
}

/// `np.sum` of a whole C-ordered 2-D `f64` array of `len` cells, in the order
/// numpy 1.23 adds it, from the array's non-zero cells alone: `cells` holds
/// `(flat index, value)` pairs in ascending index order.
///
/// numpy 1.23 reduces a whole 2-D array through its buffered iterator: it sums
/// each run of 8192 cells with [`pairwise_sum`] and adds the runs left to
/// right. (numpy 2 sums the flattened array in one pairwise pass instead; the
/// tracking reference pins 1.23.) A cell of `+0.0` changes no partial sum, so
/// leaving the zero cells out keeps every bit, which is what lets a mostly
/// empty grid be held as its non-zero cells.
///
/// # Panics
///
/// If an index is not below `len`, or the indices are not ascending.
pub(crate) fn grid_sum(len: usize, cells: &[(usize, f64)]) -> f64 {
    assert!(
        cells.windows(2).all(|w| w[0].0 < w[1].0),
        "grid_sum: cell indices must ascend"
    );
    assert!(
        cells.last().is_none_or(|&(i, _)| i < len),
        "grid_sum: a cell index is past the grid's {len} cells"
    );
    let mut total = 0.0;
    let mut rest = cells;
    while let Some(&(first, _)) = rest.first() {
        let start = first - first % NUMPY_BUFSIZE;
        let n = NUMPY_BUFSIZE.min(len - start);
        let end = rest.partition_point(|&(i, _)| i < start + n);
        total += sparse_pairwise_sum(start, n, &rest[..end]);
        rest = &rest[end..];
    }
    total
}

/// [`pairwise_sum`] of the `n` cells from `start`, given only the non-zero
/// ones: the same lanes, splits, and tail, with the zeros left out.
fn sparse_pairwise_sum(start: usize, n: usize, cells: &[(usize, f64)]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    if n < PW_UNROLL {
        return cells.iter().fold(0.0, |s, &(_, v)| s + v);
    }
    if n <= PW_BLOCKSIZE {
        let blocks_end = start + n - n % PW_UNROLL;
        let tail = cells.partition_point(|&(i, _)| i < blocks_end);
        let mut r = [0.0f64; PW_UNROLL];
        for &(i, v) in &cells[..tail] {
            r[(i - start) % PW_UNROLL] += v;
        }
        let head = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        return cells[tail..].iter().fold(head, |s, &(_, v)| s + v);
    }
    let half = n / 2;
    let split = half - half % PW_UNROLL;
    let mid = cells.partition_point(|&(i, _)| i < start + split);
    sparse_pairwise_sum(start, split, &cells[..mid])
        + sparse_pairwise_sum(start + split, n - split, &cells[mid..])
}

#[cfg(test)]
mod tests {
    use super::{NUMPY_BUFSIZE, grid_sum, pairwise_sum};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The dense definition `grid_sum` must agree with: pairwise within each
    /// buffer run, runs added left to right.
    fn dense_grid_sum(values: &[f64]) -> f64 {
        values
            .chunks(NUMPY_BUFSIZE)
            .fold(0.0, |s, run| s + pairwise_sum(run))
    }

    fn cells_of(values: &[f64]) -> Vec<(usize, f64)> {
        values
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(i, &v)| (i, v))
            .collect()
    }

    /// Sparse and dense agree to the bit over grids that cross every
    /// boundary the order has: 8 lanes, 128-cell blocks, 8192-cell runs.
    #[test]
    fn sparse_grid_sum_is_the_dense_sum() {
        let mut rng = StdRng::seed_from_u64(0x6_51D);
        for trial in 0..400 {
            let len = match trial % 4 {
                0 => rng.random_range(1..16),
                1 => rng.random_range(100..300),
                2 => rng.random_range(8000..8400),
                _ => rng.random_range(16_000..30_000),
            };
            let density = [0.01, 0.2, 0.9][trial % 3];
            let values: Vec<f64> = (0..len)
                .map(|_| {
                    if rng.random_bool(density) {
                        rng.random_range(0.0..3.0) / 7.0
                    } else {
                        0.0
                    }
                })
                .collect();
            assert_eq!(
                grid_sum(len, &cells_of(&values)).to_bits(),
                dense_grid_sum(&values).to_bits(),
                "trial {trial}, len {len}"
            );
        }
        assert_eq!(grid_sum(10, &[]), 0.0);
    }

    /// Bit patterns of `np.sum(m)` under numpy 1.23.5 for the `G × T` grid
    /// with `m[i, j] = 1 / ((7919 i + 104729 j) % 1009 + 1)` where
    /// `(31 i + 17 j) % k == 0` and 0 elsewhere. On each of these, one pairwise
    /// pass over the flattened grid lands on a different bit pattern, so they
    /// tell the buffered order from numpy 2's.
    #[test]
    fn matches_numpy_1_23_on_whole_grids() {
        for &(g, t, k, bits) in NUMPY_1_23_GRIDS {
            let values: Vec<f64> = (0..g * t)
                .map(|c| {
                    let (i, j) = (c / t, c % t);
                    if (31 * i + 17 * j) % k == 0 {
                        1.0 / ((7919 * i + 104_729 * j) % 1009 + 1) as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            assert_eq!(
                grid_sum(g * t, &cells_of(&values)).to_bits(),
                bits,
                "{g} x {t}, k = {k}"
            );
            assert_ne!(pairwise_sum(&values).to_bits(), bits, "{g} x {t}, k = {k}");
        }
    }

    const NUMPY_1_23_GRIDS: &[(usize, usize, usize, u64)] = &[
        (97, 113, 3, 0x403cc85eed89202b),
        (97, 113, 5, 0x4031d7b6fe46c16b),
        (130, 211, 1, 0x4069e39611c159c4),
        (64, 200, 3, 0x40407b20bfa4e142),
    ];

    /// A sequence whose serial and pairwise sums differ: a large value followed
    /// by many values too small to register against it one at a time.
    fn lopsided(n: usize) -> Vec<f64> {
        let mut v = vec![1e-16; n];
        v[0] = 1.0;
        v
    }

    fn serial(v: &[f64]) -> f64 {
        v.iter().fold(0.0, |s, &x| s + x)
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(pairwise_sum(&[]), 0.0);
    }

    #[test]
    fn below_unroll_is_serial() {
        // n = 7: numpy's plain loop, so pairwise and serial agree exactly.
        let v = lopsided(7);
        assert_eq!(pairwise_sum(&v), serial(&v));
        assert_eq!(pairwise_sum(&v), 1.0);
    }

    #[test]
    fn exactly_one_block_of_eight() {
        // Lanes hold one value each: 1, then seven 1e-16. The tree adds the six
        // small lanes pairwise first, (1e-16 + 1e-16) twice and so on, so they
        // reach 1.0 as 6e-16 plus the 1e-16 paired with it — enough to round up.
        let v = lopsided(8);
        let expected = ((1.0 + 1e-16) + (1e-16 + 1e-16)) + ((1e-16 + 1e-16) + (1e-16 + 1e-16));
        assert_eq!(pairwise_sum(&v), expected);
        assert_eq!(serial(&v), 1.0);
        assert_ne!(pairwise_sum(&v), serial(&v));
    }

    #[test]
    fn nine_adds_the_tail_serially() {
        let v = lopsided(9);
        let head = ((1.0 + 1e-16) + (1e-16 + 1e-16)) + ((1e-16 + 1e-16) + (1e-16 + 1e-16));
        assert_eq!(pairwise_sum(&v), head + 1e-16);
        assert_ne!(pairwise_sum(&v), serial(&v));
    }

    #[test]
    fn full_block_of_128_uses_strided_lanes() {
        // Lane j holds values j, j + 8, ..., j + 120: 16 values each.
        let v = lopsided(128);
        let small_lane = (0..16).fold(0.0, |s, _| s + 1e-16);
        let lane0 = (0..15).fold(1.0, |s, _| s + 1e-16);
        let expected = ((lane0 + small_lane) + (small_lane + small_lane))
            + ((small_lane + small_lane) + (small_lane + small_lane));
        assert_eq!(pairwise_sum(&v), expected);
        assert_ne!(pairwise_sum(&v), serial(&v));
    }

    #[test]
    fn above_blocksize_splits_at_a_multiple_of_eight() {
        // n = 129: half is 64, already a multiple of 8, so 64 + 65.
        let v = lopsided(129);
        assert_eq!(
            pairwise_sum(&v),
            pairwise_sum(&v[..64]) + pairwise_sum(&v[64..])
        );
        assert_ne!(pairwise_sum(&v), serial(&v));
        // n = 1000: half is 500, rounded down to 496; each half recurses again.
        let v = lopsided(1000);
        let left = pairwise_sum(&v[..248]) + pairwise_sum(&v[248..496]);
        let right = pairwise_sum(&v[496..748]) + pairwise_sum(&v[748..]);
        assert_eq!(pairwise_sum(&v), left + right);
        assert_ne!(pairwise_sum(&v), serial(&v));
    }

    /// Bit patterns of `np.sum(np.linspace(1, n, n) / 7.0)`, recorded with
    /// numpy 2.4.3 on arm64. Each term is a correctly rounded division, so Rust
    /// builds the same inputs; for n = 8, 9, and 128 the serial sum lands on a
    /// different bit pattern. To regenerate:
    ///
    /// ```text
    /// uv run python -c "import numpy as np
    /// for n in (7, 8, 9, 128, 129, 1000):
    ///     print(n, hex(np.sum(np.linspace(1, n, n) / 7.0).view(np.uint64)))"
    /// ```
    #[test]
    fn matches_numpy_on_sevenths() {
        for &(n, bits) in NUMPY_SEVENTHS {
            let v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / 7.0).collect();
            assert_eq!(pairwise_sum(&v).to_bits(), bits, "n = {n}");
        }
    }

    const NUMPY_SEVENTHS: &[(usize, u64)] = &[
        (7, 0x4010000000000000),
        (8, 0x4014924924924925),
        (9, 0x4019b6db6db6db6e),
        (128, 0x40926db6db6db6dc),
        (129, 0x4092b76db6db6db6),
        (1000, 0x40f174c000000000),
    ];
}
