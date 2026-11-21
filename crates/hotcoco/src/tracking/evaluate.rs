//! One sequence's matching: HOTA's two passes, CLEAR, and Identity over a
//! cleaned pair of tables, as `trackeval/metrics/{hota,clear,identity}.py`
//! match. The formulas that turn the counts into scores are
//! [`crate::metrics::tracking`]'s.
//!
//! Three things differ from the reference, and none changes a number:
//!
//! - **Similarity is not kept.** A frame's IoU is computed when a pass needs
//!   it and dropped after. The first pass serves HOTA's soft counts, CLEAR,
//!   and Identity from one computation; HOTA's second pass computes it again.
//! - **HOTA's per-threshold match grids are sparse**: only the track pairs
//!   that matched are held, and [`association_scores`] sums them in the
//!   order numpy sums the dense grid.
//! - **Sequences run in parallel** (the driver's choice; nothing here is
//!   shared between sequences).
//!
//! Identity keeps the reference's `(G + T)²` assignment.

use rustc_hash::FxHashMap;

use crate::metrics::sum::buffered_pairwise_sum;
use crate::metrics::tracking::{
    AssociationScores, ClearCounts, ClearResult, CountResult, HotaResult, IdentityCounts,
    IdentityResult, NUM_ALPHAS, association_scores, fragmentations, hota_alphas, track_coverage,
};
use crate::primitives::assign::lsap;
use crate::primitives::sim::{bbox_iou_corners_flat, iou_from_weights};
use crate::table::TrackTable;

use super::TrackingResult;
use super::preprocess::EPS;

/// CLEAR's score for continuing the previous frame's match: more than any
/// IoU, so continuity decides first and IoU breaks ties.
const CONTINUITY_BONUS: f64 = 1000.0;

/// The cost Identity's assignment puts on a pairing that is not allowed.
const FORBIDDEN: f64 = 1e10;

/// Every metric of one cleaned sequence.
pub(super) fn evaluate(
    gt: &TrackTable,
    pred: &TrackTable,
    clear_threshold: f64,
    identity_threshold: f64,
) -> TrackingResult {
    let num_frames = gt.num_frames();
    let (n_gt, n_pred) = (gt.len() as u64, pred.len() as u64);
    let count = CountResult::new(
        n_pred,
        n_gt,
        pred.num_tracks() as u64,
        gt.num_tracks() as u64,
        num_frames as u64,
    );
    if gt.is_empty() || pred.is_empty() {
        // TrackEval returns before matching anything.
        return TrackingResult {
            hota: HotaResult::from_counts(
                [0; NUM_ALPHAS],
                [n_gt; NUM_ALPHAS],
                [n_pred; NUM_ALPHAS],
                [AssociationScores::default(); NUM_ALPHAS],
                [0.0; NUM_ALPHAS],
            ),
            clear: ClearResult::unmatched(n_gt, gt.num_tracks() as u64, n_pred),
            identity: IdentityCounts {
                idtp: 0,
                idfn: n_gt,
                idfp: n_pred,
            }
            .scores(),
            count,
        };
    }

    let tracks = Tracks::of(gt, pred);
    let mut soft = vec![0.0f64; tracks.cells()];
    let mut id_matches = vec![0u64; tracks.cells()];
    let mut clear = ClearPass::new(gt.num_tracks(), clear_threshold);
    for f in 0..num_frames {
        let (g, p) = (gt.frame(f), pred.frame(f));
        let (gids, pids) = (gt.tracks(g.clone()), pred.tracks(p.clone()));
        if gids.is_empty() || pids.is_empty() {
            clear.empty_frame(gids.len(), pids.len());
            continue;
        }
        let sim = bbox_iou_corners_flat(gt.boxes(g), pred.boxes(p));
        add_soft_matches(&sim, gids, pids, &tracks, &mut soft);
        for (i, &gt_track) in gids.iter().enumerate() {
            for (j, &pred_track) in pids.iter().enumerate() {
                // Identity compares with `>=` and no tolerance.
                if sim[i * pids.len() + j] >= identity_threshold {
                    id_matches[tracks.cell(gt_track, pred_track)] += 1;
                }
            }
        }
        clear.frame(&sim, gids, pids);
    }

    TrackingResult {
        hota: hota_second_pass(gt, pred, &tracks, soft),
        clear: clear.finish(&tracks.gt_dets, num_frames),
        identity: identity(&tracks, &id_matches, n_gt),
        count,
    }
}

/// The two sides' track counts, and the grid of track pairs HOTA and
/// Identity count into, ground-truth major as TrackEval lays it out.
struct Tracks {
    gt_dets: Vec<u64>,
    pred_dets: Vec<u64>,
}

impl Tracks {
    fn of(gt: &TrackTable, pred: &TrackTable) -> Self {
        Tracks {
            gt_dets: gt.track_lengths(),
            pred_dets: pred.track_lengths(),
        }
    }

    fn cells(&self) -> usize {
        self.gt_dets.len() * self.pred_dets.len()
    }

    fn cell(&self, gt_track: usize, pred_track: usize) -> usize {
        gt_track * self.pred_dets.len() + pred_track
    }
}

/// HOTA's first pass over one frame: each similarity normalized by
/// `sim / (column sum + row sum − sim)` where that is above `ε`, added to its
/// track pair's soft count. Row sums are numpy 1.23's reductions of a
/// contiguous row; column sums are its serial ones down a column, except in a
/// frame with one prediction, where numpy drops the length-1 axis and reduces
/// the one column as it reduces a row.
fn add_soft_matches(
    sim: &[f64],
    gids: &[usize],
    pids: &[usize],
    tracks: &Tracks,
    soft: &mut [f64],
) {
    let np = pids.len();
    let row_sums: Vec<f64> = sim.chunks_exact(np).map(buffered_pairwise_sum).collect();
    let col_sums = if np == 1 {
        vec![buffered_pairwise_sum(sim)]
    } else {
        let mut col_sums = vec![0.0f64; np];
        for row in sim.chunks_exact(np) {
            for (c, &s) in col_sums.iter_mut().zip(row) {
                *c += s;
            }
        }
        col_sums
    };
    for (i, &gt_track) in gids.iter().enumerate() {
        for (j, &pred_track) in pids.iter().enumerate() {
            let s = sim[i * np + j];
            soft[tracks.cell(gt_track, pred_track)] +=
                iou_from_weights(s, col_sums[j], row_sums[i], EPS);
        }
    }
}

/// HOTA's second pass: each frame matched once on `alignment × similarity`,
/// the match counted at every threshold its similarity clears.
fn hota_second_pass(
    gt: &TrackTable,
    pred: &TrackTable,
    tracks: &Tracks,
    mut soft: Vec<f64>,
) -> HotaResult {
    // The global alignment of every track pair, in place of its soft count:
    // TrackEval's `potential_matches_count / (gt_id_count + tracker_id_count
    // - potential_matches_count)`. Every track is in a frame, so the union is
    // at least 1 and the floor never applies: TrackEval divides unguarded.
    let np_tracks = tracks.pred_dets.len();
    for (cell, s) in soft.iter_mut().enumerate() {
        let (g, t) = (cell / np_tracks, cell % np_tracks);
        *s = iou_from_weights(
            *s,
            tracks.gt_dets[g] as f64,
            tracks.pred_dets[t] as f64,
            0.0,
        );
    }
    let alignment = soft;

    let alphas = hota_alphas();
    let mut tp = [0u64; NUM_ALPHAS];
    let mut fn_ = [0u64; NUM_ALPHAS];
    let mut fp = [0u64; NUM_ALPHAS];
    let mut similarity_sum = [0.0f64; NUM_ALPHAS];
    // Per matched track pair, how many frames it matched in at each
    // threshold: the sparse form of TrackEval's 19 dense grids.
    let mut grid: FxHashMap<usize, [u64; NUM_ALPHAS]> = FxHashMap::default();
    for f in 0..gt.num_frames() {
        let (g, p) = (gt.frame(f), pred.frame(f));
        let (gids, pids) = (gt.tracks(g.clone()), pred.tracks(p.clone()));
        if gids.is_empty() || pids.is_empty() {
            for a in 0..NUM_ALPHAS {
                fn_[a] += gids.len() as u64;
                fp[a] += pids.len() as u64;
            }
            continue;
        }
        let (ng, np) = (gids.len(), pids.len());
        let sim = bbox_iou_corners_flat(gt.boxes(g), pred.boxes(p));
        let mut score = Vec::with_capacity(sim.len());
        for &gt_track in gids {
            for &pred_track in pids {
                score.push(alignment[tracks.cell(gt_track, pred_track)]);
            }
        }
        for (s, &v) in score.iter_mut().zip(&sim) {
            *s *= v;
        }
        let (rows, cols) = lsap(&score, ng, np, true);
        let mut matched = [0u64; NUM_ALPHAS];
        // Each threshold's sum of this frame's matched similarities, row by
        // row from 0, as Python's `sum` over the matches; added to the
        // sequence's sum after, as TrackEval adds it.
        let mut frame_sum = [0.0f64; NUM_ALPHAS];
        for (r, c) in rows.into_iter().zip(cols) {
            let s = sim[r * np + c];
            let cleared = alphas.iter().take_while(|&&alpha| s >= alpha - EPS).count();
            if cleared == 0 {
                continue;
            }
            let counts = grid
                .entry(tracks.cell(gids[r], pids[c]))
                .or_insert([0; NUM_ALPHAS]);
            for a in 0..cleared {
                matched[a] += 1;
                frame_sum[a] += s;
                counts[a] += 1;
            }
        }
        for a in 0..NUM_ALPHAS {
            tp[a] += matched[a];
            fn_[a] += ng as u64 - matched[a];
            fp[a] += np as u64 - matched[a];
            if matched[a] > 0 {
                similarity_sum[a] += frame_sum[a];
            }
        }
    }

    let mut pairs: Vec<(usize, [u64; NUM_ALPHAS])> = grid.into_iter().collect();
    pairs.sort_unstable_by_key(|&(cell, _)| cell);
    let association = std::array::from_fn(|a| {
        let matches: Vec<(usize, usize, u64)> = pairs
            .iter()
            .filter(|(_, counts)| counts[a] > 0)
            .map(|&(cell, counts)| (cell / np_tracks, cell % np_tracks, counts[a]))
            .collect();
        association_scores(&matches, &tracks.gt_dets, &tracks.pred_dets, tp[a])
    });
    HotaResult::from_counts(tp, fn_, fp, association, similarity_sum)
}

/// CLEAR's one pass, frame by frame in order.
struct ClearPass {
    threshold: f64,
    counts: ClearCounts,
    /// Frames each ground-truth track was matched in.
    matched: Vec<u64>,
    /// Times each went from untracked to tracked.
    starts: Vec<u64>,
    /// The prediction each was last matched to, in any earlier frame: what
    /// an identity switch is counted against.
    last_match: Vec<Option<usize>>,
    /// The prediction each was matched to in the last frame both sides had
    /// rows in, if any: what the continuity bonus rewards. TrackEval clears
    /// it only in such a frame, so it survives a frame without them.
    held: Vec<Option<usize>>,
    /// The tracks set in `held`, so a frame clears only those.
    held_set: Vec<usize>,
}

impl ClearPass {
    fn new(gt_tracks: usize, threshold: f64) -> Self {
        ClearPass {
            threshold,
            counts: ClearCounts::default(),
            matched: vec![0; gt_tracks],
            starts: vec![0; gt_tracks],
            last_match: vec![None; gt_tracks],
            held: vec![None; gt_tracks],
            held_set: Vec::new(),
        }
    }

    /// A frame with no ground truth or no predictions: every box unmatched,
    /// and the memory of the last frame left as it is.
    fn empty_frame(&mut self, gt: usize, pred: usize) {
        self.counts.fn_ += gt as u64;
        self.counts.fp += pred as u64;
    }

    fn frame(&mut self, sim: &[f64], gids: &[usize], pids: &[usize]) {
        let (ng, np) = (gids.len(), pids.len());
        let mut score = Vec::with_capacity(sim.len());
        for (i, &gt_track) in gids.iter().enumerate() {
            for (j, &pred_track) in pids.iter().enumerate() {
                let s = sim[i * np + j];
                let bonus = if self.held[gt_track] == Some(pred_track) {
                    CONTINUITY_BONUS
                } else {
                    0.0
                };
                score.push(if s < self.threshold - EPS {
                    0.0
                } else {
                    bonus + s
                });
            }
        }
        let (rows, cols) = lsap(&score, ng, np, true);
        let kept: Vec<(usize, usize)> = rows
            .into_iter()
            .zip(cols)
            .filter(|&(r, c)| score[r * np + c] > EPS)
            .collect();

        let mut frame_sum = 0.0;
        for &(r, c) in &kept {
            let (g, t) = (gids[r], pids[c]);
            if self.last_match[g].is_some_and(|last| last != t) {
                self.counts.idsw += 1;
            }
            if self.held[g].is_none() {
                self.starts[g] += 1;
            }
            self.matched[g] += 1;
            self.last_match[g] = Some(t);
            frame_sum += sim[r * np + c];
        }
        for g in self.held_set.drain(..) {
            self.held[g] = None;
        }
        for &(r, c) in &kept {
            self.held[gids[r]] = Some(pids[c]);
            self.held_set.push(gids[r]);
        }

        let n = kept.len() as u64;
        self.counts.tp += n;
        self.counts.fn_ += ng as u64 - n;
        self.counts.fp += np as u64 - n;
        if n > 0 {
            self.counts.motp_sum += frame_sum;
        }
    }

    fn finish(mut self, gt_dets: &[u64], num_frames: usize) -> ClearResult {
        (self.counts.mt, self.counts.pt, self.counts.ml) = track_coverage(&self.matched, gt_dets);
        self.counts.frag = fragmentations(&self.starts);
        self.counts.frames = num_frames as u64;
        self.counts.scores()
    }
}

/// Identity: one assignment over the `(G + T)²` matrix of misses plus false
/// positives, as TrackEval builds it, with [`FORBIDDEN`] where a track may
/// not go.
fn identity(tracks: &Tracks, id_matches: &[u64], n_gt: u64) -> IdentityResult {
    let (ng, np) = (tracks.gt_dets.len(), tracks.pred_dets.len());
    let n = ng + np;
    // The misses and the false positives of one cell of the matrix.
    let cell = |r: usize, c: usize| -> (f64, f64) {
        match (r < ng, c < np) {
            (true, true) => {
                let m = id_matches[tracks.cell(r, c)] as f64;
                (tracks.gt_dets[r] as f64 - m, tracks.pred_dets[c] as f64 - m)
            }
            (true, false) => (
                if c - np == r {
                    tracks.gt_dets[r] as f64
                } else {
                    FORBIDDEN
                },
                0.0,
            ),
            (false, true) => (
                0.0,
                if r - ng == c {
                    tracks.pred_dets[c] as f64
                } else {
                    FORBIDDEN
                },
            ),
            (false, false) => (0.0, 0.0),
        }
    };
    let mut cost = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            let (missed, spurious) = cell(r, c);
            cost.push(missed + spurious);
        }
    }
    let (rows, cols) = lsap(&cost, n, n, false);
    let (mut idfn, mut idfp) = (0.0f64, 0.0f64);
    for (r, c) in rows.into_iter().zip(cols) {
        let (missed, spurious) = cell(r, c);
        idfn += missed;
        idfp += spurious;
    }
    let (idfn, idfp) = (idfn as u64, idfp as u64);
    IdentityCounts {
        idtp: n_gt - idfn,
        idfn,
        idfp,
    }
    .scores()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::sum::pairwise_sum;

    /// HOTA's first pass against TrackEval's lines run under numpy 1.23.5 on
    /// a 3 × 12 frame: row sums are pairwise along a row, and a serial row
    /// sum would land 20 of these 36 cells on other bits. Nothing reported
    /// shows this directly — the soft counts only steer the second pass's
    /// matching — so the parity fixture cannot catch it and this test does.
    #[test]
    fn soft_matches_are_trackevals_bits() {
        let (ng, np) = (3, 12);
        let sim: Vec<f64> = (0..ng * np)
            .map(|k| {
                let (i, j) = (k / np, k % np);
                ((i * 13 + j * 13) % 29 + 1) as f64 / 31.0
            })
            .collect();
        let tracks = Tracks {
            gt_dets: vec![1; ng],
            pred_dets: vec![1; np],
        };
        let gids: Vec<usize> = (0..ng).collect();
        let pids: Vec<usize> = (0..np).collect();
        let mut soft = vec![0.0; tracks.cells()];
        add_soft_matches(&sim, &gids, &pids, &tracks, &mut soft);
        let bits: Vec<u64> = soft.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, NUMPY_1_23_SIM_IOU);
        let serial = |r: &[f64]| r.iter().fold(0.0, |s, &v| s + v);
        assert!(
            sim.chunks_exact(np).any(|r| serial(r) != pairwise_sum(r)),
            "the frame no longer tells pairwise row sums from serial ones"
        );
    }

    /// The same pass on a 12 × 1 frame, one prediction: numpy drops the
    /// length-1 axis of `similarity.sum(0)` and sums the column pairwise, as
    /// it sums a row, where a serial column sum lands every one of these
    /// cells on other bits.
    #[test]
    fn a_one_prediction_frame_sums_its_column_as_numpy_does() {
        let ng = 12;
        let sim: Vec<f64> = (0..ng).map(|i| ((i * 37) % 97 + 1) as f64 / 97.0).collect();
        let tracks = Tracks {
            gt_dets: vec![1; ng],
            pred_dets: vec![1],
        };
        let gids: Vec<usize> = (0..ng).collect();
        let mut soft = vec![0.0; tracks.cells()];
        add_soft_matches(&sim, &gids, &[0], &tracks, &mut soft);
        let bits: Vec<u64> = soft.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, NUMPY_1_23_ONE_COLUMN_SIM_IOU);
        let serial = sim.iter().fold(0.0, |s, &v| s + v);
        assert_ne!(
            serial,
            pairwise_sum(&sim),
            "the frame no longer tells a pairwise column sum from a serial one"
        );
    }

    /// `sim_iou` of that frame from the same loop under numpy 1.23.5.
    const NUMPY_1_23_ONE_COLUMN_SIM_IOU: [u64; 12] = [
        0x3f5fe01fe01fe01e,
        0x3fb2ed12ed12ed12,
        0x3fc2ad52ad52ad52,
        0x3f9de21de21de21d,
        0x3fb9e619e619e619,
        0x3fc629d629d629d5,
        0x3face31ce31ce31c,
        0x3fc06f906f906f90,
        0x3f87e817e817e817,
        0x3fb56a956a956a94,
        0x3fc3ec13ec13ec13,
        0x3fa3ec13ec13ec13,
    ];

    /// `sim_iou` from `trackeval/metrics/hota.py`'s first loop, as hex.
    const NUMPY_1_23_SIM_IOU: [u64; 36] = [
        0x3f730d190130d191,
        0x3fb0e7d95bc609a9,
        0x3fc0892fc534ab7c,
        0x3fab56fd83ba6864,
        0x3fbe441938bfaf4b,
        0x3fa47ae147ae147b,
        0x3fbb4a1231617642,
        0x3f9a6449e59bb61a,
        0x3fb8202ae3da78a2,
        0x3f82e025c04b8097,
        0x3fb1f1a515885fb4,
        0x3fc0fc5c3562465f,
        0x3fb0d387cfecc51b,
        0x3fc0750750750751,
        0x3fa7dd49c34115b2,
        0x3fbe1e1e1e1e1e1f,
        0x3fa1cf06ada2811c,
        0x3fbb26c9b26c9b27,
        0x3f96db6db6db6db7,
        0x3fb8000000000001,
        0x3f82c9fb4d812ca0,
        0x3fb1dc47711dc477,
        0x3fc0e7d95bc609a9,
        0x3fa9d0ac19d0ac19,
        0x3fc0610fc5c35624,
        0x3fa7c382b34eda32,
        0x3fba41a41a41a41b,
        0x3fa1bb4a4046ed29,
        0x3fb79435e50d7944,
        0x3f96c16c16c16c17,
        0x3fb4c1bacf914c1c,
        0x3f82b404ad012b40,
        0x3fb1c71c71c71c73,
        0x3fc0d387cfecc51c,
        0x3fa9b5055b0bc84c,
        0x3fbb1e5f75270d05,
    ];
}
