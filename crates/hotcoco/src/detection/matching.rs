//! Per-image matching: one (image, category, area range, max_det) cell of the
//! evaluation.
//!
//! [`evaluate.rs`](super::evaluate) is the outer driver — it resolves parameters,
//! collects the sparse (image, category) pairs, and fans out across them. This
//! module is what each of those fan-out slots runs, and it owns no control flow
//! above the single pair.
//!
//! A fan-out slot is one *pair*, not one cell: [`gather_pair`] resolves the
//! annotations and orders the detections once, then [`match_area`] runs each
//! area range over that shared view. Only the ignore/area flags and the
//! partition they induce differ between ranges, so gathering per range meant
//! re-resolving every annotation id and re-sorting every detection four times
//! for the same answer.
//!
//! The assignment algorithm itself is not here either: that is
//! [`crate::primitives::greedy::greedy_match_masked`]. What lives here is everything
//! COCO-specific *around* it — deciding which ground truths are ignored, ordering
//! detections by score, reordering the IoU matrix to match, and translating the
//! matcher's indices back into annotation ids. The matcher's contract requires the
//! caller to own exactly this, so this module is that caller.
//!
//! Naming note: `matching`, not `match` — the latter is a reserved word and
//! `mod match;` does not compile without `r#` escaping.

use std::collections::HashMap;
use std::ops::Range;

use rayon::prelude::*;

use crate::coco::COCO;
use crate::metrics::counts::descending_score_key;
use crate::params::Params;
use crate::primitives::greedy::{GreedyMatches, GtMasks, ThreshMatrix};
use crate::types::Annotation;

use super::iou::AnnIds;

/// Everything one (image, category) pair contributes that does **not** depend on
/// the area range: the resolved annotations, the detection score order, and the
/// pair's similarity matrix.
///
/// This is the range-invariant work the module docs describe, hoisted so the
/// four default area ranges share one resolution pass: resolving every
/// annotation id, sorting the detections by score, and the `(img, cat)` lookup
/// into the IoU cache. Everything that *does* vary by range lives in
/// [`MatchScratch`] instead.
///
/// [`gather_pair`] refills one in place, so a run of pairs reuses its
/// vectors rather than allocating them per pair.
#[derive(Default)]
pub(super) struct PairCell<'a> {
    img_id: u64,
    cat_id: u64,
    max_det: usize,
    /// Ground truths in load order — the order `gt_ids` arrived in.
    gt_anns: Vec<&'a Annotation>,
    /// Index into `gt_anns` -> column in the pair's IoU matrix.
    gt_iou_indices: Vec<usize>,
    /// Detections score-descending and truncated to `max_det`.
    dt_anns: Vec<&'a Annotation>,
    /// Position in `dt_anns` -> row in the pair's IoU matrix.
    dt_iou_indices: Vec<usize>,
    dt_scores: Vec<f64>,
    /// `(position in the pair's id list, detection)`, the score sort's
    /// working array.
    dt_sort: Vec<(usize, &'a Annotation)>,
    /// The pair's similarity matrix, looked up once instead of per area range.
    iou_matrix: Option<&'a IouMatrix>,
}

/// Ground truths for one cell, partitioned non-ignored-first.
///
/// The matcher's contract requires that partition: positions
/// `[0, num_not_ignored)` are non-ignored and the rest are ignored, so a
/// position's ignore flag is [`ignored`](Self::ignored). Fields suffixed
/// `_sorted` are in partitioned order; [`order`](Self::order) maps a position
/// back to the pair's load-order `gt_anns`. [`partition_gt`] refills it per
/// area range.
#[derive(Default)]
struct GtView {
    /// [`gt_flags`] per ground truth in load order: what the partition reads.
    flags: Vec<(bool, bool)>,
    /// Partitioned position -> index into the pair's `gt_anns`.
    order: Vec<usize>,
    /// Whether each GT counts toward the recall denominator — see
    /// [`EvalImg::gt_in_denominator`].
    in_denominator_sorted: Vec<bool>,
    iscrowd_sorted: Vec<bool>,
    /// Open Images only; empty otherwise. Guarded by `is_oid` at every use.
    is_group_of_sorted: Vec<bool>,
    num_not_ignored: usize,
    /// Count of `in_denominator_sorted` — [`EvalImg::num_gt_in_denominator`].
    num_in_denominator: usize,
}

impl GtView {
    fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether matching ignores the GT at a partitioned position.
    fn ignored(&self, gi: usize) -> bool {
        gi >= self.num_not_ignored
    }

    /// Annotation id at a partitioned position.
    fn id_at(&self, pair: &PairCell<'_>, gi: usize) -> u64 {
        pair.gt_anns[self.order[gi]].id
    }
}

/// One area range's working state: the ground-truth partition, the reordered
/// IoU matrix, and the match outcome. [`match_area`] refills it, so a run of
/// pairs allocates these once; allocated per cell, they were a dozen small
/// vectors, and on many small cells that cost more than the matching.
#[derive(Default)]
pub(super) struct MatchScratch {
    gt: GtView,
    /// [`ignored_unless_matched`] per detection, in score order.
    dt_ignore_unless_matched: Vec<bool>,
    /// The pair's IoU matrix in the layout the matcher reads — see
    /// [`reordered_iou`].
    iou_flat: Vec<f64>,
    /// Open Images only: the group-of flags inverted.
    phase2_eligible: Vec<bool>,
    greedy: GreedyMatches,
    outcome: MatchOutcome,
}

/// What one fan-out slot reuses from pair to pair: the gathered pair and the
/// matcher's working state.
#[derive(Default)]
pub(super) struct PairScratch<'a> {
    pair: PairCell<'a>,
    cell: MatchScratch,
}

/// What `accumulate()` reads from every (image, category) pair `evaluate()`
/// gathered, and nothing else: the score list once, and per area range the
/// ground-truth denominator plus a matched bit and an ignore bit per detection
/// per IoU threshold — in flat arenas, one header per pair pointing into them.
///
/// An [`EvalImg`] carries the same cell with every id, the ground-truth side of
/// the match, and its own copy of the scores, one per area range. This holds
/// no vector per pair and no record per area range. `evaluate()` builds it
/// through [`Cells::build`]; `EvalImg`s are built by
/// [`COCOeval::eval_imgs`](super::COCOeval::eval_imgs) on first access.
#[derive(Debug, Clone, Default)]
pub(super) struct Cells {
    /// One per pair, plus a sentinel whose offsets are the arenas' lengths,
    /// so a pair's score count is the gap to the next header.
    pairs: Vec<PairHeader>,
    /// IoU thresholds and area ranges at evaluate time: how many flags each
    /// detection has per area range, and how many area-range blocks a pair has.
    n_thr: usize,
    n_areas: usize,
    /// Every pair's scores, score-descending and cut at `max_det`, back to back.
    scores: Vec<f64>,
    /// Per pair, per area range: how many ground truths count toward recall.
    num_gt: Vec<u32>,
    /// Per pair, from a word boundary: `n_areas` blocks of `nd` fields of
    /// `2 * n_thr` bits — per detection, its matched flag at each threshold,
    /// then its ignore flag at each. Detection-major because `accumulate()`
    /// reads a detection's flags at every threshold together.
    bits: Vec<u64>,
}

/// One pair's place in the [`Cells`] arenas.
#[derive(Debug, Clone, Copy, Default)]
struct PairHeader {
    image_id: u64,
    category_id: u64,
    scores_start: u32,
    /// Word index into `bits`.
    bits_start: u32,
}

impl PairHeader {
    /// The header past the last pair: its offsets are the arenas' lengths, so
    /// the last pair's score count is the gap to it like every other's.
    fn sentinel(n_scores: usize, n_words: usize) -> Self {
        PairHeader {
            scores_start: arena_index(n_scores),
            bits_start: arena_index(n_words),
            ..PairHeader::default()
        }
    }
}

/// One (image, category, area range) cell: a pair's index in [`Cells`] and
/// which of its area ranges, by evaluate-time position.
#[derive(Debug, Clone, Copy)]
pub(super) struct CellRef {
    pair: u32,
    area: u32,
}

impl CellRef {
    pub(super) fn new(pair: usize, area: usize) -> Self {
        CellRef {
            pair: arena_index(pair),
            area: arena_index(area),
        }
    }

    pub(super) fn pair(self) -> usize {
        self.pair as usize
    }

    pub(super) fn area(self) -> usize {
        self.area as usize
    }
}

/// Arena positions are `u32`, like annotation index positions.
pub(super) fn arena_index(i: usize) -> u32 {
    u32::try_from(i).expect("cell arena positions are u32")
}

/// The bit words a pair of `nd` detections takes: `2 * n_thr` flags per
/// detection per area range, packed end to end.
fn words_for(n_thr: usize, n_areas: usize, nd: usize) -> usize {
    (2 * n_thr * n_areas * nd).div_ceil(64)
}

impl Cells {
    /// The arena for `pairs`, sized exactly before anything is written: `nd`
    /// says how many scores a pair will push (`None` for one with nothing to
    /// gather), then `fill` writes each run of pairs through its
    /// [`CellWriter`] into that run's window of every arena. With
    /// `Some(run_len)`, runs of `run_len` pairs (at least one) are counted and
    /// written in parallel; with `None`, all of `pairs` is one run on the
    /// calling thread. `fill` must push exactly the pairs `nd` admitted, with
    /// those counts.
    pub(super) fn build(
        params: &Params,
        pairs: &[(u64, u64)],
        run_len: Option<usize>,
        nd: impl Fn(u64, u64) -> Option<usize> + Sync,
        fill: impl Fn(&[(u64, u64)], &mut CellWriter<'_>) + Sync,
    ) -> Self {
        let (n_thr, n_areas) = (params.iou_thrs.len(), params.area_ranges.len());
        // Per run: pairs kept, scores, and bit words.
        let count_run = |run: &[(u64, u64)]| {
            run.iter()
                .filter_map(|&(img_id, cat_id)| nd(img_id, cat_id))
                .fold((0, 0, 0), |(p, s, w), nd| {
                    (p + 1, s + nd, w + words_for(n_thr, n_areas, nd))
                })
        };
        let runs: Vec<(usize, usize, usize)> = match run_len {
            Some(run_len) => pairs.par_chunks(run_len).map(count_run).collect(),
            None => vec![count_run(pairs)],
        };
        let (n_pairs, n_scores, n_words) =
            runs.iter().fold((0, 0, 0), |(p, s, w), &(rp, rs, rw)| {
                (p + rp, s + rs, w + rw)
            });
        let mut cells = Cells {
            pairs: vec![PairHeader::default(); n_pairs + 1],
            n_thr,
            n_areas,
            scores: vec![0.0; n_scores],
            num_gt: vec![0; n_pairs * n_areas],
            bits: vec![0; n_words],
        };
        cells.pairs[n_pairs] = PairHeader::sentinel(n_scores, n_words);
        // One writer per run, over disjoint windows of every arena.
        let (mut ph, mut sc, mut ng, mut bw) = (
            &mut cells.pairs[..n_pairs],
            &mut cells.scores[..],
            &mut cells.num_gt[..],
            &mut cells.bits[..],
        );
        let mut writers = Vec::with_capacity(runs.len());
        let (mut scores_base, mut bits_base) = (0, 0);
        for &(p, s, w) in &runs {
            let (pairs, rest) = std::mem::take(&mut ph).split_at_mut(p);
            ph = rest;
            let (scores, rest) = std::mem::take(&mut sc).split_at_mut(s);
            sc = rest;
            let (num_gt, rest) = std::mem::take(&mut ng).split_at_mut(p * n_areas);
            ng = rest;
            let (bits, rest) = std::mem::take(&mut bw).split_at_mut(w);
            bw = rest;
            writers.push(CellWriter {
                n_thr,
                n_areas,
                pairs,
                scores,
                num_gt,
                bits,
                scores_base,
                bits_base,
                n_pairs: 0,
                n_scores: 0,
                n_words: 0,
                nd: 0,
                areas_done: n_areas,
                bit: 0,
                fields: Vec::new(),
            });
            scores_base += s;
            bits_base += w;
        }
        let write = |(writer, run): (&mut CellWriter<'_>, &[(u64, u64)])| {
            fill(run, writer);
            debug_assert!(writer.is_full(), "a run wrote what its layout pass counted");
        };
        match run_len {
            Some(run_len) => writers
                .par_iter_mut()
                .zip(pairs.par_chunks(run_len))
                .for_each(write),
            None => write((&mut writers[0], pairs)),
        }
        cells
    }

    /// One arena from runs of pairs picked out of others, in the order
    /// given, with their offsets rebased: `finalize()`'s per-image gather in
    /// `StreamingEval`. Every pair's bits start on a word boundary
    /// ([`words_for`] rounds up per pair), so a run's words copy without
    /// repacking. `params` fixes the row and block counts, which every source
    /// must share.
    pub(super) fn gather<'a>(
        params: &Params,
        runs: impl IntoIterator<Item = (&'a Cells, std::ops::Range<usize>)>,
    ) -> Self {
        let (n_thr, n_areas) = (params.iou_thrs.len(), params.area_ranges.len());
        let mut out = Cells {
            pairs: Vec::new(),
            n_thr,
            n_areas,
            scores: Vec::new(),
            num_gt: Vec::new(),
            bits: Vec::new(),
        };
        for (src, range) in runs {
            debug_assert_eq!((src.n_thr, src.n_areas), (n_thr, n_areas));
            debug_assert!(range.end <= src.len());
            let (first, end) = (&src.pairs[range.start], &src.pairs[range.end]);
            let (s0, s1) = (first.scores_start as usize, end.scores_start as usize);
            let (w0, w1) = (first.bits_start as usize, end.bits_start as usize);
            let (scores_base, bits_base) = (out.scores.len(), out.bits.len());
            out.pairs
                .extend(src.pairs[range.clone()].iter().map(|h| PairHeader {
                    scores_start: arena_index(scores_base + h.scores_start as usize - s0),
                    bits_start: arena_index(bits_base + h.bits_start as usize - w0),
                    ..*h
                }));
            out.scores.extend_from_slice(&src.scores[s0..s1]);
            out.num_gt
                .extend_from_slice(&src.num_gt[range.start * n_areas..range.end * n_areas]);
            out.bits.extend_from_slice(&src.bits[w0..w1]);
        }
        out.pairs
            .push(PairHeader::sentinel(out.scores.len(), out.bits.len()));
        out
    }

    /// The arenas as little-endian bytes: per pair header (ids, then the two
    /// arena offsets, sentinel included), the scores, `num_gt`, and the bit
    /// words. The inverse of [`Cells::from_le_bytes`]; counts travel apart.
    pub(super) fn write_le_bytes(&self, out: &mut Vec<u8>) {
        for h in &self.pairs {
            out.extend_from_slice(&h.image_id.to_le_bytes());
            out.extend_from_slice(&h.category_id.to_le_bytes());
            out.extend_from_slice(&h.scores_start.to_le_bytes());
            out.extend_from_slice(&h.bits_start.to_le_bytes());
        }
        for s in &self.scores {
            out.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        for n in &self.num_gt {
            out.extend_from_slice(&n.to_le_bytes());
        }
        for w in &self.bits {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    /// `(pairs, scores, bit words)` — the counts [`Cells::from_le_bytes`] needs.
    pub(super) fn arena_sizes(&self) -> (usize, usize, usize) {
        (self.len(), self.scores.len(), self.bits.len())
    }

    /// Rebuild an arena from [`Cells::write_le_bytes`]'s output, checking
    /// every offset so that malformed input is an error here rather than an
    /// out-of-bounds read in `accumulate()`.
    pub(super) fn from_le_bytes(
        (n_thr, n_areas): (usize, usize),
        (n_pairs, n_scores, n_words): (usize, usize, usize),
        bytes: &[u8],
    ) -> Result<Self, String> {
        // Every size is checked: the counts come from the caller's header,
        // and on a 32-bit target even an honest 64-bit state can overflow.
        let sizes = (|| {
            let head = n_pairs.checked_add(1)?.checked_mul(24)?;
            let sc = n_scores.checked_mul(8)?;
            let ng = n_pairs.checked_mul(n_areas)?.checked_mul(4)?;
            let bw = n_words.checked_mul(8)?;
            let total = head.checked_add(sc)?.checked_add(ng)?.checked_add(bw)?;
            Some((head, sc, ng, total))
        })();
        let (head_len, sc_len, ng_len, want) = sizes.ok_or("arena sizes overflow")?;
        if bytes.len() != want {
            return Err(format!(
                "expected {want} bytes of cells, found {}",
                bytes.len()
            ));
        }
        let (head, rest) = bytes.split_at(head_len);
        let (sc, rest) = rest.split_at(sc_len);
        let (ng, bw) = rest.split_at(ng_len);
        let u64_at = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8 bytes"));
        let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 bytes"));
        let pairs: Vec<PairHeader> = head
            .chunks_exact(24)
            .map(|c| PairHeader {
                image_id: u64_at(&c[0..8]),
                category_id: u64_at(&c[8..16]),
                scores_start: u32_at(&c[16..20]),
                bits_start: u32_at(&c[20..24]),
            })
            .collect();
        // Offsets: start at zero, end at the arena lengths, and every pair's
        // words are exactly what its score count needs.
        let sentinel = &pairs[n_pairs];
        if pairs[0].scores_start != 0
            || pairs[0].bits_start != 0
            || sentinel.scores_start as usize != n_scores
            || sentinel.bits_start as usize != n_words
        {
            return Err("pair offsets do not span the arenas".into());
        }
        for w in pairs.windows(2) {
            let nd = (w[1].scores_start as usize)
                .checked_sub(w[0].scores_start as usize)
                .ok_or("pair score offsets decrease")?;
            let words = (w[1].bits_start as usize).checked_sub(w[0].bits_start as usize);
            // `words_for`, checked: `n_thr` and `n_areas` come from the header.
            let need = 2usize
                .checked_mul(n_thr)
                .and_then(|b| b.checked_mul(n_areas))
                .and_then(|b| b.checked_mul(nd))
                .map(|bits| bits.div_ceil(64));
            if need.is_none() || words != need {
                return Err("pair bit offsets do not match their score counts".into());
            }
        }
        Ok(Cells {
            pairs,
            n_thr,
            n_areas,
            scores: sc
                .chunks_exact(8)
                .map(|b| f64::from_bits(u64_at(b)))
                .collect(),
            num_gt: ng.chunks_exact(4).map(u32_at).collect(),
            bits: bw.chunks_exact(8).map(u64_at).collect(),
        })
    }

    /// The number of pairs.
    pub(super) fn len(&self) -> usize {
        self.pairs.len().saturating_sub(1)
    }

    /// The pair's image and category ids.
    pub(super) fn ids(&self, pair: usize) -> (u64, u64) {
        let h = &self.pairs[pair];
        (h.image_id, h.category_id)
    }

    /// The pair's scores, descending, cut at evaluate time's `max_det`.
    pub(super) fn scores(&self, pair: usize) -> &[f64] {
        let (start, end) = (
            self.pairs[pair].scores_start as usize,
            self.pairs[pair + 1].scores_start as usize,
        );
        &self.scores[start..end]
    }

    /// [`EvalImg::num_gt_in_denominator`] for the cell.
    pub(super) fn num_gt(&self, cell: CellRef) -> u32 {
        debug_assert!(cell.area() < self.n_areas);
        self.num_gt[cell.pair() * self.n_areas + cell.area()]
    }

    /// The cell's matched and ignore flags.
    pub(super) fn block(&self, cell: CellRef) -> Block<'_> {
        debug_assert!(cell.area() < self.n_areas);
        let nd = self.scores(cell.pair()).len();
        Block {
            words: &self.bits,
            base: self.pairs[cell.pair()].bits_start as usize * 64
                + cell.area() * 2 * self.n_thr * nd,
            nd,
            n_thr: self.n_thr,
        }
    }
}

/// One cell's bits: per detection, `n_thr` matched flags then `n_thr` ignore
/// flags.
pub(super) struct Block<'a> {
    words: &'a [u64],
    /// Bit offset of the first detection's flags.
    base: usize,
    nd: usize,
    n_thr: usize,
}

impl Block<'_> {
    /// Detection `d`'s `EvalImg::dt_matched` flags at the thresholds in
    /// `rows`, at most 64 of them: threshold `rows.start + i` in bit `i`.
    pub(super) fn matched(&self, d: usize, rows: Range<usize>) -> u64 {
        debug_assert!(rows.end <= self.n_thr);
        self.field(d, rows)
    }

    /// Detection `d`'s `EvalImg::dt_ignore` flags, as [`matched`](Self::matched).
    pub(super) fn ignore(&self, d: usize, rows: Range<usize>) -> u64 {
        debug_assert!(rows.end <= self.n_thr);
        self.field(d, self.n_thr + rows.start..self.n_thr + rows.end)
    }

    fn field(&self, d: usize, bits: Range<usize>) -> u64 {
        debug_assert!(d < self.nd && bits.len() <= 64);
        read_bits(
            self.words,
            self.base + d * 2 * self.n_thr + bits.start,
            bits.len(),
        )
    }
}

/// The `len` bits (at most 64) of `words` from bit `start`, the first in bit 0.
fn read_bits(words: &[u64], start: usize, len: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    let (word, offset) = (start / 64, start % 64);
    let mut bits = words[word] >> offset;
    if offset + len > 64 {
        bits |= words[word + 1] << (64 - offset);
    }
    if len < 64 {
        bits & ((1 << len) - 1)
    } else {
        bits
    }
}

/// ORs the low `len` bits (at most 64) of `value` into `words` from bit
/// `start`: [`read_bits`]'s inverse on zeroed words.
fn write_bits(words: &mut [u64], start: usize, len: usize, value: u64) {
    if len == 0 {
        return;
    }
    let (word, offset) = (start / 64, start % 64);
    words[word] |= value << offset;
    if offset + len > 64 {
        words[word + 1] |= value >> (64 - offset);
    }
}

/// One run's window of the [`Cells`] arenas, written pair by pair: a
/// [`begin_pair`](Self::begin_pair), then one [`push_area`](Self::push_area)
/// or [`push_area_unmatched`](Self::push_area_unmatched) per area range in
/// `params.area_ranges` order.
pub(super) struct CellWriter<'a> {
    n_thr: usize,
    n_areas: usize,
    pairs: &'a mut [PairHeader],
    scores: &'a mut [f64],
    num_gt: &'a mut [u32],
    bits: &'a mut [u64],
    /// Where the window starts in the whole arena: what headers record.
    scores_base: usize,
    bits_base: usize,
    /// How much of the window is written.
    n_pairs: usize,
    n_scores: usize,
    n_words: usize,
    /// The pair being written: its score count, how many of its area ranges
    /// are in, and the bit cursor within `bits`.
    nd: usize,
    areas_done: usize,
    bit: usize,
    /// [`push_area`](Self::push_area)'s working space.
    fields: Vec<u64>,
}

impl CellWriter<'_> {
    /// Start a pair with its scores; its bit words are reserved here.
    pub(super) fn begin_pair(&mut self, image_id: u64, category_id: u64, scores: &[f64]) {
        debug_assert_eq!(
            self.areas_done, self.n_areas,
            "the previous pair pushed every area range"
        );
        self.pairs[self.n_pairs] = PairHeader {
            image_id,
            category_id,
            scores_start: arena_index(self.scores_base + self.n_scores),
            bits_start: arena_index(self.bits_base + self.n_words),
        };
        self.scores[self.n_scores..self.n_scores + scores.len()].copy_from_slice(scores);
        self.nd = scores.len();
        self.areas_done = 0;
        self.bit = self.n_words * 64;
        self.n_pairs += 1;
        self.n_scores += scores.len();
        self.n_words += words_for(self.n_thr, self.n_areas, scores.len());
    }

    /// The current pair's next area range: its denominator, then each
    /// detection's matched flags and ignore flags, one row per threshold.
    pub(super) fn push_area(
        &mut self,
        num_gt: u32,
        matched: &ThreshMatrix<bool>,
        ignore: &ThreshMatrix<bool>,
    ) {
        debug_assert!(self.areas_done < self.n_areas);
        debug_assert_eq!(
            (matched.num_rows(), matched.row_len()),
            (self.n_thr, self.nd)
        );
        debug_assert_eq!((ignore.num_rows(), ignore.row_len()), (self.n_thr, self.nd));
        self.num_gt[(self.n_pairs - 1) * self.n_areas + self.areas_done] = num_gt;
        // Each detection's flags as words of up to 64 thresholds, built a whole
        // threshold row at a time (rows are what the matrices hold
        // contiguously), laid out flag, then word, then detection. The write
        // below reads them back detection-major.
        let (nd, chunks) = (self.nd, self.n_thr.div_ceil(64));
        let fields = &mut self.fields;
        fields.clear();
        fields.resize(2 * chunks * nd, 0);
        for (f, flags) in [matched, ignore].into_iter().enumerate() {
            for t in 0..self.n_thr {
                let words = &mut fields[(f * chunks + t / 64) * nd..][..nd];
                for (w, &flag) in words.iter_mut().zip(flags.row(t)) {
                    *w |= (flag as u64) << (t % 64);
                }
            }
        }
        for d in 0..nd {
            for f in 0..2 {
                for c in 0..chunks {
                    let len = (self.n_thr - c * 64).min(64);
                    write_bits(self.bits, self.bit, len, fields[(f * chunks + c) * nd + d]);
                    self.bit += len;
                }
            }
        }
        self.areas_done += 1;
    }

    /// The current pair's next area range when nothing in it can match: its
    /// denominator, no detection matched, and detection `d` ignored at every
    /// threshold where `ignored(d)`.
    pub(super) fn push_area_unmatched(&mut self, num_gt: u32, ignored: impl Fn(usize) -> bool) {
        debug_assert!(self.areas_done < self.n_areas);
        self.num_gt[(self.n_pairs - 1) * self.n_areas + self.areas_done] = num_gt;
        for d in 0..self.nd {
            // The arena starts zeroed, so the matched flags need only be skipped.
            self.bit += self.n_thr;
            if ignored(d) {
                for rows in (0..self.n_thr).step_by(64) {
                    let len = (self.n_thr - rows).min(64);
                    write_bits(self.bits, self.bit + rows, len, u64::MAX >> (64 - len));
                }
            }
            self.bit += self.n_thr;
        }
        self.areas_done += 1;
    }

    /// Whether every arena window is written to its end.
    fn is_full(&self) -> bool {
        self.areas_done == self.n_areas
            && self.n_pairs == self.pairs.len()
            && self.n_scores == self.scores.len()
            && self.n_words == self.bits.len()
    }
}

/// Per-threshold match bookkeeping — the payload of an [`EvalImg`].
#[derive(Default)]
struct MatchOutcome {
    dt_matches: ThreshMatrix<u64>,
    gt_matches: ThreshMatrix<u64>,
    dt_matched: ThreshMatrix<bool>,
    gt_matched: ThreshMatrix<bool>,
    dt_ignore: ThreshMatrix<bool>,
}

/// The pair's ground-truth and detection ids, or `None` when both are empty.
///
/// pycocotools' `evaluateImg` skips a cell only when `len(gt) == 0 and
/// len(dt) == 0` on the *raw* per-(image, category) lists — before any area
/// range ignores anything and before the `max_det` cut — and that is the only
/// skip here too. Anything narrower is wrong in a way AP never shows: a cell
/// with detections but no ground truth, every one of them outside the area
/// range, has nothing to match and moves no counter, yet its detections still
/// occupy ranks in `accumulate()`'s score order, and the score sampled at a
/// recall threshold (`eval["scores"]`) is read off that order. Dropping such
/// cells shifted those samples onto later detections.
fn pair_ids<'a>(
    ctx: &EvalImgContext<'a>,
    img_id: u64,
    cat_id: u64,
) -> Option<(AnnIds<'a>, AnnIds<'a>)> {
    let gt_ids = super::COCOeval::get_anns_static(ctx.coco_gt, ctx.params, img_id, cat_id);
    let dt_ids = super::COCOeval::get_anns_static(ctx.coco_dt, ctx.params, img_id, cat_id);
    (!gt_ids.is_empty() || !dt_ids.is_empty()).then_some((gt_ids, dt_ids))
}

/// Resolve one (image, category) pair's annotations into `pair`, once for all
/// area ranges, reusing its vectors.
///
/// `false` for a pair with no ids on either side, leaving `pair` stale: the
/// skip that holds for every range.
///
/// The detection cap is applied *after* sorting, so it keeps the highest-scoring
/// detections rather than the first-loaded ones.
fn gather_pair<'a>(
    ctx: &EvalImgContext<'a>,
    img_id: u64,
    cat_id: u64,
    max_det: usize,
    pair: &mut PairCell<'a>,
) -> bool {
    let Some((gt_ids, dt_ids)) = pair_ids(ctx, img_id, cat_id) else {
        return false;
    };
    pair.img_id = img_id;
    pair.cat_id = cat_id;
    pair.max_det = max_det;

    pair.gt_iou_indices.clear();
    pair.gt_anns.clear();
    for (iou_idx, &id) in gt_ids.iter().enumerate() {
        if let Some(ann) = ctx.coco_gt.get_ann(id) {
            pair.gt_iou_indices.push(iou_idx);
            pair.gt_anns.push(ann);
        }
    }

    pair.dt_sort.clear();
    pair.dt_sort.extend(
        dt_ids
            .iter()
            .enumerate()
            .filter_map(|(iou_idx, &id)| Some((iou_idx, ctx.coco_dt.get_ann(id)?))),
    );
    // Stable, so tied scores keep load order; `accumulate()` ranks on the same key.
    pair.dt_sort
        .sort_by_key(|&(_, ann)| descending_score_key(ann.score.unwrap_or(0.0)));
    pair.dt_sort.truncate(max_det);
    // The index holds every annotation, so nothing above drops an id:
    // `lean_scores_len` sizes the cell arenas on that.
    debug_assert_eq!(
        Some(pair.dt_sort.len()),
        lean_scores_len(ctx, img_id, cat_id, max_det)
    );

    pair.dt_iou_indices.clear();
    pair.dt_anns.clear();
    pair.dt_scores.clear();
    for &(iou_idx, ann) in &pair.dt_sort {
        pair.dt_iou_indices.push(iou_idx);
        pair.dt_anns.push(ann);
        pair.dt_scores.push(ann.score.unwrap_or(0.0));
    }

    // `evaluate()` stores a matrix only where both sides have annotations.
    pair.iou_matrix = if pair.gt_anns.is_empty() || pair.dt_anns.is_empty() {
        None
    } else {
        ctx.ious.get(&(img_id, cat_id))
    };
    true
}

/// Decide which of the pair's ground truths this area range ignores, and
/// partition them non-ignored-first into `gt`.
///
/// Ignore rules are mode-dependent: Open Images ignores group-of boxes and does
/// not care about `iscrowd`; COCO/LVIS ignore crowds, and keypoint evaluation
/// additionally ignores annotations with no labeled keypoints.
fn partition_gt(
    gt: &mut GtView,
    pair: &PairCell<'_>,
    area_rng: [f64; 2],
    is_kp: bool,
    is_oid: bool,
) {
    let anns = pair.gt_anns.as_slice();
    let GtView {
        flags,
        order,
        in_denominator_sorted,
        iscrowd_sorted,
        is_group_of_sorted,
        num_not_ignored,
        num_in_denominator,
    } = gt;

    flags.clear();
    flags.extend(
        anns.iter()
            .map(|ann| gt_flags(ann, area_rng, is_kp, is_oid)),
    );

    // Stable partition on the ignore flag: non-ignored first, load order kept
    // within each side. Tie order is observable through `evalImgs`.
    order.clear();
    order.extend((0..anns.len()).filter(|&i| !flags[i].0));
    *num_not_ignored = order.len();
    order.extend((0..anns.len()).filter(|&i| flags[i].0));

    in_denominator_sorted.clear();
    in_denominator_sorted.extend(order.iter().map(|&i| flags[i].1));
    *num_in_denominator = in_denominator_sorted.iter().filter(|&&x| x).count();
    iscrowd_sorted.clear();
    iscrowd_sorted.extend(order.iter().map(|&i| anns[i].iscrowd));
    is_group_of_sorted.clear();
    if is_oid {
        is_group_of_sorted.extend(order.iter().map(|&i| anns[i].is_group_of.unwrap_or(false)));
    }
}

/// One ground truth under one area range: whether matching ignores it, and
/// whether it counts toward the recall denominator.
///
/// `ignore` governs *matching*; `in_denominator` governs the *recall
/// denominator*. COCO's single `gtIgnore` cannot express "not matchable here"
/// and "counted" at once, which Open Images group-of boxes need: they are
/// held out of matching (the second pass in `match_cell` absorbs them) yet
/// still count as one ground truth. See `EvalImg::gt_in_denominator`.
fn gt_flags(ann: &Annotation, area_rng: [f64; 2], is_kp: bool, is_oid: bool) -> (bool, bool) {
    let a = ann.area.unwrap_or(0.0);
    let area_ignore = a < area_rng[0] || a > area_rng[1];
    if is_oid {
        (
            ann.is_group_of.unwrap_or(false) || area_ignore,
            !area_ignore,
        )
    } else {
        let mut ignore = ann.iscrowd || area_ignore;
        if is_kp {
            ignore = ignore || ann.num_visible_keypoints() == 0;
        }
        (ignore, !ignore)
    }
}

/// Whether a detection is ignored when nothing matches it: it falls outside
/// the area range, or LVIS does not count its category on the image
/// (`not_exhaustive_cat`). A match replaces this with the matched ground
/// truth's ignore flag.
fn ignored_unless_matched(ann: &Annotation, area_rng: [f64; 2], not_exhaustive_cat: bool) -> bool {
    let a = ann.area.unwrap_or(0.0);
    not_exhaustive_cat || a < area_rng[0] || a > area_rng[1]
}

/// Flag the pair's detections that are ignored unless matched under this area
/// range, into `out`.
fn area_filter_dt(
    out: &mut Vec<bool>,
    pair: &PairCell<'_>,
    area_rng: [f64; 2],
    not_exhaustive_cat: bool,
) {
    out.clear();
    out.extend(
        pair.dt_anns
            .iter()
            .map(|ann| ignored_unless_matched(ann, area_rng, not_exhaustive_cat)),
    );
}

/// Reorder the cell's IoU matrix into `flat`, the row-major `[D*G]` layout the
/// matcher expects, with rows in score order and columns non-ignored-first.
fn reordered_iou(flat: &mut Vec<f64>, iou_mat: &IouMatrix, pair: &PairCell<'_>, gt: &GtView) {
    let (d, g) = (pair.dt_anns.len(), gt.len());
    flat.clear();
    flat.resize(d * g, 0.0);
    for di in 0..d {
        // One row borrow per detection: the row index and its bounds test are
        // invariant across the whole GT scan below.
        let Some(row) = iou_mat.row(pair.dt_iou_indices[di]) else {
            continue;
        };
        for (gi_sorted, &gi_orig) in gt.order.iter().enumerate() {
            if let Some(&v) = row.get(pair.gt_iou_indices[gi_orig]) {
                flat[di * g + gi_sorted] = v;
            }
        }
    }
}

/// Run the matcher over one cell into `s.outcome`, its indices translated
/// back to annotation ids. Reads the partition and detection flags
/// [`match_area`] left in `s`.
fn match_cell(ctx: &EvalImgContext<'_>, pair: &PairCell<'_>, s: &mut MatchScratch) {
    let is_oid = ctx.is_oid;
    let MatchScratch {
        gt,
        dt_ignore_unless_matched,
        iou_flat,
        phase2_eligible,
        greedy,
        outcome,
        ..
    } = s;
    let MatchOutcome {
        dt_matches,
        gt_matches,
        dt_matched,
        gt_matched,
        dt_ignore,
    } = outcome;
    let dt_anns = pair.dt_anns.as_slice();
    let (d, g) = (dt_anns.len(), gt.len());
    let num_iou_thrs = ctx.params.iou_thrs.len();

    dt_matches.reset(num_iou_thrs, d, 0);
    gt_matches.reset(num_iou_thrs, g, 0);
    dt_matched.reset(num_iou_thrs, d, false);
    // Seeded with what an unmatched detection's ignore status is, so it holds
    // even when there is no IoU data at all; a match overwrites it below.
    dt_ignore.reset_repeat_row(num_iou_thrs, dt_ignore_unless_matched);

    let Some(iou_mat) = pair.iou_matrix else {
        // No detections and/or no ground truths: nothing matched.
        gt_matched.reset(num_iou_thrs, g, false);
        return;
    };

    reordered_iou(iou_flat, iou_mat, pair, gt);

    // The per-GT policy flags encode the mode: crowd GTs are re-matchable
    // (COCO/LVIS only); under OID `iscrowd` is irrelevant and group-of GTs are
    // held out of the fallback phase, to be matched in the separate pass below.
    //
    // Each mode's *other* mask is uniform, and `None` says so without building
    // it: materializing both was two vectors per cell to restate the matcher's
    // defaults. Only OID fills one, to invert the group-of flags.
    phase2_eligible.clear();
    if is_oid {
        phase2_eligible.extend(gt.is_group_of_sorted.iter().map(|&x| !x));
    }

    // Both matching phases share `ctx.match_floors` — pycocotools' clamped
    // thresholds. See the policy table in `primitives::greedy`.
    crate::primitives::greedy::greedy_match_into(
        iou_flat,
        d,
        g,
        gt.num_not_ignored,
        GtMasks {
            rematchable: (!is_oid).then_some(gt.iscrowd_sorted.as_slice()),
            phase2_eligible: is_oid.then_some(phase2_eligible.as_slice()),
        },
        ctx.match_floors,
        greedy,
    );

    // Translate matched indices into annotation ids + ignore flags. Unmatched
    // detections keep the `ignore_unless_matched` flag they were seeded with.
    for t_idx in 0..num_iou_thrs {
        for (di, dt_ann) in dt_anns.iter().enumerate() {
            if let Some(gi) = greedy.dt_gt[(t_idx, di)] {
                dt_matches[(t_idx, di)] = gt.id_at(pair, gi);
                gt_matches[(t_idx, gi)] = dt_ann.id;
                dt_matched[(t_idx, di)] = true;
                // A detection matched to an ignored GT is itself ignored.
                dt_ignore[(t_idx, di)] = gt.ignored(gi);
            }
        }
    }
    // The matcher's buffer becomes the outcome's; the outcome's old one is
    // the matcher's to reset next cell.
    std::mem::swap(gt_matched, &mut greedy.gt_matched);

    // Open Images second pass — group-of boxes.
    //
    // The protocol (https://storage.googleapis.com/openimages/web/evaluation.html):
    //
    //   "If at least one detection is inside group-of box a single True Positive
    //    is scored. ... Multiple correct detections inside the same group-of box
    //    is still count as a single True Positive. Otherwise, the group-of box is
    //    counted as a single False Negative."
    //
    // So a group-of box is worth exactly one ground truth: the best-scoring
    // detection inside it becomes a true positive, every other detection inside it
    // is ignored (neither TP nor FP), and if nothing is inside it the box is a
    // miss. This is the Open Images *Challenge* metric, equivalently TensorFlow's
    // `group_of_weight = 1.0`, and it is what FiftyOne implements unconditionally.
    //
    // "Inside" is IoA, not IoU — see the note in `iou.rs` where group-of GT
    // columns are flagged crowd so `sim` selects intersection-over-detection-area.
    //
    // Group-of GTs are excluded from both greedy phases (`phase2_eligible`),
    // so this is their only matching route. That exclusion is load-bearing: an IoA
    // column saturates at 1.0 for any detection inside the region, so a group-of
    // box left in phase 1 would outbid the real object a detection is sitting on
    // and turn that object into a false negative.
    //
    // Detections arrive score-descending, so the first one to claim a given box is
    // the highest-scoring one — the same choice TF makes with
    // `scores_group_of[gt_id] = max(scores_group_of[gt_id], scores[i])`.
    //
    // `is_group_of_sorted` doubles as the candidate mask: under OID `ignore` is
    // `is_group_of || area_ignore`, so every group-of box is ignored and therefore
    // already sorted into the `[num_not_ignored, g)` tail. A separate eligibility
    // vector would only restate that invariant.
    //
    // The guard skips the whole pass for cells with no group-of GT — the common
    // case, since group-of is a minority annotation — which otherwise costs a full
    // `d x g` scan per threshold for a guaranteed-empty result.
    if is_oid && gt.is_group_of_sorted.iter().any(|&x| x) {
        for (t_idx, &iou_thr) in ctx.match_floors.iter().enumerate() {
            for di in 0..d {
                if dt_matched[(t_idx, di)] {
                    continue;
                }
                // Best enclosing group-of box. The reference does
                // `np.argmax(ioa, axis=1)` then tests the threshold, which is the
                // same selection and the same first-wins tie-break that
                // `best_above_floor` owns — see its docs for why the tie matters.
                let row = &iou_flat[di * g..(di + 1) * g];
                let Some(gi) = crate::primitives::greedy::best_above_floor(
                    row,
                    &gt.is_group_of_sorted,
                    iou_thr,
                ) else {
                    continue;
                };

                dt_matches[(t_idx, di)] = gt.id_at(pair, gi);
                dt_matched[(t_idx, di)] = true;
                // `gt_matched` *is* the "already credited" flag: group-of boxes are
                // excluded from both greedy phases, so it is false on entry here and
                // only this loop ever sets it — no separate `credited` vector.
                if gt_matched[(t_idx, gi)] {
                    // The box already has its true positive; absorb this one.
                    dt_ignore[(t_idx, di)] = true;
                } else {
                    // First (highest-scoring) detection inside this box scores it.
                    dt_ignore[(t_idx, di)] = false;
                    gt_matches[(t_idx, gi)] = dt_anns[di].id;
                    gt_matched[(t_idx, gi)] = true;
                }
            }
        }
    }
}

/// Match one area range of a gathered pair into `s`: the partitioned ground
/// truth and the match outcome, with the LVIS not-exhaustive rule applied.
///
/// `not_exhaustive_cat` — when true (LVIS mode), unmatched detections are ignored
/// rather than counted as false positives.
///
/// Every gathered pair yields a cell for every area range; the one skip is in
/// [`gather_pair`].
fn match_area(
    ctx: &EvalImgContext<'_>,
    pair: &PairCell<'_>,
    area_rng: [f64; 2],
    not_exhaustive_cat: bool,
    s: &mut MatchScratch,
) {
    partition_gt(&mut s.gt, pair, area_rng, ctx.is_kp, ctx.is_oid);
    area_filter_dt(
        &mut s.dt_ignore_unless_matched,
        pair,
        area_rng,
        not_exhaustive_cat,
    );
    match_cell(ctx, pair, s);
}

/// How many scores [`push_pair_lean`] will push for a pair, or `None` when
/// [`gather_pair`] would find nothing: the layout pass of [`Cells::build`].
pub(super) fn lean_scores_len(
    ctx: &EvalImgContext<'_>,
    img_id: u64,
    cat_id: u64,
    max_det: usize,
) -> Option<usize> {
    pair_ids(ctx, img_id, cat_id).map(|(_, dt_ids)| dt_ids.len().min(max_det))
}

/// One pair under every area range, written to `cells` as the record
/// `accumulate()` reads; nothing when the pair has neither ground truth nor
/// detections.
pub(super) fn push_pair_lean<'a>(
    ctx: &EvalImgContext<'a>,
    (img_id, cat_id): (u64, u64),
    max_det: usize,
    not_exhaustive_cat: bool,
    scratch: &mut PairScratch<'a>,
    cells: &mut CellWriter<'_>,
) {
    let PairScratch { pair, cell: s } = scratch;
    if !gather_pair(ctx, img_id, cat_id, max_det, pair) {
        return;
    }
    let pair = &*pair;
    cells.begin_pair(img_id, cat_id, &pair.dt_scores);
    // Without an IoU matrix — a pair with no ground truth or no detections —
    // `match_cell` matches nothing, so the outcome needs no matcher: no
    // detection is matched, each keeps its `ignored_unless_matched` flag, and
    // the denominator counts what `gt_flags` counts. At 10x COCO val2017 that
    // is 96% of pairs. Debug builds check every such pair against the matcher.
    if pair.iou_matrix.is_none() {
        for ar in &ctx.params.area_ranges {
            let num_gt = pair
                .gt_anns
                .iter()
                .filter(|ann| gt_flags(ann, ar.range, ctx.is_kp, ctx.is_oid).1)
                .count();
            let ignored =
                |d: usize| ignored_unless_matched(pair.dt_anns[d], ar.range, not_exhaustive_cat);
            #[cfg(debug_assertions)]
            {
                match_area(ctx, pair, ar.range, not_exhaustive_cat, s);
                debug_assert_eq!(s.gt.num_in_denominator, num_gt);
                for t in 0..ctx.params.iou_thrs.len() {
                    for d in 0..pair.dt_anns.len() {
                        debug_assert!(!s.outcome.dt_matched[(t, d)]);
                        debug_assert_eq!(s.outcome.dt_ignore[(t, d)], ignored(d));
                    }
                }
            }
            cells.push_area_unmatched(num_gt as u32, ignored);
        }
        return;
    }
    for ar in &ctx.params.area_ranges {
        match_area(ctx, pair, ar.range, not_exhaustive_cat, s);
        cells.push_area(
            s.gt.num_in_denominator as u32,
            &s.outcome.dt_matched,
            &s.outcome.dt_ignore,
        );
    }
}

/// One pair under the area ranges at `area_idxs` (indices into
/// `ctx.params.area_ranges`), as full [`EvalImg`]s written into `out` — one
/// slot per index, left `None` when the pair has neither ground truth nor
/// detections.
pub(super) fn evaluate_pair_full<'a>(
    ctx: &EvalImgContext<'a>,
    (img_id, cat_id): (u64, u64),
    max_det: usize,
    not_exhaustive_cat: bool,
    area_idxs: &[usize],
    scratch: &mut PairScratch<'a>,
    out: &mut [Option<EvalImg>],
) {
    let PairScratch { pair, cell: s } = scratch;
    if !gather_pair(ctx, img_id, cat_id, max_det, pair) {
        return;
    }
    for (slot, &a_idx) in out.iter_mut().zip(area_idxs) {
        let area_rng = ctx.params.area_ranges[a_idx].range;
        match_area(ctx, pair, area_rng, not_exhaustive_cat, s);
        // The outcome matrices move out; the next cell's `reset` refills them.
        let (gt, outcome) = (&s.gt, &mut s.outcome);
        *slot = Some(EvalImg {
            image_id: pair.img_id,
            category_id: pair.cat_id,
            area_rng,
            max_det: pair.max_det,
            dt_ids: pair.dt_anns.iter().map(|a| a.id).collect(),
            gt_ids: (0..gt.len()).map(|gi| gt.id_at(pair, gi)).collect(),
            dt_matches: std::mem::take(&mut outcome.dt_matches),
            gt_matches: std::mem::take(&mut outcome.gt_matches),
            dt_matched: std::mem::take(&mut outcome.dt_matched),
            gt_matched: std::mem::take(&mut outcome.gt_matched),
            dt_scores: pair.dt_scores.clone(),
            gt_ignore: (0..gt.len()).map(|gi| gt.ignored(gi)).collect(),
            gt_in_denominator: gt.in_denominator_sorted.clone(),
            dt_ignore: std::mem::take(&mut outcome.dt_ignore),
        });
    }
}

/// One cell's similarity matrix: a row per detection and a column per ground
/// truth, in the order of the cell's id lists, in one row-major buffer. A
/// vector per row cost an allocation per detection, and with a ground truth or
/// two per cell the row headers outweighed the values.
pub(in crate::detection) struct IouMatrix {
    cols: usize,
    values: Vec<f64>,
}

impl IouMatrix {
    /// `values` as rows of `cols`.
    pub(in crate::detection) fn new(values: Vec<f64>, cols: usize) -> Self {
        debug_assert!(cols > 0 && values.len() % cols == 0);
        Self { cols, values }
    }

    /// Detection `di`'s similarities, or `None` past the last row.
    pub(in crate::detection) fn row(&self, di: usize) -> Option<&[f64]> {
        self.values.get(di * self.cols..(di + 1) * self.cols)
    }
}

/// Per-image, per-category evaluation result.
///
/// `#[non_exhaustive]`: evaluation families added later (panoptic, tracking) will
/// need fields here, and this keeps that additive rather than breaking. Construct
/// via evaluation, not by struct literal.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EvalImg {
    pub image_id: u64,
    pub category_id: u64,
    pub area_rng: [f64; 2],
    pub max_det: usize,
    /// Detection annotation IDs (sorted by score descending, truncated to max_det)
    pub dt_ids: Vec<u64>,
    /// Ground truth annotation IDs (sorted: non-ignored first, then ignored)
    pub gt_ids: Vec<u64>,
    /// Matched GT annotation id per IoU threshold: `dt_matches[(t, d)]` = GT id, or **0 as a
    /// sentinel for unmatched**. Do not use a non-zero check for presence — `Annotation.id`
    /// defaults to 0, so a real GT can have id=0. Use `dt_matched[(t, d)]` instead.
    pub dt_matches: ThreshMatrix<u64>,
    /// Matched DT annotation id per IoU threshold: `gt_matches[(t, g)]` = DT id, or **0 as a
    /// sentinel for unmatched**. Same caveat as `dt_matches`. Use `gt_matched[(t, g)]` instead.
    pub gt_matches: ThreshMatrix<u64>,
    /// Whether each detection was matched at each IoU threshold. Authoritative presence check;
    /// avoids the id=0 sentinel ambiguity in `dt_matches`.
    pub dt_matched: ThreshMatrix<bool>,
    /// Whether each GT was matched at each IoU threshold. Authoritative presence check;
    /// avoids the id=0 sentinel ambiguity in `gt_matches`.
    pub gt_matched: ThreshMatrix<bool>,
    /// Detection scores
    pub dt_scores: Vec<f64>,
    /// Whether each GT is ignored *for matching*
    pub gt_ignore: Vec<bool>,
    /// Whether each GT counts toward the recall denominator.
    ///
    /// Equal to `!gt_ignore` in every mode except Open Images, where a group-of
    /// box is held out of matching yet still counts as one ground truth — the
    /// protocol scores an undetected group-of box as a single false negative.
    /// Consumers computing `num_gt` must read this, not `gt_ignore`.
    pub gt_in_denominator: Vec<bool>,
    /// Whether each detection is ignored per IoU threshold
    pub dt_ignore: ThreshMatrix<bool>,
}

impl EvalImg {
    /// How many ground truths in this cell count toward recall.
    ///
    /// Use this rather than counting `!gt_ignore` — see
    /// [`gt_in_denominator`](Self::gt_in_denominator) for when the two differ.
    pub fn num_gt_in_denominator(&self) -> usize {
        self.gt_in_denominator.iter().filter(|&&x| x).count()
    }

    /// Whether ground truth `gi` is a *scored* miss when unmatched.
    ///
    /// The false-negative counterpart of
    /// [`num_gt_in_denominator`](Self::num_gt_in_denominator). Tallying false
    /// negatives from `!gt_ignore` instead disagrees with the recall the same
    /// evaluation reports.
    pub fn counts_as_miss(&self, gi: usize) -> bool {
        self.gt_in_denominator.get(gi).copied().unwrap_or(false)
    }
}

/// Read-only context shared across all [`gather_pair`]/[`match_area`] calls
/// within a single [`COCOeval::evaluate`](super::COCOeval::evaluate) invocation.
pub(super) struct EvalImgContext<'a> {
    pub(super) coco_gt: &'a COCO,
    pub(super) coco_dt: &'a COCO,
    pub(super) params: &'a Params,
    pub(super) ious: &'a HashMap<(u64, u64), IouMatrix>,
    /// `params.iou_type` is keypoints, and the evaluator runs Open Images:
    /// resolved once, since the ground-truth ignore rule reads both per pair.
    pub(super) is_kp: bool,
    pub(super) is_oid: bool,
    /// `params.iou_thrs` with pycocotools' match floor applied
    /// ([`crate::primitives::greedy::coco_match_floor`]). Resolved once per
    /// `evaluate()` rather than per image-category pair: this is read inside a
    /// rayon fan-out over every (category, area range, image) tuple, so deriving
    /// it at the call site would allocate a short `Vec` hundreds of thousands of
    /// times per evaluation.
    pub(super) match_floors: &'a [f64],
}
