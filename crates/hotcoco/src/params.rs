use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// A single area-range filter: a human-readable label paired with its `[min, max]` bounds.
///
/// Used in [`Params::area_ranges`] to keep labels and ranges in sync.
/// Standard COCO labels are `"all"`, `"small"`, `"medium"`, `"large"`.
#[derive(Debug, Clone)]
pub struct AreaRange {
    pub label: String,
    pub range: [f64; 2],
}

/// The type of IoU (intersection over union) computation to use.
///
/// Serializes as the lowercase name — `"bbox"`, `"segm"`, `"keypoints"`, `"obb"`
/// — the same spelling [`Display`](fmt::Display) and [`FromStr`] use. The derive
/// defaulted to the variant name (`"Bbox"`), so a saved `results.json` could not
/// be round-tripped through `FromStr` without a `.lower()` somewhere; the CLI,
/// the Python bindings and the plotting layer each patched it separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum IouType {
    /// Bounding box IoU.
    Bbox,
    /// Segmentation mask IoU (RLE-based).
    Segm,
    /// Keypoint OKS (object keypoint similarity).
    Keypoints,
    /// Oriented bounding box IoU (rotated rectangle intersection).
    Obb,
}

impl fmt::Display for IouType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IouType::Bbox => write!(f, "bbox"),
            IouType::Segm => write!(f, "segm"),
            IouType::Keypoints => write!(f, "keypoints"),
            IouType::Obb => write!(f, "obb"),
        }
    }
}

impl From<IouType> for crate::primitives::sim::SimKind {
    /// Project this eval-config axis onto the geometry axis.
    ///
    /// The two are deliberately separate types: an `IouType` says what the user
    /// asked to evaluate (Tier-1 config surface, serialized, fixed at four
    /// variants), while a [`SimKind`](crate::primitives::sim::SimKind) says which
    /// kernel computes it (`#[non_exhaustive]`, expected to grow). The mapping is
    /// total *today*, which is why this is `From` and not `TryFrom`, and why it
    /// only goes this direction.
    ///
    /// It lives here rather than beside `SimKind` so that `primitives` — the
    /// bottom layer every family builds on — does not depend on the eval-config
    /// module above it.
    fn from(iou_type: IouType) -> Self {
        use crate::primitives::sim::SimKind;
        match iou_type {
            IouType::Bbox => SimKind::Bbox,
            IouType::Segm => SimKind::Mask,
            IouType::Keypoints => SimKind::Oks,
            IouType::Obb => SimKind::Obb,
        }
    }
}

impl FromStr for IouType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bbox" => Ok(IouType::Bbox),
            "segm" => Ok(IouType::Segm),
            "keypoints" => Ok(IouType::Keypoints),
            "obb" => Ok(IouType::Obb),
            _ => Err(format!(
                "Unknown iou_type: '{}'. Expected 'bbox', 'segm', 'keypoints', or 'obb'",
                s
            )),
        }
    }
}

/// Small-object area upper bound: 32² = 1024 px².
pub(crate) const AREA_SMALL: f64 = 32.0 * 32.0;

/// Medium/large-object area boundary: 96² = 9216 px².
pub(crate) const AREA_LARGE: f64 = 96.0 * 96.0;

/// Default OKS sigmas for the 17 COCO keypoints (nose, eyes, ears, shoulders, …, ankles).
pub(crate) const KPT_OKS_SIGMAS: [f64; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107,
    0.087, 0.087, 0.089, 0.089,
];

/// `numpy.linspace(start, stop, num, endpoint=True)`, bit-for-bit.
///
/// pycocotools builds both threshold grids with `np.linspace`, and the obvious
/// Rust spellings do not reproduce it. `0.5 + 0.05 * i` disagrees at 2 of the 10
/// IoU thresholds, and `i / 100.0` disagrees at 10 of the 101 recall thresholds —
/// each by one ulp, because numpy computes a single `step` once and multiplies,
/// where those forms round twice or divide exactly.
///
/// One ulp sounds harmless and is not, on the recall grid. `rc[d] = tp / num_gt`
/// is a ratio of small integers, so it lands *exactly* on a grid point often: at
/// `num_gt = 20, tp = 7` the recall equals the old `rec_thrs[35]` bit-for-bit
/// while sitting strictly below numpy's. The two-pointer scan in
/// [`metrics::counts`](crate::metrics::counts) then stops one detection earlier
/// and reports a slightly different precision there. Neither grid is more
/// *correct* — both approximate 0.35 — so matching the reference is free, and
/// not matching it put a permanent floor under parity.
///
/// numpy's algorithm: `y[i] = i * step + start` with `step = (stop - start) /
/// (num - 1)`, then `y[num - 1] = stop` assigned exactly rather than computed.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    // numpy pins the endpoint instead of trusting the arithmetic to land on it.
    out[num - 1] = stop;
    out
}

/// Generate the default COCO IoU threshold range: 0.50, 0.55, …, 0.95.
pub(crate) fn default_iou_thrs() -> Vec<f64> {
    linspace(0.5, 0.95, 10)
}

/// COCO's 101-point recall grid: 0.00, 0.01, …, 1.00.
///
/// The x-axis every AP in the crate is interpolated onto. Public because the
/// metric functions in [`metrics`](crate::metrics) take it as a parameter, so a
/// caller reaching for them directly needs the same grid `Params` defaults to —
/// otherwise their AP is on a different axis than `COCOeval`'s and the two
/// silently disagree.
pub fn default_rec_thrs() -> Vec<f64> {
    linspace(0.0, 1.0, 101)
}

/// Largest per-point gap at which a threshold grid counts as the default grid
/// rounded through a narrower float type.
///
/// A grid computed in `f32` and read back as `f64` is at most about 3.6e-8 from
/// the default; 1e-6 clears that with room and sits far below any deliberate
/// change to a grid.
///
/// The grid is **snapped** to the default, not merely tolerated, because the
/// difference is not harmless: recall `k / n` lands exactly on a recall-grid
/// point, and a grid point one ulp higher excludes it, so `accumulate()` picks
/// the next precision. On a category with 20 ground truths that moved 240 of
/// 12,120 precision cells, by up to 0.33. A comparison with a tolerance would
/// have called such a run comparable to the reference while its numbers were
/// not; snapping makes them the reference's.
const GRID_SNAP_TOL: f64 = 1e-6;

/// `grid`, or `default` when `grid` is `default` rounded through a narrower
/// float: same length, every point within [`GRID_SNAP_TOL`]. NaN never snaps.
fn snap_to_default(grid: Vec<f64>, default: &[f64]) -> Vec<f64> {
    let rounded = grid.len() == default.len()
        && grid
            .iter()
            .zip(default)
            .all(|(g, d)| (g - d).abs() <= GRID_SNAP_TOL);
    if rounded { default.to_vec() } else { grid }
}

/// Evaluation parameters controlling IoU thresholds, area ranges, and detection limits.
///
/// Defaults match pycocotools: 10 IoU thresholds (0.50:0.05:0.95), 101 recall
/// thresholds, and standard COCO area ranges. Keypoint evaluation uses different
/// defaults (3 area ranges instead of 4, max 20 detections instead of 1/10/100).
#[derive(Debug, Clone)]
pub struct Params {
    /// IoU computation type (bbox, segm, keypoints, or obb).
    pub iou_type: IouType,
    /// Image IDs to evaluate (empty = all images).
    pub img_ids: Vec<u64>,
    /// Category IDs to evaluate (empty = all categories).
    pub cat_ids: Vec<u64>,
    /// IoU thresholds for matching (default: 0.50, 0.55, ..., 0.95).
    pub iou_thrs: Vec<f64>,
    /// Recall thresholds for interpolated precision (default: 0.00, 0.01, ..., 1.00).
    pub rec_thrs: Vec<f64>,
    /// Maximum detections per image for each summary metric (default: [1, 10, 100]).
    pub max_dets: Vec<usize>,
    /// Area ranges for filtering, each with a label and `[min, max]` bounds.
    /// Default labels: `"all"`, `"small"`, `"medium"`, `"large"` (3 ranges for keypoints).
    pub area_ranges: Vec<AreaRange>,
    /// Whether to evaluate per-category (true) or pool all categories (false).
    pub use_cats: bool,
    /// Per-keypoint OKS sigmas (default: 17 COCO keypoint sigmas).
    pub kpt_oks_sigmas: Vec<f64>,
    /// Whether to expand detections up the category hierarchy (OID mode).
    /// Default: false (only GT is expanded).
    pub expand_dt: bool,
}

impl Params {
    /// Index of the area range with the given label, or `None` if not found.
    pub fn area_range_idx(&self, label: &str) -> Option<usize> {
        self.area_ranges.iter().position(|ar| ar.label == label)
    }

    /// Index of the `"all"` area range, falling back to the first.
    ///
    /// Every whole-dataset metric is reported at `area="all"`, so this lookup runs
    /// in the summarize, report, calibration, diagnostics, and TIDE paths. The
    /// fallback matters: a caller with custom area labels and no `"all"` still gets
    /// a defined index rather than a panic, and index 0 is the widest range by
    /// convention.
    pub fn all_area_idx(&self) -> usize {
        self.area_range_idx("all").unwrap_or(0)
    }

    /// The `[min, max]` bounds of the `"all"` area range.
    ///
    /// The value-side twin of [`all_area_idx`](Self::all_area_idx), for the
    /// consumers that compare against an [`EvalImg`](crate::EvalImg)'s stored
    /// `area_rng` rather than indexing an axis.
    ///
    /// # Panics
    ///
    /// On empty `area_ranges` — an evaluator with no area ranges has no cells to
    /// filter, and every caller indexes the axis anyway.
    pub fn all_area_range(&self) -> [f64; 2] {
        self.area_ranges[self.all_area_idx()].range
    }

    /// The per-image detection cap: the largest entry in `max_dets`, or 100 if empty.
    ///
    /// The one owner of this lookup — the maximum, never `max_dets.last()`. The
    /// two agree on the sorted default `[1, 10, 100]` and diverge on unsorted
    /// input, and a run where `evaluate()` stamps cells with one and an analysis
    /// filters on the other silently matches nothing. pycocotools sidesteps the
    /// question by sorting `maxDets` inside `evaluate()`; hotcoco does not mutate
    /// caller params, because the accumulated M axis follows the caller's order.
    pub fn max_det(&self) -> usize {
        self.max_dets.iter().copied().max().unwrap_or(100)
    }

    /// Position of [`max_det`](Self::max_det)'s value in `max_dets` — the M-axis
    /// slot every headline metric is read from.
    ///
    /// The index-side twin of [`max_det`](Self::max_det), and required for the
    /// same reason: `shape.m - 1` is the *last* slot, not the slot holding the
    /// cap. The two agree on the sorted default `[1, 10, 100]` and diverge on
    /// anything else — under `max_dets = [100, 10, 1]` the last slot is
    /// `max_det = 1`, so per-class AP, the F-scores and `report()`'s PR curves
    /// would all be read at a cap the headline `AP` was not computed at.
    ///
    /// Falls back to `0` when the cap is not in the list, which only happens for
    /// an empty `max_dets`.
    pub fn max_det_idx(&self) -> usize {
        let cap = self.max_det();
        self.max_dets.iter().position(|&d| d == cap).unwrap_or(0)
    }

    /// Index of the IoU threshold *equal* to `thr` (within 1e-9), or `None`.
    ///
    /// The owner of the **exact-match** policy, paired with
    /// [`nearest_iou_thr_idx`](Self::nearest_iou_thr_idx), which snaps
    /// unconditionally. The two answer different questions and must not be
    /// interchanged: a metric named `AP50` means the 0.50 slice or nothing —
    /// reporting the 0.65 slice under that name because it happened to be
    /// closest is a wrong number, not a fallback — while a caller passing an
    /// analysis threshold ("classify at ~0.6") wants the nearest grid point it
    /// actually has.
    ///
    /// Three copies of the exact form existed — calibration, summarize, and a
    /// test — and they disagreed: two took the *first* threshold within tolerance
    /// and one took the *nearest*, so a params list holding two thresholds that
    /// close resolved `iou_thr = 0.5` differently depending on which path asked.
    ///
    /// The tolerance is not a fudge. `thr` is a caller-supplied `f64` compared
    /// against a grid built by `linspace` to match `numpy.linspace` bit-for-bit,
    /// so exact equality would reject values that are 0.5 in every sense a caller
    /// means. Nearest-wins among those within tolerance makes the answer
    /// single-valued regardless.
    pub fn iou_thr_idx(&self, thr: f64) -> Option<usize> {
        self.iou_thrs
            .iter()
            .enumerate()
            .filter(|&(_, &t)| (t - thr).abs() < 1e-9)
            .min_by(|&(_, &a), &(_, &b)| (a - thr).abs().total_cmp(&(b - thr).abs()))
            .map(|(i, _)| i)
    }

    /// Index of the IoU threshold closest to `thr`, snapping unconditionally.
    ///
    /// The owner of the **snap-always** policy — the counterpart to
    /// [`iou_thr_idx`](Self::iou_thr_idx)'s exact-within-1e-9 match. TIDE and
    /// per-image diagnostics take an analysis threshold from the user and report
    /// which grid point they landed on, so "nothing within tolerance" is not a
    /// failure there; it is a snap. Both hand-rolled the scan, and one of them
    /// tie-broke on `partial_cmp` while the other did not.
    ///
    /// Returns `0` for an empty threshold list, which is the same degenerate
    /// answer both call sites already produced via `map_or(0, …)` — there is no
    /// index to report and every consumer indexes with it.
    pub fn nearest_iou_thr_idx(&self, thr: f64) -> usize {
        self.iou_thrs
            .iter()
            .enumerate()
            .min_by(|&(_, &a), &(_, &b)| (a - thr).abs().total_cmp(&(b - thr).abs()))
            .map_or(0, |(i, _)| i)
    }

    /// Set `iou_thrs`, turning a rounded copy of the default grid into the
    /// default grid.
    ///
    /// A grid built in `f32` — `torch.linspace`, which torchmetrics uses — and
    /// read back as `f64` sits up to 3e-8 from 0.50:0.05:0.95. Such a grid
    /// becomes [`default_iou_thrs`] exactly; any other grid is kept as given.
    /// See [`GRID_SNAP_TOL`] for what counts as rounded and why snapping beats
    /// tolerating the difference.
    ///
    /// The field is public and assigning it directly stores the grid as given;
    /// this setter is what the Python bindings call.
    pub fn set_iou_thrs(&mut self, thrs: Vec<f64>) {
        self.iou_thrs = snap_to_default(thrs, &default_iou_thrs());
    }

    /// Set `rec_thrs`, turning a rounded copy of the 101-point default grid into
    /// the default grid. Same rule as [`set_iou_thrs`](Self::set_iou_thrs).
    pub fn set_rec_thrs(&mut self, thrs: Vec<f64>) {
        self.rec_thrs = snap_to_default(thrs, &default_rec_thrs());
    }

    /// Create default parameters for the given evaluation type.
    ///
    /// Keypoint evaluation uses 3 area ranges (all/medium/large) and a single
    /// max-detections value of 20. All other types use 4 area ranges
    /// (all/small/medium/large) and max-detections of [1, 10, 100].
    pub fn new(iou_type: IouType) -> Self {
        let (max_dets, area_ranges) = match iou_type {
            IouType::Keypoints => (
                vec![20],
                vec![
                    AreaRange {
                        label: "all".into(),
                        range: [0.0, 1e10],
                    },
                    AreaRange {
                        label: "medium".into(),
                        range: [AREA_SMALL, AREA_LARGE],
                    },
                    AreaRange {
                        label: "large".into(),
                        range: [AREA_LARGE, 1e10],
                    },
                ],
            ),
            _ => (
                vec![1, 10, 100],
                vec![
                    AreaRange {
                        label: "all".into(),
                        range: [0.0, 1e10],
                    },
                    AreaRange {
                        label: "small".into(),
                        range: [0.0, AREA_SMALL],
                    },
                    AreaRange {
                        label: "medium".into(),
                        range: [AREA_SMALL, AREA_LARGE],
                    },
                    AreaRange {
                        label: "large".into(),
                        range: [AREA_LARGE, 1e10],
                    },
                ],
            ),
        };

        let kpt_oks_sigmas = KPT_OKS_SIGMAS.to_vec();
        let iou_thrs = default_iou_thrs();
        let rec_thrs = default_rec_thrs();

        Params {
            iou_type,
            img_ids: Vec::new(),
            cat_ids: Vec::new(),
            iou_thrs,
            rec_thrs,
            max_dets,
            area_ranges,
            use_cats: true,
            kpt_oks_sigmas,
            expand_dt: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default grids must equal `numpy.linspace` bit-for-bit.
    ///
    /// Raw bit patterns, captured from the exact `np.linspace` calls pycocotools
    /// makes, because the whole point is the last ulp — an approximate comparison
    /// would pass against the very constructions this replaced.
    ///
    /// Regenerate with:
    /// ```text
    /// np.linspace(.5, 0.95, int(np.round((0.95-.5)/.05))+1, endpoint=True)
    /// np.linspace(.0, 1.00, int(np.round((1.00-.0)/.01))+1, endpoint=True)
    /// ```
    #[test]
    fn default_grids_match_numpy_linspace_bitwise() {
        const IOU_BITS: [u64; 10] = [
            4602678819172646912,
            4603129179135383962,
            4603579539098121011,
            4604029899060858061,
            4604480259023595110,
            4604930618986332160,
            4605380978949069210,
            4605831338911806259,
            4606281698874543308,
            4606732058837280358,
        ];
        const REC_BITS: [u64; 101] = [
            0,
            4576918229304087675,
            4581421828931458171,
            4584304132692975288,
            4585925428558828667,
            4587366580439587226,
            4588807732320345784,
            4589708452245819884,
            4590429028186199163,
            4591149604126578442,
            4591870180066957722,
            4592590756007337001,
            4593311331947716280,
            4593851763903000740,
            4594212051873190380,
            4594572339843380019,
            4594932627813569659,
            4595292915783759299,
            4595653203753948938,
            4596013491724138578,
            4596373779694328218,
            4596734067664517857,
            4597094355634707497,
            4597454643604897137,
            4597814931575086776,
            4598175219545276416,
            4598355363530371236,
            4598535507515466056,
            4598715651500560876,
            4598895795485655695,
            4599075939470750515,
            4599256083455845335,
            4599436227440940155,
            4599616371426034975,
            4599796515411129795,
            4599976659396224615,
            4600156803381319434,
            4600336947366414254,
            4600517091351509074,
            4600697235336603894,
            4600877379321698714,
            4601057523306793534,
            4601237667291888353,
            4601417811276983173,
            4601597955262077993,
            4601778099247172813,
            4601958243232267633,
            4602138387217362453,
            4602318531202457272,
            4602498675187552092,
            4602678819172646912,
            4602768891165194322,
            4602858963157741732,
            4602949035150289142,
            4603039107142836552,
            4603129179135383962,
            4603219251127931372,
            4603309323120478782,
            4603399395113026191,
            4603489467105573601,
            4603579539098121011,
            4603669611090668421,
            4603759683083215831,
            4603849755075763241,
            4603939827068310651,
            4604029899060858061,
            4604119971053405471,
            4604210043045952881,
            4604300115038500291,
            4604390187031047701,
            4604480259023595111,
            4604570331016142520,
            4604660403008689930,
            4604750475001237340,
            4604840546993784750,
            4604930618986332160,
            4605020690978879570,
            4605110762971426980,
            4605200834963974390,
            4605290906956521800,
            4605380978949069210,
            4605471050941616620,
            4605561122934164030,
            4605651194926711440,
            4605741266919258849,
            4605831338911806259,
            4605921410904353669,
            4606011482896901079,
            4606101554889448489,
            4606191626881995899,
            4606281698874543309,
            4606371770867090719,
            4606461842859638129,
            4606551914852185539,
            4606641986844732949,
            4606732058837280359,
            4606822130829827768,
            4606912202822375178,
            4607002274814922588,
            4607092346807469998,
            4607182418800017408,
        ];

        let iou = default_iou_thrs();
        assert_eq!(iou.len(), IOU_BITS.len());
        for (i, (&got, &want)) in iou.iter().zip(IOU_BITS.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want,
                "iou_thrs[{i}] = {got:?}, numpy has {:?}",
                f64::from_bits(want)
            );
        }

        let rec = default_rec_thrs();
        assert_eq!(rec.len(), REC_BITS.len());
        for (i, (&got, &want)) in rec.iter().zip(REC_BITS.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want,
                "rec_thrs[{i}] = {got:?}, numpy has {:?}",
                f64::from_bits(want)
            );
        }

        // numpy pins the endpoint rather than computing it; so must we.
        assert_eq!(iou[9], 0.95);
        assert_eq!(rec[100], 1.0);
    }

    /// `max_det_idx` must follow the *value*, not the position. The whole point
    /// is that unsorted `max_dets` puts the cap somewhere other than last.
    #[test]
    fn max_det_idx_follows_the_cap_not_the_last_slot() {
        let mut p = Params::new(IouType::Bbox);
        assert_eq!(p.max_det(), 100);
        assert_eq!(p.max_det_idx(), 2); // [1, 10, 100] — last slot, coincidentally

        p.max_dets = vec![100, 10, 1];
        assert_eq!(p.max_det(), 100);
        assert_eq!(p.max_det_idx(), 0); // not `len - 1`

        p.max_dets = vec![10, 300, 100];
        assert_eq!(p.max_det_idx(), 1);

        p.max_dets = vec![];
        assert_eq!(p.max_det_idx(), 0);
    }

    #[test]
    fn all_area_range_agrees_with_all_area_idx() {
        let p = Params::new(IouType::Bbox);
        assert_eq!(p.all_area_range(), p.area_ranges[p.all_area_idx()].range);
        assert_eq!(p.all_area_range(), [0.0, 1e10]);
    }

    /// What `torch.linspace` hands over once read back as `f64`: the default
    /// grid rounded through `f32`. The tests assert it really differs, so a
    /// passing snap test cannot be a grid that was already exact.
    fn through_f32(grid: &[f64]) -> Vec<f64> {
        grid.iter().map(|&x| f64::from(x as f32)).collect()
    }

    #[test]
    fn a_rounded_default_grid_snaps_to_the_default() {
        let mut p = Params::new(IouType::Bbox);
        let (iou, rec) = (through_f32(&p.iou_thrs), through_f32(&p.rec_thrs));
        assert_ne!(iou, default_iou_thrs(), "fixture must drift");
        assert_ne!(rec, default_rec_thrs(), "fixture must drift");
        p.set_iou_thrs(iou);
        p.set_rec_thrs(rec);
        assert_eq!(p.iou_thrs, default_iou_thrs());
        assert_eq!(p.rec_thrs, default_rec_thrs());
    }

    /// The snap exists to make *identical* grids, not to forgive different
    /// ones: a grid that departs by more than rounding must be kept as set,
    /// so `reference_deviations` still sees it.
    #[test]
    fn a_grid_beyond_rounding_is_kept_as_set() {
        let mut p = Params::new(IouType::Bbox);
        let mut grid = default_iou_thrs();
        grid[3] += 2e-6;
        p.set_iou_thrs(grid.clone());
        assert_eq!(p.iou_thrs, grid, "2e-6 is past the 1e-6 tolerance");

        let mut near = default_rec_thrs();
        near[50] += 5e-7;
        p.set_rec_thrs(near);
        assert_eq!(p.rec_thrs, default_rec_thrs(), "5e-7 is inside it");
    }

    #[test]
    fn a_grid_of_another_length_is_never_snapped() {
        let mut p = Params::new(IouType::Bbox);
        let short = through_f32(&default_rec_thrs()[..100]);
        p.set_rec_thrs(short.clone());
        assert_eq!(p.rec_thrs, short);
        p.set_iou_thrs(vec![0.5]);
        assert_eq!(p.iou_thrs, vec![0.5]);
        p.set_iou_thrs(Vec::new());
        assert!(p.iou_thrs.is_empty());
    }

    #[test]
    fn linspace_handles_degenerate_lengths() {
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(0.25, 1.0, 1), vec![0.25]);
        assert_eq!(linspace(0.0, 1.0, 2), vec![0.0, 1.0]);
    }
}
