//! The panoptic metric family: PQ, SQ, and RQ.
//!
//! [`PanopticEval`] is the driver — a port of panopticapi's `pq_compute`,
//! checked against it on COCO val2017 (`scripts/parity_panoptic.py`) and on
//! synthetic label maps in CI (`tests/test_parity_panoptic.py`). It owns the
//! control flow only: pairing ground truth with predictions by image,
//! turning each image into two label maps, and summing counts per category.
//! The matching rules are [`crate::primitives::panoptic`]'s, the formulas
//! [`crate::metrics::panoptic`]'s.
//!
//! ```no_run
//! use std::path::Path;
//! use hotcoco::panoptic::{PanopticDataset, PanopticEval};
//! # fn main() -> hotcoco::error::Result<()> {
//! let gt = PanopticDataset::from_file(Path::new("panoptic_val2017.json"))?;
//! let pred = PanopticDataset::from_file(Path::new("predictions.json"))?;
//! let mut ev = PanopticEval::new(gt, pred);
//! ev.run()?;
//! ev.summarize();                      // the panopticapi table
//! let pq = ev.result().expect("ran").all.scores.pq;
//! # Ok(())
//! # }
//! ```
//!
//! # Two ways in
//!
//! The COCO panoptic format is a JSON file plus a folder of PNG files, one per
//! image, where each pixel's color encodes a segment id —
//! [`PanopticDataset::from_file`] reads that. The other way needs no PNG
//! files: a detection-style [`Dataset`](crate::types::Dataset) whose
//! annotations carry masks, one annotation per segment, through
//! [`PanopticDataset::from_dataset`]. Both become the same label maps, so
//! the two paths score identically on the same pixels.
//!
//! # What the reference does with bad input, and what this does
//!
//! panopticapi raises on a prediction segment that is in the JSON but not
//! the PNG, on a PNG id missing from the JSON, on an unknown prediction
//! category, and on a ground-truth image with no prediction. Those are
//! errors here too, reported from [`PanopticEval::run`] with the image id.
//! Where the reference would divide by zero — a things or stuff split with
//! no evaluable category — this reports the `-1.0` "not computed" sentinel
//! instead, with `n == 0` saying why.

mod dataset;
mod raster;

pub use self::dataset::{PanopticAnnotation, PanopticDataset, SegmentInfo};

use std::collections::BTreeMap;
use std::path::Path;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::metrics::panoptic::{PqCounts, PqScores, pq_average};
use crate::primitives::panoptic::{Overlaps, Segment, VOID, match_segments};
use crate::report::{EvalReport, Provenance};
use crate::types::{Category, Segmentation, placeholder_cat_name};

use self::raster::LabelMap;

/// The headline metrics, in the order [`PanopticResult::stats`] reports them:
/// PQ, SQ, RQ over all categories, then over things, then over stuff.
pub const METRIC_NAMES: [&str; 9] = [
    "PQ", "SQ", "RQ", "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st",
];

/// PQ, SQ, RQ averaged over a set of categories, and how many were averaged.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PqSplit {
    /// The three means — the sentinel when `n == 0`.
    pub scores: PqScores,
    /// Categories with at least one segment on either side.
    pub n: usize,
}

/// What a run produced: counts per category and the three averages.
///
/// `#[non_exhaustive]`: a breakdown added later must not be a breaking change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PanopticResult {
    /// Counts for every ground-truth category, by id. A category nothing
    /// was seen for holds zeros and is left out of the averages.
    pub per_category: BTreeMap<u64, PqCounts>,
    /// Every category.
    pub all: PqSplit,
    /// Categories with `isthing: true`.
    pub things: PqSplit,
    /// Categories with `isthing: false`.
    pub stuff: PqSplit,
    /// Images evaluated: one per ground-truth annotation.
    pub n_images: usize,
}

impl PanopticResult {
    /// The nine headline values, in [`METRIC_NAMES`] order.
    pub fn stats(&self) -> [f64; 9] {
        let s = |split: &PqSplit| [split.scores.pq, split.scores.sq, split.scores.rq];
        let [a, b, c] = s(&self.all);
        let [d, e, f] = s(&self.things);
        let [g, h, i] = s(&self.stuff);
        [a, b, c, d, e, f, g, h, i]
    }
}

/// The panoptic evaluator.
pub struct PanopticEval {
    gt: PanopticDataset,
    pred: PanopticDataset,
    result: Option<PanopticResult>,
}

/// One side of one image: its label map and, for each segment the matcher
/// will see, the label painted for it. The reference's dict semantics are
/// already applied — a repeated id keeps the last record at the first
/// record's position.
struct Side<'a> {
    map: LabelMap,
    segments: Vec<(u32, &'a SegmentInfo)>,
}

/// Read-only state shared by every image's evaluation.
struct Context<'a> {
    gt_folder: Option<&'a Path>,
    pred_folder: Option<&'a Path>,
    /// `image_id -> (height, width)` from the ground truth's images: the
    /// canvas a mask-backed ground truth is painted on. A prediction is
    /// painted on whatever canvas the ground truth turned out to have.
    gt_dims: FxHashMap<u64, (u32, u32)>,
    category_ids: FxHashSet<u64>,
}

impl PanopticEval {
    /// An evaluator over two datasets. The ground truth's `categories` are
    /// the ones scored; the prediction's are ignored, as in panopticapi.
    pub fn new(gt: PanopticDataset, pred: PanopticDataset) -> Self {
        PanopticEval {
            gt,
            pred,
            result: None,
        }
    }

    /// The result of the last [`run`](Self::run), if any.
    pub fn result(&self) -> Option<&PanopticResult> {
        self.result.as_ref()
    }

    /// Evaluate every ground-truth image against its prediction.
    ///
    /// Images run in parallel; counts are summed in annotation order, so the
    /// numbers do not depend on the thread count.
    ///
    /// # Errors
    ///
    /// The conditions panopticapi raises on, each naming the image: a
    /// ground-truth image with no prediction, a predicted segment in the
    /// JSON but not in the mask or the other way around, a prediction with
    /// a category the ground truth does not define, label maps of different
    /// sizes, a PNG that cannot be read, or a mask that cannot be
    /// rasterized. Nothing is recorded then; `result()` keeps its previous
    /// value.
    pub fn run(&mut self) -> crate::error::Result<()> {
        let pred_by_image: FxHashMap<u64, &PanopticAnnotation> = self
            .pred
            .annotations
            .iter()
            .map(|a| (a.image_id, a))
            .collect();
        let mut pairs = Vec::with_capacity(self.gt.annotations.len());
        for gt_ann in &self.gt.annotations {
            let pred_ann = pred_by_image.get(&gt_ann.image_id).ok_or_else(|| {
                format!("no prediction for the image with id {}", gt_ann.image_id)
            })?;
            pairs.push((gt_ann, *pred_ann));
        }

        let ctx = Context {
            gt_folder: self.gt.folder.as_deref(),
            pred_folder: self.pred.folder.as_deref(),
            gt_dims: self
                .gt
                .images
                .iter()
                .map(|img| (img.id, (img.height, img.width)))
                .collect(),
            category_ids: self.gt.categories.iter().map(|c| c.id).collect(),
        };

        // `collect` into a `Result` stops at the first failing image instead
        // of finishing the corpus to report it.
        let per_image: Vec<Vec<(u64, PqCounts)>> = pairs
            .par_iter()
            .map(|(gt_ann, pred_ann)| eval_image(gt_ann, pred_ann, &ctx))
            .collect::<crate::error::Result<_>>()?;

        let cats: Vec<&Category> = unique_categories(&self.gt.categories).collect();
        let mut per_category: BTreeMap<u64, PqCounts> =
            cats.iter().map(|c| (c.id, PqCounts::default())).collect();
        for counts in per_image {
            for (category_id, c) in counts {
                // A ground-truth category the file does not define is counted
                // by panopticapi and never reported; the same here.
                if let Some(total) = per_category.get_mut(&category_id) {
                    total.add(&c);
                }
            }
        }

        let split = |want: Option<bool>| {
            let (scores, n) = pq_average(
                cats.iter()
                    .filter(|c| want.is_none() || c.isthing == want)
                    .map(|c| &per_category[&c.id]),
            );
            PqSplit { scores, n }
        };
        self.result = Some(PanopticResult {
            all: split(None),
            things: split(Some(true)),
            stuff: split(Some(false)),
            per_category,
            n_images: pairs.len(),
        });
        Ok(())
    }

    /// Ways this run departs from what panopticapi would compute.
    ///
    /// Empty means the numbers are comparable to the reference. The one
    /// condition today: a category without `isthing`, which the reference
    /// requires. Such a category is scored in `All` but belongs to neither
    /// split, so the things and stuff numbers no longer mean what the
    /// reference's do. Drives both the [`summarize`](Self::summarize)
    /// warnings and [`provenance`](Self::provenance).
    pub fn reference_deviations(&self) -> Vec<String> {
        let mut out = Vec::new();
        let (total, missing) = unique_categories(&self.gt.categories).fold((0, 0), |(t, m), c| {
            (t + 1, m + usize::from(c.isthing.is_none()))
        });
        if missing > 0 {
            out.push(format!(
                "{missing} of {total} categories have no `isthing`; panopticapi requires it, so \
                 the things and stuff splits (PQ_th, PQ_st, …) cover only the categories that \
                 state it and are not comparable to the reference."
            ));
        }
        out
    }

    /// [`Provenance::ParityVerified`] when [`reference_deviations`](Self::reference_deviations)
    /// is empty, [`Provenance::Extension`] otherwise.
    pub fn provenance(&self) -> Provenance {
        if self.reference_deviations().is_empty() {
            Provenance::ParityVerified
        } else {
            Provenance::Extension
        }
    }

    /// The summary table as lines, without printing — panopticapi's layout,
    /// values in percent. A split with nothing to average shows `-`.
    pub fn summarize_lines(&self) -> Vec<String> {
        let Some(result) = &self.result else {
            return Vec::new();
        };
        let row = |name: &str, cells: [String; 3], n: &str| {
            format!(
                "{name:<10}| {:>5}  {:>5}  {:>5} {n:>5}",
                cells[0], cells[1], cells[2]
            )
        };
        let mut lines = vec![
            row("", ["PQ".into(), "SQ".into(), "RQ".into()], "N"),
            "-".repeat(10 + 7 * 4),
        ];
        for (name, split) in [
            ("All", &result.all),
            ("Things", &result.things),
            ("Stuff", &result.stuff),
        ] {
            let s = split.scores;
            let cells = if s.is_computed() {
                [s.pq, s.sq, s.rq].map(|v| format!("{:.1}", 100.0 * v))
            } else {
                ["-".into(), "-".into(), "-".into()]
            };
            lines.push(row(name, cells, &split.n.to_string()));
        }
        lines
    }

    /// Print the summary table to stdout, after any comparability warnings
    /// on stderr.
    pub fn summarize(&self) {
        if self.result.is_none() {
            eprintln!("Please run run() first.");
            return;
        }
        for w in &self.reference_deviations() {
            eprintln!("Warning: {w}");
        }
        for line in self.summarize_lines() {
            println!("{line}");
        }
    }

    /// The run as an [`EvalReport`]: the nine headline metrics, PQ/SQ/RQ per
    /// category name, the three splits as groups with their `n`, no curves.
    /// Categories and splits nothing was computed for are left out rather
    /// than recorded with the sentinel.
    ///
    /// # Errors
    ///
    /// If [`run`](Self::run) has not completed.
    pub fn report(&self) -> crate::error::Result<EvalReport> {
        let result = self
            .result
            .as_ref()
            .ok_or("run() must be called before report()")?;
        let mut report = EvalReport::new("panoptic", self.provenance())
            .with_metrics(METRIC_NAMES.iter().copied().zip(result.stats()))
            .with_params(serde_json::json!({
                "n_images": result.n_images,
                "n_categories": result.per_category.len(),
                "gt_folder": self.gt.folder.as_ref().map(|p| p.display().to_string()),
                "pred_folder": self.pred.folder.as_ref().map(|p| p.display().to_string()),
            }));
        for cat in unique_categories(&self.gt.categories) {
            let Some(scores) = result.per_category[&cat.id].scores() else {
                continue;
            };
            let name = if cat.name.is_empty() {
                placeholder_cat_name(cat.id)
            } else {
                cat.name.clone()
            };
            report = report
                .with_class_metric(name.clone(), "PQ", scores.pq)
                .with_class_metric(name.clone(), "SQ", scores.sq)
                .with_class_metric(name, "RQ", scores.rq);
        }
        for (group, split) in [
            ("all", &result.all),
            ("things", &result.things),
            ("stuff", &result.stuff),
        ] {
            if split.n == 0 {
                continue;
            }
            report = report
                .with_group_metric(group, "PQ", split.scores.pq)
                .with_group_metric(group, "SQ", split.scores.sq)
                .with_group_metric(group, "RQ", split.scores.rq)
                .with_group_metric(group, "n", split.n as f64);
        }
        Ok(report)
    }
}

/// `items` in input order, each key once, with the record of the key's
/// *last* occurrence at the position of its *first* — what a Python dict
/// built from the list holds, which is how panopticapi reads both the
/// category list and each image's `segments_info`.
fn dict_order<T>(items: &[T], key: impl Fn(&T) -> u64) -> Vec<&T> {
    let mut position: FxHashMap<u64, usize> = FxHashMap::default();
    let mut out: Vec<&T> = Vec::with_capacity(items.len());
    for item in items {
        match position.get(&key(item)) {
            Some(&i) => out[i] = item,
            None => {
                position.insert(key(item), out.len());
                out.push(item);
            }
        }
    }
    out
}

/// Categories as the reference's `{id: category}` dict holds them.
fn unique_categories(categories: &[Category]) -> impl Iterator<Item = &Category> {
    dict_order(categories, |c| c.id).into_iter()
}

/// One side of one image as a label map plus the label of each segment.
///
/// A dataset with a PNG folder reads `folder / file_name`; otherwise the
/// segments' own masks are painted on `canvas`, which is the image's size
/// for the ground truth and the ground truth's map size for the prediction.
fn load_side<'a>(
    ann: &'a PanopticAnnotation,
    folder: Option<&Path>,
    canvas: Option<(u32, u32)>,
    side: &str,
) -> crate::error::Result<Side<'a>> {
    if let (Some(folder), Some(file_name)) = (folder, &ann.file_name) {
        let map = raster::read_png(&folder.join(file_name))?;
        // The reference's dict over `segments_info` applies here, where the
        // id is the PNG color. An id a 24-bit color cannot spell is not in
        // the PNG; `u32::MAX` is a label no PNG carries.
        let segments = dict_order(&ann.segments_info, |s| s.id)
            .into_iter()
            .map(|s| (u32::try_from(s.id).unwrap_or(u32::MAX), s))
            .collect();
        return Ok(Side { map, segments });
    }

    let (h, w) = match canvas {
        Some((h, w)) if h > 0 && w > 0 => (h, w),
        _ => {
            return Err(format!(
                "image {} has no height and width, so its masks cannot be rasterized",
                ann.image_id
            )
            .into());
        }
    };
    // Masks carry their own labels, so a repeated id — common in detection
    // files, where converters emit `id: 0` — paints and scores every segment.
    let segments = &ann.segments_info;
    let mut masks: Vec<(u32, &Segmentation)> = Vec::with_capacity(segments.len());
    let mut labeled = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let mask = seg.segmentation.as_ref().ok_or_else(|| {
            format!(
                "{side} segment {} in image {} has no segmentation, and no PNG folder is set \
                 to read it from",
                seg.id, ann.image_id
            )
        })?;
        let label = i as u32 + 1;
        masks.push((label, mask));
        labeled.push((label, seg));
    }
    let map = raster::paint(&masks, h, w).map_err(|e| format!("image {}: {e}", ann.image_id))?;
    Ok(Side {
        map,
        segments: labeled,
    })
}

/// Evaluate one image: per-category counts, in no particular order.
fn eval_image(
    gt_ann: &PanopticAnnotation,
    pred_ann: &PanopticAnnotation,
    ctx: &Context<'_>,
) -> crate::error::Result<Vec<(u64, PqCounts)>> {
    let image_id = gt_ann.image_id;
    let gt = load_side(
        gt_ann,
        ctx.gt_folder,
        ctx.gt_dims.get(&image_id).copied(),
        "ground-truth",
    )?;
    let pred = load_side(
        pred_ann,
        ctx.pred_folder,
        Some((gt.map.h, gt.map.w)),
        "predicted",
    )?;
    if (gt.map.h, gt.map.w) != (pred.map.h, pred.map.w) {
        return Err(format!(
            "image {image_id}: ground truth is {}×{} but the prediction is {}×{}",
            gt.map.h, gt.map.w, pred.map.h, pred.map.w
        )
        .into());
    }

    let overlaps = Overlaps::compute(&gt.map.labels, &pred.map.labels);
    let area_of = |areas: &[(u32, u64)], label: u32| {
        areas
            .binary_search_by_key(&label, |&(l, _)| l)
            .ok()
            .map(|i| areas[i].1)
    };

    // Prediction sanity checks, panopticapi's three, then areas from the map.
    let pred_areas = overlaps.pred_areas();
    let known: FxHashSet<u32> = pred.segments.iter().map(|&(label, _)| label).collect();
    for &(label, _) in &pred_areas {
        if label != VOID && !known.contains(&label) {
            return Err(format!(
                "image {image_id}: predicted segment {label} is in the PNG but not in segments_info"
            )
            .into());
        }
    }
    let mut pred_segments = Vec::with_capacity(pred.segments.len());
    for &(label, seg) in &pred.segments {
        let area = area_of(&pred_areas, label).ok_or_else(|| {
            format!(
                "image {image_id}: predicted segment {} is in segments_info but not in the mask",
                seg.id
            )
        })?;
        if !ctx.category_ids.contains(&seg.category_id) {
            return Err(format!(
                "image {image_id}: predicted segment {} has unknown category_id {}",
                seg.id, seg.category_id
            )
            .into());
        }
        pred_segments.push(Segment {
            label,
            category_id: seg.category_id,
            area,
            iscrowd: false,
        });
    }

    let gt_areas = overlaps.gt_areas();
    let gt_segments: Vec<Segment> = gt
        .segments
        .iter()
        .map(|&(label, seg)| Segment {
            label,
            category_id: seg.category_id,
            area: seg
                .area
                .unwrap_or_else(|| area_of(&gt_areas, label).unwrap_or(0)),
            iscrowd: seg.iscrowd,
        })
        .collect();

    let matches = match_segments(&gt_segments, &pred_segments, &overlaps);
    let mut counts: FxHashMap<u64, PqCounts> = FxHashMap::default();
    for (gi, _, iou) in matches.matched {
        if !iou.is_finite() {
            return Err(format!(
                "image {image_id}: ground-truth segment {} states an area smaller than its \
                 pixels overlap, so its IoU is not a number",
                gt.segments[gi].1.id
            )
            .into());
        }
        let c = counts.entry(gt_segments[gi].category_id).or_default();
        c.tp += 1;
        c.iou += iou;
    }
    for gi in matches.missed {
        counts.entry(gt_segments[gi].category_id).or_default().fn_ += 1;
    }
    for pi in matches.false_positives {
        counts.entry(pred_segments[pi].category_id).or_default().fp += 1;
    }
    Ok(counts.into_iter().collect())
}
