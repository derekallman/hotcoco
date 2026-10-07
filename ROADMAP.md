# Roadmap

What's planned, in order. This page is forward-looking only — for what has already
shipped, see the [CHANGELOG](CHANGELOG.md).

hotcoco is one engine — `primitives` for similarity and matching, `metrics` for the
numbers — with a family driver per task on top. Detection ships today. Each family
below lands as an additive minor release on the same layering, so nothing about
detection changes when a sibling arrives. The families are listed in the order they
are planned; a version number is assigned when one is ready to ship.

## Near term

- **HTML evaluation report** — one self-contained HTML file per eval run:
  headline metrics, provenance, PR curves, and error impact at a glance, with
  a sortable per-category table and hover detail below the fold.
  `ev.report().to_html(path)` plus a CLI flag; print CSS covers casual PDF
  needs. Charts move from Plotly to vendored Observable Plot, browse's
  dashboard tab migrates to the same templates, and the matplotlib PDF report
  retires once the HTML report ships.

## Panoptic segmentation

PQ = SQ × RQ for unified "stuff" and "things" evaluation: per-class, things, and
stuff breakdowns, COCO panoptic JSON + PNG input (plus an RLE-native path that
needs no PNG files), and `coco panoptic eval` in the CLI. Verified against
panopticapi before release, the same way detection is verified against
pycocotools.

## Multi-object tracking

HOTA, CLEAR (MOTA/MOTP), and Identity (IDF1), verified against TrackEval, with
Track AP (TAO) as the natural extension once the video model is in. Tracking
brings the video data model — videos, tracks, `track_id` on annotations —
along with MOTChallenge and video-COCO interchange, video-aware `merge`,
`split`, and `sample`, and healthcheck rules for video datasets.

## Concept segmentation

Promptable concept evaluation in the SAM 3 style: cgF1 for images, pHOTA for
video. Gated on a feasibility spike — it ships only if a reference oracle solid
enough to verify against exists, the same bar every other family clears.

## Rust API cleanups

Rust-visible breaks ship in minor releases while the crate has no dependents
outside this repository; each one is named in the CHANGELOG. Nothing is queued
at the moment.

## Not tied to a release

- **Frozen primitives API** — the similarity kernels, matchers, and count
  structs are callable today but provisional. Once the families above have
  exercised them, their signatures freeze, and batched variants land for
  tracking-scale work.
- **Open-vocabulary detection guide** — evaluating open-vocabulary detector
  outputs (Grounding DINO, OWL-ViT, YOLO-World) with hotcoco. OV-LVIS is
  federated LVIS AP over rare categories, which hotcoco already computes; the
  work is documentation, with a `grounding` family (Recall@k over box–phrase
  matches, RefCOCO-style accuracy) to follow only if demand shows up.
- **Ecosystem backends** — a FiftyOne evaluation backend surfacing TIDE errors
  and confusion matrices in its UI; `MeanAveragePrecision(backend="hotcoco")`
  for torchmetrics; a Hugging Face `evaluate` metric module. The torchmetrics
  backend needs a live `COCO.dataset` first: for bbox+segm, torchmetrics
  switches each prediction's `area` by editing `coco_preds.dataset` in place,
  and hotcoco's `dataset` is a copy.
- **Bounded-memory evaluation** — a chunked results-file reader for
  `StreamingEval`, and an `accumulate()` whose working set does not scale with
  total detections, for evaluation sets past Objects365 scale.
- **Compact detection results** — keep a results set's detections as columns
  (ids, boxes, scores, categories) instead of one 200-byte annotation record
  each, sized for segmentations, keypoints, and extra keys a box detection
  never has. For RF-DETR's 1.5 million detections, the records and their index
  take about 370 MB, most of what evaluation adds to peak memory.
- **Browse enhancements** — model A/B overlay toggle, failure clustering by
  TIDE error type, PR-curve click-through, aggregate → category → image
  drill-down.
- **CrowdPose** — crowded-scene keypoint evaluation with the modified OKS
  crowd factor.
- **Per-sequence video breakdowns** — per-clip metric trends, following the
  tracking release's video model.

## Not planned

Caption metrics (CIDEr, SPICE) and model-in-the-loop metrics (CLIPScore, FID).
Their values depend on a model checkpoint, so no parity claim is possible — the
exact failure the `provenance` field exists to prevent.
