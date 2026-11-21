# Roadmap

What's planned, in order. This page is forward-looking only — for what has already
shipped, see the [CHANGELOG](CHANGELOG.md).

hotcoco is one engine — `primitives` for similarity and matching, `metrics` for the
numbers — with a family driver per task on top. Detection, panoptic, semantic, and tracking on MOTChallenge boxes ship today. Each
family below lands as an additive minor release on the same layering, so nothing
about the shipped families changes when a sibling arrives. The families are listed in the order they
are planned; a version number is assigned when one is ready to ship.

## Near term

- **HTML evaluation report** — one self-contained HTML file per eval run:
  headline metrics, provenance, PR curves, and error impact at a glance, with
  a sortable per-category table and hover detail below the fold.
  `ev.report().to_html(path)` plus a CLI flag; print CSS covers casual PDF
  needs. The charts are the browse dashboard's Plotly charts, which already
  draw the same six figures, so the two surfaces share one template set, and
  the matplotlib PDF report retires once the HTML report prints well.
- **Plot quality pass** — the matplotlib figures need design work beyond the
  Cyanotype color swap: typography, spacing, legends, and defaults that look
  right in a paper and a notebook without arguments.

## Multi-object tracking, beyond MOTChallenge boxes

HOTA, CLEAR, and Identity on more of what TrackEval covers, each verified
against it with a frozen fixture of its own: MOTS (masks and ignore regions),
TAO and COCO-video JSON with per-class evaluation and class averaging, the
other TrackEval datasets (KITTI, BDD100K), and Track AP as the natural
extension once the video model is in. That model — videos, tracks, `track_id`
on annotations — brings MOTChallenge and video-COCO interchange, video-aware
`merge`, `split`, and `sample`, healthcheck rules for video datasets, and a
`coco tracking` CLI subcommand. Also planned: a parity run on real MOT17
data, against TrackEval's example tracker.

## Concept segmentation

Promptable concept evaluation in the SAM 3 style: cgF1 for images, pHOTA for
video. Gated on a feasibility spike — it ships only if a reference oracle solid
enough to verify against exists, the same bar every other family clears.

## Not tied to a release

- **Frozen primitives API** — the similarity kernels, matchers, and count
  structs are callable today but provisional. Once the families above have
  exercised them, their signatures freeze, and batched variants land for
  tracking-scale work.
- **Open-vocabulary detection guide** — evaluating open-vocabulary detector
  outputs (Grounding DINO, OWL-ViT, YOLO-World) with hotcoco. OV-LVIS is
  federated LVIS AP over rare categories, which hotcoco already computes; the
  work is documentation. Grounding metrics (Recall@k over box–phrase matches,
  RefCOCO-style accuracy) would join the concept segmentation family above,
  and only if demand shows up.
- **Ecosystem backends** — a FiftyOne evaluation backend surfacing TIDE errors
  and confusion matrices in its UI; `MeanAveragePrecision(backend="hotcoco")`
  for torchmetrics; a Hugging Face `evaluate` metric module.
- **Bounded-memory evaluation** — a chunked results-file reader for
  `StreamingEval`, and an `accumulate()` whose working set does not scale with
  total detections, for evaluation sets past Objects365 scale.
- **Compact detection results** — `load_res` keeps an array of boxes as
  columns, but a results file, a list of dicts, and an array with masks still
  build one 200-byte annotation record per detection. Columns for those, masks
  included, would carry the saving to segm and to file-based evaluation.
- **Browse enhancements** — model A/B overlay toggle, failure clustering by
  TIDE error type, PR-curve click-through, aggregate → category → image
  drill-down.
- **Report tables as DataFrames** — `report()`'s `per_class` and `per_group`
  are tables in disguise (rows are categories or groups, columns are the
  metrics), and the idiom today is `pd.DataFrame(report["per_class"]).T`. A
  helper that returns those two as frames, pandas optional, the way sklearn's
  `classification_report(output_dict=True)` is used. The headline `metrics`
  stay a dict and `curves` stay arrays; neither is a table.
- **CrowdPose** — crowded-scene keypoint evaluation with the modified OKS
  crowd factor.

## Not planned

Caption metrics (CIDEr, SPICE) and model-in-the-loop metrics (CLIPScore, FID).
Their values depend on a model checkpoint, so no parity claim is possible — the
exact failure the `provenance` field exists to prevent.
