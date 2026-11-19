# Multi-object tracking

Multi-object tracking (MOT) evaluation scores two things at once: whether each
object was found in each frame, and whether it kept one identity from frame to
frame. Three metric families divide that up — **HOTA**, **CLEAR** (MOTA,
MOTP), and **Identity** (IDF1) — and the reference implementation published
numbers come from is [TrackEval](https://github.com/JonathonLuiten/TrackEval).
hotcoco computes all three from MOTChallenge rows, in memory or from a
benchmark's folders, under TrackEval's field names, and reports them in the
same [`EvalReport`](results.md#the-evaluation-report) shape as the other
families.

This release covers MOTChallenge 2D boxes: MOT15, MOT16, MOT17, MOT20, and
data of your own in the same row format. [What is not covered
yet](#what-is-not-covered-yet) lists the rest.

## What the metrics measure

Boxes are matched frame by frame by IoU, with an optimal assignment.

- **HOTA** is `√(DetA × AssA)`, computed at 19 localization thresholds from
  0.05 to 0.95 and averaged. **DetA** is detection accuracy,
  `TP / (TP + FN + FP)`; **AssA** is how well matched boxes keep to one
  track pair over the whole sequence, so a tracker that finds every person but
  swaps identities scores high DetA and low AssA. **LocA** is the mean IoU of
  the matches.
- **MOTA** is `1 − (FN + FP + IDSW) / ground-truth boxes` at IoU 0.5, so it
  counts each identity switch once and is dominated by detection errors.
  **MOTP** is the mean IoU of its matches.
- **IDF1** is the F1 score of boxes matched under one track-to-track
  assignment fixed for the whole sequence: a tracker gets credit only for the
  frames where it follows the ground-truth track it was assigned to.

`IDSW` (identity switches), `Frag` (fragmentations), `MT`, `PT`, and `ML`
(mostly tracked, partly tracked, and mostly lost ground-truth tracks) are the
counts beside them. Every name is TrackEval's, so a number reads the same here
as in a TrackEval printout.

## Evaluate arrays

Each `update` takes one sequence: the ground truth and the predictions as 2-D
arrays of MOTChallenge rows, one box per row, and the sequence's frame count.
The columns are `frame, id, x, y, w, h, conf, class, visibility`, the layout of
`gt.txt` and of a tracker's output file. Frames count from 1.

```python
import numpy as np
from hotcoco.tracking import TrackingEval

# frame, id, x, y, w, h, conf, class, visibility
gt = np.array([
    [1, 1, 10, 10, 20, 40, 1, 1, 1.0],
    [1, 2, 60, 10, 20, 40, 1, 1, 1.0],
    [2, 1, 12, 10, 20, 40, 1, 1, 1.0],
    [2, 2, 62, 10, 20, 40, 1, 1, 1.0],
    [3, 1, 14, 10, 20, 40, 1, 1, 1.0],
    [3, 2, 64, 10, 20, 40, 1, 1, 1.0],
    [4, 1, 16, 10, 20, 40, 1, 1, 1.0],
])
# frame, id, x, y, w, h, conf: the tracker swaps its two ids at frame 3
# and loses the first pedestrian at frame 4
pred = np.array([
    [1, 7, 11, 10, 20, 40, 0.9],
    [1, 8, 61, 11, 20, 40, 0.8],
    [2, 7, 13, 10, 20, 40, 0.9],
    [2, 8, 63, 10, 20, 40, 0.8],
    [3, 8, 14, 11, 20, 40, 0.9],
    [3, 7, 65, 10, 20, 40, 0.8],
])

ev = TrackingEval()                       # no benchmark: every row is evaluated as given
ev.update("walk-01", gt, pred, num_frames=4)
ev.summarize()
```

```
HOTA             HOTA     DetA     AssA    DetRe    DetPr    AssRe    AssPr     LocA     OWTA  HOTA(0)  LocA(0) HOTALocA(0)
walk-01        52.574   80.420   34.626   81.203   94.737   46.827   53.918   90.834   52.916   55.635   90.548      50.376
COMBINED       52.574   80.420   34.626   81.203   94.737   46.827   53.918   90.834   52.916   55.635   90.548      50.376

CLEAR            MOTA     MOTP     MODA   CLR_Re   CLR_Pr      MTR      PTR      MLR    sMOTA   CLR_TP   CLR_FN   CLR_FP     IDSW       MT       PT       ML     Frag
walk-01        57.143   90.548   85.714   85.714  100.000   50.000   50.000    0.000   49.041        6        1        0        2        1        1        0        0
COMBINED       57.143   90.548   85.714   85.714  100.000   50.000   50.000    0.000   49.041        6        1        0        2        1        1        0        0

Identity         IDF1      IDR      IDP     IDTP     IDFN     IDFP
walk-01        61.538   57.143   66.667        4        3        2
COMBINED       61.538   57.143   66.667        4        3        2

Count            Dets  GT_Dets      IDs   GT_IDs
walk-01             6        7        2        2
COMBINED            6        7        2        2
```

The swap shows where each family puts it: DetA stays at 80% because every box
but one was found, AssA drops to 35%, MOTA loses two identity switches, and
IDF1 credits only the frames before the swap.

`num_frames` is required because the rows cannot say it: a frame with no boxes
still counts toward `CLR_Frames` and `FP_per_frame`, and a row in a frame past
`num_frames` is an error. A sequence name can be added once. An empty array is
a side with no boxes.

Ground truth needs the first 8 columns when a benchmark is set, since column 8
is the class, and 6 without one. Predictions need 6; a prediction row with no
class column is a pedestrian. Any integer or float dtype is read, and so is a
list of rows.

!!! warning "A one-row file loads as a 1-D array"
    `np.loadtxt("gt.txt", delimiter=",")` returns a 1-D array for a file with
    one line, which `update` refuses with `TypeError`. Pass `ndmin=2`:
    `np.loadtxt(path, delimiter=",", ndmin=2)`.

## Evaluate a MOTChallenge folder

`update_folders` reads a benchmark laid out as TrackEval reads it and
evaluates its sequences in parallel:

```
data/gt/mot_challenge/MOT17-train/
├── MOT17-02-FRCNN/
│   ├── gt/gt.txt
│   └── seqinfo.ini             # [Sequence] … seqLength=600
├── MOT17-04-FRCNN/
│   ├── gt/gt.txt
│   └── seqinfo.ini
└── …
data/trackers/mot_challenge/MOT17-train/MyTracker/data/
├── MOT17-02-FRCNN.txt
├── MOT17-04-FRCNN.txt
└── …
```

```python
ev = TrackingEval("MOT17")
ev.update_folders(
    "data/gt/mot_challenge/MOT17-train",
    "data/trackers/mot_challenge/MOT17-train/MyTracker/data",
)
ev.summarize()
```

Each folder in the ground-truth directory is a sequence, and its frame count
is the `seqLength` in its `seqinfo.ini`. To evaluate a subset, pass a
TrackEval seqmap — a header line, then one sequence name per line — as
`seqmap="seqmaps/MOT17-train.txt"`. A file is comma-separated, or
whitespace-separated on a line with no comma. A missing folder or file raises
`FileNotFoundError` naming it.

## What each benchmark setting cleans up

MOTChallenge evaluates pedestrians only, and its ground truth carries rows that
are not meant to be scored. A `benchmark` applies TrackEval's clean-up for that
benchmark before matching:

| `benchmark` | Ground truth evaluated | Predictions removed | Errors |
|---|---|---|---|
| `None` (the default) | every row | none | none from the class column, which is not read |
| `"MOT15"` | rows whose `conf` is not 0 | none | a prediction class above 1 |
| `"MOT16"`, `"MOT17"` | rows whose `conf` is not 0 and whose class is 1, pedestrian | a prediction matched to a distractor: a person on a vehicle (2), a static person (7), a distractor (8), or a reflection (12) | a prediction class above 1; a ground-truth class outside 1 to 13 in a frame with predictions |
| `"MOT20"` | as MOT17 | as MOT17, and a prediction matched to a non-motorized vehicle (6) | as MOT17 |

`conf` is truncated toward zero first, as TrackEval reads it, so `0.5` also
means "do not evaluate". A prediction on a "do not evaluate" pedestrian stays,
as a false positive. The match to a distractor is an optimal assignment at IoU
0.5 over every ground-truth row in the frame, whatever its class.

Use `None` for a dataset of your own. It is TrackEval with its clean-up off,
with two differences on input TrackEval refuses: ground truth may have 6
columns, and a prediction class above 1 is not an error. TrackEval with its
clean-up off still drops ground truth whose `conf` is 0; `None` evaluates
those rows, so drop them first to compare with that setting. A run that kept
one says so: `reference_deviations()` has an entry, `summarize()` warns, and
the report's provenance is `"extension"`.

Under every setting, predictions may have 6 columns, where TrackEval needs 7,
and malformed rows raise `ValueError`, naming the sequence, where TrackEval
truncates them or crashes: a fractional or negative frame or id, NaN in a box
or a confidence, or an unknown benchmark name. A frame outside
`1..num_frames` and the same track id twice in one frame are `ValueError`s
too, as they are errors in TrackEval.

## Results and the report

```python
r = ev.results()                    # the combined run: {"HOTA": 0.5257, "MOTA": 0.5714, "IDF1": 0.6154, "IDSW": 2, ...}
ev.sequence_results()["walk-01"]    # the same 44 fields for one sequence, plus "Frames"
ev.curves()                         # {"alphas": [0.05, ..., 0.95], "HOTA": [...], "DetA": [...], ...}
ev.curves("walk-01")                # one sequence's curves
ev.stats                            # the 44 values of results(), in tracking.METRIC_NAMES order
ev.report()                         # metrics, per_group (one entry per sequence), curves, provenance
```

`results()` is TrackEval's `COMBINED_SEQ` row under TrackEval's field names,
after the three keys every family's `results()` starts with
(`hotcoco_version`, `provenance`, `reference_deviations`): counts summed over sequences,
and scores combined as TrackEval combines them — recomputed from the summed
counts, with `AssA`, `AssRe`, `AssPr`, and `LocA` weighted by each sequence's
true positives. The HOTA family is the mean over the 19 thresholds, as
TrackEval's summary prints it; `curves()` has the value at each threshold.
Scores are fractions and counts are ints.

A sequence with no ground truth, or no predictions, left after clean-up gets
TrackEval's early-return row: every CLEAR score is 0, `MLR` is 1, `CLR_Frames`
is 0, and `LocA` is 1. The combined row recomputes from the counts, so it does
not inherit those values.

### Distributed runs

Each sequence is evaluated when it is added, and only its result is kept. An
evaluator pickles, and two evaluators over different sequences merge into the
result of one evaluator over all of them:

```python
a, b = TrackingEval("MOT17"), TrackingEval("MOT17")
a.update_folders(gt_root, pred_root, seqmap="half-a.txt")      # on one machine
b.update_folders(gt_root, pred_root, seqmap="half-b.txt")      # on another, pickled back
a.merge(b)                          # ValueError on a different benchmark or threshold, or a shared sequence
a.results()
```

## What is not covered yet

The first release of the family is MOTChallenge 2D boxes. Not in it:

- **MOTS** — mask tracking and its ignore regions.
- **TAO and COCO-video JSON** — no `video_id`, `frame_index`, or `track_id` on
  a COCO record yet; tracks are read from MOTChallenge rows only.
- **Per-class evaluation** and class averaging. With a benchmark, only
  pedestrians are scored; without one, every row is one class.
- **KITTI, BDD100K, and the other TrackEval datasets**, each of which has its
  own clean-up rules.
- **TrackMAP, VACE, IDEucl, and J&F.**
- **A CLI subcommand.** `coco` and `coco-eval` have none for tracking.

The [roadmap](https://github.com/derekallman/hotcoco/blob/main/ROADMAP.md)
has what comes next.

## Verification

Every field TrackEval's HOTA, CLEAR, Identity, and Count metrics report — per
threshold, per sequence, and combined — is compared with TrackEval's own
output on synthetic MOTChallenge sequences built to exercise each rule on this
page, and every number matches to the bit. There is no run on real MOT17 data
yet. [Tracking parity](../benchmarks.md#tracking) has what is compared and
how.
