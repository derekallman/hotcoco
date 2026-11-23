# Stability and versions

hotcoco has three kinds of API, and each makes a different promise. Version
numbers follow [Semantic Versioning](https://semver.org/), with one
deliberate allowance, described below, for results that were silently wrong.

## The three tiers

| Tier | What is in it | Promise |
|---|---|---|
| Native | Each family's evaluator — `update()`, `merge()`, `compute()` — and [`Report`](../api/report.md) | Names and shapes change only in a major release |
| Shims | The pycocotools and lvis-api surface — `COCO`, `COCOeval`, `Params`, `mask`, `LVISeval`, `LVISResults`, `init_as_pycocotools()`, `init_as_lvis()` — and panopticapi's `pq_compute` | Permanent. A shim copies its reference's answers, defaults, and output; where the reference crashes or returns garbage, it raises hotcoco's error instead |
| Extras on the shims | hotcoco's own methods on `COCOeval` and the 1.4 readers on the family evaluators: `results()`, `report()`, `summary_lines()`, `stats`, `tide_errors()`, and the like | Working through 1.5, a deprecation warning from 1.6, removed at 2.0. Their replacements are on the native API |

[`hotcoco.primitives`](../api/primitives.md) and
[`hotcoco.metrics`](../api/metrics.md) are provisional: public so you can build
on them, and free to grow, but their signatures freeze only once the metric
families have exercised them.

## What a minor release may change

- **A silently wrong result may become an error or a warning.** A detection with
  no score ranked as 0 and produced a believable AP; since 1.5 it raises
  `ValueError`. These changes are marked **Breaking** in the
  [changelog](https://github.com/derekallman/hotcoco/blob/main/CHANGELOG.md), with
  what to do instead.
- **New warnings, new methods, new keys.** Additions do not change what existing
  code gets back.
- **The Rust crate's API.** A breaking change to a Rust signature can land in a
  minor release; it is marked **Breaking (Rust)** in the changelog.

A metric's value never changes in a minor release unless the old value was
wrong against the reference, and then the changelog says so under **Fixed**.
