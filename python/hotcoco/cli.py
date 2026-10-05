# PYTHON_ARGCOMPLETE_OK
"""
hotcoco command-line interface.

Usage:
    coco eval --gt <gt.json> --dt <dt.json> [--iou-type bbox|segm|keypoints]
        [--lvis] [--tide] [--calibration] [--report out.pdf] [--slices slices.json]
    coco healthcheck <annotation_file> [--dt <detections.json>]
    coco stats <annotation_file>
    coco filter <file> -o <output> [options]
    coco merge <file1> <file2> ... -o <output>
    coco split <file> -o <prefix> [options]
    coco sample <file> -o <output> [options]
    coco compare --gt <gt.json> --dt-a <a.json> --dt-b <b.json> [--bootstrap 1000]
    coco convert --from coco --to yolo --input <file> --output <dir>
    coco convert --from yolo --to coco --input <dir> --output <file> [--images-dir <dir>]
    coco convert --from oid --to coco --input <csv> --output <file> [--class-descriptions <csv>]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json as json_mod
import os
import sys
import textwrap
from typing import NoReturn

from hotcoco._style import Spinner, Timer, dim, error, fmt_metric, green, red, section, status, warning, yellow


def _table(columns, rows, footer=None):
    """Print an aligned table with ─ separators.

    columns: [("Name", "<"), ("Value", ">")]  — header text + alignment
    rows:    [["Loc", "0.0432"], ...]          — pre-formatted cell strings
    footer:  optional rows after a second separator
    """
    widths = [len(c[0]) for c in columns]
    for row in rows + (footer or []):
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            align = columns[i][1]
            parts.append(f"{cell:{align}{widths[i]}}")
        return "  " + "  ".join(parts)

    print(fmt_row([c[0] for c in columns]))
    print("  " + "  ".join("─" * w for w in widths))
    for row in rows:
        print(fmt_row(row))
    if footer is not None:
        print("  " + "  ".join("─" * w for w in widths))
        for row in footer:
            print(fmt_row(row))


def _print_findings(findings, *, tag: str, color, show_ids: bool = True, stream=None) -> None:
    """Print healthcheck findings, one line each.

    ``show_ids`` adds a follow-up line listing up to 10 affected IDs.
    """
    stream = stream if stream is not None else sys.stdout
    for finding in findings:
        badge = color(f"{tag:<5} [{finding['code']}]")
        print(f"{badge} {finding['message']}", file=stream)
        if show_ids and finding["affected_ids"]:
            ids = finding["affected_ids"]
            ids_str = ", ".join(str(i) for i in ids[:10])
            suffix = f" ... ({len(ids)} total)" if len(ids) > 10 else ""
            print(f"       IDs: {ids_str}{suffix}", file=stream)


def cmd_stats(args):
    coco = _load_coco(args.annotation_file, quiet=args.json, reraise=args.json)
    s = coco.stats()

    if args.json:
        return s

    filename = os.path.basename(args.annotation_file)

    ann_count = s["annotation_count"]
    crowd_count = s["crowd_count"]
    crowd_pct = 100.0 * crowd_count / ann_count if ann_count > 0 else 0.0

    print(filename)
    print()
    print(f"  Images:      {s['image_count']:>6,}")
    print(f"  Annotations: {ann_count:>6,}")
    print(f"  Categories:  {s['category_count']:>6,}")
    print(f"  Crowd:       {crowd_count:>6,}  ({crowd_pct:.1f}%)")

    per_cat = s["per_category"]
    if per_cat:
        shown = per_cat if args.all_cats else per_cat[:20]
        max_name_len = max(len(c["name"]) for c in shown)
        max_name_len = max(max_name_len, 8)
        print()
        label = "Per-cat"
        if not args.all_cats and len(per_cat) > 20:
            label += f" (top 20 of {len(per_cat)}, use --all-cats for full list)"
        print(f"{label}:")
        for c in shown:
            name = c["name"].ljust(max_name_len)
            print(f"  {name}  {c['ann_count']:>6,} anns   {c['img_count']:>5,} imgs")

    w = s["image_width"]
    h = s["image_height"]
    print()
    print("Image dimensions:")
    print(f"  width   min={w['min']:.0f}    max={w['max']:.0f}   mean={w['mean']:.1f}  median={w['median']:.1f}")
    print(f"  height  min={h['min']:.0f}    max={h['max']:.0f}   mean={h['mean']:.1f}  median={h['median']:.1f}")

    a = s["annotation_area"]
    print()
    print("Annotation areas:")
    print(f"  min={a['min']:.1f}   max={a['max']:.1f}   mean={a['mean']:.1f}   median={a['median']:.1f}")


def _maybe_spinner(message: str, quiet: bool):
    """Spinner unless *quiet* — `--json` runs must not narrate to the terminal."""
    return contextlib.nullcontext() if quiet else Spinner(message)


def _extension_import_failed(e: ImportError) -> NoReturn:
    """Report a failed ``from hotcoco import ...`` and exit.

    This CLI ships inside the hotcoco package, so the import cannot fail
    because hotcoco "is not installed" — it fails when the compiled extension
    is broken: missing from the install, or built for a different Python.
    """
    error(f"failed to import the hotcoco extension: {e}")
    print(
        f"  {dim('hint')}: the compiled extension is missing or built for a different"
        " Python/platform; reinstall hotcoco",
        file=sys.stderr,
    )
    sys.exit(1)


def _load_coco(path, *, quiet: bool = False, reraise: bool = False):
    """Load a COCO annotation file, printing errors and exiting on failure.

    ``quiet`` drops the spinner and the status line. ``reraise`` propagates the
    failure instead of exiting, so ``main()`` can render it as JSON.
    """
    try:
        from hotcoco import COCO
    except ImportError as e:
        if reraise:
            raise
        _extension_import_failed(e)
    try:
        with _maybe_spinner(f"Loading {dim(os.path.basename(path))}...", quiet), Timer() as t:
            coco = COCO(path)
        n_imgs = len(coco.dataset.get("images", []))
        n_anns = len(coco.dataset.get("annotations", []))
        if not quiet:
            status(
                "Loaded",
                f"{dim(os.path.basename(path))} ({n_imgs:,} images, {n_anns:,} annotations)",
                elapsed=t.elapsed,
            )
        return coco
    except Exception as e:
        if reraise:
            raise
        error(f"loading {path}: {e}")
        sys.exit(1)


def _load_res(coco, path, *, quiet: bool = False, reraise: bool = False):
    """Load detection results, printing errors and exiting on failure.

    ``quiet`` and ``reraise`` mean what they do in :func:`_load_coco`.
    """
    try:
        with _maybe_spinner(f"Loading {dim(os.path.basename(path))}...", quiet), Timer() as t:
            dt = coco.load_res(path)
        n_dets = len(dt.dataset.get("annotations", []))
        if not quiet:
            status("Loaded", f"{dim(os.path.basename(path))} ({n_dets:,} detections)", elapsed=t.elapsed)
        return dt
    except Exception as e:
        if reraise:
            raise
        error(f"loading detections: {e}")
        sys.exit(1)


def _counts(coco) -> dict:
    """Return ``{"images": n, "annotations": n}`` for a loaded COCO dataset."""
    return {"images": len(coco.dataset["images"]), "annotations": len(coco.dataset["annotations"])}


def _report_before_after(args, *, verb: str, before: dict, after: dict, elapsed: float | None = None):
    """Report a transform's image/annotation counts, before and after.

    Returns the JSON dict when ``args.json``; otherwise prints the styled
    status line plus the before/after counts and returns ``None``. ``before``
    and ``after`` are ``{"images": n, "annotations": n}`` dicts. ``elapsed``
    is passed straight to :func:`status`, so omitting it omits the timing.
    Reads the source path from ``args.annotation_file``.
    """
    if args.json:
        return {"before": before, "after": after, "output": args.output}

    status(
        verb, f"{dim(os.path.basename(args.annotation_file))} → {dim(os.path.basename(args.output))}", elapsed=elapsed
    )
    print(f"  before: {before['images']:,} images, {before['annotations']:,} annotations")
    print(f"  after:  {after['images']:,} images, {after['annotations']:,} annotations")


def cmd_filter(args):
    coco = _load_coco(args.annotation_file, quiet=args.json, reraise=args.json)
    before = _counts(coco)

    cat_ids = [int(x) for x in args.cat_ids.split(",")] if args.cat_ids else None
    img_ids = [int(x) for x in args.img_ids.split(",")] if args.img_ids else None
    area_rng = None
    if args.area_rng:
        parts = args.area_rng.split(",")
        if len(parts) != 2:
            error("--area-rng must be MIN,MAX")
            sys.exit(1)
        area_rng = [float(parts[0]), float(parts[1])]

    drop_empty = not args.keep_empty_images
    with Timer() as t:
        result = coco.filter(cat_ids=cat_ids, img_ids=img_ids, area_rng=area_rng, drop_empty_images=drop_empty)
    result.save(args.output)

    return _report_before_after(args, verb="Filtered", before=before, after=_counts(result), elapsed=t.elapsed)


def cmd_merge(args):
    from hotcoco import COCO

    cocos = [_load_coco(f, quiet=args.json, reraise=args.json) for f in args.files]
    n_imgs_total = sum(len(c.dataset["images"]) for c in cocos)
    n_anns_total = sum(len(c.dataset["annotations"]) for c in cocos)

    try:
        with Timer() as t:
            merged = COCO.merge(cocos)
    except Exception as e:
        if args.json:
            raise  # main() renders it as {"error": ...} JSON
        error(str(e))
        sys.exit(1)

    merged.save(args.output)
    n_imgs_out = len(merged.dataset["images"])
    n_anns_out = len(merged.dataset["annotations"])

    if args.json:
        return {
            "inputs": [{"file": f, **_counts(c)} for f, c in zip(args.files, cocos)],
            "output": {"file": args.output, "images": n_imgs_out, "annotations": n_anns_out},
        }

    status("Merged", f"{len(args.files)} files → {dim(os.path.basename(args.output))}", elapsed=t.elapsed)
    print(f"  input total: {n_imgs_total:,} images, {n_anns_total:,} annotations")
    print(f"  output:      {n_imgs_out:,} images, {n_anns_out:,} annotations")


def cmd_split(args):
    coco = _load_coco(args.annotation_file, quiet=args.json, reraise=args.json)
    n_imgs = len(coco.dataset["images"])

    # `is not None`, not truthiness: an explicit --test-frac 0.0 still asks for
    # a (possibly empty) three-way split and must not collapse to two-way.
    test_frac = args.test_frac
    with Timer() as t:
        result = coco.split(val_frac=args.val_frac, test_frac=test_frac, seed=args.seed)

    # Two-way splits pair with ("train", "val"); three-way adds "test" —
    # zip stops at the shorter sequence either way.
    splits = list(zip(("train", "val", "test"), result))

    split_results = {}
    if not args.json:
        status("Split", f"{dim(os.path.basename(args.annotation_file))} ({n_imgs:,} images)", elapsed=t.elapsed)
    for name, split in splits:
        out_path = f"{args.output}_{name}.json"
        split.save(out_path)
        n = len(split.dataset["images"])
        n_anns = len(split.dataset["annotations"])
        split_results[name] = {"images": n, "annotations": n_anns, "output": out_path}
        if not args.json:
            print(f"  {name}: {n:,} images, {n_anns:,} annotations → {os.path.basename(out_path)}")

    if args.json:
        return split_results


def cmd_eval(args):
    try:
        from hotcoco import COCOeval
    except ImportError as e:
        if args.json:
            raise
        _extension_import_failed(e)

    gt = _load_coco(args.gt, quiet=args.json, reraise=args.json)
    dt = _load_res(gt, args.dt, quiet=args.json, reraise=args.json)

    hc_result = None
    if args.healthcheck:
        hc_result = gt.healthcheck(dt)
        if not args.json:
            _print_findings(hc_result["errors"], tag="ERROR", color=red, show_ids=False, stream=sys.stderr)
            _print_findings(hc_result["warnings"], tag="WARN", color=yellow, show_ids=False, stream=sys.stderr)
            if hc_result["errors"] or hc_result["warnings"]:
                print(file=sys.stderr)

    ev = COCOeval(gt, dt, args.iou_type, lvis_style=args.lvis)

    if args.img_ids:
        ev.params.imgIds = [int(x) for x in args.img_ids.split(",")]
    if args.cat_ids:
        ev.params.catIds = [int(x) for x in args.cat_ids.split(",")]
    if args.no_cats:
        ev.params.useCats = False

    with _maybe_spinner(f"Evaluating {args.iou_type}...", args.json), Timer() as t:
        ev.evaluate()
        ev.accumulate()
    if not args.json:
        status("Evaluated", f"{args.iou_type}", elapsed=t.elapsed)
        print()

    lines = ev.summary_lines()
    if not args.json:
        for line in lines:
            print(line)

    slices_result = None
    if args.slices:
        with open(args.slices) as f:
            slices = json_mod.load(f)

        slices_result = ev.slice_by(slices)

        if not args.json:
            key_metrics = [k for k in ev.metric_keys() if k.startswith("AP")]
            section("Sliced Evaluation")

            cols = [("Slice", "<"), ("N", ">")] + [(km, ">") for km in key_metrics]
            rows = []
            for name in sorted(slices_result.keys()):
                if name == "_overall":
                    continue
                sr = slices_result[name]
                cells = [name, f"{sr['num_images']:,}"]
                for km in key_metrics:
                    val = sr.get(km, -1.0)
                    cell = fmt_metric(val)
                    if val >= 0:
                        delta = sr.get("delta", {}).get(km, 0.0)
                        sign = "+" if delta >= 0 else ""
                        cell = f"{cell} ({sign}{delta:.3f})"
                    cells.append(cell)
                rows.append(cells)

            ov = slices_result["_overall"]
            ov_cells = ["_overall", f"{ov['num_images']:,}"]
            for km in key_metrics:
                ov_cells.append(fmt_metric(ov.get(km, -1.0)))
            rows.append(ov_cells)

            _table(cols, rows)

    cal_result = None
    if args.calibration:
        cal_result = ev.calibration(n_bins=args.cal_bins, iou_threshold=args.cal_iou_thr)
        if not args.json:
            _print_calibration(cal_result)

    tide_result = None
    if args.tide:
        tide_result = ev.tide_errors(pos_thr=args.tide_pos_thr, bg_thr=args.tide_bg_thr)
        if not args.json:
            _print_tide(tide_result)

    diag_result = None
    if args.diagnostics:
        diag_result = ev.image_diagnostics(iou_thr=args.diag_iou_thr, score_thr=args.diag_score_thr)
        if not args.json:
            _print_diagnostics(diag_result)

    if args.report:
        try:
            from hotcoco.plot import report
        except ImportError as e:
            if args.json:
                raise  # main() renders it as {"error": ...} JSON
            error(str(e))
            print(f"  {dim('hint')}: install plot dependencies with:  pip install hotcoco[plot]", file=sys.stderr)
            sys.exit(1)
        try:
            report(ev, save_path=args.report, gt_path=args.gt, dt_path=args.dt, title=args.title)
        except Exception as e:
            if args.json:
                raise RuntimeError(f"generating report: {e}") from e
            error(f"generating report: {e}")
            sys.exit(1)
        if not args.json:
            status("Saved", f"report to {dim(args.report)}")

    if args.json:
        result = ev.results(per_class=False)
        # `results()` already carries `provenance`; the reasons are what is missing.
        # JSON output is the easiest surface to paste into a comparison table and
        # the stderr warnings do not survive a pipe, so they travel with it.
        result["reference_deviations"] = ev.reference_deviations()
        if cal_result is not None:
            result["calibration"] = cal_result
        if tide_result is not None:
            result["tide"] = tide_result
        if slices_result is not None:
            result["slices"] = slices_result
        if hc_result is not None:
            result["healthcheck"] = {"errors": hc_result["errors"], "warnings": hc_result["warnings"]}
        if diag_result is not None:
            result["diagnostics"] = {
                "label_errors": diag_result["label_errors"],
                "worst_images": sorted(
                    [{"image_id": k, **v} for k, v in diag_result["img_summary"].items()], key=lambda x: x["f1"]
                )[:20],
                "iou_thr": diag_result["iou_thr"],
            }
        return result


def _print_tide(te):
    # Imported here rather than at module scope so plain CLI invocations do
    # not import the plot package. A --tide run still pays for it: importing
    # `hotcoco.plot.core` runs the package __init__, which pulls numpy (via
    # plot.data). That is acceptable on the one path that asked for TIDE.
    from hotcoco.plot.core import TIDE_ERROR_ORDER

    delta = te["delta_ap"]
    counts = te["counts"]
    section(
        "TIDE Error Analysis",
        f"pos_thr={te['pos_thr']:.2f}, bg_thr={te['bg_thr']:.2f}, baseline_AP={te['ap_base']:.4f}",
    )
    _table(
        [("Type", "<"), ("ΔAP", ">"), ("Count", ">")],
        [[et, f"{delta.get(et, 0.0):.4f}", f"{counts.get(et, 0):,}"] for et in TIDE_ERROR_ORDER],
        footer=[["FP", f"{delta.get('FP', 0.0):.4f}", ""], ["FN", f"{delta.get('FN', 0.0):.4f}", ""]],
    )


def _print_calibration(cal):
    section(
        "Calibration Analysis",
        f"iou_thr={cal['iou_threshold']:.2f}, bins={cal['n_bins']}, detections={cal['num_detections']:,}",
    )
    print(f"  ECE: {cal['ece']:.4f}")
    print(f"  MCE: {cal['mce']:.4f}")
    print()
    bin_rows = []
    for b in cal["bins"]:
        label = f"[{b['bin_lower']:.1f}, {b['bin_upper']:.1f})"
        if b["count"] > 0:
            bin_rows.append([label, f"{b['avg_confidence']:.3f}", f"{b['avg_accuracy']:.3f}", f"{b['count']:,}"])
        else:
            bin_rows.append([label, "─", "─", "0"])
    _table([("Bin", ">"), ("Conf", ">"), ("Acc", ">"), ("Count", ">")], bin_rows)

    per_cat = cal.get("per_category", {})
    if per_cat:
        sorted_cats = sorted(per_cat.items(), key=lambda x: x[1], reverse=True)
        top = sorted_cats[:10]
        print(f"\n  Per-category ECE {dim('(top 10 worst-calibrated)')}:")
        for name, ece in top:
            print(f"    {name:<20} {ece:.4f}")


def _print_diagnostics(diag):
    summaries = diag["img_summary"]
    n_images = len(summaries)
    label_errors = diag["label_errors"]
    wrong = [le for le in label_errors if le["type"] == "wrong_label"]
    missing = [le for le in label_errors if le["type"] == "missing_annotation"]

    section("Per-Image Diagnostics", f"iou_thr={diag['iou_thr']:.2f}, images={n_images:,}")

    # F1 distribution buckets
    f1s = [s["f1"] for s in summaries.values()]
    poor = sum(1 for f in f1s if f < 0.5)
    moderate = sum(1 for f in f1s if 0.5 <= f <= 0.8)
    good = sum(1 for f in f1s if f > 0.8)
    print(f"  F1 distribution:  {poor:,} poor (<0.5)  {moderate:,} moderate (0.5–0.8)  {good:,} good (>0.8)")

    # Worst images by F1 — the list the --help text promises, and the same
    # ordering the --json output ships as "worst_images".
    worst = sorted(summaries.items(), key=lambda kv: kv[1]["f1"])[:10]
    if worst:
        print(f"\n  Worst images {dim('(by F1)')}:")
        _table(
            [("Image", ">"), ("F1", ">"), ("TP", ">"), ("FP", ">"), ("FN", ">")],
            [[str(img_id), f"{s['f1']:.3f}", f"{s['tp']:,}", f"{s['fp']:,}", f"{s['fn']:,}"] for img_id, s in worst],
        )

    # Label error summary
    score_thr = diag.get("score_thr", 0.5)
    print(f"\n  Label errors:     {len(label_errors):,} candidates (score ≥ {score_thr:.2f})")
    if wrong:
        top_wrong = ", ".join(f"{le['dt_category']}→{le['gt_category']}" for le in wrong[:3])
        print(f"    wrong_label:        {len(wrong)}  (top: {top_wrong})")
    if missing:
        from collections import Counter

        cat_counts = Counter(le["dt_category"] for le in missing)
        top_cats = ", ".join(f"{cat} {n}" for cat, n in cat_counts.most_common(5))
        print(f"    missing_annotation: {len(missing):,}  (top categories: {top_cats})")
    if not wrong and not missing:
        print("    (none found)")

    print(f"\n  {dim('Tip: use ev.image_diagnostics() or coco explore --dt for interactive analysis.')}")


# Inbound conversions (X → COCO): display label and the loader to call. Every
# one of these has the same body — load, save, count, report — so only the parts
# that actually differ live here.
_TO_COCO = {
    "yolo": ("YOLO", lambda COCO, args: COCO.from_yolo(args.input, images_dir=args.images_dir)),
    "voc": ("VOC", lambda COCO, args: COCO.from_voc(args.input)),
    "cvat": ("CVAT", lambda COCO, args: COCO.from_cvat(args.input)),
    "dota": ("DOTA", lambda COCO, args: COCO.from_dota(args.input, images_dir=args.images_dir)),
    "oid": (
        "Open Images",
        lambda COCO, args: COCO.from_oid(
            args.input, class_descriptions=args.class_descriptions, images_dir=args.images_dir
        ),
    ),
}

# Outbound conversions (COCO → X): display label, the writer method, how to
# summarize its stats, whether the input/output paths are echoed, and which
# stat keys get a detail line when non-zero (prefixes carry their own padding).
_FROM_COCO = {
    "yolo": (
        "YOLO",
        "to_yolo",
        lambda s: f"{s['annotations']:,} annotations",
        True,
        (("skipped (crowd):   ", "skipped_crowd"), ("skipped (no bbox): ", "skipped_no_bbox")),
    ),
    "voc": (
        "VOC",
        "to_voc",
        lambda s: f"{s['annotations']:,} annotations",
        True,
        (("crowd → difficult: ", "crowd_as_difficult"), ("skipped (no bbox): ", "skipped_no_bbox")),
    ),
    "cvat": (
        "CVAT",
        "to_cvat",
        lambda s: f"{s['boxes']:,} boxes, {s['polygons']:,} polygons",
        False,
        (("skipped (no geometry): ", "skipped_no_geometry"), ("skipped (degenerate):  ", "skipped_degenerate")),
    ),
    "dota": (
        "DOTA",
        "to_dota",
        lambda s: f"{s['annotations']:,} oriented boxes",
        True,
        (("skipped (no obb):  ", "skipped_no_obb"),),
    ),
    "oid": (
        "Open Images",
        "to_oid",
        lambda s: f"{s['annotations']:,} annotations",
        False,
        (("group-of boxes:    ", "group_of"), ("skipped (no bbox): ", "skipped_no_bbox")),
    ),
}


def cmd_convert(args):
    from_fmt = args.from_fmt
    to_fmt = args.to_fmt

    if from_fmt == "coco" and to_fmt in _FROM_COCO:
        label, method, summarize, show_paths, details = _FROM_COCO[to_fmt]

        coco = _load_coco(args.input, quiet=args.json, reraise=args.json)
        try:
            with Timer() as t:
                stats = getattr(coco, method)(args.output)
        except Exception as e:
            if args.json:
                raise
            error(str(e))
            sys.exit(1)

        if args.json:
            return {"direction": f"coco_to_{to_fmt}", "input": args.input, "output": args.output, **stats}

        status("Converted", f"COCO → {label} ({summarize(stats)})", elapsed=t.elapsed)
        if show_paths:
            print(f"  input:       {os.path.basename(args.input)}")
            print(f"  output dir:  {args.output}")
        for prefix, key in details:
            if stats[key] > 0:
                print(f"  {prefix}{stats[key]:,}")
        return None

    if to_fmt == "coco" and from_fmt in _TO_COCO:
        label, load = _TO_COCO[from_fmt]

        try:
            from hotcoco import COCO
        except ImportError as e:
            if args.json:
                raise
            _extension_import_failed(e)
        try:
            with _maybe_spinner(f"Converting {label} → COCO...", args.json), Timer() as t:
                coco = load(COCO, args)
        except Exception as e:
            if args.json:
                raise
            error(str(e))
            sys.exit(1)
        try:
            coco.save(args.output)
        except Exception as e:
            if args.json:
                raise RuntimeError(f"saving {args.output}: {e}") from e
            error(f"saving {args.output}: {e}")
            sys.exit(1)

        n_imgs = len(coco.dataset["images"])
        n_anns = len(coco.dataset["annotations"])

        if args.json:
            return {
                "direction": f"{from_fmt}_to_coco",
                "input": args.input,
                "output": args.output,
                "images": n_imgs,
                "annotations": n_anns,
            }

        status("Converted", f"{label} → COCO ({n_imgs:,} images, {n_anns:,} annotations)", elapsed=t.elapsed)
        return None

    error(f"unsupported conversion: {from_fmt} → {to_fmt}")
    sys.exit(1)


def cmd_healthcheck(args):
    coco = _load_coco(args.annotation_file, quiet=args.json, reraise=args.json)

    dt_coco = _load_res(coco, args.dt, quiet=args.json, reraise=args.json) if args.dt else None

    report = coco.healthcheck(dt_coco)

    # CI-gate contract: ERROR findings exit 1 (in both output modes);
    # warnings alone exit 0. Advertised in the subcommand's --help text.
    exit_code = 1 if report["errors"] else 0

    if args.json:
        print(json_mod.dumps(report, indent=2))
        sys.exit(exit_code)

    _print_findings(report["errors"], tag="ERROR", color=red)
    _print_findings(report["warnings"], tag="WARN", color=yellow)

    s = report["summary"]
    print()
    print(f"  Images:        {s['num_images']:>6,}")
    print(f"  Annotations:   {s['num_annotations']:>6,}")
    print(f"  Categories:    {s['num_categories']:>6,}")
    print(f"  No annotations:{s['images_without_annotations']:>6,}")

    cats = s["category_counts"]
    if len(cats) >= 2:
        top_name, top_count = cats[0]
        bot_name, bot_count = cats[-1]
        print(
            f"  Cat imbalance: {s['imbalance_ratio']:>8.1f}x  ({top_name}: {top_count:,} / {bot_name}: {bot_count:,})"
        )
    else:
        print(f"  Cat imbalance: {s['imbalance_ratio']:>8.1f}x")

    if not report["errors"] and not report["warnings"]:
        print(f"\n{green('All checks passed.')}")

    if exit_code:
        sys.exit(exit_code)


def cmd_explore(args):
    try:
        from hotcoco.browse import _require_browse_deps

        _require_browse_deps()
    except ImportError:
        error("browse dependencies required. Install with: pip install hotcoco[browse]")
        sys.exit(1)

    if not os.path.isdir(args.images):
        error(f"images directory not found: {args.images}")
        sys.exit(1)

    coco = _load_coco(args.gt)
    coco.image_dir = args.images

    dt_coco = _load_res(coco, args.dt) if args.dt else None

    # Run evaluation unless --no-eval
    coco_eval = None
    if dt_coco is not None and not args.no_eval:
        try:
            from hotcoco import COCOeval

            ev = COCOeval(coco, dt_coco, args.iou_type)
            with Spinner(f"Evaluating {args.iou_type}..."), Timer() as t:
                ev.evaluate()
            coco_eval = ev
            # Print summary at default IoU threshold
            eval_index = ev.image_diagnostics(iou_thr=args.iou_thr)
            summary = {}
            for s in eval_index["img_summary"].values():
                for k in ("tp", "fp", "fn"):
                    summary[k] = summary.get(k, 0) + s[k]
            status(
                "Evaluated",
                f"{args.iou_type}  TP={summary.get('tp', 0):,}  "
                f"FP={summary.get('fp', 0):,}  FN={summary.get('fn', 0):,}",
                elapsed=t.elapsed,
            )
        except Exception as e:
            warning(f"eval failed ({e}), launching without eval coloring")

    # Load slices
    slices = None
    if args.slices:
        with open(args.slices) as f:
            slices = json_mod.load(f)

    from hotcoco.server import create_app, run_server

    app = create_app(
        coco, batch_size=args.batch_size, dt_coco=dt_coco, coco_eval=coco_eval, slices=slices, iou_thr=args.iou_thr
    )
    run_server(app, port=args.port, open_browser=True)


def cmd_sample(args):
    # Validate the flag combination before paying to load the dataset.
    n = args.n
    frac = args.frac
    if n is None and frac is None:
        error("provide --n or --frac")
        sys.exit(1)
    if n is not None and frac is not None:
        error("provide either --n or --frac, not both")
        sys.exit(1)

    coco = _load_coco(args.annotation_file, quiet=args.json, reraise=args.json)
    before = _counts(coco)

    result = coco.sample(n=n, frac=frac, seed=args.seed)
    result.save(args.output)

    return _report_before_after(args, verb="Sampled", before=before, after=_counts(result))


def cmd_compare(args):
    try:
        from hotcoco import COCOeval, compare
    except ImportError as e:
        if args.json:
            raise
        _extension_import_failed(e)

    gt = _load_coco(args.gt, quiet=args.json, reraise=args.json)
    dt_a = _load_res(gt, args.dt_a, quiet=args.json, reraise=args.json)
    dt_b = _load_res(gt, args.dt_b, quiet=args.json, reraise=args.json)

    with _maybe_spinner(f"Evaluating {args.iou_type}...", args.json), Timer() as t:
        ev_a = COCOeval(gt, dt_a, args.iou_type, lvis_style=args.lvis)
        ev_a.evaluate()
        ev_b = COCOeval(gt, dt_b, args.iou_type, lvis_style=args.lvis)
        ev_b.evaluate()
    if not args.json:
        status("Evaluated", f"both models ({args.iou_type})", elapsed=t.elapsed)

    with _maybe_spinner("Comparing models...", args.json), Timer() as t:
        result = compare(ev_a, ev_b, n_bootstrap=args.bootstrap, seed=args.seed, confidence=args.confidence)
    if not args.json:
        bootstrap_note = f", {args.bootstrap:,} bootstrap samples" if args.bootstrap else ""
        status("Compared", f"{args.name_a} vs {args.name_b}{bootstrap_note}", elapsed=t.elapsed)

    if args.json:
        result["name_a"] = args.name_a
        result["name_b"] = args.name_b
        return result

    name_a = args.name_a
    name_b = args.name_b
    n_images = result["num_images"]
    iou_type = args.iou_type

    section("Model Comparison", f"{n_images:,} images, {iou_type}")

    has_ci = result["ci"] is not None
    ci_pct = f"{int(args.confidence * 100)}% CI"
    cols = [("Metric", "<"), (name_a, ">"), (name_b, ">"), ("Delta", ">")]
    if has_ci:
        cols.append((ci_pct, ">"))

    ordered_keys = result["metric_keys"]
    rows = []
    for key in ordered_keys:
        val_a = result["metrics_a"].get(key, -1.0)
        val_b = result["metrics_b"].get(key, -1.0)
        delta = result["deltas"].get(key, 0.0)
        sign = "+" if delta >= 0 else ""
        cells = [key, fmt_metric(val_a), fmt_metric(val_b), f"{sign}{delta:.3f}"]
        if has_ci:
            ci = result["ci"].get(key)
            if ci:
                sig = "*" if ci["lower"] > 0 or ci["upper"] < 0 else " "
                cells.append(f"[{ci['lower']:+.3f}, {ci['upper']:+.3f}]{sig}")
            else:
                cells.append("")
        rows.append(cells)

    _table(cols, rows)

    if has_ci:
        print(f"\n  {dim('* = statistically significant (CI excludes zero)')}")

    # Per-category section
    cats = result["per_category"]
    if cats:
        n_show = min(5, len(cats))

        regressions = [c for c in cats if c["delta"] < 0][:n_show]
        improvements = [c for c in reversed(cats) if c["delta"] > 0][:n_show]

        if regressions or improvements:
            print(f"\n  Per-Category AP {dim('(top regressions and improvements)')}:")
            cat_cols = [("Category", "<"), (name_a, ">"), (name_b, ">"), ("Delta", ">")]
            cat_rows = []
            for c in regressions:
                ap_a, ap_b = fmt_metric(c["ap_a"]), fmt_metric(c["ap_b"])
                cat_rows.append([c["cat_name"], ap_a, ap_b, f"{c['delta']:+.3f}  {red('↓')}"])

            if regressions and improvements:
                cat_rows.append(["···", "", "", ""])

            for c in reversed(improvements):
                ap_a, ap_b = fmt_metric(c["ap_a"]), fmt_metric(c["ap_b"])
                cat_rows.append([c["cat_name"], ap_a, ap_b, f"{c['delta']:+.3f}  {green('↑')}"])

            _table(cat_cols, cat_rows)

    print()


def main():
    parser = argparse.ArgumentParser(
        prog="coco",
        description="hotcoco — fast COCO dataset tools",
        epilog=textwrap.dedent("""\
            examples:
              coco eval --gt ann.json --dt det.json              evaluate detections (bbox)
              coco eval --gt ann.json --dt det.json --tide       evaluation + error analysis
              coco eval --gt ann.json --dt det.json --json       JSON output for CI/CD
              coco stats ann.json                                dataset overview
              coco healthcheck ann.json                          validate annotations
              coco compare --gt ann.json --dt-a a.json --dt-b b.json  compare two models
              coco filter ann.json -o out.json --cat-ids 1,2,3     keep only specific categories
              coco convert --from coco --to yolo --input ann.json --output labels/
              coco convert --from oid --to coco --input boxes.csv --output ann.json  Open Images CSV
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # The installed distribution's version, not a constant to keep in sync.
    parser.add_argument("--version", action="version", version=f"%(prog)s {importlib.metadata.version('hotcoco')}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Shared parent parser that adds --json to every subcommand
    _json_parent = argparse.ArgumentParser(add_help=False)
    _json_parent.add_argument(
        "--json", action="store_true", help="output results as JSON to stdout (for CI/CD pipelines)"
    )

    eval_parser = subparsers.add_parser(
        "eval",
        parents=[_json_parent],
        help="evaluate detections against ground truth (bbox, segm, keypoints)",
        description=(
            "Run COCO evaluation and print AP/AR metrics. Supports bbox, segmentation, and "
            "keypoint evaluation with optional TIDE error analysis, sliced evaluation, and PDF reports."
        ),
        epilog=textwrap.dedent("""\
            examples:
              coco eval --gt ann.json --dt det.json
              coco eval --gt ann.json --dt det.json --iou-type segm
              coco eval --gt ann.json --dt det.json --tide --json
              coco eval --gt ann.json --dt det.json --report report.pdf
              coco eval --gt ann.json --dt det.json --lvis
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_parser.add_argument("--gt", required=True, help="ground truth annotations (COCO JSON)")
    eval_parser.add_argument("--dt", required=True, help="detection results (COCO JSON or list of dicts)")
    eval_parser.add_argument(
        "--iou-type",
        dest="iou_type",
        default="bbox",
        choices=["bbox", "segm", "keypoints"],
        help="evaluation type (default: bbox)",
    )
    eval_parser.add_argument("--img-ids", metavar="1,2,3", help="evaluate only these image IDs (comma-separated)")
    eval_parser.add_argument("--cat-ids", metavar="1,2,3", help="evaluate only these category IDs (comma-separated)")
    eval_parser.add_argument(
        "--no-cats", dest="no_cats", action="store_true", help="pool all categories (class-agnostic evaluation)"
    )
    eval_parser.add_argument(
        "--tide", action="store_true", help="print TIDE error decomposition after standard metrics"
    )
    eval_parser.add_argument(
        "--tide-pos-thr",
        dest="tide_pos_thr",
        type=float,
        default=0.5,
        metavar="THR",
        help="IoU threshold for TP/FP classification in TIDE (default: 0.5)",
    )
    eval_parser.add_argument(
        "--tide-bg-thr",
        dest="tide_bg_thr",
        type=float,
        default=0.1,
        metavar="THR",
        help="minimum IoU with any GT for Loc/Both/Bkg distinction in TIDE (default: 0.1)",
    )
    eval_parser.add_argument(
        "--calibration", action="store_true", help="compute confidence calibration (ECE/MCE) after standard metrics"
    )
    eval_parser.add_argument(
        "--diagnostics", action="store_true", help="per-image diagnostics: worst images by F1, label error candidates"
    )
    eval_parser.add_argument(
        "--diag-iou-thr",
        dest="diag_iou_thr",
        type=float,
        default=0.5,
        metavar="THR",
        help="IoU threshold for diagnostics TP/FP classification (default: 0.5)",
    )
    eval_parser.add_argument(
        "--diag-score-thr",
        dest="diag_score_thr",
        type=float,
        default=0.5,
        metavar="THR",
        help="min detection score for label error candidates (default: 0.5)",
    )
    eval_parser.add_argument(
        "--cal-bins",
        dest="cal_bins",
        type=int,
        default=10,
        metavar="N",
        help="number of calibration bins (default: 10)",
    )
    eval_parser.add_argument(
        "--cal-iou-thr",
        dest="cal_iou_thr",
        type=float,
        default=0.5,
        metavar="THR",
        help="IoU threshold for calibration TP/FP (default: 0.5)",
    )
    eval_parser.add_argument(
        "--lvis", action="store_true", help="use LVIS-style evaluation (max 300 dets, freq-group AP)"
    )
    eval_parser.add_argument(
        "--report",
        metavar="report.pdf",
        default=None,
        help="save a PDF evaluation report to this path (requires hotcoco[plot])",
    )
    eval_parser.add_argument(
        # None, not a literal: report() derives the title from the eval mode. A
        # default here would title every LVIS, keypoints and Open Images PDF
        # "COCO Evaluation Report".
        "--title",
        default=None,
        help="report title (default: derived from eval mode)",
    )
    eval_parser.add_argument(
        "--slices",
        metavar="slices.json",
        default=None,
        help='JSON file mapping slice names to image ID lists, for example {"daytime": [1,2,3]}',
    )
    eval_parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="run dataset healthcheck before evaluation (warnings printed to stderr)",
    )

    healthcheck_parser = subparsers.add_parser(
        "healthcheck",
        parents=[_json_parent],
        help="validate a COCO dataset for common errors",
        description=(
            "Check a COCO annotation file for common errors and warnings, including duplicate IDs, "
            "missing references, invalid bounding boxes, and annotation/image mismatches. "
            "Exits 1 when any ERROR-level finding is present (so it can gate CI); "
            "warnings alone exit 0."
        ),
        epilog=textwrap.dedent("""\
            examples:
              coco healthcheck ann.json
              coco healthcheck ann.json --dt det.json
              coco healthcheck ann.json --json

            exit status:
              0  no ERROR-level findings (warnings allowed)
              1  one or more ERROR-level findings
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    healthcheck_parser.add_argument("annotation_file", help="path to COCO annotation JSON")
    healthcheck_parser.add_argument("--dt", help="path to detection results JSON (enables GT/DT checks)")

    stats_parser = subparsers.add_parser(
        "stats", parents=[_json_parent], help="show dataset statistics (counts, dimensions, areas)"
    )
    stats_parser.add_argument("annotation_file", help="path to COCO annotation JSON file")
    stats_parser.add_argument("--all-cats", action="store_true", help="show all categories instead of top 20")

    filter_parser = subparsers.add_parser(
        "filter", parents=[_json_parent], help="filter a dataset by category, image, or area"
    )
    filter_parser.add_argument("annotation_file", help="input COCO JSON file")
    filter_parser.add_argument("-o", "--output", required=True, help="output JSON file")
    filter_parser.add_argument("--cat-ids", metavar="1,2,3", help="comma-separated category IDs to keep")
    filter_parser.add_argument("--img-ids", metavar="1,2,3", help="comma-separated image IDs to keep")
    filter_parser.add_argument("--area-rng", metavar="MIN,MAX", help="annotation area range (inclusive)")
    filter_parser.add_argument(
        "--keep-empty-images", action="store_true", help="keep images with no matching annotations"
    )

    merge_parser = subparsers.add_parser("merge", parents=[_json_parent], help="merge multiple datasets into one")
    merge_parser.add_argument("files", nargs="+", help="input COCO JSON files")
    merge_parser.add_argument("-o", "--output", required=True, help="output JSON file")

    split_parser = subparsers.add_parser(
        "split", parents=[_json_parent], help="split a dataset into train/val[/test] subsets"
    )
    split_parser.add_argument("annotation_file", help="input COCO JSON file")
    split_parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="PREFIX",
        help="output prefix; writes <prefix>_train.json, <prefix>_val.json, [<prefix>_test.json]",
    )
    split_parser.add_argument(
        "--val-frac", type=float, default=0.2, help="fraction of images for validation (default 0.2)"
    )
    split_parser.add_argument(
        "--test-frac", type=float, default=None, help="fraction of images for test set (optional)"
    )
    split_parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")

    sample_parser = subparsers.add_parser("sample", parents=[_json_parent], help="sample a random subset of images")
    sample_parser.add_argument("annotation_file", help="input COCO JSON file")
    sample_parser.add_argument("-o", "--output", required=True, help="output JSON file")
    sample_parser.add_argument("--n", type=int, default=None, help="number of images to sample")
    sample_parser.add_argument("--frac", type=float, default=None, help="fraction of images to sample")
    sample_parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")

    convert_parser = subparsers.add_parser(
        "convert",
        parents=[_json_parent],
        help="convert between annotation formats (COCO ↔ YOLO/VOC/CVAT/DOTA/Open Images)",
    )
    convert_parser.add_argument(
        "--from",
        dest="from_fmt",
        required=True,
        choices=["coco", "yolo", "voc", "cvat", "dota", "oid"],
        help="source format",
    )
    convert_parser.add_argument(
        "--to",
        dest="to_fmt",
        required=True,
        choices=["coco", "yolo", "voc", "cvat", "dota", "oid"],
        help="target format",
    )
    convert_parser.add_argument(
        "--input", required=True, help="input file (COCO JSON / Open Images CSV) or directory (YOLO, VOC, DOTA labels)"
    )
    convert_parser.add_argument(
        "--output",
        required=True,
        help="output file (COCO JSON / Open Images CSV) or directory (YOLO, VOC, DOTA labels)",
    )
    convert_parser.add_argument(
        "--images-dir",
        dest="images_dir",
        default=None,
        help="directory of images (YOLO/DOTA/Open Images → COCO; read image dimensions via Pillow)",
    )
    convert_parser.add_argument(
        "--class-descriptions",
        dest="class_descriptions",
        default=None,
        help="Open Images class-descriptions-boxable.csv (oid → COCO; resolves /m/ MIDs to names)",
    )

    explore_parser = subparsers.add_parser(
        "explore", help="browse a COCO dataset interactively (requires hotcoco[browse])"
    )
    explore_parser.add_argument("--gt", required=True, metavar="PATH", help="path to COCO annotation JSON")
    explore_parser.add_argument("--images", required=True, metavar="DIR", help="directory containing images")
    explore_parser.add_argument(
        "--dt", metavar="PATH", default=None, help="detection results JSON (enables detection overlay)"
    )
    explore_parser.add_argument(
        "--iou-type",
        dest="iou_type",
        default="bbox",
        choices=["bbox", "segm", "keypoints"],
        help="evaluation type for TP/FP/FN coloring (default: bbox)",
    )
    explore_parser.add_argument(
        "--iou-thr",
        dest="iou_thr",
        type=float,
        default=0.5,
        metavar="THR",
        help="initial IoU threshold for TP/FP classification; sets the UI slider's "
        "starting position, 0.50-0.95 in steps of 0.05 (default: 0.5)",
    )
    explore_parser.add_argument(
        "--no-eval",
        dest="no_eval",
        action="store_true",
        help="disable automatic evaluation (show detections without TP/FP/FN coloring)",
    )
    explore_parser.add_argument(
        "--slices",
        metavar="slices.json",
        default=None,
        help='JSON file mapping slice names to image ID lists, for example {"daytime": [1,2,3]}',
    )
    explore_parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=12, metavar="N", help="images per batch (default 12)"
    )
    explore_parser.add_argument("--port", type=int, default=7860, help="local server port (default 7860)")

    compare_parser = subparsers.add_parser(
        "compare",
        parents=[_json_parent],
        help="compare two model evaluations on the same dataset",
        description=(
            "Pairwise model comparison with metric deltas, per-category AP breakdown, "
            "and optional bootstrap confidence intervals."
        ),
        epilog=textwrap.dedent("""\
            examples:
              coco compare --gt ann.json --dt-a baseline.json --dt-b improved.json
              coco compare --gt ann.json --dt-a a.json --dt-b b.json --bootstrap 1000
              coco compare --gt ann.json --dt-a a.json --dt-b b.json --iou-type segm --json
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_parser.add_argument("--gt", required=True, help="ground truth annotations (COCO JSON)")
    compare_parser.add_argument("--dt-a", dest="dt_a", required=True, help="detections from model A (COCO JSON)")
    compare_parser.add_argument("--dt-b", dest="dt_b", required=True, help="detections from model B (COCO JSON)")
    compare_parser.add_argument(
        "--iou-type",
        dest="iou_type",
        default="bbox",
        choices=["bbox", "segm", "keypoints"],
        help="evaluation type (default: bbox)",
    )
    compare_parser.add_argument("--lvis", action="store_true", help="use LVIS-style federated evaluation")
    compare_parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="number of bootstrap samples for confidence intervals (0 = disabled)",
    )
    compare_parser.add_argument("--seed", type=int, default=42, help="random seed for bootstrap (default: 42)")
    compare_parser.add_argument(
        "--confidence", type=float, default=0.95, help="confidence level for bootstrap CIs (default: 0.95)"
    )
    compare_parser.add_argument(
        "--name-a",
        dest="name_a",
        default="Model A",
        metavar="NAME",
        help="display name for model A (default: 'Model A')",
    )
    compare_parser.add_argument(
        "--name-b",
        dest="name_b",
        default="Model B",
        metavar="NAME",
        help="display name for model B (default: 'Model B')",
    )
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "eval": cmd_eval,
        "healthcheck": cmd_healthcheck,
        "stats": cmd_stats,
        "filter": cmd_filter,
        "merge": cmd_merge,
        "split": cmd_split,
        "sample": cmd_sample,
        "convert": cmd_convert,
        "explore": cmd_explore,
        "compare": cmd_compare,
    }

    try:
        result = dispatch[args.command](args)
        if getattr(args, "json", False) and result is not None:
            print(json_mod.dumps(result, indent=2))
    except SystemExit:
        raise
    except Exception as e:
        if getattr(args, "json", False):
            print(json_mod.dumps({"error": str(e)}))
            sys.exit(1)
        else:
            error(str(e))
            sys.exit(1)


if __name__ == "__main__":
    main()
