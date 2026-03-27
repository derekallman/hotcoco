# PYTHON_ARGCOMPLETE_OK
"""
hotcoco command-line interface.

Usage:
    coco eval --gt <gt.json> --dt <dt.json> [--iou-type bbox|segm|keypoints] [--lvis] [--tide] [--calibration] [--report out.pdf] [--slices slices.json]
    coco healthcheck <annotation_file> [--dt <detections.json>]
    coco stats <annotation_file>
    coco filter <file> -o <output> [options]
    coco merge <file1> <file2> ... -o <output>
    coco split <file> -o <prefix> [options]
    coco sample <file> -o <output> [options]
    coco compare --gt <gt.json> --dt-a <a.json> --dt-b <b.json> [--bootstrap 1000]
    coco convert --from coco --to yolo --input <file> --output <dir>
    coco convert --from yolo --to coco --input <dir> --output <file> [--images-dir <dir>]
"""

import argparse
import json as json_mod
import os
import sys
import textwrap

from hotcoco._style import Timer, Spinner, dim, error, green, red, status, warning, yellow


def cmd_stats(args):
    coco = _load_coco(args.annotation_file)
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


def _load_coco(path):
    """Load a COCO annotation file, printing errors and exiting on failure."""
    try:
        from hotcoco import COCO
    except ImportError:
        error("hotcoco is not installed")
        sys.exit(1)
    try:
        with Spinner(f"Loading {dim(os.path.basename(path))}..."), Timer() as t:
            coco = COCO(path)
        n_imgs = len(coco.dataset.get("images", []))
        n_anns = len(coco.dataset.get("annotations", []))
        status("Loaded", f"{dim(os.path.basename(path))} ({n_imgs:,} images, {n_anns:,} annotations)", elapsed=t.elapsed)
        return coco
    except Exception as e:
        error(f"loading {path}: {e}")
        sys.exit(1)


def _load_res(coco, path):
    """Load detection results, printing errors and exiting on failure."""
    try:
        with Spinner(f"Loading {dim(os.path.basename(path))}..."), Timer() as t:
            dt = coco.load_res(path)
        n_dets = len(dt.dataset.get("annotations", []))
        status("Loaded", f"{dim(os.path.basename(path))} ({n_dets:,} detections)", elapsed=t.elapsed)
        return dt
    except Exception as e:
        error(f"loading detections: {e}")
        sys.exit(1)


def _summary(coco, label):
    """Print a one-line image/annotation count summary."""
    n_imgs = len(coco.dataset["images"])
    n_anns = len(coco.dataset["annotations"])
    print(f"  {label}: {n_imgs:,} images, {n_anns:,} annotations")


def cmd_filter(args):
    coco = _load_coco(args.annotation_file)
    n_imgs_before = len(coco.dataset["images"])
    n_anns_before = len(coco.dataset["annotations"])

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

    n_imgs_after = len(result.dataset["images"])
    n_anns_after = len(result.dataset["annotations"])

    if args.json:
        return {
            "before": {"images": n_imgs_before, "annotations": n_anns_before},
            "after": {"images": n_imgs_after, "annotations": n_anns_after},
            "output": args.output,
        }

    status("Filtered", f"{dim(os.path.basename(args.annotation_file))} → {dim(os.path.basename(args.output))}", elapsed=t.elapsed)
    print(f"  before: {n_imgs_before:,} images, {n_anns_before:,} annotations")
    print(f"  after:  {n_imgs_after:,} images, {n_anns_after:,} annotations")


def cmd_merge(args):
    cocos = [_load_coco(f) for f in args.files]
    n_imgs_total = sum(len(c.dataset["images"]) for c in cocos)
    n_anns_total = sum(len(c.dataset["annotations"]) for c in cocos)

    try:
        with Timer() as t:
            merged = COCO.merge(cocos)
    except Exception as e:
        error(str(e))
        sys.exit(1)

    merged.save(args.output)
    n_imgs_out = len(merged.dataset["images"])
    n_anns_out = len(merged.dataset["annotations"])

    if args.json:
        return {
            "inputs": [{"file": f, "images": len(c.dataset["images"]), "annotations": len(c.dataset["annotations"])} for f, c in zip(args.files, cocos)],
            "output": {"file": args.output, "images": n_imgs_out, "annotations": n_anns_out},
        }

    status("Merged", f"{len(args.files)} files → {dim(os.path.basename(args.output))}", elapsed=t.elapsed)
    print(f"  input total: {n_imgs_total:,} images, {n_anns_total:,} annotations")
    print(f"  output:      {n_imgs_out:,} images, {n_anns_out:,} annotations")


def cmd_split(args):
    coco = _load_coco(args.annotation_file)
    n_imgs = len(coco.dataset["images"])

    test_frac = args.test_frac if args.test_frac else None
    with Timer() as t:
        result = coco.split(val_frac=args.val_frac, test_frac=test_frac, seed=args.seed)

    if test_frac is not None:
        train, val, test = result
        splits = [("train", train), ("val", val), ("test", test)]
    else:
        train, val = result
        splits = [("train", train), ("val", val)]

    split_results = {}
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
        from hotcoco import COCO, COCOeval
    except ImportError:
        error("hotcoco is not installed")
        sys.exit(1)

    try:
        with Spinner(f"Loading ground truth {dim(os.path.basename(args.gt))}..."), Timer() as t:
            gt = COCO(args.gt)
        n_imgs = len(gt.dataset.get("images", []))
        n_anns = len(gt.dataset.get("annotations", []))
        if not args.json:
            status("Loaded", f"ground truth {dim(os.path.basename(args.gt))} ({n_imgs:,} images, {n_anns:,} annotations)", elapsed=t.elapsed)
    except Exception as e:
        if args.json:
            raise
        error(f"loading ground truth: {e}")
        sys.exit(1)

    try:
        with Spinner(f"Loading detections {dim(os.path.basename(args.dt))}..."), Timer() as t:
            dt = gt.load_res(args.dt)
        n_dets = len(dt.dataset.get("annotations", []))
        if not args.json:
            status("Loaded", f"detections {dim(os.path.basename(args.dt))} ({n_dets:,} results)", elapsed=t.elapsed)
    except Exception as e:
        if args.json:
            raise
        error(f"loading detections: {e}")
        sys.exit(1)

    hc_result = None
    if args.healthcheck:
        hc_result = gt.healthcheck(dt)
        if not args.json:
            for f in hc_result["errors"]:
                code = f["code"]
                print(f" {red('ERROR [' + code + ']')} {f['message']}", file=sys.stderr)
            for f in hc_result["warnings"]:
                code = f["code"]
                print(f" {yellow('WARN  [' + code + ']')} {f['message']}", file=sys.stderr)
            if hc_result["errors"] or hc_result["warnings"]:
                print(file=sys.stderr)

    ev = COCOeval(gt, dt, args.iou_type, lvis_style=args.lvis)

    if args.img_ids:
        ev.params.imgIds = [int(x) for x in args.img_ids.split(",")]
    if args.cat_ids:
        ev.params.catIds = [int(x) for x in args.cat_ids.split(",")]
    if args.no_cats:
        ev.params.useCats = False

    with Spinner(f"Evaluating {args.iou_type}..."), Timer() as t:
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

            # Column width matches "0.578 (+0.020)" = 14 chars
            col_w = 14
            header = f"  {'Slice':<20} {'N':>6}"
            for km in key_metrics:
                header += f"  {km:>{col_w}}"
            print()
            print(header)
            print("  " + "-" * (len(header) - 2))

            for name in sorted(slices_result.keys()):
                if name == "_overall":
                    continue
                sr = slices_result[name]
                row = f"  {name:<20} {sr['num_images']:>6}"
                for km in key_metrics:
                    val = sr.get(km, -1.0)
                    delta = sr.get("delta", {}).get(km, 0.0)
                    if val < 0:
                        row += f"  {'n/a':>{col_w}}"
                    else:
                        sign = "+" if delta >= 0 else ""
                        row += f"  {val:.3f} ({sign}{delta:.3f})"
                print(row)

            ov = slices_result["_overall"]
            row = f"  {'_overall':<20} {ov['num_images']:>6}"
            for km in key_metrics:
                val = ov.get(km, -1.0)
                if val < 0:
                    row += f"  {'n/a':>{col_w}}"
                else:
                    row += f"  {val:.3f}{'':>9}"
            print(row)

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
            error(str(e))
            print(f"  {dim('hint')}: install plot dependencies with:  pip install hotcoco[plot]", file=sys.stderr)
            sys.exit(1)
        try:
            report(ev, save_path=args.report, gt_path=args.gt, dt_path=args.dt, title=args.title)
        except Exception as e:
            error(f"generating report: {e}")
            sys.exit(1)
        if not args.json:
            status("Saved", f"report to {dim(args.report)}")

    if args.json:
        result = ev.results(per_class=False)
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
                    [{"image_id": k, **v} for k, v in diag_result["img_summary"].items()],
                    key=lambda x: x["f1"],
                )[:20],
                "iou_thr": diag_result["iou_thr"],
            }
        return result


def _print_tide(te):
    delta = te["delta_ap"]
    counts = te["counts"]
    print(
        f"\nTIDE Error Analysis"
        f"  (pos_thr={te['pos_thr']:.2f}, bg_thr={te['bg_thr']:.2f},"
        f" baseline_AP={te['ap_base']:.4f})\n"
    )
    print(f"  {'Type':<6}  {'ΔAP':>7}  {'Count':>7}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 7}")
    for error_type in ("Loc", "Cls", "Both", "Dupe", "Bkg", "Miss"):
        dap = delta.get(error_type, 0.0)
        cnt = counts.get(error_type, 0)
        print(f"  {error_type:<6}  {dap:>7.4f}  {cnt:>7,}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 7}")
    print(f"  {'FP':<6}  {delta.get('FP', 0.0):>7.4f}")
    print(f"  {'FN':<6}  {delta.get('FN', 0.0):>7.4f}")


def _print_calibration(cal):
    print(
        f"\nCalibration Analysis"
        f"  (iou_thr={cal['iou_threshold']:.2f},"
        f" bins={cal['n_bins']}, detections={cal['num_detections']:,})\n"
    )
    print(f"  ECE: {cal['ece']:.4f}")
    print(f"  MCE: {cal['mce']:.4f}")
    print()
    print(f"  {'Bin':>11}  {'Conf':>6}  {'Acc':>6}  {'Count':>7}")
    print(f"  {'─' * 11}  {'─' * 6}  {'─' * 6}  {'─' * 7}")
    for b in cal["bins"]:
        label = f"[{b['bin_lower']:.1f}, {b['bin_upper']:.1f})"
        if b["count"] > 0:
            print(f"  {label:>11}  {b['avg_confidence']:>6.3f}  {b['avg_accuracy']:>6.3f}  {b['count']:>7,}")
        else:
            print(f"  {label:>11}  {'─':>6}  {'─':>6}  {0:>7}")

    per_cat = cal.get("per_category", {})
    if per_cat:
        sorted_cats = sorted(per_cat.items(), key=lambda x: x[1], reverse=True)
        top = sorted_cats[:10]
        print()
        print("  Per-category ECE (top 10 worst-calibrated):")
        for name, ece in top:
            print(f"    {name:<20} {ece:.4f}")


def _print_diagnostics(diag):
    summaries = diag["img_summary"]
    n_images = len(summaries)
    label_errors = diag["label_errors"]
    wrong = [le for le in label_errors if le["type"] == "wrong_label"]
    missing = [le for le in label_errors if le["type"] == "missing_annotation"]

    print(
        f"\nPer-Image Diagnostics"
        f"  (iou_thr={diag['iou_thr']:.2f}, images={n_images:,})\n"
    )

    # F1 distribution buckets
    f1s = [s["f1"] for s in summaries.values()]
    poor = sum(1 for f in f1s if f < 0.5)
    moderate = sum(1 for f in f1s if 0.5 <= f <= 0.8)
    good = sum(1 for f in f1s if f > 0.8)
    print(f"  F1 distribution:  {poor:,} poor (<0.5)  {moderate:,} moderate (0.5–0.8)  {good:,} good (>0.8)")

    # Label error summary
    score_thr = diag.get("score_thr", 0.5)
    print(f"\n  Label errors:     {len(label_errors):,} candidates (score ≥ {score_thr:.2f})")
    if wrong:
        top_wrong = ", ".join(f"{le['dt_category']}→{le['gt_category']}" for le in wrong[:3])
        print(f"    wrong_label:        {len(wrong)}  (top: {top_wrong})")
    if missing:
        # Aggregate by category
        from collections import Counter
        cat_counts = Counter(le["dt_category"] for le in missing)
        top_cats = ", ".join(f"{cat} {n}" for cat, n in cat_counts.most_common(5))
        print(f"    missing_annotation: {len(missing):,}  (top categories: {top_cats})")
    if not wrong and not missing:
        print("    (none found)")

    print(f"\n  Tip: use ev.image_diagnostics() or coco explore --dt for interactive analysis.")


def cmd_convert(args):
    from_fmt = args.from_fmt
    to_fmt = args.to_fmt

    if from_fmt == "coco" and to_fmt == "yolo":
        coco = _load_coco(args.input)
        try:
            with Timer() as t:
                stats = coco.to_yolo(args.output)
        except Exception as e:
            error(str(e))
            sys.exit(1)

        if args.json:
            return {"direction": "coco_to_yolo", "input": args.input, "output": args.output, **stats}

        status("Converted", f"COCO → YOLO ({stats['annotations']:,} annotations)", elapsed=t.elapsed)
        print(f"  input:       {os.path.basename(args.input)}")
        print(f"  output dir:  {args.output}")
        if stats["skipped_crowd"] > 0:
            print(f"  skipped (crowd):   {stats['skipped_crowd']:,}")
        if stats["missing_bbox"] > 0:
            print(f"  skipped (no bbox): {stats['missing_bbox']:,}")

    elif from_fmt == "yolo" and to_fmt == "coco":
        try:
            from hotcoco import COCO
        except ImportError:
            error("hotcoco is not installed")
            sys.exit(1)
        try:
            with Spinner("Converting YOLO → COCO..."), Timer() as t:
                coco = COCO.from_yolo(args.input, images_dir=args.images_dir)
        except Exception as e:
            error(str(e))
            sys.exit(1)
        try:
            coco.save(args.output)
        except Exception as e:
            error(f"saving {args.output}: {e}")
            sys.exit(1)
        n_imgs = len(coco.dataset["images"])
        n_anns = len(coco.dataset["annotations"])

        if args.json:
            return {"direction": "yolo_to_coco", "input": args.input, "output": args.output, "images": n_imgs, "annotations": n_anns}

        status("Converted", f"YOLO → COCO ({n_imgs:,} images, {n_anns:,} annotations)", elapsed=t.elapsed)

    elif from_fmt == "coco" and to_fmt == "voc":
        coco = _load_coco(args.input)
        try:
            with Timer() as t:
                stats = coco.to_voc(args.output)
        except Exception as e:
            error(str(e))
            sys.exit(1)

        if args.json:
            return {"direction": "coco_to_voc", "input": args.input, "output": args.output, **stats}

        status("Converted", f"COCO → VOC ({stats['annotations']:,} annotations)", elapsed=t.elapsed)
        print(f"  input:       {os.path.basename(args.input)}")
        print(f"  output dir:  {args.output}")
        if stats["crowd_as_difficult"] > 0:
            print(f"  crowd → difficult: {stats['crowd_as_difficult']:,}")
        if stats["missing_bbox"] > 0:
            print(f"  skipped (no bbox): {stats['missing_bbox']:,}")

    elif from_fmt == "voc" and to_fmt == "coco":
        try:
            from hotcoco import COCO
        except ImportError:
            error("hotcoco is not installed")
            sys.exit(1)
        try:
            with Spinner("Converting VOC → COCO..."), Timer() as t:
                coco = COCO.from_voc(args.input)
        except Exception as e:
            error(str(e))
            sys.exit(1)
        try:
            coco.save(args.output)
        except Exception as e:
            error(f"saving {args.output}: {e}")
            sys.exit(1)
        n_imgs = len(coco.dataset["images"])
        n_anns = len(coco.dataset["annotations"])

        if args.json:
            return {"direction": "voc_to_coco", "input": args.input, "output": args.output, "images": n_imgs, "annotations": n_anns}

        status("Converted", f"VOC → COCO ({n_imgs:,} images, {n_anns:,} annotations)", elapsed=t.elapsed)

    elif from_fmt == "coco" and to_fmt == "cvat":
        coco = _load_coco(args.input)
        try:
            with Timer() as t:
                stats = coco.to_cvat(args.output)
        except Exception as e:
            error(str(e))
            sys.exit(1)

        if args.json:
            return {"direction": "coco_to_cvat", "input": args.input, "output": args.output, **stats}

        status("Converted", f"COCO → CVAT ({stats['boxes']:,} boxes, {stats['polygons']:,} polygons)", elapsed=t.elapsed)
        if stats["skipped_no_geometry"] > 0:
            print(f"  skipped (no geometry): {stats['skipped_no_geometry']:,}")

    elif from_fmt == "cvat" and to_fmt == "coco":
        try:
            from hotcoco import COCO
        except ImportError:
            error("hotcoco is not installed")
            sys.exit(1)
        try:
            with Spinner("Converting CVAT → COCO..."), Timer() as t:
                coco = COCO.from_cvat(args.input)
        except Exception as e:
            error(str(e))
            sys.exit(1)
        try:
            coco.save(args.output)
        except Exception as e:
            error(f"saving {args.output}: {e}")
            sys.exit(1)
        n_imgs = len(coco.dataset["images"])
        n_anns = len(coco.dataset["annotations"])

        if args.json:
            return {"direction": "cvat_to_coco", "input": args.input, "output": args.output, "images": n_imgs, "annotations": n_anns}

        status("Converted", f"CVAT → COCO ({n_imgs:,} images, {n_anns:,} annotations)", elapsed=t.elapsed)

    else:
        error(f"unsupported conversion: {from_fmt} → {to_fmt}")
        sys.exit(1)


def cmd_healthcheck(args):
    coco = _load_coco(args.annotation_file)

    dt_coco = _load_res(coco, args.dt) if args.dt else None

    report = coco.healthcheck(dt_coco)

    if args.json:
        return report

    for finding in report["errors"]:
        code = finding["code"]
        print(f"{red('ERROR [' + code + ']')} {finding['message']}")
        if finding["affected_ids"]:
            ids_str = ", ".join(str(i) for i in finding["affected_ids"][:10])
            suffix = f" ... ({len(finding['affected_ids'])} total)" if len(finding["affected_ids"]) > 10 else ""
            print(f"       IDs: {ids_str}{suffix}")

    for finding in report["warnings"]:
        code = finding["code"]
        print(f"{yellow('WARN  [' + code + ']')} {finding['message']}")
        if finding["affected_ids"]:
            ids_str = ", ".join(str(i) for i in finding["affected_ids"][:10])
            suffix = f" ... ({len(finding['affected_ids'])} total)" if len(finding["affected_ids"]) > 10 else ""
            print(f"       IDs: {ids_str}{suffix}")

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
        print(f"  Cat imbalance: {s['imbalance_ratio']:>8.1f}x  ({top_name}: {top_count:,} / {bot_name}: {bot_count:,})")
    else:
        print(f"  Cat imbalance: {s['imbalance_ratio']:>8.1f}x")

    if not report["errors"] and not report["warnings"]:
        print(f"\n{green('All checks passed.')}")


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
            status("Evaluated", f"{args.iou_type}  TP={summary.get('tp', 0):,}  FP={summary.get('fp', 0):,}  FN={summary.get('fn', 0):,}", elapsed=t.elapsed)
        except Exception as e:
            warning(f"eval failed ({e}), launching without eval coloring")

    # Load slices
    slices = None
    if args.slices:
        with open(args.slices) as f:
            slices = json_mod.load(f)

    from hotcoco.server import create_app, run_server

    app = create_app(coco, batch_size=args.batch_size, dt_coco=dt_coco, coco_eval=coco_eval, slices=slices)
    run_server(app, port=args.port, open_browser=True)


def cmd_sample(args):
    coco = _load_coco(args.annotation_file)
    n_imgs_before = len(coco.dataset["images"])
    n_anns_before = len(coco.dataset["annotations"])

    n = args.n
    frac = args.frac
    if n is None and frac is None:
        error("provide --n or --frac")
        sys.exit(1)
    if n is not None and frac is not None:
        error("provide either --n or --frac, not both")
        sys.exit(1)

    result = coco.sample(n=n, frac=frac, seed=args.seed)
    result.save(args.output)

    n_imgs_after = len(result.dataset["images"])
    n_anns_after = len(result.dataset["annotations"])

    if args.json:
        return {
            "before": {"images": n_imgs_before, "annotations": n_anns_before},
            "after": {"images": n_imgs_after, "annotations": n_anns_after},
            "output": args.output,
        }

    status("Sampled", f"{dim(os.path.basename(args.annotation_file))} → {dim(os.path.basename(args.output))}")
    print(f"  before: {n_imgs_before:,} images, {n_anns_before:,} annotations")
    print(f"  after:  {n_imgs_after:,} images, {n_anns_after:,} annotations")


def cmd_compare(args):
    try:
        from hotcoco import COCO, COCOeval, compare
    except ImportError:
        error("hotcoco is not installed")
        sys.exit(1)

    gt = _load_coco(args.gt)
    dt_a = _load_res(gt, args.dt_a)
    dt_b = _load_res(gt, args.dt_b)

    with Spinner(f"Evaluating {args.iou_type}..."), Timer() as t:
        ev_a = COCOeval(gt, dt_a, args.iou_type, lvis_style=args.lvis)
        ev_a.evaluate()
        ev_b = COCOeval(gt, dt_b, args.iou_type, lvis_style=args.lvis)
        ev_b.evaluate()
    status("Evaluated", f"both models ({args.iou_type})", elapsed=t.elapsed)

    with Spinner("Comparing models..."), Timer() as t:
        result = compare(
            ev_a,
            ev_b,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            confidence=args.confidence,
        )
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

    print(f"\nModel Comparison ({n_images:,} images, {iou_type})\n")

    # Header
    has_ci = result["ci"] is not None
    ci_pct = f"{int(args.confidence * 100)}% CI"
    if has_ci:
        print(f"  {'Metric':<10}  {name_a:>10}  {name_b:>10}  {'Delta':>10}  {ci_pct:>20}")
        print(f"  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 20}")
    else:
        print(f"  {'Metric':<10}  {name_a:>10}  {name_b:>10}  {'Delta':>10}")
        print(f"  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

    ordered_keys = result["metric_keys"]

    for key in ordered_keys:
        val_a = result["metrics_a"].get(key, -1.0)
        val_b = result["metrics_b"].get(key, -1.0)
        delta = result["deltas"].get(key, 0.0)

        sign = "+" if delta >= 0 else ""
        line = f"  {key:<10}  {val_a:>10.3f}  {val_b:>10.3f}  {sign}{delta:>9.3f}"

        if has_ci:
            ci = result["ci"].get(key)
            if ci:
                sig = "*" if ci["lower"] > 0 or ci["upper"] < 0 else " "
                line += f"  [{ci['lower']:+.3f}, {ci['upper']:+.3f}]{sig}"

        print(line)

    if has_ci:
        print(f"\n  * = statistically significant (CI excludes zero)")

    # Per-category section
    cats = result["per_category"]
    if cats:
        n_show = min(5, len(cats))

        # Regressions (most negative deltas)
        regressions = [c for c in cats if c["delta"] < 0][:n_show]
        # Improvements (most positive deltas, reversed from end)
        improvements = [c for c in reversed(cats) if c["delta"] > 0][:n_show]

        if regressions or improvements:
            print(f"\n  Per-Category AP (top regressions and improvements):\n")
            print(f"  {'Category':<20}  {name_a:>10}  {name_b:>10}  {'Delta':>10}")
            print(f"  {'─' * 20}  {'─' * 10}  {'─' * 10}  {'─' * 10}")

            def _cat_row(c, arrow, color_fn):
                ap_a = f"{c['ap_a']:.3f}" if c["ap_a"] >= 0 else "   n/a"
                ap_b = f"{c['ap_b']:.3f}" if c["ap_b"] >= 0 else "   n/a"
                print(f"  {c['cat_name']:<20}  {ap_a:>10}  {ap_b:>10}  {c['delta']:>+10.3f}  {color_fn(arrow)}")

            for c in regressions:
                _cat_row(c, "↓", red)

            if regressions and improvements:
                print(f"  {'···':^54}")

            for c in reversed(improvements):
                _cat_row(c, "↑", green)

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
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # Shared parent parser that adds --json to every subcommand
    _json_parent = argparse.ArgumentParser(add_help=False)
    _json_parent.add_argument(
        "--json",
        action="store_true",
        help="output results as JSON to stdout (for CI/CD pipelines)",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        parents=[_json_parent],
        help="evaluate detections against ground truth (bbox, segm, keypoints)",
        description="Run COCO evaluation and print AP/AR metrics. Supports bbox, segmentation, and keypoint evaluation with optional TIDE error analysis, sliced evaluation, and PDF reports.",
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
        "--diagnostics", action="store_true", help="per-image diagnostics: worst images by F1/AP, label error candidates"
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
        "--title", default="COCO Evaluation Report", help="report title (default: 'COCO Evaluation Report')"
    )
    eval_parser.add_argument(
        "--slices",
        metavar="slices.json",
        default=None,
        help='JSON file mapping slice names to image ID lists, e.g. {"daytime": [1,2,3]}',
    )
    eval_parser.add_argument(
        "--healthcheck", action="store_true",
        help="run dataset healthcheck before evaluation (warnings printed to stderr)",
    )

    healthcheck_parser = subparsers.add_parser(
        "healthcheck",
        parents=[_json_parent],
        help="validate a COCO dataset for common errors",
        description="Check a COCO annotation file for common errors and warnings, including duplicate IDs, missing references, invalid bounding boxes, and annotation/image mismatches.",
        epilog=textwrap.dedent("""\
            examples:
              coco healthcheck ann.json
              coco healthcheck ann.json --dt det.json
              coco healthcheck ann.json --json
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    healthcheck_parser.add_argument("annotation_file", help="path to COCO annotation JSON")
    healthcheck_parser.add_argument("--dt", help="path to detection results JSON (enables GT/DT checks)")

    stats_parser = subparsers.add_parser("stats", parents=[_json_parent], help="show dataset statistics (counts, dimensions, areas)")
    stats_parser.add_argument("annotation_file", help="path to COCO annotation JSON file")
    stats_parser.add_argument("--all-cats", action="store_true", help="show all categories instead of top 20")

    filter_parser = subparsers.add_parser("filter", parents=[_json_parent], help="filter a dataset by category, image, or area")
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

    split_parser = subparsers.add_parser("split", parents=[_json_parent], help="split a dataset into train/val[/test] subsets")
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

    convert_parser = subparsers.add_parser("convert", parents=[_json_parent], help="convert between annotation formats (COCO ↔ YOLO/VOC/CVAT)")
    convert_parser.add_argument(
        "--from", dest="from_fmt", required=True, choices=["coco", "yolo", "voc", "cvat"], help="source format"
    )
    convert_parser.add_argument("--to", dest="to_fmt", required=True, choices=["coco", "yolo", "voc", "cvat"], help="target format")
    convert_parser.add_argument("--input", required=True, help="input file (COCO JSON) or directory (YOLO labels / VOC Annotations)")
    convert_parser.add_argument("--output", required=True, help="output file (COCO JSON) or directory (YOLO labels / VOC Annotations)")
    convert_parser.add_argument(
        "--images-dir",
        dest="images_dir",
        default=None,
        help="directory of images (YOLO → COCO only; used to read image dimensions via Pillow)",
    )

    explore_parser = subparsers.add_parser("explore", help="browse a COCO dataset interactively (requires hotcoco[browse])")
    explore_parser.add_argument("--gt", required=True, metavar="PATH", help="path to COCO annotation JSON")
    explore_parser.add_argument("--images", required=True, metavar="DIR", help="directory containing images")
    explore_parser.add_argument("--dt", metavar="PATH", default=None, help="detection results JSON (enables detection overlay)")
    explore_parser.add_argument(
        "--iou-type", dest="iou_type", default="bbox", choices=["bbox", "segm", "keypoints"],
        help="evaluation type for TP/FP/FN coloring (default: bbox)",
    )
    explore_parser.add_argument(
        "--iou-thr", dest="iou_thr", type=float, default=0.5, metavar="THR",
        help="IoU threshold for TP/FP classification (default: 0.5)",
    )
    explore_parser.add_argument(
        "--no-eval", dest="no_eval", action="store_true",
        help="disable automatic evaluation (show detections without TP/FP/FN coloring)",
    )
    explore_parser.add_argument(
        "--slices", metavar="slices.json", default=None,
        help='JSON file mapping slice names to image ID lists, e.g. {"daytime": [1,2,3]}',
    )
    explore_parser.add_argument("--batch-size", dest="batch_size", type=int, default=12, metavar="N", help="images per batch (default 12)")
    explore_parser.add_argument("--port", type=int, default=7860, help="local server port (default 7860)")

    compare_parser = subparsers.add_parser(
        "compare",
        parents=[_json_parent],
        help="compare two model evaluations on the same dataset",
        description="Pairwise model comparison with metric deltas, per-category AP breakdown, and optional bootstrap confidence intervals.",
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
        "--iou-type", dest="iou_type", default="bbox", choices=["bbox", "segm", "keypoints"],
        help="evaluation type (default: bbox)",
    )
    compare_parser.add_argument("--lvis", action="store_true", help="use LVIS-style federated evaluation")
    compare_parser.add_argument("--bootstrap", type=int, default=0, metavar="N", help="number of bootstrap samples for confidence intervals (0 = disabled)")
    compare_parser.add_argument("--seed", type=int, default=42, help="random seed for bootstrap (default: 42)")
    compare_parser.add_argument("--confidence", type=float, default=0.95, help="confidence level for bootstrap CIs (default: 0.95)")
    compare_parser.add_argument("--name-a", dest="name_a", default="Model A", metavar="NAME", help="display name for model A (default: 'Model A')")
    compare_parser.add_argument("--name-b", dest="name_b", default="Model B", metavar="NAME", help="display name for model B (default: 'Model B')")
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
