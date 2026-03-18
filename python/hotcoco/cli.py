# PYTHON_ARGCOMPLETE_OK
"""
hotcoco command-line interface.

Usage:
    coco eval --gt <gt.json> --dt <dt.json> [--iou-type bbox|segm|keypoints] [--lvis] [--tide] [--report out.pdf] [--slices slices.json]
    coco healthcheck <annotation_file> [--dt <detections.json>]
    coco stats <annotation_file>
    coco filter <file> -o <output> [options]
    coco merge <file1> <file2> ... -o <output>
    coco split <file> -o <prefix> [options]
    coco sample <file> -o <output> [options]
    coco convert --from coco --to yolo --input <file> --output <dir>
    coco convert --from yolo --to coco --input <dir> --output <file> [--images-dir <dir>]
"""

import argparse
import json as json_mod
import os
import sys
import textwrap


def cmd_stats(args):
    try:
        from hotcoco import COCO
    except ImportError:
        print("error: hotcoco is not installed", file=sys.stderr)
        sys.exit(1)

    try:
        coco = COCO(args.annotation_file)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

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
        print("error: hotcoco is not installed", file=sys.stderr)
        sys.exit(1)
    try:
        return COCO(path)
    except Exception as e:
        print(f"error loading {path}: {e}", file=sys.stderr)
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
            print("error: --area-rng must be MIN,MAX", file=sys.stderr)
            sys.exit(1)
        area_rng = [float(parts[0]), float(parts[1])]

    drop_empty = not args.keep_empty_images
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

    print(f"filter: {os.path.basename(args.annotation_file)} → {os.path.basename(args.output)}")
    print(f"  before: {n_imgs_before:,} images, {n_anns_before:,} annotations")
    print(f"  after:  {n_imgs_after:,} images, {n_anns_after:,} annotations")


def cmd_merge(args):
    try:
        from hotcoco import COCO
    except ImportError:
        print("error: hotcoco is not installed", file=sys.stderr)
        sys.exit(1)

    cocos = [_load_coco(f) for f in args.files]
    n_imgs_total = sum(len(c.dataset["images"]) for c in cocos)
    n_anns_total = sum(len(c.dataset["annotations"]) for c in cocos)

    try:
        merged = COCO.merge(cocos)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    merged.save(args.output)
    n_imgs_out = len(merged.dataset["images"])
    n_anns_out = len(merged.dataset["annotations"])

    if args.json:
        return {
            "inputs": [{"file": f, "images": len(c.dataset["images"]), "annotations": len(c.dataset["annotations"])} for f, c in zip(args.files, cocos)],
            "output": {"file": args.output, "images": n_imgs_out, "annotations": n_anns_out},
        }

    print(f"merge: {len(args.files)} files → {os.path.basename(args.output)}")
    print(f"  input total: {n_imgs_total:,} images, {n_anns_total:,} annotations")
    print(f"  output:      {n_imgs_out:,} images, {n_anns_out:,} annotations")


def cmd_split(args):
    coco = _load_coco(args.annotation_file)
    n_imgs = len(coco.dataset["images"])

    test_frac = args.test_frac if args.test_frac else None
    result = coco.split(val_frac=args.val_frac, test_frac=test_frac, seed=args.seed)

    if test_frac is not None:
        train, val, test = result
        splits = [("train", train), ("val", val), ("test", test)]
    else:
        train, val = result
        splits = [("train", train), ("val", val)]

    split_results = {}
    print(f"split: {os.path.basename(args.annotation_file)} ({n_imgs:,} images)")
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
        print("error: hotcoco is not installed", file=sys.stderr)
        sys.exit(1)

    try:
        gt = COCO(args.gt)
    except Exception as e:
        if args.json:
            raise
        print(f"error loading ground truth: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        dt = gt.load_res(args.dt)
    except Exception as e:
        if args.json:
            raise
        print(f"error loading detections: {e}", file=sys.stderr)
        sys.exit(1)

    hc_result = None
    if args.healthcheck:
        hc_result = gt.healthcheck(dt)
        if not args.json:
            for f in hc_result["errors"]:
                print(f" \033[91mERROR [{f['code']}]\033[0m {f['message']}", file=sys.stderr)
            for f in hc_result["warnings"]:
                print(f" \033[93mWARN  [{f['code']}]\033[0m {f['message']}", file=sys.stderr)
            if hc_result["errors"] or hc_result["warnings"]:
                print(file=sys.stderr)

    ev = COCOeval(gt, dt, args.iou_type, lvis_style=args.lvis)

    if args.img_ids:
        ev.params.imgIds = [int(x) for x in args.img_ids.split(",")]
    if args.cat_ids:
        ev.params.catIds = [int(x) for x in args.cat_ids.split(",")]
    if args.no_cats:
        ev.params.useCats = False

    ev.evaluate()
    ev.accumulate()

    if args.json:
        # summarize() writes directly to the stdout fd from Rust; suppress at OS level
        stdout_fd = sys.stdout.fileno()
        saved_fd = os.dup(stdout_fd)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            try:
                ev.summarize()
            finally:
                os.dup2(saved_fd, stdout_fd)
                os.close(saved_fd)
    else:
        ev.summarize()

    slices_result = None
    if args.slices:
        with open(args.slices) as f:
            slices = json_mod.load(f)

        slices_result = ev.slice_by(slices)

        if not args.json:
            # Pick metric names based on eval mode
            if args.lvis:
                key_metrics = ["AP", "AP50", "AP75", "APr", "APc", "APf"]
            elif args.iou_type == "keypoints":
                key_metrics = ["AP", "AP50", "AP75", "APm", "APl"]
            else:
                key_metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

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

    tide_result = None
    if args.tide:
        tide_result = ev.tide_errors(pos_thr=args.tide_pos_thr, bg_thr=args.tide_bg_thr)
        if not args.json:
            _print_tide(tide_result)

    if args.report:
        try:
            from hotcoco.plot import report
        except ImportError as e:
            print(f"error: {e}", file=sys.stderr)
            print("hint: install plot dependencies with:  pip install hotcoco[plot]", file=sys.stderr)
            sys.exit(1)
        try:
            report(ev, save_path=args.report, gt_path=args.gt, dt_path=args.dt, title=args.title)
        except Exception as e:
            print(f"error generating report: {e}", file=sys.stderr)
            sys.exit(1)
        if not args.json:
            print(f"report saved to {args.report}")

    if args.json:
        result = ev.results(per_class=False)
        if tide_result is not None:
            result["tide"] = tide_result
        if slices_result is not None:
            result["slices"] = slices_result
        if hc_result is not None:
            result["healthcheck"] = {"errors": hc_result["errors"], "warnings": hc_result["warnings"]}
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


def cmd_convert(args):
    from_fmt = args.from_fmt
    to_fmt = args.to_fmt

    if from_fmt == "coco" and to_fmt == "yolo":
        coco = _load_coco(args.input)
        try:
            stats = coco.to_yolo(args.output)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            return {"direction": "coco_to_yolo", "input": args.input, "output": args.output, **stats}

        print("convert: COCO → YOLO")
        print(f"  input:       {os.path.basename(args.input)}")
        print(f"  output dir:  {args.output}")
        print(f"  images:      {stats['images']:,}")
        print(f"  annotations: {stats['annotations']:,}")
        if stats["skipped_crowd"] > 0:
            print(f"  skipped (crowd):   {stats['skipped_crowd']:,}")
        if stats["missing_bbox"] > 0:
            print(f"  skipped (no bbox): {stats['missing_bbox']:,}")

    elif from_fmt == "yolo" and to_fmt == "coco":
        try:
            from hotcoco import COCO
        except ImportError:
            print("error: hotcoco is not installed", file=sys.stderr)
            sys.exit(1)
        try:
            coco = COCO.from_yolo(args.input, images_dir=args.images_dir)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        try:
            coco.save(args.output)
        except Exception as e:
            print(f"error saving {args.output}: {e}", file=sys.stderr)
            sys.exit(1)
        n_imgs = len(coco.dataset["images"])
        n_anns = len(coco.dataset["annotations"])

        if args.json:
            return {"direction": "yolo_to_coco", "input": args.input, "output": args.output, "images": n_imgs, "annotations": n_anns}

        print("convert: YOLO → COCO")
        print(f"  input dir:   {args.input}")
        print(f"  output:      {os.path.basename(args.output)}")
        print(f"  images:      {n_imgs:,}")
        print(f"  annotations: {n_anns:,}")

    else:
        print(f"error: unsupported conversion: {from_fmt} → {to_fmt}", file=sys.stderr)
        sys.exit(1)


def cmd_healthcheck(args):
    coco = _load_coco(args.annotation_file)

    dt_coco = None
    if args.dt:
        try:
            dt_coco = coco.load_res(args.dt)
        except Exception as e:
            print(f"error loading detections: {e}", file=sys.stderr)
            sys.exit(1)

    report = coco.healthcheck(dt_coco)

    if args.json:
        return report

    for finding in report["errors"]:
        print(f"\033[91mERROR [{finding['code']}]\033[0m {finding['message']}")
        if finding["affected_ids"]:
            ids_str = ", ".join(str(i) for i in finding["affected_ids"][:10])
            suffix = f" ... ({len(finding['affected_ids'])} total)" if len(finding["affected_ids"]) > 10 else ""
            print(f"       IDs: {ids_str}{suffix}")

    for finding in report["warnings"]:
        print(f"\033[93mWARN  [{finding['code']}]\033[0m {finding['message']}")
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
        print("\n\033[92mAll checks passed.\033[0m")


def cmd_explore(args):
    try:
        import gradio  # noqa: F401
    except ImportError:
        print("error: gradio is required. Install with: pip install hotcoco[browse]", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.images):
        print(f"error: images directory not found: {args.images}", file=sys.stderr)
        sys.exit(1)

    coco = _load_coco(args.gt)
    coco.image_dir = args.images

    dt_coco = None
    if args.dt:
        try:
            dt_coco = coco.load_res(args.dt)
        except Exception as e:
            print(f"error loading detections: {e}", file=sys.stderr)
            sys.exit(1)

    from hotcoco import browse as _browse

    app = _browse.build_app(coco, batch_size=args.batch_size, dt_coco=dt_coco)
    app.launch(server_port=args.port, share=args.share)


def cmd_sample(args):
    coco = _load_coco(args.annotation_file)
    n_imgs_before = len(coco.dataset["images"])
    n_anns_before = len(coco.dataset["annotations"])

    n = args.n
    frac = args.frac
    if n is None and frac is None:
        print("error: provide --n or --frac", file=sys.stderr)
        sys.exit(1)
    if n is not None and frac is not None:
        print("error: provide either --n or --frac, not both", file=sys.stderr)
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

    print(f"sample: {os.path.basename(args.annotation_file)} → {os.path.basename(args.output)}")
    print(f"  before: {n_imgs_before:,} images, {n_anns_before:,} annotations")
    print(f"  after:  {n_imgs_after:,} images, {n_anns_after:,} annotations")


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
              coco filter ann.json -o out.json --cat-ids 1,2,3  keep only specific categories
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

    convert_parser = subparsers.add_parser("convert", parents=[_json_parent], help="convert between annotation formats (COCO ↔ YOLO)")
    convert_parser.add_argument(
        "--from", dest="from_fmt", required=True, choices=["coco", "yolo"], help="source format"
    )
    convert_parser.add_argument("--to", dest="to_fmt", required=True, choices=["coco", "yolo"], help="target format")
    convert_parser.add_argument("--input", required=True, help="input file (COCO JSON) or directory (YOLO labels)")
    convert_parser.add_argument("--output", required=True, help="output file (COCO JSON) or directory (YOLO labels)")
    convert_parser.add_argument(
        "--images-dir",
        dest="images_dir",
        default=None,
        help="directory of images (YOLO → COCO only; used to read image dimensions via Pillow)",
    )

    explore_parser = subparsers.add_parser("explore", help="browse a COCO dataset interactively (requires gradio)")
    explore_parser.add_argument("--gt", required=True, metavar="PATH", help="path to COCO annotation JSON")
    explore_parser.add_argument("--images", required=True, metavar="DIR", help="directory containing images")
    explore_parser.add_argument("--dt", metavar="PATH", default=None, help="detection results JSON (enables detection overlay)")
    explore_parser.add_argument("--batch-size", dest="batch_size", type=int, default=12, metavar="N", help="images per batch (default 12)")
    explore_parser.add_argument("--port", type=int, default=7860, help="local server port (default 7860)")
    explore_parser.add_argument("--share", action="store_true", help="create a public Gradio share link")

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
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
