# PYTHON_ARGCOMPLETE_OK
"""
hotcoco command-line interface.

Usage:
    coco eval --gt <gt.json> --dt <dt.json> [--iou-type bbox|segm|keypoints] [--tide]
    coco stats <annotation_file>
    coco filter <file> -o <output> [options]
    coco merge <file1> <file2> ... -o <output>
    coco split <file> -o <prefix> [options]
    coco sample <file> -o <output> [options]
    coco convert --from coco --to yolo --input <file> --output <dir>
    coco convert --from yolo --to coco --input <dir> --output <file> [--images-dir <dir>]
"""

import argparse
import os
import sys


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
            print(
                f"  {name}  {c['ann_count']:>6,} anns   {c['img_count']:>5,} imgs"
            )

    w = s["image_width"]
    h = s["image_height"]
    print()
    print("Image dimensions:")
    print(
        f"  width   min={w['min']:.0f}    max={w['max']:.0f}"
        f"   mean={w['mean']:.1f}  median={w['median']:.1f}"
    )
    print(
        f"  height  min={h['min']:.0f}    max={h['max']:.0f}"
        f"   mean={h['mean']:.1f}  median={h['median']:.1f}"
    )

    a = s["annotation_area"]
    print()
    print("Annotation areas:")
    print(
        f"  min={a['min']:.1f}   max={a['max']:.1f}"
        f"   mean={a['mean']:.1f}   median={a['median']:.1f}"
    )


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

    print(f"split: {os.path.basename(args.annotation_file)} ({n_imgs:,} images)")
    for name, split in splits:
        out_path = f"{args.output}_{name}.json"
        split.save(out_path)
        n = len(split.dataset["images"])
        n_anns = len(split.dataset["annotations"])
        print(f"  {name}: {n:,} images, {n_anns:,} annotations → {os.path.basename(out_path)}")


def cmd_eval(args):
    try:
        from hotcoco import COCO, COCOeval
    except ImportError:
        print("error: hotcoco is not installed", file=sys.stderr)
        sys.exit(1)

    try:
        gt = COCO(args.gt)
    except Exception as e:
        print(f"error loading ground truth: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        dt = gt.load_res(args.dt)
    except Exception as e:
        print(f"error loading detections: {e}", file=sys.stderr)
        sys.exit(1)

    ev = COCOeval(gt, dt, args.iou_type)

    if args.img_ids:
        ev.params.imgIds = [int(x) for x in args.img_ids.split(",")]
    if args.cat_ids:
        ev.params.catIds = [int(x) for x in args.cat_ids.split(",")]
    if args.no_cats:
        ev.params.useCats = False

    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    if args.tide:
        te = ev.tide_errors(pos_thr=args.tide_pos_thr, bg_thr=args.tide_bg_thr)
        _print_tide(te)


def _print_tide(te):
    delta = te["delta_ap"]
    counts = te["counts"]
    print(
        f"\nTIDE Error Analysis"
        f"  (pos_thr={te['pos_thr']:.2f}, bg_thr={te['bg_thr']:.2f},"
        f" baseline_AP={te['ap_base']:.4f})\n"
    )
    print(f"  {'Type':<6}  {'ΔAP':>7}  {'Count':>7}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}")
    for error_type in ("Loc", "Cls", "Both", "Dupe", "Bkg", "Miss"):
        dap = delta.get(error_type, 0.0)
        cnt = counts.get(error_type, 0)
        print(f"  {error_type:<6}  {dap:>7.4f}  {cnt:>7,}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}")
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
        print(f"convert: COCO → YOLO")
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
        print(f"convert: YOLO → COCO")
        print(f"  input dir:   {args.input}")
        print(f"  output:      {os.path.basename(args.output)}")
        print(f"  images:      {n_imgs:,}")
        print(f"  annotations: {n_anns:,}")

    else:
        print(f"error: unsupported conversion: {from_fmt} → {to_fmt}", file=sys.stderr)
        sys.exit(1)


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
    print(f"sample: {os.path.basename(args.annotation_file)} → {os.path.basename(args.output)}")
    print(f"  before: {n_imgs_before:,} images, {n_anns_before:,} annotations")
    print(f"  after:  {n_imgs_after:,} images, {n_anns_after:,} annotations")


def main():
    parser = argparse.ArgumentParser(
        prog="coco",
        description="hotcoco command-line tools for COCO datasets",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    eval_parser = subparsers.add_parser(
        "eval",
        help="evaluate detections against ground truth",
    )
    eval_parser.add_argument("--gt", required=True, help="path to ground truth annotations JSON")
    eval_parser.add_argument("--dt", required=True, help="path to detection results JSON")
    eval_parser.add_argument(
        "--iou-type",
        dest="iou_type",
        default="bbox",
        choices=["bbox", "segm", "keypoints"],
        help="evaluation type (default: bbox)",
    )
    eval_parser.add_argument(
        "--img-ids", metavar="1,2,3", help="evaluate only these image IDs (comma-separated)"
    )
    eval_parser.add_argument(
        "--cat-ids", metavar="1,2,3", help="evaluate only these category IDs (comma-separated)"
    )
    eval_parser.add_argument(
        "--no-cats",
        dest="no_cats",
        action="store_true",
        help="pool all categories (class-agnostic evaluation)",
    )
    eval_parser.add_argument(
        "--tide",
        action="store_true",
        help="print TIDE error decomposition after standard metrics",
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

    stats_parser = subparsers.add_parser(
        "stats",
        help="print dataset health-check statistics",
    )
    stats_parser.add_argument(
        "annotation_file",
        help="path to COCO annotation JSON file",
    )
    stats_parser.add_argument(
        "--all-cats",
        action="store_true",
        help="show all categories instead of top 20",
    )

    filter_parser = subparsers.add_parser(
        "filter",
        help="filter a dataset by category, image, or area",
    )
    filter_parser.add_argument("annotation_file", help="input COCO JSON file")
    filter_parser.add_argument("-o", "--output", required=True, help="output JSON file")
    filter_parser.add_argument(
        "--cat-ids", metavar="1,2,3", help="comma-separated category IDs to keep"
    )
    filter_parser.add_argument(
        "--img-ids", metavar="1,2,3", help="comma-separated image IDs to keep"
    )
    filter_parser.add_argument(
        "--area-rng", metavar="MIN,MAX", help="annotation area range (inclusive)"
    )
    filter_parser.add_argument(
        "--keep-empty-images",
        action="store_true",
        help="keep images with no matching annotations",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="merge multiple datasets into one",
    )
    merge_parser.add_argument("files", nargs="+", help="input COCO JSON files")
    merge_parser.add_argument("-o", "--output", required=True, help="output JSON file")

    split_parser = subparsers.add_parser(
        "split",
        help="split a dataset into train/val[/test] subsets",
    )
    split_parser.add_argument("annotation_file", help="input COCO JSON file")
    split_parser.add_argument(
        "-o", "--output", required=True,
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

    sample_parser = subparsers.add_parser(
        "sample",
        help="sample a random subset of images",
    )
    sample_parser.add_argument("annotation_file", help="input COCO JSON file")
    sample_parser.add_argument("-o", "--output", required=True, help="output JSON file")
    sample_parser.add_argument("--n", type=int, default=None, help="number of images to sample")
    sample_parser.add_argument("--frac", type=float, default=None, help="fraction of images to sample")
    sample_parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")

    convert_parser = subparsers.add_parser(
        "convert",
        help="convert between annotation formats (COCO ↔ YOLO)",
    )
    convert_parser.add_argument(
        "--from",
        dest="from_fmt",
        required=True,
        choices=["coco", "yolo"],
        help="source format",
    )
    convert_parser.add_argument(
        "--to",
        dest="to_fmt",
        required=True,
        choices=["coco", "yolo"],
        help="target format",
    )
    convert_parser.add_argument(
        "--input",
        required=True,
        help="input file (COCO JSON) or directory (YOLO labels)",
    )
    convert_parser.add_argument(
        "--output",
        required=True,
        help="output file (COCO JSON) or directory (YOLO labels)",
    )
    convert_parser.add_argument(
        "--images-dir",
        dest="images_dir",
        default=None,
        help="directory of images (YOLO → COCO only; used to read image dimensions via Pillow)",
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

    if args.command == "eval":
        cmd_eval(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "filter":
        cmd_filter(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "sample":
        cmd_sample(args)
    elif args.command == "convert":
        cmd_convert(args)


if __name__ == "__main__":
    main()
