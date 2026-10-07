"""Download COCO panoptic val2017 and generate a perturbed prediction set.

The official archive is 860 MB, almost all of it train2017 PNG files. The val2017
members — the JSON and the 5,000 PNG masks — are 35 MB together, and the server
honors byte ranges, so this script reads the zip's central directory over HTTP
and fetches only those two members. Nothing else is downloaded.

Writes:

    data/annotations/panoptic_val2017.json
    data/annotations/panoptic_val2017/            5,000 PNG files
    data/panoptic_val2017_results.json            perturbed predictions
    data/panoptic_val2017_results/                their PNG files

The predictions are the ground truth perturbed deterministically (seed 42) by
`helpers.perturb_label_map`, the recipe the CI test uses on synthetic maps:
segments dropped, relabeled to another category, merged into a neighbor, eroded,
or shifted, plus spurious segments. That gives PQ well inside (0, 1) with every
matching rule exercised — crowd absorption, void-majority predictions, category
mismatches, and IoU on both sides of 0.5 — which is what `just parity-panoptic`
compares against panopticapi.

Usage:
    uv run python scripts/download_panoptic.py
    just download-panoptic

Flags:
    --force           Overwrite files that already exist
    --skip-download   Skip the download (generate predictions only)
    --skip-generate   Skip prediction generation (download only)
"""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from helpers import DATA_DIR, PANOPTIC_VAL2017, perturb_label_map
from panopticapi.utils import id2rgb, rgb2id
from PIL import Image

ARCHIVE_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
JSON_MEMBER = "annotations/panoptic_val2017.json"
PNG_ZIP_MEMBER = "annotations/panoptic_val2017.zip"


class _RangeFile(io.RawIOBase):
    """A seekable, read-only file over an HTTP resource that supports ranges.

    `zipfile.ZipFile` reads the central directory from the end of the file and
    then seeks to each member it is asked for, so this is enough to extract two
    members from an 860 MB archive without downloading the rest.
    """

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        if head.headers.get("Accept-Ranges") != "bytes":
            raise RuntimeError(f"{url} does not support byte ranges")
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.fetched = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        self.fetched += len(data)
        return data


def _copy_member(zf: zipfile.ZipFile, member: str, dest: Path, label: str) -> None:
    info = zf.getinfo(member)
    done = 0
    with zf.open(info) as src, open(dest, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
            done += len(chunk)
            print(f"\r  {label}: {done / 1_048_576:.1f} / {info.file_size / 1_048_576:.1f} MB", end="", flush=True)
    print()


def download(force: bool) -> None:
    paths = PANOPTIC_VAL2017
    json_dest = paths["gt"]
    png_dir = paths["gt_folder"]
    if not force and json_dest.exists() and png_dir.is_dir() and any(png_dir.iterdir()):
        print(f"  exists: {json_dest}")
        print(f"  exists: {png_dir} ({sum(1 for _ in png_dir.glob('*.png'))} PNG files)")
        return

    json_dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading the archive index from {ARCHIVE_URL}")
    remote = _RangeFile(ARCHIVE_URL)
    with zipfile.ZipFile(remote) as zf:
        _copy_member(zf, JSON_MEMBER, json_dest, "panoptic_val2017.json")
        inner_zip = DATA_DIR / "panoptic_val2017_png.zip"
        _copy_member(zf, PNG_ZIP_MEMBER, inner_zip, "panoptic_val2017.zip")
    print(f"  fetched {remote.fetched / 1_048_576:.1f} MB of {remote.size / 1_048_576:.0f} MB")

    print(f"Extracting PNG files to {png_dir}")
    png_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(inner_zip) as zf:
        members = [m for m in zf.namelist() if m.endswith(".png")]
        for m in members:
            with zf.open(m) as src, open(png_dir / Path(m).name, "wb") as dst:
                dst.write(src.read())
    print(f"  wrote {len(members)} PNG files")
    inner_zip.unlink()


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def generate_predictions(force: bool) -> None:
    paths = PANOPTIC_VAL2017
    pred_json = paths["dt"]
    pred_dir = paths["dt_folder"]
    if not force and pred_json.exists() and pred_dir.is_dir():
        print(f"  exists: {pred_json}")
        return
    gt = json.loads(paths["gt"].read_text())
    category_ids = sorted(c["id"] for c in gt["categories"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    annotations = []
    n = len(gt["annotations"])
    for i, ann in enumerate(gt["annotations"]):
        pan = rgb2id(np.array(Image.open(paths["gt_folder"] / ann["file_name"]).convert("RGB"), dtype=np.uint32))
        pred, segs = perturb_label_map(
            pan, ann["segments_info"], category_ids, rng, shift=12, erode_px=(1, 2, 4, 8), first_id=1, spurious=3
        )
        Image.fromarray(id2rgb(pred)).save(pred_dir / ann["file_name"])
        annotations.append({"image_id": ann["image_id"], "file_name": ann["file_name"], "segments_info": segs})
        if i % 250 == 0 or i == n - 1:
            print(f"\r  predictions: {i + 1} / {n}", end="", flush=True)
    print()
    pred_json.write_text(json.dumps({"annotations": annotations, "categories": gt["categories"]}))
    print(f"  wrote {pred_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Overwrite files that already exist")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download")
    parser.add_argument("--skip-generate", action="store_true", help="Skip prediction generation")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("── Panoptic val2017 ─────────────────────────")
        download(args.force)
    if not args.skip_generate:
        print("── Perturbed predictions ────────────────────")
        generate_predictions(args.force)
    print("Done. Run `just parity-panoptic` to verify.")


if __name__ == "__main__":
    main()
