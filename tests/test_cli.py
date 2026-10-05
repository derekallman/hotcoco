"""CLI surface checks that need no data: flags every user tries first."""

from __future__ import annotations

import importlib.metadata
import json
import sys

import pytest
from hotcoco import cli


def test_version_flag_prints_installed_version(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["coco", "--version"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"coco {importlib.metadata.version('hotcoco')}"


@pytest.mark.parametrize(
    ("to_fmt", "output", "detail"),
    [
        ("yolo", "labels", "skipped (no bbox): 1"),
        ("voc", "voc", "skipped (no bbox): 1"),
        ("oid", "boxes.csv", "skipped (no bbox): 1"),
        ("cvat", "annotations.xml", "skipped (no geometry): 1"),
        ("dota", "dota", "skipped (no obb):  2"),
    ],
)
def test_convert_from_coco_prints_skip_counts(tmp_path, capsys, monkeypatch, to_fmt, output, detail):
    # The human-readable path reads detail rows out of the converter's stats;
    # `--json` skips them, so a stat key the converter does not return only
    # fails here. One annotation has no geometry so a "skipped" row prints.
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40], "area": 2000, "iscrowd": 0},
                    {"id": 2, "image_id": 1, "category_id": 1, "area": 0, "iscrowd": 0},
                ],
                "categories": [{"id": 1, "name": "car"}],
            }
        )
    )
    out = tmp_path / output
    argv = ["coco", "convert", "--from", "coco", "--to", to_fmt, "--input", str(gt), "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code in (None, 0), capsys.readouterr()
    captured = capsys.readouterr()
    assert "error" not in captured.out + captured.err
    assert detail in captured.out
