"""Tests for hotcoco.browse PIL renderer.

Run with:
    uv run pytest scripts/test_browse.py -v
"""
import os
import tempfile

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_coco(dataset_dict):
    """Build a COCO object from an in-memory dataset dict."""
    from hotcoco import COCO
    return COCO(dataset_dict)


def _minimal_dataset(with_segm=False, with_kpts=False):
    """Return (dataset_dict, tmpdir) with one 100x80 image on disk."""
    tmpdir = tempfile.mkdtemp()
    img = Image.new("RGB", (100, 80), color=(50, 100, 150))
    img.save(os.path.join(tmpdir, "img001.jpg"))

    ann = {
        "id": 1, "image_id": 1, "category_id": 1,
        "bbox": [10, 10, 30, 20], "area": 600, "iscrowd": 0,
        "segmentation": [],
    }
    if with_segm:
        # Simple rectangular polygon
        ann["segmentation"] = [[10, 10, 40, 10, 40, 30, 10, 30]]
    if with_kpts:
        ann["keypoints"] = [25, 20, 2, 0, 0, 0]  # one visible kpt, one invisible
        ann["num_keypoints"] = 1

    cat = {"id": 1, "name": "cat", "supercategory": "animal"}
    if with_kpts:
        cat["keypoints"] = ["nose", "eye"]
        cat["skeleton"] = [[1, 2]]

    dataset = {
        "images": [{"id": 1, "file_name": "img001.jpg", "width": 100, "height": 80}],
        "annotations": [ann],
        "categories": [cat],
    }
    return dataset, tmpdir


# ---------------------------------------------------------------------------
# Color palette tests
# ---------------------------------------------------------------------------

def test_assign_cat_colors_deterministic():
    from hotcoco.browse import _assign_cat_colors
    colors1 = _assign_cat_colors([1, 2, 3])
    colors2 = _assign_cat_colors([1, 2, 3])
    assert colors1 == colors2


def test_assign_cat_colors_wraps_palette():
    from hotcoco.browse import _assign_cat_colors, _PALETTE
    n = len(_PALETTE)
    colors = _assign_cat_colors([0, n, 2 * n])
    # cat 0 and cat n should share the same color
    assert colors[0] == colors[n]


def test_assign_cat_colors_returns_rgb_tuples():
    from hotcoco.browse import _assign_cat_colors
    colors = _assign_cat_colors([42])
    c = colors[42]
    assert isinstance(c, tuple) and len(c) == 3
    assert all(0 <= v <= 255 for v in c)


# ---------------------------------------------------------------------------
# _is_jupyter
# ---------------------------------------------------------------------------

def test_is_jupyter_returns_false_outside_notebook():
    from hotcoco.browse import _is_jupyter
    assert _is_jupyter() is False


# ---------------------------------------------------------------------------
# _load_image
# ---------------------------------------------------------------------------

def test_load_image_returns_pil_image(tmp_path):
    from hotcoco.browse import _load_image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (50, 40), color=(10, 20, 30)).save(str(img_path))
    result = _load_image(str(tmp_path), "test.jpg")
    assert isinstance(result, Image.Image)
    assert result.size == (50, 40)


def test_load_image_missing_returns_placeholder(tmp_path):
    from hotcoco.browse import _load_image
    result = _load_image(str(tmp_path), "nonexistent.jpg")
    assert isinstance(result, Image.Image)
    # placeholder is gray
    arr = np.array(result)
    assert arr.mean() < 200  # not white, is gray-ish


# ---------------------------------------------------------------------------
# render_thumbnail
# ---------------------------------------------------------------------------

def test_render_thumbnail_returns_pil_image():
    from hotcoco.browse import render_thumbnail
    dataset, tmpdir = _minimal_dataset()
    coco = _make_coco(dataset)
    result = render_thumbnail(coco, 1, tmpdir)
    assert isinstance(result, Image.Image)


def test_render_thumbnail_respects_max_size():
    from hotcoco.browse import render_thumbnail
    dataset, tmpdir = _minimal_dataset()
    coco = _make_coco(dataset)
    result = render_thumbnail(coco, 1, tmpdir, max_size=50)
    assert max(result.size) <= 50


# ---------------------------------------------------------------------------
# render_annotated_image
# ---------------------------------------------------------------------------

def test_render_annotated_image_returns_tuple():
    from hotcoco.browse import render_annotated_image, _assign_cat_colors
    dataset, tmpdir = _minimal_dataset(with_segm=True)
    coco = _make_coco(dataset)
    cat_colors = _assign_cat_colors([1])
    result = render_annotated_image(coco, 1, tmpdir, ["bbox", "segm"], cat_colors)
    assert isinstance(result, tuple) and len(result) == 2
    img, sections = result
    assert isinstance(img, Image.Image)
    assert isinstance(sections, list)


def test_render_annotated_image_full_resolution():
    from hotcoco.browse import render_annotated_image, _assign_cat_colors
    dataset, tmpdir = _minimal_dataset()
    coco = _make_coco(dataset)
    cat_colors = _assign_cat_colors([1])
    img, sections = render_annotated_image(coco, 1, tmpdir, ["bbox"], cat_colors)
    assert img.size == (100, 80)


def test_render_annotated_image_segm_returns_bool_mask():
    from hotcoco.browse import render_annotated_image, _assign_cat_colors
    dataset, tmpdir = _minimal_dataset(with_segm=True)
    coco = _make_coco(dataset)
    cat_colors = _assign_cat_colors([1])
    img, sections = render_annotated_image(coco, 1, tmpdir, ["segm"], cat_colors)
    assert len(sections) >= 1
    mask, label = sections[0]
    assert isinstance(mask, np.ndarray) and mask.dtype == bool


def test_render_annotated_image_bbox_returns_tuple():
    from hotcoco.browse import render_annotated_image, _assign_cat_colors
    dataset, tmpdir = _minimal_dataset()
    coco = _make_coco(dataset)
    cat_colors = _assign_cat_colors([1])
    img, sections = render_annotated_image(coco, 1, tmpdir, ["bbox"], cat_colors)
    assert len(sections) >= 1
    bbox, label = sections[0]
    assert isinstance(bbox, tuple) and len(bbox) == 4  # (x1, y1, x2, y2)


def test_render_annotated_image_with_keypoints():
    from hotcoco.browse import render_annotated_image, _assign_cat_colors
    dataset, tmpdir = _minimal_dataset(with_segm=True, with_kpts=True)
    coco = _make_coco(dataset)
    cat_colors = _assign_cat_colors([1])
    img, sections = render_annotated_image(coco, 1, tmpdir, ["bbox", "segm", "keypoints"], cat_colors)
    assert isinstance(img, Image.Image)


# ---------------------------------------------------------------------------
# build_app
# ---------------------------------------------------------------------------

def test_build_app_returns_blocks():
    pytest.importorskip("gradio")
    from hotcoco.browse import build_app
    dataset, tmpdir = _minimal_dataset()
    coco = _make_coco(dataset)
    app = build_app(coco, image_dir=tmpdir)
    import gradio as gr
    assert isinstance(app, gr.Blocks)


def test_build_app_raises_without_image_dir():
    pytest.importorskip("gradio")
    from hotcoco.browse import build_app
    dataset, tmpdir = _minimal_dataset()
    coco = _make_coco(dataset)
    with pytest.raises(ValueError, match="image_dir is required"):
        build_app(coco)


def test_build_app_falls_back_to_coco_image_dir():
    pytest.importorskip("gradio")
    from hotcoco import COCO
    from hotcoco.browse import build_app
    dataset, tmpdir = _minimal_dataset()
    coco = COCO(dataset, image_dir=tmpdir)
    import gradio as gr
    app = build_app(coco)  # no explicit image_dir
    assert isinstance(app, gr.Blocks)


# ---------------------------------------------------------------------------
# COCO.browse() Python API
# ---------------------------------------------------------------------------

def test_coco_has_image_dir_attribute():
    from hotcoco import COCO
    coco = COCO()
    assert hasattr(coco, "image_dir")
    assert coco.image_dir is None


def test_coco_image_dir_via_constructor():
    from hotcoco import COCO
    dataset, tmpdir = _minimal_dataset()
    coco = COCO(dataset, image_dir=tmpdir)
    assert coco.image_dir == tmpdir


def test_coco_image_dir_setter():
    from hotcoco import COCO
    coco = COCO()
    coco.image_dir = "/tmp/images"
    assert coco.image_dir == "/tmp/images"


def test_coco_browse_raises_without_image_dir():
    pytest.importorskip("gradio")
    from hotcoco import COCO
    dataset, _ = _minimal_dataset()
    coco = COCO(dataset)
    with pytest.raises(ValueError, match="image_dir is required"):
        coco.browse()


def test_coco_browse_returns_blocks():
    pytest.importorskip("gradio")
    import gradio as gr
    from hotcoco import COCO
    dataset, tmpdir = _minimal_dataset()
    coco = COCO(dataset, image_dir=tmpdir)
    # Don't actually launch -- just build
    from hotcoco import browse as _browse
    app = _browse.build_app(coco)
    assert isinstance(app, gr.Blocks)


# ---------------------------------------------------------------------------
# CLI: coco explore
# ---------------------------------------------------------------------------

def test_explore_argparse_help():
    """coco explore --help exits 0."""
    import subprocess
    result = subprocess.run(
        ["uv", "run", "coco", "explore", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--gt" in result.stdout
    assert "--images" in result.stdout


def test_explore_missing_gradio_exits_1(tmp_path, monkeypatch):
    """cmd_explore with no gradio installed exits with code 1."""
    import builtins
    import sys

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "gradio":
            raise ImportError("No module named 'gradio'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    # Remove cached gradio from sys.modules if present
    sys.modules.pop("gradio", None)

    from hotcoco.cli import cmd_explore
    import argparse

    args = argparse.Namespace(gt="x.json", images=str(tmp_path), batch_size=12, port=7860, share=False)
    with pytest.raises(SystemExit) as exc:
        cmd_explore(args)
    assert exc.value.code == 1
