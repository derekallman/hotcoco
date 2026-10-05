"""Tests for the hotcoco browse stack: PIL rendering helpers in
``hotcoco.browse`` and the FastAPI app factory in ``hotcoco.server``.

The server tests drive the app through its ASGI interface directly (a
20-line ``_asgi_get``) rather than ``fastapi.testclient``, because the
test client requires ``httpx``, which is not a project dependency.

Run with:
    uv run pytest tests/test_browse.py -v
"""

import os
import sys

import numpy as np
import pytest
from hotcoco import COCO
from PIL import Image


def _asgi_get(app, path: str, query: str = ""):
    """Issue a GET against an ASGI app; return (status, headers, body)."""
    import asyncio

    return asyncio.run(_asgi_get_async(app, path, query))


async def _asgi_get_async(app, path: str, query: str = ""):
    """`_asgi_get` as a coroutine, for tests that issue requests concurrently."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    out = {"status": None, "headers": {}, "chunks": []}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = {k.decode(): v.decode() for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            out["chunks"].append(message.get("body", b""))

    await app(scope, receive, send)
    return out["status"], out["headers"], b"".join(out["chunks"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _dataset(with_segm=False, with_kpts=False):
    """One 100x80 image, one annotation, one category, as a dataset dict."""
    ann = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10, 10, 30, 20],
        "area": 600,
        "iscrowd": 0,
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

    return {
        "images": [{"id": 1, "file_name": "img001.jpg", "width": 100, "height": 80}],
        "annotations": [ann],
        "categories": [cat],
    }


def _minimal_dataset(tmp_path, with_segm=False, with_kpts=False):
    """Return (dataset_dict, tmpdir) with the dataset's image written under `tmp_path`."""
    tmpdir = str(tmp_path)
    Image.new("RGB", (100, 80), color=(50, 100, 150)).save(os.path.join(tmpdir, "img001.jpg"))
    return _dataset(with_segm=with_segm, with_kpts=with_kpts), tmpdir


# ---------------------------------------------------------------------------
# Color palette tests
# ---------------------------------------------------------------------------


def test_assign_cat_colors_deterministic():
    from hotcoco.browse import _assign_cat_colors

    colors1 = _assign_cat_colors([1, 2, 3])
    colors2 = _assign_cat_colors([1, 2, 3])
    assert colors1 == colors2


def test_assign_cat_colors_wraps_palette():
    from hotcoco.browse import _PALETTE, _assign_cat_colors

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


def test_render_thumbnail_returns_pil_image(tmp_path):
    from hotcoco.browse import render_thumbnail

    dataset, tmpdir = _minimal_dataset(tmp_path)
    coco = COCO(dataset)
    result = render_thumbnail(coco, 1, tmpdir)
    assert isinstance(result, Image.Image)


def test_render_thumbnail_respects_max_size(tmp_path):
    from hotcoco.browse import render_thumbnail

    dataset, tmpdir = _minimal_dataset(tmp_path)
    coco = COCO(dataset)
    result = render_thumbnail(coco, 1, tmpdir, max_size=50)
    assert max(result.size) <= 50


# ---------------------------------------------------------------------------
# prepare_annotation_data (the client-side canvas payload — successor to the
# gradio-era render_annotated_image)
# ---------------------------------------------------------------------------


def _payload(with_segm=False, with_kpts=False, **kwargs):
    from hotcoco.browse import _assign_cat_colors, prepare_annotation_data

    dataset = _dataset(with_segm=with_segm, with_kpts=with_kpts)
    coco = COCO(dataset)
    cat_colors = _assign_cat_colors([1])
    return prepare_annotation_data(coco, 1, cat_colors, **kwargs)


def test_prepare_annotation_data_returns_payload():
    data = _payload(with_segm=True)
    assert set(data) == {"image", "annotations", "skeleton", "has_eval", "iou_thr"}
    assert isinstance(data["annotations"], list) and len(data["annotations"]) == 1
    assert data["has_eval"] is False
    assert data["iou_thr"] is None


def test_prepare_annotation_data_full_resolution():
    # The payload carries original pixel dimensions — scaling is the canvas's
    # job, so nothing here may be thumbnailed.
    data = _payload()
    assert data["image"]["id"] == 1
    assert data["image"]["width"] == 100
    assert data["image"]["height"] == 80
    assert data["image"]["file_name"] == "img001.jpg"


def test_prepare_annotation_data_segmentation_polygons():
    data = _payload(with_segm=True)
    entry = data["annotations"][0]
    assert entry["segmentation"] == [[10, 10, 40, 10, 40, 30, 10, 30]]


def test_prepare_annotation_data_bbox_entry():
    data = _payload()
    entry = data["annotations"][0]
    assert entry["bbox"] == [10, 10, 30, 20]  # COCO [x, y, w, h], unscaled
    assert entry["source"] == "gt"
    assert entry["category"] == "cat"
    assert isinstance(entry["color"], list) and len(entry["color"]) == 3
    assert all(0 <= v <= 255 for v in entry["color"])
    assert entry["score"] is None  # ground truth carries no score


def test_prepare_annotation_data_keypoints_and_skeleton():
    data = _payload(with_segm=True, with_kpts=True)
    entry = data["annotations"][0]
    assert entry["keypoints"] == [25, 20, 2, 0, 0, 0]
    assert data["skeleton"] == [[1, 2]]


# ---------------------------------------------------------------------------
# create_app (FastAPI app factory in hotcoco.server)
# ---------------------------------------------------------------------------


def test_create_app_returns_fastapi_and_serves_index(tmp_path):
    from fastapi import FastAPI
    from hotcoco.server import create_app

    dataset, tmpdir = _minimal_dataset(tmp_path)
    coco = COCO(dataset)
    app = create_app(coco, image_dir=tmpdir)
    assert isinstance(app, FastAPI)

    status, headers, body = _asgi_get(app, "/")
    assert status == 200
    assert "text/html" in headers.get("content-type", "")
    assert b"img001.jpg" in body or b"hotcoco" in body.lower()


def test_create_app_raises_without_image_dir(tmp_path):
    from hotcoco.server import create_app

    dataset, _ = _minimal_dataset(tmp_path)
    coco = COCO(dataset)
    with pytest.raises(ValueError, match="image_dir is required"):
        create_app(coco)


def test_create_app_falls_back_to_coco_image_dir(tmp_path):
    from fastapi import FastAPI
    from hotcoco.server import create_app

    dataset, tmpdir = _minimal_dataset(tmp_path)
    coco = COCO(dataset, image_dir=tmpdir)
    app = create_app(coco)  # no explicit image_dir
    assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# COCO.browse() Python API
# ---------------------------------------------------------------------------


def test_coco_has_image_dir_attribute():
    coco = COCO()
    assert hasattr(coco, "image_dir")
    assert coco.image_dir is None


def test_coco_image_dir_via_constructor(tmp_path):
    dataset, tmpdir = _minimal_dataset(tmp_path)
    coco = COCO(dataset, image_dir=tmpdir)
    assert coco.image_dir == tmpdir


def test_coco_image_dir_setter():
    coco = COCO()
    coco.image_dir = "/tmp/images"
    assert coco.image_dir == "/tmp/images"


def test_coco_browse_raises_without_image_dir(tmp_path):
    dataset, _ = _minimal_dataset(tmp_path)
    coco = COCO(dataset)
    with pytest.raises(ValueError, match="image_dir is required"):
        coco.browse()


def test_create_app_serves_gallery_and_thumbnail(tmp_path):
    # The request-level path COCO.browse() wires up: build the app (without
    # launching a server) and hit the endpoints the UI actually loads.
    from hotcoco.server import create_app

    dataset, tmpdir = _minimal_dataset(tmp_path)
    coco = COCO(dataset, image_dir=tmpdir)
    app = create_app(coco)

    status, headers, body = _asgi_get(app, "/gallery")
    assert status == 200, body[:200]
    assert "text/html" in headers.get("content-type", "")

    status, headers, body = _asgi_get(app, "/thumbnail/1")
    assert status == 200, body[:200]
    assert headers.get("content-type", "").startswith("image/")
    assert body[:8] == b"\x89PNG\r\n\x1a\n", "thumbnail should be a PNG"


# ---------------------------------------------------------------------------
# CLI: coco explore
# ---------------------------------------------------------------------------


def test_explore_argparse_help(monkeypatch, capsys):
    """coco explore --help exits 0."""
    from hotcoco import cli

    monkeypatch.setattr(sys, "argv", ["coco", "explore", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--gt" in out
    assert "--images" in out


def test_explore_missing_browse_deps_exits_1(tmp_path, monkeypatch, capsys):
    """cmd_explore without the browse extra (fastapi et al.) exits with code 1.

    The gradio-era version of this test mocked out `gradio`, which is no
    longer a dependency of anything — it kept passing only because the bogus
    `x.json` path also exits 1, i.e. it verified nothing about the deps check.
    """
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "fastapi":
            raise ImportError("No module named 'fastapi'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    import argparse

    from hotcoco.cli import cmd_explore

    # The deps check runs before any argument is touched, so a minimal
    # namespace suffices — reaching further than it would be the failure.
    args = argparse.Namespace(gt="x.json", images=str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        cmd_explore(args)
    assert exc.value.code == 1
    # Exit 1 alone is ambiguous (the bogus x.json path also exits 1); the
    # message proves the *deps* branch fired.
    assert "browse dependencies required" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Eval-aware server: dashboard and inlined JSON
# ---------------------------------------------------------------------------


def _eval_app(tmp_path, summarized=False):
    """An app over an evaluator built the way browse() and `coco explore` build it: evaluate() only.

    With ``summarized``, the caller also ran accumulate() and summarize() before browsing.
    """
    from hotcoco import COCOeval
    from hotcoco.server import create_app

    dataset, tmpdir = _minimal_dataset(tmp_path)
    coco = COCO(dataset, image_dir=tmpdir)
    dt = coco.load_res([{"image_id": 1, "category_id": 1, "bbox": [12, 11, 28, 20], "score": 0.9}])
    ev = COCOeval(coco, dt, "bbox")
    ev.evaluate()
    if summarized:
        ev.accumulate()
        ev.summarize()
    return create_app(coco, dt_coco=dt, coco_eval=ev)


def test_dashboard_on_evaluate_only_eval(tmp_path):
    pytest.importorskip("plotly")
    app = _eval_app(tmp_path)
    status, headers, body = _asgi_get(app, "/dashboard")
    assert status == 200, body[:300]
    assert "text/html" in headers.get("content-type", "")
    # Second request is served from the cache and must not re-accumulate into an error.
    status, _, _ = _asgi_get(app, "/dashboard")
    assert status == 200


class _SlowAccumulate:
    """Delegates to a real COCOeval, with an accumulate() that takes a while and counts its calls."""

    def __init__(self, ev, delay):
        self._ev = ev
        self._delay = delay
        self.accumulate_calls = 0

    def accumulate(self):
        import time

        self.accumulate_calls += 1
        time.sleep(self._delay)
        self._ev.accumulate()

    def __getattr__(self, name):
        return getattr(self._ev, name)


def test_dashboard_first_build_does_not_block_the_server(tmp_path):
    """The first /dashboard request accumulates; that must not stall other requests, or run twice."""
    import asyncio
    import time

    from hotcoco import COCOeval
    from hotcoco.server import create_app

    pytest.importorskip("plotly")
    dataset, tmpdir = _minimal_dataset(tmp_path)
    coco = COCO(dataset, image_dir=tmpdir)
    dt = coco.load_res([{"image_id": 1, "category_id": 1, "bbox": [12, 11, 28, 20], "score": 0.9}])
    real = COCOeval(coco, dt, "bbox")
    real.evaluate()
    slow = _SlowAccumulate(real, delay=0.5)
    app = create_app(coco, dt_coco=dt, coco_eval=slow)

    async def tick():
        # Scheduled before the dashboard requests; a blocked event loop holds it up.
        started = time.perf_counter()
        await asyncio.sleep(0.05)
        return time.perf_counter() - started

    async def run():
        return await asyncio.gather(tick(), _asgi_get_async(app, "/dashboard"), _asgi_get_async(app, "/dashboard"))

    tick_s, first, second = asyncio.run(run())
    assert first[0] == 200 and second[0] == 200
    assert first[2] == second[2]
    assert tick_s < 0.3, f"event loop stalled {tick_s:.2f}s while the dashboard built"
    assert slow.accumulate_calls == 1, "two concurrent first requests both accumulated"


def test_dashboard_on_summarized_eval(tmp_path):
    # `stats` is a populated numpy array here, whose truth value is ambiguous.
    pytest.importorskip("plotly")
    app = _eval_app(tmp_path, summarized=True)
    status, _, body = _asgi_get(app, "/dashboard")
    assert status == 200, body[:300]


def test_detail_query_values_cannot_close_script_tag(tmp_path):
    import json
    import re
    from urllib.parse import parse_qs, quote

    app = _eval_app(tmp_path)
    payload = "</script><script>alert(1)</script>"
    status, _, body = _asgi_get(app, "/detail/1", f"categories={quote(payload)}&slice={quote('a&b=c')}")
    assert status == 200, body[:300]
    html = body.decode()

    # Exactly the one closing tag per JSON script block — the payload cannot add another.
    m = re.search(r'<script type="application/json" id="nav-data">(.*?)</script>', html, re.S)
    assert m is not None
    raw = m.group(1)
    assert "<" not in raw and ">" not in raw
    assert "<script>alert(1)" not in html

    # The escaped form round-trips: JSON.parse (json.loads) gives back a URL-encoded
    # query whose values decode to exactly what was sent.
    nav = json.loads(raw)
    q = parse_qs(nav["query"])
    assert q["categories"] == [payload]
    assert q["slice"] == ["a&b=c"]


def test_annotation_json_escapes_category_names(tmp_path):
    # Dataset strings reach the inlined annotation JSON without passing through
    # URL encoding, so this exercises the JSON escaping on its own.
    import json
    import re

    from hotcoco.server import create_app

    dataset, tmpdir = _minimal_dataset(tmp_path)
    name = "</script><script>alert(1)</script> &"
    dataset["categories"][0]["name"] = name
    app = create_app(COCO(dataset, image_dir=tmpdir))
    status, _, body = _asgi_get(app, "/detail/1")
    assert status == 200, body[:300]
    m = re.search(r'<script type="application/json" id="annotation-data">(.*?)</script>', body.decode(), re.S)
    assert m is not None
    raw = m.group(1)
    assert "<" not in raw and ">" not in raw and "&" not in raw
    assert name in json.dumps(json.loads(raw), ensure_ascii=False)
