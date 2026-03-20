"""FastAPI server for the COCO dataset browser."""
from __future__ import annotations

import asyncio
import collections
import io
import json
import os
import random
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

from . import browse as _browse

_HERE = Path(__file__).parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"


def _build_cat_tree(cats: list[dict]) -> list[dict]:
    """Group categories by supercategory into a tree structure.

    Returns a list of ``{name, children: [{name, id}, ...]}`` dicts,
    sorted alphabetically. If no supercategory data exists, returns [].
    """
    groups: dict[str, list[dict]] = {}
    has_any = False
    for c in cats:
        sc = c.get("supercategory", "")
        if sc:
            has_any = True
        key = sc or "_ungrouped"
        groups.setdefault(key, []).append({"name": c["name"], "id": c["id"]})
    if not has_any:
        return []
    result = []
    for sc_name in sorted(groups):
        children = sorted(groups[sc_name], key=lambda x: x["name"])
        result.append({"name": sc_name, "children": children})
    return result


def create_app(coco, image_dir: str | None = None, batch_size: int = 12, dt_coco=None, coco_eval=None, slices=None) -> FastAPI:
    """Create and return a FastAPI app for browsing a COCO dataset."""
    resolved_dir = image_dir if image_dir is not None else getattr(coco, "image_dir", None)
    if resolved_dir is None:
        raise ValueError(
            "image_dir is required. Set it with:\n"
            "  coco.image_dir = '/path/to/images'\n"
            "or pass it directly:\n"
            "  coco.browse(image_dir='/path/to/images')"
        )

    has_dt = dt_coco is not None
    has_eval = coco_eval is not None

    # Category data
    all_cats = coco.load_cats(coco.get_cat_ids())
    cat_name_to_id = {c["name"]: c["id"] for c in all_cats}
    cat_colors = _browse._assign_cat_colors([c["id"] for c in all_cats])
    cat_tree = _build_cat_tree(all_cats)
    has_hierarchy = len(cat_tree) > 0

    # Slices
    has_slices = slices is not None and len(slices) > 0
    slice_img_sets: dict[str, set[int]] = {}
    slice_metrics: dict[str, dict] = {}
    if has_slices:
        for name, ids in slices.items():
            slice_img_sets[name] = set(ids)
        if has_eval:
            try:
                sliced_results = coco_eval.slice_by(slices)
                # sliced_results is a dict: {slice_name: {num_images, AP, ...delta...}, _overall: {...}}
                slice_metrics = sliced_results
            except Exception:
                pass  # Non-fatal — slices still work for filtering

    all_img_ids = list(coco.get_img_ids())
    total_images = len(all_img_ids)

    # Eval index cache: iou_thr -> eval_index (LRU, 3 entries)
    _eval_cache: collections.OrderedDict[float, dict] = collections.OrderedDict()
    _EVAL_CACHE_MAX = 3

    def _get_eval_index(iou_thr: float) -> dict | None:
        if not has_eval:
            return None
        iou_thr = round(iou_thr, 2)
        if iou_thr in _eval_cache:
            _eval_cache.move_to_end(iou_thr)
            return _eval_cache[iou_thr]
        from .eval_index import build_eval_index
        idx = build_eval_index(coco_eval, iou_thr=iou_thr)
        _eval_cache[iou_thr] = idx
        if len(_eval_cache) > _EVAL_CACHE_MAX:
            _eval_cache.popitem(last=False)
        return idx

    # Thumbnail cache: (image_id, min_score) -> PNG bytes (LRU, bounded)
    _MAX_CACHE = 500
    thumbnail_cache: collections.OrderedDict[tuple[int, float], bytes] = collections.OrderedDict()

    # Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)
    env.filters["number_format"] = lambda v: f"{v:,}"

    app = FastAPI(title="hotcoco browse")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _img_ids_cache: dict[tuple, tuple[list[int], dict[int, int]]] = {}
    _IMG_IDS_CACHE_MAX = 32

    def _resolve_img_ids(
        categories: str | None,
        shuffle_seed: int | None,
        sort: str | None = None,
        eval_filter: str | None = None,
        iou_thr: float = 0.5,
        slice_name: str | None = None,
    ) -> tuple[list[int], dict[int, int]]:
        """Filter by slice, categories, apply eval filter/sort, optionally shuffle."""
        key = (categories, shuffle_seed, sort, eval_filter, round(iou_thr, 2) if has_eval else None, slice_name)
        if key in _img_ids_cache:
            return _img_ids_cache[key]

        # Slice filter (applied first)
        if slice_name and slice_name in slice_img_sets:
            base_ids = [i for i in all_img_ids if i in slice_img_sets[slice_name]]
        else:
            base_ids = all_img_ids

        # Category filter
        if not categories:
            img_ids = list(base_ids)
        else:
            cat_ids = []
            for name in categories.split(","):
                name = name.strip()
                if name in cat_name_to_id:
                    cat_ids.append(cat_name_to_id[name])
            cat_img_ids = set(coco.get_img_ids(cat_ids=cat_ids)) if cat_ids else set(base_ids)
            img_ids = [i for i in base_ids if i in cat_img_ids]

        # Eval filter
        if has_eval and eval_filter and eval_filter != "none":
            eval_index = _get_eval_index(iou_thr)
            img_summary = eval_index["img_summary"]
            if eval_filter == "has_fp":
                img_ids = [i for i in img_ids if img_summary.get(i, {}).get("fp", 0) > 0]
            elif eval_filter == "has_fn":
                img_ids = [i for i in img_ids if img_summary.get(i, {}).get("fn", 0) > 0]
            elif eval_filter == "has_errors":
                img_ids = [i for i in img_ids if img_summary.get(i, {}).get("fp", 0) + img_summary.get(i, {}).get("fn", 0) > 0]
            elif eval_filter == "perfect":
                img_ids = [i for i in img_ids if img_summary.get(i, {}).get("fp", 0) == 0 and img_summary.get(i, {}).get("fn", 0) == 0]

        # Sort (applied before shuffle; shuffle overrides if active)
        if has_eval and sort and sort != "default":
            eval_index = _get_eval_index(iou_thr)
            img_summary = eval_index["img_summary"]
            if sort == "worst":
                img_ids.sort(key=lambda i: -(img_summary.get(i, {}).get("fp", 0) + img_summary.get(i, {}).get("fn", 0)))
            elif sort == "most_fp":
                img_ids.sort(key=lambda i: -img_summary.get(i, {}).get("fp", 0))
            elif sort == "most_fn":
                img_ids.sort(key=lambda i: -img_summary.get(i, {}).get("fn", 0))

        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(img_ids)

        id_to_pos = {img_id: i for i, img_id in enumerate(img_ids)}
        if len(_img_ids_cache) >= _IMG_IDS_CACHE_MAX:
            _img_ids_cache.pop(next(iter(_img_ids_cache)))
        _img_ids_cache[key] = (img_ids, id_to_pos)
        return img_ids, id_to_pos

    def _get_nav(img_ids: list[int], id_to_pos: dict[int, int], current_id: int) -> dict:
        idx = id_to_pos.get(current_id)
        if idx is None:
            return {"prev_id": None, "next_id": None}
        prev_id = img_ids[idx - 1] if idx > 0 else None
        next_id = img_ids[idx + 1] if idx < len(img_ids) - 1 else None
        return {"prev_id": prev_id, "next_id": next_id}

    def _build_query(categories, shuffle_seed, min_score, sort=None, eval_filter=None, iou_thr=None, slice_name=None) -> str:
        parts = []
        if categories:
            parts.append(f"categories={categories}")
        if shuffle_seed is not None:
            parts.append(f"shuffle_seed={shuffle_seed}")
        if min_score > 0:
            parts.append(f"min_score={min_score}")
        if sort and sort != "default":
            parts.append(f"sort={sort}")
        if eval_filter and eval_filter != "none":
            parts.append(f"eval_filter={eval_filter}")
        if iou_thr is not None and iou_thr != 0.5:
            parts.append(f"iou_thr={iou_thr}")
        if slice_name:
            parts.append(f"slice={slice_name}")
        return "&".join(parts)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        template = env.get_template("index.html")
        cat_choices = [c["name"] for c in all_cats]
        # Build slice info for template
        slice_info = []
        if has_slices:
            for name in sorted(slice_img_sets):
                info = {"name": name, "count": len(slice_img_sets[name])}
                sm = slice_metrics.get(name)
                if sm:
                    info["ap"] = sm.get("AP")
                    delta = sm.get("delta", {})
                    info["ap_delta"] = delta.get("AP")
                slice_info.append(info)
        html = template.render(
            cat_choices=cat_choices,
            cat_tree=cat_tree,
            has_hierarchy=has_hierarchy,
            has_dt=has_dt,
            has_eval=has_eval,
            has_slices=has_slices,
            slice_info=slice_info,
            total_images=total_images,
        )
        return HTMLResponse(html)

    @app.get("/gallery", response_class=HTMLResponse)
    async def gallery(
        page: int = Query(1, ge=1),
        categories: str | None = Query(None),
        shuffle_seed: int | None = Query(None),
        min_score: float = Query(0.0, ge=0.0, le=1.0),
        sort: str | None = Query(None),
        eval_filter: str | None = Query(None),
        iou_thr: float = Query(0.5, ge=0.5, le=0.95),
        slice: str | None = Query(None),
    ):
        img_ids, _ = _resolve_img_ids(categories, shuffle_seed, sort=sort, eval_filter=eval_filter, iou_thr=iou_thr, slice_name=slice)

        total = len(img_ids)
        start = (page - 1) * batch_size
        page_ids = img_ids[start : start + batch_size]
        has_next = start + batch_size < total
        showing = min(start + batch_size, total)

        filter_query = _build_query(categories, shuffle_seed, min_score, sort=sort, eval_filter=eval_filter, iou_thr=iou_thr, slice_name=slice)

        # Get per-image eval summaries for gallery badges
        img_summaries = None
        if has_eval:
            eval_index = _get_eval_index(iou_thr)
            img_summaries = eval_index["img_summary"]

        template = env.get_template("partials/gallery.html")
        html = template.render(
            page_ids=page_ids,
            has_next=has_next,
            next_page=page + 1,
            filter_query=filter_query,
            showing=showing,
            total=total,
            min_score=min_score,
            img_summaries=img_summaries,
        )
        return HTMLResponse(html)

    @app.get("/detail/{image_id}", response_class=HTMLResponse)
    async def detail(
        image_id: int,
        categories: str | None = Query(None),
        shuffle_seed: int | None = Query(None),
        min_score: float = Query(0.0, ge=0.0, le=1.0),
        sort: str | None = Query(None),
        eval_filter: str | None = Query(None),
        iou_thr: float = Query(0.5, ge=0.5, le=0.95),
        slice: str | None = Query(None),
    ):
        imgs = coco.load_imgs([image_id])
        if not imgs:
            return Response(status_code=404, content="Image not found")
        img_info = imgs[0]

        img_ids, id_to_pos = _resolve_img_ids(categories, shuffle_seed, sort=sort, eval_filter=eval_filter, iou_thr=iou_thr, slice_name=slice)
        nav = _get_nav(img_ids, id_to_pos, image_id)

        nav_query = _build_query(categories, shuffle_seed, min_score, sort=sort, eval_filter=eval_filter, iou_thr=iou_thr, slice_name=slice)

        eval_index = _get_eval_index(iou_thr) if has_eval else None

        annotation_data = _browse.prepare_annotation_data(
            coco, image_id, cat_colors, dt_coco=dt_coco, score_thr=min_score, img_info=img_info,
            eval_index=eval_index,
        )
        annotation_data["nav"] = nav

        template = env.get_template("partials/detail.html")
        html = template.render(
            image_id=image_id,
            img_info=img_info,
            annotation_json=json.dumps(annotation_data),
            nav=nav,
            nav_query=nav_query,
            has_dt=has_dt,
            has_eval=has_eval,
        )
        return HTMLResponse(html)

    @app.get("/thumbnail/{image_id}")
    async def thumbnail(image_id: int, min_score: float = Query(0.0, ge=0.0, le=1.0)):
        cache_key = (image_id, min_score)
        if cache_key in thumbnail_cache:
            thumbnail_cache.move_to_end(cache_key)
            return Response(content=thumbnail_cache[cache_key], media_type="image/png")

        img = _browse.render_thumbnail(
            coco, image_id, resolved_dir,
            cat_colors=cat_colors, dt_coco=dt_coco, score_thr=min_score,
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        thumbnail_cache[cache_key] = png_bytes
        if len(thumbnail_cache) > _MAX_CACHE:
            thumbnail_cache.popitem(last=False)
        return Response(content=png_bytes, media_type="image/png")

    @app.get("/images/{filename:path}")
    async def serve_image(filename: str):
        path = os.path.realpath(os.path.join(resolved_dir, filename))
        if not path.startswith(os.path.realpath(resolved_dir)):
            return Response(status_code=400, content="Invalid path")
        if not os.path.isfile(path):
            return Response(status_code=404, content="Not found")
        return FileResponse(path)

    return app


def run_server(app: FastAPI, port: int = 7860, open_browser: bool = True):
    """Run the server (blocks). Opens browser on start."""
    import uvicorn

    if open_browser:
        # Open browser after a short delay to let the server start
        def _open():
            import time
            time.sleep(0.5)
            webbrowser.open(f"http://127.0.0.1:{port}")

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def start_server_background(app: FastAPI, port: int = 7860) -> int:
    """Start server in a daemon thread. Returns actual port used.

    Tries port through port+10 on OSError (address in use).
    """
    import socket
    import time

    import uvicorn

    for attempt_port in range(port, port + 11):
        # Pre-check if port is available (avoids silent thread failures)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", attempt_port))
            sock.close()
        except OSError:
            continue

        config = uvicorn.Config(app, host="127.0.0.1", port=attempt_port, log_level="warning")
        server = uvicorn.Server(config)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # Wait for the server to start (or thread to die on error)
        for _ in range(50):
            if server.started or not thread.is_alive():
                break
            time.sleep(0.1)

        if server.started:
            return attempt_port

    raise OSError(f"Could not find an available port in range {port}-{port + 10}")
