"""COCO dataset browser — rendering utilities and annotation data preparation.

Entry points:
  - COCO.browse()         — inline Jupyter or local server
  - coco explore CLI      — standalone server
  - browse.prepare_annotation_data()  — JSON-serializable annotation data for canvas overlay
"""
from __future__ import annotations

import math
import os

# ---------------------------------------------------------------------------
# Color palette — 20 visually distinct RGB colors
# ---------------------------------------------------------------------------

_PALETTE: list[tuple[int, int, int]] = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def _assign_cat_colors(cat_ids: list[int]) -> dict[int, tuple[int, int, int]]:
    """Return a deterministic color for each category ID."""
    return {cat_id: _PALETTE[cat_id % len(_PALETTE)] for cat_id in cat_ids}


def _lighten_color(rgb: tuple[int, int, int], factor: float = 0.4) -> tuple[int, int, int]:
    """Blend RGB toward white by *factor* (0 = original, 1 = white)."""
    r, g, b = rgb
    return (
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter kernel."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]  # noqa: F821
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


def _require_browse_deps():
    """Import browse dependencies or raise a helpful error."""
    try:
        import fastapi  # noqa: F401
        import jinja2  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        raise ImportError(
            "Browse dependencies required. Install with: pip install hotcoco[browse]"
        ) from None


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _load_image(image_dir: str, file_name: str, img_info: dict | None = None):
    """Load image from disk; return a gray placeholder if file is missing."""
    from PIL import Image, ImageDraw

    path = os.path.join(image_dir, file_name)
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError:
        pass
    # Gray placeholder with filename, sized to match the image metadata
    w = img_info.get("width", 320) if img_info else 320
    h = img_info.get("height", 240) if img_info else 240
    img = Image.new("RGB", (w, h), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    short = os.path.basename(file_name)
    draw.text((10, h // 2 - 10), f"Missing:\n{short}", fill=(255, 255, 255))
    return img


def _resize_thumbnail(img, max_size: int = 320):
    """Resize image preserving aspect ratio so longest edge <= max_size."""
    from PIL import Image

    w, h = img.size
    scale = max_size / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Public rendering API
# ---------------------------------------------------------------------------

def render_thumbnail(
    coco,
    img_id: int,
    image_dir: str,
    max_size: int = 320,
    cat_colors: dict | None = None,
    dt_coco=None,
    score_thr: float = 0.0,
):
    """Return a resized thumbnail with color-coded bbox outlines.

    GT boxes are solid, DT boxes are dashed. No labels — just outlines.
    """
    from PIL import ImageDraw

    img_info = coco.load_imgs([img_id])[0]
    img = _load_image(image_dir, img_info["file_name"], img_info)
    orig_w, orig_h = img.size
    img = _resize_thumbnail(img, max_size)
    thumb_w, thumb_h = img.size
    sx = thumb_w / orig_w
    sy = thumb_h / orig_h

    draw = ImageDraw.Draw(img)
    line_width = 2

    # GT annotations — solid outlines (OBB as rotated polygon, else AABB)
    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    anns = coco.load_anns(ann_ids)
    for ann in anns:
        color = cat_colors.get(ann["category_id"], (255, 0, 0)) if cat_colors else (255, 0, 0)
        obb = ann.get("obb")
        if obb:
            corners = [(x * sx, y * sy) for x, y in _obb_to_corners(obb)]
            draw.polygon(corners, outline=color, width=line_width)
        elif ann.get("bbox"):
            x, y, w, h = ann["bbox"]
            x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

    # DT annotations — dashed outlines (OBB as rotated polygon, else AABB)
    if dt_coco is not None:
        dt_ann_ids = dt_coco.get_ann_ids(img_ids=[img_id])
        dt_anns = dt_coco.load_anns(dt_ann_ids)
        for ann in dt_anns:
            score = ann.get("score", 1.0)
            if score < score_thr:
                continue
            color = _lighten_color(cat_colors.get(ann["category_id"], (255, 0, 0))) if cat_colors else (200, 200, 255)
            obb = ann.get("obb")
            if obb:
                corners = [(x * sx, y * sy) for x, y in _obb_to_corners(obb)]
                _draw_dashed_polygon(draw, corners, color, width=line_width, dash_len=6)
            elif ann.get("bbox"):
                x, y, w, h = ann["bbox"]
                x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
                _draw_dashed_rect(draw, x0, y0, x1, y1, color, width=line_width, dash_len=6)

    return img


def _draw_dashed_rect(draw, x0, y0, x1, y1, color, width=2, dash_len=6):
    """Draw a dashed rectangle on a PIL ImageDraw."""
    for start, end in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                        ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        _draw_dashed_line(draw, start, end, color, width, dash_len)


def _obb_to_corners(obb):
    """Convert OBB [cx, cy, w, h, angle] to 4 corner points as flat list of (x, y) tuples."""
    cx, cy, w, h, angle = obb
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    hw, hh = w / 2, h / 2
    dx_w, dy_w = hw * cos_a, hw * sin_a
    dx_h, dy_h = hh * sin_a, hh * cos_a
    return [
        (cx - dx_w + dx_h, cy - dy_w - dy_h),
        (cx + dx_w + dx_h, cy + dy_w - dy_h),
        (cx + dx_w - dx_h, cy + dy_w + dy_h),
        (cx - dx_w - dx_h, cy - dy_w + dy_h),
    ]


def _draw_dashed_polygon(draw, corners, color, width=2, dash_len=6):
    """Draw a dashed polygon outline through a list of (x, y) corners."""
    n = len(corners)
    for i in range(n):
        _draw_dashed_line(draw, corners[i], corners[(i + 1) % n], color, width, dash_len)


def _draw_dashed_line(draw, start, end, color, width=2, dash_len=6):
    """Draw a dashed line between two points."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg_end = min(pos + dash_len, length)
        if drawing:
            draw.line(
                [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * seg_end, y0 + uy * seg_end)],
                fill=color, width=width,
            )
        pos = seg_end
        drawing = not drawing


def prepare_annotation_data(coco, img_id: int, cat_colors: dict, dt_coco=None, score_thr: float = 0.0, img_info: dict | None = None, eval_index=None) -> dict:
    """Prepare annotations as a JSON-serializable dict for client-side canvas rendering.

    Returns dict with: image (id/width/height), annotations (list), skeleton (links), nav.
    Annotations include: id, category, color, bbox, score, source, segmentation (polygon coords),
    keypoints, eval_status, matched_id.
    """
    if img_info is None:
        imgs = coco.load_imgs([img_id])
        if not imgs:
            return {"image": {"id": img_id, "width": 0, "height": 0, "file_name": ""}, "annotations": [], "skeleton": []}
        img_info = imgs[0]

    annotations = []
    skeleton = []

    def _collect_anns(coco_obj, source, lighten=False, min_score=0.0):
        """Extract annotations from a COCO object into the annotations list."""
        ann_ids = coco_obj.get_ann_ids(img_ids=[img_id])
        anns = coco_obj.load_anns(ann_ids)
        unique_cat_ids = list({a["category_id"] for a in anns})
        cats = {c["id"]: c for c in coco_obj.load_cats(unique_cat_ids)} if unique_cat_ids else {}

        for ann in anns:
            score = ann.get("score")
            if source == "dt" and (score if score is not None else 1.0) < min_score:
                continue
            cat = cats.get(ann["category_id"])
            if cat is None:
                continue
            base_color = cat_colors.get(ann["category_id"], (255, 0, 0))
            color = list(_lighten_color(base_color)) if lighten else list(base_color)

            ann_id = ann["id"]
            eval_status = None
            matched_id = None
            if eval_index is not None:
                if source == "dt":
                    eval_status = eval_index["dt_status"].get(ann_id)
                    matched_id = eval_index["dt_match"].get(ann_id)
                elif source == "gt":
                    gt_st = eval_index["gt_status"].get(ann_id)
                    eval_status = "fn" if gt_st == "fn" else ("tp" if gt_st == "matched" else None)
                    matched_id = eval_index["gt_match"].get(ann_id)

            entry = {
                "id": ann_id,
                "category": cat["name"],
                "color": color,
                "bbox": ann.get("bbox"),
                "obb": ann.get("obb"),
                "score": round(score, 4) if score is not None else None,
                "source": source,
                "segmentation": None,
                "keypoints": None,
                "eval_status": eval_status,
                "matched_id": matched_id,
            }

            seg = ann.get("segmentation")
            if isinstance(seg, list) and seg:
                entry["segmentation"] = seg

            if source == "gt":
                kpts = ann.get("keypoints", [])
                if kpts and any(v > 0 for v in kpts[2::3]):
                    entry["keypoints"] = kpts
                    nonlocal skeleton
                    if not skeleton and cat.get("skeleton"):
                        skeleton = cat["skeleton"]

            annotations.append(entry)

    _collect_anns(coco, "gt")
    if dt_coco is not None:
        _collect_anns(dt_coco, "dt", lighten=True, min_score=score_thr)

    return {
        "image": {
            "id": img_info["id"],
            "width": img_info.get("width", 0),
            "height": img_info.get("height", 0),
            "file_name": img_info["file_name"],
        },
        "annotations": sorted(annotations, key=lambda a: (a["source"] != "gt", -(a["score"] or 0), a["category"])),
        "skeleton": skeleton,
        "has_eval": eval_index is not None,
        "iou_thr": eval_index["iou_thr"] if eval_index else None,
    }
