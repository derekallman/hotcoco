"""Gradio-powered COCO dataset browser.

Entry points:
  - COCO.browse()         — inline Jupyter or local server
  - coco explore CLI      — standalone Gradio server
  - browse.build_app()    — returns gr.Blocks for advanced use
"""
from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gradio as gr
    from PIL.Image import Image as PILImage

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


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


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


def launch_app(app) -> None:
    """Launch the Gradio app, using inline mode when inside a Jupyter kernel."""
    app.launch(inline=_is_jupyter())


def _require_gradio():
    """Import gradio or raise a helpful ImportError."""
    try:
        import gradio as gr

        return gr
    except ImportError:
        raise ImportError(
            "gradio is required for coco.browse(). "
            "Install it with: pip install hotcoco[browse]"
        ) from None


# ---------------------------------------------------------------------------
# Theme — warm cocoa brand matching docs/stylesheets/extra.css
# ---------------------------------------------------------------------------

def _build_theme(gr):
    """Build a custom Gradio theme matching the hotcoco brand palette."""
    cocoa = gr.themes.Color(
        c50="#F4F1EE",
        c100="#EAE6E1",
        c200="#D5CECC",
        c300="#C5BDB6",
        c400="#9A908A",
        c500="#7A706A",
        c600="#5E554E",
        c700="#4A4540",
        c800="#3A3530",
        c900="#28231F",
        c950="#1C1815",
    )
    theme = gr.themes.Soft(
        primary_hue=cocoa,
        neutral_hue=gr.themes.colors.stone,
        font=[
            gr.themes.GoogleFont("DM Sans"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "monospace",
        ],
    ).set(
        # Surfaces
        body_background_fill="#F4F1EE",
        body_background_fill_dark="#1A1715",
        block_background_fill="#FFFFFF",
        block_background_fill_dark="#252120",
        block_border_color="#E7E5E4",
        block_border_color_dark="#3A3530",
        block_shadow="0 1px 4px rgba(40,35,31,0.10)",
        block_shadow_dark="0 1px 4px rgba(0,0,0,0.30)",
        # Primary buttons — cocoa brown
        button_primary_background_fill="#4A4540",
        button_primary_background_fill_dark="#7A706A",
        button_primary_background_fill_hover="#3A3530",
        button_primary_background_fill_hover_dark="#948880",
        button_primary_text_color="#F4F1EE",
        button_primary_text_color_dark="#F4F1EE",
        button_primary_border_color="#4A4540",
        button_primary_border_color_dark="#7A706A",
        # Secondary buttons — dashed outline style
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover="#EAE6E1",
        button_secondary_background_fill_hover_dark="#2A2520",
        button_secondary_text_color="#4A4540",
        button_secondary_text_color_dark="#C5BDB6",
        button_secondary_border_color="#C5BDB6",
        button_secondary_border_color_dark="#5E554E",
        # Input fields
        input_background_fill="#FFFFFF",
        input_background_fill_dark="#252120",
        input_border_color="#E7E5E4",
        input_border_color_dark="#3A3530",
    )
    return theme


# ---------------------------------------------------------------------------
# CSS — component-level styling injected via gr.Blocks(css=...)
# ---------------------------------------------------------------------------

_CSS = """
/* Full-width, warm background everywhere */
html, body, #app,
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container .gap, .gradio-container .flex,
.contain {
    max-width: 100% !important;
}
html, body, #app, .gradio-container, .contain {
    background-color: #F4F1EE !important;
}
/* .dark is placed on <html> by Gradio, so html.dark targets it directly */
html.dark, .dark body, .dark #app,
.dark .gradio-container, .dark .contain {
    background-color: #1A1715 !important;
}


/* Header banner */
#browse-header {
    background: linear-gradient(135deg, #28231F 0%, #3A3530 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 4px;
    border: none !important;
    box-shadow: 0 3px 12px rgba(40,35,31,0.18);
}
#browse-header p, #browse-header h1, #browse-header h2 {
    color: #F4F1EE !important;
    margin: 0 !important;
}
#browse-header .subtitle {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.78rem;
    color: #9A908A !important;
    margin-top: 4px !important;
    letter-spacing: 0.03em;
}

/* Sidebar card — warm cream so white inputs pop */
#sidebar {
    background: #EAE6E1;
    border: 1px solid #C5BDB6;
    border-radius: 12px;
    padding: 4px;
    box-shadow: 0 1px 4px rgba(40,35,31,0.08);
}
.dark #sidebar {
    background: #201D1B;
    border-color: #3A3530;
}

/* Dropdown (Category) — target the .wrap input container for visible border */
#cat-filter .wrap {
    border: 1.5px solid #9A908A !important;
    background: #FFFFFF !important;
    border-radius: 6px !important;
}
.dark #cat-filter .wrap {
    border-color: #7A706A !important;
    background: #2A2520 !important;
}

/* Filters label — section header with separator */
#sidebar-title {
    border-bottom: 1.5px solid #C5BDB6 !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    background: transparent !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    box-shadow: none !important;
}
#sidebar-title p {
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #5E554E !important;
    margin: 0 !important;
    text-align: center !important;
}
.dark #sidebar-title {
    border-bottom-color: #4A4540 !important;
}
.dark #sidebar-title p {
    color: #9A908A !important;
}

/* Image count — monospace bold number */
#image-count {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0 !important;
}
#image-count p {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.82rem !important;
    color: #7A706A !important;
}
#image-count strong {
    color: #28231F !important;
    font-size: 1.05rem !important;
}

/* Shuffle button */
#shuffle-btn {
    margin-bottom: 2px;
}

/* Load more button — dashed outline */
#load-more-btn button {
    border-style: dashed !important;
    border-width: 1.5px !important;
}
#load-more-btn button:hover {
    background-color: #EAE6E1 !important;
    border-color: #6B7E9A !important;
    color: #4A4540 !important;
}

/* Gallery — rounded corners, hover lift on thumbnails */
#image-gallery {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E7E5E4 !important;
}
#image-gallery .thumbnail-item img,
#image-gallery .grid-wrap img {
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    border-radius: 6px;
}
#image-gallery .thumbnail-item:hover img,
#image-gallery .grid-wrap .thumbnail-item:hover img {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(40,35,31,0.18) !important;
}

/* Lock three-column layout — never wrap even when detail panel is empty */
#main-row {
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
}

/* Detail column — prevent collapsing when panel is empty */
#detail-col {
    min-width: 240px;
    min-height: 660px;
}

/* Detail panel */
#detail-panel {
    border-radius: 12px !important;
    border: 1px solid #E7E5E4 !important;
    box-shadow: 0 1px 4px rgba(40,35,31,0.08) !important;
    overflow: hidden;
    min-height: 640px !important;
}

/* Dark mode overrides */
.dark .gradio-container {
    background-color: #1A1715 !important;
}
.dark #browse-header {
    background: linear-gradient(135deg, #120F0E 0%, #201D1B 100%);
}
.dark #sidebar {
    background: #252120;
    border-color: #3A3530;
}
.dark #image-count p { color: #857870 !important; }
.dark #image-count strong { color: #E0D8D0 !important; }
.dark #image-gallery {
    border-color: #3A3530 !important;
}
.dark #detail-panel {
    border-color: #3A3530 !important;
}
"""


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _count_md(shown: int, total: int) -> str:
    return f"**{shown}** of {total} images"


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

def render_annotated_image(
    coco,
    img_id: int,
    image_dir: str,
    ann_types: list[str],
    cat_colors: dict[int, tuple[int, int, int]],
):
    """Return (PIL.Image, annotations) for gr.AnnotatedImage.

    Segmentation masks are numpy bool arrays (H×W).
    Bounding boxes are (x1, y1, x2, y2) tuples.
    Keypoints are drawn directly onto the image (not natively supported by AnnotatedImage).

    Returns:
        tuple: (PIL.Image, list[tuple[mask_or_bbox, label]]) ready for gr.AnnotatedImage.
    """
    from PIL import ImageDraw

    img_info = coco.load_imgs([img_id])[0]
    img = _load_image(image_dir, img_info["file_name"], img_info)

    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    anns = coco.load_anns(ann_ids)

    cat_ids = list({ann["category_id"] for ann in anns})
    cats = {c["id"]: c for c in coco.load_cats(cat_ids)}

    sections: list[tuple] = []
    draw = None  # lazy-init once per image if keypoints are needed

    for ann in anns:
        cat = cats[ann["category_id"]]
        label = cat["name"]
        color = cat_colors.get(ann["category_id"], (255, 0, 0))
        has_segm = bool(ann.get("segmentation"))
        has_bbox = bool(ann.get("bbox")) and len(ann.get("bbox", [])) == 4
        kpts = ann.get("keypoints", [])
        has_kpts = bool(kpts) and any(v > 0 for v in kpts[2::3])

        if "segm" in ann_types and has_segm:
            try:
                mask = coco.ann_to_mask(ann)  # uint8 H×W
                sections.append((mask.astype(bool), label))
            except Exception:
                pass
        elif "bbox" in ann_types and has_bbox:
            x, y, w, h = ann["bbox"]
            sections.append(((int(x), int(y), int(x + w), int(y + h)), label))

        if "keypoints" in ann_types and has_kpts:
            if draw is None:
                draw = ImageDraw.Draw(img)
            for i in range(0, len(kpts), 3):
                kx, ky, v = kpts[i], kpts[i + 1], kpts[i + 2]
                if v > 0:
                    r = 4
                    draw.ellipse([kx - r, ky - r, kx + r, ky + r], fill=color)
            for link in cat.get("skeleton", []):
                i1, i2 = link[0] - 1, link[1] - 1
                if i1 * 3 + 2 < len(kpts) and i2 * 3 + 2 < len(kpts):
                    x1, y1, v1 = kpts[i1 * 3], kpts[i1 * 3 + 1], kpts[i1 * 3 + 2]
                    x2, y2, v2 = kpts[i2 * 3], kpts[i2 * 3 + 1], kpts[i2 * 3 + 2]
                    if v1 > 0 and v2 > 0:
                        draw.line([x1, y1, x2, y2], fill=color, width=2)

    return (img, sections)


def render_annotated_image_with_dt(
    gt_coco,
    dt_coco,
    img_id: int,
    image_dir: str,
    ann_types: list[str],
    cat_colors: dict[int, tuple[int, int, int]],
    show_gt: bool = True,
    show_dt: bool = True,
    score_thr: float = 0.0,
):
    """Return (PIL.Image, annotations) for gr.AnnotatedImage with GT+DT overlay.

    GT labels are prefixed ``"GT: <name>"``, DT labels ``"DT: <name>"``.
    Detection scores are drawn as text on the image near each DT bbox.

    Returns:
        tuple: (PIL.Image, list[tuple[mask_or_bbox, label]]) ready for gr.AnnotatedImage.
    """
    from PIL import ImageDraw, ImageFont

    img_info = gt_coco.load_imgs([img_id])[0]
    img = _load_image(image_dir, img_info["file_name"], img_info)
    draw = ImageDraw.Draw(img)

    sections: list[tuple] = []

    # --- Ground truth ---
    if show_gt:
        ann_ids = gt_coco.get_ann_ids(img_ids=[img_id])
        anns = gt_coco.load_anns(ann_ids)
        cat_ids = list({ann["category_id"] for ann in anns})
        cats = {c["id"]: c for c in gt_coco.load_cats(cat_ids)}

        for ann in anns:
            cat = cats[ann["category_id"]]
            label = f"GT: {cat['name']}"
            color = cat_colors.get(ann["category_id"], (255, 0, 0))
            has_segm = bool(ann.get("segmentation"))
            has_bbox = bool(ann.get("bbox")) and len(ann.get("bbox", [])) == 4
            kpts = ann.get("keypoints", [])
            has_kpts = bool(kpts) and any(v > 0 for v in kpts[2::3])

            if "segm" in ann_types and has_segm:
                try:
                    mask = gt_coco.ann_to_mask(ann)
                    sections.append((mask.astype(bool), label))
                except Exception:
                    pass
            elif "bbox" in ann_types and has_bbox:
                x, y, w, h = ann["bbox"]
                sections.append(((int(x), int(y), int(x + w), int(y + h)), label))

            if "keypoints" in ann_types and has_kpts:
                for i in range(0, len(kpts), 3):
                    kx, ky, v = kpts[i], kpts[i + 1], kpts[i + 2]
                    if v > 0:
                        r = 4
                        draw.ellipse([kx - r, ky - r, kx + r, ky + r], fill=color)
                for link in cat.get("skeleton", []):
                    i1, i2 = link[0] - 1, link[1] - 1
                    if i1 * 3 + 2 < len(kpts) and i2 * 3 + 2 < len(kpts):
                        x1, y1, v1 = kpts[i1 * 3], kpts[i1 * 3 + 1], kpts[i1 * 3 + 2]
                        x2, y2, v2 = kpts[i2 * 3], kpts[i2 * 3 + 1], kpts[i2 * 3 + 2]
                        if v1 > 0 and v2 > 0:
                            draw.line([x1, y1, x2, y2], fill=color, width=2)

    # --- Detections ---
    if show_dt and dt_coco is not None:
        dt_ann_ids = dt_coco.get_ann_ids(img_ids=[img_id])
        dt_anns = dt_coco.load_anns(dt_ann_ids)
        dt_cat_ids = list({ann["category_id"] for ann in dt_anns})
        dt_cats = {c["id"]: c for c in dt_coco.load_cats(dt_cat_ids)} if dt_cat_ids else {}

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except Exception:
            font = ImageFont.load_default()

        for ann in dt_anns:
            score = ann.get("score", 1.0)
            if score < score_thr:
                continue

            cat = dt_cats.get(ann["category_id"])
            if cat is None:
                continue
            label = f"DT: {cat['name']}"
            color = _lighten_color(cat_colors.get(ann["category_id"], (255, 0, 0)))

            has_segm = bool(ann.get("segmentation"))
            has_bbox = bool(ann.get("bbox")) and len(ann.get("bbox", [])) == 4

            bbox_coords = None
            if "segm" in ann_types and has_segm:
                try:
                    mask = dt_coco.ann_to_mask(ann)
                    sections.append((mask.astype(bool), label))
                    if has_bbox:
                        x, y, w, h = ann["bbox"]
                        bbox_coords = (int(x), int(y), int(x + w), int(y + h))
                except Exception:
                    if has_bbox:
                        x, y, w, h = ann["bbox"]
                        bbox_coords = (int(x), int(y), int(x + w), int(y + h))
                        sections.append((bbox_coords, label))
            elif "bbox" in ann_types and has_bbox:
                x, y, w, h = ann["bbox"]
                bbox_coords = (int(x), int(y), int(x + w), int(y + h))
                sections.append((bbox_coords, label))

            # Draw score text near top-left of bbox
            if bbox_coords is not None:
                score_text = f"{score:.2f}"
                tx, ty = bbox_coords[0] + 2, bbox_coords[1] + 2
                try:
                    bbox_text = draw.textbbox((tx, ty), score_text, font=font)
                    pad = 2
                    draw.rectangle(
                        [bbox_text[0] - pad, bbox_text[1] - pad, bbox_text[2] + pad, bbox_text[3] + pad],
                        fill=(255, 255, 255),
                    )
                except AttributeError:
                    # older Pillow without textbbox
                    draw.rectangle([tx - 2, ty - 2, tx + 30, ty + 14], fill=(255, 255, 255))
                draw.text((tx, ty), score_text, fill=(0, 0, 0), font=font)

    return (img, sections)


def render_thumbnail(
    coco,
    img_id: int,
    image_dir: str,
    max_size: int = 320,
):
    """Return a resized thumbnail for the image grid (no annotation overlay)."""
    img_info = coco.load_imgs([img_id])[0]
    img = _load_image(image_dir, img_info["file_name"], img_info)
    return _resize_thumbnail(img, max_size)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(coco, image_dir: str | None = None, batch_size: int = 12, dt_coco=None):
    """Build and return a Gradio Blocks app for browsing a COCO dataset.

    Args:
        coco: A COCO instance (hotcoco.COCO).
        image_dir: Root directory for image files. Falls back to coco.image_dir.
        batch_size: Number of images to load per batch (default 12).
        dt_coco: Optional detection results COCO object (from ``coco.load_res()``).
            When provided, adds a detection overlay with source toggles and a
            confidence threshold slider.

    Returns:
        gr.Blocks: The Gradio app (not yet launched).

    Raises:
        ValueError: If image_dir is None and coco.image_dir is also None.
        ImportError: If gradio is not installed.
    """
    gr = _require_gradio()

    resolved_dir = image_dir if image_dir is not None else getattr(coco, "image_dir", None)
    if resolved_dir is None:
        raise ValueError(
            "image_dir is required. Set it with:\n"
            "  coco.image_dir = '/path/to/images'\n"
            "or pass it directly:\n"
            "  coco.browse(image_dir='/path/to/images')"
        )

    has_dt = dt_coco is not None

    # Build category data
    all_cats = coco.load_cats(coco.get_cat_ids())
    cat_name_to_id = {c["name"]: c["id"] for c in all_cats}
    cat_choices = [c["name"] for c in all_cats]
    cat_colors = _assign_cat_colors([c["id"] for c in all_cats])

    # color_map for gr.AnnotatedImage — maps label string → hex color
    if has_dt:
        color_map = {}
        for c in all_cats:
            if c["id"] in cat_colors:
                color_map[f"GT: {c['name']}"] = _rgb_to_hex(cat_colors[c["id"]])
                color_map[f"DT: {c['name']}"] = _rgb_to_hex(_lighten_color(cat_colors[c["id"]]))
    else:
        color_map = {c["name"]: _rgb_to_hex(cat_colors[c["id"]]) for c in all_cats if c["id"] in cat_colors}

    all_img_ids = list(coco.get_img_ids())
    total_images = len(all_img_ids)

    theme = _build_theme(gr)

    # --- Inner helpers -------------------------------------------------------

    def _filter_ids(selected_cats: list[str]) -> list[int]:
        if not selected_cats:
            return list(all_img_ids)
        cat_ids = [cat_name_to_id[n] for n in selected_cats]
        return list(coco.get_img_ids(cat_ids=cat_ids))

    def _render_batch(img_ids: list[int]) -> list:
        return [render_thumbnail(coco, iid, resolved_dir) for iid in img_ids]

    def _filter_and_batch(selected_cats: list[str], shuffle: bool = False):
        img_ids = _filter_ids(selected_cats)
        if shuffle:
            random.shuffle(img_ids)
        first_batch = img_ids[:batch_size]
        thumbnails = _render_batch(first_batch)
        count = _count_md(len(first_batch), len(img_ids))
        return img_ids, first_batch, thumbnails, thumbnails, count, None, None

    def _render_detail(img_id, ann_type_list, sources, score_thr):
        if img_id is None:
            return None
        if has_dt:
            show_gt = "Ground truth" in sources
            show_dt = "Detections" in sources
            return render_annotated_image_with_dt(
                coco, dt_coco, img_id, resolved_dir, ann_type_list, cat_colors,
                show_gt=show_gt, show_dt=show_dt, score_thr=score_thr,
            )
        return render_annotated_image(coco, img_id, resolved_dir, ann_type_list, cat_colors)

    # --- Gradio app ----------------------------------------------------------

    with gr.Blocks(title="coco.browse()", theme=theme, css=_CSS, fill_width=True) as app:
        all_ids_state = gr.State([])
        displayed_ids_state = gr.State([])
        rendered_thumbnails_state = gr.State([])
        selected_id_state = gr.State(None)

        # Header
        with gr.Row(elem_id="browse-header"):
            gr.Markdown(
                "## hotcoco\n"
                '<p class="subtitle">Dataset browser</p>'
            )

        # Main content row
        with gr.Row(elem_id="main-row"):
            # Sidebar
            with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
                gr.Markdown("FILTERS", elem_id="sidebar-title")
                cat_filter = gr.Dropdown(
                    choices=cat_choices,
                    multiselect=True,
                    label="Category",
                    value=[],
                    elem_id="cat-filter",
                )
                ann_types = gr.CheckboxGroup(
                    choices=["bbox", "segm", "keypoints"],
                    value=["bbox", "segm", "keypoints"],
                    label="Show",
                    elem_id="ann-types",
                )
                source_toggle = gr.CheckboxGroup(
                    choices=["Ground truth", "Detections"],
                    value=["Ground truth", "Detections"],
                    label="Sources",
                    visible=has_dt,
                )
                score_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Min confidence",
                    visible=has_dt,
                )
                shuffle_btn = gr.Button("Shuffle ⇄", elem_id="shuffle-btn", variant="primary")
                count_md = gr.Markdown(
                    _count_md(0, total_images),
                    elem_id="image-count",
                )

            # Image grid
            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    label="Images",
                    height=640,
                    columns=4,
                    object_fit="cover",
                    allow_preview=False,
                    elem_id="image-gallery",
                )
                load_more_btn = gr.Button("Load more", elem_id="load-more-btn")

            # Detail panel
            with gr.Column(scale=2, elem_id="detail-col"):
                detail = gr.AnnotatedImage(
                    label="Detail",
                    color_map=color_map,
                    height=640,
                    elem_id="detail-panel",
                )

        # --- Event handlers --------------------------------------------------

        def on_filter_change(selected_cats, _):
            return _filter_and_batch(selected_cats)

        def on_ann_type_change(selected_id, ann_type_list, sources, score_thr):
            return _render_detail(selected_id, ann_type_list, sources, score_thr)

        def on_select(displayed_ids, ann_type_list, sources, score_thr, evt):
            img_id = displayed_ids[evt.index]
            return img_id, _render_detail(img_id, ann_type_list, sources, score_thr)
        on_select.__annotations__["evt"] = gr.SelectData  # can't use inline annotation: from __future__ import annotations stringifies it

        def on_load_more(all_ids, displayed_ids, rendered_thumbnails):
            offset = len(displayed_ids)
            next_batch = all_ids[offset : offset + batch_size]
            new_thumbnails = rendered_thumbnails + _render_batch(next_batch)
            new_displayed = displayed_ids + next_batch
            return new_displayed, new_thumbnails, new_thumbnails

        def on_shuffle(selected_cats, _):
            return _filter_and_batch(selected_cats, shuffle=True)

        def on_source_or_score_change(selected_id, ann_type_list, sources, score_thr):
            return _render_detail(selected_id, ann_type_list, sources, score_thr)

        # --- Wire events -----------------------------------------------------

        filter_outputs = [all_ids_state, displayed_ids_state, rendered_thumbnails_state, gallery, count_md, selected_id_state, detail]

        cat_filter.change(on_filter_change, inputs=[cat_filter, ann_types], outputs=filter_outputs)
        ann_types.change(on_ann_type_change, inputs=[selected_id_state, ann_types, source_toggle, score_slider], outputs=[detail])
        gallery.select(on_select, inputs=[displayed_ids_state, ann_types, source_toggle, score_slider], outputs=[selected_id_state, detail])
        load_more_btn.click(
            on_load_more,
            inputs=[all_ids_state, displayed_ids_state, rendered_thumbnails_state],
            outputs=[displayed_ids_state, rendered_thumbnails_state, gallery],
        )
        shuffle_btn.click(on_shuffle, inputs=[cat_filter, ann_types], outputs=filter_outputs)
        source_toggle.change(on_source_or_score_change, inputs=[selected_id_state, ann_types, source_toggle, score_slider], outputs=[detail])
        score_slider.change(on_source_or_score_change, inputs=[selected_id_state, ann_types, source_toggle, score_slider], outputs=[detail])

        # Initialize gallery on load
        app.load(on_filter_change, inputs=[cat_filter, ann_types], outputs=filter_outputs)

    return app
