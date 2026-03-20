/* hotcoco — Canvas annotation overlay for dataset browser lightbox */

// Eval color scheme
const EVAL_COLORS = {
    tp: { r: 34, g: 197, b: 94 },   // green #22c55e
    fp: { r: 239, g: 68, b: 68 },    // red #ef4444
    fn: { r: 59, g: 130, b: 246 },   // blue #3b82f6
};

const state = {
    annotations: [],
    image: null,
    skeleton: [],
    hasEval: false,
    iouThr: null,
    layers: { bbox: true, segm: true, kpts: true },
    sources: { gt: true, dt: true },
    colorMode: 'category',  // 'category' or 'eval'
    hoveredIdx: null,
    matchHighlightIdx: null,  // index of matched annotation to highlight
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    lastDragOffsetX: 0,
    lastDragOffsetY: 0,
    // Image position within container (set by sizeCanvasAndDraw)
    imgOffsetX: 0,
    imgOffsetY: 0,
    imgW: 0,
    imgH: 0,
};

// Build a lookup from annotation id to index for match highlighting
let _annIdToIdx = {};

function _resolveMatchHighlight(idx) {
    if (idx !== null && state.colorMode === 'eval') {
        const ann = state.annotations[idx];
        if (ann && ann.matched_id) {
            const mi = _annIdToIdx[ann.matched_id];
            if (mi !== undefined) return mi;
        }
    }
    return null;
}

function initOverlay() {
    const dataEl = document.getElementById('annotation-data');
    const img = document.getElementById('detail-img');
    const canvas = document.getElementById('overlay-canvas');
    if (!dataEl || !img || !canvas) return;

    let data;
    try {
        data = JSON.parse(dataEl.textContent);
    } catch (e) {
        return;
    }

    state.annotations = data.annotations || [];
    state.image = data.image;
    state.skeleton = data.skeleton || [];
    state.hasEval = data.has_eval || false;
    state.iouThr = data.iou_thr || null;
    state.hoveredIdx = null;
    state.matchHighlightIdx = null;
    state.scale = 1;
    state.offsetX = 0;
    state.offsetY = 0;

    // On first load, set default color mode; on subsequent images, preserve user choices
    if (!state._initialized) {
        state.layers = { bbox: true, segm: true, kpts: true };
        state.sources = { gt: true, dt: true };
        state.colorMode = state.hasEval ? 'eval' : 'category';
        state._initialized = true;
    }

    // Build id -> index lookup
    _annIdToIdx = {};
    for (let i = 0; i < state.annotations.length; i++) {
        _annIdToIdx[state.annotations[i].id] = i;
    }

    // Sync checkbox DOM to match persisted state
    _syncCheckboxes();
    _syncColorModeToggle();

    // The detail image loads from /images/ which may take time.
    // Always attach onload — if already complete, call directly too.
    // Use double-rAF: the first rAF runs before paint (layout may not
    // be final when the lightbox just opened from display:none), the
    // second rAF runs after paint — layout is guaranteed settled.
    function onReady() {
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                sizeCanvasAndDraw(img, canvas);
            });
        });
    }
    img.onload = onReady;
    if (img.complete && img.naturalWidth > 0) {
        onReady();
    }

    // Build annotation sidebar
    buildAnnotationList();

    // Resize canvas when container resizes (e.g. window resize while lightbox open)
    if (window._overlayResizeObserver) window._overlayResizeObserver.disconnect();
    var container = document.getElementById('image-container');
    if (container && typeof ResizeObserver !== 'undefined') {
        window._overlayResizeObserver = new ResizeObserver(function () {
            if (img.complete && img.naturalWidth > 0) {
                sizeCanvasAndDraw(img, canvas);
            }
        });
        window._overlayResizeObserver.observe(container);
    }

    // Attach mouse events
    canvas.onmousemove = function (e) { onCanvasMouseMove(e, canvas, img); };
    canvas.onmouseleave = function () { state.hoveredIdx = null; state.matchHighlightIdx = null; drawOverlays(canvas, img); syncSidebarHighlight(); };
    canvas.onwheel = function (e) { onCanvasWheel(e, canvas, img); };
    canvas.onmousedown = function (e) { onCanvasMouseDown(e); };
    canvas.onmouseup = function () { state.isDragging = false; canvas.style.cursor = state.scale > 1 ? 'grab' : ''; };
    canvas.ondblclick = function () { state.scale = 1; state.offsetX = 0; state.offsetY = 0; canvas.style.cursor = ''; drawOverlays(canvas, img); };

    // Touch: tap to highlight annotation
    canvas.addEventListener('touchstart', function (e) {
        if (e.touches.length === 1) {
            var touch = e.touches[0];
            // Simulate mousemove for hit-test
            onCanvasMouseMove(touch, canvas, img);
        }
    }, { passive: true });
}

function _getAnnColor(ann) {
    if (state.colorMode === 'eval' && ann.eval_status) {
        const ec = EVAL_COLORS[ann.eval_status];
        if (ec) return [ec.r, ec.g, ec.b];
    }
    return ann.color;
}

function sizeCanvasAndDraw(img, canvas) {
    // Reset transform before measuring so getBoundingClientRect gives true size
    img.style.transform = '';

    const imgRect = img.getBoundingClientRect();
    const container = document.getElementById('image-container');
    if (!container) return;
    const containerRect = container.getBoundingClientRect();

    // Canvas fills the entire container so zoomed content isn't clipped
    canvas.width = containerRect.width;
    canvas.height = containerRect.height;
    canvas.style.width = containerRect.width + 'px';
    canvas.style.height = containerRect.height + 'px';
    canvas.style.left = '0px';
    canvas.style.top = '0px';

    // Track where the image sits within the container
    state.imgOffsetX = imgRect.left - containerRect.left;
    state.imgOffsetY = imgRect.top - containerRect.top;
    state.imgW = imgRect.width;
    state.imgH = imgRect.height;

    // drawOverlays will re-apply the image transform via syncImageTransform
    drawOverlays(canvas, img);
}

function syncImageTransform(img) {
    if (state.scale === 1 && state.offsetX === 0 && state.offsetY === 0) {
        img.style.transform = '';
        img.style.transformOrigin = '';
    } else {
        // Transform origin at image's top-left; offset is relative to image position
        img.style.transformOrigin = '0 0';
        img.style.transform = `translate(${state.offsetX}px, ${state.offsetY}px) scale(${state.scale})`;
    }
}

function drawOverlays(canvas, img) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    syncImageTransform(img);

    if (!state.image || !img.naturalWidth) return;

    // Scale from image coords to rendered image size
    const scaleX = state.imgW / state.image.width;
    const scaleY = state.imgH / state.image.height;

    ctx.save();
    // Translate to image position within container, then apply zoom/pan
    ctx.translate(state.imgOffsetX + state.offsetX, state.imgOffsetY + state.offsetY);
    ctx.scale(state.scale, state.scale);

    for (let i = 0; i < state.annotations.length; i++) {
        const ann = state.annotations[i];

        // Filter by source
        if (!state.sources[ann.source]) continue;

        const isHovered = state.hoveredIdx === i;
        const isMatchHighlight = state.matchHighlightIdx === i;
        const hasHover = state.hoveredIdx !== null;
        const baseAlpha = hasHover ? (isHovered || isMatchHighlight ? 0.8 : 0.15) : 0.4;

        const [r, g, b] = _getAnnColor(ann);
        const activeHighlight = isHovered || isMatchHighlight;
        const strokeColor = `rgba(${r}, ${g}, ${b}, ${hasHover ? (activeHighlight ? 1 : 0.2) : 0.9})`;
        const fillColor = `rgba(${r}, ${g}, ${b}, ${baseAlpha * 0.5})`;

        // Determine dash pattern based on eval mode
        let dashPattern = [];
        if (state.colorMode === 'eval' && ann.eval_status === 'fn') {
            dashPattern = [6 / state.scale, 3 / state.scale];
        } else if (state.colorMode !== 'eval' && ann.source === 'dt') {
            dashPattern = [6 / state.scale, 3 / state.scale];
        }

        // Segmentation polygons
        if (state.layers.segm && ann.segmentation) {
            ctx.fillStyle = fillColor;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = activeHighlight ? 2.5 / state.scale : 1.5 / state.scale;
            ctx.setLineDash(dashPattern);

            for (const poly of ann.segmentation) {
                if (poly.length < 6) continue;
                ctx.beginPath();
                ctx.moveTo(poly[0] * scaleX, poly[1] * scaleY);
                for (let j = 2; j < poly.length; j += 2) {
                    ctx.lineTo(poly[j] * scaleX, poly[j + 1] * scaleY);
                }
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
            }
            ctx.setLineDash([]);
        }

        // Bounding box
        if (state.layers.bbox && ann.bbox) {
            const [bx, by, bw, bh] = ann.bbox;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = activeHighlight ? 2.5 / state.scale : 1.5 / state.scale;
            ctx.setLineDash(dashPattern);
            ctx.strokeRect(bx * scaleX, by * scaleY, bw * scaleX, bh * scaleY);
            ctx.setLineDash([]);

            // Label + score + eval status
            if (activeHighlight || !hasHover) {
                let label;
                if (state.colorMode === 'eval' && ann.eval_status) {
                    const tag = ann.eval_status.toUpperCase();
                    if (ann.source === 'dt' && ann.score !== null) {
                        label = `${tag} ${ann.score.toFixed(2)}`;
                    } else if (ann.eval_status === 'fn') {
                        label = `FN: ${ann.category}`;
                    } else {
                        label = `${tag} ${ann.category}`;
                    }
                } else {
                    label = ann.source === 'dt' && ann.score !== null
                        ? `${ann.category} ${ann.score.toFixed(2)}`
                        : ann.category;
                }
                const fontSize = Math.max(10, 12 / state.scale);
                ctx.font = `600 ${fontSize}px "DM Sans", -apple-system, sans-serif`;
                const tw = ctx.measureText(label).width;
                const pad = 3 / state.scale;
                const lx = bx * scaleX;
                const ly = by * scaleY - fontSize - pad * 2;
                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.85)`;
                ctx.fillRect(lx, ly, tw + pad * 2, fontSize + pad * 2);
                ctx.fillStyle = '#fff';
                ctx.fillText(label, lx + pad, ly + fontSize + pad - 1);
            }
        }

        // Match line: draw a connecting line between matched DT↔GT on hover
        if (state.colorMode === 'eval' && activeHighlight && ann.matched_id && ann.bbox) {
            const matchIdx = _annIdToIdx[ann.matched_id];
            if (matchIdx !== undefined) {
                const matchAnn = state.annotations[matchIdx];
                if (matchAnn && matchAnn.bbox) {
                    const [bx1, by1, bw1, bh1] = ann.bbox;
                    const [bx2, by2, bw2, bh2] = matchAnn.bbox;
                    const cx1 = (bx1 + bw1 / 2) * scaleX;
                    const cy1 = (by1 + bh1 / 2) * scaleY;
                    const cx2 = (bx2 + bw2 / 2) * scaleX;
                    const cy2 = (by2 + bh2 / 2) * scaleY;
                    ctx.beginPath();
                    ctx.moveTo(cx1, cy1);
                    ctx.lineTo(cx2, cy2);
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
                    ctx.lineWidth = 1.5 / state.scale;
                    ctx.setLineDash([4 / state.scale, 3 / state.scale]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }
        }

        // Keypoints
        if (state.layers.kpts && ann.keypoints) {
            const kpts = ann.keypoints;
            const kptAlpha = hasHover ? (activeHighlight ? 1 : 0.15) : 0.8;
            const radius = Math.max(3, 4 / state.scale);

            // Skeleton lines
            if (state.skeleton && state.skeleton.length > 0) {
                ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${kptAlpha * 0.6})`;
                ctx.lineWidth = Math.max(1, 1.5 / state.scale);
                for (const link of state.skeleton) {
                    const i1 = link[0] - 1;
                    const i2 = link[1] - 1;
                    if (i1 * 3 + 2 >= kpts.length || i2 * 3 + 2 >= kpts.length) continue;
                    const v1 = kpts[i1 * 3 + 2];
                    const v2 = kpts[i2 * 3 + 2];
                    if (v1 > 0 && v2 > 0) {
                        ctx.beginPath();
                        ctx.moveTo(kpts[i1 * 3] * scaleX, kpts[i1 * 3 + 1] * scaleY);
                        ctx.lineTo(kpts[i2 * 3] * scaleX, kpts[i2 * 3 + 1] * scaleY);
                        ctx.stroke();
                    }
                }
            }

            // Keypoint dots
            for (let k = 0; k < kpts.length; k += 3) {
                const kx = kpts[k];
                const ky = kpts[k + 1];
                const v = kpts[k + 2];
                if (v > 0) {
                    ctx.beginPath();
                    ctx.arc(kx * scaleX, ky * scaleY, radius, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${kptAlpha})`;
                    ctx.fill();
                    ctx.strokeStyle = `rgba(255, 255, 255, ${kptAlpha * 0.8})`;
                    ctx.lineWidth = 1 / state.scale;
                    ctx.stroke();
                }
            }
        }
    }

    ctx.restore();
}

let _mouseMoveRAF = null;

function onCanvasMouseMove(e, canvas, img) {
    if (state.isDragging) {
        const dx = e.clientX - state.dragStartX;
        const dy = e.clientY - state.dragStartY;
        state.offsetX = state.lastDragOffsetX + dx;
        state.offsetY = state.lastDragOffsetY + dy;
        if (!_mouseMoveRAF) {
            _mouseMoveRAF = requestAnimationFrame(function () {
                _mouseMoveRAF = null;
                drawOverlays(canvas, img);
            });
        }
        return;
    }

    // Hit test against annotation bboxes (account for image offset within container)
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - state.imgOffsetX - state.offsetX) / state.scale;
    const my = (e.clientY - rect.top - state.imgOffsetY - state.offsetY) / state.scale;

    const scaleX = state.imgW / state.image.width;
    const scaleY = state.imgH / state.image.height;

    let hit = null;
    // Iterate in reverse so topmost (last drawn) annotations get priority
    for (let i = state.annotations.length - 1; i >= 0; i--) {
        const ann = state.annotations[i];
        if (!state.sources[ann.source]) continue;
        if (!ann.bbox) continue;
        const [bx, by, bw, bh] = ann.bbox;
        if (mx >= bx * scaleX && mx <= (bx + bw) * scaleX &&
            my >= by * scaleY && my <= (by + bh) * scaleY) {
            hit = i;
            break;
        }
    }

    if (hit !== state.hoveredIdx) {
        state.hoveredIdx = hit;
        state.matchHighlightIdx = _resolveMatchHighlight(hit);
        drawOverlays(canvas, img);
        syncSidebarHighlight();
    }
}

function onCanvasWheel(e, canvas, img) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    // Mouse position relative to image origin (not container origin)
    const mx = e.clientX - rect.left - state.imgOffsetX;
    const my = e.clientY - rect.top - state.imgOffsetY;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.min(10, Math.max(1, state.scale * zoomFactor));
    if (newScale === state.scale) return;

    if (newScale === 1) {
        state.offsetX = 0;
        state.offsetY = 0;
    } else {
        // Zoom toward cursor (relative to image origin)
        state.offsetX = mx - (mx - state.offsetX) * (newScale / state.scale);
        state.offsetY = my - (my - state.offsetY) * (newScale / state.scale);
    }

    state.scale = newScale;
    canvas.style.cursor = newScale > 1 ? 'grab' : '';
    drawOverlays(canvas, img);
}

function onCanvasMouseDown(e) {
    if (state.scale > 1) {
        state.isDragging = true;
        state.dragStartX = e.clientX;
        state.dragStartY = e.clientY;
        state.lastDragOffsetX = state.offsetX;
        state.lastDragOffsetY = state.offsetY;
        const canvas = document.getElementById('overlay-canvas');
        if (canvas) canvas.style.cursor = 'grabbing';
    }
}

// Global mouseup to handle drag ending outside canvas
document.addEventListener('mouseup', function () {
    state.isDragging = false;
    const canvas = document.getElementById('overlay-canvas');
    if (canvas) canvas.style.cursor = state.scale > 1 ? 'grab' : '';
});

function _redraw() {
    const canvas = document.getElementById('overlay-canvas');
    const img = document.getElementById('detail-img');
    if (canvas && img) drawOverlays(canvas, img);
}

function toggleLayer(type) {
    state.layers[type] = !state.layers[type];
    _redraw();
}

function toggleSource(source) {
    state.sources[source] = !state.sources[source];
    _redraw();
    buildAnnotationList();
}

function toggleColorMode() {
    state.colorMode = state.colorMode === 'eval' ? 'category' : 'eval';
    _syncColorModeToggle();
    _redraw();
    buildAnnotationList();
}

function _syncColorModeToggle() {
    const btn = document.getElementById('color-mode-toggle');
    if (!btn) return;
    btn.classList.toggle('active', state.colorMode === 'eval');
}

function _syncCheckboxes() {
    // Sync all toggle UI to persisted state. Called inline from detail.html
    // (parser-blocking script) to prevent any flash before first paint.
    const toggles = document.querySelectorAll('.overlay-toggles input[type="checkbox"]');
    for (const cb of toggles) {
        const handler = cb.getAttribute('onchange') || '';
        const m = handler.match(/toggle(Layer|Source)\('(\w+)'\)/);
        if (!m) continue;
        const [, type, key] = m;
        if (type === 'Layer') cb.checked = !!state.layers[key];
        else cb.checked = !!state.sources[key];
    }
    _syncColorModeToggle();
}

let _annItems = [];
let _annIdxToItem = {};

function buildAnnotationList() {
    const list = document.getElementById('annotation-list');
    if (!list) return;
    list.innerHTML = '';
    _annItems = [];
    _annIdxToItem = {};

    // Count visible annotations
    let visibleCount = 0;
    for (let i = 0; i < state.annotations.length; i++) {
        if (state.sources[state.annotations[i].source]) visibleCount++;
    }
    const countEl = document.getElementById('ann-count');
    if (countEl) countEl.textContent = visibleCount;

    const canvas = document.getElementById('overlay-canvas');
    const img = document.getElementById('detail-img');

    for (let i = 0; i < state.annotations.length; i++) {
        const ann = state.annotations[i];
        if (!state.sources[ann.source]) continue;

        const item = document.createElement('div');
        item.className = 'ann-item';
        item.dataset.idx = i;

        const dot = document.createElement('span');
        dot.className = 'ann-color-dot';
        const [cr, cg, cb] = _getAnnColor(ann);
        const dotColor = `rgb(${cr}, ${cg}, ${cb})`;
        dot.style.background = dotColor;
        dot.style.color = dotColor;

        const label = document.createElement('span');
        label.className = 'ann-label';
        label.textContent = ann.category;

        item.appendChild(dot);

        // Eval badge (before label)
        if (state.colorMode === 'eval' && ann.eval_status) {
            const badge = document.createElement('span');
            badge.className = 'eval-badge eval-' + ann.eval_status;
            badge.textContent = ann.eval_status.toUpperCase();
            item.appendChild(badge);
        }

        item.appendChild(label);

        if (ann.score !== null) {
            const score = document.createElement('span');
            score.className = 'ann-score';
            score.textContent = ann.score.toFixed(2);
            item.appendChild(score);
        }

        const source = document.createElement('span');
        source.className = 'ann-source ' + ann.source;
        source.textContent = ann.source;
        item.appendChild(source);

        // Hover sync: sidebar → canvas (refs cached outside loop)
        item.addEventListener('mouseenter', function () {
            const idx = parseInt(this.dataset.idx);
            state.hoveredIdx = idx;
            state.matchHighlightIdx = _resolveMatchHighlight(idx);
            if (canvas && img) drawOverlays(canvas, img);
            syncSidebarHighlight();
        });
        item.addEventListener('mouseleave', function () {
            state.hoveredIdx = null;
            state.matchHighlightIdx = null;
            if (canvas && img) drawOverlays(canvas, img);
            syncSidebarHighlight();
        });

        list.appendChild(item);
        _annItems.push(item);
        _annIdxToItem[i] = item;
    }
}

var _prevHighlightedItem = null;
var _prevMatchItem = null;

function syncSidebarHighlight() {
    if (_prevHighlightedItem) {
        _prevHighlightedItem.classList.remove('highlighted');
    }
    if (_prevMatchItem) {
        _prevMatchItem.classList.remove('match-highlighted');
    }
    var item = state.hoveredIdx !== null ? _annIdxToItem[state.hoveredIdx] : null;
    if (item) {
        item.classList.add('highlighted');
        item.scrollIntoView({ block: 'nearest' });
    }
    _prevHighlightedItem = item || null;

    var matchItem = state.matchHighlightIdx !== null ? _annIdxToItem[state.matchHighlightIdx] : null;
    if (matchItem) {
        matchItem.classList.add('match-highlighted');
    }
    _prevMatchItem = matchItem || null;
}

function navigateLightbox(direction) {
    if (!window._navData) return;
    const id = direction === 'prev' ? window._navData.prev_id : window._navData.next_id;
    if (id === null) return;
    let url = '/detail/' + id;
    if (window._navData.query) url += '?' + window._navData.query;
    // htmx:afterSettle listener in index.html handles initOverlay
    htmx.ajax('GET', url, { target: '#lightbox-content', swap: 'innerHTML' });
}

// Make functions available globally
window.initOverlay = initOverlay;
window.toggleLayer = toggleLayer;
window.toggleSource = toggleSource;
window.toggleColorMode = toggleColorMode;
window.navigateLightbox = navigateLightbox;
window._syncCheckboxes = _syncCheckboxes;
