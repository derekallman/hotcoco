/* hotcoco — Canvas annotation overlay for dataset browser lightbox */

const state = {
    annotations: [],
    image: null,
    skeleton: [],
    layers: { bbox: true, segm: true, kpts: true },
    sources: { gt: true, dt: true },
    hoveredIdx: null,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    lastDragOffsetX: 0,
    lastDragOffsetY: 0,
};

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
    state.hoveredIdx = null;
    state.scale = 1;
    state.offsetX = 0;
    state.offsetY = 0;

    // Reset layer/source toggles to match checkbox state
    state.layers = { bbox: true, segm: true, kpts: true };
    state.sources = { gt: true, dt: true };

    // The detail image loads from /images/ which may take time.
    // Always attach onload — if already complete, call directly too.
    function onReady() {
        sizeCanvasAndDraw(img, canvas);
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
    canvas.onmouseleave = function () { state.hoveredIdx = null; drawOverlays(canvas, img); syncSidebarHighlight(); };
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

function sizeCanvasAndDraw(img, canvas) {
    // Match canvas size to the rendered image size
    const rect = img.getBoundingClientRect();
    const container = document.getElementById('image-container');
    if (!container) return;

    canvas.width = rect.width;
    canvas.height = rect.height;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    // Position canvas over the image
    const containerRect = container.getBoundingClientRect();
    canvas.style.left = (rect.left - containerRect.left) + 'px';
    canvas.style.top = (rect.top - containerRect.top) + 'px';

    drawOverlays(canvas, img);
}

function drawOverlays(canvas, img) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!state.image || !img.naturalWidth) return;

    // Scale from image coords to canvas coords
    const scaleX = canvas.width / state.image.width;
    const scaleY = canvas.height / state.image.height;

    ctx.save();
    // Apply zoom/pan
    ctx.translate(state.offsetX, state.offsetY);
    ctx.scale(state.scale, state.scale);

    for (let i = 0; i < state.annotations.length; i++) {
        const ann = state.annotations[i];

        // Filter by source
        if (!state.sources[ann.source]) continue;

        const isHovered = state.hoveredIdx === i;
        const hasHover = state.hoveredIdx !== null;
        const baseAlpha = hasHover ? (isHovered ? 0.8 : 0.15) : 0.4;

        const [r, g, b] = ann.color;
        const strokeColor = `rgba(${r}, ${g}, ${b}, ${hasHover ? (isHovered ? 1 : 0.2) : 0.9})`;
        const fillColor = `rgba(${r}, ${g}, ${b}, ${baseAlpha * 0.5})`;

        // Segmentation polygons
        if (state.layers.segm && ann.segmentation) {
            ctx.fillStyle = fillColor;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = isHovered ? 2.5 / state.scale : 1.5 / state.scale;

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
        }

        // Bounding box
        if (state.layers.bbox && ann.bbox) {
            const [bx, by, bw, bh] = ann.bbox;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = isHovered ? 2.5 / state.scale : 1.5 / state.scale;
            ctx.setLineDash(ann.source === 'dt' ? [6 / state.scale, 3 / state.scale] : []);
            ctx.strokeRect(bx * scaleX, by * scaleY, bw * scaleX, bh * scaleY);
            ctx.setLineDash([]);

            // Label + score
            if (isHovered || !hasHover) {
                const label = ann.source === 'dt' && ann.score !== null
                    ? `${ann.category} ${ann.score.toFixed(2)}`
                    : ann.category;
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

        // Keypoints
        if (state.layers.kpts && ann.keypoints) {
            const kpts = ann.keypoints;
            const kptAlpha = hasHover ? (isHovered ? 1 : 0.15) : 0.8;
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

    // Hit test against annotation bboxes
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - state.offsetX) / state.scale;
    const my = (e.clientY - rect.top - state.offsetY) / state.scale;

    const scaleX = canvas.width / state.image.width;
    const scaleY = canvas.height / state.image.height;

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
        drawOverlays(canvas, img);
        syncSidebarHighlight();
    }
}

function onCanvasWheel(e, canvas, img) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.min(10, Math.max(1, state.scale * zoomFactor));
    if (newScale === state.scale) return;

    if (newScale === 1) {
        state.offsetX = 0;
        state.offsetY = 0;
    } else {
        // Zoom toward cursor
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
        const dotColor = `rgb(${ann.color[0]}, ${ann.color[1]}, ${ann.color[2]})`;
        dot.style.background = dotColor;
        dot.style.color = dotColor;

        const label = document.createElement('span');
        label.className = 'ann-label';
        label.textContent = ann.category;

        item.appendChild(dot);
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
            state.hoveredIdx = parseInt(this.dataset.idx);
            if (canvas && img) drawOverlays(canvas, img);
            syncSidebarHighlight();
        });
        item.addEventListener('mouseleave', function () {
            state.hoveredIdx = null;
            if (canvas && img) drawOverlays(canvas, img);
            syncSidebarHighlight();
        });

        list.appendChild(item);
        _annItems.push(item);
        _annIdxToItem[i] = item;
    }
}

var _prevHighlightedItem = null;

function syncSidebarHighlight() {
    if (_prevHighlightedItem) {
        _prevHighlightedItem.classList.remove('highlighted');
    }
    var item = state.hoveredIdx !== null ? _annIdxToItem[state.hoveredIdx] : null;
    if (item) {
        item.classList.add('highlighted');
        item.scrollIntoView({ block: 'nearest' });
    }
    _prevHighlightedItem = item || null;
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
window.navigateLightbox = navigateLightbox;
