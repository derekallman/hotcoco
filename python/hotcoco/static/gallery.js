/* hotcoco — Gallery page controller (filters, lightbox, keyboard nav) */

(function () {
    'use strict';

    // ── State ──

    let _shuffleSeed = null;
    let _catView = localStorage.getItem('hotcoco_cat_view') || 'list';
    let _scoreTimer = null;
    let _iouTimer = null;

    // ── Category view toggle ──

    function setCatView(view) {
        _catView = view;
        localStorage.setItem('hotcoco_cat_view', view);
        const listEl = document.getElementById('cat-list');
        const treeEl = document.getElementById('cat-tree');
        document.querySelectorAll('#cat-view-toggle .view-btn').forEach(function(btn) {
            btn.classList.toggle('active', btn.dataset.view === view);
        });
        if (listEl) listEl.style.display = view === 'list' ? '' : 'none';
        if (treeEl) treeEl.style.display = view === 'tree' ? '' : 'none';
        syncCatViews();
    }

    function syncCatViews() {
        const selected = new Set();
        if (_catView === 'list') {
            document.querySelectorAll('#cat-list input[type=checkbox]:checked').forEach(function(c) {
                selected.add(c.value);
            });
            document.querySelectorAll('#cat-tree .tree-child input[type=checkbox]').forEach(function(c) {
                c.checked = selected.has(c.value);
            });
        } else {
            document.querySelectorAll('#cat-tree .tree-child input[type=checkbox]:checked').forEach(function(c) {
                selected.add(c.value);
            });
            document.querySelectorAll('#cat-list input[type=checkbox]').forEach(function(c) {
                c.checked = selected.has(c.value);
            });
        }
        document.querySelectorAll('.tree-group').forEach(function(g) {
            updateGroupState(g);
        });
    }

    // ── Category dropdown ──

    function toggleCatDropdown() {
        const dd = document.getElementById('cat-dropdown');
        dd.classList.toggle('open');
        document.getElementById('cat-panel').classList.toggle('open');
    }

    document.addEventListener('click', function(e) {
        const dd = document.getElementById('cat-dropdown');
        if (dd && !dd.contains(e.target)) {
            dd.classList.remove('open');
            document.getElementById('cat-panel').classList.remove('open');
        }
    });

    function filterCatList(query) {
        const q = query.toLowerCase();
        if (_catView === 'list') {
            document.querySelectorAll('#cat-list .cat-option').forEach(function(opt) {
                opt.style.display = opt.dataset.name.indexOf(q) >= 0 ? '' : 'none';
            });
        } else {
            document.querySelectorAll('.tree-group').forEach(function(g) {
                const children = g.querySelectorAll('.tree-child');
                let anyVisible = false;
                children.forEach(function(c) {
                    const match = !q || c.dataset.name.indexOf(q) >= 0;
                    c.style.display = match ? '' : 'none';
                    if (match) anyVisible = true;
                });
                g.style.display = anyVisible ? '' : 'none';
                if (q && anyVisible) {
                    expandTreeGroup(g.querySelector('.tree-group-header'), true);
                }
            });
        }
    }

    function getSelectedCategories() {
        if (_catView === 'tree' && document.getElementById('cat-tree')) {
            const checks = document.querySelectorAll('#cat-tree .tree-child input[type=checkbox]:checked');
            return Array.from(checks).map(function(c) { return c.value; });
        }
        const checks = document.querySelectorAll('#cat-list input[type=checkbox]:checked');
        return Array.from(checks).map(function(c) { return c.value; });
    }

    function onCatChange() {
        updateCatSummary();
        reloadGallery();
    }

    function updateCatSummary() {
        const sel = getSelectedCategories();
        const summary = document.getElementById('cat-summary');
        if (sel.length === 0) {
            summary.textContent = 'All categories';
        } else if (sel.length <= 2) {
            summary.textContent = sel.join(', ');
        } else {
            summary.textContent = sel.length + ' categories';
        }
    }

    function clearCategories() {
        document.querySelectorAll('#cat-list input[type=checkbox]:checked').forEach(function(c) { c.checked = false; });
        document.querySelectorAll('#cat-tree .tree-child input[type=checkbox]:checked').forEach(function(c) { c.checked = false; });
        document.querySelectorAll('.tree-group').forEach(function(g) { updateGroupState(g); });
        document.getElementById('cat-summary').textContent = 'All categories';
        document.getElementById('cat-search').value = '';
        filterCatList('');
        reloadGallery();
    }

    // ── Tree view ──

    function toggleTreeGroup(header) {
        const expanded = header.getAttribute('aria-expanded') === 'true';
        expandTreeGroup(header, !expanded);
    }

    function expandTreeGroup(header, expand) {
        header.setAttribute('aria-expanded', expand ? 'true' : 'false');
        const children = header.parentElement.querySelector('.tree-children');
        children.style.display = expand ? '' : 'none';
        header.querySelector('.tree-expand-icon').textContent = expand ? '\u25BE' : '\u25B6';
    }

    function toggleGroupCheck(checkbox) {
        const group = checkbox.closest('.tree-group');
        const children = group.querySelectorAll('.tree-child input[type=checkbox]');
        const checked = checkbox.checked;
        children.forEach(function(c) { c.checked = checked; });
        checkbox.indeterminate = false;
        const countEl = group.querySelector('.tree-group-count');
        if (countEl) {
            countEl.textContent = (checked ? children.length : 0) + '/' + countEl.dataset.total;
        }
        updateCatSummary();
        reloadGallery();
    }

    function onTreeChildChange(checkbox) {
        const group = checkbox.closest('.tree-group');
        updateGroupState(group);
        updateCatSummary();
        reloadGallery();
    }

    function updateGroupState(group) {
        const children = group.querySelectorAll('.tree-child input[type=checkbox]');
        const headerCheck = group.querySelector('.tree-group-check');
        let checked = 0;
        children.forEach(function(c) { if (c.checked) checked++; });
        if (headerCheck && children.length > 0) {
            headerCheck.checked = checked === children.length;
            headerCheck.indeterminate = checked > 0 && checked < children.length;
        }
        const countEl = group.querySelector('.tree-group-count');
        if (countEl) {
            countEl.textContent = checked + '/' + countEl.dataset.total;
        }
    }

    function treeKeydown(e, header) {
        if (e.key === 'ArrowRight') {
            expandTreeGroup(header, true);
            e.preventDefault();
        } else if (e.key === 'ArrowLeft') {
            expandTreeGroup(header, false);
            e.preventDefault();
        } else if (e.key === ' ') {
            const check = header.querySelector('.tree-group-check');
            check.checked = !check.checked;
            toggleGroupCheck(check);
            e.preventDefault();
        } else if (e.key === 'ArrowDown') {
            const next = header.parentElement.nextElementSibling;
            if (next) next.querySelector('.tree-group-header').focus();
            e.preventDefault();
        } else if (e.key === 'ArrowUp') {
            const prev = header.parentElement.previousElementSibling;
            if (prev) prev.querySelector('.tree-group-header').focus();
            e.preventDefault();
        }
    }

    // ── Slider controls ──

    function getMinScore() {
        const el = document.getElementById('min-score');
        return el ? parseFloat(el.value) : 0;
    }

    function onScoreChange(val) {
        const pct = (parseFloat(val) * 100).toFixed(1);
        const el = document.getElementById('min-score');
        if (el) el.style.background = 'linear-gradient(to right, var(--accent) ' + pct + '%, var(--border-default) ' + pct + '%)';
        document.getElementById('score-display').textContent = parseFloat(val).toFixed(2);
        clearTimeout(_scoreTimer);
        _scoreTimer = setTimeout(reloadGallery, 250);
    }

    function getSort() {
        const el = document.getElementById('eval-sort');
        return el ? el.value : 'default';
    }

    function getEvalFilter() {
        const el = document.getElementById('eval-filter');
        return el ? el.value : 'none';
    }

    function getIouThr() {
        const el = document.getElementById('iou-thr');
        return el ? parseFloat(el.value) : 0.5;
    }

    function onIouChange(val) {
        const v = parseFloat(val);
        document.getElementById('iou-display').textContent = v.toFixed(2);
        const pct = ((v - 0.5) / 0.45 * 100).toFixed(1);
        const el = document.getElementById('iou-thr');
        if (el) el.style.background = 'linear-gradient(to right, var(--accent) ' + pct + '%, var(--border-default) ' + pct + '%)';
        clearTimeout(_iouTimer);
        _iouTimer = setTimeout(reloadGallery, 300);
    }

    function getSlice() {
        const el = document.getElementById('slice-select');
        return el ? el.value : '';
    }

    // ── Gallery reload ──

    function buildGalleryUrl(page) {
        let url = '/gallery?page=' + (page || 1);
        const cats = getSelectedCategories().join(',');
        if (cats) url += '&categories=' + encodeURIComponent(cats);
        if (_shuffleSeed !== null) url += '&shuffle_seed=' + _shuffleSeed;
        const score = getMinScore();
        if (score > 0) url += '&min_score=' + score;
        const sort = getSort();
        if (sort !== 'default') url += '&sort=' + sort;
        const evalFilter = getEvalFilter();
        if (evalFilter !== 'none') url += '&eval_filter=' + evalFilter;
        const iou = getIouThr();
        if (iou !== 0.5) url += '&iou_thr=' + iou;
        const slice = getSlice();
        if (slice) url += '&slice=' + encodeURIComponent(slice);
        return url;
    }

    function reloadGallery() {
        htmx.ajax('GET', buildGalleryUrl(1), {target: '#gallery', swap: 'innerHTML'});
    }

    function doShuffle() {
        _shuffleSeed = Math.floor(Math.random() * 1000000);
        reloadGallery();
    }

    // ── Lightbox ──

    function openLightbox() {
        document.getElementById('lightbox').classList.add('open');
        document.body.style.overflow = 'hidden';
    }

    function closeLightbox() {
        document.getElementById('lightbox').classList.remove('open');
        document.body.style.overflow = '';
    }

    // Listen for ALL htmx swaps into lightbox-content
    document.body.addEventListener('htmx:afterSettle', function(e) {
        if (e.detail.target && e.detail.target.id === 'lightbox-content') {
            openLightbox();
            requestAnimationFrame(function() {
                if (window.initOverlay) initOverlay();
            });
        }
    });

    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        const lb = document.getElementById('lightbox');
        if (!lb || !lb.classList.contains('open')) return;
        if (e.key === 'Escape') { closeLightbox(); e.preventDefault(); }
        if (e.key === 'ArrowLeft') { navigateLightbox('prev'); e.preventDefault(); }
        if (e.key === 'ArrowRight') { navigateLightbox('next'); e.preventDefault(); }
    });

    // ── Init ──

    // Restore cat view preference
    if (document.getElementById('cat-tree') && _catView === 'tree') {
        setCatView('tree');
    }

    // Parse URL query params (used by dashboard click-through)
    const params = new URLSearchParams(window.location.search);

    const catParam = params.get('categories');
    if (catParam) {
        const catSet = new Set(catParam.split(',').map(function(c) { return c.trim(); }));
        document.querySelectorAll('#cat-list input[type=checkbox]').forEach(function(c) {
            c.checked = catSet.has(c.value);
        });
        syncCatViews();
        updateCatSummary();
    }

    const evalFilterParam = params.get('eval_filter');
    if (evalFilterParam) {
        const el = document.getElementById('eval-filter');
        if (el) el.value = evalFilterParam;
    }

    const sortParam = params.get('sort');
    if (sortParam) {
        const el = document.getElementById('eval-sort');
        if (el) el.value = sortParam;
    }

    // If any URL params were set, override the initial gallery load
    if (catParam || evalFilterParam || sortParam) {
        const gallery = document.getElementById('gallery');
        if (gallery) {
            gallery.setAttribute('hx-get', buildGalleryUrl(1));
        }
    }

    // Open lightbox for a specific image from URL (used by label errors table)
    const detailParam = params.get('detail');
    if (detailParam) {
        document.body.addEventListener('htmx:afterSettle', function onGalleryLoad(e) {
            if (e.detail.target && e.detail.target.id === 'gallery') {
                document.body.removeEventListener('htmx:afterSettle', onGalleryLoad);
                const query = buildGalleryUrl(1).replace('/gallery?', '');
                htmx.ajax('GET', '/detail/' + detailParam + '?' + query, {target: '#lightbox-content', swap: 'innerHTML'});
            }
        });
    }

    // ── HTMX error handling ──

    let _errorTimer = null;
    function showError(msg) {
        const toast = document.getElementById('error-toast');
        if (!toast) return;
        toast.textContent = msg;
        toast.classList.add('visible');
        clearTimeout(_errorTimer);
        _errorTimer = setTimeout(function() {
            toast.classList.remove('visible');
        }, 4000);
    }

    document.body.addEventListener('htmx:responseError', function(e) {
        const status = e.detail.xhr ? e.detail.xhr.status : 0;
        if (status === 404) {
            showError('Image not found');
        } else if (status >= 500) {
            showError('Server error — please try again');
        } else {
            showError('Request failed');
        }
    });

    document.body.addEventListener('htmx:sendError', function() {
        showError('Connection error — is the server running?');
    });

    // ── Expose to inline handlers ──
    // Template onclick/onchange attributes call these directly.

    window.setCatView = setCatView;
    window.toggleCatDropdown = toggleCatDropdown;
    window.filterCatList = filterCatList;
    window.onCatChange = onCatChange;
    window.clearCategories = clearCategories;
    window.toggleTreeGroup = toggleTreeGroup;
    window.toggleGroupCheck = toggleGroupCheck;
    window.onTreeChildChange = onTreeChildChange;
    window.treeKeydown = treeKeydown;
    window.onScoreChange = onScoreChange;
    window.onIouChange = onIouChange;
    window.reloadGallery = reloadGallery;
    window.doShuffle = doShuffle;
    window.closeLightbox = closeLightbox;
    window.buildGalleryUrl = buildGalleryUrl;
})();
