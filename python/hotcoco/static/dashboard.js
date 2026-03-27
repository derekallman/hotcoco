/* hotcoco — Dashboard chart resize handler.
   Plotly responsive mode only listens to window resize, not CSS grid
   container changes. This ensures charts re-layout when the sidebar
   collapses at 768px. */

(function() {
    'use strict';

    const grid = document.querySelector('.chart-grid');
    if (!grid || typeof ResizeObserver === 'undefined') return;

    let timeout;
    new ResizeObserver(function() {
        clearTimeout(timeout);
        timeout = setTimeout(function() {
            const plots = grid.querySelectorAll('.js-plotly-plot');
            for (let i = 0; i < plots.length; i++) {
                Plotly.Plots.resize(plots[i]);
            }
        }, 100);
    }).observe(grid);
})();
