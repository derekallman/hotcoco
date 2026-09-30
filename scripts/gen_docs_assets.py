"""Generate the figures embedded in the documentation site.

Every image under ``docs/assets/`` is produced here, so a contributor can
regenerate the whole set after a plotting or theming change instead of
hand-editing binaries:

    uv run python scripts/gen_docs_assets.py

The evaluation figures run against real COCO val2017 data in ``data/``, which
is gitignored — the script says what is missing and skips that figure rather
than failing, so the benchmark chart still regenerates on a fresh clone.

Every chart is drawn twice — ``cyanotype`` and ``cyanotype-dark`` — and written as
``<name>.png`` and ``<name>-dark.png``. The docs carry both and hide one via the
``#only-light`` / ``#only-dark`` fragment, so figures follow the palette toggle
instead of glaring white on the dark scheme.

``docs/assets/browse-ui.webp`` is *not* produced here — it is a screenshot of a
running server, so it is captured by hand and only when the browser UI changes:

1. ``uv run coco explore --gt data/demo_val2017/annotations.json \\
       --images data/demo_val2017/images --port 7861``
2. ``"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \\
       --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \\
       --window-size=1400,640 --virtual-time-budget=8000 \\
       --screenshot=browse.png http://localhost:7861``
3. Downscale to 1400px wide and save as WebP q90 — the grid is photographic, so
   PNG costs ~1 MB where WebP costs ~150 KB.

Being hand-captured is why it is the asset most able to drift: every generated
figure followed the Cyanotype swap and this one kept shipping the retired
palette. Re-capture it whenever the browse UI's colors change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import hotcoco  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from helpers import VAL2017, WORKSPACE  # noqa: E402
from hotcoco.plot import confusion_matrix, per_category_ap, pr_curve_iou_sweep, style, tide_errors  # noqa: E402
from hotcoco.plot.core import _configure_axes, _headroom_for_bar_labels, _save_and_return  # noqa: E402

REPO = WORKSPACE
ASSETS = REPO / "docs" / "assets"

GT = VAL2017["bbox"]["gt"]
DT = VAL2017["bbox"]["dt"]


# ---------------------------------------------------------------------------
# Benchmark chart — no dataset needed, numbers come from docs/benchmarks.md
# ---------------------------------------------------------------------------

# Keep in sync with the "Results (1x detections)" table in docs/benchmarks.md.
BENCH = {
    "bbox": {
        "pycocotools": 4.91,
        "faster-coco-eval": 1.32,
        "ultrafast-pycocotools": 0.08,
        "vernier": 0.20,
        "hotcoco": 0.06,
    },
    "segm": {
        "pycocotools": 5.72,
        "faster-coco-eval": 2.92,
        "ultrafast-pycocotools": 0.17,
        "vernier": 0.57,
        "hotcoco": 0.07,
    },
    "keypoints": {
        "pycocotools": 2.27,
        "faster-coco-eval": 1.58,
        "ultrafast-pycocotools": 0.11,
        "vernier": 0.16,
        "hotcoco": 0.05,
    },
}
# hotcoco's speedup vs pycocotools, as the table states it: the median of the
# per-run ratios, not a ratio of the two-decimal times above, which would round
# a 0.134s median into a larger multiple than was measured.
BENCH_SPEEDUP = {"bbox": 83.9, "segm": 81.0, "keypoints": 49.7}


def benchmark_chart(out: Path, theme: str) -> None:
    """Grouped horizontal bars: wall-clock eval time on COCO val2017.

    Deliberately linear, not log. On a linear axis the Rust engines' bars are
    slivers next to pycocotools' — which is the entire point of the figure. A
    log axis would make the five libraries look comparable.
    """
    libs = ["pycocotools", "faster-coco-eval", "ultrafast-pycocotools", "vernier", "hotcoco"]
    kinds = list(BENCH)

    with style(theme=theme):
        # Read the palette back off rcParams rather than the module-level
        # constants, which are the light theme's and would not follow `theme`.
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(libs)]
        tick_color = plt.rcParams["xtick.color"]
        text_color = plt.rcParams["text.color"]

        fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")

        n = len(libs)
        height = 0.16
        for i, (lib, color) in enumerate(zip(libs, colors)):
            ours = lib == "hotcoco"
            offsets = [k - (n - 1) / 2 * height + i * height for k in range(len(kinds))]
            values = [BENCH[kind][lib] for kind in kinds]
            bars = ax.barh(offsets, values, height=height, color=color, label=lib, zorder=3)
            # The speed-up multiple only means anything on our own bar; the
            # others are the baselines it is measured against.
            labels = [
                f"{v:.2f}s  ({BENCH_SPEEDUP[kind]:.0f}×)" if ours else f"{v:.2f}s" for kind, v in zip(kinds, values)
            ]
            ax.bar_label(
                bars,
                labels=labels,
                padding=3,
                fontsize=8.5,
                color=text_color if ours else tick_color,
                fontweight="bold" if ours else "normal",
                zorder=4,
            )

        ax.set_yticks(range(len(kinds)))
        ax.set_yticklabels(kinds)
        ax.invert_yaxis()
        ax.set_xlabel("Evaluation wall clock (seconds) — lower is better")
        # Room for the value labels computed from the data, not a hand-tuned
        # xlim: the longest label's width depends on the vendored face, and a
        # benchmark rerun changes the longest bar.
        _headroom_for_bar_labels(ax, [v for kind in kinds for v in BENCH[kind].values()], axis="x")
        _configure_axes(ax, value_axis="x")
        ax.set_title("COCO val2017, 36,781 detections", loc="left")
        ax.legend(loc="lower right", fontsize=9)

        _save_and_return(fig, ax, out)
        plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Evaluation figures — need real data
# ---------------------------------------------------------------------------


def load_eval():
    """Run the bbox evaluation once, or return None if the dataset is absent.

    Hoisted out of the per-theme loop: nothing here depends on the theme, so
    running it per variant doubled a dataset load and a full evaluation to
    produce identical numbers. `data/` is gitignored, so a missing file is a
    skip rather than a failure — the benchmark chart still regenerates.
    """
    if not GT.exists() or not DT.exists():
        print(f"  skipping evaluation figures: {GT.name} or {DT.name} not found (data/ is gitignored)")
        return None

    print("  loading COCO val2017 ...")
    gt = hotcoco.COCO(str(GT))
    dt = gt.load_res(str(DT))
    ev = hotcoco.COCOeval(gt, dt, "bbox")
    ev.run()
    return ev


def evaluation_figures(ev, theme: str, suffix: str) -> None:
    # AP50 and AP75 only, not the ten-threshold default. Editorial: two curves
    # make the IoU penalty legible at the width the docs render at, where ten
    # crowd each other.
    for name, draw in (
        ("pr-curve", lambda p: pr_curve_iou_sweep(ev, iou_thrs=[0.50, 0.75], theme=theme, save_path=p)),
        ("per-category-ap", lambda p: per_category_ap(ev.results(per_class=True), theme=theme, save_path=p)),
        ("tide-errors", lambda p: tide_errors(ev.tide_errors(), theme=theme, save_path=p)),
        ("confusion-matrix", lambda p: confusion_matrix(ev.confusion_matrix(), top_n=15, theme=theme, save_path=p)),
    ):
        out = ASSETS / f"{name}{suffix}.png"
        draw(out)
        plt.close("all")
        print(f"  wrote {out.relative_to(REPO)}")


# Each figure ships in both schemes; the docs swap them on the palette toggle.
VARIANTS = [("cyanotype", ""), ("cyanotype-dark", "-dark")]


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"generating docs assets into {ASSETS.relative_to(REPO)}/")
    ev = load_eval()
    for theme, suffix in VARIANTS:
        print(f"[{theme}]")
        benchmark_chart(ASSETS / f"benchmark-speed{suffix}.png", theme)
        if ev is not None:
            evaluation_figures(ev, theme, suffix)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
