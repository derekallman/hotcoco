---
name: bench
description: Run hotcoco speed benchmarks against vernier and ultrafast-pycocotools (and pycocotools and faster-coco-eval when publishing), then update the docs and README benchmark tables. Use when the user says "benchmark", "run bench", "how fast is it", or after changes that could affect evaluation performance.
---

# Bench

There are two modes. **Checking a change** is the default and takes under a minute.
**Publishing a capture** rewrites the tables in `docs/benchmarks.md` and `README.md`,
and takes about seven minutes. Never publish from a check run.

## Where the time goes

pycocotools and faster-coco-eval account for about 95% of the suite's wall time. At
1×, they take 21 of the 22 seconds per rep (measured 2026-11-18), and the other three
libraries take about a second. No hotcoco change moves those two, so leave them out
unless you are publishing.

Before you raise `--reps` on any benchmark, check what share of its wall time is the
thing being timed. If setup costs more than about 5× the timed call, fix the harness
instead. In November 2026, an RF-DETR adapter harness started a fresh process for
every timed call: 4–10 s of torch import, data building, and `update()` around a
0.1–0.2 s `compute()`. Its noise drove reps from 7 to 15, and single runs took
15 minutes. Paying the setup once per process gave the same comparison in 65 seconds,
with a third of the noise.

## Checking a change

```bash
just build
uv run python scripts/bench.py --impls ultrafast vernier hotcoco              # ~6 s
uv run python scripts/bench.py --impls ultrafast vernier hotcoco --scale 10   # ~25 s
just bench-arrays                                                             # ~10 s
```

- `just bench-arrays` (`--workload arrays --impls vernier hotcoco`) times boxes passed
  as in-memory arrays: RF-DETR's `compute()` shape, 1.5M detections, each library
  called the way RF-DETR's adapter calls it. The file benchmarks cannot see this
  path, so run it for any change to `from_arrays`, `load_res(array)`, or the
  evaluator's data handling. Add `--threads 1` to compare at one thread. It only
  covers boxes. pycocotools takes about a minute per rep there, so add it with
  `--impls` only when you want the ratio. vernier does its conversion work lazily,
  so compare the `total` row, not `load` against `load`.
- Add `--types bbox` (or `segm`, `keypoints`) when the change touches one IoU type.
- Compare hotcoco against vernier and ultrafast-pycocotools **in the same run**, not
  against the table in `docs/benchmarks.md`. Absolute times from different captures
  are not comparable (see machine state, below).
- For a before-and-after on a hotcoco change, build and run both versions in the
  same sitting, alternating them.
- Report the numbers. Don't edit any doc in this mode.

## Publishing a capture

Publish at a release, or when a check run shows a hotcoco time that moved by more
than about 5%.

1. **Machine state matters more than library versions.** The March 2026 capture ran
   about 1.8× slow across every library, from throttling or memory pressure on the
   fanless 8 GB M1 Air. Numpy and script changes were ruled out by experiment.
   Bench plugged in, on a quiet machine, and check `uptime` first. Only ratios
   within one capture are comparable.

2. Run each table's command once:

   ```bash
   just bench                                          # 1× table and its memory table, ~1.5 min
   uv run python scripts/bench.py --scale 10           # 10× table, ~5 min
   uv run python scripts/bench.py --phases --reps 1    # "Where the time goes", ~40 s
   ```

   `just bench` builds the extension first. Each cell is already the median of 3
   fresh-process runs (`--reps`), so don't rerun the suite and take medians across
   runs. An earlier version of this skill said to run the suite three times, which
   made 9 runs per cell and tripled the capture time. If every library comes out
   slow together, that is machine state: fix the machine and rerun that one
   command. Use **wall clock time**, not CPU time. ultrafast-pycocotools and vernier
   are dev extras; a missing one leaves its column blank.

3. `docs/benchmarks.md` is the **canonical owner** of every benchmark and parity
   number. Update it first, including the **version label** on each table. A table
   labeled with an old version while reporting new timings is the exact drift this
   ownership rule exists to prevent. Then:
   - Sync `README.md`'s headline table and any speedup multiplier quoted in its
     prose. Every figure in the README must match `docs/benchmarks.md` digit for
     digit.
   - `docs/index.md` carries only a headline claim and a link. Check that it still
     reads true, but don't add a table to it.
   - Refresh the "Where the time goes" phase table from the `--phases` run. It is a
     single run and is labeled as such.
   - Re-run `just docs-links` if you added or renamed any heading.

4. Table format:
   - Columns: `Eval Type | pycocotools | faster-coco-eval | ultrafast-pycocotools | vernier | hotcoco`
   - Times in seconds, 2 decimal places
   - Speedups in parentheses vs pycocotools, for example `0.74s (15.9x)`
   - The script also prints a peak-RSS table from the same runs. Carry it into
     `docs/benchmarks.md` beside the timing table it came from. Every cell runs in
     its own process, so the memory column is per library, not cumulative.

5. Only scale detections (never ground truth) for any synthetic load tests.

6. If numbers changed, run `just parity` to confirm metrics still match before
   committing updated tables.
