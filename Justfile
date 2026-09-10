# One-time environment setup (run this first, before anything else)
setup:
    uv sync --all-extras
    # rust-analyzer backs editor/LSP code intelligence. It is not in
    # rust-toolchain.toml on purpose (that would make CI download it too), and a
    # channel bump drops it, so re-run `just setup` after bumping. Note
    # ~/.cargo/bin/rust-analyzer is a rustup proxy: `which` finds it even when
    # the component is missing, so absence only shows up as a failed spawn.
    rustup component add rust-analyzer

# Build the Python extension (required before running any Python scripts)
# Run `just setup` first if maturin is missing.
build:
    uv run maturin develop --release

# Run all tests: Rust unit tests + Python parity regression tests
test: build
    cargo test
    uv run pytest scripts/test_parity.py scripts/test_stubs.py scripts/test_theme.py scripts/test_cli.py -v -x --tb=short

# Run hypothesis-based parity fuzzer (slow — for bug hunting, not CI)
fuzz: build
    uv run pytest scripts/fuzz_parity.py -v -x --tb=short

# Verify metric parity vs pycocotools on COCO val2017
parity: build
    uv run python scripts/parity.py

# Generate evaluation report PDF — just report [type=bbox|segm|kpt]
report type="bbox": build
    uv run python scripts/report.py --type {{type}}

# Run performance benchmarks
bench: build
    uv run python scripts/bench.py

# Download COCO val2017 annotations + generate parity result files (~240 MB)
download-coco:
    uv run python scripts/download_coco.py

# Download Objects365 validation set from HuggingFace (~220 MB, requires polars)
download-o365:
    uv run python scripts/download_o365.py

# Download everything needed for all benchmarks
download-all: download-coco download-o365

# Preview docs locally (installs zensical via uv tool if needed)
docs:
    uv tool install zensical --quiet
    zensical serve -o

# Check every internal docs link, heading anchor, and nav entry resolves
docs-links:
    uv run python scripts/check_docs_links.py

# Lint (warnings are errors, matches CI)
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Dependency audit (advisories/licenses/bans) — matches the CI cargo-deny gate.
# Requires cargo-deny: `cargo install cargo-deny`.
audit:
    cargo deny check

# Public Rust API check against the last published release.
# The 1.0 rewrite moves nearly every type between modules while promising the
# crate-root paths keep resolving; this is the mechanical proof of that, rather
# than re-reading lib.rs by hand.
#
# The baseline is deliberately NOT pinned. cargo-semver-checks compares the
# working tree against the newest version on crates.io and only reports a
# failure when the version bump is too small for the API change — so pinning an
# old baseline silently kills the check. Against a pinned 0.5.0, every release
# from 1.0.0 onward reads as a major bump, which permits any removal: the run
# reports "no semver update required" after executing *zero* checks.
#
# Pass a baseline explicitly only to answer a specific question, e.g.
#   just semver --baseline-version 0.5.0
# which is how the 1.0 Tier-2 audit was run while the tree was still 0.5.0.
#
# Requires cargo-semver-checks: `cargo install cargo-semver-checks --locked`.
semver *ARGS:
    cargo semver-checks check-release -p hotcoco {{ARGS}}

# Format all Rust code
fmt:
    cargo fmt --all

# Check formatting without modifying (matches CI)
fmt-check:
    cargo fmt --all -- --check

# Format Python code
py-fmt:
    pre-commit run ruff-format --all-files

# Run the formatting gate; fixes files and fails when changes are needed
py-fmt-check:
    pre-commit run ruff-format --all-files

# Lint Python code
py-lint:
    pre-commit run ruff-check --all-files

# Verify LVIS metric parity vs lvis-api (synthetic data — no data/ needed)
parity-lvis: build
    uv run python scripts/parity_lvis.py

# Verify Open Images parity vs the TensorFlow Object Detection API (frozen fixtures)
parity-oid: build
    uv run python scripts/parity_oid.py

# Regenerate the Open Images fixtures from the TF reference (needs network)
gen-oid-fixtures: build
    uv run python scripts/gen_oid_fixtures.py

# Verify TIDE error-type parity vs tidecv (needs data/)
parity-tide: build
    uv run python scripts/parity_tide.py

# Verify every hotcoco.mask operation against pycocotools.mask, bit-for-bit
parity-mask cases="400": build
    uv run python scripts/parity_mask.py --cases {{cases}}

# Fuzz oriented-box IoU against shapely
fuzz-obb: build
    uv run pytest scripts/fuzz_obb_parity.py -v -x --tb=short

# Diff per-(image, category) matching decisions against pycocotools for one fixture
adversarial fixture: build
    uv run python scripts/adversarial_harness.py {{fixture}}

# Every reference comparison that does not need data/ — what CI runs
parity-all: build
    uv run pytest scripts/test_parity.py -q
    uv run pytest scripts/test_adversarial.py -q
    uv run python scripts/parity_lvis.py
    uv run python scripts/parity_oid.py
    uv run python scripts/parity_mask.py --cases 200

# Diff per-(image, category) matching decisions against pycocotools across the
# whole adversarial corpus — the check metrics alone cannot replace
adversarial-all: build
    uv run pytest scripts/test_adversarial.py -v --tb=short

# Regenerate frozen test oracles (needs scipy/shapely/sklearn/netcal transiently)
gen-fixtures: build
    uv run --with scipy python scripts/gen_assign_fixtures.py
    uv run --with shapely python scripts/gen_obb_fixtures.py
    uv run --with scikit-learn --with netcal python scripts/gen_metrics_fixtures.py
    uv run python scripts/gen_val2017_baseline.py

# Report what target/ is costing, broken down by profile
disk:
    @du -sh target 2>/dev/null || echo "target/ does not exist"
    @du -sh target/* 2>/dev/null | sort -rh || true
    @echo "stray .o files in debug/deps: $(find target/debug/deps -maxdepth 1 -name '*.o' 2>/dev/null | wc -l | tr -d ' ')"

# Cargo has no eviction policy — stale artifacts are never removed, so a
# long-lived working copy only grows. Safe to run anytime: it costs a rebuild
# and nothing else. The compiled Python extension lives at
# python/hotcoco/hotcoco.abi3.so, outside target/, so `import hotcoco` keeps
# working across a clean.
# Reclaim target/ (costs one rebuild, nothing else)
clean:
    cargo clean
    @echo "target/ cleared. Next cargo/just command rebuilds."
