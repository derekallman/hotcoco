# One-time environment setup (run this first, before anything else)
setup:
    uv sync --all-extras

# Build the Python extension (required before running any Python scripts)
# Run `just setup` first if maturin is missing.
build:
    uv run maturin develop --release

# Run all tests: Rust unit tests + Python parity regression tests
test: build
    cargo test
    uv run pytest scripts/test_parity.py -v -x --tb=short

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

# Lint (warnings are errors, matches CI)
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Format all Rust code
fmt:
    cargo fmt --all

# Check formatting without modifying (matches CI)
fmt-check:
    cargo fmt --all -- --check

# Format Python code
py-fmt:
    uv run ruff format python/ scripts/

# Check Python formatting without modifying (CI-safe)
py-fmt-check:
    uv run ruff format --check python/ scripts/

# Lint Python code
py-lint:
    uv run ruff check python/ scripts/
