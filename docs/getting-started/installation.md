# Installation

## Python

```bash
pip install hotcoco
```

Verify the installation:

```python
from hotcoco import COCO
print("hotcoco installed successfully")
```

!!! note "numpy"
    hotcoco requires numpy, which is installed automatically. If you need a specific numpy version, install it first.

!!! tip "IDE support"
    hotcoco ships with type stubs (`.pyi`) and a `py.typed` marker. Autocomplete, hover docs, and type checking work out of the box in VS Code, PyCharm, and other editors.

??? info "Build from source"
    Install prerequisites if you don't have them:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh   # uv
    cargo install just                                  # just (requires Rust)
    ```

    Then clone and build:
    ```bash
    git clone https://github.com/derekallman/hotcoco.git
    cd hotcoco
    uv sync --all-extras
    just build
    ```
    This builds the `hotcoco` Python module and installs it into the repo's `.venv`.

## CLI

```bash
cargo install hotcoco-cli
```

This installs the `coco-eval` binary.

??? info "Build from source"
    ```bash
    git clone https://github.com/derekallman/hotcoco.git
    cd hotcoco
    cargo build --release
    # Binary is at target/release/coco-eval
    ```

## Rust library

```bash
cargo add hotcoco
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
hotcoco = "0.3"
```

Full API documentation is on [docs.rs](https://docs.rs/hotcoco).

## Benchmark data

hotcoco's parity checks and benchmarks run against COCO val2017. A single command downloads
the annotations and generates synthetic detection files:

```bash
just download-coco   # ~240 MB — val2017 annotations + parity result files
```

After that, `just parity` and `just bench` work out of the box. For Objects365 scale benchmarks:

```bash
# Requires polars: uv pip install polars
just download-o365   # ~220 MB — Objects365 validation annotations from HuggingFace
```

Expected layout after `just download-coco`:

```
data/
├── annotations/
│   ├── instances_val2017.json
│   └── person_keypoints_val2017.json
├── bbox_val2017_results.json
├── segm_val2017_results.json
└── kpt_val2017_results.json
```

Quick sanity check:

```python
from hotcoco import COCO, COCOeval

coco_gt = COCO("data/annotations/instances_val2017.json")
coco_dt = coco_gt.load_res("data/bbox_val2017_results.json")

ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()
```

!!! tip
    Images are never needed for evaluation — only the JSON annotation and result files.
