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

??? info "Build from source"
    ```bash
    git clone https://github.com/derekallman/hotcoco.git
    cd hotcoco/crates/hotcoco-pyo3
    pip install maturin
    maturin develop --release
    ```
    This builds the `hotcoco` Python module and installs it into your active environment.

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
hotcoco = "0.1"
```

Full API documentation is on [docs.rs](https://docs.rs/hotcoco).

## Sample data

The fastest way to get something running is to use the COCO val2014 annotations and the synthetic detection results from the [cocoapi repository](https://github.com/ppwwyyxx/cocoapi). Everything downloads in seconds and no images are needed.

```bash
# Annotations (~240 MB — instances + keypoints)
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

# Synthetic detection results (a few KB each)
BASE=https://raw.githubusercontent.com/ppwwyyxx/cocoapi/master/results
wget $BASE/instances_val2014_fakebbox100_results.json
wget $BASE/instances_val2014_fakesegm100_results.json
wget $BASE/person_keypoints_val2014_fakekeypoints100_results.json
```

Expected directory layout after the downloads:

```
.
├── annotations/
│   ├── instances_val2014.json
│   └── person_keypoints_val2014.json
├── instances_val2014_fakebbox100_results.json
├── instances_val2014_fakesegm100_results.json
└── person_keypoints_val2014_fakekeypoints100_results.json
```

Then run a quick sanity check:

```python
from hotcoco import COCO, COCOeval

coco_gt = COCO("annotations/instances_val2014.json")
coco_dt = coco_gt.load_res("instances_val2014_fakebbox100_results.json")

ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()
```

!!! tip
    The images (13 GB for val2014) are only needed if you're loading them for visualization or using `CocoDetection` with a dataloader. For evaluation alone, only the JSON files are required.
