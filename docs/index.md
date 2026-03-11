<div class="hero" markdown>

# hotcoco

<p class="hero-tagline">
COCO evaluation shouldn't be the slowest part of your training loop.
</p>

<div class="hero-actions" markdown>

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[API Reference](api/coco.md){ .md-button }

</div>

</div>

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<strong>Eval in under a second</strong>
<p>Up to 25× faster than pycocotools. Eval goes from a bottleneck to background noise.</p>
</div>

<div class="feature-card" markdown>
<strong>Your metrics, unchanged</strong>
<p>10,000+ parity tests against pycocotools. Your AP scores won't budge.</p>
</div>

<div class="feature-card" markdown>
<strong>Just pip install</strong>
<p>Prebuilt wheels for Linux, macOS, and Windows. No Cython, no compiler, nothing to build.</p>
</div>

<div class="feature-card" markdown>
<strong>Already works with your stack</strong>
<p><code>init_as_pycocotools()</code> patches imports in-place. Detectron2, mmdetection, RF-DETR — no code changes.</p>
</div>

</div>

## Quick start

```bash
pip install hotcoco
```

=== "Python"

    ```python
    from hotcoco import COCO, COCOeval

    coco_gt = COCO("instances_val2017.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.run()
    ```

=== "Drop-in replacement"

    ```python
    from hotcoco import init_as_pycocotools
    init_as_pycocotools()

    # All pycocotools imports now resolve to hotcoco
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ```

=== "CLI"

    ```bash
    coco eval --gt instances_val2017.json --dt detections.json --iou-type bbox
    ```

=== "Rust"

    ```rust
    use hotcoco::{COCO, COCOeval};
    use hotcoco::params::IouType;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("instances_val2017.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

## Performance

Benchmarked on COCO val2017 (5,000 images, 36,781 GT annotations, ~43,700 detections), Apple M1 MacBook Air:

<div class="benchmark-table" markdown>

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|---------|
| bbox      | 11.79s | 3.47s (3.4×) | **0.74s (15.9×)** |
| segm      | 19.49s | 10.52s (1.9×) | **1.58s (12.3×)** |
| keypoints | 4.79s  | 3.08s (1.6×) | **0.19s (25.0×)** |

</div>

All 12 metrics verified against pycocotools on COCO val2017 with a 10,000+ case parity test suite.

## License

MIT
