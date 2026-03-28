//! Evaluation with custom parameters.
//!
//! Shows how to filter by category, restrict to a subset of images,
//! and use custom IoU thresholds.
//!
//! ```bash
//! cargo run --example custom_params -- \
//!     --gt annotations/instances_val2014.json \
//!     --dt instances_val2014_fakebbox100_results.json
//! ```

use std::path::Path;
use std::path::PathBuf;

use hotcoco::params::IouType;
use hotcoco::{COCO, COCOeval};

fn load(gt_path: &Path, dt_path: &Path) -> Result<(COCO, COCO), Box<dyn std::error::Error>> {
    let coco_gt = COCO::new(gt_path)?;
    let coco_dt = coco_gt.load_res(dt_path)?;
    Ok((coco_gt, coco_dt))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 || args[1] != "--gt" || args[3] != "--dt" {
        eprintln!("Usage: custom_params --gt <gt.json> --dt <dt.json>");
        std::process::exit(1);
    }
    let gt_path = PathBuf::from(&args[2]);
    let dt_path = PathBuf::from(&args[4]);

    // --- Example 1: evaluate only the first 3 categories ---
    println!("=== Subset of categories ===");
    let (coco_gt, coco_dt) = load(&gt_path, &dt_path)?;
    let first_cats: Vec<u64> = coco_gt
        .get_cat_ids(&[], &[], &[])
        .into_iter()
        .take(3)
        .collect();
    println!("Categories: {:?}", first_cats);

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.params.cat_ids = first_cats;
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    // --- Example 2: custom IoU thresholds ---
    println!("\n=== Custom IoU thresholds [0.5, 0.75] ===");
    let (coco_gt, coco_dt) = load(&gt_path, &dt_path)?;
    let mut ev2 = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev2.params.iou_thrs = vec![0.5, 0.75];
    ev2.evaluate();
    ev2.accumulate();
    ev2.summarize();

    // --- Example 3: category-agnostic (pool all categories) ---
    println!("\n=== Class-agnostic evaluation ===");
    let (coco_gt, coco_dt) = load(&gt_path, &dt_path)?;
    let mut ev3 = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev3.params.use_cats = false;
    ev3.evaluate();
    ev3.accumulate();
    ev3.summarize();

    Ok(())
}
