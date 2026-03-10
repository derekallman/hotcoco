//! Basic COCO evaluation example.
//!
//! Loads ground truth annotations and detection results, runs bbox evaluation,
//! and prints the standard 12-metric summary.
//!
//! ```bash
//! cargo run --example basic_eval -- \
//!     --gt annotations/instances_val2014.json \
//!     --dt instances_val2014_fakebbox100_results.json
//! ```

use std::path::PathBuf;

use hotcoco::params::IouType;
use hotcoco::{COCOeval, COCO};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 || args[1] != "--gt" || args[3] != "--dt" {
        eprintln!("Usage: basic_eval --gt <gt.json> --dt <dt.json>");
        std::process::exit(1);
    }
    let gt_path = PathBuf::from(&args[2]);
    let dt_path = PathBuf::from(&args[4]);

    println!("Loading ground truth...");
    let coco_gt = COCO::new(&gt_path)?;
    println!(
        "  {} images, {} categories, {} annotations",
        coco_gt.get_img_ids(&[], &[]).len(),
        coco_gt.get_cat_ids(&[], &[], &[]).len(),
        coco_gt.get_ann_ids(&[], &[], None, None).len(),
    );

    println!("Loading detections...");
    let coco_dt = coco_gt.load_res(&dt_path)?;
    println!(
        "  {} detections",
        coco_dt.get_ann_ids(&[], &[], None, None).len()
    );

    println!("Running bbox evaluation...");
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();

    if let Some(stats) = &ev.stats {
        println!("\nKey metrics:");
        println!("  AP (IoU=0.50:0.95):  {:.3}", stats[0]);
        println!("  AP (IoU=0.50):       {:.3}", stats[1]);
        println!("  AP (IoU=0.75):       {:.3}", stats[2]);
        println!("  AR (maxDets=100):    {:.3}", stats[8]);
    }

    Ok(())
}
