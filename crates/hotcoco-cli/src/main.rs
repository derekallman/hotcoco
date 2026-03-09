use std::path::PathBuf;

use clap::Parser;
use hotcoco::params::IouType;
use hotcoco::{COCOeval, COCO};

#[derive(Parser)]
#[command(name = "coco-eval")]
#[command(
    about = "COCO evaluation tool — compute AP/AR metrics for object detection, segmentation, and keypoints"
)]
struct Cli {
    /// Path to ground truth annotations JSON file
    #[arg(long)]
    gt: PathBuf,

    /// Path to detection results JSON file
    #[arg(long)]
    dt: PathBuf,

    /// IoU type: bbox, segm, or keypoints
    #[arg(long, default_value = "bbox")]
    iou_type: IouType,

    /// Filter to specific image IDs (comma-separated)
    #[arg(long, value_delimiter = ',')]
    img_ids: Option<Vec<u64>>,

    /// Filter to specific category IDs (comma-separated)
    #[arg(long, value_delimiter = ',')]
    cat_ids: Option<Vec<u64>>,

    /// Pool all categories (disable per-category evaluation)
    #[arg(long)]
    no_cats: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    eprintln!("Loading ground truth from {:?}...", cli.gt);
    let coco_gt = COCO::new(&cli.gt)?;

    eprintln!("Loading detections from {:?}...", cli.dt);
    let coco_dt = coco_gt.load_res(&cli.dt)?;

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, cli.iou_type);

    if let Some(img_ids) = cli.img_ids {
        coco_eval.params.img_ids = img_ids;
    }
    if let Some(cat_ids) = cli.cat_ids {
        coco_eval.params.cat_ids = cat_ids;
    }
    if cli.no_cats {
        coco_eval.params.use_cats = false;
    }

    eprintln!("Evaluating...");
    coco_eval.evaluate();

    eprintln!("Accumulating...");
    coco_eval.accumulate();

    coco_eval.summarize();

    // Print machine-readable stats line for parity testing
    if let Some(ref stats) = coco_eval.stats {
        let stats_strs: Vec<String> = stats.iter().map(|v| format!("{:.15}", v)).collect();
        println!("stats: [{}]", stats_strs.join(", "));
    }

    Ok(())
}
