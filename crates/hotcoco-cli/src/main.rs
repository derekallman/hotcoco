use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anstream::stderr;
use anstyle::{AnsiColor, Color, Style};
use clap::{CommandFactory, Parser};
use clap_complete::{generate, Shell};
use hotcoco::params::IouType;
use hotcoco::{COCOeval, COCO};
use indicatif::{ProgressBar, ProgressStyle};

const GREEN: Style = Style::new().fg_color(Some(Color::Ansi(AnsiColor::Green)));
const DIM: Style = Style::new().dimmed();
const RESET: Style = Style::new();

/// Print a styled status line to stderr.
fn status(verb: &str, message: &str, elapsed: Duration) {
    let _ = writeln!(
        stderr(),
        "{GREEN}{verb}{RESET} {message} {DIM}in {:.2}s{RESET}",
        elapsed.as_secs_f64()
    );
}

/// Create a braille spinner on stderr.
fn spinner(message: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner()
        .with_message(message.to_string())
        .with_style(
            ProgressStyle::with_template("{spinner:.green} {msg}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

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

    /// Write evaluation results to a JSON file
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Print shell completion script and exit
    #[arg(long, value_name = "SHELL", hide = true)]
    completions: Option<Shell>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if let Some(shell) = cli.completions {
        generate(
            shell,
            &mut Cli::command(),
            "coco-eval",
            &mut std::io::stdout(),
        );
        return Ok(());
    }

    let gt_name = cli.gt.file_name().unwrap_or_default().to_string_lossy();
    let dt_name = cli.dt.file_name().unwrap_or_default().to_string_lossy();

    let pb = spinner(&format!("Loading ground truth {gt_name}..."));
    let start = Instant::now();
    let coco_gt = COCO::new(&cli.gt)?;
    pb.finish_and_clear();
    status(
        "Loaded",
        &format!("ground truth {DIM}{gt_name}{RESET}"),
        start.elapsed(),
    );

    let pb = spinner(&format!("Loading detections {dt_name}..."));
    let start = Instant::now();
    let coco_dt = coco_gt.load_res(&cli.dt)?;
    pb.finish_and_clear();
    status(
        "Loaded",
        &format!("detections {DIM}{dt_name}{RESET}"),
        start.elapsed(),
    );

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

    let pb = spinner(&format!("Evaluating {}...", cli.iou_type));
    let start = Instant::now();
    coco_eval.evaluate();
    coco_eval.accumulate();
    pb.finish_and_clear();
    status("Evaluated", &format!("{}", cli.iou_type), start.elapsed());

    let _ = writeln!(stderr());
    for line in coco_eval.summarize_lines() {
        println!("{}", line);
    }

    // Print machine-readable stats line for parity testing
    if let Some(stats) = coco_eval.stats() {
        let stats_strs: Vec<String> = stats.iter().map(|v| format!("{:.15}", v)).collect();
        println!("stats: [{}]", stats_strs.join(", "));
    }

    if let Some(ref output_path) = cli.output {
        let start = Instant::now();
        let results = coco_eval
            .results(true)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        results.save(output_path)?;
        status(
            "Saved",
            &format!("results to {DIM}{}{RESET}", output_path.display()),
            start.elapsed(),
        );
    }

    Ok(())
}
