use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anstream::stderr;
use anstyle::{AnsiColor, Color, Style};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{Shell, generate};
use hotcoco::params::IouType;
use hotcoco::{COCO, COCOeval, PanopticDataset, PanopticEval};
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
                .expect("valid template")
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

#[derive(Parser)]
#[command(name = "coco-eval", version)]
#[command(
    about = "COCO evaluation tool — compute AP/AR metrics for object detection, segmentation, and keypoints"
)]
// `subcommand_negates_reqs` is what lets the eval args be *required* while still
// allowing `coco-eval completions <shell>` to parse with no dataset: naming a
// subcommand drops the top-level requirements. Enforcing this manually instead
// loses the `required` markers in `--help` and the per-scope usage line in
// errors.
#[command(subcommand_negates_reqs = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    // Detection-eval arguments, also accepted bare (without the `eval`
    // subcommand). Deliberately not a doc comment: clap promotes a flattened
    // struct's doc comment into the top-level `about`, replacing the binary's own
    // description.
    #[command(flatten)]
    eval: EvalArgs,
}

#[derive(Subcommand)]
enum Command {
    /// Evaluate detections against ground truth (the default action)
    Eval(EvalArgs),

    /// Panoptic quality: PQ, SQ, RQ from two COCO panoptic JSON files
    Panoptic(PanopticArgs),

    /// Print a shell completion script
    Completions {
        /// Shell to generate completions for
        shell: Shell,
    },
}

#[derive(clap::Args)]
struct EvalArgs {
    /// Path to ground truth annotations JSON file
    #[arg(long, required_unless_present = "completions")]
    gt: Option<PathBuf>,

    /// Path to detection results JSON file
    #[arg(long, required_unless_present = "completions")]
    dt: Option<PathBuf>,

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

    /// Print shell completion script and exit (superseded by `coco-eval completions`).
    ///
    /// Lives here, not on `Cli`, so the id resolves for `required_unless_present`
    /// in both the bare and `eval` scopes.
    #[arg(long, value_name = "SHELL", hide = true)]
    completions: Option<Shell>,
}

#[derive(clap::Args)]
struct PanopticArgs {
    /// Path to the ground-truth panoptic JSON file
    #[arg(long)]
    gt: PathBuf,

    /// Path to the predicted panoptic JSON file
    #[arg(long)]
    pred: PathBuf,

    /// Folder of ground-truth PNG files (default: --gt without .json)
    #[arg(long)]
    gt_folder: Option<PathBuf>,

    /// Folder of predicted PNG files (default: --pred without .json)
    #[arg(long)]
    pred_folder: Option<PathBuf>,

    /// Write the evaluation report to a JSON file
    #[arg(long, short)]
    output: Option<PathBuf>,
}

/// Write a completion script for `shell` to stdout.
fn print_completions(shell: Shell) {
    generate(
        shell,
        &mut Cli::command(),
        "coco-eval",
        &mut std::io::stdout(),
    );
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Some(Command::Completions { shell }) => {
            print_completions(shell);
            Ok(())
        }
        Some(Command::Eval(args)) => run_eval(args),
        Some(Command::Panoptic(args)) => run_panoptic(args),
        // No subcommand: the historic bare form.
        None => run_eval(cli.eval),
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        // Report through Display, not the default Debug formatting a
        // `-> Result` main would print — a mistyped path should read
        // "failed to load ...: No such file or directory", not
        // `Os { code: 2, kind: NotFound, ... }`.
        Err(e) => {
            let _ = writeln!(stderr(), "error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run_eval(args: EvalArgs) -> Result<(), Box<dyn std::error::Error>> {
    // The pre-subcommand spelling of `coco-eval completions <shell>`.
    if let Some(shell) = args.completions {
        print_completions(shell);
        return Ok(());
    }

    // clap guarantees these are present unless `--completions` was passed, which
    // returned above.
    let (Some(gt_path), Some(dt_path)) = (args.gt.as_ref(), args.dt.as_ref()) else {
        unreachable!("clap enforces --gt/--dt via required_unless_present")
    };

    let coco_gt = timed_load("ground truth", gt_path, || COCO::new(gt_path))?;
    let coco_dt = timed_load("detections", dt_path, || coco_gt.load_res(dt_path))?;

    let mut coco_eval = COCOeval::new(coco_gt, coco_dt, args.iou_type);

    if let Some(img_ids) = args.img_ids {
        coco_eval.params.img_ids = img_ids;
    }
    if let Some(cat_ids) = args.cat_ids {
        coco_eval.params.cat_ids = cat_ids;
    }
    if args.no_cats {
        coco_eval.params.use_cats = false;
    }

    let pb = spinner(&format!("Evaluating {}...", args.iou_type));
    let start = Instant::now();
    coco_eval.evaluate()?;
    coco_eval.accumulate();
    pb.finish_and_clear();
    status("Evaluated", &format!("{}", args.iou_type), start.elapsed());

    let _ = writeln!(stderr());
    coco_eval.summarize();

    if let Some(stats) = coco_eval.stats() {
        print_stats(stats);
    }

    if let Some(ref output_path) = args.output {
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

/// Load `what` from `path` behind a spinner, reporting the time it took;
/// a failure names the file.
fn timed_load<T>(
    what: &str,
    path: &Path,
    load: impl FnOnce() -> hotcoco::error::Result<T>,
) -> Result<T, String> {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let pb = spinner(&format!("Loading {what} {name}..."));
    let start = Instant::now();
    let value = load().map_err(|e| {
        pb.finish_and_clear();
        format!("failed to load {what} {}: {e}", path.display())
    })?;
    pb.finish_and_clear();
    status(
        "Loaded",
        &format!("{what} {DIM}{name}{RESET}"),
        start.elapsed(),
    );
    Ok(value)
}

/// The machine-readable `stats:` line the parity scripts read.
fn print_stats(stats: &[f64]) {
    let stats_strs: Vec<String> = stats.iter().map(|v| format!("{v:.15}")).collect();
    println!("stats: [{}]", stats_strs.join(", "));
}

fn run_panoptic(args: PanopticArgs) -> Result<(), Box<dyn std::error::Error>> {
    let load = |path: &Path, folder: Option<PathBuf>, what: &str| {
        let dataset = timed_load(what, path, || PanopticDataset::from_file(path))?;
        Ok::<_, String>(match folder {
            Some(folder) => dataset.with_folder(folder),
            None => dataset,
        })
    };
    let gt = load(&args.gt, args.gt_folder, "ground truth")?;
    let pred = load(&args.pred, args.pred_folder, "predictions")?;

    let mut ev = PanopticEval::new(gt, pred);
    let pb = spinner("Evaluating panoptic quality...");
    let start = Instant::now();
    let outcome = ev.run();
    pb.finish_and_clear();
    outcome?;
    status("Evaluated", "panoptic quality", start.elapsed());

    let _ = writeln!(stderr());
    ev.summarize();
    if let Some(result) = ev.result() {
        print_stats(&result.stats());
    }

    if let Some(ref output_path) = args.output {
        let start = Instant::now();
        std::fs::write(output_path, ev.report()?.to_json()?)?;
        status(
            "Saved",
            &format!("report to {DIM}{}{RESET}", output_path.display()),
            start.elapsed(),
        );
    }
    Ok(())
}
