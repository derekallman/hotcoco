//! A benchmark laid out as TrackEval expects it: the sequences, their
//! `seqinfo.ini` frame counts, and the MOTChallenge text files of each side.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// The files of one sequence.
pub(super) struct SequenceFiles {
    pub(super) name: String,
    pub(super) gt: PathBuf,
    pub(super) seqinfo: PathBuf,
    pub(super) pred: PathBuf,
}

/// A text file's rows, row-major.
pub(super) struct Rows {
    pub(super) values: Vec<f64>,
    pub(super) ncols: usize,
}

/// The sequences to evaluate, in name order: those `seqmap` names, or every
/// folder of `gt_folder` whose name does not start with a dot.
pub(super) fn sequences(
    gt_folder: &Path,
    pred_folder: &Path,
    seqmap: Option<&Path>,
) -> Result<Vec<SequenceFiles>> {
    let mut names = match seqmap {
        Some(path) => read_seqmap(path)?,
        None => sequence_folders(gt_folder)?,
    };
    names.sort();
    if let Some([name, _]) = names.windows(2).find(|w| w[0] == w[1]) {
        return Err(Error::InvalidInput(format!(
            "sequence {name} is named twice in the seqmap"
        )));
    }
    if names.is_empty() {
        return Err(Error::InvalidInput(match seqmap {
            Some(path) => format!("seqmap {} names no sequence", path.display()),
            None => format!("no sequence folders in {}", gt_folder.display()),
        }));
    }
    Ok(names
        .into_iter()
        .map(|name| {
            let dir = gt_folder.join(&name);
            SequenceFiles {
                gt: dir.join("gt").join("gt.txt"),
                seqinfo: dir.join("seqinfo.ini"),
                pred: pred_folder.join(format!("{name}.txt")),
                name,
            }
        })
        .collect())
}

/// The folders directly under `gt_folder`, hidden ones left out.
fn sequence_folders(gt_folder: &Path) -> Result<Vec<String>> {
    let entries = std::fs::read_dir(gt_folder).map_err(|e| {
        Error::io(
            e,
            format_args!("cannot read ground-truth folder {}", gt_folder.display()),
        )
    })?;
    let mut names = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with('.') && entry.path().is_dir() {
            names.push(name);
        }
    }
    Ok(names)
}

/// A TrackEval seqmap: a header line, then a sequence name first on each
/// line; lines with no name are skipped.
fn read_seqmap(path: &Path) -> Result<Vec<String>> {
    let text = read(path, "seqmap")?;
    Ok(text
        .lines()
        .skip(1)
        .filter_map(|line| {
            let name = line.split(',').next().unwrap_or("").trim();
            (!name.is_empty()).then(|| name.to_owned())
        })
        .collect())
}

/// The `seqLength` of a `seqinfo.ini`'s `[Sequence]` section.
pub(super) fn seq_length(path: &Path) -> Result<usize> {
    let text = read(path, "seqinfo")?;
    let mut in_sequence = false;
    for line in text.lines() {
        let line = line.trim();
        if let Some(section) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            in_sequence = section.trim() == "Sequence";
            continue;
        }
        let Some((key, value)) = line.split_once(['=', ':']) else {
            continue;
        };
        if in_sequence && key.trim().eq_ignore_ascii_case("seqLength") {
            return value.trim().parse().map_err(|_| {
                Error::InvalidInput(format!(
                    "{}: seqLength {:?} is not a whole number",
                    path.display(),
                    value.trim()
                ))
            });
        }
    }
    Err(Error::InvalidInput(format!(
        "{}: no seqLength in a [Sequence] section",
        path.display()
    )))
}

/// The rows of a MOTChallenge text file: comma-separated, or
/// whitespace-separated on a line with no comma, one empty field after a
/// trailing comma dropped, blank lines skipped. Every row has as many fields
/// as the first.
pub(super) fn read_rows_file(path: &Path, what: &str) -> Result<Rows> {
    let text = read(path, what)?;
    let mut values = Vec::new();
    let mut ncols = 0;
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields: Vec<&str> = if line.contains(',') {
            line.split(',').map(str::trim).collect()
        } else {
            line.split_whitespace().collect()
        };
        if fields.last() == Some(&"") {
            fields.pop();
        }
        if ncols == 0 {
            ncols = fields.len();
        } else if fields.len() != ncols {
            return Err(Error::InvalidInput(format!(
                "{} line {}: {} fields, where the first row has {ncols}",
                path.display(),
                n + 1,
                fields.len()
            )));
        }
        for field in fields {
            values.push(field.parse::<f64>().map_err(|_| {
                Error::InvalidInput(format!(
                    "{} line {}: {field:?} is not a number",
                    path.display(),
                    n + 1
                ))
            })?);
        }
    }
    Ok(Rows { values, ncols })
}

/// `path` as text. Bytes that are not UTF-8 are bad content, a `ValueError`
/// as a line that is not numbers is, and as the converters and the JSON
/// loader raise them; only a filesystem failure is an I/O error.
fn read(path: &Path, what: &str) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::InvalidData {
            return Error::InvalidInput(format!("{what} file {}: {e}", path.display()));
        }
        Error::io(
            e,
            format_args!("cannot read {what} file {}", path.display()),
        )
    })
}
