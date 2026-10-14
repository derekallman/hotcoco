use std::fmt::Display;
use std::io;

use crate::convert::ConvertError;

/// Unified error type for hotcoco operations.
///
/// Each variant is a kind of failure, and the kind decides the Python
/// exception, so the same bad input raises the same exception whichever path
/// it took. There is no catch-all variant and no conversion from a bare
/// string: a new error has to say which kind it is.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error (file open, read, write). Errors from opening a named file
    /// carry the path in their message and keep their [`io::ErrorKind`], so a
    /// missing file is still `NotFound`.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// JSON parse or serialization error: loading (`COCO::new`,
    /// `COCO::load_res`), saving, and report emission.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Format conversion error (COCO ↔ YOLO).
    #[error(transparent)]
    Convert(#[from] ConvertError),

    /// Annotation ids handed to [`COCO::update_anns`](crate::COCO::update_anns)
    /// that the dataset does not have. Its own variant so a binding can map
    /// exactly this failure to a lookup error — Python's `KeyError`.
    #[error("annotation id(s) not in this dataset: {0:?}")]
    UnknownAnnIds(Vec<u64>),

    /// Category ids in a batch handed to
    /// [`StreamingEval::update`](crate::StreamingEval::update) that the
    /// evaluator was not constructed with. Its own variant for the same
    /// reason as `UnknownAnnIds`.
    #[error(
        "category id(s) {0:?} are not in this StreamingEval's categories; pass every \
         category the run will see to `categories`"
    )]
    UnknownCategoryIds(Vec<u64>),

    /// The caller's data or arguments are malformed: a value out of range,
    /// sizes that do not match, a file that does not decode, two inputs that
    /// cannot be combined. Python's `ValueError`.
    #[error("{0}")]
    InvalidInput(String),

    /// A method was called before the step it depends on, such as
    /// `results()` before `run()`. Python's `RuntimeError`.
    #[error("{0}")]
    InvalidState(String),
}

impl Error {
    /// An I/O error with what was being done, and to which path, in front of
    /// its message. The [`io::ErrorKind`] is kept, so a binding can still tell
    /// a missing file from a permission problem.
    pub fn io(e: io::Error, context: impl Display) -> Self {
        Error::Io(io::Error::new(e.kind(), format!("{context}: {e}")))
    }
}

/// Convenience alias used throughout hotcoco.
pub type Result<T> = std::result::Result<T, Error>;
