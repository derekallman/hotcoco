use std::io;

use crate::convert::ConvertError;

/// Unified error type for hotcoco operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error (file open, read, write).
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

    /// Any other error with a human-readable message.
    #[error("{0}")]
    Other(String),
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Other(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Other(s.to_string())
    }
}

/// Convenience alias used throughout hotcoco.
pub type Result<T> = std::result::Result<T, Error>;
