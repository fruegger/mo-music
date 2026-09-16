//! The shared error taxonomy (S2.3): `thiserror` in libraries, `anyhow` only in
//! `audan-cli`. Every library-crate error should ultimately convert into an
//! [`AudanError`] so the CLI can map it to the exit codes in S8.9 without
//! re-deriving the classification at the top level.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Process exit codes, fixed by S8.9. `Success` is never actually returned by
/// `AudanError` (there is no error variant for success); it is listed here so
/// the mapping is total and documented in one place.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    RuntimeError = 1,
    UsageError = 2,
    LowConfidence = 3,
    UnsupportedFormat = 4,
}

/// The top-level error taxonomy shared by every `audan-*` library crate.
/// Stage-specific errors should implement `From<StageError> for AudanError`
/// rather than being matched directly in `audan-cli`.
#[derive(Error, Debug)]
pub enum AudanError {
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("cache error: {0}")]
    Cache(String),

    #[error("decode error: {0}")]
    Decode(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("invalid or foreign input rejected: {0}")]
    InvalidInput(String),

    #[error("usage error: {0}")]
    Usage(String),

    #[error("confidence {confidence:.2} below --strict threshold {threshold:.2}")]
    LowConfidence { confidence: f32, threshold: f32 },
}

impl AudanError {
    pub fn exit_code(&self) -> ExitCode {
        match self {
            AudanError::UnsupportedFormat(_) => ExitCode::UnsupportedFormat,
            AudanError::Usage(_) => ExitCode::UsageError,
            AudanError::LowConfidence { .. } => ExitCode::LowConfidence,
            AudanError::Io(_)
            | AudanError::Cache(_)
            | AudanError::Decode(_)
            | AudanError::Model(_)
            | AudanError::InvalidInput(_) => ExitCode::RuntimeError,
        }
    }
}

pub type Result<T> = std::result::Result<T, AudanError>;
