//! Shared domain types for the `audan` workspace: [`frame::FrameTime`] /
//! [`frame::FrameGrid`] (the frame-time convention, S8.2), [`beat::BeatGrid`]
//! (the shared primitive, S8.3), [`chroma::Chroma`], [`candidate::Candidate`] /
//! [`candidate::Ranked`] (S8.5), [`signal`] (decoded-audio types), and the
//! error taxonomy (S2.3).
//!
//! No I/O, no DSP: this crate depends on nothing internal (S5.1) so that every
//! other crate in the workspace can depend on it without pulling in decode,
//! DSP, or cache machinery.

pub mod beat;
pub mod candidate;
pub mod chroma;
pub mod error;
pub mod frame;
pub mod signal;

pub use beat::{
    BeatGrid, BeatIndex, Meter, Source, TempoInfo, TempoStability, TempoStabilityClass,
};
pub use candidate::{Candidate, Confidence, EmptyCandidates, Ranked};
pub use chroma::{Chroma, PITCH_CLASSES};
pub use error::{AudanError, ExitCode, Result};
pub use frame::{FrameGrid, FrameTime, FramesMeta, PadMode, TimeRef};
pub use signal::{MonoSignal, Signal, ANALYSIS_SAMPLE_RATE, STEMS_SAMPLE_RATE};
