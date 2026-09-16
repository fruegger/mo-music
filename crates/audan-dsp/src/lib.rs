//! DSP features every analysis stage builds on: windowing, STFT, constant-Q,
//! chroma variants, onset strength envelope, tempogram, and self-similarity
//! matrices. No I/O: operates on `audan_core::MonoSignal` in, produces
//! `audan_core::Chroma` or this crate's own DSP types out (S5.1).
//!
//! Every frame timestamp in this crate comes from `FrameGrid::time_of`
//! (S8.2, ADR-5) -- there is no other route from a frame index to a time
//! anywhere in these modules.

pub mod chroma;
pub mod cqt;
pub mod onset;
pub mod ssm;
pub mod stft;
pub mod tempogram;
pub mod window;

pub use chroma::{chord_chroma, key_chroma, ChordChromaParams, KeyChromaParams};
pub use cqt::{Cqt, CqtParams};
pub use onset::{onset_envelope, pick_peaks, spectral_flux, OnsetEnvelope, Peak};
pub use ssm::{chroma_ssm, cosine_similarity_matrix};
pub use stft::Stft;
pub use tempogram::{tempogram, Tempogram, TempogramParams};
