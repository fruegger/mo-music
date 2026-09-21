//! Structural segmentation: self-similarity matrix, Foote novelty detection,
//! and beat-indexed section boundaries (architecture doc S5.1; glossary
//! "Foote novelty" / "SSM").
//!
//! Per RISK-8, section *labelling* ("this is the chorus") is out of scope for
//! this pass: [`SectionEvent::label`] is always `None`. Unlabeled boundary
//! detection is the real, tested deliverable here.
//!
//! Feature basis: [`audan_dsp::key_chroma`] (long hop, heavily
//! temporally-smoothed) rather than [`audan_dsp::chord_chroma`] (short hop,
//! harmonic-suppressed). Structural similarity wants a feature that's stable
//! *within* a section and contrasts clearly *across* sections; the
//! beat/chord-rate chroma is tuned for the opposite property (tracking
//! harmonic change quickly), which would make the SSM noisy along the
//! diagonal within a single section and wash out the block structure Foote
//! novelty depends on.

pub mod novelty;
pub mod segment;

pub use novelty::novelty_curve;
pub use segment::{segment_structure, SectionEvent, SegmentParams, StructureResult};
