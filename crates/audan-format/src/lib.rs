//! Writers (and, where S8.7 requires round-trip tests, readers) for the interchange
//! formats named in ADR-10 and the output table in S3.2: `.lab`, Audacity label
//! tracks, MIREX time lists, JAMS, MIDI tempo tracks, and click-track WAVs.
//!
//! Every function here operates on generic, format-shaped data ([`LabelInterval`],
//! `(f64, f64)` tempo points, plain `f64` times, [`JamsAnnotation`]) rather than on
//! any stage crate's domain type (a chord, a section label, a beat grid entry).
//! `audan-format` sits beside `audan-core` as a dependency of every layer above it
//! (S5.1); it must never depend on `audan-chords`, `audan-key`, `audan-struct`,
//! `audan-beats`, or `audan-stems`, or the dependency graph inverts.

mod click;
mod jams;
mod labels;
mod midi;
mod mirex;

pub use click::write_click_track;
pub use jams::{write_jams, JamsAnnotation, JamsObservation};
pub use labels::{read_audacity_labels, read_lab, write_audacity_labels, write_lab, LabelInterval};
pub use midi::{read_midi_tempo_track, write_midi_tempo_track};
pub use mirex::{read_mirex_times, write_mirex_times};

pub use audan_core::{AudanError, Result};

/// Thin convenience wrapper around `serde_json::to_string_pretty`. Most callers can
/// just call `serde_json::to_string` directly on their own `Serialize` type (e.g.
/// `BeatGrid` already derives it); this exists only for callers that want one
/// obvious entry point for "the own-JSON output" mentioned in S3.2.
pub fn write_json<T: serde::Serialize>(value: &T) -> serde_json::Result<String> {
    serde_json::to_string_pretty(value)
}
