//! The beat grid: the shared primitive of the architecture (S8.3, ADR-4).
//!
//! Key, chords, and structure all quantise to beats, and downstream artefacts
//! reference beat *indices* rather than times, so a hand-corrected grid re-times
//! everything below it without recomputation.

use serde::{Deserialize, Serialize};

use crate::candidate::Ranked;
use crate::frame::FramesMeta;

/// A beat-grid index. Downstream artefacts (chords, sections) store these
/// instead of times (ADR-4), so correcting the grid re-times them for free.
pub type BeatIndex = usize;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum TempoStabilityClass {
    /// Inter-beat intervals are near-constant to the sample: a drum machine or
    /// DAW-programmed track.
    Programmed,
    /// Natural micro-timing variance consistent with a human performance.
    Human,
    /// A consistent linear trend across the track (tape slowdown, deliberate
    /// accelerando/ritardando).
    Drifting,
    /// Neither a small constant jitter nor a clean drift -- tempo changes
    /// structurally (e.g. a DJ mix, a tempo-change composition).
    Variable,
}

#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TempoStability {
    /// Median absolute deviation of inter-beat intervals, in milliseconds.
    /// Captures micro-timing (a drummer's human feel), independent of drift.
    pub ibi_mad_ms: f64,
    /// Linear trend across the tempo curve, in BPM per minute. Captures tape
    /// slowdown or a deliberate accelerando -- a different phenomenon from
    /// jitter, deliberately not collapsed into one "+/- 2 BPM" number.
    pub drift_bpm_per_min: f64,
    pub class: TempoStabilityClass,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TempoInfo {
    pub median_bpm: f64,
    /// Ranked tempo candidates. Octave ambiguity (87 vs 174 BPM) is genuine
    /// perceptual ambiguity, not a bug to fix (Q5, ADR-11): both appear here
    /// with their confidences, never silently collapsed to one.
    pub candidates: Ranked<f64>,
    pub stability: TempoStability,
}

#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Meter {
    pub beats_per_bar: u8,
    pub confidence: f32,
}

/// Provenance: which algorithm and post-processing variant produced this
/// artefact, and its version. Carried in every output so a timing discrepancy
/// against another tool, or against a re-run after an upgrade, is diagnosable.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Source {
    pub algo: String,
    pub version: String,
    pub postproc: String,
}

/// The beat grid: beats, downbeats, meter, tempo statistics, and per-beat
/// confidences. Serializes to exactly the schema shown in S8.3.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct BeatGrid {
    pub schema_version: u32,
    /// Beat times in original-signal seconds (window-centre convention; these
    /// are pre-flattened from `FrameTime` for the public schema).
    pub beats: Vec<f64>,
    /// Indices into `beats` marking the first beat of each bar.
    pub downbeats: Vec<BeatIndex>,
    pub meter: Meter,
    pub tempo: TempoInfo,
    /// Per-beat confidence, same length as `beats`.
    pub confidence: Vec<f32>,
    pub frames: FramesMeta,
    pub source: Source,
}

impl BeatGrid {
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    /// Resolves a candidate tempo's octave-doubled/halved BPM given a beat
    /// count, for callers that want to report a specific candidate rather than
    /// the median. Pure convenience; the median is already authoritative.
    pub fn top_tempo_bpm(&self) -> f64 {
        self.tempo.candidates.top().value
    }

    pub fn beat_time(&self, index: BeatIndex) -> Option<f64> {
        self.beats.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::Candidate;
    use crate::frame::{FrameGrid, PadMode};

    fn sample_grid() -> BeatGrid {
        BeatGrid {
            schema_version: BeatGrid::CURRENT_SCHEMA_VERSION,
            beats: vec![0.512, 0.973, 1.441],
            downbeats: vec![0],
            meter: Meter {
                beats_per_bar: 4,
                confidence: 0.91,
            },
            tempo: TempoInfo {
                median_bpm: 128.02,
                candidates: Ranked::new(vec![
                    Candidate::new(128.0, 0.88),
                    Candidate::new(64.0, 0.09),
                ])
                .unwrap(),
                stability: TempoStability {
                    ibi_mad_ms: 0.8,
                    drift_bpm_per_min: 0.0,
                    class: TempoStabilityClass::Programmed,
                },
            },
            confidence: vec![0.95, 0.93, 0.90],
            frames: FrameGrid::new(22050, 512, 2048, PadMode::Reflect).into(),
            source: Source {
                algo: "beat_this".into(),
                version: "1.0".into(),
                postproc: "minimal".into(),
            },
        }
    }

    #[test]
    fn round_trips_through_json() {
        let grid = sample_grid();
        let json = serde_json::to_string(&grid).unwrap();
        let back: BeatGrid = serde_json::from_str(&json).unwrap();
        assert_eq!(back.beats, grid.beats);
        assert_eq!(back.top_tempo_bpm(), 128.0);
    }
}
