//! Chord transcription (S5.1, ADR-12): beat-synchronous chroma averaging,
//! chord-template matching, HMM smoothing via Viterbi decoding, Harte-notation
//! rendering. Follows the classic Sheh & Ellis 2003 / Bello & Pickens 2005
//! approach.
//!
//! No neural model, by deliberate choice (ADR-12, RISK-2): no chord model
//! currently has unambiguously permissive weights, and Q1 (license
//! cleanliness) outranks accuracy in this project's quality tree. This
//! yields roughly 75-80% majmin accuracy rather than a neural system's
//! 90%+ -- an accepted tradeoff, not an oversight.
//!
//! Chords are stored against **beat indices**, not times (S8.3, ADR-4): a
//! hand-corrected beat grid re-times every `ChordEvent` for free, with no
//! recomputation. Rendering to a time-based format (`.lab`) is a caller's
//! job (`audan-cli`, via `beats.beat_time(event.start_beat)`), not this
//! crate's -- this crate deliberately does not depend on `audan-format`.

mod aggregate;
mod harte;
mod hmm;
mod templates;

use serde::{Deserialize, Serialize};

use audan_core::{BeatGrid, BeatIndex, Chroma, MonoSignal, Source, PITCH_CLASSES};
use audan_dsp::ChordChromaParams;

use aggregate::beat_synchronous_chroma;
use hmm::{persistence_transition_log, viterbi};
use templates::{all_templates, cosine_similarity, Template};

pub use harte::NO_CHORD;
pub use templates::ChordQuality;

/// Softmax temperature over template cosine-similarity scores when turning
/// them into emission log-probabilities. Smaller sharpens the distribution
/// (the winning template dominates the emission signal); larger flattens it
/// (transitions/persistence dominate more). Chosen empirically so a clearly
/// C-major-shaped beat decisively outscores every other template while
/// still leaving enough headroom for the transition matrix to overrule a
/// single ambiguous beat.
const EMISSION_TEMPERATURE: f32 = 0.15;

/// The fixed pseudo-similarity assigned to the "no chord" (`N`) state,
/// competing directly against the 24 chord templates' cosine similarities in
/// the same softmax. A beat whose best chord-template similarity is below
/// this value loses to `N`; one above it wins. This makes `N` a first-class
/// state in the HMM (so persistence smoothing applies to it too) rather than
/// a post-hoc threshold bolted on after decoding.
const NO_CHORD_SIMILARITY: f32 = 0.35;

/// Self-transition probability in the Viterbi transition matrix: the
/// probability of the chord *not* changing from one beat to the next.
/// Chords typically hold for several beats, and per-beat template-matching
/// scores are noisy enough to flicker without this bias -- this constant is
/// the entire mechanism by which the HMM stage improves on naive per-beat
/// argmax (see `hmm::persistence_transition_log`). 0.9 sits in the range
/// used in the beat-synchronous chord-HMM literature; a real corpus would
/// learn it, but ADR-12 already accepts hand-picked parameters over a
/// trained model.
const SELF_TRANSITION_PROB: f32 = 0.9;

/// One decoded chord spanning a half-open beat interval `[start_beat,
/// end_beat)`. Storing beat indices rather than times is the point (ADR-4):
/// a corrected beat grid re-times this for free.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ChordEvent {
    pub start_beat: BeatIndex,
    pub end_beat: BeatIndex,
    /// Harte notation: `C:maj`, `A:min`, or `N` for no chord.
    pub chord: String,
    pub confidence: f32,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ChordSequence {
    pub schema_version: u32,
    pub chords: Vec<ChordEvent>,
    pub source: Source,
}

impl ChordSequence {
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;
}

/// Turns one beat's aggregated 12-dim chroma vector into a log-probability
/// over the 24 chord-template states plus the `N` state (index
/// `templates.len()`), via a softmax over cosine-similarity scores. Softmax
/// keeps the emission log-probabilities on the same scale as the transition
/// log-probabilities (both are log-probabilities of a normalized
/// distribution), which is what lets a fixed persistence constant compete
/// meaningfully against emission evidence in the Viterbi recursion.
fn emission_log_probs(vector: &[f32; PITCH_CLASSES], templates: &[Template]) -> Vec<f32> {
    let mut scores: Vec<f32> = templates
        .iter()
        .map(|t| cosine_similarity(vector, &t.vector))
        .collect();
    scores.push(NO_CHORD_SIMILARITY);

    let scaled: Vec<f32> = scores.iter().map(|s| s / EMISSION_TEMPERATURE).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| (e / sum).ln()).collect()
}

fn label_for_state(state: usize, templates: &[Template]) -> String {
    match templates.get(state) {
        Some(t) => harte::render(t.root, t.quality),
        None => NO_CHORD.to_string(),
    }
}

/// Estimates a beat-indexed chord sequence from a chroma sequence (as
/// produced by `audan_dsp::chord_chroma`) and a beat grid.
///
/// Pipeline: average chroma into one vector per beat, score every majmin
/// template plus `N` by cosine similarity, convert to emission
/// log-probabilities, and decode the maximum-likelihood chord sequence with
/// a persistence-favouring Viterbi HMM. Adjacent beats decoding to the same
/// chord are merged into a single `ChordEvent`.
pub fn estimate_chords(chroma: &Chroma, beats: &BeatGrid) -> ChordSequence {
    let aggregated = beat_synchronous_chroma(chroma, beats);
    let templates = all_templates();
    let n_states = templates.len() + 1; // + N

    let emission_log: Vec<Vec<f32>> = aggregated
        .iter()
        .map(|v| emission_log_probs(v, &templates))
        .collect();
    let log_trans = persistence_transition_log(n_states, SELF_TRANSITION_PROB);
    let log_prior = vec![(1.0 / n_states as f32).ln(); n_states];

    let path = viterbi(&emission_log, &log_trans, &log_prior);

    let mut chords = Vec::new();
    let mut i = 0;
    while i < path.len() {
        let mut j = i + 1;
        while j < path.len() && path[j] == path[i] {
            j += 1;
        }
        let confidence: f32 = (i..j)
            .map(|beat| emission_log[beat][path[beat]].exp())
            .sum::<f32>()
            / (j - i) as f32;
        chords.push(ChordEvent {
            start_beat: i,
            end_beat: j,
            chord: label_for_state(path[i], &templates),
            confidence,
        });
        i = j;
    }

    ChordSequence {
        schema_version: ChordSequence::CURRENT_SCHEMA_VERSION,
        chords,
        source: Source {
            algo: "template_hmm".into(),
            version: "1.0".into(),
            postproc: "viterbi".into(),
        },
    }
}

/// Convenience chaining entry point mirroring `audan-key`'s pattern:
/// chord-parameterized chroma (`audan_dsp::chord_chroma`) straight through
/// to a `ChordSequence`, for callers that have a `MonoSignal` (the L1
/// canonical analysis signal) rather than an already-computed `Chroma`.
/// Decoding audio into a `MonoSignal` is `audan-io`'s job, not this crate's
/// (S5.1's dependency rule), so this starts one layer above that.
pub fn estimate_chords_from_signal(
    signal: &MonoSignal,
    beats: &BeatGrid,
    params: &ChordChromaParams,
) -> ChordSequence {
    let chroma = audan_dsp::chord_chroma(signal, params);
    estimate_chords(&chroma, beats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::beat::{Meter, TempoInfo, TempoStability, TempoStabilityClass};
    use audan_core::candidate::{Candidate, Ranked};
    use audan_core::frame::{FrameGrid, PadMode};

    fn beat_grid(beats: Vec<f64>) -> BeatGrid {
        let n = beats.len();
        let duration_seconds = beats.last().copied().unwrap_or(0.0) + 1.0;
        BeatGrid {
            schema_version: BeatGrid::CURRENT_SCHEMA_VERSION,
            duration_seconds,
            confidence: vec![1.0; n],
            beats,
            downbeats: vec![0],
            meter: Meter {
                beats_per_bar: 4,
                confidence: 1.0,
            },
            tempo: TempoInfo {
                median_bpm: 60.0,
                candidates: Ranked::new(vec![Candidate::new(60.0, 1.0)]).unwrap(),
                stability: TempoStability {
                    ibi_mad_ms: 0.0,
                    drift_bpm_per_min: 0.0,
                    class: TempoStabilityClass::Programmed,
                },
            },
            frames: FrameGrid::new(1, 1, 1, PadMode::Zero).into(),
            source: Source {
                algo: "test".into(),
                version: "0".into(),
                postproc: "none".into(),
            },
        }
    }

    fn chord_vec(pcs: &[usize]) -> [f32; PITCH_CLASSES] {
        let mut v = [0f32; PITCH_CLASSES];
        for &pc in pcs {
            v[pc] = 1.0;
        }
        v
    }

    /// One chroma frame per beat, aligned so frame `k`'s window-centre time
    /// (via `FrameGrid::new(1, 1, 1, PadMode::Zero)`, `time_of(k) == k`)
    /// falls inside beat `k`'s `[k, k+1)` interval.
    fn one_frame_per_beat_chroma(frames: Vec<[f32; PITCH_CLASSES]>) -> Chroma {
        Chroma::from_frames(FrameGrid::new(1, 1, 1, PadMode::Zero), frames)
    }

    #[test]
    fn recovers_two_chords_and_does_not_fragment_on_a_noisy_transition_beat() {
        let c_major = chord_vec(&[0, 4, 7]);
        let g_major = chord_vec(&[7, 11, 2]);
        let noisy: [f32; PITCH_CLASSES] = {
            let mut v = [0f32; PITCH_CLASSES];
            for pc in 0..PITCH_CLASSES {
                v[pc] = (c_major[pc] + g_major[pc]) / 2.0;
            }
            v
        };

        let frames = vec![
            c_major, c_major, c_major, noisy, g_major, g_major, g_major, g_major,
        ];
        let chroma = one_frame_per_beat_chroma(frames);
        let beats = beat_grid((0..8).map(|i| i as f64).collect());

        let seq = estimate_chords(&chroma, &beats);

        assert_eq!(
            seq.chords.len(),
            2,
            "expected exactly two chord events, got {:?}",
            seq.chords
        );
        assert_eq!(seq.chords[0].chord, "C:maj");
        assert_eq!(seq.chords[1].chord, "G:maj");
        assert_eq!(seq.chords[0].start_beat, 0);
        assert_eq!(seq.chords[1].end_beat, 8);
        // The noisy beat (index 3) must land on one side or the other, not
        // spawn a third event.
        assert_eq!(seq.chords[0].end_beat, seq.chords[1].start_beat);
    }

    #[test]
    fn silence_yields_no_chord_not_a_spurious_label() {
        let frames = vec![[0f32; PITCH_CLASSES]; 4];
        let chroma = one_frame_per_beat_chroma(frames);
        let beats = beat_grid((0..4).map(|i| i as f64).collect());

        let seq = estimate_chords(&chroma, &beats);

        assert_eq!(seq.chords.len(), 1);
        assert_eq!(seq.chords[0].chord, NO_CHORD);
        assert_eq!(seq.chords[0].start_beat, 0);
        assert_eq!(seq.chords[0].end_beat, 4);
    }

    #[test]
    fn empty_beat_grid_yields_empty_sequence() {
        let chroma = one_frame_per_beat_chroma(vec![]);
        let beats = beat_grid(vec![]);
        let seq = estimate_chords(&chroma, &beats);
        assert!(seq.chords.is_empty());
    }
}
