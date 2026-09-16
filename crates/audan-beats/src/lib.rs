//! Beat tracking: mel frontend, pluggable inference backend, minimal
//! post-processing, tempo statistics, and meter estimation (S5.3 of
//! `audan-architecture-arc42.md`).
//!
//! ## The model situation, stated plainly
//!
//! The architecture calls for Beat This! (ISMIR 2024) via ONNX as the
//! primary backend, with a small default model embedded via
//! `include_bytes!` so a bare `audan beats` works offline with no network,
//! no Python, no toolchain (ADR-7, QS8, QS9). **No real trained Beat This!
//! ONNX model is available in this environment** -- it can be neither
//! trained nor sourced here. Building a backend around weights that do not
//! exist would mean either shipping something that silently does nothing or
//! faking numbers; neither is acceptable.
//!
//! So the crate is split cleanly along that line:
//!
//! - [`model::InferenceBackend`] is the real, general trait a genuine ONNX
//!   backend would implement -- see [`model::OnnxBackend`], a compiling
//!   skeleton behind the off-by-default `onnx-beats` feature with a clearly
//!   marked `TODO` where a real `rten` forward pass would go.
//! - [`model::OnsetFallbackBackend`] is the crate's actual, always-tested,
//!   zero-dependency default: a classical periodicity-based beat tracker
//!   built entirely from `audan-dsp`'s (already implemented, already
//!   tested) onset/tempogram machinery, applied over this crate's own mel
//!   spectrogram. Periodicity-based beat tracking is a real, decades-old
//!   MIR technique, not a stand-in pretending to be something else -- it is
//!   simply less accurate than a trained neural tracker. This is what makes
//!   [`track_beats_default`] actually work with no network and no model
//!   file, in the spirit of QS8/QS9 even without the specific Beat This!
//!   weights.
//!
//! ## Pipeline
//!
//! [`track_beats`] runs, in order: [`mel::MelFrontend::compute`] ->
//! `backend.run` -> [`postprocess::PostProcessor`] (beat detection, then
//! downbeat snapping once meter is known) -> [`tempo::TempoAnalyser`] ->
//! [`meter::MeterEstimator`] -> assembly into an `audan_core::BeatGrid`.

pub mod mel;
pub mod meter;
pub mod model;
pub mod postprocess;
pub mod tempo;

use audan_core::{
    AudanError, BeatGrid, FramesMeta, Meter, Result, Source, TempoInfo, TempoStability,
};

pub use mel::{MelFrontend, MelSpectrogram};
pub use meter::{MeterEstimator, MeterResult};
#[cfg(feature = "onnx-beats")]
pub use model::OnnxBackend;
pub use model::{BeatActivations, InferenceBackend, OnsetFallbackBackend};
pub use postprocess::{DetectedBeats, PostProcessor};
pub use tempo::{TempoAnalyser, TempoStats};

/// The number of mel bands fed to the backend. S5.3 specifies 128 for exact
/// parity with the Beat This! reference frontend; kept at that figure here
/// even though the default backend does not need a specific band count, so
/// a future real ONNX backend can be dropped in without a mismatch.
pub const DEFAULT_N_MELS: usize = 128;

/// Runs the full beat-tracking pipeline against a chosen backend.
pub fn track_beats(
    signal: &audan_core::MonoSignal,
    backend: &dyn InferenceBackend,
) -> Result<BeatGrid> {
    let mel = MelFrontend::compute(signal, DEFAULT_N_MELS);
    let activations = backend.run(&mel)?;

    // Tempo is estimated from the *whole* activation curve's periodicity
    // (a global tempogram search, robust to which sub-pulse happens to
    // peak-pick loudest) before any discrete beat is picked, and that
    // estimate then *informs* beat selection -- not the other way around.
    // Picking beats first via plain peak-picking and only asking "what
    // tempo does this imply" afterward (the previous pipeline order) can
    // never correct a beat sequence that locked onto the wrong pulse (e.g.
    // hi-hats at 2x the true tempo): by the time you're computing stats
    // from it, the wrong beats are already final.
    let postproc = PostProcessor::default();
    let tempo_analyser = TempoAnalyser::default();
    let candidates = tempo_analyser.tempo_candidates(&activations, mel.grid);
    let target_bpm = candidates.top().value;

    let detected = postproc.detect_beats_at_tempo(&activations, mel.grid, target_bpm);
    if detected.times.len() < 2 {
        return Err(AudanError::InvalidInput(
            "beat tracking found fewer than two beats; cannot derive tempo".into(),
        ));
    }

    // Stability (jitter/drift/class) and the headline `median_bpm` are
    // recomputed from the tempo-locked beats, so they describe the same
    // sequence the grid actually reports rather than an earlier, possibly
    // differently-paced candidate.
    let stats = tempo_analyser
        .analyse(&detected.times)
        .expect("at least two monotonically increasing beats were just confirmed above");

    let meter_estimator = MeterEstimator::default();
    let meter_result = meter_estimator.estimate(&detected.confidence);

    let downbeats = postproc.snap_downbeats(
        &detected.times,
        &activations,
        meter_result.beats_per_bar,
        mel.grid,
    );

    let frames: FramesMeta = mel.grid.into();

    Ok(BeatGrid {
        schema_version: BeatGrid::CURRENT_SCHEMA_VERSION,
        beats: detected.times,
        downbeats,
        meter: Meter {
            beats_per_bar: meter_result.beats_per_bar,
            confidence: meter_result.confidence,
        },
        tempo: TempoInfo {
            median_bpm: stats.median_bpm,
            candidates,
            stability: TempoStability {
                ibi_mad_ms: stats.ibi_mad_ms,
                drift_bpm_per_min: stats.drift_bpm_per_min,
                class: stats.class,
            },
        },
        confidence: detected.confidence,
        frames,
        source: Source {
            algo: backend.name().into(),
            version: backend.version().into(),
            postproc: "minimal".into(),
        },
    })
}

/// [`track_beats`] against the pure-DSP [`OnsetFallbackBackend`] -- the
/// path that must always work with no model, no network, and no feature
/// flag (see the module docs above).
pub fn track_beats_default(signal: &audan_core::MonoSignal) -> Result<BeatGrid> {
    track_beats(signal, &OnsetFallbackBackend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::{MonoSignal, TempoStabilityClass};

    /// Same construction as `audan-dsp`'s timing canary
    /// (`crates/audan-dsp/tests/timing_canary.rs`): a short Hann-shaped
    /// energy burst, not a bare impulse, so it behaves like a real
    /// transient under windowed analysis.
    fn add_click(signal: &mut [f32], center_sample: usize, win: usize, amplitude: f32) {
        let burst_len = ((win as f32 / 10.0).round() as usize).max(3) | 1;
        let half = burst_len / 2;
        let envelope = audan_dsp::window::hann(burst_len);
        for i in 0..burst_len {
            let offset = i as i64 - half as i64;
            let idx = center_sample as i64 + offset;
            if idx < 0 || idx as usize >= signal.len() {
                continue;
            }
            signal[idx as usize] += amplitude * envelope[i];
        }
    }

    /// A perfectly regular click train at exactly 120 BPM (clicks every
    /// 0.5s), matching the timing canary's construction style.
    fn synth_120bpm_click_train(sample_rate: u32, seconds: f64) -> Vec<f32> {
        let n = (sample_rate as f64 * seconds) as usize;
        let mut signal = vec![0.0f32; n];
        let win = 2048; // matches MelFrontend's fixed STFT window
        let mut t = 0.2; // small lead-in so early padding does not clip the first click
        while t < seconds - 0.2 {
            let center = (t * sample_rate as f64) as usize;
            add_click(&mut signal, center, win, 1.0);
            t += 0.5;
        }
        signal
    }

    #[test]
    fn default_fallback_recovers_120_bpm_click_train() {
        let sr = 22_050u32;
        let samples = synth_120bpm_click_train(sr, 16.0);
        let signal = MonoSignal {
            sample_rate: sr,
            samples,
        };

        let grid = track_beats_default(&signal).expect("click train must yield a beat grid");

        assert!(
            (grid.tempo.median_bpm - 120.0).abs() < 6.0,
            "expected ~120 BPM, got {}",
            grid.tempo.median_bpm
        );
        assert_eq!(grid.tempo.stability.class, TempoStabilityClass::Programmed);
        assert!(
            grid.beats.len() >= 20,
            "expected most of the ~30 clicks to be found, got {}",
            grid.beats.len()
        );

        // Detected beats should land close to the true 0.5s-spaced click
        // times (generous tolerance: this is a heuristic DSP tracker on a
        // burst, not a sample-accurate onset detector).
        // Clicks are at 0.2, 0.7, 1.2, ... (0.5s spacing with a 0.2s lead-in).
        for &t in &grid.beats {
            let nearest_click = ((t - 0.2) / 0.5).round() * 0.5 + 0.2;
            assert!(
                (t - nearest_click).abs() < 0.05,
                "beat at {t}s is not close to any 0.5s-spaced click time (nearest {nearest_click}s)"
            );
        }

        assert_eq!(grid.confidence.len(), grid.beats.len());
        assert!(grid.tempo.candidates.len() >= 1);
        assert_eq!(grid.frames.sr, sr);
        assert_eq!(grid.source.algo, "onset_fallback");
    }

    #[test]
    fn too_short_signal_is_a_clean_error_not_a_panic() {
        let signal = MonoSignal {
            sample_rate: 22_050,
            samples: vec![0.0; 100],
        };
        let result = track_beats_default(&signal);
        assert!(result.is_err());
    }
}
