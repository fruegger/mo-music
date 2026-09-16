//! The pluggable inference backend contract (S5.3 `BeatModel`, S8.6, ADR-6).
//!
//! `audan-architecture-arc42.md` calls for Beat This! (ISMIR 2024) via ONNX
//! as the primary backend, with a small default model embedded via
//! `include_bytes!` so a bare `audan beats` works offline (ADR-7, QS8/QS9).
//! **There is no real trained Beat This! ONNX model available in this
//! environment** -- it cannot be trained or sourced here. Rather than fake a
//! forward pass against weights that do not exist, this module defines the
//! real trait a real ONNX backend would implement (kept feature-gated and
//! genuinely off by default, see [`OnnxBackend`]), and ships
//! [`OnsetFallbackBackend`] as the crate's actual, always-available,
//! tested default: a classical periodicity-based beat tracker built
//! entirely from the mel spectrogram already computed by [`crate::mel`].
//! This is real, implementable MIR technique (onset/flux-based beat
//! tracking predates neural trackers by decades) -- it is simply less
//! accurate than a trained model, which is the honest trade being made
//! here rather than the spirit-only claim of a Beat This! port that cannot
//! run.

use audan_core::{FrameTime, Result};

use crate::mel::MelSpectrogram;

/// Per-frame beat/downbeat activation curves, in roughly `[0, 1]`. The
/// common currency every [`InferenceBackend`] produces and every
/// post-processing stage consumes, regardless of what produced it.
pub struct BeatActivations {
    pub times: Vec<FrameTime>,
    pub beat: Vec<f32>,
    pub downbeat: Vec<f32>,
}

/// A pluggable beat-activation model. Backend choice is a runtime concern in
/// the architecture (S8.6: "Backend choice is a runtime flag, never a
/// compile-time assumption"); this trait is the seam a real ONNX (or any
/// other) model would implement without the rest of the crate caring.
pub trait InferenceBackend {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn run(&self, mel: &MelSpectrogram) -> Result<BeatActivations>;
}

/// The default, always-available backend: no model file, no network, no
/// feature flag. Derives a beat-activation curve as spectral flux computed
/// directly over the mel bands already sitting in `mel.frames` -- the same
/// "sum of positive frame-to-frame differences" idea `audan_dsp::onset`
/// uses over linear-frequency STFT bins, just computed here over mel bands
/// instead, because the caller already paid for the `MelSpectrogram` and
/// recomputing a second, separate STFT-based onset envelope from the raw
/// signal would duplicate that work for no benefit -- the two are
/// equivalent in spirit, mel-band flux is just the one that reuses what is
/// already in hand.
pub struct OnsetFallbackBackend;

impl InferenceBackend for OnsetFallbackBackend {
    fn name(&self) -> &str {
        "onset_fallback"
    }

    fn version(&self) -> &str {
        "0.1"
    }

    fn run(&self, mel: &MelSpectrogram) -> Result<BeatActivations> {
        let n = mel.n_frames();
        let mut beat = Vec::with_capacity(n);
        let mut prev: Option<&Vec<f32>> = None;
        for frame in &mel.frames {
            let flux = match prev {
                None => 0.0,
                Some(p) => frame
                    .iter()
                    .zip(p.iter())
                    .map(|(a, b)| (a - b).max(0.0))
                    .sum(),
            };
            beat.push(flux);
            prev = Some(frame);
        }

        let max = beat.iter().cloned().fold(0.0f32, f32::max);
        if max > 1e-9 {
            for v in beat.iter_mut() {
                *v /= max;
            }
        }

        // Deliberately coarse: this fallback has no notion of "first beat of
        // a bar" independent of "beat", so it reuses the same curve.
        // Discriminating downbeats from beats is exactly the kind of
        // structure a trained model would learn and this heuristic cannot
        // -- real downbeat detection here happens downstream, in
        // `MeterEstimator` + `PostProcessor::snap_downbeats`'s "every Nth
        // beat at the best-scoring phase" heuristic, not in this curve.
        let downbeat = beat.clone();

        Ok(BeatActivations {
            times: mel.times.clone(),
            beat,
            downbeat,
        })
    }
}

/// A thin skeleton for a real ONNX (Beat This!-shaped) backend, behind the
/// off-by-default `onnx-beats` feature. There are no real weights available
/// to validate a forward pass against in this environment, so `run` is a
/// clearly marked stub rather than a fabricated implementation -- the goal
/// here is the right shape (a `PathBuf` to a checksum-verified model from
/// `audan_model::ModelStore`, in through `InferenceBackend`, real
/// `BeatActivations` out) for whoever wires up real `rten` inference later,
/// not a working prediction today.
#[cfg(feature = "onnx-beats")]
pub struct OnnxBackend {
    model_path: std::path::PathBuf,
}

#[cfg(feature = "onnx-beats")]
impl OnnxBackend {
    pub fn new(model_path: std::path::PathBuf) -> Self {
        Self { model_path }
    }
}

#[cfg(feature = "onnx-beats")]
impl InferenceBackend for OnnxBackend {
    fn name(&self) -> &str {
        "beat_this_onnx"
    }

    fn version(&self) -> &str {
        "1.0"
    }

    fn run(&self, _mel: &MelSpectrogram) -> Result<BeatActivations> {
        // TODO: load `self.model_path` as an `rten::Model`, build an input
        // tensor from `mel.frames` (shape `[1, n_mels, n_frames]` or
        // whatever the real Beat This! ONNX export expects), run the
        // forward pass, and unpack its beat/downbeat activation outputs
        // into `BeatActivations`. Needs a real exported graph to test
        // against, which does not exist in this environment.
        Err(audan_core::AudanError::Model(format!(
            "onnx-beats backend is a compiling skeleton only; no inference implemented (model path: {})",
            self.model_path.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::MelFrontend;
    use audan_core::{FrameGrid, MonoSignal, PadMode};

    #[test]
    fn fallback_backend_produces_nonnegative_normalized_curves() {
        let sr = 22_050u32;
        let n = sr as usize * 2;
        let samples: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * 220.0 * i as f32 / sr as f32).sin())
            .collect();
        let signal = MonoSignal {
            sample_rate: sr,
            samples,
        };
        let mel = MelFrontend::compute(&signal, 40);

        let backend = OnsetFallbackBackend;
        let activations = backend.run(&mel).unwrap();

        assert_eq!(activations.beat.len(), mel.n_frames());
        assert_eq!(activations.downbeat.len(), mel.n_frames());
        assert!(activations
            .beat
            .iter()
            .all(|&v| v >= 0.0 && v <= 1.0 + 1e-6));
        assert_eq!(activations.beat, activations.downbeat);
    }

    #[test]
    fn fallback_backend_handles_empty_signal() {
        let grid = FrameGrid::new(22_050, 512, 2048, PadMode::Reflect);
        let _ = grid; // documents the grid this crate otherwise assumes elsewhere
        let signal = MonoSignal {
            sample_rate: 22_050,
            samples: vec![],
        };
        let mel = MelFrontend::compute(&signal, 16);
        let backend = OnsetFallbackBackend;
        let activations = backend.run(&mel).unwrap();
        assert!(activations.beat.is_empty());
    }
}
