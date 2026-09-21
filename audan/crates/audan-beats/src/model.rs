//! The pluggable inference backend contract (S5.3 `BeatModel`, S8.6, ADR-6).
//!
//! Two backends implement [`InferenceBackend`]: [`OnsetFallbackBackend`], the
//! crate's always-available, model-free default (a classical
//! periodicity-based beat tracker -- real, implementable MIR technique that
//! predates neural trackers by decades, simply less accurate than a trained
//! model), and [`OnnxBackend`] (behind the `onnx-beats` feature), which runs
//! the real Beat This! (ISMIR 2024, CPJKU/beat_this, MIT-licensed code and
//! weights) model via `rten` -- see `resources/models/beat_this/README.md`
//! for how that model is produced.
//!
//! The trait takes the raw analysis-rate [`MonoSignal`], not a pre-computed
//! spectrogram: the two backends need genuinely different preprocessing
//! (different mel scale, hop/window size, frequency range, log-compression
//! constant -- see [`crate::onnx_frontend`]'s module docs), so each backend
//! owns its complete frontend-to-activations pipeline rather than being
//! handed a spectrogram computed for a different backend's contract.

use audan_core::{FrameGrid, MonoSignal, PadMode, Result};

use crate::mel::MelFrontend;

/// Per-frame beat/downbeat activation curves, in roughly `[0, 1]`. The
/// common currency every [`InferenceBackend`] produces and every
/// post-processing stage consumes, regardless of what produced it.
pub struct BeatActivations {
    pub times: Vec<audan_core::FrameTime>,
    pub beat: Vec<f32>,
    pub downbeat: Vec<f32>,
    /// The frame geometry (hop/win/sample_rate/pad) `times` was derived
    /// under -- needed downstream by [`crate::tempo::TempoAnalyser`] and
    /// [`crate::postprocess::PostProcessor`], which convert between the
    /// frame and time/BPM domains. Each backend uses its own frontend's
    /// grid, since [`OnsetFallbackBackend`] and [`OnnxBackend`] run
    /// different hop/window STFTs.
    pub grid: audan_core::FrameGrid,
}

/// A pluggable beat-activation model. Backend choice is a runtime concern in
/// the architecture (S8.6: "Backend choice is a runtime flag, never a
/// compile-time assumption"); this trait is the seam a real ONNX (or any
/// other) model would implement without the rest of the crate caring.
pub trait InferenceBackend {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    /// The frame geometry this backend's frontend produces at `sample_rate`
    /// -- used to validate a hand-corrected `--beats grid.json` (RV5)
    /// against the right shape for *this* backend, without having to run
    /// the frontend just to find out.
    fn frame_grid(&self, sample_rate: u32) -> FrameGrid;
    fn run(&self, signal: &MonoSignal) -> Result<BeatActivations>;
}

/// The default, always-available backend: no model file, no network, no
/// feature flag. Derives a beat-activation curve as spectral flux computed
/// directly over its own mel spectrogram's bands -- the same "sum of
/// positive frame-to-frame differences" idea `audan_dsp::onset` uses over
/// linear-frequency STFT bins, just computed here over mel bands instead.
pub struct OnsetFallbackBackend;

impl InferenceBackend for OnsetFallbackBackend {
    fn name(&self) -> &str {
        "onset_fallback"
    }

    fn version(&self) -> &str {
        "0.1"
    }

    fn frame_grid(&self, sample_rate: u32) -> FrameGrid {
        FrameGrid::new(sample_rate, 512, 2048, PadMode::Reflect)
    }

    fn run(&self, signal: &MonoSignal) -> Result<BeatActivations> {
        let mel = MelFrontend::compute(signal, crate::DEFAULT_N_MELS);
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
            grid: mel.grid,
        })
    }
}

/// The real Beat This! backend: an `rten` session loaded once at
/// construction, run once per fixed-[`crate::chunk::CHUNK_SIZE`]-frame chunk
/// (see [`crate::chunk`]) over the [`crate::onnx_frontend`]'s Slaney mel
/// frontend.
#[cfg(feature = "onnx-beats")]
pub struct OnnxBackend {
    model: rten::Model,
    version: String,
}

#[cfg(feature = "onnx-beats")]
impl OnnxBackend {
    /// Loads the model once so `run()` only pays inference cost, not load
    /// cost, per call. `version` is carried through to `BeatGrid.source` for
    /// cache-key and provenance purposes (S8.4) -- typically the resolved
    /// `ModelEntry.version` from the registry.
    pub fn new(model_path: &std::path::Path, version: impl Into<String>) -> Result<Self> {
        let model = rten::Model::load_file(model_path).map_err(|e| {
            audan_core::AudanError::Model(format!(
                "failed to load onnx-beats model {}: {e}",
                model_path.display()
            ))
        })?;
        Ok(Self {
            model,
            version: version.into(),
        })
    }

    fn run_chunk(&self, chunk: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<f32>)> {
        use rten_tensor::{AsView, NdTensor};

        let n_mels = chunk.first().map(|f| f.len()).unwrap_or(0);
        let flat: Vec<f32> = chunk.iter().flat_map(|frame| frame.iter().copied()).collect();
        let input: NdTensor<f32, 3> =
            NdTensor::from_data([1, chunk.len(), n_mels], flat);

        let output = self.model.run_one(input.into(), None).map_err(|e| {
            audan_core::AudanError::Model(format!("onnx-beats inference failed: {e}"))
        })?;
        let output: NdTensor<f32, 3> = output.try_into().map_err(|_| {
            audan_core::AudanError::Model(
                "onnx-beats model returned an unexpected output shape".into(),
            )
        })?;

        // Exported as `torch.stack([beat, downbeat], dim=0)`, shape
        // (2, 1, chunk_size); flattened C-order iteration therefore yields
        // the whole beat channel followed by the whole downbeat channel.
        let flat_out: Vec<f32> = output.iter().copied().collect();
        let chunk_size = chunk.len();
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let beat: Vec<f32> = flat_out[0..chunk_size].iter().copied().map(sigmoid).collect();
        let downbeat: Vec<f32> = flat_out[chunk_size..2 * chunk_size]
            .iter()
            .copied()
            .map(sigmoid)
            .collect();
        Ok((beat, downbeat))
    }
}

#[cfg(feature = "onnx-beats")]
impl InferenceBackend for OnnxBackend {
    fn name(&self) -> &str {
        "beat_this_onnx"
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn frame_grid(&self, sample_rate: u32) -> FrameGrid {
        use crate::onnx_frontend::{HOP, N_FFT};
        FrameGrid::new(sample_rate, HOP, N_FFT, PadMode::Reflect)
    }

    fn run(&self, signal: &MonoSignal) -> Result<BeatActivations> {
        let mel = crate::onnx_frontend::OnnxFrontend::compute(signal);
        let len = mel.frames.len();
        let starts = crate::chunk::chunk_starts(len);

        let mut beat_chunks = Vec::with_capacity(starts.len());
        let mut downbeat_chunks = Vec::with_capacity(starts.len());
        for &start in &starts {
            let chunk = crate::chunk::build_chunk(&mel.frames, start);
            let (beat, downbeat) = self.run_chunk(&chunk)?;
            beat_chunks.push(beat);
            downbeat_chunks.push(downbeat);
        }

        let beat = crate::chunk::aggregate(&starts, &beat_chunks, len);
        let downbeat = crate::chunk::aggregate(&starts, &downbeat_chunks, len);

        Ok(BeatActivations {
            times: mel.times,
            beat,
            downbeat,
            grid: mel.grid,
        })
    }
}

#[cfg(all(test, feature = "onnx-beats"))]
mod onnx_backend_tests {
    use super::*;
    use audan_core::MonoSignal;

    /// Not run by default (needs a real converted `.rten` file, which isn't
    /// checked into the repo -- see `resources/models/beat_this/README.md`):
    /// `AUDAN_TEST_ONNX_MODEL=/path/to/beat_this_final0.rten cargo test
    /// --features onnx-beats -- --ignored onnx_backend_end_to_end`.
    /// Exercises the real load -> chunk -> infer -> stitch pipeline against
    /// a multi-chunk-length signal (over 30s, so more than one
    /// `chunk::CHUNK_SIZE` window is exercised) end to end.
    #[test]
    #[ignore]
    fn onnx_backend_end_to_end() {
        let path = std::env::var("AUDAN_TEST_ONNX_MODEL")
            .expect("set AUDAN_TEST_ONNX_MODEL to a converted beat_this .rten file");
        let backend = OnnxBackend::new(std::path::Path::new(&path), "4.0").unwrap();

        let sr = 22_050u32;
        let n = sr as usize * 45; // 45s: spans two chunks (30s each)
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.5 * (std::f32::consts::TAU * 2.0 * t).sin() // a slow 2 Hz pulse, not silence
            })
            .collect();
        let signal = MonoSignal { sample_rate: sr, samples };

        let activations = backend.run(&signal).unwrap();
        assert!(!activations.beat.is_empty());
        assert_eq!(activations.beat.len(), activations.downbeat.len());
        assert_eq!(activations.beat.len(), activations.times.len());

        for &v in activations.beat.iter().chain(activations.downbeat.iter()) {
            assert!(v.is_finite() && (0.0..=1.0).contains(&v), "activation out of range: {v}");
        }

        // Real inference on non-silence should not be a flat constant.
        let min = activations.beat.iter().cloned().fold(f32::MAX, f32::min);
        let max = activations.beat.iter().cloned().fold(f32::MIN, f32::max);
        assert!(max - min > 1e-3, "beat activation looks constant: min={min} max={max}");

        eprintln!(
            "onnx backend: {} frames, beat range [{min}, {max}]",
            activations.beat.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::MonoSignal;

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

        let backend = OnsetFallbackBackend;
        let activations = backend.run(&signal).unwrap();
        let expected_n = MelFrontend::compute(&signal, crate::DEFAULT_N_MELS).n_frames();

        assert_eq!(activations.beat.len(), expected_n);
        assert_eq!(activations.downbeat.len(), expected_n);
        assert!(activations
            .beat
            .iter()
            .all(|&v| v >= 0.0 && v <= 1.0 + 1e-6));
        assert_eq!(activations.beat, activations.downbeat);
    }

    #[test]
    fn fallback_backend_handles_empty_signal() {
        let signal = MonoSignal {
            sample_rate: 22_050,
            samples: vec![],
        };
        let backend = OnsetFallbackBackend;
        let activations = backend.run(&signal).unwrap();
        assert!(activations.beat.is_empty());
    }
}
