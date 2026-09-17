//! Log-mel spectrogram front end matching the Beat This! (CPJKU/beat_this,
//! ISMIR 2024) reference implementation bit-for-bit, for [`crate::model::OnnxBackend`].
//!
//! Deliberately separate from [`crate::mel::MelFrontend`], which stays on its
//! own HTK-scale, hop-512/win-2048 contract for [`crate::model::OnsetFallbackBackend`].
//! The two backends need genuinely different preprocessing (different mel
//! scale formula, hop/window size, fixed vs. sample-rate-derived frequency
//! range, different log-compression constant) -- bending one frontend to
//! serve both would either compromise the fallback's own tuning or produce a
//! frontend that matches neither reference exactly.
//!
//! Ported from `beat_this.inference.LogMelSpect`, which wraps
//! `torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024,
//! hop_length=441, f_min=30, f_max=11000, n_mels=128, mel_scale="slaney",
//! normalized="frame_length", power=1)` followed by `log1p(1000 * x)`. Every
//! constant below (the Slaney mel-scale breakpoints, the frame-length STFT
//! normalisation, the periodic Hann window, the reflect-padded centred
//! framing) was read from `torchaudio`'s own source
//! (`torchaudio/functional/functional.py`), not guessed -- see
//! `resources/models/beat_this/README.md` for the export this frontend feeds.

use audan_core::{FrameGrid, FrameTime, MonoSignal, PadMode};
use audan_dsp::{stft::frame_samples, window};
use realfft::RealFftPlanner;

pub(crate) const N_FFT: usize = 1024;
pub(crate) const HOP: usize = 441;
const N_MELS: usize = 128;
const F_MIN: f32 = 30.0;
const F_MAX: f32 = 11_000.0;
const LOG_MULTIPLIER: f32 = 1000.0;

/// Slaney-scale Hz -> mel, exactly `torchaudio.functional._hz_to_mel(freq,
/// mel_scale="slaney")`: linear below 1000 Hz, log above.
fn hz_to_mel_slaney(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let mut mels = f / f_sp;
    let min_log_hz = 1000.0f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f32.ln() / 27.0;
    if f >= min_log_hz {
        mels = min_log_mel + (f / min_log_hz).ln() / logstep;
    }
    mels
}

/// Slaney-scale mel -> Hz, the inverse of [`hz_to_mel_slaney`].
fn mel_to_hz_slaney(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let mut freq = f_sp * m;
    let min_log_hz = 1000.0f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f32.ln() / 27.0;
    if m >= min_log_mel {
        freq = min_log_hz * (logstep * (m - min_log_mel)).exp();
    }
    freq
}

/// One triangular filter's nonzero span, mirroring
/// [`crate::mel::mel_filterbank`]'s sparse-span representation for the same
/// `O(n_bins)`-per-frame reason -- see that module for the rationale.
struct MelFilter {
    start_bin: usize,
    weights: Vec<f32>,
}

/// `torchaudio.functional.melscale_fbanks(n_freqs, f_min=30, f_max=11000,
/// n_mels=128, sample_rate, norm=None, mel_scale="slaney")`: `norm=None`
/// (the `LogMelSpect` call site never passes `norm="slaney"`) means no
/// per-filter area normalisation is applied -- only the frequency spacing
/// uses the Slaney formula.
fn slaney_filterbank(n_freqs: usize, sample_rate: u32) -> Vec<MelFilter> {
    let sr = sample_rate as f32;
    let nyquist = (sample_rate / 2) as f32;

    let m_min = hz_to_mel_slaney(F_MIN);
    let m_max = hz_to_mel_slaney(F_MAX);
    let mel_points: Vec<f32> = (0..N_MELS + 2)
        .map(|i| m_min + (m_max - m_min) * i as f32 / (N_MELS + 1) as f32)
        .collect();
    let f_pts: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    // `all_freqs = linspace(0, sample_rate // 2, n_freqs)`: since
    // n_freqs == n_fft/2 + 1, this lands exactly on the real STFT bin
    // frequencies `bin * sr / n_fft`, which is what we build directly here
    // rather than materialising the linspace.
    let bin_hz = |bin: usize| bin as f32 * sr / N_FFT as f32;
    let _ = nyquist; // documents the equivalence noted above; not read directly

    let mut filters = Vec::with_capacity(N_MELS);
    for m in 0..N_MELS {
        let (left, center, right) = (f_pts[m], f_pts[m + 1], f_pts[m + 2]);
        let last_bin = n_freqs.saturating_sub(1);
        // Find the bin range where this triangle can be nonzero: bin_hz is
        // monotonically increasing, so a direct search is exact (no
        // Hz->bin-index rounding the way `mel.rs`'s HTK filterbank does).
        let mut start_bin = 0usize;
        while start_bin < last_bin && bin_hz(start_bin) < left {
            start_bin += 1;
        }
        let mut end_bin = start_bin;
        while end_bin <= last_bin && bin_hz(end_bin) <= right {
            end_bin += 1;
        }
        let mut weights = Vec::with_capacity(end_bin.saturating_sub(start_bin));
        for bin in start_bin..end_bin {
            let f = bin_hz(bin);
            let down = if center > left { (f - left) / (center - left) } else { 0.0 };
            let up = if right > center { (right - f) / (right - center) } else { 0.0 };
            weights.push(down.min(up).max(0.0));
        }
        filters.push(MelFilter { start_bin, weights });
    }
    filters
}

/// The Beat This!-contract front end: fixed 22050 Hz / n_fft=1024 / hop=441
/// STFT, periodic-Hann-windowed, reflect-padded and centred exactly like
/// `torch.stft(center=True, pad_mode="reflect")`, frame-length-normalised
/// magnitude, folded through a Slaney mel filterbank, then `log1p(1000 * x)`.
///
/// Deliberately does not go through [`audan_dsp::Stft::compute`]: that
/// type's frame count for padded modes is `num_samples.div_ceil(hop)`, which
/// differs from `torch.stft(center=True)`'s `1 + num_samples / hop` (integer
/// division) whenever `num_samples` is an exact multiple of `hop` (division
/// discovered a real test case: a signal built to be exactly 1.0s at 22050
/// Hz -- 50 exact hops of 441 -- produces 50 frames the `audan_dsp` way but
/// 51 the `torch` way). `audan_dsp::Stft` is used elsewhere in this
/// workspace for other purposes with its own (equally valid) convention;
/// this frontend's whole point is bit-parity with a specific external
/// reference, so it computes its own frame count and calls the lower-level
/// [`frame_samples`] primitive directly instead.
pub struct OnnxFrontend;

/// Frame-major log-mel frames plus each frame's centre time, mirroring
/// [`crate::mel::MelSpectrogram`]'s shape for the unrelated HTK frontend.
pub struct OnnxMelFrames {
    pub frames: Vec<Vec<f32>>,
    pub times: Vec<FrameTime>,
    pub grid: FrameGrid,
}

impl OnnxFrontend {
    /// Returns frame-major log-mel frames: `frames[k]` is the 128-band
    /// vector for frame `k`, matching `LogMelSpect`'s `(time, 128)` output
    /// layout.
    pub fn compute(signal: &MonoSignal) -> OnnxMelFrames {
        let grid = FrameGrid::new(signal.sample_rate, HOP, N_FFT, PadMode::Reflect);
        let n_frames = 1 + signal.samples.len() / HOP;
        let n_bins = N_FFT / 2 + 1;
        let filters = slaney_filterbank(n_bins, signal.sample_rate);
        let hann = window::hann(N_FFT);

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut scratch = fft.make_scratch_vec();

        // `normalized="frame_length"`: divide the (otherwise unnormalised)
        // STFT by sqrt(n_fft) before taking the magnitude (power=1).
        let norm = (N_FFT as f32).sqrt();

        let mut frames = Vec::with_capacity(n_frames);
        let mut times = Vec::with_capacity(n_frames);
        for k in 0..n_frames {
            let raw = frame_samples(&signal.samples, &grid, k);
            let mut buf: Vec<f32> = raw.iter().zip(hann.iter()).map(|(s, w)| s * w).collect();
            let mut spectrum = fft.make_output_vec();
            fft.process_with_scratch(&mut buf, &mut spectrum, &mut scratch)
                .expect("realfft: buffers sized exactly by make_input/output_vec");

            let mut mel_frame = vec![0.0f32; N_MELS];
            for (m, filter) in filters.iter().enumerate() {
                let mut energy = 0.0f32;
                for (offset, &w) in filter.weights.iter().enumerate() {
                    let mag = spectrum[filter.start_bin + offset].norm() / norm;
                    energy += w * mag;
                }
                mel_frame[m] = (1.0 + LOG_MULTIPLIER * energy).ln();
            }
            frames.push(mel_frame);
            times.push(grid.time_of(k));
        }
        OnnxMelFrames { frames, times, grid }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        sample_rate: u32,
        signal: Vec<f32>,
        expected_shape: Vec<usize>,
        expected: Vec<Vec<f32>>,
    }

    /// Generated from the real `beat_this.inference.LogMelSpect` (see
    /// `resources/models/beat_this/README.md` for how to regenerate) against
    /// a deterministic two-tone-plus-noise signal -- not a pure sine, so the
    /// filterbank's overlap behaviour across many bands is exercised, not
    /// just its peak-picking on a single tone the way `mel.rs`'s own test
    /// does for the unrelated HTK frontend.
    const FIXTURE_JSON: &str = include_str!("testdata/onnx_frontend_reference.json");

    #[test]
    fn matches_the_python_reference_within_float32_tolerance() {
        let fixture: Fixture = serde_json::from_str(FIXTURE_JSON).unwrap();
        let signal = MonoSignal {
            sample_rate: fixture.sample_rate,
            samples: fixture.signal.clone(),
        };

        let mel = OnnxFrontend::compute(&signal);
        let frames = mel.frames;

        assert_eq!(frames.len(), fixture.expected_shape[0], "frame count mismatch");
        assert_eq!(frames.len(), mel.times.len(), "frames/times length mismatch");
        assert_eq!(
            frames.first().map(|f| f.len()),
            Some(fixture.expected_shape[1]),
            "band count mismatch"
        );

        let mut max_abs_diff = 0.0f32;
        for (k, (got, want)) in frames.iter().zip(fixture.expected.iter()).enumerate() {
            for (m, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                let diff = (g - w).abs();
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
                assert!(
                    diff < 1e-3,
                    "frame {k} band {m}: got {g}, want {w}, diff {diff}"
                );
            }
        }
        eprintln!("max abs diff vs. python reference: {max_abs_diff}");
    }
}
