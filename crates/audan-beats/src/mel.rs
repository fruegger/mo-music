//! Log-mel spectrogram front end.
//!
//! S5.3 describes `MelFrontend` as matching the Beat This! input contract
//! (128 mel bands) "for exact parity with the reference implementation."
//! There is no reference implementation available in this environment to be
//! exactly parity with (see the crate root docs), so this module implements
//! a real, standard mel filterbank -- triangular filters spaced on the mel
//! scale, folded over the STFT's linear-frequency magnitude spectrum, then
//! log-compressed -- rather than faking a specific contract it cannot be
//! validated against. It is entirely real and independently testable, and is
//! the shape any future ONNX backend's frontend would plug into.

use audan_core::{FrameGrid, FrameTime, MonoSignal, PadMode};
use audan_dsp::Stft;

/// A log-mel spectrogram: `n_mels` frame-major rows, one per STFT frame.
pub struct MelSpectrogram {
    pub grid: FrameGrid,
    pub times: Vec<FrameTime>,
    pub n_mels: usize,
    /// Frame-major: `frames[k]` is the `n_mels`-length mel-band vector for
    /// frame `k`, already log-compressed.
    pub frames: Vec<Vec<f32>>,
}

impl MelSpectrogram {
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }
}

fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

/// Builds `n_mels` overlapping triangular filters spanning `0..sr/2`, each
/// expressed as a sparse weight per STFT bin (`n_bins = win/2 + 1`). Standard
/// construction (as in `librosa.filters.mel` / HTK): `n_mels + 2` points
/// equally spaced in mel space give `n_mels` triangles, each rising from its
/// left neighbour's centre and falling to its right neighbour's centre.
fn mel_filterbank(n_mels: usize, n_bins: usize, sample_rate: u32, win: usize) -> Vec<Vec<f32>> {
    let sr = sample_rate as f32;
    let fmax = sr / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);

    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m) * win as f32 / sr)
        .collect();

    let mut filters = vec![vec![0.0f32; n_bins]; n_mels];
    for (m, filter) in filters.iter_mut().enumerate() {
        let (left, center, right) = (bin_points[m], bin_points[m + 1], bin_points[m + 2]);
        for (bin, weight) in filter.iter_mut().enumerate() {
            let b = bin as f32;
            if center > left && b >= left && b <= center {
                *weight = (b - left) / (center - left);
            } else if right > center && b > center && b <= right {
                *weight = (right - b) / (right - center);
            }
        }
    }
    filters
}

/// The mel front end: STFT (fixed at hop 512 / win 2048 / reflect padding,
/// matching the analysis rate's usual resolution elsewhere in the workspace)
/// folded through a triangular mel filterbank and log-compressed.
pub struct MelFrontend;

impl MelFrontend {
    pub fn compute(signal: &MonoSignal, n_mels: usize) -> MelSpectrogram {
        let grid = FrameGrid::new(signal.sample_rate, 512, 2048, PadMode::Reflect);
        let stft = Stft::compute(grid, &signal.samples);
        let filters = mel_filterbank(n_mels, stft.n_bins, signal.sample_rate, grid.win);

        let mut frames = Vec::with_capacity(stft.n_frames());
        for k in 0..stft.n_frames() {
            let mag = stft.magnitude(k);
            let mut mel_frame = vec![0.0f32; n_mels];
            for (m, filter) in filters.iter().enumerate() {
                let mut energy = 0.0f32;
                for (bin, &w) in filter.iter().enumerate() {
                    if w > 0.0 {
                        energy += w * mag[bin];
                    }
                }
                // log1p-style compression: robust at energy == 0, standard
                // for mel-spectrogram frontends feeding a neural model.
                mel_frame[m] = (1.0 + energy).ln();
            }
            frames.push(mel_frame);
        }

        MelSpectrogram {
            grid,
            times: stft.times.clone(),
            n_mels,
            frames,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_sine(n: usize, sr: u32, freq: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn frame_and_band_counts_match_expectations() {
        let sr = 22_050u32;
        let n = sr as usize * 4;
        let signal = MonoSignal {
            sample_rate: sr,
            samples: synth_sine(n, sr, 440.0),
        };
        let n_mels = 128;
        let mel = MelFrontend::compute(&signal, n_mels);

        let grid = FrameGrid::new(sr, 512, 2048, PadMode::Reflect);
        assert_eq!(mel.n_frames(), grid.frame_count(n));
        assert_eq!(mel.n_mels, n_mels);
        for frame in &mel.frames {
            assert_eq!(frame.len(), n_mels);
        }
    }

    #[test]
    fn pure_tone_concentrates_energy_in_a_plausible_mel_band() {
        let sr = 22_050u32;
        let n = sr as usize * 2;
        let freq = 440.0f32;
        let signal = MonoSignal {
            sample_rate: sr,
            samples: synth_sine(n, sr, freq),
        };
        let n_mels = 40;
        let mel = MelFrontend::compute(&signal, n_mels);

        // Average energy per band across the (steady-state, away from edges)
        // middle of the signal.
        let mid = mel.n_frames() / 2;
        let span = 10.min(mid);
        let mut avg = vec![0.0f64; n_mels];
        for k in (mid - span)..(mid + span) {
            for (m, &v) in mel.frames[k].iter().enumerate() {
                avg[m] += v as f64;
            }
        }
        let (peak_band, _) = avg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // The band containing 440 Hz, found by inverting the same
        // equal-mel-spacing construction the filterbank itself uses.
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(sr as f32 / 2.0);
        let target_mel = hz_to_mel(freq);
        let expected_band =
            ((target_mel - mel_min) / (mel_max - mel_min) * (n_mels + 1) as f32 - 1.0).round();

        assert!(
            (peak_band as f32 - expected_band).abs() <= 2.0,
            "peak band {peak_band} not close to expected band {expected_band} for a {freq} Hz tone"
        );
    }
}
