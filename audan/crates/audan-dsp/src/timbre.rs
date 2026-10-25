//! Timbre / loudness features for structure analysis: log mel-band energies,
//! temporally smoothed and standardised per band over the track.
//!
//! Chroma only sees harmony, but most section changes in pop and rock are
//! arrangement changes (a verse becoming a chorus keeps its chords and gains
//! guitars, drums and level). Frames line up one-to-one with
//! [`crate::key_chroma`]'s when built on the same `hop` and padded `PadMode`,
//! so the two can be combined into one self-similarity matrix.

use audan_core::{FrameGrid, MonoSignal, PadMode};

use crate::stft::Stft;

const N_BANDS: usize = 24;
const F_MIN_HZ: f32 = 40.0;
const F_MAX_HZ: f32 = 8000.0;
const WIN: usize = 4096;

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Triangular mel filters as (first bin, weights) per band.
fn mel_filters(sample_rate: u32, n_bins: usize) -> Vec<(usize, Vec<f32>)> {
    let nyquist = sample_rate as f32 / 2.0;
    let f_max = F_MAX_HZ.min(nyquist);
    let (m_lo, m_hi) = (hz_to_mel(F_MIN_HZ), hz_to_mel(f_max));
    let edges: Vec<f32> = (0..N_BANDS + 2)
        .map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f32 / (N_BANDS + 1) as f32))
        .collect();
    let bin_hz = nyquist / (n_bins - 1) as f32;
    (0..N_BANDS)
        .map(|b| {
            let (lo, mid, hi) = (edges[b], edges[b + 1], edges[b + 2]);
            let first = (lo / bin_hz).floor() as usize;
            let last = ((hi / bin_hz).ceil() as usize).min(n_bins - 1);
            let weights = (first..=last)
                .map(|k| {
                    let f = k as f32 * bin_hz;
                    if f <= lo || f >= hi {
                        0.0
                    } else if f <= mid {
                        (f - lo) / (mid - lo)
                    } else {
                        (hi - f) / (hi - mid)
                    }
                })
                .collect();
            (first, weights)
        })
        .collect()
}

/// One feature vector per frame of a `hop`-spaced, `pad`-padded grid:
/// `N_BANDS` log mel energies, averaged over `smoothing_radius_frames` on each
/// side and z-scored per band across the track.
pub fn timbre_features(
    signal: &MonoSignal,
    hop: usize,
    pad: PadMode,
    smoothing_radius_frames: usize,
) -> Vec<Vec<f32>> {
    let grid = FrameGrid::new(signal.sample_rate, hop, WIN, pad);
    let stft = Stft::compute(grid, &signal.samples);
    let filters = mel_filters(signal.sample_rate, stft.n_bins);

    let mut frames: Vec<Vec<f32>> = (0..stft.n_frames())
        .map(|k| {
            let mag = stft.magnitude(k);
            filters
                .iter()
                .map(|(first, weights)| {
                    let energy: f32 = weights
                        .iter()
                        .enumerate()
                        .map(|(i, w)| w * mag[first + i] * mag[first + i])
                        .sum();
                    (energy + 1e-6).ln()
                })
                .collect()
        })
        .collect();

    smooth(&mut frames, smoothing_radius_frames);
    standardise(&mut frames);
    frames
}

fn smooth(frames: &mut [Vec<f32>], radius: usize) {
    if radius == 0 || frames.len() < 2 {
        return;
    }
    let orig = frames.to_vec();
    for (k, out) in frames.iter_mut().enumerate() {
        let lo = k.saturating_sub(radius);
        let hi = (k + radius + 1).min(orig.len());
        let n = (hi - lo) as f32;
        for (b, v) in out.iter_mut().enumerate() {
            *v = orig[lo..hi].iter().map(|row| row[b]).sum::<f32>() / n;
        }
    }
}

/// Zero mean, unit variance per band (bands with no variance become zero).
fn standardise(frames: &mut [Vec<f32>]) {
    let n = frames.len() as f32;
    if n < 2.0 {
        return;
    }
    for b in 0..N_BANDS {
        let mean = frames.iter().map(|f| f[b]).sum::<f32>() / n;
        let var = frames.iter().map(|f| (f[b] - mean).powi(2)).sum::<f32>() / n;
        let sd = var.sqrt();
        for f in frames.iter_mut() {
            f[b] = if sd > 1e-6 { (f[b] - mean) / sd } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sr: u32, seconds: f32, freq: f32, amp: f32) -> Vec<f32> {
        (0..(sr as f32 * seconds) as usize)
            .map(|i| amp * (std::f32::consts::TAU * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn frame_count_matches_the_chroma_grid() {
        let sr = 22050;
        let signal = MonoSignal {
            sample_rate: sr,
            samples: tone(sr, 5.0, 440.0, 0.5),
        };
        let frames = timbre_features(&signal, 4096, PadMode::Reflect, 2);
        let chroma_frames = FrameGrid::new(sr, 4096, 8192, PadMode::Reflect).frame_count(signal.samples.len());
        assert_eq!(frames.len(), chroma_frames);
        assert!(frames.iter().all(|f| f.len() == N_BANDS));
    }

    #[test]
    fn timbre_change_separates_the_two_halves() {
        // Same level, different spectrum: a low tone then a high one.
        let sr = 22050;
        let mut samples = tone(sr, 10.0, 200.0, 0.5);
        samples.extend(tone(sr, 10.0, 3000.0, 0.5));
        let frames = timbre_features(&MonoSignal { sample_rate: sr, samples }, 4096, PadMode::Reflect, 0);
        let dist = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum() };
        let n = frames.len();
        let within = dist(&frames[n / 4], &frames[n / 4 + 3]);
        let across = dist(&frames[n / 4], &frames[3 * n / 4]);
        assert!(across > 10.0 * within.max(1e-3), "across {across}, within {within}");
    }
}
