//! Constant-Q transform.
//!
//! This is a **naive, per-bin implementation**, not a fast/pruned CQT (e.g.
//! Brown/Puckette or the sliding-CQT variants): for every log-spaced centre
//! frequency we directly correlate a Hann-windowed segment of the signal
//! against a complex exponential at that frequency (a single-bin DFT). Each
//! bin's window length is sized so that `window_len / sample_rate` covers
//! `Q` periods of that bin's frequency -- longer windows at low frequencies,
//! shorter at high ones -- which is what gives the transform constant-Q
//! behaviour (log-frequency bins with proportional bandwidth) as opposed to
//! an STFT with relabelled/interpolated axes. It's O(frames * bins *
//! max_window_len) with no pruning or recursive bin reuse; fine for the
//! signal lengths this crate is exercised on, not for real-time use.

use audan_core::{FrameGrid, FrameTime};

use crate::stft::sample_at;
use crate::window;

/// Frequency of MIDI-like pitch C1, the conventional low end of a musical CQT.
pub const C1_HZ: f32 = 32.703_2;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CqtParams {
    pub bins_per_octave: usize,
    pub n_octaves: usize,
    pub f_min: f32,
}

impl Default for CqtParams {
    fn default() -> Self {
        // 12 bins/octave (one per semitone) x 7 octaves = C1..C8, spanning
        // the practical musical range.
        CqtParams {
            bins_per_octave: 12,
            n_octaves: 7,
            f_min: C1_HZ,
        }
    }
}

impl CqtParams {
    pub fn n_bins(&self) -> usize {
        self.bins_per_octave * self.n_octaves
    }

    pub fn bin_freq(&self, k: usize) -> f32 {
        self.f_min * 2f32.powf(k as f32 / self.bins_per_octave as f32)
    }

    /// The constant-Q quality factor implied by the bin spacing: `Q = f / df`
    /// for adjacent log-spaced bins one step apart.
    fn q_factor(&self) -> f32 {
        1.0 / (2f32.powf(1.0 / self.bins_per_octave as f32) - 1.0)
    }

    pub(crate) fn window_len(&self, k: usize, sample_rate: u32) -> usize {
        let f = self.bin_freq(k);
        let n = (self.q_factor() * sample_rate as f32 / f).round() as i64;
        n.max(4) as usize
    }
}

/// A constant-Q magnitude spectrogram: `magnitudes[frame][bin]`, with `bin`
/// running from `params.f_min` upward at `params.bins_per_octave` steps per
/// octave.
pub struct Cqt {
    pub grid: FrameGrid,
    pub params: CqtParams,
    pub times: Vec<FrameTime>,
    pub magnitudes: Vec<Vec<f32>>,
}

impl Cqt {
    pub fn compute(grid: FrameGrid, params: CqtParams, signal: &[f32]) -> Self {
        let n_frames = grid.frame_count(signal.len());
        let n_bins = params.n_bins();
        let sr = grid.sample_rate;

        let bin_windows: Vec<Vec<f32>> = (0..n_bins)
            .map(|k| window::hann(params.window_len(k, sr)))
            .collect();
        let bin_norms: Vec<f64> = bin_windows
            .iter()
            .map(|w| w.iter().map(|&x| x as f64).sum::<f64>() / 2.0)
            .collect();

        let mut times = Vec::with_capacity(n_frames);
        let mut magnitudes = Vec::with_capacity(n_frames);
        for j in 0..n_frames {
            // The only route from a frame index to a time is `time_of`; we
            // then convert that canonical time back to a sample offset so
            // every bin's (differently sized) window is centred consistently
            // with the rest of the system, rather than re-deriving "centre"
            // from k*hop by hand.
            let t = grid.time_of(j);
            let center_sample = (t.as_seconds() * sr as f64).round() as i64;

            let mut row = Vec::with_capacity(n_bins);
            for k in 0..n_bins {
                let win = &bin_windows[k];
                let n_k = win.len() as i64;
                let f_k = params.bin_freq(k) as f64;
                let start = center_sample - n_k / 2;

                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for i in 0..n_k {
                    let s = sample_at(signal, start + i, grid.pad) as f64 * win[i as usize] as f64;
                    let phase = -std::f64::consts::TAU * f_k * i as f64 / sr as f64;
                    re += s * phase.cos();
                    im += s * phase.sin();
                }
                let norm = bin_norms[k].max(1e-9);
                row.push(((re * re + im * im).sqrt() / norm) as f32);
            }
            times.push(t);
            magnitudes.push(row);
        }

        Cqt {
            grid,
            params,
            times,
            magnitudes,
        }
    }

    pub fn n_frames(&self) -> usize {
        self.magnitudes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::PadMode;

    #[test]
    fn peak_bin_matches_known_pitch() {
        let sr = 22_050u32;
        // A4 = 440 Hz.
        let freq = 440.0f32;
        let duration_s = 1.0;
        let n = (sr as f32 * duration_s) as usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / sr as f32).sin())
            .collect();

        let params = CqtParams::default();
        let grid = FrameGrid::new(sr, 1024, 2048, PadMode::Reflect);
        let cqt = Cqt::compute(grid, params, &signal);

        // Middle frame, away from any edge effects.
        let mid = cqt.n_frames() / 2;
        let row = &cqt.magnitudes[mid];
        let (peak_bin, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let peak_freq = params.bin_freq(peak_bin);
        // Within a quarter semitone of 440 Hz.
        let ratio = peak_freq / freq;
        assert!(
            ratio > 2f32.powf(-0.25 / 12.0) && ratio < 2f32.powf(0.25 / 12.0),
            "peak bin {peak_bin} => {peak_freq} Hz, expected near {freq} Hz"
        );
        // And specifically the A pitch class (A1..A8 are bins 9, 21, 33, ...:
        // 9 semitones above each C).
        assert_eq!(peak_bin % 12, 9);
    }
}
