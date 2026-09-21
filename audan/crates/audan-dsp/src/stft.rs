//! Short-time Fourier transform. Frame extraction honours [`PadMode`] exactly
//! as specified in S8.2 / ADR-5, and every frame's timestamp comes from
//! [`FrameGrid::time_of`] -- never a hand-rolled `k*hop/sr`.

use audan_core::{FrameGrid, FrameTime, PadMode};
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;

use crate::window;

/// Reads `signal[idx]` under the given padding mode, where `idx` is a signed
/// offset in original-signal sample coordinates (may be negative or beyond
/// `signal.len()`). Used by both the STFT and the CQT, since both need to
/// pull windows centred on arbitrary sample positions without materialising a
/// padded copy of the whole signal.
pub(crate) fn sample_at(signal: &[f32], idx: i64, pad: PadMode) -> f32 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    match pad {
        // Callers only ever pass in-bounds indices for `PadMode::None`: its
        // frame geometry (see `frame_samples`) never reaches outside the
        // signal by construction, so this mode has no "outside" to define.
        PadMode::None => signal[idx as usize],
        PadMode::Zero => {
            if idx < 0 || idx as usize >= n {
                0.0
            } else {
                signal[idx as usize]
            }
        }
        PadMode::Reflect => {
            if n == 1 {
                return signal[0];
            }
            let n_i = n as i64;
            // Mirror without repeating the edge sample (numpy's `mode="reflect"`),
            // period 2*(n-1).
            let period = 2 * (n_i - 1);
            let mut m = idx % period;
            if m < 0 {
                m += period;
            }
            if m >= n_i {
                m = period - m;
            }
            signal[m as usize]
        }
    }
}

/// Extracts the raw (un-windowed) `win`-length sample sequence for frame `k`
/// under `grid`'s padding mode. Padded modes centre frame 0 on sample 0
/// (`start = k*hop - win/2`); `PadMode::None` starts frame `k` at `k*hop`
/// with no centring, matching `FrameGrid::time_of`'s two branches exactly.
pub fn frame_samples(signal: &[f32], grid: &FrameGrid, k: usize) -> Vec<f32> {
    let win = grid.win;
    let hop = grid.hop;
    let start: i64 = match grid.pad {
        PadMode::None => (k * hop) as i64,
        _ => (k * hop) as i64 - (win / 2) as i64,
    };
    (0..win)
        .map(|i| sample_at(signal, start + i as i64, grid.pad))
        .collect()
}

/// A short-time Fourier transform: one complex half-spectrum (`win/2 + 1`
/// bins, via `realfft`) per analysis frame, Hann-windowed.
pub struct Stft {
    pub grid: FrameGrid,
    pub n_bins: usize,
    pub times: Vec<FrameTime>,
    pub spectra: Vec<Vec<Complex32>>,
}

impl Stft {
    pub fn compute(grid: FrameGrid, signal: &[f32]) -> Self {
        let win = grid.win;
        let hann = window::hann(win);
        let n_frames = grid.frame_count(signal.len());
        let n_bins = win / 2 + 1;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(win);
        let mut scratch = fft.make_scratch_vec();

        let mut times = Vec::with_capacity(n_frames);
        let mut spectra = Vec::with_capacity(n_frames);
        for k in 0..n_frames {
            let raw = frame_samples(signal, &grid, k);
            let mut buf: Vec<f32> = raw.iter().zip(hann.iter()).map(|(s, w)| s * w).collect();
            let mut out = fft.make_output_vec();
            fft.process_with_scratch(&mut buf, &mut out, &mut scratch)
                .expect(
                "realfft: buffers are sized exactly by make_input/output_vec, so this cannot fail",
            );
            times.push(grid.time_of(k));
            spectra.push(out);
        }

        Stft {
            grid,
            n_bins,
            times,
            spectra,
        }
    }

    pub fn n_frames(&self) -> usize {
        self.spectra.len()
    }

    pub fn magnitude(&self, k: usize) -> Vec<f32> {
        self.spectra[k].iter().map(|c| c.norm()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::PadMode;

    fn synth_sine(n: usize, sr: u32, freq: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    /// Parseval / Plancherel: sum(x[n]^2) == (1/N) * sum(|X[k]|^2) over the
    /// *full* N-point DFT. `realfft` only returns bins 0..=N/2, so the other
    /// half is reconstructed from conjugate symmetry: bin 0 and (for even N)
    /// the Nyquist bin count once, every interior bin counts twice.
    fn parseval_holds(grid: FrameGrid, signal: &[f32]) {
        let win = grid.win;
        let hann = window::hann(win);
        let stft = Stft::compute(grid, signal);
        assert!(
            stft.n_frames() > 0,
            "test signal too short to produce a frame"
        );

        for k in 0..stft.n_frames() {
            let raw = frame_samples(signal, &grid, k);
            let windowed: Vec<f32> = raw.iter().zip(hann.iter()).map(|(s, w)| s * w).collect();
            let time_energy: f64 = windowed.iter().map(|&s| (s as f64).powi(2)).sum();

            let spec = &stft.spectra[k];
            let mut freq_energy_full: f64 = 0.0;
            for (bin, c) in spec.iter().enumerate() {
                let mag2 = (c.norm() as f64).powi(2);
                let is_edge = bin == 0 || (win % 2 == 0 && bin == win / 2);
                freq_energy_full += if is_edge { mag2 } else { 2.0 * mag2 };
            }
            let freq_energy = freq_energy_full / win as f64;

            assert!(
                (time_energy - freq_energy).abs() < 1e-3 * time_energy.max(1e-9),
                "frame {k}: time energy {time_energy} vs freq energy {freq_energy}"
            );
        }
    }

    #[test]
    fn parseval_holds_across_pad_modes() {
        let sr = 22_050u32;
        let signal = synth_sine(4096, sr, 440.0);
        for pad in [PadMode::None, PadMode::Zero, PadMode::Reflect] {
            let grid = FrameGrid::new(sr, 512, 1024, pad);
            parseval_holds(grid, &signal);
        }
    }

    #[test]
    fn none_mode_frame_is_a_direct_slice() {
        let sr = 22_050u32;
        let signal: Vec<f32> = (0..4096).map(|i| i as f32).collect();
        let grid = FrameGrid::new(sr, 256, 512, PadMode::None);
        let f = frame_samples(&signal, &grid, 2);
        assert_eq!(f, signal[512..1024]);
    }

    #[test]
    fn padded_mode_frame_zero_is_centred_on_sample_zero() {
        let sr = 22_050u32;
        let signal: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let grid = FrameGrid::new(sr, 32, 8, PadMode::Zero);
        let f = frame_samples(&signal, &grid, 0);
        // win=8 centred on sample 0 => samples [-4..4) => [0,0,0,0, 0,1,2,3]
        assert_eq!(f, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn reflect_mode_mirrors_without_repeating_edge() {
        let signal = vec![0.0f32, 1.0, 2.0, 3.0];
        // idx -1 should reflect to signal[1] (period = 2*(4-1) = 6, mirror
        // without duplicating the boundary sample).
        assert_eq!(sample_at(&signal, -1, PadMode::Reflect), 1.0);
        assert_eq!(sample_at(&signal, 4, PadMode::Reflect), 2.0);
    }
}
