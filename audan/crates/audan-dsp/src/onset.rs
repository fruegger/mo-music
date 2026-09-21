//! Onset strength envelope (spectral flux) and peak picking.
//!
//! This is the feature exercised by the timing canary (S8.7): it's the
//! transient detector every frame timestamp downstream ultimately traces
//! back to `FrameGrid::time_of`, so getting its timestamps right is the
//! whole point of the crate.

use audan_core::{FrameGrid, FrameTime};

use crate::stft::Stft;

pub struct OnsetEnvelope {
    pub grid: FrameGrid,
    pub times: Vec<FrameTime>,
    pub values: Vec<f32>,
}

/// Sum of positive-only frame-to-frame magnitude-spectrum differences
/// ("spectral flux"). Frame 0 has no predecessor and is defined as 0.
pub fn spectral_flux(stft: &Stft) -> OnsetEnvelope {
    let n = stft.n_frames();
    let mut values = Vec::with_capacity(n);
    let mut prev: Option<Vec<f32>> = None;
    for k in 0..n {
        let mag = stft.magnitude(k);
        let flux = match &prev {
            None => 0.0,
            Some(p) => mag
                .iter()
                .zip(p.iter())
                .map(|(a, b)| (a - b).max(0.0))
                .sum(),
        };
        values.push(flux);
        prev = Some(mag);
    }
    OnsetEnvelope {
        grid: stft.grid,
        times: stft.times.clone(),
        values,
    }
}

/// Computes the STFT internally and folds it into a spectral-flux onset
/// envelope in one call.
pub fn onset_envelope(grid: FrameGrid, signal: &[f32]) -> OnsetEnvelope {
    let stft = Stft::compute(grid, signal);
    spectral_flux(&stft)
}

/// A detected onset: an integer-frame local maximum of the envelope, refined
/// to sub-frame precision by quadratic interpolation of the three samples
/// around the peak (standard technique for sub-bin/sub-frame peak
/// localisation). The refined time is built from two calls to
/// `FrameGrid::time_of` (the peak frame and its neighbour), not a hand-rolled
/// offset formula: `time_of(k+1) - time_of(k) == hop/sr` exactly, regardless
/// of padding mode (see `audan_core::frame` tests), so interpolating linearly
/// between those two timestamps stays inside the one true timestamp
/// convention.
#[derive(Copy, Clone, Debug)]
pub struct Peak {
    pub frame: usize,
    pub time: FrameTime,
}

/// Finds local maxima of `env` at or above `threshold` and refines each to
/// sub-frame precision.
pub fn pick_peaks(env: &OnsetEnvelope, threshold: f32) -> Vec<Peak> {
    let v = &env.values;
    let n = v.len();
    let mut peaks = Vec::new();

    for i in 0..n {
        if v[i] < threshold {
            continue;
        }
        let greater_than_left = i == 0 || v[i] > v[i - 1];
        let greater_eq_right = i + 1 == n || v[i] >= v[i + 1];
        if !(greater_than_left && greater_eq_right) {
            continue;
        }

        let time = if i > 0 && i + 1 < n {
            let (l, c, r) = (v[i - 1] as f64, v[i] as f64, v[i + 1] as f64);
            let denom = l - 2.0 * c + r;
            let t0 = env.times[i].as_seconds();
            if denom.abs() > 1e-12 {
                let frac = 0.5 * (l - r) / denom; // in (-0.5, 0.5)
                let frac = frac.clamp(-0.5, 0.5);
                let frame_spacing = env.times[i + 1].as_seconds() - t0;
                FrameTime::from_seconds_centered(t0 + frac * frame_spacing)
            } else {
                FrameTime::from_seconds_centered(t0)
            }
        } else {
            env.times[i]
        };

        peaks.push(Peak { frame: i, time });
    }

    peaks
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::PadMode;

    #[test]
    fn flat_envelope_has_no_peaks() {
        let grid = FrameGrid::new(22_050, 256, 512, PadMode::Reflect);
        let env = OnsetEnvelope {
            grid,
            times: (0..10).map(|k| grid.time_of(k)).collect(),
            values: vec![0.0; 10],
        };
        assert!(pick_peaks(&env, 0.01).is_empty());
    }

    #[test]
    fn single_spike_is_detected_and_refined_toward_its_true_centre() {
        let grid = FrameGrid::new(22_050, 256, 512, PadMode::Reflect);
        let values = vec![0.0, 0.2, 1.0, 0.3, 0.0];
        let env = OnsetEnvelope {
            grid,
            times: (0..5).map(|k| grid.time_of(k)).collect(),
            values,
        };
        let peaks = pick_peaks(&env, 0.5);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].frame, 2);
        // Slightly asymmetric (0.2 vs 0.3) so the interpolated peak should
        // shift slightly toward frame 3.
        let t2 = env.times[2].as_seconds();
        let t3 = env.times[3].as_seconds();
        assert!(peaks[0].time.as_seconds() > t2 && peaks[0].time.as_seconds() < t3);
    }
}
