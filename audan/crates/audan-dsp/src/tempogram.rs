//! Tempogram: local autocorrelation of the onset strength envelope, resampled
//! onto a BPM axis. Naive `O(window^2)` per analysis frame -- fine at these
//! signal lengths, not optimized.

use audan_core::FrameTime;

use crate::onset::OnsetEnvelope;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TempogramParams {
    /// Number of onset-envelope frames per autocorrelation window.
    pub window_frames: usize,
    /// Number of onset-envelope frames to advance between tempogram frames.
    pub hop_frames: usize,
    pub min_bpm: f32,
    pub max_bpm: f32,
    pub bpm_steps: usize,
}

impl Default for TempogramParams {
    fn default() -> Self {
        TempogramParams {
            window_frames: 384,
            hop_frames: 32,
            min_bpm: 40.0,
            max_bpm: 240.0,
            bpm_steps: 200,
        }
    }
}

pub struct Tempogram {
    pub times: Vec<FrameTime>,
    pub bpms: Vec<f32>,
    /// Frame-major: `values[frame][bpm_index]`, normalized so lag 0 == 1.0.
    pub values: Vec<Vec<f32>>,
}

fn autocorrelation(x: &[f32], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let centered: Vec<f64> = x.iter().map(|&v| v as f64 - mean).collect();
    let ac0: f64 = centered.iter().map(|v| v * v).sum();
    (0..=max_lag)
        .map(|lag| {
            if ac0 <= 1e-12 {
                return 0.0;
            }
            let mut acc = 0.0;
            for i in 0..(n - lag) {
                acc += centered[i] * centered[i + lag];
            }
            acc / ac0
        })
        .collect()
}

pub fn tempogram(env: &OnsetEnvelope, params: &TempogramParams) -> Tempogram {
    let sr = env.grid.sample_rate as f64;
    let hop = env.grid.hop as f64;
    let frames_per_second = sr / hop;

    let bpm_step = if params.bpm_steps > 1 {
        (params.max_bpm - params.min_bpm) / (params.bpm_steps - 1) as f32
    } else {
        0.0
    };
    let bpms: Vec<f32> = (0..params.bpm_steps)
        .map(|i| params.min_bpm + bpm_step * i as f32)
        .collect();
    // Lag (in onset-envelope frames) corresponding to each BPM.
    let lag_for_bpm = |bpm: f32| -> f64 { 60.0 * frames_per_second / bpm as f64 };

    let n = env.values.len();
    let mut times = Vec::new();
    let mut values = Vec::new();

    let mut start = 0usize;
    while start < n {
        let end = (start + params.window_frames).min(n);
        if end - start < 4 {
            break;
        }
        let window = &env.values[start..end];
        let max_lag = (end - start) - 1;
        let ac = autocorrelation(window, max_lag);

        let row: Vec<f32> = bpms
            .iter()
            .map(|&bpm| {
                let lag = lag_for_bpm(bpm);
                if lag < 0.0 || lag as usize + 1 > max_lag {
                    return 0.0;
                }
                let lo = lag.floor() as usize;
                let hi = (lo + 1).min(max_lag);
                let frac = lag - lo as f64;
                (ac[lo] * (1.0 - frac) + ac[hi] * frac) as f32
            })
            .collect();

        // Timestamp the tempogram frame at the centre of its analysis
        // window, via the same `time_of` the onset envelope itself used --
        // never a hand-rolled `centre_frame * hop / sr`.
        let centre_frame = (start + end) / 2;
        times.push(env.times[centre_frame.min(n - 1)]);
        values.push(row);

        if end == n {
            break;
        }
        start += params.hop_frames;
    }

    Tempogram {
        times,
        bpms,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::{FrameGrid, PadMode};

    #[test]
    fn recovers_known_periodicity() {
        // A synthetic onset envelope with a spike every 20 frames.
        let grid = FrameGrid::new(22_050, 512, 2048, PadMode::Reflect);
        let n = 800;
        let period = 20;
        let values: Vec<f32> = (0..n)
            .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
            .collect();
        let env = OnsetEnvelope {
            grid,
            times: (0..n).map(|k| grid.time_of(k)).collect(),
            values,
        };

        let params = TempogramParams {
            window_frames: 400,
            hop_frames: 400,
            ..Default::default()
        };
        let tg = tempogram(&env, &params);
        assert!(!tg.values.is_empty());

        let row = &tg.values[0];
        let (best_idx, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let best_bpm = tg.bpms[best_idx];

        // Period of 20 frames at hop=512, sr=22050 => frames/sec = 43.07,
        // period_seconds = 20/43.07 = 0.4643s => 129.2 BPM.
        let expected_bpm = 60.0 * (22_050.0 / 512.0) / period as f32;
        assert!(
            (best_bpm - expected_bpm).abs() < 5.0,
            "expected ~{expected_bpm} BPM, got {best_bpm}"
        );
    }
}
