//! The timing canary (S8.7): "the single highest-value test in the project."
//!
//! Synthesise a click train at known sample positions, run the onset
//! detector, peak-pick the resulting envelope, and assert every detected
//! peak lands within 1ms of the nearest true click -- across the full
//! cross-product of padding modes, hop sizes, and sample rates. This is what
//! proves the S8.2 half-window-misalignment class of bugs cannot occur: if
//! `FrameGrid::time_of` ever used the wrong formula for a pad mode, or a
//! frame's timestamp were computed by hand somewhere instead of going
//! through it, this test would catch it as a systematic (not random) offset
//! of roughly half a window.

use audan_core::{FrameGrid, PadMode};
use audan_dsp::{onset_envelope, pick_peaks, window};

/// Adds a short Hann-shaped energy burst centred exactly on `center_sample`.
/// The burst is symmetric, so its "true" position is unambiguous. Its
/// duration is a small fraction of the analysis window (`win`) so it behaves
/// like a sharp transient relative to that window, not a sustained tone --
/// what a click train is standing in for.
fn add_click(signal: &mut [f32], center_sample: usize, win: usize, amplitude: f32) {
    let burst_len = ((win as f32 / 10.0).round() as usize).max(3) | 1; // odd, << win
    let half = burst_len / 2;
    let envelope = window::hann(burst_len);
    for i in 0..burst_len {
        let offset = i as i64 - half as i64;
        let idx = center_sample as i64 + offset;
        if idx < 0 || idx as usize >= signal.len() {
            continue;
        }
        signal[idx as usize] += amplitude * envelope[i];
    }
}

/// Builds a several-second click train at known sample positions, spread out
/// with enough margin from the signal edges that no pad mode's boundary
/// handling can plausibly perturb detection. `win` sizes each click's burst
/// relative to the analysis window under test (see `add_click`).
fn synth_click_train(sample_rate: u32, win: usize) -> (Vec<f32>, Vec<usize>) {
    let duration_s = 4.0;
    let n = (sample_rate as f32 * duration_s) as usize;
    let mut signal = vec![0.0f32; n];

    let click_times_s = [0.30, 0.77, 1.21, 1.85, 2.40, 2.93, 3.41, 3.70];
    let clicks: Vec<usize> = click_times_s
        .iter()
        .map(|&t| (t * sample_rate as f32) as usize)
        .collect();
    for &c in &clicks {
        add_click(&mut signal, c, win, 1.0);
    }
    (signal, clicks)
}

fn run_case(sample_rate: u32, hop: usize, win: usize, pad: PadMode) {
    let (signal, clicks) = synth_click_train(sample_rate, win);
    let grid = FrameGrid::new(sample_rate, hop, win, pad);
    let env = onset_envelope(grid, &signal);

    let max_flux = env.values.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_flux > 0.0,
        "onset envelope is flat for sr={sample_rate} hop={hop} win={win} pad={pad:?}"
    );
    let threshold = max_flux * 0.15;

    let peaks = pick_peaks(&env, threshold);
    assert!(
        !peaks.is_empty(),
        "no peaks detected for sr={sample_rate} hop={hop} win={win} pad={pad:?}"
    );

    let tolerance_s = 0.001; // 1ms
    for peak in &peaks {
        let peak_s = peak.time.as_seconds();
        let nearest = clicks
            .iter()
            .map(|&c| (c as f64 / sample_rate as f64 - peak_s).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest <= tolerance_s,
            "sr={sample_rate} hop={hop} win={win} pad={pad:?}: peak at frame {} ({}s) is {:.4}ms from the nearest click (tolerance 1ms)",
            peak.frame,
            peak_s,
            nearest * 1000.0
        );
    }

    // Every click should also have been found by *some* peak (no misses).
    for &c in &clicks {
        let c_s = c as f64 / sample_rate as f64;
        let found = peaks
            .iter()
            .any(|p| (p.time.as_seconds() - c_s).abs() <= tolerance_s);
        assert!(
            found,
            "sr={sample_rate} hop={hop} win={win} pad={pad:?}: click at {c_s:.4}s was not detected within 1ms"
        );
    }
}

#[test]
fn timing_canary_full_cross_product() {
    let pad_modes = [PadMode::None, PadMode::Zero, PadMode::Reflect];
    let sample_rates = [22_050u32, 44_100u32];
    // hop/win pairs (win = 2*hop, 50% overlap), in samples -- kept the same
    // across sample rates so the higher rate (finer time resolution per
    // sample) is the easier case and the lower rate is the binding one.
    let hops = [16usize, 24, 32];

    for &sr in &sample_rates {
        for &hop in &hops {
            let win = hop * 2;
            for &pad in &pad_modes {
                run_case(sr, hop, win, pad);
            }
        }
    }
}
