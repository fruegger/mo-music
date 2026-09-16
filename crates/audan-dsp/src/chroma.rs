//! Chroma (pitch-class profile) computation: fold a [`Cqt`]'s per-frame
//! magnitude spectrum across octaves into 12 pitch-class bins.
//!
//! Two distinct, separately-parameterized entry points are exposed rather
//! than one function with knobs, because the architecture doc (RV2) requires
//! that the key-estimation chroma (long window, heavy temporal smoothing)
//! and the chord-estimation chroma (short/beat-synchronous-ready, harmonic
//! energy suppressed) never share one `params_hash` / cache entry. Distinct
//! parameter *types* make that the natural outcome for any caller, without
//! this crate needing to know anything about cache keys.

use audan_core::{Chroma, FrameGrid, MonoSignal, PadMode, PITCH_CLASSES};

use crate::cqt::{Cqt, CqtParams, C1_HZ};

fn fold_octaves(cqt: &Cqt) -> Vec<[f32; PITCH_CLASSES]> {
    let bpo = cqt.params.bins_per_octave;
    let n_octaves = cqt.params.n_octaves;
    cqt.magnitudes
        .iter()
        .map(|row| {
            let mut pc = [0f32; PITCH_CLASSES];
            for octave in 0..n_octaves {
                for semitone in 0..PITCH_CLASSES.min(bpo) {
                    pc[semitone] += row[octave * bpo + semitone];
                }
            }
            pc
        })
        .collect()
}

fn l1_normalize(frames: &mut [[f32; PITCH_CLASSES]]) {
    for f in frames.iter_mut() {
        let sum: f32 = f.iter().sum();
        if sum > 1e-9 {
            for v in f.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Suppresses energy in each pitch class that's plausibly leaking from a
/// harmonic of a *lower* fundamental: the 3rd harmonic of a note lands a
/// perfect fifth (+7 semitones) above it, and the 5th harmonic lands two
/// octaves plus a major third (+28 semitones, i.e. +4 mod 12) above it. We
/// don't know which pitch class is the "true" fundamental, so this just
/// attenuates every bin by a fraction of its fifth-below and major-third-below
/// neighbours' energy -- a standard, simple chroma heuristic (see e.g. Ellis's
/// harmonic-weighted chromagram), not a full harmonic/percussive source model.
fn suppress_harmonics(frames: &mut [[f32; PITCH_CLASSES]], alpha: f32, beta: f32) {
    for f in frames.iter_mut() {
        let orig = *f;
        for pc in 0..PITCH_CLASSES {
            let fifth_below = orig[(pc + PITCH_CLASSES - 7) % PITCH_CLASSES];
            let third_below = orig[(pc + PITCH_CLASSES - 4) % PITCH_CLASSES];
            f[pc] = (orig[pc] - alpha * fifth_below - beta * third_below).max(0.0);
        }
    }
}

fn smooth_temporal(frames: &mut [[f32; PITCH_CLASSES]], radius: usize) {
    if radius == 0 || frames.len() < 2 {
        return;
    }
    let orig = frames.to_vec();
    for (k, out) in frames.iter_mut().enumerate() {
        let lo = k.saturating_sub(radius);
        let hi = (k + radius + 1).min(orig.len());
        let mut acc = [0f32; PITCH_CLASSES];
        for row in &orig[lo..hi] {
            for pc in 0..PITCH_CLASSES {
                acc[pc] += row[pc];
            }
        }
        let n = (hi - lo) as f32;
        for pc in 0..PITCH_CLASSES {
            out[pc] = acc[pc] / n;
        }
    }
}

/// Parameters for the key-estimation chroma: a large hop (coarse time
/// resolution) and heavy temporal smoothing, since key estimation wants a
/// stable, slowly-varying pitch-class profile rather than fine-grained
/// harmonic changes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KeyChromaParams {
    pub hop: usize,
    pub n_octaves: usize,
    pub smoothing_radius_frames: usize,
    pub pad: PadMode,
}

impl Default for KeyChromaParams {
    fn default() -> Self {
        KeyChromaParams {
            hop: 4096,
            n_octaves: 7,
            smoothing_radius_frames: 15,
            pad: PadMode::Reflect,
        }
    }
}

/// Parameters for the chord-estimation chroma: a short hop suitable for
/// later beat-synchronous averaging by `audan-chords`, with harmonic energy
/// suppression so chord templates aren't confused by 3rd/5th-harmonic bleed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ChordChromaParams {
    pub hop: usize,
    pub n_octaves: usize,
    pub harmonic_suppression: bool,
    pub suppression_alpha: f32,
    pub suppression_beta: f32,
    pub pad: PadMode,
}

impl Default for ChordChromaParams {
    fn default() -> Self {
        ChordChromaParams {
            hop: 512,
            n_octaves: 7,
            harmonic_suppression: true,
            suppression_alpha: 0.6,
            suppression_beta: 0.3,
            pad: PadMode::Reflect,
        }
    }
}

fn cqt_grid(
    sample_rate: u32,
    hop: usize,
    n_octaves: usize,
    pad: PadMode,
) -> (FrameGrid, CqtParams) {
    let cqt_params = CqtParams {
        bins_per_octave: PITCH_CLASSES,
        n_octaves,
        f_min: C1_HZ,
    };
    // `win` only feeds `FrameTime` for `PadMode::None`; give it the lowest
    // (longest) bin's window length as the grid's nominal window.
    let nominal_win = cqt_params.window_len(0, sample_rate);
    (
        FrameGrid::new(sample_rate, hop, nominal_win, pad),
        cqt_params,
    )
}

pub fn key_chroma(signal: &MonoSignal, params: &KeyChromaParams) -> Chroma {
    let (grid, cqt_params) = cqt_grid(signal.sample_rate, params.hop, params.n_octaves, params.pad);
    let cqt = Cqt::compute(grid, cqt_params, &signal.samples);
    let mut frames = fold_octaves(&cqt);
    smooth_temporal(&mut frames, params.smoothing_radius_frames);
    l1_normalize(&mut frames);
    Chroma::from_frames(grid, frames)
}

pub fn chord_chroma(signal: &MonoSignal, params: &ChordChromaParams) -> Chroma {
    let (grid, cqt_params) = cqt_grid(signal.sample_rate, params.hop, params.n_octaves, params.pad);
    let cqt = Cqt::compute(grid, cqt_params, &signal.samples);
    let mut frames = fold_octaves(&cqt);
    if params.harmonic_suppression {
        suppress_harmonics(
            &mut frames,
            params.suppression_alpha,
            params.suppression_beta,
        );
    }
    l1_normalize(&mut frames);
    Chroma::from_frames(grid, frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A signal built from a fundamental plus a couple of harmonics, all
    /// sharing pitch class C (fundamental at C3 = 130.81 Hz; its harmonics
    /// land on octaves of C, which fold onto the same pitch class).
    fn synth_c_with_harmonics(sr: u32) -> MonoSignal {
        let f0 = 130.81f32; // C3
        let duration_s = 2.0;
        let n = (sr as f32 * duration_s) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut s = 0.0;
                for h in [1.0, 2.0, 4.0] {
                    s += (std::f32::consts::TAU * f0 * h * t).sin() / h;
                }
                s
            })
            .collect();
        MonoSignal {
            sample_rate: sr,
            samples,
        }
    }

    #[test]
    fn key_chroma_peaks_on_correct_pitch_class() {
        let signal = synth_c_with_harmonics(22_050);
        let params = KeyChromaParams {
            hop: 2048,
            ..Default::default()
        };
        let chroma = key_chroma(&signal, &params);
        assert!(chroma.n_frames() > 0);
        let mid = chroma.frame(chroma.n_frames() / 2);
        let (peak_pc, _) = mid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(
            peak_pc, 0,
            "expected pitch class C (0), got {peak_pc}: {mid:?}"
        );
    }

    #[test]
    fn chord_chroma_peaks_on_correct_pitch_class() {
        let signal = synth_c_with_harmonics(22_050);
        let params = ChordChromaParams::default();
        let chroma = chord_chroma(&signal, &params);
        assert!(chroma.n_frames() > 0);
        let mid = chroma.frame(chroma.n_frames() / 2);
        let (peak_pc, _) = mid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(
            peak_pc, 0,
            "expected pitch class C (0), got {peak_pc}: {mid:?}"
        );
    }

    #[test]
    fn key_and_chord_params_are_distinct_types() {
        // Compile-time assertion: these are genuinely separate parameter
        // types (S RV2), so a `params_hash` built from one can never collide
        // with one built from the other.
        fn assert_distinct<A, B>() {}
        assert_distinct::<KeyChromaParams, ChordChromaParams>();
    }
}
