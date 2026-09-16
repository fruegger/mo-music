//! Key estimation: aggregate a chroma sequence into one 12-dim profile,
//! correlate it against all 24 rotated Krumhansl-Kessler profiles (12
//! tonics x major/minor), and report the top 3 as a `Ranked<KeyEstimate>`
//! (ADR-11, S8.5, QS9). Relative major and minor share 6 of 7 diatonic
//! scale degrees and are the textbook irreducible ambiguity in chroma-based
//! key estimation, so collapsing straight to a winner would silently
//! discard a real ambiguity rather than surface it.

use audan_core::{Candidate, Chroma, MonoSignal, Ranked, PITCH_CLASSES};
use audan_dsp::KeyChromaParams;

use crate::key_estimate::KeyEstimate;
use crate::mode::Mode;
use crate::pitch_class::PitchClass;
use crate::profiles::{rotated, MAJOR_PROFILE, MINOR_PROFILE};

const TOP_N: usize = 3;

fn aggregate(chroma: &Chroma) -> [f32; PITCH_CLASSES] {
    let mut acc = [0f32; PITCH_CLASSES];
    let n_frames = chroma.n_frames();
    for frame in chroma.frames() {
        for (a, v) in acc.iter_mut().zip(frame.iter()) {
            *a += v;
        }
    }
    if n_frames > 0 {
        let n = n_frames as f32;
        for a in acc.iter_mut() {
            *a /= n;
        }
    }
    let norm: f32 = acc.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for a in acc.iter_mut() {
            *a /= norm;
        }
    }
    acc
}

fn pearson(a: &[f32; PITCH_CLASSES], b: &[f32; PITCH_CLASSES]) -> f32 {
    let n = PITCH_CLASSES as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut cov = 0f32;
    let mut var_a = 0f32;
    let mut var_b = 0f32;
    for i in 0..PITCH_CLASSES {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom > 1e-9 {
        cov / denom
    } else {
        0.0
    }
}

/// Estimates the key of an already-computed key-chroma sequence (typically
/// from [`audan_dsp::key_chroma`]). Returns the top 3 candidates ranked by
/// confidence (ADR-11) -- never a bare point estimate.
///
/// Confidence is a Pearson correlation coefficient in `[-1, 1]`, linearly
/// rescaled to `[0, 1]` to fit `audan_core::Confidence`'s documented range.
/// It is not a calibrated probability, only an ordering-preserving rescale.
pub fn estimate_key(chroma: &Chroma) -> Ranked<KeyEstimate> {
    let profile = aggregate(chroma);

    let mut scored: Vec<Candidate<KeyEstimate>> = Vec::with_capacity(2 * PITCH_CLASSES);
    for i in 0..PITCH_CLASSES as u8 {
        let tonic = PitchClass::from_index(i);
        let maj_score = pearson(&profile, &rotated(&MAJOR_PROFILE, tonic));
        let min_score = pearson(&profile, &rotated(&MINOR_PROFILE, tonic));
        scored.push(Candidate::new(
            KeyEstimate::new(tonic, Mode::Major),
            (maj_score + 1.0) / 2.0,
        ));
        scored.push(Candidate::new(
            KeyEstimate::new(tonic, Mode::Minor),
            (min_score + 1.0) / 2.0,
        ));
    }

    let ranked = Ranked::new(scored).expect("24 candidates are always produced");
    let top: Vec<Candidate<KeyEstimate>> = ranked.as_slice().iter().take(TOP_N).cloned().collect();
    Ranked::new(top).expect("TOP_N >= 1 and the input was non-empty")
}

/// Convenience: `audan_dsp::key_chroma` followed by [`estimate_key`], for
/// the common case of estimating a key straight from decoded audio.
pub fn estimate_key_from_signal(
    signal: &MonoSignal,
    params: &KeyChromaParams,
) -> Ranked<KeyEstimate> {
    let chroma = audan_dsp::key_chroma(signal, params);
    estimate_key(&chroma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::{FrameGrid, PadMode, PITCH_CLASSES as N};

    fn grid() -> FrameGrid {
        FrameGrid::new(22_050, 4096, 8192, PadMode::Reflect)
    }

    fn chroma_from_profile(profile: [f32; N], n_frames: usize) -> Chroma {
        Chroma::from_frames(grid(), vec![profile; n_frames])
    }

    #[test]
    fn recovers_c_major_from_c_major_shaped_chroma() {
        let profile = rotated(&MAJOR_PROFILE, PitchClass::C);
        let chroma = chroma_from_profile(profile, 5);
        let ranked = estimate_key(&chroma);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked.top().value.name, "C major");
    }

    #[test]
    fn recovers_g_major_from_rotated_chroma() {
        let profile = rotated(&MAJOR_PROFILE, PitchClass::G);
        let chroma = chroma_from_profile(profile, 3);
        let ranked = estimate_key(&chroma);
        assert_eq!(ranked.top().value.name, "G major");
    }

    #[test]
    fn ranked_output_has_exactly_three_candidates_sorted_descending() {
        let profile = rotated(&MAJOR_PROFILE, PitchClass::D);
        let chroma = chroma_from_profile(profile, 4);
        let ranked = estimate_key(&chroma);
        assert_eq!(ranked.len(), 3);
        let confidences: Vec<f32> = ranked.as_slice().iter().map(|c| c.confidence).collect();
        assert!(confidences.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn relative_major_minor_ambiguity_surfaces_both_in_top_three() {
        // The 7 natural-minor / major-scale shared pitch classes, equally
        // weighted, with the non-diatonic 5 pitch classes at zero: a chroma
        // shape that is, by construction, exactly as consistent with C major
        // as with its relative A minor (QS9's spirit).
        let mut shared = [0f32; N];
        for pc in [0u8, 2, 4, 5, 7, 9, 11] {
            shared[pc as usize] = 1.0;
        }
        let chroma = chroma_from_profile(shared, 6);
        let ranked = estimate_key(&chroma);

        let names: Vec<&str> = ranked
            .as_slice()
            .iter()
            .map(|c| c.value.name.as_str())
            .collect();
        assert!(names.contains(&"C major"), "top-3 was {names:?}");
        assert!(names.contains(&"A minor"), "top-3 was {names:?}");
    }
}
