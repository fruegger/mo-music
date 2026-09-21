//! Krumhansl-Kessler key profiles: mean listener fit ratings for each of the
//! 12 chromatic pitch classes relative to a tonic, separately for major and
//! minor contexts (Krumhansl & Kessler 1982; reprinted as Krumhansl,
//! *Cognitive Foundations of Musical Pitch*, 1990, Table 3.1). `profile[i]`
//! is the rating for the pitch class `i` semitones above the tonic, so
//! `profile[0]` is the tonic's own rating, `profile[7]` the fifth's, etc.
//! These exact digits were cross-checked against secondary citations during
//! implementation (not against the original 1982 paper directly); the shape
//! -- high tonic/fifth/third, low on the non-diatonic degrees -- is the part
//! that actually drives the correlation and is not in doubt either way.

use audan_core::PITCH_CLASSES;

use crate::pitch_class::PitchClass;

pub const MAJOR_PROFILE: [f32; PITCH_CLASSES] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

pub const MINOR_PROFILE: [f32; PITCH_CLASSES] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Rotates a tonic-relative profile so it's expressed in absolute pitch
/// classes for the given `tonic`: `out[j] = base[(j - tonic) mod 12]`, i.e.
/// the tonic's own high rating (`base[0]`) lands on `out[tonic.index()]`.
pub fn rotated(base: &[f32; PITCH_CLASSES], tonic: PitchClass) -> [f32; PITCH_CLASSES] {
    let shift = tonic.index() as usize;
    let mut out = [0f32; PITCH_CLASSES];
    for (i, v) in base.iter().enumerate() {
        out[(i + shift) % PITCH_CLASSES] = *v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_moves_tonic_peak_to_tonic_pitch_class() {
        let rotated_a = rotated(&MAJOR_PROFILE, PitchClass::A);
        let (peak_pc, _) = rotated_a
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(peak_pc, PitchClass::A.index() as usize);
    }

    #[test]
    fn rotation_by_c_is_identity() {
        assert_eq!(rotated(&MAJOR_PROFILE, PitchClass::C), MAJOR_PROFILE);
    }
}
