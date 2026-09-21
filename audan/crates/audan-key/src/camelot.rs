//! Camelot wheel and Open Key notation.
//!
//! Both systems lay the 24 keys on the same circle-of-fifths wheel and both
//! give a major key and its relative minor the same wheel number (differing
//! only in the letter suffix) -- but the two systems do **not** start
//! numbering at the same point on the wheel. Camelot fixes C major at `8B`
//! (and so A minor, its relative minor, at `8A`); Open Key fixes C major at
//! `1d` (A minor at `1m`). This was verified against several independent
//! secondary sources during implementation, including the landmark pair
//! `11B` (A major) <-> `4d`, which only holds under a constant offset, not
//! under "same number, different letter". The offset is constant across the
//! wheel: `open_number = camelot_number - 7`, wrapped into `1..=12`.

use crate::mode::Mode;
use crate::pitch_class::PitchClass;

/// Camelot wheel number (1..=12) for the *major* key whose tonic is pitch
/// class `pc`. Built by walking the circle of fifths from C (pc 0) = 8,
/// where each ascending fifth (+7 semitones) is +1 on the wheel.
const CAMELOT_NUMBER_FOR_MAJOR_PC: [u8; 12] = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1];

fn camelot_number(tonic: PitchClass, mode: Mode) -> u8 {
    // A minor key shares its relative major's wheel number; the relative
    // major's tonic is a minor third (3 semitones) above the minor tonic.
    let major_pc = match mode {
        Mode::Major => tonic.index(),
        Mode::Minor => (tonic.index() + 3) % 12,
    };
    CAMELOT_NUMBER_FOR_MAJOR_PC[major_pc as usize]
}

pub fn to_camelot(tonic: PitchClass, mode: Mode) -> String {
    let n = camelot_number(tonic, mode);
    let letter = match mode {
        Mode::Major => 'B',
        Mode::Minor => 'A',
    };
    format!("{n}{letter}")
}

pub fn to_open_key(tonic: PitchClass, mode: Mode) -> String {
    let camelot_n = camelot_number(tonic, mode) as i32;
    let open_n = (camelot_n - 1 - 7).rem_euclid(12) + 1;
    let letter = match mode {
        Mode::Major => 'd',
        Mode::Minor => 'm',
    };
    format!("{open_n}{letter}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_major_is_camelot_8b() {
        assert_eq!(to_camelot(PitchClass::C, Mode::Major), "8B");
    }

    #[test]
    fn a_minor_is_camelot_8a() {
        assert_eq!(to_camelot(PitchClass::A, Mode::Minor), "8A");
    }

    #[test]
    fn c_major_is_open_key_1d() {
        assert_eq!(to_open_key(PitchClass::C, Mode::Major), "1d");
    }

    #[test]
    fn a_minor_is_open_key_1m() {
        assert_eq!(to_open_key(PitchClass::A, Mode::Minor), "1m");
    }

    #[test]
    fn a_major_is_camelot_11b_and_open_key_4d() {
        assert_eq!(to_camelot(PitchClass::A, Mode::Major), "11B");
        assert_eq!(to_open_key(PitchClass::A, Mode::Major), "4d");
    }

    #[test]
    fn relative_major_and_minor_share_camelot_and_open_key_number() {
        for i in 0u8..12 {
            let major_tonic = PitchClass::from_index(i);
            let relative_minor = PitchClass::from_index((i + 9) % 12);
            let cam_major = to_camelot(major_tonic, Mode::Major);
            let cam_minor = to_camelot(relative_minor, Mode::Minor);
            assert_eq!(
                &cam_major[..cam_major.len() - 1],
                &cam_minor[..cam_minor.len() - 1]
            );

            let open_major = to_open_key(major_tonic, Mode::Major);
            let open_minor = to_open_key(relative_minor, Mode::Minor);
            assert_eq!(
                &open_major[..open_major.len() - 1],
                &open_minor[..open_minor.len() - 1]
            );
        }
    }

    #[test]
    fn all_24_camelot_and_open_key_strings_are_distinct_and_well_formed() {
        let mut camelots = std::collections::HashSet::new();
        let mut open_keys = std::collections::HashSet::new();
        for i in 0u8..12 {
            for mode in [Mode::Major, Mode::Minor] {
                let tonic = PitchClass::from_index(i);
                let cam = to_camelot(tonic, mode);
                let ok = to_open_key(tonic, mode);

                let (num, letter) = cam.split_at(cam.len() - 1);
                let n: u32 = num.parse().unwrap();
                assert!((1..=12).contains(&n));
                assert!(letter == "A" || letter == "B");
                assert!(camelots.insert(cam));

                let (num, letter) = ok.split_at(ok.len() - 1);
                let n: u32 = num.parse().unwrap();
                assert!((1..=12).contains(&n));
                assert!(letter == "d" || letter == "m");
                assert!(open_keys.insert(ok));
            }
        }
        assert_eq!(camelots.len(), 24);
        assert_eq!(open_keys.len(), 24);
    }
}
