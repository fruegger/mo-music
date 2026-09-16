//! Pitch-class identity, shared by every module in this crate. Index 0 = C,
//! ascending chromatically -- the same convention `audan_core::Chroma` and
//! `audan-dsp`'s chroma folding use (confirmed from `audan-dsp`'s own test
//! fixtures), so a chroma frame's `[i]` lines up directly with
//! `PitchClass::from_index(i as u8)`.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
#[repr(u8)]
pub enum PitchClass {
    C = 0,
    CSharp = 1,
    D = 2,
    DSharp = 3,
    E = 4,
    F = 5,
    FSharp = 6,
    G = 7,
    GSharp = 8,
    A = 9,
    ASharp = 10,
    B = 11,
}

const ORDER: [PitchClass; 12] = [
    PitchClass::C,
    PitchClass::CSharp,
    PitchClass::D,
    PitchClass::DSharp,
    PitchClass::E,
    PitchClass::F,
    PitchClass::FSharp,
    PitchClass::G,
    PitchClass::GSharp,
    PitchClass::A,
    PitchClass::ASharp,
    PitchClass::B,
];

impl PitchClass {
    pub fn from_index(i: u8) -> Self {
        ORDER[(i % 12) as usize]
    }

    pub fn index(self) -> u8 {
        self as u8
    }

    pub fn name(self) -> &'static str {
        match self {
            PitchClass::C => "C",
            PitchClass::CSharp => "C#",
            PitchClass::D => "D",
            PitchClass::DSharp => "D#",
            PitchClass::E => "E",
            PitchClass::F => "F",
            PitchClass::FSharp => "F#",
            PitchClass::G => "G",
            PitchClass::GSharp => "G#",
            PitchClass::A => "A",
            PitchClass::ASharp => "A#",
            PitchClass::B => "B",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trips() {
        for i in 0u8..12 {
            assert_eq!(PitchClass::from_index(i).index(), i);
        }
    }

    #[test]
    fn wraps_modulo_twelve() {
        assert_eq!(PitchClass::from_index(12), PitchClass::C);
        assert_eq!(PitchClass::from_index(13), PitchClass::CSharp);
    }
}
