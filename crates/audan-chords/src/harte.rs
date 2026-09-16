//! Harte-notation rendering (`C:maj`, `A:min`, `N`), per the architecture
//! doc's glossary and S3.2 output row. Pitch class 0 is C, ascending
//! chromatically, matching `audan-dsp`'s chroma convention.
//!
//! Sharps only, never flats: e.g. pitch class 6 renders `F#`, never `Gb`.
//! Correct enharmonic spelling depends on the surrounding key (a Bb-major
//! passage should spell its flat degrees with flats), which this crate does
//! not model -- that's `audan-key`'s domain, not chord estimation's. A
//! fixed sharps-only spelling is the standard simplification and is what
//! Harte notation itself uses in most published corpora (e.g. Isophonics).

use crate::templates::ChordQuality;

pub const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// The Harte no-chord label.
pub const NO_CHORD: &str = "N";

pub fn render(root: u8, quality: ChordQuality) -> String {
    let name = NOTE_NAMES[(root as usize) % NOTE_NAMES.len()];
    match quality {
        ChordQuality::Maj => format!("{name}:maj"),
        ChordQuality::Min => format!("{name}:min"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_natural_roots() {
        assert_eq!(render(0, ChordQuality::Maj), "C:maj");
        assert_eq!(render(9, ChordQuality::Min), "A:min");
    }

    #[test]
    fn renders_sharps_for_black_keys() {
        assert_eq!(render(1, ChordQuality::Min), "C#:min");
        assert_eq!(render(6, ChordQuality::Maj), "F#:maj");
    }
}
