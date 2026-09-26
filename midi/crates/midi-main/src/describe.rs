//! Human readable descriptions of MIDI messages.

use crate::midi::MidiMessage;
use clap::ValueEnum;

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum KeyFormat {
    /// key name, e.g. "Bb major" or "F# minor"
    Classic,
    /// Camelot wheel code, e.g. "6B" or "11A"
    Camelot,
}

/// major keys by number of sharps (positive) or flats (negative), from 7 flats to 7 sharps
const MAJOR_KEYS: [&str; 15] = [
    "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#",
];
/// the relative minor keys, in the same order
const MINOR_KEYS: [&str; 15] = [
    "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#",
];

/// Describes `message` like its Debug form, but with time signature, key signature
/// and tempo in the notation musicians use.
pub fn describe(message: &MidiMessage, key_format: KeyFormat) -> String {
    let readable = match *message {
        MidiMessage::TimeSignature(numerator, denominator_pow, _, _) => {
            time_signature(numerator, denominator_pow).map(|ts| format!("TimeSignature({ts})"))
        }
        MidiMessage::KeySignature(sharps_flats, minor) => {
            key(sharps_flats, minor, key_format).map(|key| format!("KeySignature({key})"))
        }
        MidiMessage::SetTempo(tempo) => bpm(tempo).map(|bpm| format!("SetTempo(b={bpm})")),
        _ => None,
    };
    // values outside the spec's range are shown as stored
    readable.unwrap_or_else(|| format!("{message:?}"))
}

/// "6/8"; the denominator is stored as a power of two
fn time_signature(numerator: u8, denominator_pow: u8) -> Option<String> {
    let denominator = 1u32.checked_shl(denominator_pow as u32)?;
    Some(format!("{numerator}/{denominator}"))
}

fn key(sharps_flats: i8, minor: bool, key_format: KeyFormat) -> Option<String> {
    if !(-7..=7).contains(&sharps_flats) {
        return None;
    }
    let index = (sharps_flats + 7) as usize;
    Some(match key_format {
        KeyFormat::Classic => match minor {
            false => format!("{} major", MAJOR_KEYS[index]),
            true => format!("{} minor", MINOR_KEYS[index]),
        },
        // C major and A minor are 8, each sharp moves one step clockwise; B marks major, A minor
        KeyFormat::Camelot => {
            let number = (sharps_flats as i32 + 7).rem_euclid(12) + 1;
            format!("{number}{}", if minor { 'A' } else { 'B' })
        }
    })
}

/// Quarter notes per minute from microseconds per quarter note, with at most two decimals.
fn bpm(tempo: u32) -> Option<String> {
    if tempo == 0 {
        return None;
    }
    let bpm = format!("{:.2}", 60_000_000.0 / tempo as f64);
    Some(bpm.trim_end_matches('0').trim_end_matches('.').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classic(sharps_flats: i8, minor: bool) -> String {
        describe(
            &MidiMessage::KeySignature(sharps_flats, minor),
            KeyFormat::Classic,
        )
    }

    fn camelot(sharps_flats: i8, minor: bool) -> String {
        describe(
            &MidiMessage::KeySignature(sharps_flats, minor),
            KeyFormat::Camelot,
        )
    }

    #[test]
    fn classic_key_names() {
        assert_eq!(classic(0, false), "KeySignature(C major)");
        assert_eq!(classic(0, true), "KeySignature(A minor)");
        assert_eq!(classic(-2, false), "KeySignature(Bb major)");
        assert_eq!(classic(3, true), "KeySignature(F# minor)");
        assert_eq!(classic(7, false), "KeySignature(C# major)");
        assert_eq!(classic(-7, true), "KeySignature(Ab minor)");
    }

    #[test]
    fn camelot_codes() {
        assert_eq!(camelot(0, false), "KeySignature(8B)");
        assert_eq!(camelot(0, true), "KeySignature(8A)");
        assert_eq!(camelot(1, false), "KeySignature(9B)"); // G major
        assert_eq!(camelot(4, false), "KeySignature(12B)"); // E major
        assert_eq!(camelot(5, false), "KeySignature(1B)"); // B major
        assert_eq!(camelot(-1, false), "KeySignature(7B)"); // F major
        assert_eq!(camelot(-2, true), "KeySignature(6A)"); // G minor
        assert_eq!(camelot(3, true), "KeySignature(11A)"); // F# minor

        // Cb major and B major share a position on the wheel
        assert_eq!(camelot(-7, false), camelot(5, false));
    }

    #[test]
    fn invalid_key_is_shown_as_stored() {
        assert_eq!(classic(9, false), "KeySignature(9, false)");
    }

    #[test]
    fn time_signatures() {
        let ts = |nn, dd| {
            describe(
                &MidiMessage::TimeSignature(nn, dd, 24, 8),
                KeyFormat::Classic,
            )
        };
        assert_eq!(ts(4, 2), "TimeSignature(4/4)");
        assert_eq!(ts(6, 3), "TimeSignature(6/8)");
        assert_eq!(ts(3, 1), "TimeSignature(3/2)");
        assert_eq!(ts(3, 40), "TimeSignature(3, 40, 24, 8)");
    }

    #[test]
    fn tempos() {
        let tempo = |t| describe(&MidiMessage::SetTempo(t), KeyFormat::Classic);
        assert_eq!(tempo(500_000), "SetTempo(b=120)");
        assert_eq!(tempo(833_333), "SetTempo(b=72)");
        assert_eq!(tempo(600_000), "SetTempo(b=100)");
        assert_eq!(tempo(461_538), "SetTempo(b=130)");
        assert_eq!(tempo(470_000), "SetTempo(b=127.66)");
        assert_eq!(tempo(480_000), "SetTempo(b=125)");
        assert_eq!(tempo(0), "SetTempo(0)");
    }

    #[test]
    fn other_messages_keep_their_debug_form() {
        assert_eq!(
            describe(&MidiMessage::NoteOn(0, 60, 100), KeyFormat::Camelot),
            "NoteOn(0, 60, 100)"
        );
    }
}
