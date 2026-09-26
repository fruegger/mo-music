use thiserror::Error;

pub struct MidiFile {
    pub header: MidiHeader,
    pub tracks: Vec<MidiTrack>,
}

pub struct MidiHeader {
    pub format: MidiFormat,
    pub divisions: MidiDivision,
}

/// Meaning of a delta time tick, from the header's division word.
#[derive(Debug, PartialEq)]
pub enum MidiDivision {
    /// top bit clear: ticks per quarter note
    TicksPerQuarterNote(u16),
    /// top bit set: SMPTE frames per second (24, 25, 29 = 29.97 drop frame, 30) and ticks per frame
    Smpte {
        frames_per_second: u8,
        ticks_per_frame: u8,
    },
}

pub struct MidiTrack {
    pub events: Vec<MidiTimeEvent>,
}

#[derive(Debug)]
pub enum MidiFormat {
    SingleTrack,
    MultiTrack,
    MultiSong,
}

pub struct MidiTimeEvent {
    delta_time: u32,
    message: MidiMessage,
}

#[derive(Debug)]
pub enum MidiMessage {
    NoteOff(u8, u8, u8),          // 8n - channel key velocity
    NoteOn(u8, u8, u8),           // 9n
    AfterTouch(u8, u8, u8),       // An
    ControllerChange(u8, u8, u8), // Bn 00..77 - channel controller value
    ProgramChange(u8, u8),        // Cn - channel program
    ChannelKeyPressure(u8, u8),   // Dn - channel pressure
    PitchBend(u8, u8, u8),        // En . channel lsb msb

    AllSoundOff(u8),         // Bn 78 00 - channel
    ResetAllControllers(u8), // Bn 79 00
    LocalControl(u8, bool),  // Bn 7A - channel disconnect
    AllNotesOff(u8),         // Bn 7B - channel
    OmniModeOff(u8),         // Bn 7C
    OmniModeOn(u8),          // Bn 7D
    MonoModeOn(u8, u8),      // Bn 7E - channel  nr_channel
    PolyModeOn(u8),          // Bn 7F - channel

    SysEx(Vec<u8>), // F0 or F7 - data

    SequenceNumber(u16),             // FF 00 02 - seuqence
    TextEvent(String),               // FF 01 - text
    CopyrightNotice(String),         // FF 02
    SequenceOrTrackName(String),     // FF 03
    InstrumentName(String),          // FF 04
    Lyric(String),                   // FF 05
    Marker(String),                  // FF 06,
    CuePoint(String),                // FF 07,
    ChannelPrefix(u8),               // FF 20 01 -- channel
    EndOfTrack,                      // FF 2F 00
    SetTempo(u32),                   // FF 51 03 -- tempo - only 24 bits used
    SMPTEOffset(u8, u8, u8, u8, u8), // FF 54 05 - ho mi se fr hu
    TimeSignature(u8, u8, u8, u8),   // FF 58 04 - num den cpt 32perclk
    KeySignature(i8, bool),          // FF 59 02 - sharp-flats minor
    SequenceSpecific(u32, Vec<u8>),  // FF 7F - id data
    UnknownMeta(u8, Vec<u8>),        // FF xx - type data
}

#[derive(Error, Debug)]
pub enum MidiError {
    #[error("not a midi file")]
    NotAMidiFile(),
    #[error("invalid midi header")]
    InvalidMidiHeader(),
    #[error("invalid midi track {0}")]
    InvalidMidiTrack(u16),
    #[error("invalid midi message {0}")]
    InvalidMidiMessage(u8),
    #[error("invalid meta event {0:#04x}")]
    InvalidMetaEvent(u8),
    #[error("end of buffer reached")]
    EndOfChunk(),
}

/// Reads big-endian values from a byte buffer, tracking the read position.
/// Every read is bounds checked and fails with `EndOfChunk` instead of panicking.
struct MidiReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> MidiReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        MidiReader { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn peek(&self) -> Result<u8, MidiError> {
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or(MidiError::EndOfChunk())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], MidiError> {
        let end = self.pos.checked_add(n).ok_or(MidiError::EndOfChunk())?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(MidiError::EndOfChunk())?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, MidiError> {
        let byte = self.peek()?;
        self.pos += 1;
        Ok(byte)
    }

    fn u16(&mut self) -> Result<u16, MidiError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, MidiError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads a variable length quantity (at most 4 bytes, 7 bits each).
    fn var_len(&mut self) -> Result<u32, MidiError> {
        let mut result: u32 = 0;
        for _ in 0..4 {
            let byte = self.u8()?;
            result = result << 7 | (byte & 0x7f) as u32;
            if byte & 0x80 == 0 {
                break;
            }
        }
        Ok(result)
    }
}

pub fn parse_midi(bytes: &[u8]) -> Result<MidiFile, MidiError> {
    let mut reader = MidiReader::new(bytes);

    if reader.take(4).ok() != Some(b"MThd".as_slice()) {
        return Err(MidiError::NotAMidiFile());
    }
    let header_len = reader.u32().map_err(|_| MidiError::InvalidMidiHeader())?;
    // the header is 6 bytes today; the spec asks readers to skip anything a later version appends
    if header_len < 6 {
        return Err(MidiError::InvalidMidiHeader());
    }
    let mut header = MidiReader::new(
        reader
            .take(header_len as usize)
            .map_err(|_| MidiError::InvalidMidiHeader())?,
    );
    let format = match header.u16()? {
        0 => MidiFormat::SingleTrack,
        1 => MidiFormat::MultiTrack,
        2 => MidiFormat::MultiSong,
        _ => return Err(MidiError::InvalidMidiHeader()),
    };
    let nr_tracks = header.u16()?;
    let divisions = division(header.u16()?)?;

    let mut tracks = Vec::new();
    while tracks.len() < nr_tracks as usize {
        let invalid = |_| MidiError::InvalidMidiTrack(tracks.len() as u16 + 1);
        let kind = reader.take(4).map_err(invalid)?;
        let len = reader.u32().map_err(invalid)? as usize;
        let data = reader.take(len).map_err(invalid)?;
        // the spec asks readers to skip chunk types they don't know
        if kind == b"MTrk" {
            let track = parse_track(data).map_err(invalid)?;
            tracks.push(track);
        }
    }

    Ok(MidiFile {
        header: MidiHeader { format, divisions },
        tracks,
    })
}

fn division(word: u16) -> Result<MidiDivision, MidiError> {
    if word & 0x8000 == 0 {
        return Ok(MidiDivision::TicksPerQuarterNote(word));
    }
    // the high byte holds the frame rate as a negative two's complement number
    let [fps, ticks_per_frame] = word.to_be_bytes();
    let frames_per_second = (fps as i8).unsigned_abs();
    match frames_per_second {
        24 | 25 | 29 | 30 => Ok(MidiDivision::Smpte {
            frames_per_second,
            ticks_per_frame,
        }),
        _ => Err(MidiError::InvalidMidiHeader()),
    }
}

/// Parses the body of an MTrk chunk; having only the chunk's bytes, events cannot run into the next chunk.
fn parse_track(bytes: &[u8]) -> Result<MidiTrack, MidiError> {
    let mut chunk = MidiReader::new(bytes);

    let mut events = Vec::new();
    let mut running_status = None;
    while !chunk.is_empty() {
        let delta_time = chunk.var_len()?;
        let message = event(&mut chunk, &mut running_status)?;
        events.push(MidiTimeEvent {
            delta_time,
            message,
        });
    }
    Ok(MidiTrack { events })
}

pub fn print_midi(midi: MidiFile) {
    println!("header[");
    println!(" format:{:?}", midi.header.format);
    println!(" divisions:{:?}", midi.header.divisions);
    println!(" tracks:({}) [", midi.tracks.len());
    for track in &midi.tracks {
        println!("  track[");
        for event in &track.events {
            println!("    {} - {:?}", event.delta_time, event.message);
        }
        println!("  ]");
    }
    println!(" ]");
    println!("]");
}

/// Reads one event. `running_status` is the last channel status byte of the track;
/// it is reused when an event starts with a data byte and cleared by sysex and meta events.
fn event(
    reader: &mut MidiReader,
    running_status: &mut Option<u8>,
) -> Result<MidiMessage, MidiError> {
    let first = reader.peek()?;
    let status = if first < 0x80 {
        running_status.ok_or(MidiError::InvalidMidiMessage(first))?
    } else {
        reader.u8()?
    };

    match status {
        0x80..=0xef => {
            *running_status = Some(status);
            channel_message(reader, status)
        }
        0xf0 | 0xf7 => {
            *running_status = None;
            let len = reader.var_len()? as usize;
            Ok(MidiMessage::SysEx(reader.take(len)?.to_vec()))
        }
        0xff => {
            *running_status = None;
            meta_event(reader)
        }
        _ => Err(MidiError::InvalidMidiMessage(status)),
    }
}

fn channel_message(reader: &mut MidiReader, status: u8) -> Result<MidiMessage, MidiError> {
    let channel = status & 0x0f;
    let message = match status >> 4 {
        0x8 => MidiMessage::NoteOff(channel, reader.u8()?, reader.u8()?),
        0x9 => MidiMessage::NoteOn(channel, reader.u8()?, reader.u8()?),
        0xa => MidiMessage::AfterTouch(channel, reader.u8()?, reader.u8()?),
        0xb => {
            let controller = reader.u8()?;
            let value = reader.u8()?;
            match controller {
                0x78 => MidiMessage::AllSoundOff(channel),
                0x79 => MidiMessage::ResetAllControllers(channel),
                0x7a => MidiMessage::LocalControl(channel, value != 0),
                0x7b => MidiMessage::AllNotesOff(channel),
                0x7c => MidiMessage::OmniModeOff(channel),
                0x7d => MidiMessage::OmniModeOn(channel),
                0x7e => MidiMessage::MonoModeOn(channel, value),
                0x7f => MidiMessage::PolyModeOn(channel),
                _ => MidiMessage::ControllerChange(channel, controller, value),
            }
        }
        0xc => MidiMessage::ProgramChange(channel, reader.u8()?),
        0xd => MidiMessage::ChannelKeyPressure(channel, reader.u8()?),
        0xe => MidiMessage::PitchBend(channel, reader.u8()?, reader.u8()?),
        _ => return Err(MidiError::InvalidMidiMessage(status)),
    };
    Ok(message)
}

/// Reads a meta event (FF type len data); the leading FF is already consumed.
fn meta_event(reader: &mut MidiReader) -> Result<MidiMessage, MidiError> {
    let kind = reader.u8()?;
    let len = reader.var_len()? as usize;
    let data = reader.take(len)?;
    let text = || String::from_utf8_lossy(data).into_owned();
    let invalid = || MidiError::InvalidMetaEvent(kind);

    let message = match kind {
        0x00 => {
            let [hi, lo]: [u8; 2] = data.try_into().map_err(|_| invalid())?;
            MidiMessage::SequenceNumber(u16::from_be_bytes([hi, lo]))
        }
        0x01 => MidiMessage::TextEvent(text()),
        0x02 => MidiMessage::CopyrightNotice(text()),
        0x03 => MidiMessage::SequenceOrTrackName(text()),
        0x04 => MidiMessage::InstrumentName(text()),
        0x05 => MidiMessage::Lyric(text()),
        0x06 => MidiMessage::Marker(text()),
        0x07 => MidiMessage::CuePoint(text()),
        0x20 => {
            let [channel]: [u8; 1] = data.try_into().map_err(|_| invalid())?;
            MidiMessage::ChannelPrefix(channel)
        }
        0x2f if data.is_empty() => MidiMessage::EndOfTrack,
        0x51 => {
            let [a, b, c]: [u8; 3] = data.try_into().map_err(|_| invalid())?;
            MidiMessage::SetTempo(u32::from_be_bytes([0, a, b, c]))
        }
        0x54 => {
            let [hr, mn, se, fr, ff]: [u8; 5] = data.try_into().map_err(|_| invalid())?;
            MidiMessage::SMPTEOffset(hr, mn, se, fr, ff)
        }
        0x58 => {
            let [nn, dd, cc, bb]: [u8; 4] = data.try_into().map_err(|_| invalid())?;
            MidiMessage::TimeSignature(nn, dd, cc, bb)
        }
        0x59 => {
            let [sf, mi]: [u8; 2] = data.try_into().map_err(|_| invalid())?;
            MidiMessage::KeySignature(sf as i8, mi != 0)
        }
        0x7f => {
            // manufacturer id is one byte, or three bytes when the first one is 00
            let id_len = if data.first() == Some(&0) { 3 } else { 1 };
            if data.len() < id_len {
                return Err(invalid());
            }
            let id = data[..id_len]
                .iter()
                .fold(0u32, |id, &b| id << 8 | b as u32);
            MidiMessage::SequenceSpecific(id, data[id_len..].to_vec())
        }
        0x2f => return Err(invalid()),
        // the spec asks readers to ignore meta events they don't know
        _ => MidiMessage::UnknownMeta(kind, data.to_vec()),
    };
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_len_decodes_spec_examples() {
        let cases: [(u32, &[u8]); 12] = [
            (0x00000000, &[0x00]),
            (0x00000040, &[0x40]),
            (0x0000007F, &[0x7F]),
            (0x00000080, &[0x81, 0x00]),
            (0x00002000, &[0xC0, 0x00]),
            (0x00003FFF, &[0xFF, 0x7F]),
            (0x00004000, &[0x81, 0x80, 0x00]),
            (0x00100000, &[0xC0, 0x80, 0x00]),
            (0x001FFFFF, &[0xFF, 0xFF, 0x7F]),
            (0x00200000, &[0x81, 0x80, 0x80, 0x00]),
            (0x08000000, &[0xC0, 0x80, 0x80, 0x00]),
            (0x0FFFFFFF, &[0xFF, 0xFF, 0xFF, 0x7F]),
        ];
        for (expected, encoded) in cases {
            // trailing byte checks that decoding stops at the right position
            let mut bytes = encoded.to_vec();
            bytes.push(0x55);
            let mut reader = MidiReader::new(&bytes);
            assert_eq!(
                reader.var_len().unwrap(),
                expected,
                "decoding {:02X?}",
                encoded
            );
            assert_eq!(reader.pos, encoded.len(), "position after {:02X?}", encoded);
        }
    }

    #[test]
    fn reader_fails_at_end_of_buffer() {
        let mut reader = MidiReader::new(&[0x01, 0x02, 0x03]);
        assert!(matches!(reader.u32(), Err(MidiError::EndOfChunk())));
        assert_eq!(reader.u16().unwrap(), 0x0102);
        assert!(matches!(reader.u16(), Err(MidiError::EndOfChunk())));
        assert!(matches!(
            MidiReader::new(&[0x81]).var_len(),
            Err(MidiError::EndOfChunk())
        ));
    }

    #[test]
    fn parses_minimal_file() {
        let bytes = [
            b'M', b'T', b'h', b'd', 0, 0, 0, 6, 0, 0, 0, 1, 0, 96, b'M', b'T', b'r', b'k', 0, 0, 0,
            9, 0x00, 0x90, 60, 100, 0x81, 0x00, 0x80, 60, 0,
        ];
        let midi = parse_midi(&bytes).unwrap();
        assert_eq!(midi.header.divisions, MidiDivision::TicksPerQuarterNote(96));
        let events = &midi.tracks[0].events;
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].delta_time, 128);
        assert!(matches!(events[1].message, MidiMessage::NoteOff(0, 60, 0)));
    }

    /// header with the given length field and body, followed by one empty track
    fn file_with_header(header_len: u32, body: &[u8]) -> Vec<u8> {
        let mut bytes = b"MThd".to_vec();
        bytes.extend(header_len.to_be_bytes());
        bytes.extend(body);
        bytes.extend(b"MTrk\0\0\0\x04\x00\xff\x2f\x00");
        bytes
    }

    #[test]
    fn longer_header_is_skipped() {
        let bytes = file_with_header(8, &[0, 0, 0, 1, 0, 96, 0xaa, 0xbb]);
        let midi = parse_midi(&bytes).unwrap();
        assert_eq!(midi.header.divisions, MidiDivision::TicksPerQuarterNote(96));
        assert!(matches!(
            midi.tracks[0].events[0].message,
            MidiMessage::EndOfTrack
        ));
    }

    #[test]
    fn unknown_chunks_are_skipped() {
        let mut bytes = file_with_header(6, &[0, 1, 0, 2, 0, 96]);
        // an unknown chunk between the tracks, and one after the last track
        bytes.extend(b"XFIH\0\0\0\x03abc");
        bytes.extend(b"MTrk\0\0\0\x04\x00\xff\x2f\x00");
        bytes.extend(b"XFKM\0\0\0\x00");
        let midi = parse_midi(&bytes).unwrap();
        assert_eq!(midi.tracks.len(), 2);
        assert!(matches!(
            midi.tracks[1].events[0].message,
            MidiMessage::EndOfTrack
        ));
    }

    #[test]
    fn missing_track_is_an_error() {
        let mut bytes = file_with_header(6, &[0, 1, 0, 2, 0, 96]);
        bytes.extend(b"XFIH\0\0\0\x03abc");
        assert!(matches!(
            parse_midi(&bytes),
            Err(MidiError::InvalidMidiTrack(2))
        ));
    }

    #[test]
    fn short_header_is_an_error() {
        let bytes = file_with_header(4, &[0, 0, 0, 1]);
        assert!(matches!(
            parse_midi(&bytes),
            Err(MidiError::InvalidMidiHeader())
        ));
    }

    #[test]
    fn smpte_divisions() {
        // E7 = -25 fps, 40 ticks per frame (millisecond resolution)
        let bytes = file_with_header(6, &[0, 0, 0, 1, 0xe7, 40]);
        assert_eq!(
            parse_midi(&bytes).unwrap().header.divisions,
            MidiDivision::Smpte {
                frames_per_second: 25,
                ticks_per_frame: 40
            }
        );
        // E3 = -29 fps (29.97 drop frame)
        assert_eq!(
            division(0xe350).unwrap(),
            MidiDivision::Smpte {
                frames_per_second: 29,
                ticks_per_frame: 80
            }
        );
        // -20 is not a SMPTE frame rate
        assert!(matches!(
            division(0xec04),
            Err(MidiError::InvalidMidiHeader())
        ));
    }

    #[test]
    fn truncated_track_is_an_error() {
        let bytes = [
            b'M', b'T', b'h', b'd', 0, 0, 0, 6, 0, 0, 0, 1, 0, 96, b'M', b'T', b'r', b'k', 0, 0, 0,
            9, 0x00, 0x90,
        ];
        assert!(matches!(
            parse_midi(&bytes),
            Err(MidiError::InvalidMidiTrack(1))
        ));
    }

    fn parse_events(bytes: &[u8]) -> Result<Vec<MidiMessage>, MidiError> {
        let mut reader = MidiReader::new(bytes);
        let mut running_status = None;
        let mut messages = Vec::new();
        while !reader.is_empty() {
            messages.push(event(&mut reader, &mut running_status)?);
        }
        Ok(messages)
    }

    #[test]
    fn running_status_reuses_last_channel_status() {
        let messages = parse_events(&[0x91, 60, 100, 62, 90, 0xc2, 5, 7]).unwrap();
        assert!(matches!(messages[0], MidiMessage::NoteOn(1, 60, 100)));
        assert!(matches!(messages[1], MidiMessage::NoteOn(1, 62, 90)));
        assert!(matches!(messages[2], MidiMessage::ProgramChange(2, 5)));
        assert!(matches!(messages[3], MidiMessage::ProgramChange(2, 7)));
    }

    #[test]
    fn meta_and_sysex_events_cancel_running_status() {
        assert!(matches!(
            parse_events(&[60, 100]),
            Err(MidiError::InvalidMidiMessage(60))
        ));
        assert!(matches!(
            parse_events(&[0x90, 60, 100, 0xff, 0x2f, 0x00, 60, 100]),
            Err(MidiError::InvalidMidiMessage(60))
        ));
        assert!(matches!(
            parse_events(&[0x90, 60, 100, 0xf0, 0x01, 0xf7, 60, 100]),
            Err(MidiError::InvalidMidiMessage(60))
        ));
    }

    #[test]
    fn controller_messages_and_channel_modes() {
        let messages = parse_events(&[0xb3, 0x07, 100, 0xb3, 0x7b, 0, 0xb3, 0x7a, 127]).unwrap();
        assert!(matches!(
            messages[0],
            MidiMessage::ControllerChange(3, 0x07, 100)
        ));
        assert!(matches!(messages[1], MidiMessage::AllNotesOff(3)));
        assert!(matches!(messages[2], MidiMessage::LocalControl(3, true)));
    }

    #[test]
    fn meta_events() {
        let messages = parse_events(&[
            0xff, 0x03, 0x04, b'B', b'a', b's', b's', 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20, 0xff,
            0x58, 0x04, 6, 3, 24, 8, 0xff, 0x59, 0x02, 0xfe, 1, 0xff, 0x7f, 0x04, 0x00, 0x00, 0x41,
            0x99, 0xff, 0x21, 0x01, 0x02, 0xff, 0x2f, 0x00,
        ])
        .unwrap();
        assert!(matches!(&messages[0], MidiMessage::SequenceOrTrackName(name) if name == "Bass"));
        assert!(matches!(messages[1], MidiMessage::SetTempo(500_000)));
        assert!(matches!(
            messages[2],
            MidiMessage::TimeSignature(6, 3, 24, 8)
        ));
        assert!(matches!(messages[3], MidiMessage::KeySignature(-2, true)));
        assert!(
            matches!(&messages[4], MidiMessage::SequenceSpecific(0x41, data) if data == &[0x99])
        );
        assert!(matches!(&messages[5], MidiMessage::UnknownMeta(0x21, data) if data == &[0x02]));
        assert!(matches!(messages[6], MidiMessage::EndOfTrack));
    }

    #[test]
    fn meta_event_with_wrong_length_is_an_error() {
        assert!(matches!(
            parse_events(&[0xff, 0x51, 0x02, 0x07, 0xa1]),
            Err(MidiError::InvalidMetaEvent(0x51))
        ));
    }
}
