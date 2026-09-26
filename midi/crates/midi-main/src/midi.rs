use thiserror::Error;
use std::str;

pub struct MidiFile {
    pub header : MidiHeader,
    pub tracks : Vec<MidiTrack>
}

pub struct MidiHeader {
    pub format : MidiFormat,
    pub divisions : i16
}

pub struct MidiTrack {
    pub events : Vec<MidiTimeEvent>
}


#[derive(Debug)]
pub enum MidiFormat {
    SingleTrack,
    MultiTrack,
    MultiSong
}

pub struct MidiTimeEvent {
    delta_time : u32,
    message : MidiMessage
    }

#[derive(Debug)]
pub enum MidiMessage {
    NoteOff(u8, u8, u8), // 8n - channel key velocity
    NoteOn(u8, u8, u8), // 9n
    AfterTouch(u8, u8, u8), // An
    ControllerChange(u8, u8, u8), // Bn 00..77 - channel controller value
    ProgramChange(u8, u8), // Cn - channel program
    ChannelKeyPressure(u8, u8), // Dn - channel pressure
    PitchBend(u8, u8, u8), // En . channel lsb msb

    AllSoundOff(u8), // Bn 78 00 - channel
    ResetAllControllers(u8), // Bn 79 00
    LocalControl(u8, bool), // Bn 7A - channel disconnect
    AllNotesOff(u8), // Bn 7B - channel
    OmniModeOff(u8), // Bn 7C
    OmniModeOn(u8), // Bn 7D
    MonoModeOn(u8,u8), // Bn 7E - channel  nr_channel
    PolyModeOn(u8), // Bn 7F - channel

    SysEx(Vec<u8>), // F0 or F7 - data

    SequenceNumber(u16), // FF 00 02 - seuqence
    TextEvent(String), // FF 01 - text
    CopyrightNotice(String), // FF 02
    SequenceOrTrackName(String), // FF 03
    InstrumentName(String), // FF 04
    Lyric(String), // FF 05
    Marker(String), // FF 06,
    CuePoint(String), // FF 07,
    ChannelPrefix(u8), // FF 20 01 -- channel
    EndOfTrack, // FF 2F 00
    SetTempo(u32), // FF 51 03 -- tempo - only 24 bits used
    SMTPEOffset(u8, u8, u8, u8, u8), // FF 54 05 - ho mi se fr hu
    TimeSignature(u8, u8, u8, u8), // FF 58 04 - num den cpt 32perclk
    KeySignature(i8, bool), // FF 59 03 - sharp-flats minor
    SequenceSpecific(u32, Vec<u8>) // FF 7F - id data
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
     #[error("end of buffer reached")]
    EndOfChunk(),
}

pub fn parse_midi(bytes : &Vec<u8>) -> Result<MidiFile, MidiError> {
    let mut read_pos : usize = 0;

    if bytes.len()<14 || chunk_type(bytes,&mut read_pos)!="MThd" {
         return Err(MidiError::NotAMidiFile())
        }
    if chunk_int4(&bytes,&mut read_pos)!= 6 {
         return Err(MidiError::InvalidMidiHeader())
        }
    let format = match chunk_int2(&bytes,&mut read_pos) {
                             0 => MidiFormat::SingleTrack,
                             1 => MidiFormat::MultiTrack,
                             2 => MidiFormat::MultiSong,
                             _ => return Err(MidiError::InvalidMidiHeader())
                         };
    let nr_tracks = chunk_int2(&bytes, &mut read_pos);
    let divisions = chunk_int2(&bytes, &mut read_pos) as i16;

    let mut tracks = Vec::new();
    let mut read_pos = 14;
    for track in 0..nr_tracks {
        if bytes.len()<read_pos || chunk_type(bytes,&mut read_pos)!="MTrk" {
           return Err(MidiError::InvalidMidiTrack(track+1))
           }
        let chunk_sz = chunk_int4(&bytes,&mut read_pos);
        let mut events : Vec<MidiTimeEvent> = Vec::new();
println!("track:{}, sz:{}", track, chunk_sz);

        let mut read_pos2 = read_pos;
        while read_pos2-read_pos<chunk_sz as usize {
            let event_delta = var_length(&bytes, &mut read_pos2);
            let event_result = event(&bytes, &mut read_pos2);
            match event_result {
                Ok(message) => {
                    events.push(
                        MidiTimeEvent {
                            delta_time : event_delta,
                            message : message
                        }
                    );
                },
                Err(e) => {
                    println!("{}",e);
//                *pos +=1
                    }
                }
            }
        read_pos+=chunk_sz as usize;
            tracks.push(
                MidiTrack {
                   events : events
                   }
                );
    }

    let header = MidiHeader {
        format : format,
        divisions : divisions
    };
    let result = MidiFile {
        header : header,
        tracks: tracks
    };
    return Result::Ok(result)
}

pub fn print_midi(midi : MidiFile) {
    println!("header[");
    println!(" format:{:?}", midi.header.format);
    println!(" divisions:{}", midi.header.divisions);
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

fn chunk_type(bytes : &Vec<u8>, pos : & mut usize) ->  String {
    let p : usize = *pos;
    let result = str::from_utf8(&bytes[p..p+4]).unwrap().to_string();
    *pos+=4;
    return result;
    }

fn chunk_int4(bytes : &Vec<u8>, pos : &mut usize) -> u32 {
    let p : usize = *pos;
    let result = (bytes[p] as u32) <<24 | (bytes[p+1] as u32) <<16 | (bytes[p+2] as u32) <<8 | bytes[p+3] as u32;
    *pos+=4;
    return result;
    }

fn chunk_int2(bytes : &Vec<u8>, pos : &mut usize) -> u16 {
    let p : usize = *pos;
    let result = (bytes[p] as u16) <<8 | bytes[p+1] as u16;
    *pos+=2;
    return result;
    }

fn var_length(bytes : &Vec<u8>, pos : &mut usize) -> u32 {
    let mut result : u32 = 0;
    let mut byte = bytes[*pos];
    while byte>0x7f {
        result = result << 7 | byte as u32;
        *pos+=1;
        byte = bytes[*pos];
    }
    result = result | byte as u32;
    *pos+=1;
println!("var_len : {}", result);
    return result;
}

fn event(bytes : &Vec<u8>, pos : &mut usize) -> Result<MidiMessage, MidiError> {
    if *pos>=bytes.len() {
        return Err(MidiError::EndOfChunk())
    }

    println!("op_code : {}", bytes[*pos]);

    let op_code = bytes[*pos] >>4;
    if (op_code >=0x8 && op_code <=0xa) || op_code== 0xe {
        let channel = bytes[*pos] & 0x0f;
        *pos+=1;
        let val1 = bytes[*pos];
        *pos+=1;
        let val2 = bytes[*pos];

        let message = match op_code {
            0x8 => MidiMessage::NoteOff(channel, val1, val2),
            0x9 => MidiMessage::NoteOn(channel, val1, val2),
            0xa => MidiMessage::AfterTouch(channel, val1, val2),
            0xe => MidiMessage::PitchBend(channel, val1, val2),
            _ => return Err(MidiError::InvalidMidiMessage(bytes[*pos]))
            };
        *pos+=1;
        return Result::Ok(message);
    } else {
        let message = match bytes[*pos] {
//test only
            0xff => MidiMessage::EndOfTrack,
            _ => return Err(MidiError::InvalidMidiMessage(bytes[*pos]))
        };
        *pos+=1;
        return Result::Ok(message);
     }
}

/* var len test data
00000000	00
00000040	40
0000007F	7F
00000080	81 00
00002000	C0 00
00003FFF	FF 7F
00004000	81 80 00
00100000	C0 80 00
001FFFFF	FF FF 7F
00200000	81 80 80 00
08000000	C0 80 80 00
0FFFFFFF	FF FF FF 7F
*/