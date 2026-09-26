use clap::{ArgGroup, Parser, ValueEnum};
use describe::{describe, KeyFormat};
use midi::{MidiDivision, MidiFile, MidiFormat};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use timeline::Timeline;
mod describe;
mod midi;
mod timeline;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(group(ArgGroup::new("input").required(true).args(["file", "file_positional"])))]
struct Args {
    /// Name of the midi file to read
    #[arg(short, long, value_name = "FILE")]
    file: Option<PathBuf>,

    /// Name of the midi file to read (positional form of --file)
    #[arg(value_name = "FILE")]
    file_positional: Option<PathBuf>,

    /// How to show the time of each event
    #[arg(short, long, value_enum, default_value_t = TimeFormat::Clock)]
    time: TimeFormat,

    /// How to show key signatures
    #[arg(short, long, value_enum, default_value_t = KeyFormat::Classic)]
    key: KeyFormat,

    /// Show time signatures, key signatures and tempos as stored in the file
    #[arg(long, conflicts_with = "key")]
    raw: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum TimeFormat {
    /// ticks since the previous event, as stored in the file
    Delta,
    /// absolute bars:beats:hundredths, following the time signature events
    Bars,
    /// absolute minutes:seconds:hundredths, following the tempo events
    Clock,
}

fn main() -> ExitCode {
    let args = Args::parse();
    // the "input" group guarantees exactly one of the two is set
    let file = args.file.or(args.file_positional).unwrap();

    let contents = match fs::read(&file) {
        Ok(contents) => contents,
        Err(e) => {
            eprintln!("midi: error: cannot read {}: {e}", file.display());
            return ExitCode::FAILURE;
        }
    };

    let midi = match midi::parse_midi(&contents) {
        Ok(midi) => midi,
        Err(e) => {
            eprintln!("midi: error: {}: {e}", file.display());
            return ExitCode::FAILURE;
        }
    };

    if matches!(args.time, TimeFormat::Bars)
        && matches!(midi.header.divisions, MidiDivision::Smpte { .. })
    {
        eprintln!(
            "midi: error: {}: --time bars needs a file timed in ticks per quarter note, this one uses SMPTE frames",
            file.display()
        );
        return ExitCode::FAILURE;
    }

    let key = (!args.raw).then_some(args.key);
    print_midi(&midi, args.time, key);
    ExitCode::SUCCESS
}

/// `key` None prints every message as stored (--raw)
fn print_midi(midi: &MidiFile, time: TimeFormat, key: Option<KeyFormat>) {
    let divisions = midi.header.divisions;
    // format 2 tracks are independent songs; in formats 0 and 1 tempo and meter changes apply to all tracks
    let song_timeline = match midi.header.format {
        MidiFormat::MultiSong => None,
        _ => Some(Timeline::new(
            divisions,
            midi.tracks
                .iter()
                .flat_map(|track| track.timed_events())
                .map(|(tick, event)| (tick, &event.message)),
        )),
    };

    println!("header[");
    println!(" format:{:?}", midi.header.format);
    println!(" divisions:{:?}", divisions);
    println!(" tracks:({}) [", midi.tracks.len());
    for track in &midi.tracks {
        let track_timeline;
        let timeline = match &song_timeline {
            Some(timeline) => timeline,
            None => {
                track_timeline = Timeline::new(
                    divisions,
                    track
                        .timed_events()
                        .map(|(tick, event)| (tick, &event.message)),
                );
                &track_timeline
            }
        };

        println!("  track[");
        for (tick, event) in track.timed_events() {
            let when = match time {
                TimeFormat::Delta => event.delta_time.to_string(),
                // SMPTE files are rejected for bars in main
                TimeFormat::Bars => timeline.bars(tick).unwrap_or_default(),
                TimeFormat::Clock => timeline.clock(tick),
            };
            let what = match key {
                Some(key) => describe(&event.message, key),
                None => format!("{:?}", event.message),
            };
            println!("    {} - {}", when, what);
        }
        println!("  ]");
    }
    println!(" ]");
    println!("]");
}
