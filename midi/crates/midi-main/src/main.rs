use clap::{ArgGroup, Parser};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
mod midi;

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

    match midi::parse_midi(&contents) {
        Ok(m) => {
            midi::print_midi(m);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("midi: error: {}: {e}", file.display());
            ExitCode::FAILURE
        }
    }
}
