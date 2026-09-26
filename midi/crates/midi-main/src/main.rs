use clap::Parser;
use std::fs;
mod midi;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]

struct Args {
    /// Name of the midi file to read
    #[arg(short, long)]
    file: Option<String>,

    /// Name of the midi file to read (positional form of --file)
    file_positional: Option<String>,
}

fn main() {
    let args = Args::parse();
    let file = args
        .file
        .or(args.file_positional)
        .expect("either --file or a positional FILE argument is required");

    let contents = fs::read(&file)
            .expect("reading file {file} failed");

    let result  = midi::parse_midi(&contents);
    match result {
        Ok(m) => {
            midi::print_midi(m);
            std::process::exit(0);
            }
        Err(e) => {
            eprintln!("midi: error: {e:#}");
            std::process::exit(1);
        }
    }
}
