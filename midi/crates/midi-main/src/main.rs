use clap::Parser;
use std::fs;

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
    println!("file: {}", file);

    let contents = fs::read_to_string(&file)
            .expect("reading file {file} failed");
    println!("contents: {}", contents);

    let result : Result<(),std::io::Error> = Ok(());
    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("midi: error: {e:#}");
            std::process::exit(1);
        }
    }
}
