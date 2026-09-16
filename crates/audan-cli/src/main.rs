//! The `audan` binary (ADR-2: one binary, subcommands). Parses arguments,
//! runs the matching command, and maps errors to the exit codes fixed by
//! S8.9: 0 success, 1 runtime error, 2 usage error, 3 low confidence under
//! `--strict`, 4 unsupported format. `clap` handles its own usage errors
//! (missing/invalid arguments) with exit code 2 before `run` is ever called.

use clap::Parser;

fn main() {
    let cli = audan_cli::cli::Cli::parse();
    match audan_cli::run(cli) {
        Ok(()) => std::process::exit(audan_core::ExitCode::Success as i32),
        Err(e) => {
            eprintln!("audan: error: {e:#}");
            let code = e
                .downcast_ref::<audan_core::AudanError>()
                .map(audan_core::AudanError::exit_code)
                .unwrap_or(audan_core::ExitCode::RuntimeError);
            std::process::exit(code as i32);
        }
    }
}
