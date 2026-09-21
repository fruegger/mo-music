//! `audan-cli`: subcommand dispatch, config resolution, cache wiring,
//! TTY-vs-pipe rendering, and exit codes (S5.1's `audan-cli` row). The only
//! crate in the workspace allowed to depend on `anyhow` (S2.3).
//!
//! Exposed as a library (in addition to the `audan` binary in `main.rs`) so
//! integration tests can call [`run`] and the individual `commands::*::run`
//! functions directly instead of shelling out to a compiled binary.

pub mod beats_backend;
pub mod cache_paths;
pub mod cli;
pub mod commands;
pub mod config;
pub mod license_gate;
pub mod pipeline;
pub mod render;

use cli::{Cli, Command};

/// Resolves configuration once, then dispatches to the matching command
/// module. Returns `anyhow::Error` rather than `audan_core::AudanError`
/// directly so command bodies can use `?` freely against any error type;
/// `main` recovers the precise exit code (S8.9) by downcasting to
/// `audan_core::AudanError` where a command constructed one deliberately.
pub fn run(cli: Cli) -> anyhow::Result<()> {
    let resolved = config::resolve_all(&cli)?;

    match &cli.command {
        Command::Probe { file } => commands::probe::run(file, cli.format, cli.quiet),
        Command::Beats { files, click, model, quick } => {
            commands::beats::run(files, click.as_deref(), model.as_deref(), *quick, &cli, &resolved)
        }
        Command::Key { file } => commands::key::run(file, &cli, &resolved),
        Command::Chords { file, beats } => {
            commands::chords::run(file, beats.as_deref(), &cli, &resolved)
        }
        Command::Struct { file, beats } => {
            commands::struct_cmd::run(file, beats.as_deref(), &cli, &resolved)
        }
        Command::Stems { file, model } => commands::stems::run(file, model, &cli, &resolved),
        Command::Tag { file, write } => commands::tag::run(file, write, &cli, &resolved),
        Command::Cache { action } => commands::cache::run(action, &resolved),
    }
}
