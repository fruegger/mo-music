//! `audan tag` (F7): compute whatever `--write` asks for and write it back
//! via `audan-io::write_tags`. BPM comes from the (cached) beat grid; key
//! from key estimation.

use std::path::Path;

use audan_cache::Resolver;

use crate::beats_backend;
use crate::cli::Cli;
use crate::config::Resolved;
use crate::pipeline;
use crate::progress::{Progress, Stage, StageReporter};

pub fn run(file: &Path, write: &[String], cli: &Cli, resolved: &Resolved) -> anyhow::Result<()> {
    let want_bpm = write.iter().any(|f| f.eq_ignore_ascii_case("bpm"));
    let want_key = write.iter().any(|f| f.eq_ignore_ascii_case("key"));
    if !want_bpm && !want_key {
        return Err(audan_core::AudanError::Usage(format!(
            "`audan tag --write` requires bpm and/or key, got: {write:?}"
        ))
        .into());
    }

    // No `--model` of its own; uses the configured default backend, see
    // `commands::chords::run`. Resolved before progress starts, since it may
    // prompt for a model license on stderr.
    let backend = if want_bpm {
        Some(beats_backend::resolve(None, cli, resolved)?)
    } else {
        None
    };

    let progress = Progress::start(1, Progress::detect(cli.quiet));
    let task = progress.task(file);
    let resolver = Resolver::open(&resolved.cache_root)?;
    let l0_key = pipeline::l0_key(file, &task)?;
    let (l1_key, mono) = pipeline::resolve_l1_analysis(&resolver, &l0_key, file, &task)?;

    let mut update = audan_io::TagUpdate::default();
    if let Some(backend) = &backend {
        let (_beats_key, grid) = pipeline::resolve_beats(
            &resolver,
            &l1_key,
            &mono,
            None,
            pipeline::expected_beats_frames(backend.as_dyn()),
            backend.as_dyn(),
            &task,
        )?;
        update.bpm = Some(grid.tempo.median_bpm);
    }
    if want_key {
        let params = audan_dsp::KeyChromaParams {
            pad: resolved.pad,
            ..Default::default()
        };
        task.stage(Stage::Key);
        let ranked = audan_key::estimate_key_from_signal(&mono, &params);
        update.key = Some(ranked.top().value.camelot());
    }
    task.finish(None);
    progress.finish();

    audan_io::write_tags(file, &update)?;
    println!(
        "wrote tags to {}: bpm={:?} key={:?}",
        file.display(),
        update.bpm,
        update.key
    );
    Ok(())
}
