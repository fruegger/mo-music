//! `audan tag` (F7): compute whatever `--write` asks for and write it back
//! via `audan-io::write_tags`. BPM comes from the (cached) beat grid; key
//! from key estimation.

use std::path::Path;

use audan_cache::Resolver;

use crate::config::Resolved;
use crate::pipeline;

pub fn run(file: &Path, write: &[String], resolved: &Resolved) -> anyhow::Result<()> {
    let want_bpm = write.iter().any(|f| f.eq_ignore_ascii_case("bpm"));
    let want_key = write.iter().any(|f| f.eq_ignore_ascii_case("key"));
    if !want_bpm && !want_key {
        return Err(audan_core::AudanError::Usage(format!(
            "`audan tag --write` requires bpm and/or key, got: {write:?}"
        ))
        .into());
    }

    let resolver = Resolver::open(&resolved.cache_root)?;
    let (l0_key, signal) = pipeline::resolve_l0(&resolver, file)?;
    let (l1_key, mono) = pipeline::resolve_l1_analysis(&resolver, &l0_key, &signal)?;

    let mut update = audan_io::TagUpdate::default();
    if want_bpm {
        let (_beats_key, grid) = pipeline::resolve_beats(
            &resolver,
            &l1_key,
            &mono,
            None,
            pipeline::expected_beats_frames(),
        )?;
        update.bpm = Some(grid.tempo.median_bpm);
    }
    if want_key {
        let params = audan_dsp::KeyChromaParams {
            pad: resolved.pad,
            ..Default::default()
        };
        let ranked = audan_key::estimate_key_from_signal(&mono, &params);
        update.key = Some(ranked.top().value.camelot());
    }

    audan_io::write_tags(file, &update)?;
    println!(
        "wrote tags to {}: bpm={:?} key={:?}",
        file.display(),
        update.bpm,
        update.key
    );
    Ok(())
}
