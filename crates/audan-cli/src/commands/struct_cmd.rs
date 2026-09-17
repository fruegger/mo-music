//! `audan struct` (F4): structural segmentation, resolving the beat grid
//! through the same L3 key as `audan beats`/`audan chords` (RV2), then
//! segmentation cached at L4. `--beats <grid.json>` overrides L3 (RV5).
//! Named `struct_cmd` rather than `struct` because the latter is a Rust
//! keyword.

use std::path::Path;

use audan_cache::Resolver;

use crate::beats_backend;
use crate::cli::Cli;
use crate::config::Resolved;
use crate::pipeline;
use crate::render::{effective_format, print_json, OutputFormat};

pub fn run(
    file: &Path,
    beats_override: Option<&Path>,
    cli: &Cli,
    resolved: &Resolved,
) -> anyhow::Result<()> {
    // See `commands::chords::run`: `struct` also has no `--model` of its
    // own and uses the configured default backend.
    let backend = beats_backend::resolve(None, cli, resolved)?;

    let resolver = Resolver::open(&resolved.cache_root)?;
    let (l0_key, signal) = pipeline::resolve_l0(&resolver, file)?;
    let (l1_key, mono) = pipeline::resolve_l1_analysis(&resolver, &l0_key, &signal)?;
    let (beats_key, grid) = pipeline::resolve_beats(
        &resolver,
        &l1_key,
        &mono,
        beats_override,
        pipeline::expected_beats_frames(backend.as_dyn()),
        backend.as_dyn(),
    )?;

    let key_params = audan_dsp::KeyChromaParams {
        pad: resolved.pad,
        ..Default::default()
    };

    let result =
        pipeline::resolve_struct(&resolver, &l1_key, &beats_key, &mono, &grid, &key_params)?;

    if cli.quiet {
        println!("{} sections", result.sections.len());
        return Ok(());
    }

    // Display labels (A, B, C, ...) are a render-time convenience only:
    // `SectionEvent::label` upstream is always `None` (RISK-8 -- real
    // section-*type* labelling, "this is the chorus", is explicitly out of
    // scope). This just gives unlabeled boundaries a readable name in
    // `.lab`/Audacity output, not real labelling logic.
    let intervals: Vec<audan_format::LabelInterval> = result
        .sections
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let start = grid.beat_time(s.start_beat).unwrap_or(0.0);
            let end = pipeline::beat_end_time(&grid, s.end_beat);
            audan_format::LabelInterval {
                start,
                end,
                label: display_label(i),
            }
        })
        .collect();

    match effective_format(cli.format) {
        OutputFormat::Table => {
            for iv in &intervals {
                println!("{:>8.3} - {:<8.3} {}", iv.start, iv.end, iv.label);
            }
        }
        OutputFormat::Lab => print!("{}", audan_format::write_lab(&intervals)),
        OutputFormat::Audacity => print!("{}", audan_format::write_audacity_labels(&intervals)),
        _ => print_json(&result)?,
    }
    Ok(())
}

fn display_label(i: usize) -> String {
    let letter = (b'A' + (i % 26) as u8) as char;
    letter.to_string()
}
