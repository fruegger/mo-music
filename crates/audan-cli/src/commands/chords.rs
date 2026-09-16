//! `audan chords` (F5): resolves the beat grid through the same L3 cache key
//! `audan beats` would use (RV2's payoff -- a prior `audan beats` run on the
//! same file makes this skip beat tracking entirely), then chord estimation
//! at L4. `--beats <grid.json>` overrides L3 resolution with a
//! hand-corrected grid (RV5).

use std::path::Path;

use audan_cache::Resolver;

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
    let resolver = Resolver::open(&resolved.cache_root)?;
    let (l0_key, signal) = pipeline::resolve_l0(&resolver, file)?;
    let (l1_key, mono) = pipeline::resolve_l1_analysis(&resolver, &l0_key, &signal)?;
    let (beats_key, grid) = pipeline::resolve_beats(
        &resolver,
        &l1_key,
        &mono,
        beats_override,
        pipeline::expected_beats_frames(),
    )?;

    let chord_params = audan_dsp::ChordChromaParams {
        pad: resolved.pad,
        ..Default::default()
    };
    let vocabulary = resolved.analysis.chords.vocabulary.clone();

    let seq = pipeline::resolve_chords(
        &resolver,
        &l1_key,
        &beats_key,
        &mono,
        &grid,
        &chord_params,
        &vocabulary,
    )?;

    if cli.strict {
        let min_conf = seq
            .chords
            .iter()
            .map(|c| c.confidence)
            .fold(f32::INFINITY, f32::min);
        if min_conf.is_finite() && min_conf < resolved.strict_min_confidence {
            return Err(audan_core::AudanError::LowConfidence {
                confidence: min_conf,
                threshold: resolved.strict_min_confidence,
            }
            .into());
        }
    }

    if cli.quiet {
        println!("{} chords", seq.chords.len());
        return Ok(());
    }

    let intervals: Vec<audan_format::LabelInterval> = seq
        .chords
        .iter()
        .filter_map(|c| {
            let start = grid.beat_time(c.start_beat)?;
            let end = pipeline::beat_end_time(&grid, c.end_beat);
            Some(audan_format::LabelInterval {
                start,
                end,
                label: c.chord.clone(),
            })
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
        OutputFormat::Jams => {
            let data = seq
                .chords
                .iter()
                .filter_map(|c| {
                    let start = grid.beat_time(c.start_beat)?;
                    let end = pipeline::beat_end_time(&grid, c.end_beat);
                    Some(audan_format::JamsObservation {
                        time: start,
                        duration: (end - start).max(0.0),
                        value: serde_json::json!(c.chord),
                        confidence: Some(c.confidence),
                    })
                })
                .collect();
            let ann = audan_format::JamsAnnotation {
                namespace: "chord".into(),
                data,
            };
            let duration = grid.beats.last().copied().unwrap_or(0.0);
            println!(
                "{}",
                serde_json::to_string_pretty(&audan_format::write_jams(&[ann], duration))?
            );
        }
        _ => print_json(&seq)?,
    }
    Ok(())
}
