//! `audan beats` (F2): beat grid, cached at L0 (decode) / L1 (resample) / L3
//! (beat grid). A single file renders richly (table/`-q`/`--format`); more
//! than one file (explicit paths and/or directories, walked recursively for
//! known audio extensions) runs as a batch (RV3): a `rayon` pool sized by
//! `-j`, progress on stderr, one JSON-Lines result per line on stdout.

use std::path::{Path, PathBuf};

use audan_cache::Resolver;
use rayon::prelude::*;

use crate::beats_backend::{self, ResolvedBackend};
use crate::cli::Cli;
use crate::config::Resolved;
use crate::pipeline;
use crate::render::{effective_format, print_json, OutputFormat};

const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "flac", "mp3", "m4a", "aac", "ogg", "opus", "wv", "alac",
];

pub fn run(
    files: &[PathBuf],
    click: Option<&Path>,
    model: Option<&str>,
    cli: &Cli,
    resolved: &Resolved,
) -> anyhow::Result<()> {
    let expanded = expand_paths(files)?;
    if expanded.is_empty() {
        return Err(
            audan_core::AudanError::Usage("no input files given to `audan beats`".into()).into(),
        );
    }

    // Resolved once (not per file): loading a real model is comparatively
    // expensive, and every file in a batch shares the same backend choice.
    let backend = beats_backend::resolve(model, cli, resolved)?;

    if expanded.len() == 1 {
        return run_single(&expanded[0], click, cli, resolved, &backend);
    }

    run_batch(&expanded, click, cli, resolved, &backend)
}

fn run_single(
    file: &Path,
    click: Option<&Path>,
    cli: &Cli,
    resolved: &Resolved,
    backend: &ResolvedBackend,
) -> anyhow::Result<()> {
    let grid = compute_beats(file, resolved, backend)?;

    if cli.strict {
        let conf = grid.tempo.candidates.top().confidence;
        if conf < resolved.strict_min_confidence {
            return Err(audan_core::AudanError::LowConfidence {
                confidence: conf,
                threshold: resolved.strict_min_confidence,
            }
            .into());
        }
    }

    if let Some(click_path) = click {
        let wav = audan_format::write_click_track(&grid.beats, grid.frames.sr)?;
        std::fs::write(click_path, wav)?;
    }

    if cli.quiet {
        println!(
            "{:.2} {}/4",
            grid.tempo.median_bpm, grid.meter.beats_per_bar
        );
        return Ok(());
    }

    match effective_format(cli.format) {
        OutputFormat::Table => {
            println!("duration:   {:.3}s", grid.duration_seconds);
            println!(
                "tempo:      {:.2} bpm ({:?})",
                grid.tempo.median_bpm, grid.tempo.stability.class
            );
            println!(
                "meter:      {}/4 (confidence {:.2})",
                grid.meter.beats_per_bar, grid.meter.confidence
            );
            println!("beats:      {}", grid.beats.len());
            println!("downbeats:  {}", grid.downbeats.len());
            println!("candidates:");
            for c in grid.tempo.candidates.iter() {
                println!("  {:.2} bpm  conf={:.3}", c.value, c.confidence);
            }
        }
        OutputFormat::Jams => {
            let data = grid
                .beats
                .iter()
                .zip(grid.confidence.iter())
                .map(|(&t, &c)| audan_format::JamsObservation {
                    time: t,
                    duration: 0.0,
                    value: serde_json::json!(1),
                    confidence: Some(c),
                })
                .collect();
            let ann = audan_format::JamsAnnotation {
                namespace: "beat".into(),
                data,
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&audan_format::write_jams(&[ann], grid.duration_seconds))?
            );
        }
        OutputFormat::Lab => print!("{}", audan_format::write_lab(&beat_intervals(&grid))),
        OutputFormat::Audacity => print!(
            "{}",
            audan_format::write_audacity_labels(&beat_intervals(&grid))
        ),
        OutputFormat::Csv => {
            println!("beat_seconds");
            for t in &grid.beats {
                println!("{t:.6}");
            }
        }
        OutputFormat::Midi => {
            let bytes = audan_format::write_midi_tempo_track(&[(0.0, grid.tempo.median_bpm)], 480)?;
            use std::io::Write;
            std::io::stdout().write_all(&bytes)?;
        }
        _ => print_json(&grid)?,
    }
    Ok(())
}

fn run_batch(
    files: &[PathBuf],
    click: Option<&Path>,
    cli: &Cli,
    resolved: &Resolved,
    backend: &ResolvedBackend,
) -> anyhow::Result<()> {
    let jobs = cli.jobs.unwrap_or_else(rayon::current_num_threads).max(1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build()?;

    // Each worker opens its own `Resolver` against the same cache root; the
    // `redb` index handles concurrent access and per-key lock files prevent
    // two workers duplicating the same computation on a duplicate master
    // (audan-cache's own tests already prove concurrent resolution of one
    // key computes once), so no extra dedup logic belongs here. `backend` is
    // shared read-only across workers -- resolved once in `run` rather than
    // per file, since loading a real model is comparatively expensive.
    let results: Vec<(PathBuf, anyhow::Result<audan_core::BeatGrid>)> = pool.install(|| {
        files
            .par_iter()
            .map(|path| {
                eprintln!("audan: analysing {}", path.display());
                (path.clone(), compute_beats(path, resolved, backend))
            })
            .collect()
    });

    let mut failures = 0usize;
    for (path, result) in &results {
        let line = match result {
            Ok(grid) => serde_json::json!({ "file": path.display().to_string(), "beats": grid }),
            Err(e) => {
                failures += 1;
                serde_json::json!({ "file": path.display().to_string(), "error": e.to_string() })
            }
        };
        println!("{line}");
    }

    if let Some(click_path) = click {
        if let Some((_, Ok(grid))) = results.first() {
            let wav = audan_format::write_click_track(&grid.beats, grid.frames.sr)?;
            std::fs::write(click_path, wav)?;
        }
    }

    if failures > 0 {
        anyhow::bail!("{failures} of {} files in the batch failed", results.len());
    }
    Ok(())
}

fn beat_intervals(grid: &audan_core::BeatGrid) -> Vec<audan_format::LabelInterval> {
    grid.beats
        .windows(2)
        .map(|w| audan_format::LabelInterval {
            start: w[0],
            end: w[1],
            label: "beat".into(),
        })
        .collect()
}

fn compute_beats(
    path: &Path,
    resolved: &Resolved,
    backend: &ResolvedBackend,
) -> anyhow::Result<audan_core::BeatGrid> {
    let resolver = Resolver::open(&resolved.cache_root)?;
    let (l0_key, signal) = pipeline::resolve_l0(&resolver, path)?;
    let (l1_key, mono) = pipeline::resolve_l1_analysis(&resolver, &l0_key, &signal)?;
    let (_beats_key, grid) = pipeline::resolve_beats(
        &resolver,
        &l1_key,
        &mono,
        None,
        pipeline::expected_beats_frames(backend.as_dyn()),
        backend.as_dyn(),
    )?;
    Ok(grid)
}

fn expand_paths(paths: &[PathBuf]) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        if p.is_dir() {
            walk_dir(p, &mut out)?;
        } else {
            out.push(p.clone());
        }
    }
    Ok(out)
}

fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir(&path, out)?;
        } else if is_audio_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}
