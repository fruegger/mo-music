//! `audan key` (F3): key estimation, cached through L0/L1 like every other
//! stage. Not itself cached at L4 -- key estimation doesn't depend on the
//! beat grid, and its own chroma (long window, heavy smoothing) is cheap
//! relative to beat tracking, so it isn't worth a dedicated cache layer here.

use std::path::Path;

use audan_cache::Resolver;

use crate::cli::Cli;
use crate::config::Resolved;
use crate::pipeline;
use crate::progress::{Progress, Stage, StageReporter};
use crate::render::{effective_format, print_json, OutputFormat};

pub fn run(file: &Path, cli: &Cli, resolved: &Resolved) -> anyhow::Result<()> {
    let progress = Progress::start(1, Progress::detect(cli.quiet));
    let task = progress.task(file);
    let resolver = Resolver::open(&resolved.cache_root)?;
    let l0_key = pipeline::l0_key(file, &task)?;
    let (_l1_key, mono) = pipeline::resolve_l1_analysis(&resolver, &l0_key, file, &task)?;

    let params = audan_dsp::KeyChromaParams {
        pad: resolved.pad,
        ..Default::default()
    };

    task.stage(Stage::Key);
    let ranked = audan_key::estimate_key_from_signal(&mono, &params);
    task.finish(None);
    progress.finish();

    if cli.strict {
        let conf = ranked.top().confidence;
        if conf < resolved.strict_min_confidence {
            return Err(audan_core::AudanError::LowConfidence {
                confidence: conf,
                threshold: resolved.strict_min_confidence,
            }
            .into());
        }
    }

    if cli.quiet {
        println!("{}", short_key_label(&ranked.top().value));
        return Ok(());
    }

    match effective_format(cli.format) {
        OutputFormat::Table => {
            let top = &ranked.top().value;
            println!(
                "key:      {} (confidence {:.2})",
                top.name,
                ranked.top().confidence
            );
            println!("camelot:  {}", top.camelot());
            println!("open key: {}", top.open_key());
            println!();
            println!("ranked candidates:");
            for c in ranked.iter() {
                println!("  {:<12} conf={:.3}", c.value.name, c.confidence);
            }
        }
        _ => print_json(&ranked)?,
    }
    Ok(())
}

fn short_key_label(k: &audan_key::KeyEstimate) -> String {
    let suffix = if k.mode == audan_key::Mode::Major {
        "maj"
    } else {
        "min"
    };
    format!("{}{}", k.tonic.name(), suffix)
}
