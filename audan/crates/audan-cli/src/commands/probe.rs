//! `audan probe` (F1): format, duration, loudness, clipping, encoder delay.
//! No caching -- decode is cheap relative to what the cache targets (S1),
//! and probe is meant to stay simple (a single decode, no feature stages).

use std::path::Path;

use crate::cli::FormatArg;
use crate::render::{effective_format, print_json, OutputFormat, ProbeReportDto};

pub fn run(file: &Path, format: Option<FormatArg>, quiet: bool) -> anyhow::Result<()> {
    let report = audan_io::probe_file(file)?;

    if quiet {
        println!(
            "{:.2}s {} peak={:.3}",
            report.duration_seconds, report.format, report.peak_amplitude
        );
        return Ok(());
    }

    match effective_format(format) {
        OutputFormat::Table => {
            println!("format:          {}", report.format);
            println!("duration:        {:.3}s", report.duration_seconds);
            println!("rms amplitude:   {:.4}", report.rms_amplitude);
            println!("peak amplitude:  {:.4}", report.peak_amplitude);
            println!(
                "clipped samples: {} ({:.4}%)",
                report.clipped_sample_count,
                report.clipped_sample_fraction * 100.0
            );
            println!("delay stripped:  {} samples", report.delay_stripped_samples);
            println!(
                "decoder:         {} {}",
                report.decoder_id, report.decoder_version
            );
        }
        _ => print_json(&ProbeReportDto::from(&report))?,
    }
    Ok(())
}
