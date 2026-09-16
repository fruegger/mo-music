//! TTY-vs-pipe rendering (S8.9): JSON when stdout is piped, a human table on
//! a TTY, with `--format` overriding either way. Uses `std::io::IsTerminal`
//! (stable since Rust 1.70) rather than an `is-terminal`/`atty` dependency.

use std::io::IsTerminal;

use serde::Serialize;

use crate::cli::FormatArg;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Jams,
    Lab,
    Csv,
    Audacity,
    Midi,
    Table,
}

pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

pub fn effective_format(explicit: Option<FormatArg>) -> OutputFormat {
    match explicit {
        Some(FormatArg::Json) => OutputFormat::Json,
        Some(FormatArg::Jams) => OutputFormat::Jams,
        Some(FormatArg::Lab) => OutputFormat::Lab,
        Some(FormatArg::Csv) => OutputFormat::Csv,
        Some(FormatArg::Audacity) => OutputFormat::Audacity,
        Some(FormatArg::Midi) => OutputFormat::Midi,
        None => {
            if stdout_is_tty() {
                OutputFormat::Table
            } else {
                OutputFormat::Json
            }
        }
    }
}

pub fn print_json<T: Serialize>(value: &T) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

/// `audan_io::ProbeReport` is deliberately not `Serialize` (S5.1's
/// dependency rule keeps format concerns out of `audan-io`), so this mirrors
/// the fields we want in JSON output rather than adding a
/// dependency-inverting `Serialize` impl upstream.
#[derive(Serialize)]
pub struct ProbeReportDto {
    pub format: String,
    pub duration_seconds: f64,
    pub rms_amplitude: f32,
    pub peak_amplitude: f32,
    pub clipped_sample_count: u64,
    pub clipped_sample_fraction: f64,
    pub delay_stripped_samples: u64,
    pub decoder_id: String,
    pub decoder_version: String,
}

impl From<&audan_io::ProbeReport> for ProbeReportDto {
    fn from(r: &audan_io::ProbeReport) -> Self {
        ProbeReportDto {
            format: r.format.clone(),
            duration_seconds: r.duration_seconds,
            rms_amplitude: r.rms_amplitude,
            peak_amplitude: r.peak_amplitude,
            clipped_sample_count: r.clipped_sample_count,
            clipped_sample_fraction: r.clipped_sample_fraction,
            delay_stripped_samples: r.delay_stripped_samples,
            decoder_id: r.decoder_id.clone(),
            decoder_version: r.decoder_version.clone(),
        }
    }
}

/// `audan_struct::{StructureResult, SectionEvent}` are likewise not
/// `Serialize`; same DTO workaround. `SectionEvent::label` is always `None`
/// upstream (RISK-8: real section-type labelling is out of scope), so it's
/// dropped here rather than carried as a perpetually-null field -- the
/// `struct` command's own display labels (A/B/C) are a render-time
/// convenience computed separately, not part of this cached shape.
#[derive(Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct SectionEventDto {
    pub start_beat: usize,
    pub end_beat: usize,
}

#[derive(Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct StructureResultDto {
    pub schema_version: u32,
    pub sections: Vec<SectionEventDto>,
}

impl From<&audan_struct::StructureResult> for StructureResultDto {
    fn from(r: &audan_struct::StructureResult) -> Self {
        StructureResultDto {
            schema_version: r.schema_version,
            sections: r
                .sections
                .iter()
                .map(|s| SectionEventDto {
                    start_beat: s.start_beat,
                    end_beat: s.end_beat,
                })
                .collect(),
        }
    }
}
