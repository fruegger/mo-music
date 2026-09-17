//! Argument parsing (S8.9). Global flags (`--cache-dir`, `--strict`,
//! `--pad`, `--format`, `-q`, `-j`, `--accept-model-license`) are attached to
//! every subcommand via `global = true` rather than duplicated per command.
//!
//! `--cache-dir`/`--strict-threshold`/`--pad` use clap's `env` attribute so
//! that, by the time [`crate::config::resolve_all`] sees them, each already
//! carries "explicit CLI flag, else the matching `AUDAN_*` env var, else
//! `None`" -- the top two rungs of S8.8's four-rung precedence collapsed
//! into one `Option<T>` by clap itself.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "audan",
    version,
    about = "audan: music-information-retrieval command line utilities"
)]
pub struct Cli {
    /// Cache root directory (S7.2). Compiled default is the XDG cache dir
    /// for "audan"; overridable by $AUDAN_CACHE_DIR or this flag.
    #[arg(long, global = true, env = "AUDAN_CACHE_DIR")]
    pub cache_dir: Option<PathBuf>,

    /// Explicit analysis.toml path, overriding the XDG config-dir lookup.
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Exit 3 (S8.9) if the top candidate's confidence is below the strict
    /// threshold, instead of printing a low-confidence result normally.
    #[arg(long, global = true)]
    pub strict: bool,

    /// Override [strict].min_confidence (S8.5/S8.9).
    #[arg(long, global = true, env = "AUDAN_STRICT_MIN_CONFIDENCE")]
    pub strict_threshold: Option<f32>,

    /// Override [frames].pad (S8.2). Affects key/chord chroma only --
    /// audan-beats' mel frontend is not parameterized by padding mode.
    #[arg(long, global = true, value_enum, env = "AUDAN_PAD")]
    pub pad: Option<PadArg>,

    /// Output format. Default is json when stdout is piped, a human table
    /// on a TTY (S8.9).
    #[arg(long, global = true, value_enum)]
    pub format: Option<FormatArg>,

    /// One-line summary output (S8.9).
    #[arg(short = 'q', long = "quiet", global = true)]
    pub quiet: bool,

    /// Worker count for batch input (RV3). Defaults to available parallelism.
    #[arg(short = 'j', long = "jobs", global = true)]
    pub jobs: Option<usize>,

    /// Accept a model's license non-interactively (RV4 step 3), for
    /// scripted/CI use where stdin is not a TTY.
    #[arg(long, global = true)]
    pub accept_model_license: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadArg {
    None,
    Zero,
    Reflect,
}

impl From<PadArg> for audan_core::PadMode {
    fn from(p: PadArg) -> Self {
        match p {
            PadArg::None => audan_core::PadMode::None,
            PadArg::Zero => audan_core::PadMode::Zero,
            PadArg::Reflect => audan_core::PadMode::Reflect,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatArg {
    Json,
    Jams,
    Lab,
    Csv,
    Audacity,
    Midi,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Probe a file: format, duration, loudness, clipping, encoder delay (F1).
    Probe { file: PathBuf },

    /// Estimate the beat grid: beats, downbeats, meter, tempo (F2). Accepts
    /// multiple files (or directories, walked recursively) for batch mode
    /// (RV3); with more than one input, results stream as JSON Lines.
    Beats {
        files: Vec<PathBuf>,

        /// Write a click-track WAV at the detected beat times (S8.7's
        /// debugging instrument). In batch mode, applies to the first file.
        #[arg(long)]
        click: Option<PathBuf>,

        /// Use a real neural backend (e.g. `beat_this`) instead of the
        /// always-available onset_fallback default. Unlike `stems --model`,
        /// this is optional: omitting it (or leaving `[beats] model` unset
        /// in analysis.toml) keeps `audan beats` working fully offline with
        /// no model, no network, no license prompt (ADR-7).
        #[arg(long)]
        model: Option<String>,
    },

    /// Estimate musical key, with Camelot / Open Key notation (F3).
    Key { file: PathBuf },

    /// Transcribe the chord sequence (F5).
    Chords {
        file: PathBuf,

        /// Use a hand-corrected beat grid instead of computing/caching one
        /// (RV5). Must be a JSON `BeatGrid` whose `frames` block matches the
        /// fixed shape audan-beats always produces.
        #[arg(long)]
        beats: Option<PathBuf>,
    },

    /// Segment into structural sections (unlabeled A/B/C) (F4).
    Struct {
        file: PathBuf,

        /// See `chords --beats` (RV5).
        #[arg(long)]
        beats: Option<PathBuf>,
    },

    /// Separate instrument stems (F6). Requires an explicit `--model`;
    /// audan-stems ships no default backend (RISK-1).
    Stems {
        file: PathBuf,

        #[arg(long)]
        model: String,
    },

    /// Write results back into file metadata tags (F7).
    Tag {
        file: PathBuf,

        /// Comma-separated fields to write: `bpm`, `key`, or both.
        #[arg(long, value_delimiter = ',')]
        write: Vec<String>,
    },

    /// Inspect or manage the content-addressed cache (S5.2).
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
}

#[derive(Subcommand, Debug)]
pub enum CacheAction {
    Stats,
    Prune {
        /// Byte budget to prune down to. Defaults to [cache].budget_bytes.
        #[arg(long)]
        budget_bytes: Option<u64>,
    },
    Clear,
}
