//! Shared decode -> resample -> beat-grid glue, wired through
//! `audan-cache::Resolver` (S8.1/S5.2, RV1/RV2). This is what makes
//! `audan chords` after `audan beats` on the same file hit a warm L3 cache
//! entry instead of re-tracking beats.
//!
//! Cache-key design (a pragmatic, documented instance of S8.1's scheme):
//!
//! - **L0** (`decode`): keyed on `blake3(file_bytes)` plus the pinned
//!   `symphonia` version, with no parent. The true `decoder_id` (S8.4) isn't
//!   known until *after* decoding runs (it depends on whether the
//!   `symphonia`-native path or the `ffmpeg` fallback actually handled the
//!   file), so this approximates it with the file hash plus the primary
//!   decoder's pinned version -- the common path. An `ffmpeg`-fallback
//!   decode of a file that would normally be `symphonia`-native, decoded
//!   with a different `ffmpeg` version, would not get a distinct L0 key from
//!   that. Accepted simplification for this integration layer; a stricter
//!   implementation would thread the real `DecodeReport.decoder_id` back
//!   into the key, which is only possible by decoding unconditionally first
//!   (defeating the point of keying on it) or restructuring `audan-io`'s
//!   API to expose decoder identity without decoding. `decode_file` is still
//!   only actually invoked on a miss, so the real payoff (skipping decode
//!   entirely on a warm cache) holds regardless.
//! - **L1** (`resample`): parent = L0 key, params = target sample rate.
//! - **L3** (`beats`): parent = L1 key, params name the backend
//!   (`track_beats_default` always uses `onset_fallback`, so bumping that
//!   string is the invalidation lever if a real backend is added later).
//! - **L4** (`chords`/`struct`): parent = L1 key (the chroma's real parent),
//!   with the L3 beat-grid key's hex folded into `params` as an extra field
//!   -- `KeyDeriver::derive` only takes one `parent`, so this is the
//!   documented way to make a two-upstream-dependency key still change
//!   whenever *either* upstream (the mono signal or the beat grid) changes.

use std::path::Path;

use audan_cache::{CacheKey, CacheLayer, KeyDeriver, Resolver};
use audan_core::{
    BeatGrid, FrameGrid, FramesMeta, MonoSignal, PadMode, Signal, ANALYSIS_SAMPLE_RATE,
};
use serde::Serialize;

use crate::render::StructureResultDto;

pub fn hash_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Serialize)]
struct L0Params<'a> {
    file_hash: &'a str,
    decoder_version: &'a str,
}

pub fn resolve_l0(resolver: &Resolver, path: &Path) -> anyhow::Result<(CacheKey, Signal)> {
    let file_hash = hash_file(path)?;
    let key = KeyDeriver::derive(
        None,
        &CacheLayer::L0.stage_id("decode"),
        1,
        &L0Params {
            file_hash: &file_hash,
            decoder_version: audan_io::decode::SYMPHONIA_VERSION,
        },
    );
    let owned_path = path.to_path_buf();
    let signal: Signal = resolver.resolve(key, move || {
        let (signal, _report) = audan_io::decode_file(&owned_path)?;
        Ok(signal)
    })?;
    Ok((key, signal))
}

pub fn resolve_l1_analysis(
    resolver: &Resolver,
    l0_key: &CacheKey,
    signal: &Signal,
) -> anyhow::Result<(CacheKey, MonoSignal)> {
    let key = KeyDeriver::derive(
        Some(l0_key),
        &CacheLayer::L1.stage_id("resample"),
        1,
        &ANALYSIS_SAMPLE_RATE,
    );
    let mono: MonoSignal = resolver.resolve(key, || audan_io::to_analysis_rate(signal))?;
    Ok((key, mono))
}

/// `audan-beats`' mel frontend hardcodes hop=512/win=2048/`PadMode::Reflect`
/// internally (`audan_beats::mel::MelFrontend::compute`) and is not
/// parameterized by `--pad` at all -- that flag only affects the key/chord
/// chroma stages (`audan_dsp::{KeyChromaParams,ChordChromaParams}::pad`). A
/// hand-corrected grid supplied via `--beats` (RV5) must therefore be
/// validated against this fixed shape, independent of the resolved `--pad`
/// config, or a perfectly valid grid computed with a non-default `--pad`
/// elsewhere in the pipeline would be rejected for the wrong reason.
pub fn expected_beats_frames() -> FramesMeta {
    FrameGrid::new(ANALYSIS_SAMPLE_RATE, 512, 2048, PadMode::Reflect).into()
}

#[derive(Serialize)]
struct BeatsParams<'a> {
    backend: &'a str,
}

/// Resolves the beat grid used by `beats`/`chords`/`struct`/`tag`: normally
/// through the L3 cache (RV1 cold, RV2 warm), or -- when `beats_override` is
/// `Some` (RV5's `--beats grid.json`) -- loaded directly from a
/// hand-corrected file, validated, and never written back into the L3 cache
/// namespace (it's the user's own result, not a re-derivable cache entry).
pub fn resolve_beats(
    resolver: &Resolver,
    l1_key: &CacheKey,
    mono: &MonoSignal,
    beats_override: Option<&Path>,
    expected_frames: FramesMeta,
) -> anyhow::Result<(CacheKey, BeatGrid)> {
    if let Some(path) = beats_override {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read --beats {}: {e}", path.display()))?;
        let grid: BeatGrid = serde_json::from_str(&text).map_err(|e| {
            audan_core::AudanError::InvalidInput(format!(
                "--beats {}: not a valid beat grid: {e}",
                path.display()
            ))
        })?;
        if grid.frames != expected_frames {
            return Err(audan_core::AudanError::InvalidInput(format!(
                "--beats {}: frames block {:?} does not match the fixed shape audan-beats \
                 always produces ({:?}) -- foreign grids are rejected loudly on mismatch \
                 rather than silently misaligning (S6.5)",
                path.display(),
                grid.frames,
                expected_frames
            ))
            .into());
        }
        let key = KeyDeriver::derive(
            Some(l1_key),
            &CacheLayer::L3.stage_id("beats-override"),
            1,
            &blake3::hash(text.as_bytes()).to_hex().to_string(),
        );
        Ok((key, grid))
    } else {
        let key = KeyDeriver::derive(
            Some(l1_key),
            &CacheLayer::L3.stage_id("beats"),
            1,
            &BeatsParams {
                backend: "onset_fallback",
            },
        );
        let grid: BeatGrid = resolver.resolve(key, || audan_beats::track_beats_default(mono))?;
        Ok((key, grid))
    }
}

/// `end_beat` in a `[start_beat, end_beat)` interval may equal
/// `beats.len()` (the half-open upper bound past the last real beat index),
/// which `BeatGrid::beat_time` returns `None` for by construction -- fall
/// back to the last known beat time in that case.
pub fn beat_end_time(grid: &BeatGrid, end_beat: usize) -> f64 {
    grid.beat_time(end_beat)
        .unwrap_or_else(|| grid.beats.last().copied().unwrap_or(0.0))
}

#[derive(Serialize)]
struct ChordsParams<'a> {
    beat_grid_key: &'a str,
    vocabulary: &'a str,
}

pub fn resolve_chords(
    resolver: &Resolver,
    l1_key: &CacheKey,
    beats_key: &CacheKey,
    mono: &MonoSignal,
    grid: &BeatGrid,
    params: &audan_dsp::ChordChromaParams,
    vocabulary: &str,
) -> anyhow::Result<audan_chords::ChordSequence> {
    let key = KeyDeriver::derive(
        Some(l1_key),
        &CacheLayer::L4.stage_id("chords"),
        1,
        &ChordsParams {
            beat_grid_key: &beats_key.to_hex(),
            vocabulary,
        },
    );
    let seq: audan_chords::ChordSequence = resolver.resolve(key, || {
        Ok(audan_chords::estimate_chords_from_signal(
            mono, grid, params,
        ))
    })?;
    Ok(seq)
}

#[derive(Serialize)]
struct StructParams<'a> {
    beat_grid_key: &'a str,
}

pub fn resolve_struct(
    resolver: &Resolver,
    l1_key: &CacheKey,
    beats_key: &CacheKey,
    mono: &MonoSignal,
    grid: &BeatGrid,
    key_params: &audan_dsp::KeyChromaParams,
) -> anyhow::Result<StructureResultDto> {
    let key = KeyDeriver::derive(
        Some(l1_key),
        &CacheLayer::L4.stage_id("struct"),
        1,
        &StructParams {
            beat_grid_key: &beats_key.to_hex(),
        },
    );
    let dto: StructureResultDto = resolver.resolve(key, || {
        let chroma = audan_dsp::key_chroma(mono, key_params);
        let result =
            audan_struct::segment_structure(&chroma, grid, &audan_struct::SegmentParams::default());
        Ok(StructureResultDto::from(&result))
    })?;
    Ok(dto)
}
