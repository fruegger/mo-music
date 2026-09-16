//! Configuration resolution (S8.8): compiled defaults -> user
//! `analysis.toml` -> `AUDAN_*` env vars -> CLI flags, lowest to highest.
//!
//! The compiled default is `resources/config/analysis.toml`, embedded via
//! `include_str!` so the binary never depends on that file existing on disk
//! at runtime. A present user file is merged on top of it *section by
//! section* (`merge_toml` below) rather than requiring every field to be
//! `Option`, so a user who only sets `[strict]` doesn't have to restate
//! `[frames]`/`[beats]`/etc. Only three fields are actually threaded through
//! to the DSP/estimator layer end-to-end (cache dir, strict threshold, pad
//! mode) per the task's scope -- the rest of `AnalysisConfig` exists so the
//! precedence mechanism and the resource file's shape are both exercised
//! honestly, without claiming to wire every knob into every stage.

use std::path::{Path, PathBuf};

use audan_core::PadMode;
use serde::Deserialize;

const COMPILED_DEFAULT_TOML: &str = include_str!("../../../resources/config/analysis.toml");

#[derive(Debug, Clone, Deserialize)]
pub struct FramesConfig {
    pub sample_rate: u32,
    pub hop: usize,
    pub win: usize,
    pub pad: PadMode,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BeatsConfig {
    pub model: String,
    pub model_version: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KeyConfig {
    pub profile: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChordsConfig {
    pub vocabulary: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StructConfig {
    pub label_sections: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StemsConfig {
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    pub budget_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StrictConfig {
    pub min_confidence: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalysisConfig {
    pub frames: FramesConfig,
    pub beats: BeatsConfig,
    pub key: KeyConfig,
    pub chords: ChordsConfig,
    #[serde(rename = "struct")]
    pub structure: StructConfig,
    pub stems: StemsConfig,
    pub cache: CacheConfig,
    pub strict: StrictConfig,
}

impl AnalysisConfig {
    pub fn compiled_default() -> Self {
        toml::from_str(COMPILED_DEFAULT_TOML)
            .expect("embedded resources/config/analysis.toml must parse")
    }
}

/// Recursively merges `overlay` onto `base`: a table key present in both
/// sides is merged recursively (so a user's `[frames]` table only needs the
/// keys it actually overrides); any other value in `overlay` replaces
/// `base` wholesale. This is what lets a user `analysis.toml` override just
/// the sections/fields it sets, per S8.8, without every field in
/// `AnalysisConfig` needing to be `Option<T>`.
fn merge_toml(base: &mut toml::Value, overlay: toml::Value) {
    match overlay {
        toml::Value::Table(overlay_table) => {
            if let toml::Value::Table(base_table) = base {
                for (k, v) in overlay_table {
                    match base_table.get_mut(&k) {
                        Some(existing) => merge_toml(existing, v),
                        None => {
                            base_table.insert(k, v);
                        }
                    }
                }
            } else {
                *base = toml::Value::Table(overlay_table);
            }
        }
        other => *base = other,
    }
}

/// Parses the compiled-default TOML and, if `user_toml` is given, merges it
/// on top (S8.8's second rung). Returns a fully-populated config either way.
pub fn load_merged(user_toml: Option<&str>) -> anyhow::Result<AnalysisConfig> {
    let mut value: toml::Value = toml::from_str(COMPILED_DEFAULT_TOML)?;
    if let Some(text) = user_toml {
        let overlay: toml::Value = toml::from_str(text)?;
        merge_toml(&mut value, overlay);
    }
    // Round-trip through a string rather than `Value::try_into` directly:
    // keeps this independent of exactly which conversion API a given `toml`
    // crate version exposes on `Value`, at the cost of one extra reserialize.
    let merged_text = toml::to_string(&value)?;
    Ok(toml::from_str(&merged_text)?)
}

pub fn user_config_path(config_dir_override: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = config_dir_override {
        return Some(dir.join("analysis.toml"));
    }
    directories::ProjectDirs::from("", "", "audan").map(|d| d.config_dir().join("analysis.toml"))
}

/// Generic four-rung precedence resolution (S8.8): the last `Some` among
/// `file`, `env`, `flag` wins; `compiled` is the fallback when none are set.
/// `flag`/`env` are typically already-parsed `Option<T>`s from a `clap`
/// field (CLI flag) and `std::env::var(...).ok().and_then(|s| s.parse().ok())`
/// respectively.
pub fn resolve<T>(compiled: T, file: Option<T>, env: Option<T>, flag: Option<T>) -> T {
    flag.or(env).or(file).unwrap_or(compiled)
}

#[derive(Debug, Clone)]
pub struct Resolved {
    pub cache_root: PathBuf,
    pub strict_min_confidence: f32,
    pub pad: PadMode,
    pub analysis: AnalysisConfig,
}

pub fn resolve_all(cli: &crate::cli::Cli) -> anyhow::Result<Resolved> {
    let user_toml_path = cli.config.clone().or_else(|| user_config_path(None));
    let user_toml = user_toml_path
        .as_deref()
        .and_then(|p| std::fs::read_to_string(p).ok());
    let analysis = load_merged(user_toml.as_deref())?;

    let cache_root = crate::cache_paths::resolve_cache_root(cli.cache_dir.clone());

    // `cli.strict_threshold`/`cli.pad` are already "explicit CLI flag, else
    // AUDAN_STRICT_MIN_CONFIDENCE/AUDAN_PAD env var, else None" by the time
    // we get here -- clap's `env` attribute folded that merge in at parse
    // time (see cli.rs). So the `env` rung passed to `resolve` below is
    // deliberately `None`: it has already been absorbed into the `flag`
    // rung. `analysis.strict.min_confidence`/`analysis.frames.pad` already
    // reflect "user file, else compiled default" from `load_merged` above.
    // Passing both that and a freshly-parsed compiled default here is
    // therefore redundant in practice, but it keeps this call site an
    // honest instance of the same four-rung `resolve` helper the precedence
    // unit tests exercise directly, rather than a special case.
    let compiled = AnalysisConfig::compiled_default();
    let strict_min_confidence = resolve(
        compiled.strict.min_confidence,
        Some(analysis.strict.min_confidence),
        None,
        cli.strict_threshold,
    );
    let pad = resolve(
        compiled.frames.pad,
        Some(analysis.frames.pad),
        None,
        cli.pad.map(Into::into),
    );

    Ok(Resolved {
        cache_root,
        strict_min_confidence,
        pad,
        analysis,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_default_matches_resource_file() {
        let compiled = AnalysisConfig::compiled_default();
        assert_eq!(compiled.strict.min_confidence, 0.5);
        assert_eq!(compiled.frames.pad, PadMode::Reflect);
        assert_eq!(compiled.beats.model, "beat_this");
        assert_eq!(compiled.cache.budget_bytes, 5_000_000_000);
    }

    #[test]
    fn strict_threshold_precedence_flag_beats_env_beats_file_beats_compiled() {
        let compiled = 0.5f32;
        assert_eq!(
            resolve(compiled, None, None, None),
            0.5,
            "nothing set anywhere: compiled default wins"
        );
        assert_eq!(
            resolve(compiled, Some(0.7), None, None),
            0.7,
            "file set, nothing else: file wins"
        );
        assert_eq!(
            resolve(compiled, Some(0.7), Some(0.8), None),
            0.8,
            "file + env: env wins"
        );
        assert_eq!(
            resolve(compiled, Some(0.7), Some(0.8), Some(0.9)),
            0.9,
            "file + env + flag: flag wins"
        );
    }

    #[test]
    fn pad_mode_precedence_follows_the_same_four_rungs() {
        let compiled = PadMode::Reflect;
        assert_eq!(resolve(compiled, None, None, None), PadMode::Reflect);
        assert_eq!(
            resolve(compiled, Some(PadMode::Zero), None, None),
            PadMode::Zero
        );
        assert_eq!(
            resolve(compiled, Some(PadMode::Zero), Some(PadMode::None), None),
            PadMode::None
        );
        assert_eq!(
            resolve(
                compiled,
                Some(PadMode::Zero),
                Some(PadMode::None),
                Some(PadMode::Reflect)
            ),
            PadMode::Reflect
        );
    }

    #[test]
    fn cache_dir_precedence_is_flag_or_env_then_compiled_xdg_default() {
        let compiled = PathBuf::from("/compiled/default/cache");
        assert_eq!(resolve(compiled.clone(), None, None, None), compiled);
        let flag_or_env = PathBuf::from("/overridden/cache");
        assert_eq!(
            resolve(compiled, None, None, Some(flag_or_env.clone())),
            flag_or_env
        );
    }

    #[test]
    fn user_file_overrides_only_the_sections_it_sets() {
        let file_toml = "[strict]\nmin_confidence = 0.9\n";
        let merged = load_merged(Some(file_toml)).unwrap();
        assert_eq!(merged.strict.min_confidence, 0.9);
        assert_eq!(
            merged.frames.pad,
            PadMode::Reflect,
            "untouched section keeps the compiled default"
        );
        assert_eq!(
            merged.beats.model, "beat_this",
            "untouched section keeps the compiled default"
        );
    }

    #[test]
    fn user_file_can_override_a_single_scalar_leaving_siblings_at_the_compiled_default() {
        let file_toml = "[frames]\npad = \"none\"\n";
        let merged = load_merged(Some(file_toml)).unwrap();
        assert_eq!(merged.frames.pad, PadMode::None);
        assert_eq!(
            merged.frames.hop, 512,
            "sibling field in the same table keeps the compiled default"
        );
    }

    #[test]
    fn nothing_set_anywhere_resolves_to_the_compiled_default() {
        let merged = load_merged(None).unwrap();
        assert_eq!(merged.strict.min_confidence, 0.5);
        assert_eq!(merged.frames.pad, PadMode::Reflect);
    }
}
