//! Resolves which `audan_beats::InferenceBackend` a beats-consuming command
//! should use (S8.6: "backend choice is a runtime flag, never a
//! compile-time assumption"): an explicit `--model` name (only `audan
//! beats` exposes that flag directly) if given, else the user's configured
//! default (`[beats] model` in analysis.toml -- empty in the compiled
//! default, so a bare `audan beats`/`chords`/`struct`/`tag` keeps working
//! fully offline per ADR-7/QS8/QS9), else the always-available
//! `OnsetFallbackBackend`. Mirrors `commands::stems::run`'s use of
//! `audan_model`/`license_gate`/`cache_paths` for the model-resolution part.

use crate::cli::Cli;
use crate::config::Resolved;

pub enum ResolvedBackend {
    Fallback(audan_beats::OnsetFallbackBackend),
    Onnx(audan_beats::OnnxBackend),
}

impl ResolvedBackend {
    pub fn as_dyn(&self) -> &dyn audan_beats::InferenceBackend {
        match self {
            ResolvedBackend::Fallback(b) => b,
            ResolvedBackend::Onnx(b) => b,
        }
    }
}

/// `model_flag` is the `--model` value on subcommands that expose one (only
/// `audan beats` does); `chords`/`struct`/`tag` pass `None` and rely solely
/// on the configured default.
pub fn resolve(
    model_flag: Option<&str>,
    cli: &Cli,
    resolved: &Resolved,
) -> anyhow::Result<ResolvedBackend> {
    let configured = resolved.analysis.beats.model.trim();
    let name = model_flag.or((!configured.is_empty()).then_some(configured));

    let Some(name) = name else {
        return Ok(ResolvedBackend::Fallback(audan_beats::OnsetFallbackBackend));
    };

    let registry = audan_model::Registry::default_embedded();
    let models_dir = crate::cache_paths::models_dir(&resolved.cache_root);
    let store = audan_model::ModelStore::new(models_dir);
    let gate = crate::license_gate::select_gate(cli.accept_model_license);
    let path =
        audan_beats::resolve::resolve_backend_model(&registry, &store, gate.as_ref(), name)?;
    let entry = registry
        .find(name)
        .expect("resolve_backend_model already validated this name exists");
    let backend = audan_beats::OnnxBackend::new(&path, entry.version.clone())?;
    Ok(ResolvedBackend::Onnx(backend))
}
