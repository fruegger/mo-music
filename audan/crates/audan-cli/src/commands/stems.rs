//! `audan stems` (F6): proves the plumbing up through "the weights are on
//! disk, license accepted" (`resolve_backend_model`, RV4), then returns a
//! clear, honest error -- there is no real `Separator` implementation
//! anywhere in this workspace (RISK-1: no permissively-licensed weights
//! exist to bundle or default to), so faking output here would be worse
//! than refusing.

use std::path::Path;

use crate::cli::Cli;
use crate::config::Resolved;

pub fn run(file: &Path, model: &str, cli: &Cli, resolved: &Resolved) -> anyhow::Result<()> {
    let (signal, _decode_report) = audan_io::decode_file(file)?;
    let _stems_rate_signal = audan_io::to_stems_rate(&signal)?;

    let registry = audan_model::Registry::default_embedded();
    let models_dir = crate::cache_paths::models_dir(&resolved.cache_root);
    let store = audan_model::ModelStore::new(models_dir);
    let gate = crate::license_gate::select_gate(cli.accept_model_license);

    let _weights_path =
        audan_stems::resolve_backend_model(&registry, &store, gate.as_ref(), model)?;

    Err(audan_core::AudanError::Model(format!(
        "no separation backend is implemented for '{model}' yet -- audan-stems ships the trait \
         and model-resolution plumbing but no inference backend; see RISK-1"
    ))
    .into())
}
