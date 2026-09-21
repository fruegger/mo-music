//! Glue between "the user asked for stems with backend X" (S6.4 RV4) and
//! "the weights are on disk, license accepted" -- `audan-model`'s job, not
//! reimplemented here.

use std::path::PathBuf;

use audan_core::{AudanError, Result};
use audan_model::{LicenseGate, ModelStore, Registry};

/// Resolves `backend_name` (e.g. `"htdemucs"`) against `registry`, then asks
/// `store` to ensure the weights are downloaded, checksum-verified, and
/// license-accepted via `gate` (S6.4 RV4).
///
/// `audan-stems` ships no default backend (RISK-1: Demucs weights licensing
/// is genuinely unresolved), so an unrecognised or unspecified name is not a
/// fallback opportunity -- it's a clear, actionable error.
pub fn resolve_backend_model(
    registry: &Registry,
    store: &ModelStore,
    gate: &dyn LicenseGate,
    backend_name: &str,
) -> Result<PathBuf> {
    let entry = registry.find(backend_name).ok_or_else(|| {
        AudanError::Model(format!(
            "no separation backend configured; '{backend_name}' is not in the model \
             registry. audan-stems ships no default backend (RISK-1: Demucs weights \
             licensing is unresolved) -- choose one explicitly via `--model <id>` \
             (e.g. `--model htdemucs`) and accept its license terms"
        ))
    })?;
    store.ensure(entry, gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct PanicsIfCalled;
    impl LicenseGate for PanicsIfCalled {
        fn confirm(&self, _entry: &audan_model::ModelEntry) -> bool {
            panic!("gate must not be consulted when the model is already installed");
        }
    }

    #[test]
    fn unknown_backend_name_errors_clearly_without_touching_the_store() {
        let registry = Registry::default_embedded();
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf());

        let err = resolve_backend_model(&registry, &store, &PanicsIfCalled, "not-a-real-backend")
            .expect_err("unknown backend name must error");

        let message = err.to_string();
        assert!(message.contains("not-a-real-backend"));
        assert!(message.contains("RISK-1") || message.contains("--model"));
    }

    #[test]
    fn known_backend_name_resolves_in_the_registry() {
        let registry = Registry::default_embedded();
        assert!(registry.find("htdemucs").is_some());
    }

    /// `ModelStore::ensure` always does a real `ureq` HTTP call on a cache
    /// miss, so this test never lets that happen: it pre-populates the
    /// store's cache directory with a fake already-installed file (mirroring
    /// the on-disk layout `audan-model`'s own tests exercise --
    /// `root/<name>/<version>/<file name from the url>`) so `ensure` returns
    /// via its `is_installed` short-circuit before ever consulting the
    /// license gate or the network. Testing past this point (an actual
    /// download) needs a live network or a mock HTTP server, which is out of
    /// scope here since `audan-model::ModelStore::ensure`'s fetch step isn't
    /// swappable from outside that crate.
    #[test]
    fn already_installed_backend_resolves_without_network_or_gate() {
        let registry = Registry::default_embedded();
        let entry = registry
            .find("htdemucs")
            .expect("htdemucs must be in the embedded registry");

        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join(&entry.name).join(&entry.version);
        fs::create_dir_all(&model_dir).unwrap();
        let file_name = entry.url.rsplit('/').next().unwrap();
        let expected_path = model_dir.join(file_name);
        fs::write(&expected_path, b"fake pre-installed weights").unwrap();

        let store = ModelStore::new(dir.path().to_path_buf());
        assert!(store.is_installed(entry));

        let resolved = resolve_backend_model(&registry, &store, &PanicsIfCalled, "htdemucs")
            .expect("already-installed model must resolve without consulting the gate");

        assert_eq!(resolved, expected_path);
    }

    #[test]
    fn declined_license_on_a_cache_miss_errors() {
        struct AlwaysDecline;
        impl LicenseGate for AlwaysDecline {
            fn confirm(&self, _entry: &audan_model::ModelEntry) -> bool {
                false
            }
        }

        let registry = Registry::default_embedded();
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf());

        let err = resolve_backend_model(&registry, &store, &AlwaysDecline, "htdemucs")
            .expect_err("declining the license must error rather than fetch");
        assert!(err.to_string().contains("license"));
    }
}
