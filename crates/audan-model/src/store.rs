use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use audan_core::{AudanError, Result};

use crate::checksum::verify_checksum;
use crate::registry::ModelEntry;

/// A trait rather than a concrete prompt implementation: `audan-model` must
/// not need to know whether the caller is an interactive TTY confirmation or
/// a non-interactive `--accept-model-license` flag check (RV4 step 3). That
/// policy decision belongs to `audan-cli`.
pub trait LicenseGate {
    fn confirm(&self, entry: &ModelEntry) -> bool;
}

pub struct ModelStore {
    root: PathBuf,
}

impl ModelStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Resolves `$XDG_CACHE_HOME/audan/models/` with platform-appropriate
    /// fallbacks (S7.2). `audan-cli` decides whether to use this or honour
    /// `AUDAN_CACHE_DIR` instead -- that override lookup happens above this
    /// crate, which only needs an already-resolved root.
    pub fn default_cache_dir() -> Option<PathBuf> {
        directories::ProjectDirs::from("", "", "audan").map(|dirs| dirs.cache_dir().join("models"))
    }

    pub fn is_installed(&self, entry: &ModelEntry) -> bool {
        self.model_path(entry).is_file()
    }

    pub fn ensure(&self, entry: &ModelEntry, gate: &dyn LicenseGate) -> Result<PathBuf> {
        self.ensure_with_fetch(entry, gate, fetch_bytes)
    }

    fn ensure_with_fetch(
        &self,
        entry: &ModelEntry,
        gate: &dyn LicenseGate,
        fetch: impl FnOnce(&str) -> Result<Vec<u8>>,
    ) -> Result<PathBuf> {
        let final_path = self.model_path(entry);
        if final_path.is_file() {
            return Ok(final_path);
        }

        if !gate.confirm(entry) {
            return Err(AudanError::Model(format!(
                "license for model '{}' {} was not accepted; not downloaded",
                entry.name, entry.version
            )));
        }

        let bytes = fetch(&entry.url)?;
        verify_checksum(&bytes, &entry.sha256)?;

        let dir = self.model_dir(entry);
        fs::create_dir_all(&dir)?;
        write_license(entry, &dir)?;
        install_atomically(&dir, &final_path, &bytes)?;

        Ok(final_path)
    }

    fn model_dir(&self, entry: &ModelEntry) -> PathBuf {
        self.root.join(&entry.name).join(&entry.version)
    }

    fn model_path(&self, entry: &ModelEntry) -> PathBuf {
        self.model_dir(entry).join(model_file_name(entry))
    }
}

fn model_file_name(entry: &ModelEntry) -> String {
    entry
        .url
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("model.bin")
        .to_string()
}

fn write_license(entry: &ModelEntry, dir: &Path) -> Result<()> {
    let mut text = format!("license: {}\n", entry.license_id);
    if let Some(url) = &entry.license_text_url {
        text.push_str(&format!("license text: {url}\n"));
    }
    if let Some(note) = &entry.license_note {
        text.push('\n');
        text.push_str(note);
        text.push('\n');
    }
    fs::write(dir.join("LICENSE"), text)?;
    Ok(())
}

fn install_atomically(dir: &Path, final_path: &Path, bytes: &[u8]) -> Result<()> {
    let file_name = final_path
        .file_name()
        .expect("model path always has a file name")
        .to_string_lossy();
    let tmp_path = dir.join(format!(".{file_name}.part"));
    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, final_path)?;
    Ok(())
}

fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| AudanError::Model(format!("failed to fetch model from {url}: {e}")))?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| AudanError::Model(format!("failed to read model body from {url}: {e}")))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sha256_hex(bytes: &[u8]) -> String {
        use std::fmt::Write;
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        let mut out = String::with_capacity(64);
        for byte in digest {
            write!(out, "{byte:02x}").unwrap();
        }
        out
    }

    fn test_entry(url: &str, bytes: &[u8]) -> ModelEntry {
        ModelEntry {
            name: "htdemucs".to_string(),
            version: "4.0".to_string(),
            url: url.to_string(),
            sha256: sha256_hex(bytes),
            size_bytes: bytes.len() as u64,
            license_id: "unclear".to_string(),
            license_text_url: None,
            license_note: Some("RISK-1: weights licensing is unresolved.".to_string()),
        }
    }

    struct AlwaysAccept;
    impl LicenseGate for AlwaysAccept {
        fn confirm(&self, _entry: &ModelEntry) -> bool {
            true
        }
    }

    struct AlwaysDecline;
    impl LicenseGate for AlwaysDecline {
        fn confirm(&self, _entry: &ModelEntry) -> bool {
            false
        }
    }

    struct PanicsIfCalled;
    impl LicenseGate for PanicsIfCalled {
        fn confirm(&self, _entry: &ModelEntry) -> bool {
            panic!("gate must not be consulted when the model is already installed");
        }
    }

    #[test]
    fn declined_license_returns_error_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf());
        let bytes = b"fake model weights";
        let entry = test_entry("https://models.audan.dev/htdemucs/4.0/model.onnx", bytes);

        let result = store.ensure_with_fetch(&entry, &AlwaysDecline, |_url| {
            panic!("must not fetch when the license was declined")
        });

        assert!(result.is_err());
        assert!(!store.is_installed(&entry));
        assert!(!dir.path().join("htdemucs").exists());
    }

    #[test]
    fn accepted_license_downloads_verifies_and_installs_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf());
        let bytes = b"fake model weights".to_vec();
        let entry = test_entry("https://models.audan.dev/htdemucs/4.0/model.onnx", &bytes);

        let fetch_calls = AtomicUsize::new(0);
        let path = store
            .ensure_with_fetch(&entry, &AlwaysAccept, |url| {
                fetch_calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(url, entry.url);
                Ok(bytes.clone())
            })
            .expect("ensure should succeed when the license is accepted");

        assert_eq!(fetch_calls.load(Ordering::SeqCst), 1);
        assert!(path.is_file());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(path.file_name().unwrap(), "model.onnx");

        let license_path = dir.path().join("htdemucs").join("4.0").join("LICENSE");
        assert!(license_path.is_file());
        let license_text = fs::read_to_string(&license_path).unwrap();
        assert!(license_text.contains("unclear"));
        assert!(license_text.contains("RISK-1"));

        for entry_result in fs::read_dir(dir.path().join("htdemucs").join("4.0")).unwrap() {
            let name = entry_result.unwrap().file_name();
            assert!(
                !name.to_string_lossy().ends_with(".part"),
                "no partial file should remain: {name:?}"
            );
        }

        assert!(store.is_installed(&entry));

        let second = store
            .ensure_with_fetch(&entry, &PanicsIfCalled, |_url| {
                panic!("must not fetch again once already installed")
            })
            .expect("second ensure() call must be a no-op success");
        assert_eq!(second, path);
    }

    #[test]
    fn checksum_mismatch_is_rejected_and_nothing_is_installed() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::new(dir.path().to_path_buf());
        let mut entry = test_entry(
            "https://models.audan.dev/htdemucs/4.0/model.onnx",
            b"expected bytes",
        );
        entry.sha256 = sha256_hex(b"some other bytes entirely");

        let result =
            store.ensure_with_fetch(&entry, &AlwaysAccept, |_url| Ok(b"expected bytes".to_vec()));

        assert!(result.is_err());
        assert!(!store.is_installed(&entry));
    }
}
