use std::path::Path;

use audan_core::{AudanError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub version: String,
    pub url: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub license_id: String,
    #[serde(default)]
    pub license_text_url: Option<String>,
    /// Surfaces cases like RISK-1 (Demucs weights licensing is genuinely
    /// unresolved) verbatim to the user rather than folding an ambiguous
    /// situation into a single reassuring `license_id`.
    #[serde(default)]
    pub license_note: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Registry {
    pub models: Vec<ModelEntry>,
}

impl Registry {
    pub fn find(&self, name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.name == name)
    }

    /// A manifest baked into the binary at compile time so lookups (though
    /// not downloads) work with no network, matching the "bare `audan beats`
    /// works offline" spirit of ADR-7.
    pub fn default_embedded() -> Registry {
        const EMBEDDED: &str = include_str!("../../../resources/models/registry.json");
        serde_json::from_str(EMBEDDED).expect("embedded resources/models/registry.json must parse")
    }
}

pub fn load(path: &Path) -> Result<Registry> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| {
        AudanError::Model(format!("invalid registry manifest {}: {e}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_registry_parses_and_has_expected_models() {
        let registry = Registry::default_embedded();
        let beat_this = registry.find("beat_this").expect("beat_this entry present");
        assert_eq!(beat_this.license_id, "MIT");
        assert!(beat_this.license_note.is_some());

        let htdemucs = registry.find("htdemucs").expect("htdemucs entry present");
        assert_eq!(htdemucs.license_id, "unclear");
        let note = htdemucs
            .license_note
            .as_ref()
            .expect("htdemucs must carry a license_note");
        assert!(note.contains("RISK-1") || note.contains("unresolved"));
    }

    #[test]
    fn load_reads_the_resource_file_from_disk() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/models/registry.json");
        let registry = load(&path).expect("resources/models/registry.json must load");
        assert_eq!(registry.models.len(), 2);
    }

    #[test]
    fn minimal_hand_written_json_parses_including_license_note() {
        let hash = "0".repeat(64);
        let json = format!(
            r#"{{
                "models": [
                    {{
                        "name": "example",
                        "version": "0.1",
                        "url": "https://models.audan.dev/example/0.1/model.onnx",
                        "sha256": "{hash}",
                        "size_bytes": 1,
                        "license_id": "unclear",
                        "license_note": "ambiguous on purpose"
                    }}
                ]
            }}"#
        );
        let registry: Registry = serde_json::from_str(&json).expect("minimal JSON must parse");
        let entry = &registry.models[0];
        assert_eq!(entry.name, "example");
        assert_eq!(entry.license_note.as_deref(), Some("ambiguous on purpose"));
        assert!(entry.license_text_url.is_none());
    }
}
