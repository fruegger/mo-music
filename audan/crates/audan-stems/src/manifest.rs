//! The stem manifest sidecar (S3.2 "Stems" output row: "WAV or FLAC plus a
//! JSON manifest"). Writing the actual audio files is `audan-io`'s /
//! `audan-cli`'s job; this crate only owns the manifest shape.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StemManifest {
    pub schema_version: u32,
    pub backend: String,
    pub sample_rate: u32,
    /// Track names, matching the WAV/FLAC files written alongside this
    /// manifest.
    pub stems: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_manifest_round_trips_through_json() {
        let manifest = StemManifest {
            schema_version: 1,
            backend: "htdemucs".to_string(),
            sample_rate: audan_core::STEMS_SAMPLE_RATE,
            stems: vec![
                "vocals".to_string(),
                "drums".to_string(),
                "bass".to_string(),
                "other".to_string(),
            ],
        };

        let json = serde_json::to_string(&manifest).expect("serialize must succeed");
        let round_tripped: StemManifest =
            serde_json::from_str(&json).expect("deserialize must succeed");

        assert_eq!(round_tripped, manifest);
    }

    #[test]
    fn stem_manifest_json_carries_schema_version_field() {
        let manifest = StemManifest {
            schema_version: 1,
            backend: "htdemucs".to_string(),
            sample_rate: 44_100,
            stems: vec!["vocals".to_string()],
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"schema_version\":1"));
    }
}
