use serde_json::{json, Value};

/// One JAMS observation: `{"time", "duration", "value", "confidence"}`. `value` is
/// an arbitrary `serde_json::Value` so a chord caller can pass `json!("C:maj")`, a
/// beat caller a bare number, a section caller a label string, etc. -- this crate
/// has no chord/beat/section types of its own to be specific about.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JamsObservation {
    pub time: f64,
    pub duration: f64,
    pub value: Value,
    pub confidence: Option<f32>,
}

/// One JAMS annotation: a namespace name plus its observations.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct JamsAnnotation {
    pub namespace: String,
    pub data: Vec<JamsObservation>,
}

/// Renders a JAMS (JSON Annotated Music Specification) document:
/// `{"file_metadata": {...}, "annotations": [...]}`. Export-only (ADR-10) --
/// nothing in `audan` reads JAMS back in, so there is no matching reader.
///
/// Fields with no home in JAMS (e.g. `audan`'s tempo stability class) are simply
/// not written; that loss is expected and documented per ADR-10.
pub fn write_jams(annotations: &[JamsAnnotation], file_duration_seconds: f64) -> Value {
    let annotations_json: Vec<Value> = annotations
        .iter()
        .map(|ann| {
            let data: Vec<Value> = ann
                .data
                .iter()
                .map(|obs| {
                    json!({
                        "time": obs.time,
                        "duration": obs.duration,
                        "value": obs.value,
                        "confidence": obs.confidence,
                    })
                })
                .collect();
            json!({
                "namespace": ann.namespace,
                "data": data,
                "annotation_metadata": {
                    "corpus": "",
                    "version": "",
                    "annotator": {},
                    "annotation_tools": "audan",
                    "annotation_rules": "",
                    "validation": "",
                    "data_source": "audan",
                },
                "sandbox": {},
            })
        })
        .collect();

    json!({
        "file_metadata": {
            "title": "",
            "artist": "",
            "duration": file_duration_seconds,
            "identifiers": {},
            "jams_version": "0.3.0",
        },
        "annotations": annotations_json,
        "sandbox": {},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_a_chord_annotation() {
        let ann = JamsAnnotation {
            namespace: "chord".into(),
            data: vec![
                JamsObservation {
                    time: 0.0,
                    duration: 0.512,
                    value: json!("N"),
                    confidence: None,
                },
                JamsObservation {
                    time: 0.512,
                    duration: 0.461,
                    value: json!("C:maj"),
                    confidence: Some(0.83),
                },
            ],
        };
        let doc = write_jams(std::slice::from_ref(&ann), 180.0);

        assert_eq!(doc["file_metadata"]["duration"], 180.0);
        let data = doc["annotations"][0]["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(doc["annotations"][0]["namespace"], "chord");
        assert_eq!(data[1]["value"], "C:maj");
        assert!((data[1]["confidence"].as_f64().unwrap() - 0.83).abs() < 1e-6);
        assert!(data[0]["confidence"].is_null());
    }

    #[test]
    fn multiple_annotations_are_independent() {
        let chords = JamsAnnotation {
            namespace: "chord".into(),
            data: vec![],
        };
        let beats = JamsAnnotation {
            namespace: "beat".into(),
            data: vec![JamsObservation {
                time: 0.512,
                duration: 0.0,
                value: json!(1),
                confidence: Some(0.95),
            }],
        };
        let doc = write_jams(&[chords, beats], 10.0);
        assert_eq!(doc["annotations"].as_array().unwrap().len(), 2);
        assert_eq!(doc["annotations"][1]["namespace"], "beat");
    }
}
