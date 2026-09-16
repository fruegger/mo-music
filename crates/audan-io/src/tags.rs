//! Read/write BPM and musical-key tags via `lofty`: ID3v2 `TBPM`/`TKEY` for
//! MP3/ID3-tagged containers, Vorbis-comment `BPM`/`KEY` for FLAC/Ogg (S3.2,
//! `audan tag --write bpm,key`). `lofty`'s [`lofty::tag::ItemKey::IntegerBpm`]
//! and [`lofty::tag::ItemKey::InitialKey`] already unify the ID3v2 and
//! Vorbis-comment field names, so a single write path covers both families;
//! see `read_tags`/`write_tags` below for the one place (`Bpm` vs
//! `IntegerBpm`) where the two families genuinely disagree on precision.

use std::path::Path;

use lofty::config::WriteOptions;
use lofty::prelude::*;
use lofty::probe::Probe;
use lofty::tag::Tag;

use audan_core::error::{AudanError, Result};

/// BPM/key values read back from a file's tags. `None` means the field was
/// absent, not that it was zero.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TagSnapshot {
    pub bpm: Option<f64>,
    pub key: Option<String>,
}

/// A partial update: only the fields that are `Some` are written, so a
/// caller can set BPM and key independently (`audan tag --write bpm,key`
/// vs `--write bpm`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TagUpdate {
    pub bpm: Option<f64>,
    pub key: Option<String>,
}

fn lofty_err(context: &str, path: &Path, e: impl std::fmt::Display) -> AudanError {
    AudanError::Decode(format!("{context} '{}': {e}", path.display()))
}

/// Read the BPM and musical key from `path`'s primary tag (falling back to
/// the first tag present, if any). Returns an empty [`TagSnapshot`] if the
/// file has no tags at all.
pub fn read_tags(path: &Path) -> Result<TagSnapshot> {
    let tagged_file = Probe::open(path)
        .map_err(|e| lofty_err("failed to open for tag reading", path, e))?
        .read()
        .map_err(|e| lofty_err("failed to read tags", path, e))?;

    let tag = match tagged_file
        .primary_tag()
        .or_else(|| tagged_file.first_tag())
    {
        Some(tag) => tag,
        None => return Ok(TagSnapshot::default()),
    };

    // ID3v2 restricts TBPM to integers (`IntegerBpm`); Vorbis comments carry
    // a decimal BPM field (`Bpm`). Prefer whichever is present, integer
    // first since it is the more common of the two in practice.
    let bpm = tag
        .get_string(&ItemKey::IntegerBpm)
        .or_else(|| tag.get_string(&ItemKey::Bpm))
        .and_then(|s| s.trim().parse::<f64>().ok());

    let key = tag.get_string(&ItemKey::InitialKey).map(str::to_string);

    Ok(TagSnapshot { bpm, key })
}

/// Write the fields present in `update` into `path`'s primary tag, creating
/// one if the file doesn't have it yet, and save in place. Fields left as
/// `None` in `update` are left untouched in the file.
pub fn write_tags(path: &Path, update: &TagUpdate) -> Result<()> {
    if update.bpm.is_none() && update.key.is_none() {
        return Ok(());
    }

    let mut tagged_file = Probe::open(path)
        .map_err(|e| lofty_err("failed to open for tag writing", path, e))?
        .read()
        .map_err(|e| lofty_err("failed to read tags before writing", path, e))?;

    let tag_type = tagged_file.primary_tag_type();
    if tagged_file.tag(tag_type).is_none() {
        tagged_file.insert_tag(Tag::new(tag_type));
    }
    let tag = tagged_file
        .tag_mut(tag_type)
        .expect("primary tag was just inserted if it was missing");

    if let Some(bpm) = update.bpm {
        // Write both representations; each tag format's own serializer
        // keeps only the item key(s) it actually maps to a frame/field
        // (S3.2: ID3v2 TBPM, Vorbis-comment BPM), so this is safe to do
        // unconditionally rather than switching on `tag_type`.
        tag.insert_text(ItemKey::IntegerBpm, format!("{}", bpm.round() as i64));
        tag.insert_text(ItemKey::Bpm, format!("{bpm}"));
    }
    if let Some(ref key) = update.key {
        tag.insert_text(ItemKey::InitialKey, key.clone());
    }

    tag.save_to_path(path, WriteOptions::default())
        .map_err(|e| lofty_err("failed to save tags", path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_wav(path: &Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..44100u32 {
            let t = i as f32 / 44100.0;
            let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            writer.write_sample((v * i16::MAX as f32) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn round_trips_bpm_and_key_on_wav_id3v2() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tagged.wav");
        synth_wav(&path);

        let update = TagUpdate {
            bpm: Some(128.0),
            key: Some("8B".to_string()),
        };
        write_tags(&path, &update).unwrap();

        let snapshot = read_tags(&path).unwrap();
        assert_eq!(snapshot.bpm, Some(128.0));
        assert_eq!(snapshot.key.as_deref(), Some("8B"));
    }

    #[test]
    fn partial_update_only_touches_requested_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("partial.wav");
        synth_wav(&path);

        write_tags(
            &path,
            &TagUpdate {
                bpm: Some(120.0),
                key: None,
            },
        )
        .unwrap();
        write_tags(
            &path,
            &TagUpdate {
                bpm: None,
                key: Some("5A".to_string()),
            },
        )
        .unwrap();

        let snapshot = read_tags(&path).unwrap();
        assert_eq!(snapshot.bpm, Some(120.0));
        assert_eq!(snapshot.key.as_deref(), Some("5A"));
    }

    #[test]
    fn no_op_update_does_not_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("untouched.wav");
        synth_wav(&path);
        write_tags(&path, &TagUpdate::default()).unwrap();
    }

    #[test]
    fn missing_tags_read_back_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("untagged.wav");
        synth_wav(&path);
        let snapshot = read_tags(&path).unwrap();
        assert_eq!(snapshot, TagSnapshot::default());
    }
}
