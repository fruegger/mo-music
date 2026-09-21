//! Decode a file to an [`audan_core::Signal`] via `symphonia`, stripping
//! encoder delay where the container/codec combination exposes it (S8.4).
//!
//! Falls back to the [`crate::ffmpeg`] subprocess path (ADR-8, RISK-5) when
//! `symphonia` cannot open the container or construct a decoder for its
//! codec.

use std::fs::File;
use std::io::ErrorKind;
use std::path::Path;

use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::{MetadataOptions, Value};
use symphonia::core::probe::Hint;

use audan_core::error::AudanError;
use audan_core::signal::Signal;

use crate::ffmpeg::{self, FfmpegFallbackError};

/// Tracks the workspace-pinned `symphonia` release (see the root
/// `Cargo.toml` `[workspace.dependencies]` entry). `symphonia` does not
/// expose its own version as a runtime constant, so this must be bumped by
/// hand alongside that pin.
pub const SYMPHONIA_VERSION: &str = "0.5";

/// Identity and encoder-delay bookkeeping for one decode, alongside the
/// [`Signal`] it produced. `decoder_id`/`decoder_version` are pinned into the
/// L0 cache key by `audan-cache` because two decoders (or two `symphonia`
/// versions) can disagree on where the music starts (S8.4).
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeReport {
    /// Which decoder produced this signal, e.g. `"symphonia"` or
    /// `"ffmpeg 6.1.1"` (the `ffmpeg` version string is folded in per S8.4).
    pub decoder_id: String,
    pub decoder_version: String,
    /// Leading samples (frames) discarded because the container/codec
    /// declared encoder delay. Always `0` for WAV/FLAC, which have no such
    /// concept.
    pub delay_stripped_samples: u64,
    /// Best-effort container/codec label, currently the lowercased file
    /// extension (e.g. `"mp3"`, `"flac"`, `"m4a"`).
    pub container_format: String,
}

pub(crate) enum NativeDecodeError {
    Io(std::io::Error),
    Unsupported(String),
}

impl std::fmt::Display for NativeDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeDecodeError::Io(e) => write!(f, "I/O error: {e}"),
            NativeDecodeError::Unsupported(msg) => write!(f, "{msg}"),
        }
    }
}

fn container_format_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Decode `path` to a [`Signal`], trying `symphonia` first and falling back
/// to the `ffmpeg` subprocess (S3.2, ADR-8, RISK-5) when `symphonia` cannot
/// handle the container or codec.
pub fn decode_file(path: &Path) -> audan_core::error::Result<(Signal, DecodeReport)> {
    match decode_with_symphonia(path) {
        Ok(result) => Ok(result),
        Err(NativeDecodeError::Io(e)) => Err(AudanError::Io(e)),
        Err(NativeDecodeError::Unsupported(native_msg)) => match ffmpeg::decode_via_ffmpeg(path) {
            Ok(result) => Ok(result),
            Err(FfmpegFallbackError::NotAvailable) => {
                let ext = container_format_of(path);
                Err(AudanError::UnsupportedFormat(format!(
                    "'{}': symphonia cannot decode this {ext} file ({native_msg}); \
                         ffmpeg was not found on PATH -- install ffmpeg to decode this format",
                    path.display()
                )))
            }
            Err(FfmpegFallbackError::Failed(ffmpeg_msg)) => {
                let ext = container_format_of(path);
                Err(AudanError::UnsupportedFormat(format!(
                    "'{}': symphonia cannot decode this {ext} file ({native_msg}); \
                         ffmpeg fallback also failed: {ffmpeg_msg}",
                    path.display()
                )))
            }
        },
    }
}

pub(crate) fn decode_with_symphonia(
    path: &Path,
) -> Result<(Signal, DecodeReport), NativeDecodeError> {
    let file = File::open(path).map_err(NativeDecodeError::Io)?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let metadata_opts = MetadataOptions::default();

    let mut probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| NativeDecodeError::Unsupported(format!("no container reader: {e}")))?;

    let track = probed
        .format
        .default_track()
        .cloned()
        .ok_or_else(|| NativeDecodeError::Unsupported("no default audio track".to_string()))?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| NativeDecodeError::Unsupported(format!("no decoder for codec: {e}")))?;

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut out_spec: Option<SignalSpec> = None;
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match probed.format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(SymphoniaError::ResetRequired) => {
                decoder = symphonia::default::get_codecs()
                    .make(&track.codec_params, &DecoderOptions::default())
                    .map_err(|e| {
                        NativeDecodeError::Unsupported(format!("decoder reset failed: {e}"))
                    })?;
                continue;
            }
            Err(e) => return Err(NativeDecodeError::Unsupported(format!("demux error: {e}"))),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = *audio_buf.spec();
                if sample_buf.is_none() {
                    sample_buf = Some(SampleBuffer::<f32>::new(audio_buf.capacity() as u64, spec));
                    out_spec = Some(spec);
                }
                if let Some(buf) = sample_buf.as_mut() {
                    buf.copy_interleaved_ref(audio_buf);
                    samples.extend_from_slice(buf.samples());
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(NativeDecodeError::Unsupported(format!("decode error: {e}"))),
        }
    }

    let spec = out_spec
        .ok_or_else(|| NativeDecodeError::Unsupported("stream produced no audio".to_string()))?;
    let channels = spec.channels.count() as u16;

    // MP3 (LAME/Xing) and Ogg Vorbis expose delay/padding through
    // `symphonia`'s gapless-playback support (`FormatOptions::enable_gapless`);
    // when present, the demuxer/decoder pair already trimmed the samples
    // above and recorded how much on the track's codec parameters.
    let native_delay = track.codec_params.delay.unwrap_or(0) as u64;

    let mut delay_stripped_samples = native_delay;

    // TODO: MP4 containers (AAC-LC, ALAC) carry encoder delay/padding in the
    // `iTunSMPB` freeform atom, not through any mechanism `symphonia` 0.5's
    // `FormatReader`/`Decoder` traits expose as packet trim info (verified
    // against symphonia-format-isomp4 0.5.5: it parses `elst` edit-list atoms
    // into an unused `EdtsAtom`, and never calls
    // `symphonia_core::formats::util::trim_packet`, and its AAC/ALAC decoders
    // never call `AudioBuffer::trim`). We therefore parse `iTunSMPB`
    // ourselves below and trim manually. A raw ADTS `.aac` stream (no MP4
    // container) has no equivalent tag at all and is not handled -- such
    // streams report `delay_stripped_samples == 0` even if the encoder
    // inserted priming samples, because there is nowhere standard to read
    // that information from.
    if native_delay == 0 {
        if let Some((delay, padding)) = find_itunsmpb(&mut probed) {
            trim_frames(&mut samples, channels, delay, padding);
            delay_stripped_samples = u64::from(delay);
        }
    }

    let signal = Signal {
        sample_rate: spec.rate,
        channels,
        samples,
    };

    let report = DecodeReport {
        decoder_id: "symphonia".to_string(),
        decoder_version: SYMPHONIA_VERSION.to_string(),
        delay_stripped_samples,
        container_format: container_format_of(path),
    };

    Ok((signal, report))
}

fn trim_frames(samples: &mut Vec<f32>, channels: u16, delay: u32, padding: u32) {
    let channels = channels as usize;
    if channels == 0 {
        return;
    }
    let total_frames = samples.len() / channels;
    let delay = (delay as usize).min(total_frames);
    let remaining_frames = total_frames - delay;
    let padding = (padding as usize).min(remaining_frames);

    let start = delay * channels;
    let end = samples.len() - padding * channels;
    if start >= end {
        samples.clear();
    } else {
        samples.drain(..start);
        let new_len = end - start;
        samples.truncate(new_len);
    }
}

/// Look for an iTunes `iTunSMPB` freeform tag (`com.apple.iTunes:iTunSMPB`)
/// in either the out-of-band metadata read during probing or the container's
/// own metadata revision, and parse its encoder delay / padding fields.
///
/// Format (space-separated hex fields, as written by `afconvert`/iTunes):
/// ` 00000000 <delay> <padding> <original-sample-count, 16 hex digits> ...`.
fn find_itunsmpb(probed: &mut symphonia::core::probe::ProbeResult) -> Option<(u32, u32)> {
    if let Some(metadata) = probed.metadata.get() {
        if let Some(rev) = metadata.current() {
            if let Some(parsed) = parse_itunsmpb_tags(rev.tags()) {
                return Some(parsed);
            }
        }
    }
    if let Some(rev) = probed.format.metadata().current() {
        if let Some(parsed) = parse_itunsmpb_tags(rev.tags()) {
            return Some(parsed);
        }
    }
    None
}

fn parse_itunsmpb_tags(tags: &[symphonia::core::meta::Tag]) -> Option<(u32, u32)> {
    tags.iter()
        .find(|t| t.key.to_ascii_lowercase().ends_with("itunsmpb"))
        .and_then(|t| match &t.value {
            Value::String(s) => parse_itunsmpb_value(s),
            _ => None,
        })
}

fn parse_itunsmpb_value(value: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let delay = u32::from_str_radix(parts[1], 16).ok()?;
    let padding = u32::from_str_radix(parts[2], 16).ok()?;
    Some((delay, padding))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_itunsmpb_value() {
        let value = " 00000000 00000840 00000164 0000000000078A58 00000000 00000000 \
                      00000000 00000000 00000000 00000000 00000000";
        let (delay, padding) = parse_itunsmpb_value(value).unwrap();
        assert_eq!(delay, 0x840);
        assert_eq!(padding, 0x164);
    }

    #[test]
    fn rejects_malformed_itunsmpb_value() {
        assert!(parse_itunsmpb_value("not enough fields").is_none());
    }

    #[test]
    fn trims_delay_and_padding_from_interleaved_frames() {
        // 2 channels, 10 frames: values 0..10 on ch0, 100..110 on ch1.
        let mut samples: Vec<f32> = (0..10).flat_map(|i| [i as f32, (100 + i) as f32]).collect();
        trim_frames(&mut samples, 2, 2, 3);
        // Frames 2..7 remain (5 frames).
        let expected: Vec<f32> = (2..7).flat_map(|i| [i as f32, (100 + i) as f32]).collect();
        assert_eq!(samples, expected);
    }
}
