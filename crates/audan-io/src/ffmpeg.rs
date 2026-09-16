//! Subprocess fallback for formats `symphonia` cannot open (ADR-8, RISK-5):
//! Opus, WMA, WavPack, and video containers. `ffmpeg` is invoked as a child
//! process and never linked, so its license never attaches to `audan`
//! (S2.2). Absence degrades gracefully: [`decode_via_ffmpeg`] reports
//! [`FfmpegFallbackError::NotAvailable`] rather than failing loudly, and the
//! caller (`decode::decode_file`) turns that into an error naming the format
//! and suggesting `ffmpeg`.

use std::path::Path;
use std::process::Command;

use crate::decode::{decode_with_symphonia, DecodeReport, NativeDecodeError};
use audan_core::signal::Signal;

pub enum FfmpegFallbackError {
    /// No usable `ffmpeg` was found on `PATH`.
    NotAvailable,
    /// `ffmpeg` was found but the transcode or the subsequent WAV decode
    /// failed.
    Failed(String),
}

/// Runs `ffmpeg -version` and extracts the version token from its first
/// output line (`"ffmpeg version 6.1.1 Copyright ..."`). Returns `None` if
/// `ffmpeg` is not on `PATH` or does not behave like `ffmpeg`.
pub fn detect_ffmpeg_version() -> Option<String> {
    let output = Command::new("ffmpeg").arg("-version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let first_line = text.lines().next()?;
    let version = first_line.split_whitespace().nth(2)?;
    Some(version.to_string())
}

/// Transcode `path` to a temporary WAV file via an `ffmpeg` subprocess, then
/// decode that WAV natively. The temporary file is cleaned up when the
/// backing `TempDir` is dropped, including on early return.
pub fn decode_via_ffmpeg(path: &Path) -> Result<(Signal, DecodeReport), FfmpegFallbackError> {
    let version = detect_ffmpeg_version().ok_or(FfmpegFallbackError::NotAvailable)?;

    let tmp_dir = tempfile::Builder::new()
        .prefix("audan-io-ffmpeg-")
        .tempdir()
        .map_err(|e| FfmpegFallbackError::Failed(format!("could not create temp dir: {e}")))?;
    let wav_path = tmp_dir.path().join("transcoded.wav");

    let output = Command::new("ffmpeg")
        .arg("-nostdin")
        .arg("-y")
        .arg("-i")
        .arg(path)
        .arg("-vn")
        .arg("-acodec")
        .arg("pcm_f32le")
        .arg(&wav_path)
        .output()
        .map_err(|e| FfmpegFallbackError::Failed(format!("failed to spawn ffmpeg: {e}")))?;

    if !output.status.success() {
        return Err(FfmpegFallbackError::Failed(format!(
            "ffmpeg exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    // Decoding the ffmpeg-produced WAV natively (rather than looping back
    // through `decode::decode_file`) avoids any risk of recursing into this
    // fallback again, and WAV always succeeds natively.
    let (signal, _wav_report) = decode_with_symphonia(&wav_path).map_err(|e| match e {
        NativeDecodeError::Io(io_err) => {
            FfmpegFallbackError::Failed(format!("could not read ffmpeg output: {io_err}"))
        }
        NativeDecodeError::Unsupported(msg) => {
            FfmpegFallbackError::Failed(format!("could not decode ffmpeg output: {msg}"))
        }
    })?;

    // Whatever encoder delay the original container declared was stripped
    // (or not) by ffmpeg's own decoder before it ever reached the WAV we
    // just read; we have no independent way to observe that from here, so
    // `delay_stripped_samples` is reported as 0 rather than guessed at.
    let container_format = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_else(|| "unknown".to_string());

    let report = DecodeReport {
        decoder_id: format!("ffmpeg {version}"),
        decoder_version: version,
        delay_stripped_samples: 0,
        container_format,
    };

    Ok((signal, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_ffmpeg_or_reports_absence_cleanly() {
        match detect_ffmpeg_version() {
            Some(v) => println!("ffmpeg detected: {v}"),
            None => println!("ffmpeg not found on PATH; treating as absent (expected in CI)"),
        }
    }
}
