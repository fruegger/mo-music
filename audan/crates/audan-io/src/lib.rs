//! `audan-io` -- L0/L1 of the pipeline (S5.1): decode to `f32`
//! ([`decode`]), strip encoder delay (S8.4, inside [`decode`]), resample to
//! the canonical analysis rates ([`resample`]), read/write BPM/key tags
//! ([`tags`]), and fall back to an `ffmpeg` subprocess for formats
//! `symphonia` doesn't cover natively ([`ffmpeg`], ADR-8). [`probe`] builds
//! the small report `audan probe` (F1) needs on top of the other modules.

pub mod decode;
pub mod ffmpeg;
pub mod probe;
pub mod resample;
pub mod tags;

pub use decode::{decode_file, DecodeReport};
pub use ffmpeg::{decode_via_ffmpeg, detect_ffmpeg_version, FfmpegFallbackError};
pub use probe::{probe_file, probe_signal, ProbeReport};
pub use resample::{mixdown_mono, resample_mono, to_analysis_rate, to_stems_rate};
pub use tags::{read_tags, write_tags, TagSnapshot, TagUpdate};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use audan_core::signal::{ANALYSIS_SAMPLE_RATE, STEMS_SAMPLE_RATE};

    fn synth_wav_file(dir: &std::path::Path, seconds: f32) -> std::path::PathBuf {
        let path = dir.join("tone.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        let n_frames = (44100.0 * seconds) as u32;
        for i in 0..n_frames {
            let t = i as f32 / 44100.0;
            let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let sample = (v * i16::MAX as f32 * 0.8) as i16;
            writer.write_sample(sample).unwrap();
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        path
    }

    #[test]
    fn decode_round_trips_known_wav_duration() {
        let dir = tempfile::tempdir().unwrap();
        let path = synth_wav_file(dir.path(), 2.0);

        let (signal, report) = decode_file(&path).unwrap();

        assert_eq!(signal.sample_rate, 44100);
        assert_eq!(signal.channels, 2);
        assert_eq!(signal.num_frames(), 44100 * 2);
        assert!((signal.duration_seconds() - 2.0).abs() < 1e-3);
        assert_eq!(
            report.delay_stripped_samples, 0,
            "WAV has no encoder delay concept"
        );
        assert_eq!(report.container_format, "wav");
        assert_eq!(report.decoder_id, "symphonia");
    }

    #[test]
    fn decode_then_resample_to_analysis_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = synth_wav_file(dir.path(), 3.0);

        let (signal, _report) = decode_file(&path).unwrap();
        let mono = to_analysis_rate(&signal).unwrap();

        assert_eq!(mono.sample_rate, ANALYSIS_SAMPLE_RATE);
        let duration_diff = (mono.duration_seconds() - signal.duration_seconds()).abs();
        assert!(duration_diff < 0.05, "duration drifted by {duration_diff}s");
    }

    #[test]
    fn decode_then_resample_to_stems_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = synth_wav_file(dir.path(), 1.0);

        let (signal, _report) = decode_file(&path).unwrap();
        let mono = to_stems_rate(&signal).unwrap();

        assert_eq!(mono.sample_rate, STEMS_SAMPLE_RATE);
    }

    #[test]
    fn probe_reports_sane_fields_for_synthesized_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = synth_wav_file(dir.path(), 1.0);

        let report = probe_file(&path).unwrap();

        assert!((report.duration_seconds - 1.0).abs() < 1e-3);
        assert_eq!(report.format, "wav");
        assert_eq!(report.delay_stripped_samples, 0);
        // A 0.8-scaled sine never reaches the clip threshold.
        assert_eq!(report.clipped_sample_count, 0);
        assert!(report.peak_amplitude < 0.9);
    }

    #[test]
    fn ffmpeg_fallback_transcodes_and_redecodes_when_available() {
        if ffmpeg::detect_ffmpeg_version().is_none() {
            println!("ffmpeg not found on PATH; skipping ffmpeg fallback test");
            return;
        }

        // We have no real Opus/WMA fixture to force symphonia to refuse, so
        // this exercises `decode_via_ffmpeg` directly against our
        // synthesized WAV: spawn ffmpeg, transcode to a temp WAV, redecode
        // it, and check the reported decoder identity and audio content.
        let dir = tempfile::tempdir().unwrap();
        let path = synth_wav_file(dir.path(), 1.0);

        let (signal, report) = ffmpeg::decode_via_ffmpeg(&path).unwrap_or_else(|e| match e {
            ffmpeg::FfmpegFallbackError::NotAvailable => {
                panic!("ffmpeg reported available moments ago but is now NotAvailable")
            }
            ffmpeg::FfmpegFallbackError::Failed(msg) => panic!("ffmpeg fallback failed: {msg}"),
        });

        assert!(report.decoder_id.starts_with("ffmpeg "));
        assert_eq!(report.container_format, "wav");
        assert!((signal.duration_seconds() - 1.0).abs() < 0.05);
    }

    #[test]
    fn unsupported_extension_with_garbage_bytes_errors_clearly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mystery.opus");
        std::fs::write(&path, b"not actually audio").unwrap();

        let err = decode_file(&path).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("opus") || message.contains("ffmpeg"),
            "expected a clear format/ffmpeg message, got: {message}"
        );
    }
}
