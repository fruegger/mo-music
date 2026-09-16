//! `audan probe` support (F1, S3.2): format, duration, a simple loudness
//! measure, clipping, and decoder/delay identity, computed from an already
//! decoded [`Signal`] plus its [`DecodeReport`]. Rendering this to JSON/table
//! output is `audan-format`'s and `audan-cli`'s job, not this crate's.

use std::path::Path;

use audan_core::error::Result;
use audan_core::signal::Signal;

use crate::decode::{decode_file, DecodeReport};

/// A sample at or above this absolute value counts as clipped. Slightly
/// below `1.0` to also catch inter-sample peaks that round to exactly
/// full-scale after decode.
const CLIP_THRESHOLD: f32 = 0.999;

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeReport {
    /// Container/codec label, currently the lowercased file extension.
    pub format: String,
    pub duration_seconds: f64,
    /// Root-mean-square level over the whole signal, in linear amplitude
    /// (not dBFS, not LUFS). This is a simplification, not full EBU R128
    /// integrated loudness -- no gating, no K-weighting, no channel
    /// weighting. Adequate as a rough "how hot is this file" signal, not as
    /// a loudness-standard measurement.
    pub rms_amplitude: f32,
    pub peak_amplitude: f32,
    pub clipped_sample_count: u64,
    pub clipped_sample_fraction: f64,
    pub delay_stripped_samples: u64,
    pub decoder_id: String,
    pub decoder_version: String,
}

/// Compute a [`ProbeReport`] from an already-decoded signal and its decode
/// report. Kept separate from [`probe_file`] so callers that already
/// decoded (e.g. to also resample) don't pay for a second decode.
pub fn probe_signal(signal: &Signal, decode_report: &DecodeReport) -> ProbeReport {
    let n = signal.samples.len();

    let (sum_sq, peak, clipped) =
        signal
            .samples
            .iter()
            .fold((0f64, 0f32, 0u64), |(sum_sq, peak, clipped), &s| {
                let a = s.abs();
                let clipped = clipped + u64::from(a >= CLIP_THRESHOLD);
                (sum_sq + f64::from(s) * f64::from(s), peak.max(a), clipped)
            });

    let rms_amplitude = if n > 0 {
        (sum_sq / n as f64).sqrt() as f32
    } else {
        0.0
    };
    let clipped_sample_fraction = if n > 0 {
        clipped as f64 / n as f64
    } else {
        0.0
    };

    ProbeReport {
        format: decode_report.container_format.clone(),
        duration_seconds: signal.duration_seconds(),
        rms_amplitude,
        peak_amplitude: peak,
        clipped_sample_count: clipped,
        clipped_sample_fraction,
        delay_stripped_samples: decode_report.delay_stripped_samples,
        decoder_id: decode_report.decoder_id.clone(),
        decoder_version: decode_report.decoder_version.clone(),
    }
}

/// Decode `path` and compute its [`ProbeReport`] in one call.
pub fn probe_file(path: &Path) -> Result<ProbeReport> {
    let (signal, report) = decode_file(path)?;
    Ok(probe_signal(&signal, &report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(decoder_id: &str, delay: u64) -> DecodeReport {
        DecodeReport {
            decoder_id: decoder_id.to_string(),
            decoder_version: "0.5".to_string(),
            delay_stripped_samples: delay,
            container_format: "wav".to_string(),
        }
    }

    #[test]
    fn quiet_sine_has_no_clipping() {
        let n = 44100;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let signal = Signal {
            sample_rate: 44100,
            channels: 1,
            samples,
        };

        let report = probe_signal(&signal, &report("symphonia", 0));

        assert!((report.duration_seconds - 1.0).abs() < 1e-6);
        assert_eq!(report.clipped_sample_count, 0);
        assert!(report.peak_amplitude <= 0.5 + 1e-6);
        assert!(report.rms_amplitude > 0.0 && report.rms_amplitude < 0.5);
    }

    #[test]
    fn full_scale_square_wave_is_all_clipped() {
        let samples = vec![1.0f32, -1.0, 1.0, -1.0];
        let signal = Signal {
            sample_rate: 44100,
            channels: 1,
            samples,
        };

        let report = probe_signal(&signal, &report("symphonia", 0));

        assert_eq!(report.clipped_sample_count, 4);
        assert!((report.clipped_sample_fraction - 1.0).abs() < 1e-9);
    }

    #[test]
    fn carries_delay_and_decoder_identity_through() {
        let signal = Signal {
            sample_rate: 44100,
            channels: 1,
            samples: vec![0.0; 100],
        };
        let report = probe_signal(&signal, &report("ffmpeg 6.1.1", 1057));
        assert_eq!(report.delay_stripped_samples, 1057);
        assert_eq!(report.decoder_id, "ffmpeg 6.1.1");
    }
}
