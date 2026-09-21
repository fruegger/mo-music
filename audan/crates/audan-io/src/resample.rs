//! Mono mixdown and resampling to the canonical L1 analysis rates
//! (S5.2): [`audan_core::ANALYSIS_SAMPLE_RATE`] for beat/key/chord/structure
//! work and [`audan_core::STEMS_SAMPLE_RATE`] for stem separation.

use rubato::{FftFixedIn, Resampler};

use audan_core::error::{AudanError, Result};
use audan_core::signal::{MonoSignal, Signal, ANALYSIS_SAMPLE_RATE, STEMS_SAMPLE_RATE};

/// Number of input frames processed per resampler call. Arbitrary but must
/// be sizeable relative to typical sample-rate ratios; 4096 keeps the number
/// of `process`/`process_partial` calls reasonable for a multi-minute track
/// while staying small enough not to matter for memory.
const CHUNK_FRAMES: usize = 4096;

/// Average all channels of an interleaved [`Signal`] down to one channel,
/// at the signal's native sample rate.
pub fn mixdown_mono(signal: &Signal) -> Vec<f32> {
    let channels = signal.channels as usize;
    if channels <= 1 {
        return signal.samples.clone();
    }

    let frames = signal.num_frames();
    let mut mono = Vec::with_capacity(frames);
    for frame in signal.samples.chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        mono.push(sum / channels as f32);
    }
    mono
}

/// Mix `signal` down to mono and resample it to `target_rate` using
/// `rubato`'s FFT-based fixed-input resampler.
pub fn resample_mono(signal: &Signal, target_rate: u32) -> Result<MonoSignal> {
    let mono = mixdown_mono(signal);

    if mono.is_empty() {
        return Ok(MonoSignal {
            sample_rate: target_rate,
            samples: Vec::new(),
        });
    }
    if signal.sample_rate == target_rate {
        return Ok(MonoSignal {
            sample_rate: target_rate,
            samples: mono,
        });
    }

    let mut resampler = FftFixedIn::<f32>::new(
        signal.sample_rate as usize,
        target_rate as usize,
        CHUNK_FRAMES,
        2,
        1,
    )
    .map_err(|e| AudanError::Decode(format!("failed to construct resampler: {e}")))?;

    let estimated_out_len =
        (mono.len() as u64 * u64::from(target_rate) / u64::from(signal.sample_rate)) as usize;
    let mut out_samples = Vec::with_capacity(estimated_out_len + CHUNK_FRAMES);

    let mut pos = 0usize;
    while pos < mono.len() {
        let need = resampler.input_frames_next();
        let end = (pos + need).min(mono.len());
        let chunk = [&mono[pos..end]];
        let is_partial = end - pos < need;

        let produced = if is_partial {
            resampler.process_partial(Some(&chunk), None)
        } else {
            resampler.process(&chunk, None)
        }
        .map_err(|e| AudanError::Decode(format!("resample failed: {e}")))?;

        out_samples.extend_from_slice(&produced[0]);
        pos = end;
    }

    // The final chunk is zero-padded up to `need` frames before being fed to
    // the resampler (see `process_partial`'s doc comment), so it yields a
    // few extra resampled frames derived from that padding. Truncate to the
    // exact length implied by the input length and the rate ratio so
    // `duration_seconds()` matches the source signal rather than overrunning
    // by up to one resampler chunk.
    let expected_out_len =
        (mono.len() as u128 * u128::from(target_rate) / u128::from(signal.sample_rate)) as usize;
    out_samples.truncate(expected_out_len);

    Ok(MonoSignal {
        sample_rate: target_rate,
        samples: out_samples,
    })
}

/// Resample to [`ANALYSIS_SAMPLE_RATE`] (22050 Hz), the L1 signal consumed
/// by beat/key/chord/structure analysis.
pub fn to_analysis_rate(signal: &Signal) -> Result<MonoSignal> {
    resample_mono(signal, ANALYSIS_SAMPLE_RATE)
}

/// Resample to [`STEMS_SAMPLE_RATE`] (44100 Hz), the L1 signal consumed by
/// stem separation.
pub fn to_stems_rate(signal: &Signal) -> Result<MonoSignal> {
    resample_mono(signal, STEMS_SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(sample_rate: u32, channels: u16, freq: f32, seconds: f32) -> Signal {
        let n_frames = (sample_rate as f32 * seconds) as usize;
        let mut samples = Vec::with_capacity(n_frames * channels as usize);
        for i in 0..n_frames {
            let t = i as f32 / sample_rate as f32;
            let v = (2.0 * std::f32::consts::PI * freq * t).sin();
            for _ in 0..channels {
                samples.push(v);
            }
        }
        Signal {
            sample_rate,
            channels,
            samples,
        }
    }

    #[test]
    fn mixdown_averages_channels() {
        let signal = Signal {
            sample_rate: 44100,
            channels: 2,
            samples: vec![1.0, -1.0, 0.5, 0.5],
        };
        let mono = mixdown_mono(&signal);
        assert_eq!(mono, vec![0.0, 0.5]);
    }

    #[test]
    fn mixdown_is_identity_for_mono() {
        let signal = Signal {
            sample_rate: 44100,
            channels: 1,
            samples: vec![0.1, 0.2, 0.3],
        };
        assert_eq!(mixdown_mono(&signal), vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn resamples_44100_to_22050_preserving_duration() {
        let signal = sine_signal(44100, 2, 440.0, 2.0);
        let mono = resample_mono(&signal, 22050).unwrap();

        assert_eq!(mono.sample_rate, 22050);

        let expected_frames = signal.num_frames() / 2;
        let tolerance = (expected_frames as f64 * 0.02).max(64.0) as usize;
        let diff = mono.samples.len().abs_diff(expected_frames);
        assert!(
            diff <= tolerance,
            "expected ~{expected_frames} frames, got {} (diff {diff}, tolerance {tolerance})",
            mono.samples.len()
        );

        let duration_diff = (mono.duration_seconds() - signal.duration_seconds()).abs();
        assert!(duration_diff < 0.05, "duration drifted by {duration_diff}s");
    }

    #[test]
    fn resample_is_noop_when_rate_already_matches() {
        let signal = sine_signal(22050, 1, 220.0, 0.5);
        let mono = resample_mono(&signal, 22050).unwrap();
        assert_eq!(mono.samples, signal.samples);
    }
}
