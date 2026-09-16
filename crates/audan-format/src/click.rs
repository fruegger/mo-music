use std::io::Cursor;

use audan_core::{AudanError, Result};
use hound::{SampleFormat, WavSpec, WavWriter};

const CLICK_FREQUENCY_HZ: f64 = 1000.0;
const CLICK_DURATION_SECONDS: f64 = 0.015;
const TAIL_SECONDS: f64 = 0.25;

/// Synthesises a click-track WAV (S3.2, "the primary debugging instrument",
/// S8.7): a brief Hann-windowed sine burst at each beat time, mixed into silence.
/// The window keeps the burst's onset and offset smooth, so it sounds like an
/// audible tick rather than a harsh digital pop.
pub fn write_click_track(beats: &[f64], sample_rate: u32) -> Result<Vec<u8>> {
    if sample_rate == 0 {
        return Err(AudanError::InvalidInput(
            "sample_rate must be greater than zero".into(),
        ));
    }

    let last_beat = beats.iter().cloned().fold(0.0f64, f64::max);
    let duration_seconds = last_beat + CLICK_DURATION_SECONDS + TAIL_SECONDS;
    let total_samples = (duration_seconds * f64::from(sample_rate)).ceil() as usize;
    let mut samples = vec![0.0f32; total_samples.max(1)];

    let click_len = ((CLICK_DURATION_SECONDS * f64::from(sample_rate)).round() as usize).max(1);
    for &beat_time in beats {
        if beat_time.is_sign_negative() {
            continue;
        }
        let start = (beat_time * f64::from(sample_rate)).round() as usize;
        for i in 0..click_len {
            let idx = start + i;
            let Some(slot) = samples.get_mut(idx) else {
                break;
            };
            let t = i as f64 / f64::from(sample_rate);
            let phase = 2.0 * std::f64::consts::PI * CLICK_FREQUENCY_HZ * t;
            let window_phase = if click_len > 1 {
                i as f64 / (click_len - 1) as f64
            } else {
                0.0
            };
            let window = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * window_phase).cos();
            *slot += (phase.sin() * window) as f32;
        }
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)
            .map_err(|e| AudanError::InvalidInput(format!("failed to open WAV writer: {e}")))?;
        for sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let quantised = (clamped * f32::from(i16::MAX)) as i16;
            writer.write_sample(quantised).map_err(|e| {
                AudanError::InvalidInput(format!("failed to write click-track sample: {e}"))
            })?;
        }
        writer.finalize().map_err(|e| {
            AudanError::InvalidInput(format!("failed to finalize click-track WAV: {e}"))
        })?;
    }
    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::WavReader;

    #[test]
    fn sample_count_matches_expected_duration() {
        let beats = vec![0.5, 1.0, 1.5];
        let sample_rate = 8_000;
        let bytes = write_click_track(&beats, sample_rate).unwrap();

        let reader = WavReader::new(Cursor::new(bytes)).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, sample_rate);
        assert_eq!(spec.channels, 1);

        let expected_duration = 1.5 + CLICK_DURATION_SECONDS + TAIL_SECONDS;
        let expected_samples = (expected_duration * f64::from(sample_rate)).ceil() as u32;
        assert_eq!(reader.duration(), expected_samples);
    }

    #[test]
    fn has_energy_near_each_beat() {
        let beats = vec![0.1, 0.2, 0.3];
        let sample_rate = 8_000;
        let bytes = write_click_track(&beats, sample_rate).unwrap();

        let mut reader = WavReader::new(Cursor::new(bytes)).unwrap();
        let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();

        for &beat in &beats {
            let centre = (beat * f64::from(sample_rate)).round() as usize;
            let window = &samples[centre.saturating_sub(5)..(centre + 5).min(samples.len())];
            let energy: i64 = window.iter().map(|&s| i64::from(s) * i64::from(s)).sum();
            assert!(energy > 0, "expected nonzero energy near beat at {beat}s");
        }
    }

    #[test]
    fn rejects_zero_sample_rate() {
        assert!(write_click_track(&[0.5], 0).is_err());
    }

    #[test]
    fn empty_beats_still_produces_a_valid_wav() {
        let bytes = write_click_track(&[], 8_000).unwrap();
        let reader = WavReader::new(Cursor::new(bytes)).unwrap();
        assert!(reader.duration() > 0);
    }
}
