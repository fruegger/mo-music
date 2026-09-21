use audan_core::{AudanError, Result};
use midly::num::{u15, u24, u28};
use midly::{Format, Header, MetaMessage, Smf, Timing, Track, TrackEvent, TrackEventKind};

/// Writes a Standard MIDI File containing a single track that is nothing but a
/// tempo map: each `(time_seconds, bpm)` pair becomes a `Set Tempo` meta event at
/// the tick offset that time converts to, given the tempo in effect since the
/// previous point (the first point's own bpm governs the run-up from tick 0).
pub fn write_midi_tempo_track(tempo_map: &[(f64, f64)], ticks_per_quarter: u16) -> Result<Vec<u8>> {
    if tempo_map.is_empty() {
        return Err(AudanError::InvalidInput(
            "tempo_map must contain at least one (time, bpm) point".into(),
        ));
    }

    let mut track: Track = Vec::with_capacity(tempo_map.len() + 1);
    let mut prev_time = 0.0f64;
    let mut prev_bpm = tempo_map[0].1;
    for &(time, bpm) in tempo_map {
        if !(bpm > 0.0) {
            return Err(AudanError::InvalidInput(format!("non-positive bpm {bpm}")));
        }
        let dt_seconds = (time - prev_time).max(0.0);
        let seconds_per_tick = 60.0 / (prev_bpm * f64::from(ticks_per_quarter));
        let delta_ticks = (dt_seconds / seconds_per_tick).round() as u32;
        let micros_per_quarter = (60_000_000.0 / bpm)
            .round()
            .clamp(1.0, f64::from(u24::max_value().as_int()))
            as u32;

        track.push(TrackEvent {
            delta: u28::new(delta_ticks),
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(micros_per_quarter))),
        });

        prev_time = time;
        prev_bpm = bpm;
    }
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    let smf = Smf {
        header: Header::new(
            Format::SingleTrack,
            Timing::Metrical(u15::new(ticks_per_quarter)),
        ),
        tracks: vec![track],
    };
    let mut buf = Vec::new();
    smf.write_std(&mut buf)?;
    Ok(buf)
}

/// Reads a Set Tempo track back into `(time_seconds, bpm)` pairs, in the order the
/// tempo events occur. The inverse of [`write_midi_tempo_track`], for the S8.7
/// round-trip test; only the first track is consulted.
pub fn read_midi_tempo_track(bytes: &[u8]) -> Result<Vec<(f64, f64)>> {
    let smf = Smf::parse(bytes)
        .map_err(|e| AudanError::InvalidInput(format!("failed to parse MIDI: {e}")))?;
    let ticks_per_quarter = match smf.header.timing {
        Timing::Metrical(t) => f64::from(t.as_int()),
        Timing::Timecode(..) => {
            return Err(AudanError::InvalidInput(
                "SMPTE timing is not supported for tempo-track round-trips".into(),
            ))
        }
    };
    let track = smf
        .tracks
        .first()
        .ok_or_else(|| AudanError::InvalidInput("MIDI file has no tracks".into()))?;

    let mut out = Vec::new();
    let mut elapsed_seconds = 0.0f64;
    let mut micros_per_quarter = 500_000.0f64;
    for event in track {
        let seconds_per_tick = micros_per_quarter / 1_000_000.0 / ticks_per_quarter;
        elapsed_seconds += f64::from(event.delta.as_int()) * seconds_per_tick;
        if let TrackEventKind::Meta(MetaMessage::Tempo(t)) = event.kind {
            micros_per_quarter = f64::from(t.as_int());
            let bpm = 60_000_000.0 / micros_per_quarter;
            out.push((elapsed_seconds, bpm));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_tempo_map() {
        let tempo_map = vec![(0.0, 120.0), (4.0, 140.0), (10.0, 90.0)];
        let bytes = write_midi_tempo_track(&tempo_map, 480).unwrap();
        let back = read_midi_tempo_track(&bytes).unwrap();

        assert_eq!(back.len(), tempo_map.len());
        for ((t_time, t_bpm), (r_time, r_bpm)) in tempo_map.iter().zip(back.iter()) {
            assert!((t_time - r_time).abs() < 1e-2, "time {t_time} vs {r_time}");
            assert!((t_bpm - r_bpm).abs() < 1e-2, "bpm {t_bpm} vs {r_bpm}");
        }
    }

    #[test]
    fn rejects_empty_tempo_map() {
        assert!(write_midi_tempo_track(&[], 480).is_err());
    }

    #[test]
    fn rejects_non_positive_bpm() {
        assert!(write_midi_tempo_track(&[(0.0, 0.0)], 480).is_err());
    }
}
