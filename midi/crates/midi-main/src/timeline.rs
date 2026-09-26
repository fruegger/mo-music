//! Converts absolute tick positions into musical time (bars:beats) or clock time (minutes:seconds).

use crate::midi::{MidiDivision, MidiMessage};

/// tempo a file plays at until it sets one: 120 quarter notes per minute
const DEFAULT_TEMPO: u32 = 500_000;

struct TempoSegment {
    start_tick: u64,
    /// microseconds per quarter note from start_tick on
    tempo: u32,
    /// sum of ticks * tempo up to start_tick; divided by the ticks per quarter note it gives microseconds
    start_tick_tempo: u128,
}

struct MeterSegment {
    start_tick: u64,
    /// bars completed before start_tick
    start_bar: u64,
    numerator: u64,
    /// the denominator as a power of two, as stored in the time signature event
    denominator_pow: u32,
}

impl MeterSegment {
    /// Ticks since the segment start, scaled by the denominator so that a beat is exactly 4 * ppq long.
    fn scaled(&self, tick: u64) -> u64 {
        (tick - self.start_tick) << self.denominator_pow
    }
}

/// Tempo and time signature changes of a song, used to place ticks in time.
pub struct Timeline {
    division: MidiDivision,
    tempos: Vec<TempoSegment>,
    meters: Vec<MeterSegment>,
}

impl Timeline {
    /// Collects the SetTempo and TimeSignature events from `events`, given as (absolute tick, message).
    pub fn new<'a>(
        division: MidiDivision,
        events: impl IntoIterator<Item = (u64, &'a MidiMessage)>,
    ) -> Self {
        let mut tempo_changes = vec![(0, DEFAULT_TEMPO)];
        let mut meter_changes = vec![(0, 4, 2)]; // 4/4
        for (tick, message) in events {
            match *message {
                MidiMessage::SetTempo(tempo) if tempo > 0 => tempo_changes.push((tick, tempo)),
                MidiMessage::TimeSignature(nn, dd, _, _) if nn > 0 && dd <= 16 => {
                    meter_changes.push((tick, nn as u64, dd as u32))
                }
                _ => {}
            }
        }
        // the sort is stable, so of several changes at the same tick the last one in the file wins
        tempo_changes.sort_by_key(|&(tick, _)| tick);
        meter_changes.sort_by_key(|&(tick, _, _)| tick);

        let mut tempos: Vec<TempoSegment> = Vec::new();
        for (start_tick, tempo) in tempo_changes {
            let start_tick_tempo = match tempos.last() {
                Some(prev) => {
                    prev.start_tick_tempo
                        + (start_tick - prev.start_tick) as u128 * prev.tempo as u128
                }
                None => 0,
            };
            tempos.push(TempoSegment {
                start_tick,
                tempo,
                start_tick_tempo,
            });
        }

        let mut meters: Vec<MeterSegment> = Vec::new();
        if let MidiDivision::TicksPerQuarterNote(ppq) = division {
            let beat_len = 4 * ppq as u64;
            for (start_tick, numerator, denominator_pow) in meter_changes {
                let start_bar = match meters.last() {
                    // a change in the middle of a bar starts a new bar
                    Some(prev) => {
                        prev.start_bar + prev.scaled(start_tick).div_ceil(prev.numerator * beat_len)
                    }
                    None => 0,
                };
                meters.push(MeterSegment {
                    start_tick,
                    start_bar,
                    numerator,
                    denominator_pow,
                });
            }
        }

        Timeline {
            division,
            tempos,
            meters,
        }
    }

    /// `bar:beat:hundredths`, bar and beat counted from 1; a beat is the time signature's denominator.
    /// None for SMPTE timed files, whose ticks have no musical length.
    pub fn bars(&self, tick: u64) -> Option<String> {
        let MidiDivision::TicksPerQuarterNote(ppq) = self.division else {
            return None;
        };
        let meter = &self.meters[self.meters.partition_point(|m| m.start_tick <= tick) - 1];
        let beat_len = 4 * ppq as u64;
        let bar_len = meter.numerator * beat_len;
        let pos = meter.scaled(tick);
        let in_bar = pos % bar_len;
        Some(format!(
            "{}:{}:{:02}",
            meter.start_bar + pos / bar_len + 1,
            in_bar / beat_len + 1,
            in_bar % beat_len * 100 / beat_len
        ))
    }

    /// `minutes:seconds:hundredths` from the start of the song.
    pub fn clock(&self, tick: u64) -> String {
        let micros = match self.division {
            MidiDivision::TicksPerQuarterNote(ppq) => {
                let tempo = &self.tempos[self.tempos.partition_point(|t| t.start_tick <= tick) - 1];
                (tempo.start_tick_tempo + (tick - tempo.start_tick) as u128 * tempo.tempo as u128)
                    / ppq as u128
            }
            MidiDivision::Smpte {
                frames_per_second,
                ticks_per_frame,
            } => {
                // 29 stands for 29.97 (30000 / 1001) frames per second
                let (fps_num, fps_den) = match frames_per_second {
                    29 => (30_000, 1001),
                    fps => (fps as u128, 1),
                };
                tick as u128 * 1_000_000 * fps_den / (fps_num * ticks_per_frame as u128)
            }
        };
        let hundredths = micros / 10_000;
        format!(
            "{}:{:02}:{:02}",
            hundredths / 6000,
            hundredths / 100 % 60,
            hundredths % 100
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PPQ: MidiDivision = MidiDivision::TicksPerQuarterNote(96);

    fn timeline(division: MidiDivision, events: &[(u64, MidiMessage)]) -> Timeline {
        Timeline::new(division, events.iter().map(|(tick, m)| (*tick, m)))
    }

    #[test]
    fn clock_uses_default_tempo_without_tempo_events() {
        let t = timeline(PPQ, &[]);
        assert_eq!(t.clock(0), "0:00:00");
        // 120 bpm: two quarter notes per second
        assert_eq!(t.clock(2 * 96), "0:01:00");
        assert_eq!(t.clock(2 * 96 * 61 + 96), "1:01:50");
    }

    #[test]
    fn clock_follows_tempo_changes() {
        // 1 s at 120 bpm, then 60 bpm
        let t = timeline(PPQ, &[(192, MidiMessage::SetTempo(1_000_000))]);
        assert_eq!(t.clock(192), "0:01:00");
        assert_eq!(t.clock(192 + 96), "0:02:00");
        assert_eq!(t.clock(192 + 96 + 24), "0:02:25");
    }

    #[test]
    fn clock_for_smpte_divisions() {
        let t = timeline(
            MidiDivision::Smpte {
                frames_per_second: 25,
                ticks_per_frame: 40,
            },
            &[(0, MidiMessage::SetTempo(1_000_000))],
        );
        // 1000 ticks per second; the tempo does not matter
        assert_eq!(t.clock(61_500), "1:01:50");
        assert_eq!(t.bars(61_500), None);

        let drop_frame = timeline(
            MidiDivision::Smpte {
                frames_per_second: 29,
                ticks_per_frame: 1,
            },
            &[],
        );
        // 30 frames at 29.97 fps take 1.001 s
        assert_eq!(drop_frame.clock(30), "0:01:00");
        assert_eq!(drop_frame.clock(30 * 60), "1:00:06");
    }

    #[test]
    fn bars_default_to_four_four() {
        let t = timeline(PPQ, &[]);
        assert_eq!(t.bars(0).unwrap(), "1:1:00");
        assert_eq!(t.bars(96 + 48).unwrap(), "1:2:50");
        assert_eq!(t.bars(4 * 96).unwrap(), "2:1:00");
    }

    #[test]
    fn bars_count_beats_in_the_denominator() {
        // 6/8: a beat is an eighth note (48 ticks), a bar 288 ticks
        let t = timeline(PPQ, &[(0, MidiMessage::TimeSignature(6, 3, 24, 8))]);
        assert_eq!(t.bars(48).unwrap(), "1:2:00");
        assert_eq!(t.bars(288 + 24).unwrap(), "2:1:50");
    }

    #[test]
    fn time_signature_change_starts_a_new_bar() {
        // 4/4 for two bars, then 3/4
        let t = timeline(PPQ, &[(8 * 96, MidiMessage::TimeSignature(3, 2, 24, 8))]);
        assert_eq!(t.bars(8 * 96).unwrap(), "3:1:00");
        assert_eq!(t.bars(11 * 96).unwrap(), "4:1:00");

        // a change in the middle of bar 2 (beat 3) still starts bar 3
        let t = timeline(PPQ, &[(6 * 96, MidiMessage::TimeSignature(3, 2, 24, 8))]);
        assert_eq!(t.bars(5 * 96).unwrap(), "2:2:00");
        assert_eq!(t.bars(6 * 96).unwrap(), "3:1:00");
    }

    #[test]
    fn last_change_at_the_same_tick_wins() {
        let t = timeline(
            PPQ,
            &[
                (0, MidiMessage::SetTempo(250_000)),
                (0, MidiMessage::SetTempo(1_000_000)),
            ],
        );
        assert_eq!(t.clock(96), "0:01:00");
    }
}
