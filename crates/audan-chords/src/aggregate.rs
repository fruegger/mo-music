//! Beat-synchronous chroma aggregation: fold the fine-grained chroma frames
//! `audan_dsp::chord_chroma` produces down to one 12-dim vector per beat, by
//! averaging every frame whose window-centre time falls inside that beat's
//! interval.

use audan_core::{BeatGrid, Chroma, PITCH_CLASSES};

/// Averages chroma frames into one vector per beat.
///
/// Beat `i`'s interval is `[beats[i], beats[i+1])`, except the last beat,
/// which has no successor to bound it: it is extended through every
/// remaining chroma frame rather than to some fixed duration past the beat.
/// This needs no extra parameter (a fixed duration would have to guess at a
/// track's true end) and is the natural reading of "the last beat owns
/// whatever chroma comes after it" -- a beat grid's final beat is not
/// promised to coincide with the audio's end.
///
/// A beat with no frames in its interval (a pathologically coarse beat grid
/// relative to the chroma hop, or an empty chroma) yields an all-zero
/// vector, which downstream template matching treats as "no chord" rather
/// than a crash.
pub fn beat_synchronous_chroma(chroma: &Chroma, beats: &BeatGrid) -> Vec<[f32; PITCH_CLASSES]> {
    let n_beats = beats.beats.len();
    if n_beats == 0 {
        return Vec::new();
    }
    let mut sums = vec![[0f32; PITCH_CLASSES]; n_beats];
    let mut counts = vec![0usize; n_beats];

    let mut beat_idx = 0usize;
    for k in 0..chroma.n_frames() {
        let t = chroma.grid.time_of(k).as_seconds();
        if t < beats.beats[0] {
            continue;
        }
        while beat_idx + 1 < n_beats && t >= beats.beats[beat_idx + 1] {
            beat_idx += 1;
        }
        let frame = chroma.frame(k);
        for pc in 0..PITCH_CLASSES {
            sums[beat_idx][pc] += frame[pc];
        }
        counts[beat_idx] += 1;
    }

    for (sum, count) in sums.iter_mut().zip(counts.iter()) {
        if *count > 0 {
            for v in sum.iter_mut() {
                *v /= *count as f32;
            }
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::beat::{Meter, Source, TempoInfo, TempoStability, TempoStabilityClass};
    use audan_core::candidate::{Candidate, Ranked};
    use audan_core::frame::{FrameGrid, PadMode};

    fn grid(beats: Vec<f64>) -> BeatGrid {
        let duration_seconds = beats.last().copied().unwrap_or(0.0) + 1.0;
        BeatGrid {
            schema_version: BeatGrid::CURRENT_SCHEMA_VERSION,
            duration_seconds,
            confidence: vec![1.0; beats.len()],
            beats,
            downbeats: vec![0],
            meter: Meter {
                beats_per_bar: 4,
                confidence: 1.0,
            },
            tempo: TempoInfo {
                median_bpm: 120.0,
                candidates: Ranked::new(vec![Candidate::new(120.0, 1.0)]).unwrap(),
                stability: TempoStability {
                    ibi_mad_ms: 0.0,
                    drift_bpm_per_min: 0.0,
                    class: TempoStabilityClass::Programmed,
                },
            },
            frames: FrameGrid::new(22050, 512, 2048, PadMode::Reflect).into(),
            source: Source {
                algo: "test".into(),
                version: "0".into(),
                postproc: "none".into(),
            },
        }
    }

    #[test]
    fn averages_frames_within_each_beat_interval() {
        let frame_grid = FrameGrid::new(10, 1, 1, PadMode::Zero); // time_of(k) = k / 10
        let mut frames = Vec::new();
        for k in 0..10 {
            let mut f = [0f32; PITCH_CLASSES];
            f[0] = k as f32; // distinct per frame so we can check averaging
            frames.push(f);
        }
        let chroma = Chroma::from_frames(frame_grid, frames);
        let beats = grid(vec![0.0, 0.5]); // beat 0: frames 0..5 (t=0.0..0.5), beat 1: frames 5..10
        let agg = beat_synchronous_chroma(&chroma, &beats);
        assert_eq!(agg.len(), 2);
        assert!((agg[0][0] - 2.0).abs() < 1e-6); // mean of 0..4
        assert!((agg[1][0] - 7.0).abs() < 1e-6); // mean of 5..9
    }

    #[test]
    fn empty_chroma_yields_zero_vectors() {
        let frame_grid = FrameGrid::new(10, 1, 1, PadMode::Zero);
        let chroma = Chroma::from_frames(frame_grid, vec![]);
        let beats = grid(vec![0.0, 0.5]);
        let agg = beat_synchronous_chroma(&chroma, &beats);
        assert_eq!(agg, vec![[0.0; PITCH_CLASSES]; 2]);
    }
}
