//! Minimal post-processing (S5.3 `PostProcessor`).
//!
//! Deliberately **not** a DBN (dynamic Bayesian network): the Beat This!
//! authors show DBN post-processing adds unwanted metrical rigidity, and
//! madmom's DBN implementation is CC BY-NC-SA (non-commercial) anyway
//! (S2.2, S5.3). This is a scope boundary the architecture draws on
//! purpose, not a shortcut taken here for lack of time -- so this module
//! stays peak-picking + deduplication + phase-based downbeat snapping, and
//! should stay that way even if a future contributor is tempted to bolt a
//! Viterbi/HMM smoother on top.

use audan_core::{BeatIndex, FrameGrid};
use audan_dsp::{pick_peaks, OnsetEnvelope};

use crate::model::BeatActivations;

pub struct DetectedBeats {
    pub times: Vec<f64>,
    /// Per-beat confidence, drawn from the activation strength at the
    /// accepted peak -- not a constant.
    pub confidence: Vec<f32>,
}

pub struct PostProcessor {
    /// Minimum time between two accepted beats. Peaks closer together than
    /// this are collapsed into one (the stronger of the two), which is the
    /// entire "deduplication" step -- no smoothing model behind it.
    pub min_beat_interval_s: f64,
    /// Absolute activation threshold peaks must clear to be considered at
    /// all. The fallback backend normalizes its curve to a max of 1.0, so a
    /// fraction of that range is a reasonable default; a backend with a
    /// differently scaled curve should pass its own.
    pub activation_threshold: f32,
}

impl Default for PostProcessor {
    fn default() -> Self {
        // Chosen so that: (a) a 120 BPM click train's ~0.5s inter-beat
        // spacing is never itself mistaken for a duplicate (min interval is
        // well under any plausible IBI, including at very fast tempi up to
        // ~240 BPM / 0.25s), and (b) low-level noise well below a real
        // transient does not spawn spurious beats. Both are tunable
        // heuristics, not derived constants.
        Self {
            min_beat_interval_s: 0.15,
            activation_threshold: 0.15,
        }
    }
}

impl PostProcessor {
    /// Peak-picks `activations.beat`, reusing `audan_dsp::pick_peaks`'s
    /// already-tested local-maximum-plus-quadratic-refinement logic (built
    /// by wrapping the activation curve in the same `OnsetEnvelope` shape
    /// it expects, rather than reimplementing peak picking here), then
    /// deduplicates near-duplicate peaks by `min_beat_interval_s`.
    pub fn detect_beats(&self, activations: &BeatActivations, grid: FrameGrid) -> DetectedBeats {
        let env = OnsetEnvelope {
            grid,
            times: activations.times.clone(),
            values: activations.beat.clone(),
        };
        let peaks = pick_peaks(&env, self.activation_threshold);

        let mut times: Vec<f64> = Vec::new();
        let mut confidence: Vec<f32> = Vec::new();
        for peak in peaks {
            let t = peak.time.as_seconds();
            let v = activations.beat.get(peak.frame).copied().unwrap_or(0.0);
            if let (Some(&last_t), Some(last_v)) = (times.last(), confidence.last_mut()) {
                if t - last_t < self.min_beat_interval_s {
                    // Collapse: keep whichever of the two peaks is stronger.
                    if v > *last_v {
                        *times.last_mut().expect("just checked non-empty") = t;
                        *last_v = v;
                    }
                    continue;
                }
            }
            times.push(t);
            confidence.push(v);
        }

        DetectedBeats { times, confidence }
    }

    /// Chooses which detected beats are downbeats by testing every phase
    /// `0..beats_per_bar` and picking the one whose beats land, on average,
    /// on the strongest `activations.downbeat` energy. For the fallback
    /// backend, `downbeat` is identical to `beat` (see `model.rs`), so this
    /// degrades exactly to the documented fallback: "every Nth beat
    /// starting from the best-scoring phase," using the meter estimate as
    /// N. A backend with a genuinely discriminative downbeat curve gets a
    /// real snap for free from the same code.
    pub fn snap_downbeats(
        &self,
        beat_times: &[f64],
        activations: &BeatActivations,
        beats_per_bar: u8,
    ) -> Vec<BeatIndex> {
        let n = beats_per_bar as usize;
        if beat_times.is_empty() || n == 0 {
            return Vec::new();
        }

        let sample_at = |t: f64| -> f32 {
            if activations.times.is_empty() {
                return 0.0;
            }
            let mut best_idx = 0usize;
            let mut best_d = f64::MAX;
            for (i, ft) in activations.times.iter().enumerate() {
                let d = (ft.as_seconds() - t).abs();
                if d < best_d {
                    best_d = d;
                    best_idx = i;
                }
            }
            activations.downbeat.get(best_idx).copied().unwrap_or(0.0)
        };

        let mut best_phase = 0usize;
        let mut best_score = f32::MIN;
        for phase in 0..n {
            let score: f32 = beat_times
                .iter()
                .enumerate()
                .filter(|(i, _)| i % n == phase)
                .map(|(_, &t)| sample_at(t))
                .sum();
            if score > best_score {
                best_score = score;
                best_phase = phase;
            }
        }

        (0..beat_times.len())
            .filter(|i| i % n == best_phase)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::{FrameGrid, FrameTime, PadMode};

    fn activations_from_values(grid: FrameGrid, values: Vec<f32>) -> BeatActivations {
        let times: Vec<FrameTime> = (0..values.len()).map(|k| grid.time_of(k)).collect();
        let downbeat = values.clone();
        BeatActivations {
            times,
            beat: values,
            downbeat,
        }
    }

    #[test]
    fn two_close_spurious_peaks_collapse_to_one() {
        let grid = FrameGrid::new(22_050, 256, 512, PadMode::Reflect);
        // Two distinct local maxima (frames 1 and 3) very close in time --
        // at this hop/sample-rate a couple of frames apart is a few ms,
        // well inside any sensible min-distance -- separated from a third,
        // clearly distinct peak much later.
        let values = vec![
            0.0, 0.6, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        let activations = activations_from_values(grid, values);

        let pp = PostProcessor {
            min_beat_interval_s: 0.05,
            activation_threshold: 0.1,
        };
        let detected = pp.detect_beats(&activations, grid);

        // Without dedup this would find two close peaks near frame 1-2 plus
        // the frame-10 peak; with dedup, the close pair collapses to one.
        assert_eq!(
            detected.times.len(),
            2,
            "expected the close pair to collapse to one beat: {:?}",
            detected.times
        );
        assert!(detected.times[1] > detected.times[0] + 0.05);
    }

    #[test]
    fn snap_downbeats_picks_the_strongest_phase() {
        let grid = FrameGrid::new(22_050, 256, 512, PadMode::Reflect);
        // 8 evenly spaced beats; every 4th (phase 0) lands on a strong
        // downbeat-activation value.
        let beat_times: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();
        let mut downbeat = vec![0.1f32; 40];
        for i in 0..8 {
            let frame = (beat_times[i] * 22_050.0 / 256.0).round() as usize;
            if i % 4 == 0 {
                let idx = frame.min(downbeat.len() - 1);
                downbeat[idx] = 1.0;
            }
        }
        let times: Vec<FrameTime> = (0..downbeat.len()).map(|k| grid.time_of(k)).collect();
        let activations = BeatActivations {
            times,
            beat: downbeat.clone(),
            downbeat,
        };

        let pp = PostProcessor::default();
        let downbeats = pp.snap_downbeats(&beat_times, &activations, 4);
        assert_eq!(downbeats, vec![0, 4]);
    }
}
