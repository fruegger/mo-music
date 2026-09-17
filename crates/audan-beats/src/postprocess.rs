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

    /// Like [`Self::detect_beats`], but constrained to a known target tempo.
    ///
    /// Plain peak-picking (`detect_beats`) accepts *every* activation peak
    /// that clears the threshold and isn't a near-duplicate, with no notion
    /// of periodicity -- on real music this routinely locks onto the
    /// densest strong transient (e.g. hi-hats at a clean subdivision of the
    /// true beat) rather than the beat itself, since a hi-hat pattern can be
    /// just as strong and far more frequent than the kick/snare pulse a
    /// listener would call "the beat." The result is an internally
    /// consistent but *wrong* beat grid: twice as many beats as the track
    /// actually has, at twice the true tempo.
    ///
    /// Given an externally supplied `target_bpm` (from
    /// [`crate::tempo::TempoAnalyser::tempo_candidates`], which estimates
    /// tempo from the *whole* activation curve's periodicity via a
    /// tempogram rather than from any single peak sequence, and is
    /// therefore not fooled by which sub-pulse happens to peak-pick
    /// loudest), this greedily walks forward in steps of the target
    /// inter-beat interval, at each step keeping only the strongest raw
    /// peak within a tolerance window around the expected position --
    /// exactly the mechanism that turns "every onset, regardless of
    /// periodicity" into "one onset per true beat." A brief dropout (no
    /// peak found near an expected position) advances the expectation by
    /// one period anyway rather than desynchronising every later beat.
    pub fn detect_beats_at_tempo(
        &self,
        activations: &BeatActivations,
        grid: FrameGrid,
        target_bpm: f64,
    ) -> DetectedBeats {
        if !(target_bpm.is_finite() && target_bpm > 0.0) {
            return self.detect_beats(activations, grid);
        }
        let raw = self.detect_beats(activations, grid);
        if raw.times.len() < 2 {
            return raw;
        }

        let target_ibi = 60.0 / target_bpm;
        // Wide enough to absorb real tempo jitter/drift, tight enough to
        // reject a neighbouring subdivision peak roughly half a period away.
        let tol = (target_ibi * 0.35).max(0.02);

        let mut times = Vec::new();
        let mut confidence = Vec::new();
        let mut expected = raw.times[0];
        let mut idx = 0usize;
        while idx < raw.times.len() {
            let mut best: Option<(usize, f64, f32)> = None;
            let mut j = idx;
            while j < raw.times.len() && raw.times[j] < expected + tol {
                if raw.times[j] >= expected - tol {
                    let c = raw.confidence[j];
                    if best.map(|(_, _, bc)| c > bc).unwrap_or(true) {
                        best = Some((j, raw.times[j], c));
                    }
                }
                j += 1;
            }
            match best {
                Some((_, t, c)) => {
                    times.push(t);
                    confidence.push(c);
                    expected = t + target_ibi;
                    idx = j;
                }
                None => {
                    // Nothing near the expected slot: advance one period and
                    // skip past any raw peaks now behind the new window,
                    // rather than getting stuck re-examining them forever.
                    expected += target_ibi;
                    while idx < raw.times.len() && raw.times[idx] < expected - tol {
                        idx += 1;
                    }
                }
            }
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
        grid: FrameGrid,
    ) -> Vec<BeatIndex> {
        let n = beats_per_bar as usize;
        if beat_times.is_empty() || n == 0 {
            return Vec::new();
        }

        // `activations.times` are evenly spaced (`MelFrontend` always uses a
        // padded grid, where `FrameGrid::time_of(k) == k*hop/sr` exactly), so
        // the nearest frame index can be computed directly in O(1) rather
        // than by scanning every activation frame for every beat -- with a
        // beat count and activation-frame count that both scale with track
        // length, the earlier linear scan made this whole function
        // effectively O(beats_per_bar * n_beats * n_frames).
        let sr = grid.sample_rate as f64;
        let hop = grid.hop as f64;
        let last_idx = activations.downbeat.len().saturating_sub(1);
        let sample_at = |t: f64| -> f32 {
            if activations.downbeat.is_empty() {
                return 0.0;
            }
            let idx = ((t * sr / hop).round() as i64).clamp(0, last_idx as i64) as usize;
            activations.downbeat[idx]
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
            grid,
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
    fn tempo_locked_detection_rejects_a_subdivision_pulse() {
        // A strong beat pulse every 40 frames, plus an equally-strong
        // subdivision pulse every 20 frames in between (a hi-hat playing
        // twice as fast as the true beat -- exactly the real-world failure
        // mode this method exists to correct). Plain `detect_beats` cannot
        // tell the two apart and keeps every peak; `detect_beats_at_tempo`,
        // told the true beat's tempo, should keep only the beat-period ones.
        let grid = FrameGrid::new(22_050, 256, 512, PadMode::Reflect);
        let n_frames = 400;
        let beat_period = 40usize;
        let sub_period = 20usize;
        let mut values = vec![0.0f32; n_frames];
        for i in (0..n_frames).step_by(sub_period) {
            values[i] = 0.6;
        }
        for i in (0..n_frames).step_by(beat_period) {
            values[i] = 0.6; // same strength as the subdivision peaks
        }
        let activations = activations_from_values(grid, values);

        let pp = PostProcessor::default();
        let raw = pp.detect_beats(&activations, grid);
        assert_eq!(
            raw.times.len(),
            n_frames / sub_period,
            "sanity check: plain peak-picking keeps every subdivision peak"
        );

        // frames/sec = 22050/256 = 86.13; beat_period=40 frames => IBI =
        // 40/86.13 = 0.4645s => ~129.2 BPM.
        let target_bpm = 60.0 * (22_050.0 / 256.0) / beat_period as f64;
        let locked = pp.detect_beats_at_tempo(&activations, grid, target_bpm);

        assert!(
            locked.times.len() >= n_frames / beat_period - 1
                && locked.times.len() <= n_frames / beat_period + 1,
            "expected roughly {} tempo-locked beats, got {}: {:?}",
            n_frames / beat_period,
            locked.times.len(),
            locked.times
        );
        for w in locked.times.windows(2) {
            let ibi = w[1] - w[0];
            let target_ibi = 60.0 / target_bpm;
            assert!(
                (ibi - target_ibi).abs() < target_ibi * 0.5,
                "consecutive locked beats {ibi}s apart, expected ~{target_ibi}s"
            );
        }
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
            grid,
        };

        let pp = PostProcessor::default();
        let downbeats = pp.snap_downbeats(&beat_times, &activations, 4, grid);
        assert_eq!(downbeats, vec![0, 4]);
    }
}
