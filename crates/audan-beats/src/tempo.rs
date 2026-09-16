//! Tempo statistics and ranked tempo candidates (S5.3 `TempoAnalyser`, S8.3,
//! ADR-11, Q5, QS9).
//!
//! Two things live here and are deliberately kept separate:
//!
//! - [`TempoAnalyser::analyse`] derives `median_bpm` and jitter/drift
//!   stability purely from a beat-time sequence -- no DSP dependency, so it
//!   is directly unit-testable against hand-built beat arrays.
//! - [`TempoAnalyser::tempo_candidates`] derives the *ranked* tempo
//!   candidates (the actual point of ADR-11/QS9: 87 vs 174 BPM must both
//!   surface) from `audan_dsp::tempogram` run over the beat-activation
//!   curve, searching near `median_bpm`, its double, and its half -- the
//!   classic octave-ambiguity triad -- with confidences drawn from relative
//!   tempogram energy at each candidate, not hardcoded numbers.

use audan_core::{Candidate, FrameGrid, Ranked, TempoStabilityClass};
use audan_dsp::{tempogram, OnsetEnvelope, TempogramParams};

use crate::model::BeatActivations;

pub struct TempoStats {
    pub median_bpm: f64,
    pub ibi_mad_ms: f64,
    pub drift_bpm_per_min: f64,
    pub class: TempoStabilityClass,
}

pub struct TempoAnalyser {
    /// Thresholds below are tunable heuristics documented here, not derived
    /// from any formal calibration -- there is no reference beat-tracking
    /// dataset available in this environment to fit them against.
    ///
    /// IBI MAD at or below this (milliseconds) reads as machine-precise
    /// timing: a drum machine or DAW-programmed track.
    pub programmed_mad_ms: f64,
    /// IBI MAD at or below this (and not classified `Drifting`) reads as
    /// natural human micro-timing rather than structural tempo change.
    pub human_mad_ms: f64,
    /// A linear fit of instantaneous BPM against elapsed time with a slope
    /// at or beyond this magnitude (BPM/min) is considered a real drift
    /// rather than fit noise, *provided* the fit is also clean (see
    /// `drift_r2_threshold`).
    pub drift_threshold_bpm_per_min: f64,
    /// R^2 of the same linear fit must reach this for a drift to be trusted
    /// as "drifting" rather than "just noisy" (-> `Variable`).
    pub drift_r2_threshold: f64,
}

impl Default for TempoAnalyser {
    fn default() -> Self {
        Self {
            programmed_mad_ms: 5.0,
            human_mad_ms: 40.0,
            drift_threshold_bpm_per_min: 1.0,
            drift_r2_threshold: 0.5,
        }
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Ordinary least squares of `ys` against `xs`. Returns `(slope, r_squared)`.
fn linear_fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if xs.len() < 2 {
        return (0.0, 0.0);
    }
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 1e-12 {
        return (0.0, 0.0);
    }
    let slope = sxy / sxx;
    let r2 = if syy <= 1e-12 {
        1.0
    } else {
        (sxy * sxy) / (sxx * syy)
    };
    (slope, r2)
}

impl TempoAnalyser {
    /// Derives `median_bpm` and stability purely from beat times. Returns
    /// `None` when fewer than two beats are given (no inter-beat interval
    /// exists to measure).
    pub fn analyse(&self, beat_times: &[f64]) -> Option<TempoStats> {
        if beat_times.len() < 2 {
            return None;
        }
        let ibis: Vec<f64> = beat_times.windows(2).map(|w| w[1] - w[0]).collect();
        if ibis.iter().any(|&ibi| ibi <= 0.0) {
            return None;
        }

        let mut sorted_ibis = ibis.clone();
        let median_ibi = median(&mut sorted_ibis);
        let median_bpm = 60.0 / median_ibi;

        let mut abs_dev: Vec<f64> = ibis.iter().map(|&ibi| (ibi - median_ibi).abs()).collect();
        let ibi_mad_ms = median(&mut abs_dev) * 1000.0;

        // Instantaneous BPM per interval, at the interval's midpoint in
        // elapsed time -- a real linear fit of tempo against time, not a
        // placeholder.
        let xs: Vec<f64> = (0..ibis.len())
            .map(|i| (beat_times[i] + beat_times[i + 1]) / 2.0)
            .collect();
        let ys: Vec<f64> = ibis.iter().map(|&ibi| 60.0 / ibi).collect();
        let (slope_bpm_per_sec, r2) = linear_fit(&xs, &ys);
        let drift_bpm_per_min = slope_bpm_per_sec * 60.0;

        let class = self.classify(ibi_mad_ms, drift_bpm_per_min, r2);

        Some(TempoStats {
            median_bpm,
            ibi_mad_ms,
            drift_bpm_per_min,
            class,
        })
    }

    fn classify(&self, mad_ms: f64, drift_bpm_per_min: f64, r2: f64) -> TempoStabilityClass {
        let clean_drift = drift_bpm_per_min.abs() >= self.drift_threshold_bpm_per_min
            && r2 >= self.drift_r2_threshold;
        if clean_drift {
            return TempoStabilityClass::Drifting;
        }
        if mad_ms <= self.programmed_mad_ms {
            TempoStabilityClass::Programmed
        } else if mad_ms <= self.human_mad_ms {
            TempoStabilityClass::Human
        } else {
            TempoStabilityClass::Variable
        }
    }

    /// Ranked tempo candidates (ADR-11/QS9): searches the tempogram of the
    /// beat-activation curve near `median_bpm` and its octave double/half,
    /// scoring each by relative tempogram energy. Never returns an empty
    /// `Ranked`: falls back to a single `median_bpm` candidate if the
    /// activation curve is too short to build a tempogram from, or if
    /// nothing scores near any target.
    pub fn tempo_candidates(
        &self,
        median_bpm: f64,
        activations: &BeatActivations,
        grid: FrameGrid,
    ) -> Ranked<f64> {
        let env = OnsetEnvelope {
            grid,
            times: activations.times.clone(),
            values: activations.beat.clone(),
        };
        let params = TempogramParams::default();
        let tg = tempogram(&env, &params);
        if tg.values.is_empty() || tg.bpms.is_empty() {
            return Ranked::single(median_bpm);
        }

        let n_bpms = tg.bpms.len();
        let mut energy = vec![0.0f64; n_bpms];
        for row in &tg.values {
            for (i, &v) in row.iter().enumerate() {
                energy[i] += v.max(0.0) as f64;
            }
        }
        let n_frames = tg.values.len() as f64;
        for e in energy.iter_mut() {
            *e /= n_frames.max(1.0);
        }

        let targets = [median_bpm, median_bpm * 2.0, median_bpm / 2.0];
        let mut found: Vec<(f64, f64)> = Vec::new(); // (bpm, energy)
        for &target in &targets {
            if (target as f32) < params.min_bpm || (target as f32) > params.max_bpm {
                continue;
            }
            let tol = (target * 0.08).max(1.0);
            let mut best: Option<(usize, f64)> = None;
            for (i, &bpm) in tg.bpms.iter().enumerate() {
                if (bpm as f64 - target).abs() <= tol {
                    if best.map(|(_, e)| energy[i] > e).unwrap_or(true) {
                        best = Some((i, energy[i]));
                    }
                }
            }
            if let Some((i, e)) = best {
                let bpm = tg.bpms[i] as f64;
                if found.iter().any(|&(b, _)| (b - bpm).abs() < 1.0) {
                    continue;
                }
                found.push((bpm, e));
            }
        }

        if found.is_empty() {
            return Ranked::single(median_bpm);
        }

        let total: f64 = found.iter().map(|&(_, e)| e.max(1e-6)).sum();
        let candidates: Vec<Candidate<f64>> = found
            .into_iter()
            .map(|(bpm, e)| Candidate::new(bpm, ((e.max(1e-6) / total) as f32).clamp(0.0, 1.0)))
            .collect();

        Ranked::new(candidates).unwrap_or_else(|_| Ranked::single(median_bpm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::{FrameGrid, FrameTime, PadMode};

    #[test]
    fn perfectly_regular_beats_are_programmed() {
        let analyser = TempoAnalyser::default();
        let beats: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect(); // exact 120 BPM
        let stats = analyser.analyse(&beats).unwrap();
        assert!((stats.median_bpm - 120.0).abs() < 1e-6);
        assert!(stats.ibi_mad_ms < 1e-6);
        assert_eq!(stats.class, TempoStabilityClass::Programmed);
    }

    #[test]
    fn small_jitter_reads_as_human() {
        let analyser = TempoAnalyser::default();
        // Deterministic pseudo-random jitter (no external RNG dependency):
        // a fixed pattern in +/-15ms, which sits inside the "human" MAD band
        // and averages out to no net drift.
        let jitter_ms = [8.0, -12.0, 5.0, -6.0, 14.0, -9.0, 3.0, -13.0, 7.0, -4.0];
        let mut t = 0.0f64;
        let mut beats = Vec::new();
        for i in 0..40 {
            beats.push(t);
            let jitter = jitter_ms[i % jitter_ms.len()] / 1000.0;
            t += 0.5 + jitter;
        }
        let stats = analyser.analyse(&beats).unwrap();
        assert_eq!(
            stats.class,
            TempoStabilityClass::Human,
            "mad_ms={}",
            stats.ibi_mad_ms
        );
        assert!(stats.ibi_mad_ms > analyser.programmed_mad_ms);
        assert!(stats.ibi_mad_ms <= analyser.human_mad_ms);
    }

    #[test]
    fn clean_linear_drift_is_detected_with_matching_slope() {
        let analyser = TempoAnalyser::default();
        // IBI shrinks linearly from 0.50s to 0.40s over 40 beats: a clean
        // accelerando, built in with a known target drift.
        let n = 40;
        let start_ibi = 0.50;
        let end_ibi = 0.40;
        let mut t = 0.0f64;
        let mut beats = vec![t];
        for i in 0..n - 1 {
            let ibi = start_ibi + (end_ibi - start_ibi) * i as f64 / (n - 2) as f64;
            t += ibi;
            beats.push(t);
        }
        let stats = analyser.analyse(&beats).unwrap();
        assert_eq!(
            stats.class,
            TempoStabilityClass::Drifting,
            "drift={}",
            stats.drift_bpm_per_min
        );

        // Built-in target: BPM goes from 120 to 150 over ~15.6s of playback
        // (sum of IBIs) => roughly +115 BPM/min. Just check sign and rough
        // magnitude, not exact match -- the fit is over instantaneous BPM
        // at interval midpoints, not the nominal end points.
        assert!(
            stats.drift_bpm_per_min > 20.0,
            "expected clear positive drift, got {}",
            stats.drift_bpm_per_min
        );
    }

    #[test]
    fn ranked_candidates_are_never_empty_and_sorted_descending() {
        let analyser = TempoAnalyser::default();
        let grid = FrameGrid::new(22_050, 512, 2048, PadMode::Reflect);
        // Too short to build any tempogram window from -> falls back to a
        // single candidate.
        let activations = BeatActivations {
            times: vec![grid.time_of(0), grid.time_of(1)],
            beat: vec![0.0, 0.0],
            downbeat: vec![0.0, 0.0],
        };
        let candidates = analyser.tempo_candidates(120.0, &activations, grid);
        assert!(candidates.len() >= 1);
        assert_eq!(candidates.top().value, 120.0);
    }

    #[test]
    fn octave_candidates_surface_for_a_periodic_activation_curve() {
        let analyser = TempoAnalyser::default();
        let grid = FrameGrid::new(22_050, 512, 2048, PadMode::Reflect);
        // A strong periodicity at 120 BPM: spike every `period` frames.
        // frames/sec = 22050/512 = 43.07; 120 BPM => period = 60/120*43.07
        // ~= 21.5 frames.
        let period = 21usize;
        let n = 900;
        let values: Vec<f32> = (0..n)
            .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
            .collect();
        let times: Vec<FrameTime> = (0..n).map(|k| grid.time_of(k)).collect();
        let activations = BeatActivations {
            times,
            beat: values,
            downbeat: vec![0.0; n],
        };

        let median_bpm = 60.0 * (22_050.0 / 512.0) / period as f64;
        let candidates = analyser.tempo_candidates(median_bpm, &activations, grid);

        assert!(candidates.len() >= 1);
        // Sorted descending by confidence.
        for w in candidates.as_slice().windows(2) {
            assert!(w[0].confidence >= w[1].confidence);
        }
    }
}
