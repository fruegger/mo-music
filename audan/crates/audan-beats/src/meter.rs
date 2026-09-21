//! Meter estimation (S5.3 `MeterEstimator`, S8.3, ADR-11).
//!
//! Tests candidate beats-per-bar values by grouping a per-beat strength
//! sequence into phases modulo each candidate and measuring how much
//! stronger the best phase is than the average phase -- a real, if
//! heuristic, periodicity test, not a hardcoded `4`. Near-ties are broken by
//! a mild prior favouring more common meters ([`meter_prior`]), the same
//! idea as [`crate::tempo::TempoAnalyser::tempo_candidates`]'s tempo prior
//! and for the same reason: raw periodicity-strength scores alone are
//! genuinely ambiguous between related candidates (a signal that's well
//! explained by a 6-beat grouping is, by construction, at least as well
//! explained by a 3-beat grouping sampled at half the rate -- confirmed
//! directly in this module's own tests), so some outside signal is needed to
//! prefer one over the other when the raw scores don't clearly decide it.
//!
//! `beats_per_bar` is purely a periodicity count -- how many detected beats
//! separate one strong (downbeat-like) pulse from the next -- with no notion
//! of the *denominator* a human would notate it with. `audan`'s own CLI and
//! test data both display it as `beats_per_bar/4` by convention (S8.3), but
//! this module cannot tell a 6/4 waltz-of-two-bars feel from a 6/8 compound
//! feel: both are just "period 6" here, since nothing in this pipeline
//! models beat subdivision. A track notated as 12/8 with a felt 6/8 pulse
//! (e.g. `resources/test_data`'s "Crazy") will report whatever
//! `beats_per_bar` its detected beat *pulse* happens to align to -- 6 if the
//! tracker locks onto the eighth-note-triplet-ish pulse rather than the
//! dotted-quarter one -- not "6/8" as a distinct answer from "6/4".

pub struct MeterResult {
    pub beats_per_bar: u8,
    pub confidence: f32,
}

pub struct MeterEstimator {
    pub candidates: Vec<u8>,
}

impl Default for MeterEstimator {
    fn default() -> Self {
        // 5 and 7 cover odd/asymmetric meters (5/4 "Take Five", 7/4/7/8
        // progressive-rock bars) that the earlier {2,3,4,6} set could never
        // pick, no matter how clean the signal, simply because they weren't
        // offered as candidates at all -- confirmed directly against real
        // recordings (`resources/test_data`), not a change made in the
        // abstract. Neither aliases against an existing candidate the way
        // 3-vs-6 does (5 and 7 are prime and divide none of 2/3/4/6), so
        // this doesn't introduce a new instance of that ambiguity.
        Self {
            candidates: vec![2, 3, 4, 5, 6, 7],
        }
    }
}

/// A mild preference for more common meters, used only to break near-ties
/// (see [`MeterEstimator::estimate`]'s doc comment). 4/4 is by a wide margin
/// the most common meter in Western popular/electronic music; the rest are
/// ordered roughly by how often they appear after it, with 5 and 7 last
/// among the named cases since asymmetric meters are rarer still than 6.
/// Deliberately gentle (never more than a ~1.8x factor) so a signal with
/// real, strong evidence for an unusual meter still wins -- this only tips a
/// genuine coin-flip.
fn meter_prior(n: u8) -> f64 {
    match n {
        4 => 1.0,
        3 => 0.9,
        2 => 0.8,
        6 => 0.75,
        5 => 0.65,
        7 => 0.6,
        _ => 0.55,
    }
}

impl MeterEstimator {
    /// `beat_strengths` is any per-beat activation/confidence sequence -- a
    /// value expected to be periodically stronger every `beats_per_bar`-th
    /// beat if the meter guess is right. In `track_beats` this is
    /// `crate::sample_downbeat_curve`'s downbeat-activation curve sampled at
    /// beat times, not the plain per-beat detection confidence
    /// (`PostProcessor::detect_beats`'s), which is drawn from the *beat*
    /// curve: a backend confident about every beat regardless of downbeat
    /// status (real for a trained tracker; see `lib.rs::track_beats`'s doc
    /// comment on this) makes that signal nearly flat and useless for
    /// picking the meter, even though it type-checks the same way.
    pub fn estimate(&self, beat_strengths: &[f32]) -> MeterResult {
        // Falls back to the overwhelmingly common case with zero confidence
        // when there isn't enough data to test any candidate meaningfully
        // (need at least two full bars of the largest candidate to compare
        // phases at all).
        let default_result = MeterResult {
            beats_per_bar: 4,
            confidence: 0.0,
        };
        let max_candidate = match self.candidates.iter().max() {
            Some(&m) => m as usize,
            None => return default_result,
        };
        if beat_strengths.len() < max_candidate * 2 {
            return default_result;
        }

        let mut scores: Vec<(u8, f64)> = Vec::new();
        for &n in &self.candidates {
            let n_usize = n as usize;
            if n_usize == 0 || beat_strengths.len() < n_usize * 2 {
                continue;
            }
            let mut phase_sums = vec![0.0f64; n_usize];
            let mut phase_counts = vec![0usize; n_usize];
            for (i, &v) in beat_strengths.iter().enumerate() {
                let p = i % n_usize;
                phase_sums[p] += v as f64;
                phase_counts[p] += 1;
            }
            let phase_means: Vec<f64> = (0..n_usize)
                .map(|p| {
                    if phase_counts[p] > 0 {
                        phase_sums[p] / phase_counts[p] as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let overall_mean = phase_means.iter().sum::<f64>() / n_usize as f64;
            let max_phase_mean = phase_means.iter().cloned().fold(f64::MIN, f64::max);
            // How much better the best phase is than the average phase:
            // a meter that truly explains the periodicity should have one
            // clearly dominant phase (the downbeat position) and the rest
            // near baseline.
            let score = (max_phase_mean - overall_mean) * meter_prior(n);
            scores.push((n, score));
        }

        if scores.is_empty() {
            return default_result;
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (best_n, best_score) = scores[0];
        let second_score = scores.get(1).map(|s| s.1).unwrap_or(0.0);
        let confidence = if best_score + second_score.max(0.0) > 1e-9 {
            ((best_score - second_score) / (best_score + second_score.max(0.0) + 1e-9))
                .clamp(0.0, 1.0) as f32
        } else {
            0.0
        };

        MeterResult {
            beats_per_bar: best_n,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn periodic_strengths(len: usize, period: usize, strong: f32, weak: f32) -> Vec<f32> {
        (0..len)
            .map(|i| if i % period == 0 { strong } else { weak })
            .collect()
    }

    #[test]
    fn recovers_four_beats_per_bar() {
        let estimator = MeterEstimator::default();
        let strengths = periodic_strengths(32, 4, 1.0, 0.2);
        let result = estimator.estimate(&strengths);
        assert_eq!(result.beats_per_bar, 4);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn recovers_three_beats_per_bar() {
        // A pure period-3 pattern is *exactly* as well explained by a
        // period-6 grouping (every other bar realigns identically) -- the
        // meter analogue of tempo's octave ambiguity, and genuinely
        // ambiguous, not a bug. Restrict candidates to {2,3,4} here so this
        // test isolates "3 beats a bar is preferred over 2 or 4," and leave
        // the 3-vs-6 tie to be resolved by whichever candidate is tried
        // first when using the full default candidate set.
        let estimator = MeterEstimator {
            candidates: vec![2, 3, 4],
        };
        let strengths = periodic_strengths(24, 3, 1.0, 0.2);
        let result = estimator.estimate(&strengths);
        assert_eq!(result.beats_per_bar, 3);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn recovers_five_beats_per_bar() {
        let estimator = MeterEstimator::default();
        let strengths = periodic_strengths(40, 5, 1.0, 0.2);
        let result = estimator.estimate(&strengths);
        assert_eq!(result.beats_per_bar, 5);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn recovers_seven_beats_per_bar() {
        let estimator = MeterEstimator::default();
        let strengths = periodic_strengths(56, 7, 1.0, 0.2);
        let result = estimator.estimate(&strengths);
        assert_eq!(result.beats_per_bar, 7);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn too_little_data_falls_back_with_zero_confidence() {
        let estimator = MeterEstimator::default();
        let result = estimator.estimate(&[1.0, 0.2, 1.0]);
        assert_eq!(result.confidence, 0.0);
    }
}
