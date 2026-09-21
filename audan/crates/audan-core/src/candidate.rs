//! Ranked, non-bare estimates (architecture doc S8.5, ADR-11, Q5).
//!
//! Tempo octave ambiguity and relative-key confusion are genuine perceptual
//! ambiguities, not bugs. Reporting a single number hides irreducible
//! ambiguity, so every estimator in `audan` returns a [`Ranked<T>`] instead of
//! a bare point estimate.

use serde::{Deserialize, Serialize};

/// A confidence in `[0.0, 1.0]`. Not a probability in the measure-theoretic
/// sense -- estimators are free to define it as they see fit -- but always
/// comparable within one estimator's output.
pub type Confidence = f32;

/// One candidate value with the estimator's confidence in it.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Candidate<T> {
    pub value: T,
    pub confidence: Confidence,
}

impl<T> Candidate<T> {
    pub fn new(value: T, confidence: Confidence) -> Self {
        Self { value, confidence }
    }
}

/// A non-empty, descending-by-confidence list of candidates. The top candidate
/// is always `ranked.top()`; human-readable output shows only that one unless
/// `-v`, but JSON always carries the full ranking (S8.5).
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(try_from = "Vec<Candidate<T>>", into = "Vec<Candidate<T>>")]
pub struct Ranked<T: Clone>(Vec<Candidate<T>>);

/// Constructing a [`Ranked<T>`] from an empty list is a programmer error in
/// this codebase: every estimator must produce at least one candidate.
#[derive(thiserror::Error, Debug)]
#[error("Ranked<T> requires at least one candidate")]
pub struct EmptyCandidates;

impl<T: Clone> Ranked<T> {
    /// Sorts `candidates` descending by confidence and wraps them. Fails if
    /// `candidates` is empty.
    pub fn new(mut candidates: Vec<Candidate<T>>) -> Result<Self, EmptyCandidates> {
        if candidates.is_empty() {
            return Err(EmptyCandidates);
        }
        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(Ranked(candidates))
    }

    /// Wraps a single candidate at confidence 1.0. Convenient for estimators
    /// that are not (yet) ambiguity-aware.
    pub fn single(value: T) -> Self {
        Ranked(vec![Candidate::new(value, 1.0)])
    }

    pub fn top(&self) -> &Candidate<T> {
        // Invariant upheld by `new` and `single`: never empty.
        &self.0[0]
    }

    pub fn iter(&self) -> impl Iterator<Item = &Candidate<T>> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn as_slice(&self) -> &[Candidate<T>] {
        &self.0
    }
}

impl<T: Clone> TryFrom<Vec<Candidate<T>>> for Ranked<T> {
    type Error = EmptyCandidates;

    fn try_from(value: Vec<Candidate<T>>) -> Result<Self, Self::Error> {
        Ranked::new(value)
    }
}

impl<T: Clone> From<Ranked<T>> for Vec<Candidate<T>> {
    fn from(r: Ranked<T>) -> Self {
        r.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_descending_by_confidence() {
        let r = Ranked::new(vec![
            Candidate::new(87.0, 0.09),
            Candidate::new(174.0, 0.88),
        ])
        .unwrap();
        assert_eq!(r.top().value, 174.0);
        assert_eq!(r.as_slice()[1].value, 87.0);
    }

    #[test]
    fn rejects_empty() {
        assert!(Ranked::<f32>::new(vec![]).is_err());
    }

    #[test]
    fn round_trips_through_serde() {
        let r = Ranked::new(vec![Candidate::new("Amin", 0.95)]).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: Ranked<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.top().value, "Amin");
    }
}
