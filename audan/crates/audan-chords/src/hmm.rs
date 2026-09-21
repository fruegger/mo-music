//! A minimal hidden Markov model and Viterbi decoder, generic over the state
//! space so [`crate::estimate_chords`] can wire it to the (24 chord + 1
//! no-chord) state space without this module knowing about chords at all.
//!
//! Follows the classic Sheh & Ellis 2003 / Bello & Pickens 2005 approach:
//! noisy per-beat template-matching scores are treated as emission
//! log-likelihoods and smoothed by a transition matrix that strongly favours
//! *staying* in the current chord, since chords typically persist for
//! several beats and per-beat chroma is noisy enough that a naive per-beat
//! argmax flickers between neighbouring chords. This is the actual
//! contribution of the HMM stage over template matching alone.

/// Builds an ergodic (fully connected) transition matrix biased toward
/// self-transition. `p_self` is the probability of staying in the same
/// state from one beat to the next; the remaining probability mass is
/// spread uniformly over every other state. A real corpus would learn this
/// from beat-annotated data (as Bello & Pickens do); a single hand-picked
/// persistence constant is the standard simplification when no training
/// corpus is in scope (ADR-12's whole premise), and 0.9 is in the range the
/// literature uses for beat-synchronous chord HMMs.
pub fn persistence_transition_log(n_states: usize, p_self: f32) -> Vec<Vec<f32>> {
    assert!(
        n_states > 1,
        "need at least two states for a transition matrix"
    );
    let p_other = (1.0 - p_self) / (n_states - 1) as f32;
    let log_self = p_self.ln();
    let log_other = p_other.ln();
    (0..n_states)
        .map(|i| {
            (0..n_states)
                .map(|j| if i == j { log_self } else { log_other })
                .collect()
        })
        .collect()
}

/// Decodes the maximum-likelihood state sequence given per-beat emission
/// log-likelihoods (`emission_log[t][s]`), a transition log-probability
/// matrix (`log_trans[i][j]`, from state `i` to state `j`), and a log prior
/// over the first beat's state. Returns one state index per beat; empty
/// input yields an empty path.
pub fn viterbi(emission_log: &[Vec<f32>], log_trans: &[Vec<f32>], log_prior: &[f32]) -> Vec<usize> {
    let n_t = emission_log.len();
    if n_t == 0 {
        return Vec::new();
    }
    let n_s = log_prior.len();

    let mut delta = vec![vec![f32::NEG_INFINITY; n_s]; n_t];
    let mut psi = vec![vec![0usize; n_s]; n_t];

    for s in 0..n_s {
        delta[0][s] = log_prior[s] + emission_log[0][s];
    }

    for t in 1..n_t {
        for s in 0..n_s {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_prev = 0usize;
            for i in 0..n_s {
                let score = delta[t - 1][i] + log_trans[i][s];
                if score > best_score {
                    best_score = score;
                    best_prev = i;
                }
            }
            delta[t][s] = best_score + emission_log[t][s];
            psi[t][s] = best_prev;
        }
    }

    let mut path = vec![0usize; n_t];
    path[n_t - 1] = (0..n_s)
        .max_by(|&a, &b| delta[n_t - 1][a].partial_cmp(&delta[n_t - 1][b]).unwrap())
        .unwrap();
    for t in (1..n_t).rev() {
        path[t - 1] = psi[t][path[t]];
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_matrix_favours_self_transition() {
        let m = persistence_transition_log(3, 0.9);
        assert!(m[0][0] > m[0][1]);
        assert!(m[1][1] > m[1][0]);
    }

    #[test]
    fn empty_input_yields_empty_path() {
        let path = viterbi(&[], &persistence_transition_log(2, 0.9), &[0.0, 0.0]);
        assert!(path.is_empty());
    }

    #[test]
    fn smooths_a_single_noisy_flicker_between_persistent_states() {
        // Three states; state 0 is strongly favoured throughout except a
        // single noisy beat that (weakly) favours state 1. A naive per-beat
        // argmax would flicker to state 1 for one beat; Viterbi with a
        // strong persistence prior should not.
        let log_trans = persistence_transition_log(3, 0.9);
        let log_prior = [0.0f32, 0.0, 0.0];
        let strong = vec![2.0f32, 0.0, 0.0];
        let weak_flicker = vec![0.0f32, 0.3, 0.0];
        let emission_log = vec![
            strong.clone(),
            strong.clone(),
            weak_flicker,
            strong.clone(),
            strong,
        ];

        let path = viterbi(&emission_log, &log_trans, &log_prior);
        assert_eq!(path, vec![0, 0, 0, 0, 0]);
    }
}
