//! Window functions. Only Hann is exposed: S8.2 deliberately keeps the window
//! function out of the public/CLI surface (it's folded into `params_hash` but
//! not a user-facing flag), so there is no reason for this module to offer a
//! menu of window shapes.

/// Periodic Hann window of length `n` (i.e. `sym=False` in scipy/librosa
/// terms: `w[n]` is *not* included, only `w[0..n]` of the length-`n+1`
/// symmetric window). This is the variant that satisfies the constant-overlap-add
/// property for `hop = n/4` or `n/2`, which the symmetric variant does not.
pub fn hann(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..n)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_requested_length() {
        assert_eq!(hann(0).len(), 0);
        assert_eq!(hann(1).len(), 1);
        assert_eq!(hann(1024).len(), 1024);
    }

    #[test]
    fn starts_at_zero_and_peaks_near_one() {
        let w = hann(1024);
        assert!(w[0].abs() < 1e-6);
        let max = w.iter().cloned().fold(f32::MIN, f32::max);
        assert!(max > 0.999 && max <= 1.0 + 1e-6);
    }

    #[test]
    fn is_never_negative_and_never_exceeds_one() {
        for &v in &hann(512) {
            assert!((0.0..=1.0 + 1e-6).contains(&v));
        }
    }

    #[test]
    fn periodic_variant_does_not_repeat_the_endpoint() {
        // For the periodic (sym=False) window, w[0] == 0 but the value that
        // *would* be w[n] (0.5 - 0.5*cos(2*pi)) is never materialised, so
        // w[n-1] is strictly less than the symmetric window's peak of 1.0
        // for even n -- it's one sample shy of completing the cosine cycle.
        let w = hann(8);
        assert_ne!(w[0], w[7]);
    }
}
