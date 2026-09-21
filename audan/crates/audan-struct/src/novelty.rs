//! Foote novelty detection (architecture glossary "Foote novelty", RISK-8): a
//! checkerboard kernel correlated along an SSM's main diagonal, producing a
//! 1-D curve whose peaks are candidate structural boundaries.

/// Computes the novelty curve for a self-similarity matrix `ssm` using a
/// `2 * kernel_half_size` square checkerboard kernel centred on each frame.
/// The output has the same length as `ssm`.
///
/// The checkerboard is Gaussian-tapered rather than hard-edged, which is the
/// standard Foote treatment: a hard-edged kernel's sharp cutoff means every
/// frame the kernel's edge slides past produces a step discontinuity in the
/// correlation sum, injecting high-frequency ringing into the novelty curve
/// that has nothing to do with real structure. Weighting samples down
/// smoothly towards the kernel's edges removes that ringing without blurring
/// away genuine boundaries, since the boundary frame itself still sits at the
/// kernel's centre where the taper weight is highest.
///
/// Frames within `kernel_half_size` of either end of the SSM don't have a
/// full kernel footprint and are left at `0.0` rather than computed from a
/// shrunk kernel: a shrunk kernel implicitly averages over fewer quadrant
/// cells, so its value would sit on a different scale than interior frames
/// and wouldn't be comparable for peak-picking. `segment::segment_structure`
/// always forces beat `0` and the final beat as boundaries anyway, so there's
/// nothing to gain from estimating novelty in that margin.
pub fn novelty_curve(ssm: &[Vec<f32>], kernel_half_size: usize) -> Vec<f32> {
    let n = ssm.len();
    let mut out = vec![0f32; n];
    if kernel_half_size == 0 || n == 0 {
        return out;
    }

    let l = kernel_half_size as isize;
    let sigma = kernel_half_size as f32 / 2.0;
    let two_sigma_sq = 2.0 * sigma * sigma;

    // The tapered, signed checkerboard kernel depends only on the offset
    // from its centre, so it's identical at every interior frame -- build it
    // once rather than per-frame.
    let size = 2 * kernel_half_size;
    let mut kernel = vec![vec![0f32; size]; size];
    for (ia, a) in (-l..l).enumerate() {
        for (ib, b) in (-l..l).enumerate() {
            let sign = if (a < 0) == (b < 0) { 1.0 } else { -1.0 };
            let dist_sq = (a * a + b * b) as f32;
            kernel[ia][ib] = sign * (-dist_sq / two_sigma_sq).exp();
        }
    }

    let hi = n.saturating_sub(kernel_half_size);
    for i in kernel_half_size..hi {
        let mut acc = 0f32;
        for (ia, a) in (-l..l).enumerate() {
            let ii = (i as isize + a) as usize;
            let row = &ssm[ii];
            let krow = &kernel[ia];
            for (ib, b) in (-l..l).enumerate() {
                let jj = (i as isize + b) as usize;
                acc += krow[ib] * row[jj];
            }
        }
        out[i] = acc;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_dsp::cosine_similarity_matrix;

    fn block_diagonal_ssm() -> Vec<Vec<f32>> {
        let mut features = Vec::new();
        for _ in 0..20 {
            features.push(vec![1.0, 0.0, 0.0]);
        }
        for _ in 0..20 {
            features.push(vec![0.0, 1.0, 0.0]);
        }
        for _ in 0..20 {
            features.push(vec![0.0, 0.0, 1.0]);
        }
        cosine_similarity_matrix(&features)
    }

    #[test]
    fn peaks_at_block_boundaries_and_low_within_blocks() {
        let ssm = block_diagonal_ssm();
        let novelty = novelty_curve(&ssm, 6);

        // The true boundaries are at frames 20 and 40.
        let boundary_peak = |centre: usize| {
            (centre - 3..=centre + 3)
                .map(|i| novelty[i])
                .fold(f32::MIN, f32::max)
        };
        let interior_value =
            |centre: usize| (centre - 3..=centre + 3).map(|i| novelty[i]).sum::<f32>() / 7.0;

        let peak_20 = boundary_peak(20);
        let peak_40 = boundary_peak(40);
        let interior_a = interior_value(10);
        let interior_b = interior_value(30);
        let interior_c = interior_value(50);

        assert!(peak_20 > interior_a && peak_20 > interior_b);
        assert!(peak_40 > interior_b && peak_40 > interior_c);
    }

    #[test]
    fn edges_within_half_kernel_are_zero() {
        let ssm = block_diagonal_ssm();
        let l = 6;
        let novelty = novelty_curve(&ssm, l);
        for i in 0..l {
            assert_eq!(novelty[i], 0.0);
        }
        for i in (ssm.len() - l)..ssm.len() {
            assert_eq!(novelty[i], 0.0);
        }
    }

    #[test]
    fn empty_ssm_and_zero_kernel_do_not_panic() {
        assert_eq!(novelty_curve(&[], 4), Vec::<f32>::new());
        let ssm = block_diagonal_ssm();
        assert_eq!(novelty_curve(&ssm, 0), vec![0f32; ssm.len()]);
    }
}
