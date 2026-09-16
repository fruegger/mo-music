//! Chunked inference and overlap-add reconstruction (S5.1 `audan-stems` row).
//!
//! Generic DSP plumbing, independent of any specific model: a real
//! separation backend processes fixed-size chunks rather than an
//! arbitrary-length track, and stitching the chunk outputs back together
//! with a hard cut at each boundary produces audible clicks. These utilities
//! do the windowed crossfade instead, and are usable by any future
//! [`crate::Separator`] implementation.

/// One fixed-length, possibly zero-padded window over a longer signal, plus
/// the sample offset (in the original signal) it starts at.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub start: usize,
    pub samples: Vec<f32>,
}

/// Splits `signal` into overlapping, fixed-length chunks of `chunk_len`
/// samples, advancing by `chunk_len - overlap` each step. The final chunk is
/// zero-padded on the right if the signal doesn't divide evenly -- every
/// chunk is always exactly `chunk_len` samples, since that's what a chunked
/// model expects to be fed.
///
/// # Panics
/// If `chunk_len` is zero or `overlap >= chunk_len`.
pub fn chunk_signal(signal: &[f32], chunk_len: usize, overlap: usize) -> Vec<Chunk> {
    assert!(chunk_len > 0, "chunk_len must be non-zero");
    assert!(
        overlap < chunk_len,
        "overlap must be smaller than chunk_len"
    );

    if signal.is_empty() {
        return Vec::new();
    }

    let stride = chunk_len - overlap;
    let mut chunks = Vec::new();
    let mut start = 0usize;
    loop {
        let end = (start + chunk_len).min(signal.len());
        let mut samples = vec![0.0f32; chunk_len];
        samples[..end - start].copy_from_slice(&signal[start..end]);
        chunks.push(Chunk { start, samples });
        if end >= signal.len() {
            break;
        }
        start += stride;
    }
    chunks
}

/// Reassembles `chunks` (each exactly `chunk_len` samples, positioned at
/// `i * (chunk_len - overlap)` as produced by [`chunk_signal`]) into a
/// `total_len`-sample signal.
///
/// Overlapping regions are blended with a raised-cosine taper rather than a
/// hard cut, which is what makes this safe to use on real (non-identity)
/// per-chunk model output without introducing boundary clicks: each chunk is
/// weighted by the taper and every output sample is renormalised by the sum
/// of the weights that landed on it, so the taper's shape cancels out
/// exactly for constant (or slowly varying) content across the overlap and
/// only actually matters where consecutive chunks disagree.
///
/// # Panics
/// If `chunk_len` is zero, `overlap >= chunk_len`, or any chunk is not
/// exactly `chunk_len` samples long.
pub fn overlap_add_reconstruct(
    chunks: &[Vec<f32>],
    chunk_len: usize,
    overlap: usize,
    total_len: usize,
) -> Vec<f32> {
    assert!(chunk_len > 0, "chunk_len must be non-zero");
    assert!(
        overlap < chunk_len,
        "overlap must be smaller than chunk_len"
    );
    for c in chunks {
        assert_eq!(
            c.len(),
            chunk_len,
            "every chunk must be exactly chunk_len samples"
        );
    }

    let stride = chunk_len - overlap;
    let taper = raised_cosine_taper(chunk_len, overlap);

    let mut out = vec![0.0f32; total_len];
    let mut weight = vec![0.0f32; total_len];

    for (i, chunk) in chunks.iter().enumerate() {
        let start = i * stride;
        for (j, &sample) in chunk.iter().enumerate() {
            let pos = start + j;
            if pos >= total_len {
                break;
            }
            out[pos] += sample * taper[j];
            weight[pos] += taper[j];
        }
    }

    for i in 0..total_len {
        if weight[i] > 1e-8 {
            out[i] /= weight[i];
        }
    }

    out
}

/// A window that is 1.0 across the flat middle of a chunk and ramps with a
/// raised cosine (Hann-style: smooth, zero-derivative at both ends) over
/// `overlap` samples at each edge. Never exactly zero, so a sample that is
/// only ever covered by one chunk (the very first/last `overlap` samples of
/// the whole signal) still renormalises to its true value rather than
/// dividing by zero.
fn raised_cosine_taper(chunk_len: usize, overlap: usize) -> Vec<f32> {
    let mut w = vec![1.0f32; chunk_len];
    if overlap == 0 {
        return w;
    }
    for i in 0..overlap {
        let t = (i as f32 + 1.0) / (overlap as f32 + 1.0);
        let ramp = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
        w[i] = ramp;
        w[chunk_len - 1 - i] = ramp;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_signal_produces_expected_count_length_and_overlap() {
        let signal: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let chunks = chunk_signal(&signal, 8, 3);

        assert_eq!(chunks.len(), 4);
        for c in &chunks {
            assert_eq!(c.samples.len(), 8);
        }
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[1].start, 5);
        assert_eq!(chunks[2].start, 10);
        assert_eq!(chunks[3].start, 15);

        // stride = chunk_len - overlap = 5, so consecutive chunks overlap by
        // exactly 3 samples in original-signal coordinates.
        assert_eq!(chunks[0].start + 8 - chunks[1].start, 3);

        // Last chunk covers [15, 20) of real signal, zero-padded for the
        // remaining 3 samples.
        assert_eq!(&chunks[3].samples[..5], &signal[15..20]);
        assert_eq!(&chunks[3].samples[5..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn chunk_signal_exact_multiple_has_no_padding() {
        let signal: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let chunks = chunk_signal(&signal, 8, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].samples, signal[0..8]);
        assert_eq!(chunks[1].samples, signal[8..16]);
    }

    #[test]
    fn chunk_signal_empty_signal_yields_no_chunks() {
        assert!(chunk_signal(&[], 8, 2).is_empty());
    }

    #[test]
    #[should_panic(expected = "overlap must be smaller")]
    fn chunk_signal_rejects_overlap_not_smaller_than_chunk_len() {
        chunk_signal(&[1.0, 2.0, 3.0], 4, 4);
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "sample {i}: {a} vs {e} (tolerance {tol})"
            );
        }
    }

    #[test]
    fn overlap_add_reconstructs_identity_chunks_closely() {
        let total_len = 50usize;
        let signal: Vec<f32> = (0..total_len).map(|i| (i as f32 * 0.3).sin()).collect();
        let chunk_len = 16;
        let overlap = 4;

        let chunks = chunk_signal(&signal, chunk_len, overlap);
        let identity: Vec<Vec<f32>> = chunks.iter().map(|c| c.samples.clone()).collect();

        let reconstructed = overlap_add_reconstruct(&identity, chunk_len, overlap, total_len);

        // Identity-processed chunks reconstruct the original signal; allow a
        // small tolerance for the taper's edge behaviour rather than
        // demanding bit-exact equality.
        assert_close(&reconstructed, &signal, 1e-4);
    }

    #[test]
    fn overlap_add_with_zero_overlap_is_exact_hard_cut() {
        let total_len = 16usize;
        let signal: Vec<f32> = (0..total_len).map(|i| i as f32).collect();
        let chunk_len = 8;
        let overlap = 0;

        let chunks = chunk_signal(&signal, chunk_len, overlap);
        let identity: Vec<Vec<f32>> = chunks.iter().map(|c| c.samples.clone()).collect();
        let reconstructed = overlap_add_reconstruct(&identity, chunk_len, overlap, total_len);

        assert_eq!(reconstructed, signal);
    }

    #[test]
    fn overlap_add_truncates_to_total_len() {
        let signal: Vec<f32> = vec![1.0; 20];
        let chunk_len = 8;
        let overlap = 2;
        let chunks = chunk_signal(&signal, chunk_len, overlap);
        let identity: Vec<Vec<f32>> = chunks.iter().map(|c| c.samples.clone()).collect();

        // total_len shorter than the padded chunk coverage.
        let reconstructed = overlap_add_reconstruct(&identity, chunk_len, overlap, 20);
        assert_eq!(reconstructed.len(), 20);
        assert_close(&reconstructed, &signal, 1e-4);
    }
}
