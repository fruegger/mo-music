//! Splits arbitrary-length mel-frame sequences into fixed-size, overlapping
//! chunks for [`crate::model::OnnxBackend`] (whose exported model only
//! accepts exactly [`CHUNK_SIZE`]-frame input), runs them, and stitches the
//! per-chunk outputs back into one full-length sequence.
//!
//! Ported from `beat_this.inference`'s `split_piece` / `aggregate_prediction`
//! (traced from the upstream source, not guessed): stride = `CHUNK_SIZE -
//! 2*BORDER_SIZE`, the first/last chunk are zero-padded by `BORDER_SIZE` at
//! the absolute start/end of the piece only, `BORDER_SIZE` frames are
//! discarded from each end of every chunk's *output* before stitching, and
//! `"keep_first"` overlap resolution applies (earlier chunks win in the
//! region where the length-preserving last chunk overlaps its predecessor).
//!
//! One deliberate deviation from upstream: pieces shorter than `CHUNK_SIZE -
//! 2*BORDER_SIZE` get a single chunk zero-padded up to exactly
//! [`CHUNK_SIZE`] frames, rather than upstream's shorter, variably-sized
//! chunk (which the fixed-shape ONNX export this backend uses cannot
//! accept). This is the same padding formula as the normal multi-chunk case
//! -- see [`chunk_bounds`] -- just with a right-pad wide enough to reach
//! `CHUNK_SIZE` instead of being capped at `BORDER_SIZE`; that generality is
//! what lets one formula cover both cases without a special case.

pub const CHUNK_SIZE: usize = 1500;
pub const BORDER_SIZE: usize = 6;
const STRIDE: usize = CHUNK_SIZE - 2 * BORDER_SIZE;

/// The starting frame index (in real-signal coordinates, may be negative) of
/// each chunk needed to cover a piece of `len` frames. Mirrors
/// `split_piece`'s `starts` with `avoid_short_end=True`.
pub fn chunk_starts(len: usize) -> Vec<isize> {
    if len <= STRIDE {
        return vec![-(BORDER_SIZE as isize)];
    }
    let mut starts = Vec::new();
    let mut s: isize = -(BORDER_SIZE as isize);
    while s < len as isize - BORDER_SIZE as isize {
        starts.push(s);
        s += STRIDE as isize;
    }
    let last = starts.len() - 1;
    starts[last] = len as isize - (CHUNK_SIZE as isize - BORDER_SIZE as isize);
    starts
}

/// For chunk `start`, the `[region_start, region_end)` span of real frames it
/// covers and how many zero-frames to pad on the left/right to reach exactly
/// [`CHUNK_SIZE`]. `left + (region_end - region_start) + right ==
/// CHUNK_SIZE` always holds by construction.
pub fn chunk_bounds(start: isize, len: usize) -> (usize, usize, usize, usize) {
    let region_start = start.max(0) as usize;
    let region_end = ((start + CHUNK_SIZE as isize).min(len as isize)).max(0) as usize;
    let left = (-start).max(0) as usize;
    let right = CHUNK_SIZE - left - (region_end - region_start);
    (region_start, region_end, left, right)
}

/// Builds one zero-padded, exactly-[`CHUNK_SIZE`]-frame chunk from `frames`
/// (frame-major, `frames.len() == len`) for the given chunk `start`.
pub fn build_chunk(frames: &[Vec<f32>], start: isize) -> Vec<Vec<f32>> {
    let n_mels = frames.first().map(|f| f.len()).unwrap_or(0);
    let (region_start, region_end, left, right) = chunk_bounds(start, frames.len());
    let mut chunk = Vec::with_capacity(CHUNK_SIZE);
    chunk.extend(std::iter::repeat(vec![0.0f32; n_mels]).take(left));
    chunk.extend(frames[region_start..region_end].iter().cloned());
    chunk.extend(std::iter::repeat(vec![0.0f32; n_mels]).take(right));
    debug_assert_eq!(chunk.len(), CHUNK_SIZE);
    chunk
}

/// Stitches per-chunk model outputs (each exactly [`CHUNK_SIZE`] values, one
/// per frame) back into one `len`-length sequence: discards [`BORDER_SIZE`]
/// values from each end of every chunk, then applies `"keep_first"` overlap
/// resolution -- chunks are folded in reverse order so that predictions from
/// earlier (lower-indexed) chunks overwrite predictions from later ones
/// wherever the length-preserving last chunk overlaps its predecessor.
pub fn aggregate(starts: &[isize], chunk_outputs: &[Vec<f32>], len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    let mut written = vec![false; len];

    for (&start, chunk_out) in starts.iter().zip(chunk_outputs.iter()).rev() {
        debug_assert_eq!(chunk_out.len(), CHUNK_SIZE);
        let kept = &chunk_out[BORDER_SIZE..CHUNK_SIZE - BORDER_SIZE];
        // kept[i] is the prediction for real-signal position start + BORDER_SIZE + i.
        let kept_start = start + BORDER_SIZE as isize;
        let target_start = kept_start.max(0) as usize;
        let target_end = (kept_start + kept.len() as isize).min(len as isize).max(0) as usize;
        let skip = (target_start as isize - kept_start) as usize;
        for (offset, pos) in (target_start..target_end).enumerate() {
            out[pos] = kept[skip + offset];
            written[pos] = true;
        }
    }
    debug_assert!(written.iter().all(|&w| w), "chunk_starts must cover every frame");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frames(len: usize) -> Vec<Vec<f32>> {
        (0..len).map(|i| vec![i as f32]).collect()
    }

    #[test]
    fn short_piece_gets_a_single_chunk() {
        let starts = chunk_starts(100);
        assert_eq!(starts, vec![-6]);
    }

    #[test]
    fn short_piece_chunk_is_padded_to_exactly_chunk_size() {
        let frames = make_frames(100);
        let chunk = build_chunk(&frames, -6);
        assert_eq!(chunk.len(), CHUNK_SIZE);
        // first 6 are left padding (zero), next 100 are the real frames,
        // rest is right padding (zero).
        for f in &chunk[..6] {
            assert_eq!(f[0], 0.0);
        }
        for (i, f) in chunk[6..106].iter().enumerate() {
            assert_eq!(f[0], i as f32);
        }
        for f in &chunk[106..] {
            assert_eq!(f[0], 0.0);
        }
    }

    #[test]
    fn short_piece_aggregate_recovers_exactly_len_frames() {
        let len = 100;
        let starts = chunk_starts(len);
        // Pretend the "model" just returns the chunk's own mel value (index
        // 0 channel) as its activation, so we can check aggregation by eye.
        let frames = make_frames(len);
        let chunk_outputs: Vec<Vec<f32>> = starts
            .iter()
            .map(|&s| build_chunk(&frames, s).iter().map(|f| f[0]).collect())
            .collect();
        let out = aggregate(&starts, &chunk_outputs, len);
        assert_eq!(out.len(), len);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i as f32, "position {i}");
        }
    }

    #[test]
    fn long_piece_needs_multiple_chunks_and_they_are_all_exactly_chunk_size() {
        let len = 3000; // > STRIDE (1488), so this exercises the multi-chunk path
        let starts = chunk_starts(len);
        assert!(starts.len() > 1);
        let frames = make_frames(len);
        for &s in &starts {
            let chunk = build_chunk(&frames, s);
            assert_eq!(chunk.len(), CHUNK_SIZE);
        }
        // avoid_short_end: the last start must place the chunk's end exactly
        // at the end of the piece.
        let last = *starts.last().unwrap();
        assert_eq!(last, len as isize - (CHUNK_SIZE as isize - BORDER_SIZE as isize));
    }

    #[test]
    fn long_piece_aggregate_recovers_every_position_exactly() {
        let len = 3000;
        let starts = chunk_starts(len);
        let frames = make_frames(len);
        let chunk_outputs: Vec<Vec<f32>> = starts
            .iter()
            .map(|&s| build_chunk(&frames, s).iter().map(|f| f[0]).collect())
            .collect();
        let out = aggregate(&starts, &chunk_outputs, len);
        assert_eq!(out.len(), len);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i as f32, "position {i}");
        }
    }

    #[test]
    fn exact_multiple_of_stride_length_is_still_fully_covered() {
        // A length chosen so the naive avoid_short_end shift could
        // theoretically leave a gap or duplicate coverage if the boundary
        // math were off by one.
        for len in [STRIDE, STRIDE + 1, STRIDE * 2, STRIDE * 2 + 1, STRIDE * 3 - 1] {
            let starts = chunk_starts(len);
            let frames = make_frames(len);
            let chunk_outputs: Vec<Vec<f32>> = starts
                .iter()
                .map(|&s| build_chunk(&frames, s).iter().map(|f| f[0]).collect())
                .collect();
            let out = aggregate(&starts, &chunk_outputs, len);
            for (i, &v) in out.iter().enumerate() {
                assert_eq!(v, i as f32, "len {len}, position {i}");
            }
        }
    }
}
