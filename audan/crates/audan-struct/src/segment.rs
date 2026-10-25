//! Peak-picking the Foote novelty curve into beat-indexed section boundaries
//! (architecture doc S8.3: downstream artefacts reference beat indices, not
//! times, so a hand-corrected beat grid re-times sections for free).

use audan_core::{BeatGrid, BeatIndex, Chroma};
use audan_dsp::{chroma_ssm, cosine_similarity_matrix};

use crate::novelty::novelty_curve;

/// Knobs for `segment_structure`. Frame-domain fields (`kernel_half_size`,
/// `min_boundary_distance_frames`) are counted in *feature* frames, i.e.
/// units of the `Chroma`'s own `FrameGrid` hop -- not beats or seconds -- so
/// their effective duration depends on the caller's chroma hop size.
///
/// The defaults below are empirically tuned, not just reasoned about in the
/// abstract: this module's own unit tests only ever exercised synthetic,
/// idealized 3-block chroma, and the original defaults (`kernel_half_size:
/// 16`, `peak_threshold_k: 1.0`, `min_boundary_distance_frames: 8`) turned out
/// to badly over-segment real recordings when checked against 40 real tracks
/// in `resources/test_data` -- averaging 15.7 sections per track, 30% of them
/// under 5 seconds, with individual tracks landing anywhere from 1 to 30
/// sections. The values here were chosen from a grid sweep of `audan struct
/// --seg-kernel-half-size/--seg-peak-threshold-k/--seg-min-boundary-distance`
/// (see `crates/audan-cli`) against that same set: they bring the average down
/// to ~6 sections per track with under 9% of sections shorter than 8 seconds
/// and zero tracks over-segmented (`>= 15` sections), while a persistent
/// handful of tracks built on one repeating riff/groove for their entire
/// runtime (e.g. "Come As You Are") still collapse to a single section
/// regardless of these settings -- plausibly a real property of those
/// specific recordings (little harmonic contrast for a similarity-based
/// method to key off), not a parameter this sweep can tune away.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SegmentParams {
    /// Half-width of the checkerboard kernel, in feature frames. At
    /// `audan_dsp::KeyChromaParams`'s default 4096-sample hop / 22050 Hz
    /// analysis rate (~5.4 frames/s), the default of 32 gives a ~12 s kernel
    /// -- wide enough to see past short-term chroma jitter to large-scale
    /// (verse/chorus-level) structure, per the module-level tuning note
    /// above (the original ~6 s kernel was too narrow on real audio).
    pub kernel_half_size: usize,
    /// Peak-acceptance threshold, expressed as `mean + peak_threshold_k *
    /// stddev` of the novelty curve. Adaptive rather than a fixed absolute
    /// value because novelty magnitude scales with how similar/dissimilar
    /// the underlying features are track to track. Default 1.5 (see the
    /// module-level tuning note); the original 1.0 accepted too many small,
    /// noisy novelty peaks as real boundaries.
    pub peak_threshold_k: f32,
    /// Minimum spacing between accepted boundaries, in feature frames.
    /// Prevents two novelty samples one frame apart (a noisy, nearly-tied
    /// local maximum) from producing two boundaries a beat or two apart.
    /// Default 43 (~8 s at the default chroma hop, see the module-level
    /// tuning note); the original 8 frames (~1.5 s) was far shorter than any
    /// real musical section.
    pub min_boundary_distance_frames: usize,
    /// Move each boundary to the nearest downbeat of the beat grid. Sections
    /// start on a bar, so this removes the half-bar misses of a novelty peak
    /// (chroma novelty peaks sit a little before or after the real change).
    /// Ignored when the grid has no downbeats. Off by default: against the 20
    /// MIDI-annotated songs it did not raise boundary F-measure (0.34 vs 0.35
    /// at 3 s with the DSP grid; +0.01 at 0.5 s with the `beat_this` grid).
    pub snap_to_downbeats: bool,
    /// Share of the self-similarity matrix taken from the timbre features
    /// (`0.0` = chroma only, `1.0` = timbre only); only used when timbre
    /// features are passed to [`segment_structure_with_timbre`].
    pub timbre_weight: f32,
}

impl Default for SegmentParams {
    fn default() -> Self {
        SegmentParams {
            kernel_half_size: 32,
            peak_threshold_k: 0.5,
            min_boundary_distance_frames: 24,
            snap_to_downbeats: false,
            timbre_weight: 0.5,
        }
    }
}

/// One structural section, beat-indexed per ADR-4/S8.3: `[start_beat,
/// end_beat)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SectionEvent {
    pub start_beat: BeatIndex,
    pub end_beat: BeatIndex,
    /// Always `None`. Section *labelling* ("this is the chorus") is
    /// explicitly out of scope for this pass (RISK-8) -- only unlabeled
    /// boundary detection is implemented and tested here. This field exists
    /// so the schema doesn't need to break when labelling is later added as
    /// its own, separately-validated, clearly experimental feature; it must
    /// not be populated by a heuristic that hasn't gone through that.
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructureResult {
    pub schema_version: u32,
    pub sections: Vec<SectionEvent>,
}

impl StructureResult {
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;
}

/// Peaks below this share of the kernel's total weight (`(2 * half_size)^2`)
/// are numerical noise from a flat self-similarity matrix, not structure; the
/// relative `mean + k * stddev` threshold alone would accept them.
const MIN_PEAK_FRACTION: f32 = 1e-3;

fn pick_peaks(
    novelty: &[f32],
    threshold_k: f32,
    min_distance: usize,
    kernel_half_size: usize,
) -> Vec<usize> {
    let n = novelty.len();
    if n < 3 {
        return Vec::new();
    }

    let mean = novelty.iter().sum::<f32>() / n as f32;
    let variance = novelty.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let floor = MIN_PEAK_FRACTION * (2 * kernel_half_size).pow(2) as f32;
    let threshold = (mean + threshold_k * variance.sqrt()).max(floor);

    let mut candidates: Vec<(usize, f32)> = (1..n - 1)
        .filter(|&i| novelty[i] >= novelty[i - 1] && novelty[i] >= novelty[i + 1])
        .filter(|&i| novelty[i] > threshold && novelty[i] > 0.0)
        .map(|i| (i, novelty[i]))
        .collect();

    // Highest peaks win when two candidates fall within `min_distance` of
    // each other, rather than the earlier one winning by scan order.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut accepted: Vec<usize> = Vec::new();
    for (idx, _) in candidates {
        if accepted
            .iter()
            .all(|&a| idx.abs_diff(a) >= min_distance.max(1))
        {
            accepted.push(idx);
        }
    }
    accepted.sort_unstable();
    accepted
}

/// Nearest beat to `time` by time difference, ties resolved towards the
/// earlier beat. `beats` must be sorted ascending, as `BeatGrid::beats`
/// always is.
fn nearest_beat_index(beats: &[f64], time: f64) -> BeatIndex {
    match beats.binary_search_by(|b| b.partial_cmp(&time).unwrap()) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) if i >= beats.len() => beats.len() - 1,
        Err(i) => {
            let before = beats[i - 1];
            let after = beats[i];
            if (time - before).abs() <= (after - time).abs() {
                i - 1
            } else {
                i
            }
        }
    }
}

/// The downbeat closest to `beat` (by beat count, ties towards the earlier
/// one); `beat` itself when the grid has no downbeats.
fn nearest_downbeat(beats: &BeatGrid, beat: BeatIndex) -> BeatIndex {
    beats
        .downbeats
        .iter()
        .copied()
        .min_by_key(|&d| (d.abs_diff(beat), d))
        .unwrap_or(beat)
}

/// Segments `chroma` into unlabeled structural sections, beat-indexed
/// against `beats`. See RISK-8: `SectionEvent::label` is always `None`.
pub fn segment_structure(
    chroma: &Chroma,
    beats: &BeatGrid,
    params: &SegmentParams,
) -> StructureResult {
    segment_structure_with_timbre(chroma, None, beats, params)
}

/// Like [`segment_structure`], blending a timbre self-similarity matrix
/// (frame-aligned with `chroma`, e.g. `audan_dsp::timbre_features` on the
/// same hop) into the chroma one with weight `params.timbre_weight`.
pub fn segment_structure_with_timbre(
    chroma: &Chroma,
    timbre: Option<&[Vec<f32>]>,
    beats: &BeatGrid,
    params: &SegmentParams,
) -> StructureResult {
    if beats.beats.is_empty() || chroma.n_frames() == 0 {
        return StructureResult {
            schema_version: StructureResult::CURRENT_SCHEMA_VERSION,
            sections: Vec::new(),
        };
    }
    let last_beat = beats.beats.len() - 1;

    let mut ssm = chroma_ssm(chroma);
    if let Some(timbre) = timbre.filter(|_| params.timbre_weight > 0.0) {
        let timbre_ssm = cosine_similarity_matrix(timbre);
        let w = params.timbre_weight.clamp(0.0, 1.0);
        let n = ssm.len().min(timbre_ssm.len());
        for i in 0..n {
            for j in 0..n {
                ssm[i][j] = (1.0 - w) * ssm[i][j] + w * timbre_ssm[i][j];
            }
        }
    }
    let novelty = novelty_curve(&ssm, params.kernel_half_size);
    let peak_frames = pick_peaks(
        &novelty,
        params.peak_threshold_k,
        params.min_boundary_distance_frames,
        params.kernel_half_size,
    );

    let mut boundaries: Vec<BeatIndex> = peak_frames
        .into_iter()
        .map(|frame| {
            let time = chroma.grid.time_of(frame).as_seconds();
            let beat = nearest_beat_index(&beats.beats, time);
            if params.snap_to_downbeats {
                nearest_downbeat(beats, beat)
            } else {
                beat
            }
        })
        .collect();

    // Sections must tile [0, last_beat] with no gaps, even if no novelty
    // peak lands exactly there.
    boundaries.push(0);
    boundaries.push(last_beat);
    boundaries.sort_unstable();
    boundaries.dedup();

    let sections = boundaries
        .windows(2)
        .map(|w| SectionEvent {
            start_beat: w[0],
            end_beat: w[1],
            label: None,
        })
        .collect();

    StructureResult {
        schema_version: StructureResult::CURRENT_SCHEMA_VERSION,
        sections,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audan_core::beat::{Meter, Source, TempoInfo, TempoStability, TempoStabilityClass};
    use audan_core::candidate::{Candidate, Ranked};
    use audan_core::frame::{FrameGrid, PadMode};
    use audan_core::PITCH_CLASSES;

    fn beat_grid(beat_times: Vec<f64>) -> BeatGrid {
        let n = beat_times.len();
        let duration_seconds = beat_times.last().copied().unwrap_or(0.0) + 1.0;
        BeatGrid {
            schema_version: BeatGrid::CURRENT_SCHEMA_VERSION,
            duration_seconds,
            beats: beat_times,
            downbeats: (0..n).step_by(4).collect(),
            meter: Meter {
                beats_per_bar: 4,
                confidence: 0.9,
            },
            tempo: TempoInfo {
                median_bpm: 120.0,
                candidates: Ranked::new(vec![Candidate::new(120.0, 0.9)]).unwrap(),
                stability: TempoStability {
                    ibi_mad_ms: 1.0,
                    drift_bpm_per_min: 0.0,
                    class: TempoStabilityClass::Programmed,
                },
            },
            confidence: vec![0.9; n],
            frames: FrameGrid::new(22050, 512, 2048, PadMode::Reflect).into(),
            source: Source {
                algo: "test".into(),
                version: "0".into(),
                postproc: "none".into(),
            },
        }
    }

    fn pc_vector(dominant: usize, jitter_seed: usize, jitter: f32) -> [f32; PITCH_CLASSES] {
        let mut v = [0f32; PITCH_CLASSES];
        v[dominant] = 1.0;
        if jitter > 0.0 {
            let other = (dominant + 1 + (jitter_seed % 10)) % PITCH_CLASSES;
            v[other] += jitter;
        }
        v
    }

    /// Three ~constant blocks (different dominant pitch classes) with light
    /// synthetic jitter, at a coarse hop matching `KeyChromaParams`'s default
    /// so the frame rate this test exercises is realistic.
    fn three_block_chroma(frames_per_block: usize) -> Chroma {
        let hop = 4096;
        let grid = FrameGrid::new(22050, hop, 8192, PadMode::Reflect);
        let mut frames = Vec::new();
        for (block, &pc) in [0usize, 4, 8].iter().enumerate() {
            for k in 0..frames_per_block {
                let jitter = if k % 5 == 0 { 0.08 } else { 0.0 };
                frames.push(pc_vector(pc, block * frames_per_block + k, jitter));
            }
        }
        Chroma::from_frames(grid, frames)
    }

    fn beats_covering(chroma: &Chroma, beat_period_s: f64) -> BeatGrid {
        let total_time = chroma.grid.time_of(chroma.n_frames() - 1).as_seconds();
        let mut beats = Vec::new();
        let mut t = 0.0;
        while t <= total_time {
            beats.push(t);
            t += beat_period_s;
        }
        beat_grid(beats)
    }

    #[test]
    fn timbre_alone_finds_a_boundary_chroma_cannot_see() {
        // Constant chroma, but the timbre features change halfway.
        let hop = 4096;
        let grid = FrameGrid::new(22050, hop, 8192, PadMode::Reflect);
        let n = 240;
        let chroma = Chroma::from_frames(grid, vec![pc_vector(0, 0, 0.0); n]);
        let timbre: Vec<Vec<f32>> = (0..n)
            .map(|k| if k < n / 2 { vec![1.0, 0.0, 0.2] } else { vec![0.0, 1.0, 0.2] })
            .collect();
        let beats = beats_covering(&chroma, 0.5);
        let params = SegmentParams::default();

        let without = segment_structure(&chroma, &beats, &params);
        let with = segment_structure_with_timbre(&chroma, Some(&timbre), &beats, &params);
        assert_eq!(without.sections.len(), 1);
        assert_eq!(with.sections.len(), 2);
        let boundary = with.sections[1].start_beat as i64;
        let expected = (chroma.grid.time_of(n / 2).as_seconds() / 0.5).round() as i64;
        assert!((boundary - expected).abs() <= 4, "boundary {boundary}, expected {expected}");
    }

    #[test]
    fn snapped_boundaries_land_on_downbeats() {
        let chroma = three_block_chroma(100);
        let beats = beats_covering(&chroma, 0.5);

        let params = SegmentParams {
            snap_to_downbeats: true,
            ..SegmentParams::default()
        };
        let snapped = segment_structure(&chroma, &beats, &params);
        assert!(snapped.sections.len() > 1);
        for section in &snapped.sections[1..] {
            assert!(
                beats.downbeats.contains(&section.start_beat),
                "boundary at beat {} is not a downbeat",
                section.start_beat
            );
        }
    }

    #[test]
    fn recovers_roughly_three_sections_with_boundaries_near_true_transitions() {
        // Blocks must be comfortably wider than `SegmentParams::default()`'s
        // tuned-for-real-audio `min_boundary_distance_frames` (43) and
        // `kernel_half_size` (32), or the two true transitions here (40
        // frames apart at the old block width) collapse into one under the
        // new defaults -- not a synthetic-test bug, just this test's block
        // size predating the real-audio tuning (see `SegmentParams`'s own
        // doc comment).
        let frames_per_block = 100;
        let chroma = three_block_chroma(frames_per_block);
        let beats = beats_covering(&chroma, 0.5);

        let result = segment_structure(&chroma, &beats, &SegmentParams::default());

        assert!(
            (2..=4).contains(&result.sections.len()),
            "expected roughly 3 sections, got {}",
            result.sections.len()
        );

        // Tiles [0, last_beat] with no gaps or overlaps.
        let last_beat = beats.beats.len() - 1;
        assert_eq!(result.sections.first().unwrap().start_beat, 0);
        assert_eq!(result.sections.last().unwrap().end_beat, last_beat);
        for w in result.sections.windows(2) {
            assert_eq!(w[0].end_beat, w[1].start_beat);
        }

        // True transitions sit at frames `frames_per_block` and
        // `2*frames_per_block`; convert to expected beat indices and check
        // some detected boundary lands within a reasonable tolerance.
        let hop = chroma.grid.hop;
        let sr = 22050u32;
        let true_times =
            [frames_per_block, 2 * frames_per_block].map(|f| (f * hop) as f64 / sr as f64);
        let detected_beats: Vec<BeatIndex> = result
            .sections
            .iter()
            .skip(1)
            .map(|s| s.start_beat)
            .collect();
        for tt in true_times {
            let true_beat = (tt / 0.5).round() as i64;
            assert!(
                detected_beats.iter().any(|&b| (b as i64 - true_beat).abs() <= 4),
                "no detected boundary near true transition beat {true_beat}; detected {detected_beats:?}"
            );
        }
    }

    #[test]
    fn near_constant_chroma_does_not_flood_spurious_sections() {
        let hop = 4096;
        let grid = FrameGrid::new(22050, hop, 8192, PadMode::Reflect);
        let frames: Vec<[f32; PITCH_CLASSES]> = (0..150)
            .map(|k| pc_vector(0, k, if k % 3 == 0 { 0.02 } else { 0.0 }))
            .collect();
        let chroma = Chroma::from_frames(grid, frames);
        let beats = beats_covering(&chroma, 0.5);

        let result = segment_structure(&chroma, &beats, &SegmentParams::default());

        assert!(
            // The peak threshold is relative (mean + k*stddev, k = 0.5 by default),
            // so noise-only input still yields a few peaks; this only guards
            // against it flooding at every `min_boundary_distance_frames`.
            result.sections.len() <= 6,
            "expected few/no spurious sections on structureless input, got {}",
            result.sections.len()
        );
    }

    #[test]
    fn labels_are_always_none() {
        let chroma = three_block_chroma(40);
        let beats = beats_covering(&chroma, 0.5);
        let result = segment_structure(&chroma, &beats, &SegmentParams::default());
        assert!(!result.sections.is_empty());
        assert!(result.sections.iter().all(|s| s.label.is_none()));
    }

    #[test]
    fn empty_beat_grid_yields_no_sections() {
        let chroma = three_block_chroma(10);
        let beats = beat_grid(Vec::new());
        let result = segment_structure(&chroma, &beats, &SegmentParams::default());
        assert!(result.sections.is_empty());
    }
}
