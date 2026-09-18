//! Peak-picking the Foote novelty curve into beat-indexed section boundaries
//! (architecture doc S8.3: downstream artefacts reference beat indices, not
//! times, so a hand-corrected beat grid re-times sections for free).

use audan_core::{BeatGrid, BeatIndex, Chroma};
use audan_dsp::chroma_ssm;

use crate::novelty::novelty_curve;

/// Knobs for `segment_structure`. Frame-domain fields (`kernel_half_size`,
/// `min_boundary_distance_frames`) are counted in *feature* frames, i.e.
/// units of the `Chroma`'s own `FrameGrid` hop -- not beats or seconds -- so
/// their effective duration depends on the caller's chroma hop size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SegmentParams {
    /// Half-width of the checkerboard kernel, in feature frames. The default
    /// (16) pairs with `audan_dsp::KeyChromaParams`'s default 4096-sample hop
    /// at the 22050 Hz analysis rate (~5.4 frames/s), giving a ~6 s kernel --
    /// a reasonable span to distinguish one section from its neighbours
    /// without being swamped by within-section variation.
    pub kernel_half_size: usize,
    /// Peak-acceptance threshold, expressed as `mean + peak_threshold_k *
    /// stddev` of the novelty curve. Adaptive rather than a fixed absolute
    /// value because novelty magnitude scales with how similar/dissimilar
    /// the underlying features are track to track.
    pub peak_threshold_k: f32,
    /// Minimum spacing between accepted boundaries, in feature frames.
    /// Prevents two novelty samples one frame apart (a noisy, nearly-tied
    /// local maximum) from producing two boundaries a beat or two apart.
    pub min_boundary_distance_frames: usize,
}

impl Default for SegmentParams {
    fn default() -> Self {
        SegmentParams {
            kernel_half_size: 16,
            peak_threshold_k: 1.0,
            min_boundary_distance_frames: 8,
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

fn pick_peaks(novelty: &[f32], threshold_k: f32, min_distance: usize) -> Vec<usize> {
    let n = novelty.len();
    if n < 3 {
        return Vec::new();
    }

    let mean = novelty.iter().sum::<f32>() / n as f32;
    let variance = novelty.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let threshold = mean + threshold_k * variance.sqrt();

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

/// Segments `chroma` into unlabeled structural sections, beat-indexed
/// against `beats`. See RISK-8: `SectionEvent::label` is always `None`.
pub fn segment_structure(
    chroma: &Chroma,
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

    let ssm = chroma_ssm(chroma);
    let novelty = novelty_curve(&ssm, params.kernel_half_size);
    let peak_frames = pick_peaks(
        &novelty,
        params.peak_threshold_k,
        params.min_boundary_distance_frames,
    );

    let mut boundaries: Vec<BeatIndex> = peak_frames
        .into_iter()
        .map(|frame| {
            let time = chroma.grid.time_of(frame).as_seconds();
            nearest_beat_index(&beats.beats, time)
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
            downbeats: vec![0],
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
    fn recovers_roughly_three_sections_with_boundaries_near_true_transitions() {
        let frames_per_block = 40;
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
            result.sections.len() <= 4,
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
