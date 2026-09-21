//! Chord templates: binary pitch-class masks for each root/quality, rotated
//! over all 12 roots.
//!
//! Scope cut (documented, not accidental): this crate implements majmin
//! (major and minor triads) only, not the extended 7th/inversion vocabulary
//! the architecture doc's example labels (`G:7/3`) hint at. ADR-12 already
//! accepts a lower accuracy ceiling than a neural model in exchange for
//! license cleanliness (~75-80% majmin per the ADR); adding 7ths, inversions,
//! and slash-chord bass detection multiplies the template count and the
//! template-confusion surface for a vocabulary that reference chord-eval
//! tooling (e.g. `mir_eval`) usually scores separately from majmin anyway.
//! Majmin is therefore a deliberate, sufficient scope for this pass.

use audan_core::PITCH_CLASSES;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ChordQuality {
    Maj,
    Min,
}

impl ChordQuality {
    pub fn intervals(self) -> [u8; 3] {
        match self {
            ChordQuality::Maj => [0, 4, 7],
            ChordQuality::Min => [0, 3, 7],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Template {
    pub root: u8,
    pub quality: ChordQuality,
    pub vector: [f32; PITCH_CLASSES],
}

fn triad_vector(root: u8, intervals: [u8; 3]) -> [f32; PITCH_CLASSES] {
    let mut v = [0f32; PITCH_CLASSES];
    for iv in intervals {
        v[(root as usize + iv as usize) % PITCH_CLASSES] = 1.0;
    }
    v
}

/// All 24 majmin templates (12 roots x {major, minor}). Order is stable and
/// is relied on elsewhere in this crate to map a template index back to a
/// `(root, quality)` pair.
pub fn all_templates() -> Vec<Template> {
    let mut templates = Vec::with_capacity(24);
    for root in 0..12u8 {
        for quality in [ChordQuality::Maj, ChordQuality::Min] {
            templates.push(Template {
                root,
                quality,
                vector: triad_vector(root, quality.intervals()),
            });
        }
    }
    templates
}

/// Cosine similarity, guarded against a zero-norm operand (returns 0.0
/// rather than NaN -- a near-silent aggregated chroma vector should compete
/// on equal footing with every template, not poison the comparison).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 1e-9 || nb <= 1e-9 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_major_template_beats_rotated_and_wrong_quality_templates() {
        let templates = all_templates();
        let c_major_shaped = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let c_major = templates
            .iter()
            .find(|t| t.root == 0 && t.quality == ChordQuality::Maj)
            .unwrap();
        let g_major = templates
            .iter()
            .find(|t| t.root == 7 && t.quality == ChordQuality::Maj)
            .unwrap();
        let c_minor = templates
            .iter()
            .find(|t| t.root == 0 && t.quality == ChordQuality::Min)
            .unwrap();

        let sim_c_major = cosine_similarity(&c_major_shaped, &c_major.vector);
        let sim_g_major = cosine_similarity(&c_major_shaped, &g_major.vector);
        let sim_c_minor = cosine_similarity(&c_major_shaped, &c_minor.vector);

        assert!((sim_c_major - 1.0).abs() < 1e-6);
        assert!(sim_c_major > sim_g_major);
        assert!(sim_c_major > sim_c_minor);
    }

    #[test]
    fn zero_vector_similarity_is_zero_not_nan() {
        let templates = all_templates();
        let sim = cosine_similarity(&[0.0; PITCH_CLASSES], &templates[0].vector);
        assert_eq!(sim, 0.0);
    }
}
