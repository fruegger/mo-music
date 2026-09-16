//! Self-similarity matrix: pairwise cosine similarity between all frame pairs
//! of a feature sequence. `O(n^2)`, which is the standard approach for this
//! (`audan-struct` consumes it for Foote novelty).

use audan_core::Chroma;

pub fn cosine_similarity_matrix(features: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = features.len();
    let norms: Vec<f32> = features
        .iter()
        .map(|f| f.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();

    let mut m = vec![vec![0f32; n]; n];
    for i in 0..n {
        m[i][i] = if norms[i] > 1e-12 { 1.0 } else { 0.0 };
        for j in (i + 1)..n {
            let dot: f32 = features[i]
                .iter()
                .zip(features[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            let denom = norms[i] * norms[j];
            let sim = if denom > 1e-12 { dot / denom } else { 0.0 };
            m[i][j] = sim;
            m[j][i] = sim;
        }
    }
    m
}

pub fn chroma_ssm(chroma: &Chroma) -> Vec<Vec<f32>> {
    let feats: Vec<Vec<f32>> = chroma.frames().map(|f| f.to_vec()).collect();
    cosine_similarity_matrix(&feats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_block_signal_is_block_diagonal() {
        let mut features = Vec::new();
        for _ in 0..10 {
            features.push(vec![1.0, 0.0, 0.0]);
        }
        for _ in 0..10 {
            features.push(vec![0.0, 1.0, 0.0]);
        }
        let m = cosine_similarity_matrix(&features);

        let within_a: f32 = (0..10)
            .flat_map(|i| (0..10).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| m[i][j])
            .sum::<f32>()
            / 90.0;
        let within_b: f32 = (10..20)
            .flat_map(|i| (10..20).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j)
            .map(|(i, j)| m[i][j])
            .sum::<f32>()
            / 90.0;
        let cross: f32 = (0..10)
            .flat_map(|i| (10..20).map(move |j| (i, j)))
            .map(|(i, j)| m[i][j])
            .sum::<f32>()
            / 100.0;

        assert!((within_a - 1.0).abs() < 1e-6);
        assert!((within_b - 1.0).abs() < 1e-6);
        assert!(cross.abs() < 1e-6);
        assert!(within_a > cross + 0.9);
        assert!(within_b > cross + 0.9);
    }

    #[test]
    fn diagonal_is_self_similar() {
        let features = vec![vec![1.0, 2.0, 3.0], vec![0.5, -1.0, 4.0]];
        let m = cosine_similarity_matrix(&features);
        assert!((m[0][0] - 1.0).abs() < 1e-6);
        assert!((m[1][1] - 1.0).abs() < 1e-6);
        assert!((m[0][1] - m[1][0]).abs() < 1e-6);
    }
}
