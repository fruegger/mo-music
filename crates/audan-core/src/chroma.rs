//! The chroma / pitch-class profile domain type. `audan-dsp` computes these;
//! `audan-core` only defines the shape so it can be a stable cache and
//! interchange type without pulling DSP dependencies into `audan-core` (S5.1).

use serde::{Deserialize, Serialize};

use crate::frame::FrameGrid;

pub const PITCH_CLASSES: usize = 12;

/// A time series of 12-dimensional, octave-collapsed pitch-class energy
/// vectors, plus the [`FrameGrid`] that gives each frame a time.
///
/// Stored flat (`frame * PITCH_CLASSES + pitch_class`) rather than as nested
/// `Vec<Vec<f32>>` so it serializes compactly and slices cheaply.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Chroma {
    pub grid: FrameGrid,
    n_frames: usize,
    data: Vec<f32>,
}

impl Chroma {
    /// Builds a `Chroma` from frame-major pitch-class vectors. Panics if any
    /// row's length differs from [`PITCH_CLASSES`] -- a stage-internal
    /// invariant violation, not a runtime condition callers should handle.
    pub fn from_frames(grid: FrameGrid, frames: Vec<[f32; PITCH_CLASSES]>) -> Self {
        let n_frames = frames.len();
        let mut data = Vec::with_capacity(n_frames * PITCH_CLASSES);
        for row in frames {
            data.extend_from_slice(&row);
        }
        Chroma {
            grid,
            n_frames,
            data,
        }
    }

    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    pub fn frame(&self, k: usize) -> &[f32] {
        let start = k * PITCH_CLASSES;
        &self.data[start..start + PITCH_CLASSES]
    }

    pub fn frames(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(PITCH_CLASSES)
    }

    pub fn as_flat_slice(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::PadMode;

    #[test]
    fn indexes_frames_correctly() {
        let grid = FrameGrid::new(22050, 512, 2048, PadMode::Reflect);
        let mut a = [0.0f32; PITCH_CLASSES];
        a[0] = 1.0;
        let mut b = [0.0f32; PITCH_CLASSES];
        b[9] = 1.0; // A
        let chroma = Chroma::from_frames(grid, vec![a, b]);
        assert_eq!(chroma.n_frames(), 2);
        assert_eq!(chroma.frame(1)[9], 1.0);
    }
}
