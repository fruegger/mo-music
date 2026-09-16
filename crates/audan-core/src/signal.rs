//! Decoded-audio domain types. Shared by `audan-io` (produces), `audan-cache`
//! (stores as L0/L1 blobs), and `audan-dsp`/`audan-stems` (consume), so the
//! shape lives here rather than in any one of them.

use serde::{Deserialize, Serialize};

/// Decoded PCM at native rate and channel count, interleaved. The L0 cache
/// layer's content (S5.2).
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Signal {
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved samples: `[c0s0, c1s0, c0s1, c1s1, ...]`.
    pub samples: Vec<f32>,
}

impl Signal {
    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }

    pub fn duration_seconds(&self) -> f64 {
        self.num_frames() as f64 / self.sample_rate as f64
    }
}

/// The canonical analysis signal: mono `f32` at a fixed rate (22050 Hz for
/// analysis, 44100 Hz for stems -- S5.2, L1). The input every DSP stage and
/// every stage above L1 actually operates on.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct MonoSignal {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

impl MonoSignal {
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// The canonical analysis rate for beat/key/chord/structure work (S5.2, L1).
pub const ANALYSIS_SAMPLE_RATE: u32 = 22_050;

/// The canonical rate for stem separation and reconstruction (S5.2, L1).
pub const STEMS_SAMPLE_RATE: u32 = 44_100;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_frames_divides_by_channel_count() {
        let s = Signal {
            sample_rate: 44100,
            channels: 2,
            samples: vec![0.0; 8],
        };
        assert_eq!(s.num_frames(), 4);
    }
}
