//! The frame-time convention (architecture doc S8.2, ADR-5).
//!
//! Two decisions are routinely conflated elsewhere in the MIR ecosystem and the
//! conflation is the bug: whether the signal is padded at the edges, and whether
//! a frame's timestamp denotes its window's start or its centre. `audan` treats
//! padding as a parameter and fixes the timestamp reference for good: every
//! [`FrameTime`] in the system is the centre of its analysis window, in
//! original-signal seconds. There is no public way to build a `FrameTime` from a
//! raw `f64` outside this module, and no route from a frame index to a time
//! except [`FrameGrid::time_of`], which is the only place that knows the padding
//! mode. A stage physically cannot apply the wrong formula.

use serde::{Deserialize, Serialize};

/// How the signal is padded at its edges before framing.
///
/// This genuinely affects edge behaviour, frame count, and onset detection near
/// t=0, so unlike the timestamp reference it *is* a legitimate per-invocation
/// parameter (exposed as `--pad`) and is folded into `params_hash`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PadMode {
    /// No padding. Frame `k` spans `[k*hop, k*hop+win)`; frame 0 is *not*
    /// centred on sample 0.
    None,
    /// Zero-padded so frame 0 is centred on sample 0.
    Zero,
    /// Reflect-padded so frame 0 is centred on sample 0. The default.
    Reflect,
}

impl Default for PadMode {
    fn default() -> Self {
        PadMode::Reflect
    }
}

/// Seconds at the CENTRE of an analysis window, in original-signal coordinates
/// (i.e. with any padding already accounted for). There is no other time
/// convention in this codebase (ADR-5).
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FrameTime(f64);

impl FrameTime {
    /// Escape hatch for stages that already hold a value known -- by
    /// construction, not convention -- to be a window-centre time (e.g. a time
    /// read back from a validated foreign `frames` block, or zero). Prefer
    /// [`FrameGrid::time_of`] wherever a frame index is available.
    pub fn from_seconds_centered(seconds: f64) -> Self {
        FrameTime(seconds)
    }

    pub fn as_seconds(self) -> f64 {
        self.0
    }
}

/// The geometry of a framing operation: sample rate, hop, window length, and
/// padding mode. The only way to turn a frame index into a [`FrameTime`].
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct FrameGrid {
    pub sample_rate: u32,
    pub hop: usize,
    pub win: usize,
    pub pad: PadMode,
}

impl FrameGrid {
    pub fn new(sample_rate: u32, hop: usize, win: usize, pad: PadMode) -> Self {
        Self {
            sample_rate,
            hop,
            win,
            pad,
        }
    }

    /// The number of frames a signal of `num_samples` produces under this grid.
    pub fn frame_count(&self, num_samples: usize) -> usize {
        match self.pad {
            PadMode::None => {
                if num_samples < self.win {
                    0
                } else {
                    (num_samples - self.win) / self.hop + 1
                }
            }
            // Padded: one frame is centred on every hop-multiple sample,
            // including the edges.
            _ => num_samples.div_ceil(self.hop),
        }
    }

    /// The window-centre time of frame `k`, in original-signal seconds. This is
    /// the *only* route from a frame index to a time, and the only place the
    /// start-vs-centre formula is allowed to be written (S8.2).
    pub fn time_of(&self, k: usize) -> FrameTime {
        match self.pad {
            // Unpadded: frame k spans [k*hop, k*hop+win), so its centre is
            // offset by half the window from the frame's nominal start.
            PadMode::None => {
                FrameTime((k * self.hop + self.win / 2) as f64 / self.sample_rate as f64)
            }
            // Padded: frame 0 is centred on sample 0 by construction.
            _ => FrameTime((k * self.hop) as f64 / self.sample_rate as f64),
        }
    }
}

/// The timestamp reference convention. Always `window_center` in this system;
/// carried in output only so foreign data can be validated and rejected loudly
/// rather than misaligning silently (S8.2).
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeRef {
    WindowCenter,
}

impl Default for TimeRef {
    fn default() -> Self {
        TimeRef::WindowCenter
    }
}

/// The `frames` block stamped onto every output that carries frame-referenced
/// data, so a foreign file can be validated instead of silently misaligning.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct FramesMeta {
    pub sr: u32,
    pub hop: usize,
    pub win: usize,
    pub pad: PadMode,
    pub t_ref: TimeRef,
}

impl From<FrameGrid> for FramesMeta {
    fn from(g: FrameGrid) -> Self {
        FramesMeta {
            sr: g.sample_rate,
            hop: g.hop,
            win: g.win,
            pad: g.pad,
            t_ref: TimeRef::WindowCenter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpadded_frame_zero_is_offset_by_half_window() {
        let grid = FrameGrid::new(22050, 512, 2048, PadMode::None);
        let t = grid.time_of(0);
        assert!((t.as_seconds() - (1024.0 / 22050.0)).abs() < 1e-12);
    }

    #[test]
    fn padded_frame_zero_is_centred_on_sample_zero() {
        let grid = FrameGrid::new(22050, 512, 2048, PadMode::Reflect);
        let t = grid.time_of(0);
        assert_eq!(t.as_seconds(), 0.0);
    }

    #[test]
    fn hop_advances_time_linearly_regardless_of_padding() {
        for pad in [PadMode::None, PadMode::Zero, PadMode::Reflect] {
            let grid = FrameGrid::new(22050, 512, 2048, pad);
            let t0 = grid.time_of(3).as_seconds();
            let t1 = grid.time_of(4).as_seconds();
            assert!((t1 - t0 - 512.0 / 22050.0).abs() < 1e-12);
        }
    }
}
