//! Shared vocabulary for the cache layers from S5.2's table, so downstream
//! crates use one set of `stage_id` prefixes instead of inventing ad hoc
//! strings.

use std::fmt;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum CacheLayer {
    /// Decoded PCM at native rate and channel count.
    L0,
    /// Canonical analysis signal: mono f32, 22050 Hz (analysis) / 44100 Hz (stems).
    L1,
    /// Cheap features: onset envelope, CQT, chroma, tempogram.
    L2,
    /// Beat grid.
    L3,
    /// Key, chords, sections (beat-referenced).
    L4,
    /// Stems, raw model outputs.
    L5,
}

impl CacheLayer {
    pub const fn stage_prefix(self) -> &'static str {
        match self {
            CacheLayer::L0 => "l0",
            CacheLayer::L1 => "l1",
            CacheLayer::L2 => "l2",
            CacheLayer::L3 => "l3",
            CacheLayer::L4 => "l4",
            CacheLayer::L5 => "l5",
        }
    }

    /// Build a `stage_id` for [`KeyDeriver::derive`](crate::KeyDeriver::derive)
    /// by namespacing a stage name under this layer, e.g.
    /// `CacheLayer::L3.stage_id("beats-this")` -> `"l3:beats-this"`.
    pub fn stage_id(self, name: &str) -> String {
        format!("{}:{name}", self.stage_prefix())
    }
}

impl fmt::Display for CacheLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.stage_prefix())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_id_namespaces_by_layer() {
        assert_eq!(CacheLayer::L3.stage_id("beats-this"), "l3:beats-this");
        assert_eq!(CacheLayer::L2.stage_id("chroma"), "l2:chroma");
    }
}
