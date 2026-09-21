//! The pluggable backend contract (S5.1 `audan-stems` row, S8.6 "Backends").
//!
//! `audan-stems` ships no implementation of this trait by default -- see the
//! crate-level docs and RISK-1 in `audan-architecture-arc42.md`.

use audan_core::{Result, Signal};

/// One isolated track produced by a [`Separator`], e.g. `"vocals"`, `"drums"`,
/// `"bass"`, `"other"`. Names are backend-defined; a manifest written
/// alongside the audio (see [`crate::StemManifest`]) records which ones a
/// particular run produced.
pub struct StemTrack {
    pub name: String,
    pub signal: Signal,
}

/// The full set of tracks a [`Separator`] produced from one input signal.
pub struct Stems {
    pub tracks: Vec<StemTrack>,
}

/// A pluggable source-separation backend.
pub trait Separator {
    fn name(&self) -> &str;

    /// `signal` is expected at [`audan_core::STEMS_SAMPLE_RATE`]. Whether it
    /// must be mono or stereo is the implementor's own requirement --
    /// document it on the implementing type, since this trait does not
    /// constrain it.
    fn separate(&self, signal: &Signal) -> Result<Stems>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves the trait's shape is actually usable end to end, not just
    /// declared. Splits the input deterministically into two stems rather
    /// than doing any real separation -- there is no licensed model to call
    /// here (RISK-1).
    struct SignFlipSeparator;

    impl Separator for SignFlipSeparator {
        fn name(&self) -> &str {
            "sign-flip-fake"
        }

        fn separate(&self, signal: &Signal) -> Result<Stems> {
            let positive = Signal {
                sample_rate: signal.sample_rate,
                channels: signal.channels,
                samples: signal.samples.clone(),
            };
            let negated = Signal {
                sample_rate: signal.sample_rate,
                channels: signal.channels,
                samples: signal.samples.iter().map(|s| -s).collect(),
            };
            Ok(Stems {
                tracks: vec![
                    StemTrack {
                        name: "positive".to_string(),
                        signal: positive,
                    },
                    StemTrack {
                        name: "negated".to_string(),
                        signal: negated,
                    },
                ],
            })
        }
    }

    #[test]
    fn fake_separator_produces_expected_tracks() {
        let signal = Signal {
            sample_rate: audan_core::STEMS_SAMPLE_RATE,
            channels: 1,
            samples: vec![0.1, -0.2, 0.3],
        };

        let separator = SignFlipSeparator;
        assert_eq!(separator.name(), "sign-flip-fake");

        let stems = separator
            .separate(&signal)
            .expect("separation must succeed");
        assert_eq!(stems.tracks.len(), 2);

        let positive = &stems.tracks[0];
        assert_eq!(positive.name, "positive");
        assert_eq!(positive.signal.samples, vec![0.1, -0.2, 0.3]);

        let negated = &stems.tracks[1];
        assert_eq!(negated.name, "negated");
        assert_eq!(negated.signal.samples, vec![-0.1, 0.2, -0.3]);
        assert_eq!(negated.signal.sample_rate, signal.sample_rate);
    }
}
