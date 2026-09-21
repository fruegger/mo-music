//! Key estimation (architecture doc S5.1): aggregate a key-chroma sequence
//! into one pitch-class profile, correlate it against the 24 Krumhansl-
//! Kessler major/minor key profiles, and report the top 3 ranked candidates
//! (ADR-11) converted to conventional name, Camelot, and Open Key notation
//! (S3.2).

mod camelot;
mod estimate;
mod key_estimate;
mod mode;
mod pitch_class;
mod profiles;

pub use estimate::{estimate_key, estimate_key_from_signal};
pub use key_estimate::KeyEstimate;
pub use mode::Mode;
pub use pitch_class::PitchClass;
pub use profiles::{MAJOR_PROFILE, MINOR_PROFILE};
