use serde::{Deserialize, Serialize};

use crate::camelot::{to_camelot, to_open_key};
use crate::mode::Mode;
use crate::pitch_class::PitchClass;

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct KeyEstimate {
    pub tonic: PitchClass,
    pub mode: Mode,
    pub name: String,
}

impl KeyEstimate {
    pub fn new(tonic: PitchClass, mode: Mode) -> Self {
        let name = format!("{} {}", tonic.name(), mode.name());
        KeyEstimate { tonic, mode, name }
    }

    pub fn camelot(&self) -> String {
        to_camelot(self.tonic, self.mode)
    }

    pub fn open_key(&self) -> String {
        to_open_key(self.tonic, self.mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_matches_expected_format() {
        let k = KeyEstimate::new(PitchClass::C, Mode::Major);
        assert_eq!(k.name, "C major");
        assert_eq!(k.camelot(), "8B");
        assert_eq!(k.open_key(), "1d");
    }
}
