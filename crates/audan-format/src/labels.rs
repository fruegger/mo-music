use audan_core::{AudanError, Result};
use serde::{Deserialize, Serialize};

/// One labelled time interval: `start`/`end` in seconds, `label` free text.
///
/// The shared shape behind both `.lab` (Isophonics/Harte convention: chords like
/// `C:maj`, `G:7/3`, or section names) and Audacity label tracks -- any upstream
/// stage that produces interval-shaped results (chords, sections, anything else)
/// goes through the same reader/writer pair regardless of which stage produced it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelInterval {
    pub start: f64,
    pub end: f64,
    pub label: String,
}

pub fn write_lab(intervals: &[LabelInterval]) -> String {
    write_delimited(intervals, ' ')
}

pub fn read_lab(input: &str) -> Result<Vec<LabelInterval>> {
    parse_delimited(input)
}

pub fn write_audacity_labels(intervals: &[LabelInterval]) -> String {
    write_delimited(intervals, '\t')
}

pub fn read_audacity_labels(input: &str) -> Result<Vec<LabelInterval>> {
    parse_delimited(input)
}

fn write_delimited(intervals: &[LabelInterval], sep: char) -> String {
    let mut out = String::new();
    for iv in intervals {
        out.push_str(&format!(
            "{:.6}{sep}{:.6}{sep}{}\n",
            iv.start, iv.end, iv.label
        ));
    }
    out
}

/// Both `.lab` and Audacity label tracks parse the same way: the first two
/// whitespace-delimited fields are the start/end times, and everything after is
/// the label (rejoined with single spaces, since a label may itself contain
/// spaces -- this is what makes the tab-separated Audacity format work).
fn parse_delimited(input: &str) -> Result<Vec<LabelInterval>> {
    let mut out = Vec::new();
    for (lineno, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let start_str = fields.next().ok_or_else(|| {
            AudanError::InvalidInput(format!("line {}: missing start time", lineno + 1))
        })?;
        let end_str = fields.next().ok_or_else(|| {
            AudanError::InvalidInput(format!("line {}: missing end time", lineno + 1))
        })?;
        let label = fields.collect::<Vec<_>>().join(" ");
        if label.is_empty() {
            return Err(AudanError::InvalidInput(format!(
                "line {}: missing label",
                lineno + 1
            )));
        }
        let start: f64 = start_str.parse().map_err(|_| {
            AudanError::InvalidInput(format!(
                "line {}: invalid start time {start_str:?}",
                lineno + 1
            ))
        })?;
        let end: f64 = end_str.parse().map_err(|_| {
            AudanError::InvalidInput(format!("line {}: invalid end time {end_str:?}", lineno + 1))
        })?;
        out.push(LabelInterval { start, end, label });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<LabelInterval> {
        vec![
            LabelInterval {
                start: 0.0,
                end: 0.512,
                label: "N".into(),
            },
            LabelInterval {
                start: 0.512,
                end: 0.973,
                label: "C:maj".into(),
            },
            LabelInterval {
                start: 0.973,
                end: 1.441,
                label: "G:7/3".into(),
            },
        ]
    }

    #[test]
    fn lab_round_trips() {
        let intervals = sample();
        let text = write_lab(&intervals);
        assert_eq!(text.lines().next().unwrap(), "0.000000 0.512000 N");
        let back = read_lab(&text).unwrap();
        assert_eq!(back, intervals);
    }

    #[test]
    fn audacity_round_trips() {
        let intervals = sample();
        let text = write_audacity_labels(&intervals);
        assert_eq!(text.lines().next().unwrap(), "0.000000\t0.512000\tN");
        let back = read_audacity_labels(&text).unwrap();
        assert_eq!(back, intervals);
    }

    #[test]
    fn label_with_internal_spaces_round_trips() {
        let intervals = vec![LabelInterval {
            start: 1.0,
            end: 2.0,
            label: "verse 1".into(),
        }];
        let text = write_audacity_labels(&intervals);
        let back = read_audacity_labels(&text).unwrap();
        assert_eq!(back, intervals);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let text = "0.0 1.0 A\n\n1.0 2.0 B\n";
        let back = read_lab(text).unwrap();
        assert_eq!(back.len(), 2);
    }

    #[test]
    fn missing_label_is_an_error() {
        let text = "0.0 1.0\n";
        assert!(read_lab(text).is_err());
    }
}
