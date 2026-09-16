use audan_core::{AudanError, Result};

/// MIREX beat-evaluation convention: one timestamp per line, in seconds.
pub fn write_mirex_times(times: &[f64]) -> String {
    let mut out = String::new();
    for t in times {
        out.push_str(&format!("{t:.6}\n"));
    }
    out
}

pub fn read_mirex_times(input: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for (lineno, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let t: f64 = line.parse().map_err(|_| {
            AudanError::InvalidInput(format!("line {}: invalid timestamp {line:?}", lineno + 1))
        })?;
        out.push(t);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let times = vec![0.512, 0.973, 1.441, 1.905];
        let text = write_mirex_times(&times);
        assert_eq!(text, "0.512000\n0.973000\n1.441000\n1.905000\n");
        let back = read_mirex_times(&text).unwrap();
        assert_eq!(back, times);
    }

    #[test]
    fn blank_lines_are_skipped() {
        let back = read_mirex_times("0.1\n\n0.2\n").unwrap();
        assert_eq!(back, vec![0.1, 0.2]);
    }

    #[test]
    fn invalid_line_is_an_error() {
        assert!(read_mirex_times("not-a-number\n").is_err());
    }
}
