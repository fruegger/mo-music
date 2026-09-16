//! `audan-model::LicenseGate` implementations (RV4 step 3). `audan-model`
//! deliberately doesn't know whether its caller is an interactive TTY
//! confirmation or a scripted `--accept-model-license` check -- that policy
//! decision lives here, in the only crate allowed to make UI decisions.

use std::io::{IsTerminal, Write};

use audan_model::{LicenseGate, ModelEntry};

/// Prompts on stderr and reads a `y/N` line from stdin. RV4 step 3 is
/// explicit that presenting the license (name, size, URL, `license_id`, and
/// `license_note` verbatim when present) is not a formality, so all of it is
/// printed before the prompt, not summarised away.
pub struct InteractiveGate;

impl InteractiveGate {
    /// The actual decision logic, parameterized on whether stdin is a TTY
    /// rather than reading `std::io::stdin().is_terminal()` itself, so it can
    /// be exercised deterministically in tests without touching the real
    /// process stdin (which may or may not be a TTY depending on how the
    /// test binary itself was launched).
    fn confirm_impl(&self, entry: &ModelEntry, stdin_is_tty: bool) -> bool {
        if !stdin_is_tty {
            eprintln!(
                "model '{}' {} requires license acceptance but stdin is not a TTY; \
                 pass --accept-model-license to accept non-interactively",
                entry.name, entry.version
            );
            return false;
        }

        eprintln!("Model:   {} {}", entry.name, entry.version);
        eprintln!("Size:    {} bytes", entry.size_bytes);
        eprintln!("Source:  {}", entry.url);
        eprintln!("License: {}", entry.license_id);
        if let Some(url) = &entry.license_text_url {
            eprintln!("License text: {url}");
        }
        if let Some(note) = &entry.license_note {
            eprintln!();
            eprintln!("{note}");
            eprintln!();
        }
        eprint!("Accept this license and download? [y/N] ");
        let _ = std::io::stderr().flush();

        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            return false;
        }
        matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
    }
}

impl LicenseGate for InteractiveGate {
    fn confirm(&self, entry: &ModelEntry) -> bool {
        self.confirm_impl(entry, std::io::stdin().is_terminal())
    }
}

/// Non-interactive: `--accept-model-license` accepts every model outright
/// (there is only ever one model to confirm per invocation), otherwise
/// refuses without touching stdin at all.
pub struct FlagGate {
    pub accepted: bool,
}

impl LicenseGate for FlagGate {
    fn confirm(&self, entry: &ModelEntry) -> bool {
        if self.accepted {
            eprintln!(
                "accepted license for model '{}' {} via --accept-model-license",
                entry.name, entry.version
            );
        } else {
            eprintln!(
                "model '{}' {} requires license acceptance; pass --accept-model-license \
                 for non-interactive use, or run interactively to confirm",
                entry.name, entry.version
            );
        }
        self.accepted
    }
}

/// Picks the right gate for the current invocation: `--accept-model-license`
/// short-circuits to a non-interactive accept; otherwise fall back to an
/// interactive TTY prompt, which itself declines cleanly (not a hang) when
/// stdin isn't a TTY.
pub fn select_gate(accept_model_license: bool) -> Box<dyn LicenseGate> {
    if accept_model_license {
        Box::new(FlagGate { accepted: true })
    } else {
        Box::new(InteractiveGate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry() -> ModelEntry {
        ModelEntry {
            name: "htdemucs".into(),
            version: "4.0".into(),
            url: "https://example.invalid/model.onnx".into(),
            sha256: "0".repeat(64),
            size_bytes: 123,
            license_id: "unclear".into(),
            license_text_url: None,
            license_note: Some("ambiguous on purpose".into()),
        }
    }

    #[test]
    fn interactive_gate_declines_cleanly_when_stdin_is_not_a_tty() {
        // The scenario RV4 step 3 calls out for non-interactive contexts:
        // must return false promptly, never block on a `read_line` that will
        // never receive input.
        let gate = InteractiveGate;
        assert!(!gate.confirm_impl(&sample_entry(), false));
    }

    #[test]
    fn flag_gate_accepts_without_touching_stdin() {
        let gate = FlagGate { accepted: true };
        assert!(gate.confirm(&sample_entry()));
    }

    #[test]
    fn flag_gate_declines_when_not_passed() {
        let gate = FlagGate { accepted: false };
        assert!(!gate.confirm(&sample_entry()));
    }

    #[test]
    fn select_gate_chooses_flag_gate_when_accept_model_license_is_set() {
        let gate = select_gate(true);
        assert!(gate.confirm(&sample_entry()));
    }
}
