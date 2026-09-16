use audan_core::{AudanError, Result};
use sha2::{Digest, Sha256};

pub fn verify_checksum(bytes: &[u8], expected_sha256_hex: &str) -> Result<()> {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let actual_hex = hex_encode(&hasher.finalize());

    if actual_hex.eq_ignore_ascii_case(expected_sha256_hex) {
        Ok(())
    } else {
        Err(AudanError::Model(format!(
            "checksum mismatch: expected {expected_sha256_hex}, got {actual_hex}"
        )))
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex_encode(&hasher.finalize())
    }

    #[test]
    fn accepts_a_matching_hash() {
        let data = b"audan-model test fixture bytes";
        let expected = sha256_hex(data);
        assert!(verify_checksum(data, &expected).is_ok());
    }

    #[test]
    fn accepts_a_matching_hash_regardless_of_case() {
        let data = b"audan-model test fixture bytes";
        let expected = sha256_hex(data).to_uppercase();
        assert!(verify_checksum(data, &expected).is_ok());
    }

    #[test]
    fn rejects_a_mismatched_hash() {
        let data = b"audan-model test fixture bytes";
        let wrong = sha256_hex(b"different bytes entirely");
        assert!(verify_checksum(data, &wrong).is_err());
    }
}
