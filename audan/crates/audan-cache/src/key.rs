//! [`CacheKey`] and [`KeyDeriver`] (S8.1). This is the only place key
//! composition logic exists; every stage goes through [`KeyDeriver::derive`]
//! so that the chaining property (parent hash inside child key) always holds.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A content-addressed cache key: a blake3 digest of a chain of parent
/// hashes, a stage identity, and canonicalised stage parameters. See S8.1.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct CacheKey(blake3::Hash);

impl CacheKey {
    pub fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }

    pub fn to_hex(&self) -> String {
        self.0.to_hex().to_string()
    }

    pub fn from_hex(s: &str) -> std::result::Result<Self, CacheKeyParseError> {
        blake3::Hash::from_hex(s)
            .map(CacheKey)
            .map_err(|e| CacheKeyParseError(e.to_string()))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("invalid cache key hex: {0}")]
pub struct CacheKeyParseError(String);

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CacheKey({})", self.to_hex())
    }
}

impl Serialize for CacheKey {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for CacheKey {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        CacheKey::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Domain separator baked into every derived key, so a future change to the
/// derivation scheme itself (not just a stage's `stage_version`) can be
/// rolled out as a new constant without ever colliding with keys from the
/// old scheme.
const DOMAIN: &[u8] = b"audan-cache-key-v1";

/// The single place key-composition logic exists (S5.2). Builds a
/// [`CacheKey`] from `(parent, stage_id, stage_version, params)`.
pub struct KeyDeriver;

impl KeyDeriver {
    /// Derive a cache key.
    ///
    /// `params` must implement `Serialize` (not `std::hash::Hash` — see the
    /// module-level note below for why that distinction matters). Fields are
    /// canonicalised by routing through `serde_json::Value` and then
    /// re-encoding it with object keys explicitly sorted (`canonical_bytes`
    /// below) — this does NOT rely on `serde_json::Value`'s own map
    /// representation defaulting to a `BTreeMap`, because that default is a
    /// crate-level Cargo feature (`preserve_order`) unified across the whole
    /// workspace's dependency graph: if any other crate anywhere in the build
    /// enables it for its own reasons, `Value`'s map silently becomes
    /// insertion-ordered for every crate that uses it, this one included. So
    /// a `params` struct that embeds a `std::collections::HashMap` field
    /// could otherwise leak that map's randomised per-process iteration order
    /// into the key, depending on what else happens to be in the dependency
    /// tree that week. Sorting explicitly makes canonicalisation independent
    /// of that.
    pub fn derive(
        parent: Option<&CacheKey>,
        stage_id: &str,
        stage_version: u32,
        params: &impl Serialize,
    ) -> CacheKey {
        let canonical = canonical_bytes(params)
            .expect("cache key params must be representable as serde_json::Value");

        let mut hasher = blake3::Hasher::new();
        hasher.update(DOMAIN);
        match parent {
            Some(p) => {
                hasher.update(&[1u8]);
                hasher.update(p.as_bytes());
            }
            None => {
                hasher.update(&[0u8]);
            }
        }
        // Length-prefix the variable-length fields so that, e.g., stage_id
        // "ab" + params "c" cannot hash identically to stage_id "a" + params
        // "bc".
        hasher.update(&(stage_id.len() as u64).to_le_bytes());
        hasher.update(stage_id.as_bytes());
        hasher.update(&stage_version.to_le_bytes());
        hasher.update(&(canonical.len() as u64).to_le_bytes());
        hasher.update(&canonical);
        CacheKey(hasher.finalize())
    }
}

fn canonical_bytes(params: &impl Serialize) -> serde_json::Result<Vec<u8>> {
    let value = serde_json::to_value(params)?;
    let mut buf = Vec::new();
    write_canonical(&value, &mut buf);
    Ok(buf)
}

/// Serialises a `serde_json::Value` with object keys sorted and every
/// variable-length field length-prefixed, so structurally different values
/// (e.g. `{"ab":"c"}` vs `{"a":"bc"}`) can never collide.
fn write_canonical(value: &serde_json::Value, buf: &mut Vec<u8>) {
    use serde_json::Value;
    match value {
        Value::Null => buf.push(0),
        Value::Bool(b) => {
            buf.push(1);
            buf.push(*b as u8);
        }
        Value::Number(n) => {
            buf.push(2);
            let s = n.to_string();
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        Value::String(s) => {
            buf.push(3);
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        Value::Array(arr) => {
            buf.push(4);
            buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
            for v in arr {
                write_canonical(v, buf);
            }
        }
        Value::Object(map) => {
            buf.push(5);
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            buf.extend_from_slice(&(entries.len() as u64).to_le_bytes());
            for (k, v) in entries {
                buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                buf.extend_from_slice(k.as_bytes());
                write_canonical(v, buf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Serialize)]
    struct Params {
        a: u32,
        b: String,
        tags: HashMap<String, u32>,
    }

    fn sample_params(insertion_order: &[&str]) -> Params {
        // Values are derived from the key itself (not the insertion index),
        // so every permutation of `insertion_order` builds the exact same
        // logical map -- only HashMap's internal iteration order differs.
        let mut tags = HashMap::new();
        for k in insertion_order {
            tags.insert(k.to_string(), k.len() as u32);
        }
        Params {
            a: 7,
            b: "beats".to_string(),
            tags,
        }
    }

    #[test]
    fn identical_params_produce_identical_keys() {
        let k1 = KeyDeriver::derive(None, "beats", 1, &sample_params(&["x", "y", "z"]));
        let k2 = KeyDeriver::derive(None, "beats", 1, &sample_params(&["z", "y", "x"]));
        assert_eq!(
            k1, k2,
            "HashMap insertion order must not affect the derived key"
        );
    }

    #[test]
    fn stage_version_bump_changes_key() {
        let params = sample_params(&["x"]);
        let k1 = KeyDeriver::derive(None, "beats", 1, &params);
        let k2 = KeyDeriver::derive(None, "beats", 2, &params);
        assert_ne!(k1, k2);
    }

    #[test]
    fn hex_round_trip() {
        let k = KeyDeriver::derive(None, "beats", 1, &sample_params(&["x"]));
        let hex = k.to_hex();
        let parsed = CacheKey::from_hex(&hex).unwrap();
        assert_eq!(k, parsed);
    }

    #[test]
    fn serde_round_trip() {
        let k = KeyDeriver::derive(None, "beats", 1, &sample_params(&["x"]));
        let json = serde_json::to_string(&k).unwrap();
        let parsed: CacheKey = serde_json::from_str(&json).unwrap();
        assert_eq!(k, parsed);
    }
}
