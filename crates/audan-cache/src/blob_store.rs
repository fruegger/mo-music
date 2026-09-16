//! Content-addressed storage of arbitrary bytes under a [`CacheKey`] (S5.2).
//! Blobs are zstd-compressed on write and sharded by the first two hex chars
//! of the key so a large cache never puts tens of thousands of files in one
//! directory.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use audan_core::error::{AudanError, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::key::CacheKey;

pub struct BlobStore {
    blobs_dir: PathBuf,
}

impl BlobStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let blobs_dir = root.as_ref().join("blobs");
        fs::create_dir_all(&blobs_dir)?;
        Ok(Self { blobs_dir })
    }

    fn shard_dir(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.blobs_dir.join(&hex[..2])
    }

    fn path_for(&self, key: &CacheKey) -> PathBuf {
        self.shard_dir(key).join(key.to_hex())
    }

    pub fn contains(&self, key: &CacheKey) -> bool {
        self.path_for(key).is_file()
    }

    /// Store raw bytes under `key`, zstd-compressed. Returns the compressed
    /// size in bytes (what actually lands on disk; what `Index`/`Evictor`
    /// should budget against).
    ///
    /// Writes go to a fresh temp file in the same shard directory and are
    /// then renamed over the final path. `rename` on both POSIX and NTFS
    /// replaces the destination atomically, so a reader never observes a
    /// half-written file: it either sees the previous blob (if any) or the
    /// complete new one, never a partial one. A process killed mid-write
    /// leaves an orphaned temp file, never a corrupt blob.
    pub fn put_bytes(&self, key: &CacheKey, bytes: &[u8]) -> Result<u64> {
        let compressed = zstd::encode_all(bytes, zstd::DEFAULT_COMPRESSION_LEVEL)?;

        let shard = self.shard_dir(key);
        fs::create_dir_all(&shard)?;
        let mut tmp = tempfile::NamedTempFile::new_in(&shard)?;
        tmp.write_all(&compressed)?;
        tmp.flush()?;
        tmp.persist(self.path_for(key))
            .map_err(|e| AudanError::Cache(format!("atomic blob rename failed: {e}")))?;

        Ok(compressed.len() as u64)
    }

    pub fn get_bytes(&self, key: &CacheKey) -> Result<Option<Vec<u8>>> {
        let path = self.path_for(key);
        let compressed = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let bytes = zstd::decode_all(&compressed[..])?;
        Ok(Some(bytes))
    }

    pub fn remove(&self, key: &CacheKey) -> Result<()> {
        match fs::remove_file(self.path_for(key)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// Serialise `value` with `serde_json` and store it. Fine for scaffold
    /// purposes; a production build might prefer `bincode` for large `f32`
    /// arrays (L1/L2 signal and feature blobs), where JSON's per-number
    /// overhead is significant even after zstd.
    pub fn put<T: Serialize>(&self, key: &CacheKey, value: &T) -> Result<u64> {
        let json = serde_json::to_vec(value)
            .map_err(|e| AudanError::Cache(format!("blob serialise failed: {e}")))?;
        self.put_bytes(key, &json)
    }

    pub fn get<T: DeserializeOwned>(&self, key: &CacheKey) -> Result<Option<T>> {
        match self.get_bytes(key)? {
            Some(bytes) => {
                let value = serde_json::from_slice(&bytes)
                    .map_err(|e| AudanError::Cache(format!("blob deserialise failed: {e}")))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_raw_bytes_through_compression() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();
        let key = crate::KeyDeriver::derive(None, "raw", 1, &"round-trip");

        let payload: Vec<u8> = (0u32..5000).map(|i| (i % 256) as u8).collect();
        store.put_bytes(&key, &payload).unwrap();

        let read_back = store.get_bytes(&key).unwrap().unwrap();
        assert_eq!(read_back, payload);
    }

    #[test]
    fn round_trips_typed_values() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();
        let key = crate::KeyDeriver::derive(None, "typed", 1, &"round-trip");

        #[derive(Serialize, serde::Deserialize, PartialEq, Debug)]
        struct Grid {
            values: Vec<f32>,
        }
        let grid = Grid {
            values: vec![0.1, 0.2, 0.3],
        };
        store.put(&key, &grid).unwrap();

        let read_back: Grid = store.get(&key).unwrap().unwrap();
        assert_eq!(read_back, grid);
    }

    #[test]
    fn missing_key_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();
        let key = crate::KeyDeriver::derive(None, "missing", 1, &"nope");
        assert!(store.get_bytes(&key).unwrap().is_none());
    }

    #[test]
    fn interrupted_write_never_corrupts_existing_blob() {
        let dir = tempfile::tempdir().unwrap();
        let store = BlobStore::open(dir.path()).unwrap();
        let key = crate::KeyDeriver::derive(None, "atomic", 1, &"target");

        store.put_bytes(&key, b"original-good-blob").unwrap();

        // Simulate a writer that started (created a temp file in the shard
        // directory) but crashed before the rename that would publish it:
        // drop a bogus file into the shard directory without ever renaming
        // it over the real path.
        let shard = store.shard_dir(&key);
        let fake_partial = shard.join("some-other-writer.tmp");
        fs::write(&fake_partial, b"garbage-from-a-crashed-writer").unwrap();

        // The real blob is untouched: readers never see the partial file,
        // because the path they read from is only ever reached by rename.
        let read_back = store.get_bytes(&key).unwrap().unwrap();
        assert_eq!(read_back, b"original-good-blob");

        // A subsequent real write still lands atomically and fully replaces
        // the old content.
        store.put_bytes(&key, b"second-good-blob").unwrap();
        let read_back = store.get_bytes(&key).unwrap().unwrap();
        assert_eq!(read_back, b"second-good-blob");
    }
}
