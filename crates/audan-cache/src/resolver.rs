//! The single entry point stages use (S5.2): ask for a key's value, compute
//! it only on a miss, and never let two workers compute the same key twice
//! (S6.3 / RV3).

use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use audan_core::error::{AudanError, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::blob_store::BlobStore;
use crate::evictor::Evictor;
use crate::index::Index;
use crate::key::CacheKey;

/// How long a per-key lock file may persist before a waiting resolver
/// assumes the process that created it died without cleaning up, and
/// reclaims it. Generous relative to any single analysis stage.
const STALE_LOCK_TIMEOUT: Duration = Duration::from_secs(600);

pub struct Resolver {
    root: PathBuf,
    blobs: BlobStore,
    index: Index,
}

impl Resolver {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("locks"))?;
        let blobs = BlobStore::open(&root)?;
        let index = Index::open(root.join("index.redb"))?;
        Ok(Self { root, blobs, index })
    }

    pub fn blobs(&self) -> &BlobStore {
        &self.blobs
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    pub fn evictor(&self) -> Evictor<'_> {
        Evictor::new(&self.blobs, &self.index)
    }

    fn lock_path(&self, key: &CacheKey) -> PathBuf {
        self.root
            .join("locks")
            .join(format!("{}.lock", key.to_hex()))
    }

    /// Hit: return the cached value (and record the access). Miss: run
    /// `compute`, store its result, and return it. Concurrent callers
    /// (threads within one process, or separate `audan` processes racing
    /// over a shared cache directory, per RV3) resolving the *same* key
    /// serialise on a per-key lock file so `compute` runs exactly once; every
    /// other caller waits for the winner's write to land and then reads it
    /// back as a normal hit.
    pub fn resolve<T, F>(&self, key: CacheKey, compute: F) -> Result<T>
    where
        T: Serialize + DeserializeOwned,
        F: FnOnce() -> Result<T>,
    {
        loop {
            if let Some(value) = self.blobs.get::<T>(&key)? {
                self.index.touch(&key)?;
                return Ok(value);
            }

            let lock_path = self.lock_path(&key);
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_lock_file) => {
                    let _guard = LockGuard {
                        path: lock_path.clone(),
                    };

                    // Re-check: another worker may have finished and cleared
                    // its own lock in the window between our check above and
                    // our successful lock creation.
                    if let Some(value) = self.blobs.get::<T>(&key)? {
                        self.index.touch(&key)?;
                        return Ok(value);
                    }

                    let value = compute()?;
                    let size = self.blobs.put(&key, &value)?;
                    self.index.insert_new(&key, size)?;
                    return Ok(value);
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    self.wait_for_lock(&lock_path);
                    continue;
                }
                Err(e) => return Err(AudanError::Io(e)),
            }
        }
    }

    fn wait_for_lock(&self, lock_path: &Path) {
        let start = Instant::now();
        let mut backoff = Duration::from_millis(2);
        while lock_path.exists() {
            thread::sleep(backoff);
            backoff = (backoff * 2).min(Duration::from_millis(50));
            if start.elapsed() > STALE_LOCK_TIMEOUT {
                let _ = fs::remove_file(lock_path);
                return;
            }
        }
    }
}

struct LockGuard {
    path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miss_then_hit() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = Resolver::open(dir.path()).unwrap();
        let key = crate::KeyDeriver::derive(None, "s", 1, &"p");

        let calls = std::sync::atomic::AtomicUsize::new(0);
        let v: u32 = resolver
            .resolve(key, || {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(42)
            })
            .unwrap();
        assert_eq!(v, 42);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        let v2: u32 = resolver
            .resolve(key, || {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(99)
            })
            .unwrap();
        assert_eq!(v2, 42, "a hit must return the stored value, not recompute");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a resolver hit must not call compute at all"
        );
    }

    #[test]
    fn compute_error_does_not_poison_the_key() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = Resolver::open(dir.path()).unwrap();
        let key = crate::KeyDeriver::derive(None, "s", 1, &"p");

        let err: Result<u32> = resolver.resolve(key, || Err(AudanError::Cache("boom".into())));
        assert!(err.is_err());

        // Lock file must have been cleaned up so a retry can proceed.
        let v: u32 = resolver.resolve(key, || Ok(7)).unwrap();
        assert_eq!(v, 7);
    }
}
