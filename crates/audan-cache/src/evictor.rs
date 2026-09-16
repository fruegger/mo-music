//! LRU eviction against a configurable size budget (S5.2); backs
//! `audan cache stats|prune|clear`.

use audan_core::error::Result;

use crate::blob_store::BlobStore;
use crate::index::Index;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub count: u64,
    pub total_bytes: u64,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PruneReport {
    pub evicted_count: u64,
    pub evicted_bytes: u64,
    pub remaining_count: u64,
    pub remaining_bytes: u64,
}

pub struct Evictor<'a> {
    blobs: &'a BlobStore,
    index: &'a Index,
}

impl<'a> Evictor<'a> {
    pub fn new(blobs: &'a BlobStore, index: &'a Index) -> Self {
        Self { blobs, index }
    }

    pub fn stats(&self) -> Result<CacheStats> {
        let entries = self.index.iter()?;
        let total_bytes = entries.iter().map(|m| m.size_bytes).sum();
        Ok(CacheStats {
            count: entries.len() as u64,
            total_bytes,
        })
    }

    /// Evict least-recently-accessed blobs until total size is at or under
    /// `budget_bytes`.
    pub fn prune(&self, budget_bytes: u64) -> Result<PruneReport> {
        let mut entries = self.index.iter()?;
        entries.sort_by_key(|m| m.last_accessed_unix);

        let mut total_bytes: u64 = entries.iter().map(|m| m.size_bytes).sum();
        let mut remaining_count = entries.len() as u64;
        let mut evicted_count = 0u64;
        let mut evicted_bytes = 0u64;

        for meta in entries {
            if total_bytes <= budget_bytes {
                break;
            }
            self.blobs.remove(&meta.key)?;
            self.index.remove(&meta.key)?;
            total_bytes = total_bytes.saturating_sub(meta.size_bytes);
            evicted_bytes += meta.size_bytes;
            evicted_count += 1;
            remaining_count -= 1;
        }

        Ok(PruneReport {
            evicted_count,
            evicted_bytes,
            remaining_count,
            remaining_bytes: total_bytes,
        })
    }

    pub fn clear(&self) -> Result<()> {
        self.prune(0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{KeyDeriver, Resolver};

    #[test]
    fn stats_prune_clear() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = Resolver::open(dir.path()).unwrap();

        for i in 0..5u32 {
            let key = KeyDeriver::derive(None, "s", 1, &i);
            let _: u32 = resolver.resolve(key, || Ok(i)).unwrap();
        }

        let stats = resolver.evictor().stats().unwrap();
        assert_eq!(stats.count, 5);
        assert!(stats.total_bytes > 0);

        let report = resolver.evictor().prune(0).unwrap();
        assert_eq!(report.evicted_count, 5);
        assert_eq!(report.remaining_count, 0);
        assert_eq!(report.remaining_bytes, 0);

        let stats = resolver.evictor().stats().unwrap();
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn prune_keeps_most_recently_accessed() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = Resolver::open(dir.path()).unwrap();

        let old_key = KeyDeriver::derive(None, "s", 1, &"old");
        let new_key = KeyDeriver::derive(None, "s", 1, &"new");
        let _: u32 = resolver.resolve(old_key, || Ok(1)).unwrap();
        let _: u32 = resolver.resolve(new_key, || Ok(2)).unwrap();

        // `Index` timestamps are second-resolution; sleep past a second
        // boundary so the touch below is unambiguously later than either
        // insert, then touch `old` more recently than `new` so `new` becomes
        // the eviction target despite being inserted later.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        resolver.index().touch(&old_key).unwrap();

        let stats_before = resolver.evictor().stats().unwrap();
        let budget = stats_before.total_bytes - 1;
        resolver.evictor().prune(budget).unwrap();

        assert!(resolver.blobs().contains(&old_key));
        assert!(!resolver.blobs().contains(&new_key));
    }

    #[test]
    fn clear_empties_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = Resolver::open(dir.path()).unwrap();
        let key = KeyDeriver::derive(None, "s", 1, &"p");
        let _: u32 = resolver.resolve(key, || Ok(1)).unwrap();

        resolver.evictor().clear().unwrap();
        assert_eq!(resolver.evictor().stats().unwrap().count, 0);
        assert!(!resolver.blobs().contains(&key));
    }
}
