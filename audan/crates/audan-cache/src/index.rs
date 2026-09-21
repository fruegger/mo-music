//! `redb`-backed index mapping [`CacheKey`] to blob metadata (S5.2). Pure
//! Rust, no SQLite C dependency.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use audan_core::error::{AudanError, Result};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::key::CacheKey;

const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("audan-cache-index-v1");

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobMeta {
    pub key: CacheKey,
    pub size_bytes: u64,
    pub created_unix: u64,
    pub last_accessed_unix: u64,
}

pub struct Index {
    db: Database,
}

fn cache_err(e: impl std::fmt::Display) -> AudanError {
    AudanError::Cache(e.to_string())
}

pub(crate) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Index {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = Database::create(path.as_ref()).map_err(cache_err)?;
        // Touch the table once so it exists even before the first `put`
        // (lets `iter`/`stats` on a brand new cache return an empty result
        // rather than a "table not found" error).
        let write_txn = db.begin_write().map_err(cache_err)?;
        {
            let _ = write_txn.open_table(TABLE).map_err(cache_err)?;
        }
        write_txn.commit().map_err(cache_err)?;
        Ok(Self { db })
    }

    pub fn get(&self, key: &CacheKey) -> Result<Option<BlobMeta>> {
        let read_txn = self.db.begin_read().map_err(cache_err)?;
        let table = read_txn.open_table(TABLE).map_err(cache_err)?;
        match table.get(key.as_bytes().as_slice()).map_err(cache_err)? {
            Some(guard) => {
                let meta: BlobMeta = serde_json::from_slice(guard.value()).map_err(cache_err)?;
                Ok(Some(meta))
            }
            None => Ok(None),
        }
    }

    pub fn put(&self, meta: &BlobMeta) -> Result<()> {
        let bytes = serde_json::to_vec(meta).map_err(cache_err)?;
        let write_txn = self.db.begin_write().map_err(cache_err)?;
        {
            let mut table = write_txn.open_table(TABLE).map_err(cache_err)?;
            table
                .insert(meta.key.as_bytes().as_slice(), bytes.as_slice())
                .map_err(cache_err)?;
        }
        write_txn.commit().map_err(cache_err)?;
        Ok(())
    }

    /// Record a fresh blob: `created_unix` and `last_accessed_unix` both set
    /// to now.
    pub fn insert_new(&self, key: &CacheKey, size_bytes: u64) -> Result<()> {
        let now = now_unix();
        self.put(&BlobMeta {
            key: *key,
            size_bytes,
            created_unix: now,
            last_accessed_unix: now,
        })
    }

    pub fn touch(&self, key: &CacheKey) -> Result<()> {
        if let Some(mut meta) = self.get(key)? {
            meta.last_accessed_unix = now_unix();
            self.put(&meta)?;
        }
        Ok(())
    }

    pub fn remove(&self, key: &CacheKey) -> Result<()> {
        let write_txn = self.db.begin_write().map_err(cache_err)?;
        {
            let mut table = write_txn.open_table(TABLE).map_err(cache_err)?;
            table.remove(key.as_bytes().as_slice()).map_err(cache_err)?;
        }
        write_txn.commit().map_err(cache_err)?;
        Ok(())
    }

    pub fn iter(&self) -> Result<Vec<BlobMeta>> {
        let read_txn = self.db.begin_read().map_err(cache_err)?;
        let table = read_txn.open_table(TABLE).map_err(cache_err)?;
        let mut out = Vec::new();
        for entry in table.iter().map_err(cache_err)? {
            let (_k, v) = entry.map_err(cache_err)?;
            let meta: BlobMeta = serde_json::from_slice(v.value()).map_err(cache_err)?;
            out.push(meta);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_touch_remove() {
        let dir = tempfile::tempdir().unwrap();
        let index = Index::open(dir.path().join("index.redb")).unwrap();
        let key = crate::KeyDeriver::derive(None, "s", 1, &"p");

        assert!(index.get(&key).unwrap().is_none());

        index.insert_new(&key, 123).unwrap();
        let meta = index.get(&key).unwrap().unwrap();
        assert_eq!(meta.size_bytes, 123);
        assert_eq!(meta.created_unix, meta.last_accessed_unix);

        index.touch(&key).unwrap();
        let touched = index.get(&key).unwrap().unwrap();
        assert!(touched.last_accessed_unix >= meta.last_accessed_unix);

        index.remove(&key).unwrap();
        assert!(index.get(&key).unwrap().is_none());
    }

    #[test]
    fn iter_lists_all_entries() {
        let dir = tempfile::tempdir().unwrap();
        let index = Index::open(dir.path().join("index.redb")).unwrap();
        for i in 0..5u32 {
            let key = crate::KeyDeriver::derive(None, "s", 1, &i);
            index.insert_new(&key, 10).unwrap();
        }
        assert_eq!(index.iter().unwrap().len(), 5);
    }
}
