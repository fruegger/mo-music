//! `redb`-backed index mapping [`CacheKey`] to blob metadata (S5.2). Pure
//! Rust, no SQLite C dependency.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use audan_core::error::{AudanError, Result};
use redb::{Database, DatabaseError, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::key::CacheKey;

const TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("audan-cache-index-v1");

/// How long to keep retrying while another handle (another `Index` in this
/// process, or another `audan` process) has the database open. Each holder
/// only keeps it for one short transaction, so this is very generous.
const OPEN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobMeta {
    pub key: CacheKey,
    pub size_bytes: u64,
    pub created_unix: u64,
    pub last_accessed_unix: u64,
}

/// The database is opened per operation rather than held for the `Index`'s
/// lifetime: `redb` allows one open handle per file -- across threads *and*
/// processes -- so a long-lived handle made every batch worker after the
/// first (and any second `audan` process sharing the cache) fail with
/// "Database already open". Threads sharing one `Index` queue on `lock`;
/// other handles are waited out by retrying the open (see `with_db`).
pub struct Index {
    path: PathBuf,
    lock: Mutex<()>,
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

fn open_read_table(
    txn: &redb::ReadTransaction,
) -> Result<Option<redb::ReadOnlyTable<&'static [u8], &'static [u8]>>> {
    match txn.open_table(TABLE) {
        Ok(table) => Ok(Some(table)),
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(None),
        Err(e) => Err(cache_err(e)),
    }
}

impl Index {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let index = Self {
            path: path.as_ref().to_path_buf(),
            lock: Mutex::new(()),
        };
        // Create the file and table up front for a brand new cache; an
        // existing file needs no open here (each open costs ~25 ms, mostly
        // its clean close). Reads still treat a missing table as empty, for
        // the moment between another process creating the file and its table.
        if !index.path.exists() {
            index.write(|_| Ok(()))?;
        }
        Ok(index)
    }

    /// Runs `f` against the database, opened for just this call.
    fn with_db<R>(&self, f: impl FnOnce(&Database) -> Result<R>) -> Result<R> {
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let start = Instant::now();
        let mut backoff = Duration::from_millis(1);
        let db = loop {
            match Database::create(&self.path) {
                Ok(db) => break db,
                Err(DatabaseError::DatabaseAlreadyOpen) if start.elapsed() < OPEN_TIMEOUT => {
                    thread::sleep(backoff);
                    backoff = (backoff * 2).min(Duration::from_millis(20));
                }
                Err(e) => return Err(cache_err(e)),
            }
        };
        f(&db)
    }

    fn write(&self, f: impl FnOnce(&mut redb::Table<&[u8], &[u8]>) -> Result<()>) -> Result<()> {
        self.with_db(|db| {
            let write_txn = db.begin_write().map_err(cache_err)?;
            {
                let mut table = write_txn.open_table(TABLE).map_err(cache_err)?;
                f(&mut table)?;
            }
            write_txn.commit().map_err(cache_err)
        })
    }

    pub fn get(&self, key: &CacheKey) -> Result<Option<BlobMeta>> {
        self.with_db(|db| {
            let read_txn = db.begin_read().map_err(cache_err)?;
            let Some(table) = open_read_table(&read_txn)? else {
                return Ok(None);
            };
            let value = table.get(key.as_bytes().as_slice()).map_err(cache_err)?;
            value
                .map(|guard| serde_json::from_slice(guard.value()).map_err(cache_err))
                .transpose()
        })
    }

    pub fn put(&self, meta: &BlobMeta) -> Result<()> {
        let bytes = serde_json::to_vec(meta).map_err(cache_err)?;
        self.write(|table| {
            table
                .insert(meta.key.as_bytes().as_slice(), bytes.as_slice())
                .map_err(cache_err)?;
            Ok(())
        })
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
        let k = key.as_bytes();
        self.write(|table| {
            let meta = table
                .get(k.as_slice())
                .map_err(cache_err)?
                .map(|guard| serde_json::from_slice::<BlobMeta>(guard.value()).map_err(cache_err))
                .transpose()?;
            if let Some(mut meta) = meta {
                meta.last_accessed_unix = now_unix();
                let bytes = serde_json::to_vec(&meta).map_err(cache_err)?;
                table.insert(k.as_slice(), bytes.as_slice()).map_err(cache_err)?;
            }
            Ok(())
        })
    }

    pub fn remove(&self, key: &CacheKey) -> Result<()> {
        self.write(|table| {
            table.remove(key.as_bytes().as_slice()).map_err(cache_err)?;
            Ok(())
        })
    }

    pub fn iter(&self) -> Result<Vec<BlobMeta>> {
        self.with_db(|db| {
            let read_txn = db.begin_read().map_err(cache_err)?;
            let Some(table) = open_read_table(&read_txn)? else {
                return Ok(Vec::new());
            };
            let mut out = Vec::new();
            for entry in table.iter().map_err(cache_err)? {
                let (_k, v) = entry.map_err(cache_err)?;
                let meta: BlobMeta = serde_json::from_slice(v.value()).map_err(cache_err)?;
                out.push(meta);
            }
            Ok(out)
        })
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

    #[test]
    fn many_handles_on_one_file_work_concurrently() {
        // One `Index` per thread on the same file, as batch workers (each
        // with its own `Resolver`) and separate processes have: used to fail
        // with "Database already open" for all but the first handle.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.redb");
        std::thread::scope(|scope| {
            for t in 0..6u32 {
                let path = &path;
                scope.spawn(move || {
                    let index = Index::open(path).unwrap();
                    for i in 0..3u32 {
                        let key = crate::KeyDeriver::derive(None, "s", 1, &(t, i));
                        index.insert_new(&key, 10).unwrap();
                        index.touch(&key).unwrap();
                    }
                });
            }
        });
        assert_eq!(Index::open(&path).unwrap().iter().unwrap().len(), 18);
    }
}
