//! Content-addressed cache (S5.2) — the heart of the `audan` design. Every
//! analysis stage asks a [`Resolver`] for its result before computing, keyed
//! by a chain of hashes (audio content -> decoded signal -> each stage's
//! parameters, S8.1), so identical work is never repeated and invalidation
//! needs no manual bookkeeping: bumping a stage's version changes its key
//! and every descendant key automatically.
//!
//! Components: [`KeyDeriver`] composes keys, [`BlobStore`] stores
//! zstd-compressed blobs with atomic (temp-then-rename) writes, [`Index`] is
//! a `redb`-backed map from key to blob metadata, [`Resolver`] ties the two
//! together and de-duplicates concurrent computation of the same key
//! (S6.3/RV3), and [`Evictor`] backs `audan cache stats|prune|clear`.

mod blob_store;
mod evictor;
mod index;
mod key;
mod layer;
mod resolver;

pub use blob_store::BlobStore;
pub use evictor::{CacheStats, Evictor, PruneReport};
pub use index::{BlobMeta, Index};
pub use key::{CacheKey, CacheKeyParseError, KeyDeriver};
pub use layer::CacheLayer;
pub use resolver::Resolver;

pub use audan_core::error::{AudanError, Result};
