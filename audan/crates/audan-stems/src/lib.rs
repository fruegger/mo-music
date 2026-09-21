//! Pluggable stem separation: the [`Separator`] trait, chunked-inference and
//! overlap-add reconstruction utilities, model-resolution glue against
//! `audan-model`, and the [`StemManifest`] sidecar format (S5.1 `audan-stems`
//! row, S8.6 "Backends").
//!
//! `audan-stems` ships **no default separation backend**. Demucs weights
//! licensing is genuinely unresolved (RISK-1 in `audan-architecture-arc42.md`):
//! the code is MIT, but the author stated in repository issues (#267, #327,
//! #508) that the weights are research-only per MUSDB dataset terms, while a
//! later Hugging Face model card tags the same weights `mit`. That
//! contradiction is unresolved, so this crate refuses to pick a default and
//! requires the caller to name a backend explicitly and accept its license
//! terms through `audan-model` (see [`resolve_backend_model`]) before any
//! inference happens.
//!
//! This crate's tested value is the plumbing -- the trait contract, the
//! chunking/reconstruction DSP, the manifest shape, and the resolution glue
//! -- not any specific neural model, since there is currently no
//! permissively-licensed one to bundle.

mod backend;
mod chunk;
mod manifest;
mod resolve;
mod separator;

pub use backend::InferenceBackend;
pub use chunk::{chunk_signal, overlap_add_reconstruct, Chunk};
pub use manifest::StemManifest;
pub use resolve::resolve_backend_model;
pub use separator::{Separator, StemTrack, Stems};
