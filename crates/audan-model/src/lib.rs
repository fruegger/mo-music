//! Model registry: manifest parsing, checksum verification, and the
//! license-gated fetch/install flow described in S6.4 (RV4) and S8.6 of
//! `audan-architecture-arc42.md`.
//!
//! `audan` ships no bundled model weights beyond a small default beat model
//! (owned by `audan-beats`). Full-accuracy models are fetched on first use,
//! checksum-verified against a signed registry manifest, and their license
//! terms -- including any unresolved ambiguity such as RISK-1 -- must be
//! explicitly accepted before download. This is the mechanism that keeps
//! `audan`'s own license posture clean (Q1, ADR-7) while remaining useful.

mod checksum;
mod registry;
mod store;

pub use checksum::verify_checksum;
pub use registry::{load, ModelEntry, Registry};
pub use store::{LicenseGate, ModelStore};
