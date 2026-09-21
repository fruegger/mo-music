//! A documented seam for a real inference backend, not an implementation of
//! one.
//!
//! There is no permissively-licensed default separation model to bundle or
//! even smoke-test against (RISK-1 in `audan-architecture-arc42.md`), so
//! there is no real ONNX forward pass for this crate to call. Faking tensor
//! math here would be worse than not having it.

use std::path::Path;

/// What a real chunked-ONNX-inference [`crate::Separator`] implementation
/// would need: an ONNX session (`rten` by default, `ort` optional per
/// ADR-9) built from the path [`crate::resolve_backend_model`] returns, fed
/// chunks from [`crate::chunk_signal`] at the model's expected chunk length
/// and overlap, with each stem's output chunks stitched back together via
/// [`crate::overlap_add_reconstruct`].
pub trait InferenceBackend {
    fn model_path(&self) -> &Path;

    // TODO: once a licensed model exists to wire this up against (RISK-1),
    // add something like:
    //   fn forward(&self, chunk: &[f32]) -> audan_core::Result<Vec<Vec<f32>>>;
    // returning one output chunk per stem, run through an ONNX session
    // built from `model_path()`.
}
