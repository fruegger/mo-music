//! Cache-root resolution (S7.2): `$AUDAN_CACHE_DIR`/`--cache-dir` if given,
//! else the XDG cache dir for "audan". `audan-cache::Resolver::open` creates
//! `<root>/blobs`, `<root>/index.redb`, and `<root>/locks` under whatever
//! root we hand it; `audan-model::ModelStore` gets `<root>/models` so model
//! weights live under the same overridable root rather than always using
//! `ModelStore::default_cache_dir()` (which ignores `AUDAN_CACHE_DIR`).

use std::path::{Path, PathBuf};

pub fn resolve_cache_root(cli_or_env: Option<PathBuf>) -> PathBuf {
    cli_or_env.unwrap_or_else(compiled_default_cache_root)
}

pub fn compiled_default_cache_root() -> PathBuf {
    directories::ProjectDirs::from("", "", "audan")
        .map(|d| d.cache_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".audan-cache"))
}

pub fn models_dir(cache_root: &Path) -> PathBuf {
    cache_root.join("models")
}
