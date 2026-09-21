//! `audan cache stats|prune|clear` (S8.9): thin wrappers over
//! `audan-cache::Evictor`.

use audan_cache::Resolver;

use crate::cli::CacheAction;
use crate::config::Resolved;

pub fn run(action: &CacheAction, resolved: &Resolved) -> anyhow::Result<()> {
    let resolver = Resolver::open(&resolved.cache_root)?;
    match action {
        CacheAction::Stats => {
            let stats = resolver.evictor().stats()?;
            println!("cache root:  {}", resolved.cache_root.display());
            println!("entries:     {}", stats.count);
            println!("total bytes: {}", stats.total_bytes);
        }
        CacheAction::Prune { budget_bytes } => {
            let budget = budget_bytes.unwrap_or(resolved.analysis.cache.budget_bytes);
            let report = resolver.evictor().prune(budget)?;
            println!(
                "evicted:   {} entries, {} bytes",
                report.evicted_count, report.evicted_bytes
            );
            println!(
                "remaining: {} entries, {} bytes",
                report.remaining_count, report.remaining_bytes
            );
        }
        CacheAction::Clear => {
            resolver.evictor().clear()?;
            println!("cache cleared");
        }
    }
    Ok(())
}
