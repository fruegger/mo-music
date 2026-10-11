//! Times each stage of the beats pipeline separately, without the cache in
//! the way: hash, decode, resample, cache encode/decode of the L0/L1 blobs,
//! and (with `--model`) the beat_this frontend and inference, including a
//! check of how inference scales when 30s slices run on parallel threads.
//!
//! cargo run --release -p audan-cli --example stage_timings -- [--model <file.rten>] <audio>...

use std::path::{Path, PathBuf};
use std::time::Instant;

use audan_beats::InferenceBackend;
use audan_cache::{BlobStore, KeyDeriver};
use audan_core::MonoSignal;

fn main() -> anyhow::Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().map(PathBuf::from),
            _ => files.push(PathBuf::from(arg)),
        }
    }
    if files.is_empty() {
        anyhow::bail!("usage: stage_timings [--model <file.rten>] <audio>...");
    }
    let backend = model
        .map(|m| audan_beats::OnnxBackend::new(&m, "bench"))
        .transpose()?;

    for file in &files {
        time_file(file, backend.as_ref())?;
    }
    Ok(())
}

fn time_file(path: &Path, backend: Option<&audan_beats::OnnxBackend>) -> anyhow::Result<()> {
    println!("{}", path.display());
    let total = Instant::now();

    let hash = lap("hash", || audan_cli::pipeline::hash_file(path))?;
    let (signal, _) = lap("decode", || audan_io::decode_file(path))?;
    let mono = lap("resample", || audan_io::to_analysis_rate(&signal))?;
    println!(
        "  {:<22} {:.1}s audio, {} ch @ {} Hz",
        "",
        mono.duration_seconds(),
        signal.channels,
        signal.sample_rate
    );

    let dir = tempfile::tempdir()?;
    let store = BlobStore::open(dir.path())?;
    let k0 = KeyDeriver::derive(None, "bench", 1, &hash);
    let k1 = KeyDeriver::derive(Some(&k0), "bench", 1, &hash);
    let l0 = lap("cache put L0", || store.put(&k0, &signal))?;
    let l1 = lap("cache put L1", || store.put(&k1, &mono))?;
    println!("  {:<22} L0 {} KiB, L1 {} KiB on disk", "", l0 / 1024, l1 / 1024);
    let _: Option<audan_core::Signal> = lap("cache get L0", || store.get(&k0))?;
    let _: Option<MonoSignal> = lap("cache get L1", || store.get(&k1))?;

    lap("beats (DSP)", || audan_beats::track_beats_default(&mono))?;

    if let Some(backend) = backend {
        lap("onnx frontend (mel)", || {
            Ok::<_, anyhow::Error>(audan_beats::onnx_frontend::OnnxFrontend::compute(&mono))
        })?;
        lap("onnx run (all chunks)", || backend.run(&mono))?;

        // ~30s slices = one inference chunk each: sequential vs. one
        // thread per slice says whether rten already uses all cores.
        let slices = slices(&mono, 30.0, 4);
        lap("4 slices, sequential", || {
            slices.iter().try_for_each(|s| backend.run(s).map(drop))
        })?;
        lap("4 slices, parallel", || {
            std::thread::scope(|scope| {
                let handles: Vec<_> = slices
                    .iter()
                    .map(|s| scope.spawn(move || backend.run(s).map(drop)))
                    .collect();
                handles.into_iter().try_for_each(|h| h.join().unwrap())
            })
        })?;
    }
    println!("  {:<22} {:>7.2}s", "total", total.elapsed().as_secs_f64());
    Ok(())
}

fn slices(mono: &MonoSignal, seconds: f64, n: usize) -> Vec<MonoSignal> {
    let len = (seconds * mono.sample_rate as f64) as usize;
    mono.samples
        .chunks(len)
        .take(n)
        .map(|c| MonoSignal {
            sample_rate: mono.sample_rate,
            samples: c.to_vec(),
        })
        .collect()
}

fn lap<T, E>(label: &str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = Instant::now();
    let out = f();
    println!("  {:<22} {:>7.2}s", label, start.elapsed().as_secs_f64());
    out
}
