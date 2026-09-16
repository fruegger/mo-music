//! End-to-end smoke tests, a cache-hit test against the real `Resolver`, and
//! a dispatch-level exit-code test. Calls `audan_cli::run` (the library
//! entry point) directly rather than shelling out to the compiled binary.

use clap::Parser;

fn synth_wav_file(dir: &std::path::Path, seconds: f32) -> std::path::PathBuf {
    let path = dir.join("tone.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).unwrap();
    let n_frames = (44100.0 * seconds) as u32;
    for i in 0..n_frames {
        let t = i as f32 / 44100.0;
        let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        writer
            .write_sample((v * i16::MAX as f32 * 0.8) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
    path
}

#[test]
fn probe_produces_a_sane_report() {
    let dir = tempfile::tempdir().unwrap();
    let wav = synth_wav_file(dir.path(), 2.0);

    let report = audan_io::probe_file(&wav).unwrap();

    assert!((report.duration_seconds - 2.0).abs() < 1e-3);
    assert_eq!(report.format, "wav");
    assert_eq!(report.delay_stripped_samples, 0);
}

#[test]
fn beats_end_to_end_produces_a_sane_beat_grid() {
    let dir = tempfile::tempdir().unwrap();
    let wav = synth_wav_file(dir.path(), 6.0);
    let cache_dir = dir.path().join("cache");

    let cli = audan_cli::cli::Cli::try_parse_from([
        "audan",
        "--cache-dir",
        cache_dir.to_str().unwrap(),
        "beats",
        wav.to_str().unwrap(),
    ])
    .unwrap();

    audan_cli::run(cli).unwrap();

    // The L0/L1/L3 entries the run above must have produced are directly
    // inspectable through the same `Resolver`/`Evictor` API `audan cache
    // stats` uses.
    let resolver = audan_cache::Resolver::open(&cache_dir).unwrap();
    let stats = resolver.evictor().stats().unwrap();
    assert!(
        stats.count >= 3,
        "expected at least L0 + L1 + L3 cache entries, got {}",
        stats.count
    );
    assert!(stats.total_bytes > 0);
}

#[test]
fn a_second_run_against_the_same_cache_dir_is_a_cache_hit_not_a_recompute() {
    let dir = tempfile::tempdir().unwrap();
    let wav = synth_wav_file(dir.path(), 6.0);
    let cache_dir = dir.path().join("cache");

    let run = || {
        let cli = audan_cli::cli::Cli::try_parse_from([
            "audan",
            "--cache-dir",
            cache_dir.to_str().unwrap(),
            "beats",
            wav.to_str().unwrap(),
        ])
        .unwrap();
        audan_cli::run(cli).unwrap();
    };

    run(); // cold: L0/L1/L3 computed and stored.

    // `Resolver::open` holds an exclusive lock on the `redb` index for as
    // long as it's alive, and `run()` opens its own `Resolver` internally,
    // so each stats snapshot must be taken (and dropped) before the next
    // `run()` call, not held open across it.
    let stats_after_first = {
        audan_cache::Resolver::open(&cache_dir)
            .unwrap()
            .evictor()
            .stats()
            .unwrap()
    };

    run(); // warm: must resolve every layer as a hit, adding no new entries.

    let stats_after_second = audan_cache::Resolver::open(&cache_dir)
        .unwrap()
        .evictor()
        .stats()
        .unwrap();
    assert_eq!(
        stats_after_second.count, stats_after_first.count,
        "a warm second run against the same cache dir must not add cache entries"
    );
}

#[test]
fn chords_after_beats_reuses_the_cached_beat_grid_rv2() {
    let dir = tempfile::tempdir().unwrap();
    let wav = synth_wav_file(dir.path(), 6.0);
    let cache_dir = dir.path().join("cache");

    let beats_cli = audan_cli::cli::Cli::try_parse_from([
        "audan",
        "--cache-dir",
        cache_dir.to_str().unwrap(),
        "beats",
        wav.to_str().unwrap(),
    ])
    .unwrap();
    audan_cli::run(beats_cli).unwrap();

    let stats_after_beats = audan_cache::Resolver::open(&cache_dir)
        .unwrap()
        .evictor()
        .stats()
        .unwrap();

    let chords_cli = audan_cli::cli::Cli::try_parse_from([
        "audan",
        "--cache-dir",
        cache_dir.to_str().unwrap(),
        "chords",
        wav.to_str().unwrap(),
    ])
    .unwrap();
    audan_cli::run(chords_cli).unwrap();

    let stats_after_chords = audan_cache::Resolver::open(&cache_dir)
        .unwrap()
        .evictor()
        .stats()
        .unwrap();
    // `chords` adds exactly one new entry (the L4 chord sequence); the L0,
    // L1, and L3 beat-grid entries from the `beats` run above are hits, not
    // recomputations.
    assert_eq!(
        stats_after_chords.count,
        stats_after_beats.count + 1,
        "chords after beats should only add the L4 chord-sequence entry, reusing L0/L1/L3"
    );
}

#[test]
fn missing_file_surfaces_an_audan_error_with_the_runtime_error_exit_code() {
    let cli = audan_cli::cli::Cli::try_parse_from([
        "audan",
        "probe",
        "/definitely/does/not/exist-12345.mp3",
    ])
    .unwrap();

    let err = audan_cli::run(cli).unwrap_err();
    let audan_err = err
        .downcast_ref::<audan_core::AudanError>()
        .expect("expected an AudanError");
    assert_eq!(
        audan_err.exit_code() as i32,
        audan_core::ExitCode::RuntimeError as i32
    );
}

#[test]
fn exit_code_mapping_matches_the_documented_table() {
    use audan_core::{AudanError, ExitCode};

    assert_eq!(
        AudanError::Usage("x".into()).exit_code() as i32,
        ExitCode::UsageError as i32
    );
    assert_eq!(
        AudanError::UnsupportedFormat("x".into()).exit_code() as i32,
        ExitCode::UnsupportedFormat as i32
    );
    assert_eq!(
        AudanError::LowConfidence {
            confidence: 0.1,
            threshold: 0.5
        }
        .exit_code() as i32,
        ExitCode::LowConfidence as i32
    );
    assert_eq!(
        AudanError::Model("x".into()).exit_code() as i32,
        ExitCode::RuntimeError as i32
    );
    assert_eq!(
        AudanError::InvalidInput("x".into()).exit_code() as i32,
        ExitCode::RuntimeError as i32
    );
}

#[test]
fn stems_without_a_permissively_licensed_backend_fails_honestly_rather_than_faking_output() {
    let dir = tempfile::tempdir().unwrap();
    let wav = synth_wav_file(dir.path(), 1.0);
    let cache_dir = dir.path().join("cache");

    let cli = audan_cli::cli::Cli::try_parse_from([
        "audan",
        "--cache-dir",
        cache_dir.to_str().unwrap(),
        "stems",
        wav.to_str().unwrap(),
        "--model",
        "htdemucs",
    ])
    .unwrap();

    // Declines the license (no --accept-model-license, non-TTY test stdin),
    // so this should fail at the license gate before ever claiming to
    // separate anything.
    let err = audan_cli::run(cli).unwrap_err();
    let audan_err = err
        .downcast_ref::<audan_core::AudanError>()
        .expect("expected an AudanError");
    assert!(matches!(audan_err, audan_core::AudanError::Model(_)));
}
