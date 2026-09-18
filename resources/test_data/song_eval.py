#!/usr/bin/env python3
"""Standalone evaluation harness -- no Rust toolchain needed at runtime.

Runs a real `audan` operation over every song in songs.txt (joined against
files.txt by `#`, both simple `|`-delimited tables in this directory) and
prints a table comparing the hand-curated original values against what the
already-built `audan` CLI actually measures. Replaces the earlier Rust
integration test (crates/audan-cli/tests/song_eval.rs) with a script that
just shells out to the compiled binary -- same per-song cost (dominated by
decode + model inference, not process start-up), but no cargo/rustc needed
to run it, and repeat runs get faster once audan's own L0/L1/L3 cache is
warm.

Requires a built `audan` binary:
    cargo build --release -p audan-cli

Usage:
    python song_eval.py --library-dir "C:\\...\\repertoire"

Caching note: by default this uses a fresh, throwaway cache directory per
run, so results always reflect the current `audan` build rather than a
possibly-stale cache entry from an older binary (audan's L3 beats cache key
is only `backend name/version`, e.g. "beat_this_onnx/1.0" -- it does *not*
change when the beat-tracking code itself changes, so a warm cache can
silently serve results computed by old code). Pass --cache-dir to reuse a
persistent cache across runs instead (faster on repeat runs, at that risk).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_DIR = Path(__file__).resolve().parent

T = TypeVar("T")

# ---------------------------------------------------------------------------
# generic: load songs.txt/files.txt, run an op against each file, print a
# table comparing original vs. computed values.
# ---------------------------------------------------------------------------


@dataclass
class SongRecord:
    """One row of songs.txt: the hand-curated ground truth for a song,
    keyed by `#` against files.txt's `#` to find the audio file on disk."""

    nr: int
    title: str
    artist: str
    tempo: str
    meter: str
    key: str
    chords: str


def _parse_pipe_rows(text: str) -> list[list[str]]:
    lines = text.splitlines()[1:]  # header
    return [line.split("|") for line in lines if line.strip()]


def load_songs() -> list[SongRecord]:
    text = (TEST_DATA_DIR / "songs.txt").read_text(encoding="utf-8")
    return [
        SongRecord(
            nr=int(f[0]),
            title=f[1],
            artist=f[2],
            tempo=f[3],
            meter=f[4],
            key=f[5],
            chords=f[6],
        )
        for f in _parse_pipe_rows(text)
    ]


def load_files() -> dict[int, str]:
    """`#` -> filename, from files.txt (columns: `#|Size(KB)|File`)."""
    text = (TEST_DATA_DIR / "files.txt").read_text(encoding="utf-8")
    return {int(f[0]): f[2] for f in _parse_pipe_rows(text)}


@dataclass
class EvalRow:
    song: SongRecord
    value: object | None  # T on success
    error: str | None  # set on failure instead


def run_eval(
    songs: list[SongRecord],
    files: dict[int, str],
    library_dir: Path,
    op: Callable[[Path], T],
) -> list[EvalRow]:
    """Runs `op` against every song that has a matching, existing file under
    `library_dir`, pairing each song with `op`'s result (or an error --
    missing file, no files.txt entry, or `op` itself raising). A partial
    table from a large personal library is more useful than an all-or-
    nothing crash on the first missing or corrupt file, so nothing here
    aborts the run; print_table() renders failures inline instead.

    `op` is typically slow (a whole-track CLI invocation), so progress
    prints to stderr as each song finishes, flushed immediately rather than
    only appearing once the whole (possibly many-minute) run completes.
    """
    total = len(songs)
    rows: list[EvalRow] = []
    for i, song in enumerate(songs, start=1):
        print(f"[{i:2d}/{total}] {song.title} - {song.artist}... ", end="", file=sys.stderr, flush=True)
        start = time.monotonic()

        value: T | None = None
        error: str | None = None
        filename = files.get(song.nr)
        if filename is None:
            error = f"no files.txt entry for #{song.nr}"
        else:
            path = library_dir / filename
            if not path.exists():
                error = f"file not found: {path}"
            else:
                try:
                    value = op(path)
                except Exception as e:  # noqa: BLE001 - reported per-song, not fatal
                    error = str(e)

        elapsed = time.monotonic() - start
        if error is None:
            print(f"ok ({elapsed:.1f}s)", file=sys.stderr)
        else:
            print(f"FAILED ({elapsed:.1f}s): {error}", file=sys.stderr)

        rows.append(EvalRow(song, value, error))
    return rows


def print_table(columns: list[str], rows: list[EvalRow], to_row: Callable[[SongRecord, T], list[str]]) -> None:
    """Prints a `|`-delimited table: `columns` as the header, then one row
    per song via `to_row` on success, or `Nr|Song|ERROR: ...` on failure."""
    print("|".join(columns))
    failures = 0
    for row in rows:
        if row.error is None:
            print("|".join(to_row(row.song, row.value)))
        else:
            failures += 1
            print(f"{row.song.nr}|{row.song.title}|ERROR: {row.error}")
    if failures:
        print(f"{failures} of {len(rows)} songs failed (see ERROR rows above)", file=sys.stderr)


# ---------------------------------------------------------------------------
# specific example: `audan beats <file> --model beat_this`
# ---------------------------------------------------------------------------


@dataclass
class BeatsResult:
    tempo_bpm: float
    meter: str


def run_beats_beat_this(path: Path, audan_exe: Path, cache_dir: Path) -> BeatsResult:
    """Shells out to the real `audan beats ... --model beat_this` and parses
    its JSON output -- exactly what a user would get running the CLI by
    hand, not a re-implementation of it."""
    cmd = [
        str(audan_exe),
        "beats",
        str(path),
        "--model",
        "beat_this",
        "--accept-model-license",
        "--format",
        "json",
        "--cache-dir",
        str(cache_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"audan exited {result.returncode}: {result.stderr.strip()[:200]}")
    data = json.loads(result.stdout)
    return BeatsResult(
        tempo_bpm=data["tempo"]["median_bpm"],
        meter=f"{data['meter']['beats_per_bar']}/4",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--library-dir",
        required=True,
        type=Path,
        help="folder containing the audio files named in files.txt",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=REPO_ROOT / "target" / "release" / "audan.exe",
        help="path to the built audan binary (default: target/release/audan.exe)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="reuse this cache dir across runs instead of a fresh throwaway one (see the "
        "module docstring's caching note before using this with a binary you just changed)",
    )
    args = parser.parse_args()

    if not args.exe.exists():
        sys.exit(f"audan binary not found at {args.exe} -- build it first: cargo build --release -p audan-cli")

    songs = load_songs()
    files = load_files()

    cache_dir = args.cache_dir
    cleanup_cache_dir = False
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="audan-song-eval-"))
        cleanup_cache_dir = True

    try:
        rows = run_eval(
            songs,
            files,
            args.library_dir,
            lambda path: run_beats_beat_this(path, args.exe, cache_dir),
        )

        print_table(
            ["Nr", "Song", "Tempo_O", "Tempo_C", "Meter_O", "Meter_C"],
            rows,
            lambda song, r: [
                str(song.nr),
                song.title,
                song.tempo,
                f"{r.tempo_bpm:.1f}",
                song.meter,
                r.meter,
            ],
        )
    finally:
        if cleanup_cache_dir:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
