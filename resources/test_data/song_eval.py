#!/usr/bin/env python3
"""Standalone evaluation harness -- no Rust toolchain needed at runtime.

Runs a real `audan` operation over every song in songs.txt (joined against
files.txt by `#`, both simple `|`-delimited tables in this directory) and
prints a table comparing the hand-curated original values against what the
already-built `audan` CLI actually measures. Shells out to the compiled
binary rather than linking against the Rust crates directly -- same
per-song cost (dominated by decode + analysis, not process start-up), but
no cargo/rustc needed to run it, and repeat runs get faster once audan's
own L0/L1/L3 cache is warm.

Several operations are available (see OPS below -- beats with the default
fallback backend, beats with beat_this, key, chords), each with its own
table columns since what's being compared differs (tempo/meter vs. key vs.
chord vocabulary). Pick one with --op, or leave it out to choose from a
menu.

Requires a built `audan` binary:
    cargo build --release -p audan-cli

Usage:
    python song_eval.py --library-dir "C:\\...\\repertoire" [--op beats-beat-this]

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
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_DIR = Path(__file__).resolve().parent

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
    value: Any | None  # set on success
    error: str | None  # set on failure instead


def run_eval(
    songs: list[SongRecord],
    files: dict[int, str],
    library_dir: Path,
    op: Callable[[Path], Any],
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

        value: Any | None = None
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


def print_table(columns: list[str], rows: list[EvalRow], to_row: Callable[[SongRecord, Any], list[str]]) -> None:
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


def run_audan(audan_exe: Path, cache_dir: Path, *args: str) -> dict[str, Any] | list[Any]:
    """Runs `audan <args...> --format json --cache-dir <cache_dir>` and
    parses the JSON it prints -- the one place every op below actually
    shells out, so a change to how audan is invoked (a new global flag,
    error-message parsing) only needs to happen here."""
    cmd = [str(audan_exe), *args, "--format", "json", "--cache-dir", str(cache_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"audan exited {result.returncode}: {result.stderr.strip()[:200]}")
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# ops: each is a (run, columns, to_row) triple registered in OPS below.
# `run` always has the signature (path, audan_exe, cache_dir) -> result;
# `to_row` turns (song, result) into the row cells named by `columns`.
# ---------------------------------------------------------------------------


@dataclass
class BeatsResult:
    tempo_bpm: float
    meter: str


def run_beats(path: Path, audan_exe: Path, cache_dir: Path, model: str | None = None) -> BeatsResult:
    """`audan beats`, optionally with `--model <model>`. `model=None` uses
    the always-available onset_fallback backend (no model, no license)."""
    args = ["beats", str(path)]
    if model is not None:
        args += ["--model", model, "--accept-model-license"]
    data = run_audan(audan_exe, cache_dir, *args)
    return BeatsResult(
        tempo_bpm=data["tempo"]["median_bpm"],
        meter=f"{data['meter']['beats_per_bar']}/4",
    )


def beats_row(song: SongRecord, r: BeatsResult) -> list[str]:
    return [str(song.nr), song.title, song.tempo, f"{r.tempo_bpm:.1f}", song.meter, r.meter]


BEATS_COLUMNS = ["Nr", "Song", "Tempo_O", "Tempo_C", "Meter_O", "Meter_C"]


@dataclass
class KeyResult:
    name: str  # e.g. "F# minor", straight from audan's own KeyEstimate.name
    short: str  # e.g. "F#:min", matching songs.txt's Key column convention
    confidence: float


def run_key(path: Path, audan_exe: Path, cache_dir: Path) -> KeyResult:
    data = run_audan(audan_exe, cache_dir, "key", str(path))  # [{"value": {"name": ...}, "confidence": ...}, ...]
    top = data[0]
    name = top["value"]["name"]
    tonic, _, mode_word = name.rpartition(" ")
    short = f"{tonic}:{'maj' if mode_word == 'major' else 'min'}"
    return KeyResult(name=name, short=short, confidence=top["confidence"])


def key_row(song: SongRecord, r: KeyResult) -> list[str]:
    return [str(song.nr), song.title, song.key, r.short, f"{r.confidence:.2f}"]


KEY_COLUMNS = ["Nr", "Song", "Key_O", "Key_C", "Confidence"]


@dataclass
class ChordsResult:
    vocabulary: str  # distinct chords, in first-occurrence order, Harte notation
    n_events: int


def run_chords(path: Path, audan_exe: Path, cache_dir: Path) -> ChordsResult:
    data = run_audan(audan_exe, cache_dir, "chords", str(path))  # {"chords": [{"chord": "E:min", ...}, ...], ...}
    seen: list[str] = []
    for event in data["chords"]:
        if event["chord"] not in seen:
            seen.append(event["chord"])
    return ChordsResult(vocabulary=",".join(seen), n_events=len(data["chords"]))


def chords_row(song: SongRecord, r: ChordsResult) -> list[str]:
    return [str(song.nr), song.title, song.chords, r.vocabulary]


CHORDS_COLUMNS = ["Nr", "Song", "Chords_O", "Chords_C"]


@dataclass
class OpSpec:
    description: str
    columns: list[str]
    run: Callable[[Path, Path, Path], Any]
    to_row: Callable[[SongRecord, Any], list[str]]


def build_ops() -> dict[str, OpSpec]:
    return {
        "beats": OpSpec(
            description="audan beats (default onset_fallback backend, no model)",
            columns=BEATS_COLUMNS,
            run=lambda path, exe, cache: run_beats(path, exe, cache, model=None),
            to_row=beats_row,
        ),
        "beats-beat-this": OpSpec(
            description="audan beats --model beat_this",
            columns=BEATS_COLUMNS,
            run=lambda path, exe, cache: run_beats(path, exe, cache, model="beat_this"),
            to_row=beats_row,
        ),
        "key": OpSpec(
            description="audan key",
            columns=KEY_COLUMNS,
            run=run_key,
            to_row=key_row,
        ),
        "chords": OpSpec(
            description="audan chords (distinct chord vocabulary used, deduped in order)",
            columns=CHORDS_COLUMNS,
            run=run_chords,
            to_row=chords_row,
        ),
    }


def choose_op(ops: dict[str, OpSpec], requested: str | None) -> str:
    """Returns the chosen op's key: `requested` if given (validated against
    `ops`), otherwise an interactive numbered menu on stderr (stdout stays
    clean for the table itself). Refuses to guess when stdin isn't a TTY --
    a script piping this in without --op almost certainly wants a clear
    error, not a menu prompt it can't answer."""
    if requested is not None:
        if requested not in ops:
            sys.exit(f"unknown --op {requested!r}; choices: {', '.join(ops)}")
        return requested

    if not sys.stdin.isatty():
        sys.exit(f"no --op given and stdin is not a TTY; pass one of: {', '.join(ops)}")

    keys = list(ops)
    print("Choose an operation to run:", file=sys.stderr)
    print(f"  0. exit", file=sys.stderr)
    for i, k in enumerate(keys, start=1):
        print(f"  {i}. {k} -- {ops[k].description}", file=sys.stderr)
    while True:
        choice = input(f"[0-{len(keys)}]: ").strip()
        if int(choice) == 0:
            return "exit"
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return keys[int(choice) - 1]
        print("invalid choice, try again", file=sys.stderr)


def main() -> None:
    ops = build_ops()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--library-dir",
        required=True,
        type=Path,
        help="folder containing the audio files named in files.txt",
    )
    parser.add_argument(
        "--op",
        choices=sorted(ops),
        default=None,
        help="which audan operation to run; omit to choose from a menu",
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

    choice = choose_op(ops, args.op)

    if choice!="exit":
        op = ops[choice]
        songs = load_songs()
        files = load_files()

        cache_dir = args.cache_dir
        cleanup_cache_dir = False
        if cache_dir is None:
            cache_dir = Path(tempfile.mkdtemp(prefix="audan-song-eval-"))
            cleanup_cache_dir = True

        try:
            rows = run_eval(songs, files, args.library_dir, lambda path: op.run(path, args.exe, cache_dir))
            print_table(op.columns, rows, op.to_row)
        finally:
            if cleanup_cache_dir:
                shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
