"""Multitrack sources: tracks made of labelled stems, loaded lazily."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import mido
import numpy as np
import soundfile as sf
import yaml

from . import onset_detect
from .vocab import MUSDB_CLASSES, SLAKH_CLASSES

SAMPLE_RATE = 44100


@dataclass
class Stem:
    name: str
    label: str | None  # class in the shared vocabulary; None = unknown mixture (MUSDB18 "other")
    source_class: str  # the dataset's own class name
    load: Callable[[int, int], np.ndarray]  # (start, length) in frames -> (length, channels) float32
    is_drum: bool = False
    detail: str = ""  # finer identity where the dataset has it (Slakh: MIDI program name)
    onsets: Callable[[], np.ndarray] | None = None  # note onsets in seconds from the track start, if known
    onset_source: str | None = None  # "midi" (exact) or "estimated" (from the audio)

    @property
    def ambiguous(self) -> bool:
        return self.label is None


@dataclass
class Track:
    dataset: str
    name: str
    sample_rate: int
    n_frames: int
    stems: list[Stem] = field(default_factory=list)


def _read(path: Path, start: int, length: int) -> np.ndarray:
    audio, _ = sf.read(path, start=start, frames=length, dtype="float32", always_2d=True)
    if len(audio) < length:  # a stem shorter than the mix is silent after its end
        audio = np.pad(audio, ((0, length - len(audio)), (0, 0)))
    return audio


def _slice(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    chunk = audio[start : start + length]
    return chunk if len(chunk) == length else np.pad(chunk, ((0, length - len(chunk)), (0, 0)))


def midi_onsets(path: Path) -> np.ndarray:
    """Sorted, distinct note-on times of a MIDI file in seconds, following its
    tempo map (a note-on with velocity 0 is a note-off)."""
    t, out = 0.0, set()
    for msg in mido.MidiFile(path):  # iterating yields messages with delta times in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            out.add(t)
    return np.array(sorted(out), dtype=np.float64)


def _lazy(fn, *args):
    """Calls `fn(*args)` on first use and keeps the result."""
    box = []

    def get():
        if not box:
            box.append(fn(*args))
        return box[0]

    return get


def slakh_tracks(root: Path, split: str) -> Iterator[Track]:
    """Slakh2100: `<split>/TrackNNNNN/{mix.flac, stems/Sxx.flac, metadata.yaml}`."""
    for track_dir in sorted((root / split).glob("Track*")):
        meta = yaml.safe_load((track_dir / "metadata.yaml").read_text(encoding="utf-8"))
        stems = []
        for sid, info in meta["stems"].items():
            path = track_dir / "stems" / f"{sid}.flac"
            if not info.get("audio_rendered") or not path.exists():
                continue
            midi = track_dir / "MIDI" / f"{sid}.mid"
            stems.append(
                Stem(
                    sid,
                    SLAKH_CLASSES[info["inst_class"]],
                    info["inst_class"],
                    lambda s, n, p=path: _read(p, s, n),
                    bool(info["is_drum"]),
                    info.get("midi_program_name", ""),
                    _lazy(midi_onsets, midi) if midi.exists() else None,
                    "midi" if midi.exists() else None,
                )
            )
        mix = sf.info(track_dir / "mix.flac")
        yield Track("slakh2100", track_dir.name, mix.samplerate, mix.frames, stems)


# MUSDB18 .stem.mp4 holds five AAC streams, in this order.
MUSDB_STREAMS = ["mixture", "drums", "bass", "other", "vocals"]
MUSDB_STEMS = MUSDB_STREAMS[1:]
ONSETS_FILE = "onsets.json"


def decode_musdb(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """Decodes a MUSDB18 `.stem.mp4` into `{stem: (frames, 2) float32}` and the
    sample rate (PyAV, which bundles the codecs: no ffmpeg needed). Takes a few
    seconds per track, hence `prepare_musdb`."""
    import av

    stems = {}
    rate = SAMPLE_RATE
    for index, name in enumerate(MUSDB_STREAMS):
        if name == "mixture":
            continue
        with av.open(str(path)) as container:
            stream = container.streams.audio[index]
            rate = stream.rate
            chunks = [frame.to_ndarray() for frame in container.decode(stream)]
        stems[name] = np.ascontiguousarray(np.concatenate(chunks, axis=1).T, dtype=np.float32)
    n = min(len(a) for a in stems.values())
    return {k: v[:n] for k, v in stems.items()}, rate


def estimate_onsets(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return onset_detect.detect_onsets(audio, sample_rate)


def prepare_musdb(src: Path, dst: Path, splits=("train", "test"), limit: int | None = None) -> int:
    """Decodes MUSDB18 once into `dst/<split>/<track>/{drums,bass,other,vocals}.flac`
    plus `onsets.json` (onsets estimated from each stem's energy), so training
    reads stems by random access like Slakh. Returns the number of tracks."""
    done = 0
    for split in splits:
        for path in sorted((Path(src) / split).glob("*.stem.mp4")):
            if limit is not None and done >= limit:
                return done
            out = Path(dst) / split / path.name.removesuffix(".stem.mp4")
            out.mkdir(parents=True, exist_ok=True)
            stems, rate = decode_musdb(path)
            onsets = {}
            for name, audio in stems.items():
                sf.write(out / f"{name}.flac", audio, rate)
                onsets[name] = [round(float(t), 4) for t in estimate_onsets(audio, rate)]
            meta = {"sample_rate": rate, "detector": "mixer.onset_detect", "onsets": onsets}
            (out / ONSETS_FILE).write_text(json.dumps(meta), encoding="utf-8")
            done += 1
    return done


def _musdb_stem(name: str, load, onsets) -> Stem:
    return Stem(name, MUSDB_CLASSES[name], name, load, name == "drums", "", onsets, "estimated")


def musdb_tracks(root: Path, split: str) -> Iterator[Track]:
    """MUSDB18, either as prepared by `prepare_musdb` (`<split>/<track>/*.flac`,
    fast random access) or as the original `<split>/*.stem.mp4` (each track is
    decoded whole on first use, several seconds each: fine for a look, too slow
    for training). Onsets are estimated from the stems' energy; there is no MIDI."""
    for track_dir in sorted(p for p in (root / split).glob("*") if p.is_dir() and (p / ONSETS_FILE).exists()):
        meta = json.loads((track_dir / ONSETS_FILE).read_text(encoding="utf-8"))
        stems = [
            _musdb_stem(
                name,
                lambda s, n, p=track_dir / f"{name}.flac": _read(p, s, n),
                lambda n=name, m=meta: np.array(m["onsets"][n], dtype=np.float64),
            )
            for name in MUSDB_STEMS
        ]
        info = sf.info(track_dir / "drums.flac")
        yield Track("musdb18", track_dir.name, info.samplerate, info.frames, stems)

    for path in sorted((root / split).glob("*.stem.mp4")):
        decoded = _lazy(decode_musdb, path)
        stems = [
            _musdb_stem(
                name,
                lambda s, n, nm=name, d=decoded: _slice(d()[0][nm], s, n),
                _lazy(lambda nm=name, d=decoded: estimate_onsets(d()[0][nm], d()[1])),
            )
            for name in MUSDB_STEMS
        ]
        # The length and rate are needed up front; the first stream's header has them.
        import av

        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            rate, n_frames = stream.rate, int(float(stream.duration * stream.time_base) * stream.rate)
        yield Track("musdb18", path.name.removesuffix(".stem.mp4"), rate, n_frames, stems)


LOADERS = {"slakh2100": slakh_tracks, "musdb18": musdb_tracks}


def iter_tracks(dataset: str, root: Path, split: str) -> Iterator[Track]:
    return LOADERS[dataset.lower()](Path(root), split)
