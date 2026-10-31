"""Multitrack sources: tracks made of labelled stems, loaded lazily."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import mido
import numpy as np
import soundfile as sf
import yaml

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
                )
            )
        mix = sf.info(track_dir / "mix.flac")
        yield Track("slakh2100", track_dir.name, mix.samplerate, mix.frames, stems)


MUSDB_STEMS = ["drums", "bass", "other", "vocals"]  # stream order after the mixture


def musdb_tracks(root: Path, split: str) -> Iterator[Track]:
    """MUSDB18: `<split>/*.stem.mp4`, streams mixture/drums/bass/other/vocals.
    Needs `stempeg` (and ffmpeg); a file is decoded whole on first access."""
    import stempeg

    for path in sorted((root / split).glob("*.stem.mp4")):
        info = stempeg.Info(str(path))
        n_frames = int(info.duration(0) * info.rate(0))
        cache: dict[str, np.ndarray] = {}

        def load(start, length, idx, path=path, cache=cache):
            if "audio" not in cache:
                cache["audio"], _ = stempeg.read_stems(str(path), dtype=np.float32)
            return cache["audio"][idx + 1, start : start + length]

        stems = [
            Stem(name, MUSDB_CLASSES[name], name, lambda s, n, i=i, f=load: f(s, n, i), name == "drums")
            for i, name in enumerate(MUSDB_STEMS)
        ]
        yield Track("musdb18", path.name.removesuffix(".stem.mp4"), int(info.rate(0)), n_frames, stems)


LOADERS = {"slakh2100": slakh_tracks, "musdb18": musdb_tracks}


def iter_tracks(dataset: str, root: Path, split: str) -> Iterator[Track]:
    return LOADERS[dataset.lower()](Path(root), split)
