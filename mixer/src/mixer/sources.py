"""Multitrack sources: tracks made of labelled stems, loaded lazily."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import soundfile as sf
import yaml

SAMPLE_RATE = 44100


@dataclass
class Stem:
    name: str
    label: str  # instrument class
    load: Callable[[int, int], np.ndarray]  # (start, length) in frames -> (length, channels) float32
    is_drum: bool = False


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


def slakh_tracks(root: Path, split: str) -> Iterator[Track]:
    """Slakh2100: `<split>/TrackNNNNN/{mix.flac, stems/Sxx.flac, metadata.yaml}`."""
    for track_dir in sorted((root / split).glob("Track*")):
        meta = yaml.safe_load((track_dir / "metadata.yaml").read_text(encoding="utf-8"))
        stems = []
        for sid, info in meta["stems"].items():
            path = track_dir / "stems" / f"{sid}.flac"
            if not info.get("audio_rendered") or not path.exists():
                continue
            stems.append(
                Stem(sid, info["inst_class"], lambda s, n, p=path: _read(p, s, n), bool(info["is_drum"]))
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
            Stem(label, label, lambda s, n, i=i, f=load: f(s, n, i), label == "drums")
            for i, label in enumerate(MUSDB_STEMS)
        ]
        yield Track("musdb18", path.name.removesuffix(".stem.mp4"), int(info.rate(0)), n_frames, stems)


LOADERS = {"slakh2100": slakh_tracks, "musdb18": musdb_tracks}


def iter_tracks(dataset: str, root: Path, split: str) -> Iterator[Track]:
    return LOADERS[dataset.lower()](Path(root), split)
