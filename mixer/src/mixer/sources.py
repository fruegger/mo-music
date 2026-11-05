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
    """Slakh2100: `<split>/TrackNNNNN/{mix.flac (optional), stems/Sxx.flac, metadata.yaml}`."""
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
        # The mix only supplies the length and sample rate, and every stem has the same. It can be left out
        # of a copy of the dataset (it is a tenth of the size), so fall back to the first stem.
        mix_path = track_dir / "mix.flac"
        if mix_path.exists():
            ref = sf.info(mix_path)
        elif stems:
            ref = sf.info(track_dir / "stems" / f"{stems[0].name}.flac")
        else:
            continue
        yield Track("slakh2100", track_dir.name, ref.samplerate, ref.frames, stems)


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


def prepare_musdb(src: Path, dst: Path, splits=("train", "test"), limit: int | None = None,
                  force: bool = False, verbose: bool = False) -> int:
    """Decodes MUSDB18 once into `dst/<split>/<track>/{drums,bass,other,vocals}.flac`
    plus `onsets.json` (onsets estimated from each stem's energy), so training
    reads stems by random access like Slakh. `onsets.json` is written last, so a
    track that has it is finished: an interrupted run continues where it stopped
    (`force` redoes everything). `limit` counts tracks decoded in this call.
    Returns the number of tracks decoded."""
    done = 0
    for split in splits:
        paths = sorted((Path(src) / split).glob("*.stem.mp4"))
        for i, path in enumerate(paths, start=1):
            if limit is not None and done >= limit:
                return done
            out = Path(dst) / split / path.name.removesuffix(".stem.mp4")
            if (out / ONSETS_FILE).exists() and not force:
                if verbose:
                    print(f"[{split} {i}/{len(paths)}] {out.name}: already done", flush=True)
                continue
            if verbose:
                print(f"[{split} {i}/{len(paths)}] {out.name}", flush=True)
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


STUDIO_FILE = "stems.json"
STUDIO_AUDIO = (".mp3", ".wav", ".flac")


def studio_label(name: str) -> tuple[str | None, str, bool]:
    """`(class, detail, is_drum)` of a studio stem from its file name (bass, drums, guit1, piano, organ, voc2, ...).
    A stem whose content cannot be named (a `keys` bounce of several keyboard sounds) has no class, like MUSDB18's
    `other`: it is never read as a negative for any instrument. Edit `stems.json` to correct a label."""
    n = name.lower()
    if n.startswith("bass"):
        return "Bass", name, False
    if "drum" in n:
        return "Drums", name, True
    if "guit" in n:
        return "Guitar", name, False
    if n.startswith("piano"):
        return "Piano", name, False
    if n.startswith("organ"):
        return "Organ", name, False
    if n.startswith("voc"):
        return "Vocals", name, False
    return None, name, False


def prepare_studio(src: Path, dst: Path, test_songs=(), exclude=("original_timewarp",), force: bool = False,
                   verbose: bool = False) -> int:
    """Turns a folder of bounced stems, `src/<song>/tracks/<stem>.mp3|wav|flac` (all the same start, one file per
    instrument), into `dst/<split>/<song>/<stem>.flac` plus `stems.json` (class of every stem). Every stem becomes
    stereo (mono is duplicated) and is cut to the shortest one, so the stems stay aligned and always sum to the mix.
    Songs named in `test_songs` go to `test`, the others to `train`; stems named in `exclude` are left out.
    `stems.json` is written last, so a finished song is skipped unless `force`. Returns the number of songs written."""
    done = 0
    for song in sorted(p for p in Path(src).iterdir() if (p / "tracks").is_dir()):
        split = "test" if song.name in test_songs else "train"
        out = Path(dst) / split / song.name
        if (out / STUDIO_FILE).exists() and not force:
            if verbose:
                print(f"[{split}] {song.name}: already done", flush=True)
            continue
        files = sorted(f for f in (song / "tracks").iterdir() if f.suffix.lower() in STUDIO_AUDIO and f.stem not in exclude)
        if not files:
            continue
        if verbose:
            print(f"[{split}] {song.name}: {', '.join(f.stem for f in files)}", flush=True)
        audio = {}
        for f in files:
            data, sr = sf.read(f, dtype="float32", always_2d=True)
            if sr != SAMPLE_RATE:
                raise ValueError(f"{f}: sample rate {sr}, expected {SAMPLE_RATE}")
            audio[f.stem] = np.repeat(data, 2, axis=1) if data.shape[1] == 1 else data[:, :2]
        n = min(len(a) for a in audio.values())
        out.mkdir(parents=True, exist_ok=True)
        stems = []
        for name, a in audio.items():
            sf.write(out / f"{name}.flac", a[:n], SAMPLE_RATE)
            label, detail, is_drum = studio_label(name)
            stems.append({"name": name, "file": f"{name}.flac", "label": label, "source_class": name, "detail": detail,
                          "is_drum": is_drum, "peak": round(float(np.abs(a[:n]).max()), 4)})
        meta = {"sample_rate": SAMPLE_RATE, "frames": n, "stems": stems}
        (out / STUDIO_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        done += 1
    return done


def studio_tracks(root: Path, split: str) -> Iterator[Track]:
    """Songs written by `prepare_studio`: aligned stereo FLAC stems with the classes in `stems.json`. No onsets."""
    for song in sorted(p for p in (root / split).glob("*") if (p / STUDIO_FILE).exists()):
        meta = json.loads((song / STUDIO_FILE).read_text(encoding="utf-8"))
        stems = [
            Stem(
                info["name"], info["label"], info["source_class"],
                lambda s, n, p=song / info["file"]: _read(p, s, n),
                info["is_drum"], info["detail"],
            )
            for info in meta["stems"]
        ]
        yield Track("studio", song.name, meta["sample_rate"], meta["frames"], stems)


# "musdb18-prepared" is the same loader on the folder `prepare_musdb` wrote (its own line in
# sources.log, so the original path stays on record for preparing).
LOADERS = {"slakh2100": slakh_tracks, "musdb18": musdb_tracks, "musdb18-prepared": musdb_tracks, "studio": studio_tracks}


def iter_tracks(dataset: str, root: Path, split: str) -> Iterator[Track]:
    return LOADERS[dataset.lower()](Path(root), split)
