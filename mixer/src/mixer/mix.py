"""Snippet rendering: crop, per-stem effects and gain, sum, label.

Label file (schema 1), one JSON object per snippet:

    schema, mixer_version, effects_backend     provenance
    dataset, track, seed, config               enough to re-render it
    sample_rate, channels, seconds, frames     the audio
    start_frame, start_seconds                 where in the track the crop starts
    mix_scale                                  gain applied to avoid clipping
    stems[]    stem, class (shared vocabulary or null), source_class, detail,
               ambiguous, is_drum, gain_db, rms_db, active, onsets, effects[]
    activity   fps, n_frames, per_stem {stem: "0110..."}  energy above
               `activity_threshold_db`, per frame, in the final mix
    classes    classes with at least one active, unambiguous stem
    ambiguous  true while an unlabelled mixture stem (MUSDB18 "other") is
               audible: the truth for classes it could contain is unknown
    times      seconds from the snippet start:
                 onsets            every audible note onset, sorted
                 onsets_by_class   the same, per class (unambiguous stems)
                 onsets_complete   false if an active stem has no onset data
                                   (MUSDB18): onsets are then incomplete and
                                   must not be used as negatives
                 boundaries        reserved, empty
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from . import __version__
from .effects import DEFAULT_CHAIN, apply_chain, backend_version, db_to_gain
from .sources import Track

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MixConfig:
    seconds: float = 8.0
    gain_db_range: tuple[float, float] = (-12.0, 6.0)  # relative stem volume
    p_stem_dropped: float = 0.15  # chance a stem is left out of the mix
    min_stems: int = 1
    effects: bool = True
    peak_target: float = 0.95  # the final mix is scaled down to this peak if it clips
    activity_fps: int = 10
    activity_threshold_db: float = -50.0  # frame RMS (dBFS, in the final mix) above which a stem counts as active
    onset_window_s: float = 0.05  # a note onset is kept if its stem exceeds that level within this window after it

    @classmethod
    def from_dict(cls, data: dict) -> "MixConfig":
        data = dict(data)
        data["gain_db_range"] = tuple(data["gain_db_range"])
        return cls(**data)


@dataclass
class Snippet:
    mix: np.ndarray  # (frames, channels) float32
    sample_rate: int
    stems: np.ndarray  # (n_stems_in_mix, frames, channels): each stem as heard in the mix
    activity: np.ndarray  # (n_stems_in_mix, n_frames) bool
    labels: dict = field(default_factory=dict)


def _stable_hash(text: str) -> int:
    """Python's `hash()` is salted per process; seeds must not be (FNV-1a)."""
    h = 2166136261
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * 16777619) % 2**32
    return h


def onset_times(stem_info: list[dict]) -> dict:
    by_class: dict[str, list[float]] = {}
    for info in stem_info:
        if info["onsets"] and info["class"] is not None:
            by_class.setdefault(info["class"], []).extend(info["onsets"])
    return {
        "onsets": sorted({o for info in stem_info for o in (info["onsets"] or [])}),
        "onsets_by_class": {c: sorted(set(v)) for c, v in sorted(by_class.items())},
        "onsets_complete": all(info["onsets"] is not None for info in stem_info if info["active"]),
        "boundaries": [],
    }


def frame_activity(stems: np.ndarray, sample_rate: int, fps: int, threshold_db: float) -> np.ndarray:
    """(n_stems, n_frames) bool: frame RMS above `threshold_db` dBFS."""
    hop = sample_rate // fps
    n_frames = stems.shape[1] // hop
    power = (stems[:, : n_frames * hop] ** 2).mean(axis=2).reshape(len(stems), n_frames, hop).mean(axis=2)
    return 10.0 * np.log10(power + 1e-12) > threshold_db


def audible_onsets(
    onsets: np.ndarray | None, audio: np.ndarray, start_s: float, config: MixConfig, sample_rate: int
) -> list[float] | None:
    """Onsets of one stem inside the crop, in seconds from the snippet start,
    keeping those that are audible: the stem's RMS in the `onset_window_s`
    after the onset exceeds `activity_threshold_db`. None when the stem has no
    onset data."""
    if onsets is None:
        return None
    rel = onsets[(onsets >= start_s) & (onsets < start_s + config.seconds)] - start_s
    window = max(1, int(config.onset_window_s * sample_rate))
    kept = []
    for o in rel:
        i = int(o * sample_rate)
        seg = audio[i : i + window]
        if len(seg) and 10.0 * np.log10(float((seg**2).mean()) + 1e-12) > config.activity_threshold_db:
            kept.append(round(float(o), 4))
    return kept


def render_snippet(track: Track, seed: int, config: MixConfig = MixConfig()) -> Snippet:
    """Deterministic in (track, seed, config)."""
    rng = np.random.default_rng([seed, _stable_hash(track.name)])
    length = int(config.seconds * track.sample_rate)
    start = int(rng.integers(0, max(1, track.n_frames - length)))

    kept = [s for s in track.stems if rng.random() >= config.p_stem_dropped]
    if len(kept) < config.min_stems:
        kept = list(track.stems[: config.min_stems])

    rendered, chosen = [], []
    for stem in kept:
        audio = stem.load(start, length)
        gain_db = float(rng.uniform(*config.gain_db_range))
        records = []
        if config.effects:
            audio, records = apply_chain(audio, track.sample_rate, rng, DEFAULT_CHAIN)
        rendered.append(audio * db_to_gain(gain_db))
        chosen.append((stem, gain_db, records))

    stems = np.stack(rendered)
    mix = stems.sum(axis=0)
    peak = float(np.abs(mix).max())
    scale = config.peak_target / peak if peak > config.peak_target else 1.0
    mix, stems = (mix * scale).astype(np.float32), (stems * scale).astype(np.float32)

    start_s = start / track.sample_rate
    onsets = [
        audible_onsets(None if stem.onsets is None else stem.onsets(), samples, start_s, config, track.sample_rate)
        for (stem, _, _), samples in zip(chosen, stems)
    ]
    activity = frame_activity(stems, track.sample_rate, config.activity_fps, config.activity_threshold_db)
    stem_info = []
    for (stem, gain_db, records), samples, active, stem_onsets in zip(chosen, stems, activity, onsets):
        rms = float(np.sqrt((samples**2).mean()))
        stem_info.append(
            {
                "stem": stem.name,
                "class": stem.label,
                "source_class": stem.source_class,
                "detail": stem.detail,
                "ambiguous": stem.ambiguous,
                "is_drum": stem.is_drum,
                "gain_db": gain_db,
                "rms_db": float(20.0 * np.log10(rms + 1e-12)),
                "active": bool(active.any()),
                "onsets": stem_onsets,
                "effects": records,
            }
        )

    labels = {
        "schema": SCHEMA_VERSION,
        "mixer_version": __version__,
        "effects_backend": backend_version(),
        "dataset": track.dataset,
        "track": track.name,
        "seed": seed,
        "config": asdict(config),
        "sample_rate": track.sample_rate,
        "channels": int(mix.shape[1]),
        "seconds": config.seconds,
        "frames": int(mix.shape[0]),
        "start_frame": start,
        "start_seconds": start_s,
        "mix_scale": scale,
        "stems": stem_info,
        "activity": {
            "fps": config.activity_fps,
            "n_frames": int(activity.shape[1]),
            "per_stem": {
                info["stem"]: "".join("1" if a else "0" for a in row) for info, row in zip(stem_info, activity)
            },
        },
        "classes": sorted({i["class"] for i in stem_info if i["active"] and i["class"] is not None}),
        "ambiguous": any(i["ambiguous"] and i["active"] for i in stem_info),
        "times": onset_times(stem_info),
    }
    return Snippet(mix, track.sample_rate, stems, activity, labels)
