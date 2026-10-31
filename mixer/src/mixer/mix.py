"""Snippet rendering: crop, per-stem effects and gain, sum, label."""

from dataclasses import dataclass, field

import numpy as np

from .effects import DEFAULT_CHAIN, apply_chain, db_to_gain
from .sources import Track


@dataclass(frozen=True)
class MixConfig:
    seconds: float = 8.0
    gain_db_range: tuple[float, float] = (-12.0, 6.0)  # relative stem volume
    p_stem_dropped: float = 0.15  # chance a stem is left out of the mix
    min_stems: int = 1
    effects: bool = True
    peak_target: float = 0.95  # the final mix is scaled down to this peak if it clips


@dataclass
class Snippet:
    mix: np.ndarray  # (frames, channels) float32
    sample_rate: int
    stems: np.ndarray  # (n_stems_in_mix, frames, channels): each stem as heard in the mix
    labels: dict = field(default_factory=dict)


def _stable_hash(text: str) -> int:
    """Python's `hash()` is salted per process; seeds must not be (FNV-1a)."""
    h = 2166136261
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * 16777619) % 2**32
    return h


def render_snippet(track: Track, seed: int, config: MixConfig = MixConfig()) -> Snippet:
    """Deterministic in (track, seed, config)."""
    rng = np.random.default_rng([seed, _stable_hash(track.name)])
    length = int(config.seconds * track.sample_rate)
    start = int(rng.integers(0, max(1, track.n_frames - length)))

    kept = [s for s in track.stems if rng.random() >= config.p_stem_dropped]
    if len(kept) < config.min_stems:
        kept = list(track.stems[: config.min_stems])

    rendered, stem_info = [], []
    for stem in kept:
        audio = stem.load(start, length)
        gain_db = float(rng.uniform(*config.gain_db_range))
        used = {}
        if config.effects:
            audio, used = apply_chain(audio, track.sample_rate, rng, DEFAULT_CHAIN)
        rendered.append(audio * db_to_gain(gain_db))
        stem_info.append(
            {"stem": stem.name, "class": stem.label, "is_drum": stem.is_drum, "gain_db": gain_db, "effects": used}
        )

    stems = np.stack(rendered)
    mix = stems.sum(axis=0)
    peak = float(np.abs(mix).max())
    scale = config.peak_target / peak if peak > config.peak_target else 1.0

    labels = {
        "dataset": track.dataset,
        "track": track.name,
        "seed": seed,
        "start_frame": start,
        "classes": sorted({s["class"] for s in stem_info}),
        "stems": stem_info,
        "mix_scale": scale,
    }
    return Snippet((mix * scale).astype(np.float32), track.sample_rate, (stems * scale).astype(np.float32), labels)
