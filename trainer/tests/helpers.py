"""Synthetic multitrack material for the trainer tests."""

import numpy as np
from mixer.sources import SAMPLE_RATE, Stem, Track


def tone(freq: float, seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, 0.8 * x], axis=1)


def clicks(period: float, seconds: float) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    x = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    for start in np.arange(0.1, seconds - 0.1, period):
        i = int(start * SAMPLE_RATE)
        x[i : i + 2000] = (0.5 * np.exp(-np.arange(2000) / 300) * rng.standard_normal(2000)).astype(np.float32)
    return np.stack([x, x], axis=1)


def make_stem(name, label, audio, source_class=None, is_drum=False):
    return Stem(name, label, source_class or (label or "other"), lambda s, k, a=audio: a[s : s + k], is_drum)


def synthetic_track(name="TrackA", seconds=20, with_unknown=False) -> Track:
    stems = [
        make_stem("S0", "Bass", tone(80, seconds)),
        make_stem("S1", "Drums", clicks(0.5, seconds), is_drum=True),
        make_stem("S2", "Piano", tone(440, seconds)),
        make_stem("S3", "Guitar", tone(660, seconds)),
    ]
    if with_unknown:
        stems.append(make_stem("S4", None, tone(330, seconds), source_class="other"))
    return Track("synthetic", name, SAMPLE_RATE, SAMPLE_RATE * seconds, stems)
