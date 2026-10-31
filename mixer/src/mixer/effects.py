"""Per-stem effects with seeded parameter sampling.

Each effect draws its parameters from the rng it is given, returns them for
the label file, and applies them to a `(frames, channels)` float32 array.
The effects use `pedalboard` (install the `effects` extra) and are skipped
when it is missing; the rng is consumed the same way either way.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

Params = dict[str, float]


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


@dataclass(frozen=True)
class Effect:
    name: str
    probability: float  # chance a stem gets this effect
    sample: Callable[[np.random.Generator], Params]
    apply: Callable[[np.ndarray, int, Params], np.ndarray]


def _pedalboard():
    try:
        import pedalboard
    except ImportError:
        return None
    return pedalboard


def _board(make):
    def apply(audio, sr, params):
        pb = _pedalboard()
        board = pb.Pedalboard(make(pb, params))
        return board(audio.T, sr).T.astype(np.float32)

    return apply


EQ = Effect(
    "eq", 0.5,
    lambda r: {
        "freq_hz": float(np.exp(r.uniform(np.log(80), np.log(8000)))),
        "gain_db": float(r.uniform(-9, 9)),
        "q": float(r.uniform(0.5, 2.0)),
    },
    _board(lambda pb, p: [pb.PeakFilter(p["freq_hz"], p["gain_db"], p["q"])]),
)
COMPRESSOR = Effect(
    "compressor", 0.4,
    lambda r: {"threshold_db": float(r.uniform(-30, -8)), "ratio": float(r.uniform(2, 8))},
    _board(lambda pb, p: [pb.Compressor(threshold_db=p["threshold_db"], ratio=p["ratio"])]),
)
DISTORTION = Effect(
    "distortion", 0.15,
    lambda r: {"drive_db": float(r.uniform(5, 30))},
    _board(lambda pb, p: [pb.Distortion(drive_db=p["drive_db"])]),
)
ECHO = Effect(
    "echo", 0.15,
    lambda r: {
        "delay_s": float(r.uniform(0.1, 0.5)),
        "feedback": float(r.uniform(0.1, 0.5)),
        "mix": float(r.uniform(0.1, 0.4)),
    },
    _board(lambda pb, p: [pb.Delay(delay_seconds=p["delay_s"], feedback=p["feedback"], mix=p["mix"])]),
)
REVERB = Effect(
    "reverb", 0.3,
    lambda r: {"room_size": float(r.uniform(0.1, 0.9)), "wet_level": float(r.uniform(0.05, 0.4))},
    _board(lambda pb, p: [pb.Reverb(room_size=p["room_size"], wet_level=p["wet_level"], dry_level=1.0)]),
)

DEFAULT_CHAIN = [EQ, COMPRESSOR, DISTORTION, ECHO, REVERB]


def apply_chain(
    audio: np.ndarray, sr: int, rng: np.random.Generator, chain=DEFAULT_CHAIN
) -> tuple[np.ndarray, list[dict]]:
    """Applies each effect of `chain` that fires, in order. Returns the audio
    and one record per effect: `fired` is the random draw, `applied` whether it
    actually ran (it does not when pedalboard is missing). The rng is consumed
    identically either way, so seeds stay comparable across machines."""
    records = []
    have_backend = _pedalboard() is not None
    for effect in chain:
        fired = bool(rng.random() < effect.probability)
        params = effect.sample(rng)
        applied = fired and have_backend
        if applied:
            audio = effect.apply(audio, sr, params)
        records.append({"name": effect.name, "fired": fired, "applied": applied, "params": params})
    return audio, records


def backend_version() -> str | None:
    pb = _pedalboard()
    return None if pb is None else getattr(pb, "__version__", "unknown")
