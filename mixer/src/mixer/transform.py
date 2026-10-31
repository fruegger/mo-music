"""Tempo change and pitch shift of a stem (Rubber Band, via pedalboard)."""

import numpy as np

from . import effects


def available() -> bool:
    return effects._pedalboard() is not None


def stretch(audio: np.ndarray, sample_rate: int, ratio: float, semitones: float, high_quality: bool) -> np.ndarray:
    """`audio` is `(frames, channels)`. `ratio` is the speed factor (1.1 = 10%
    faster and shorter); `semitones` shifts the pitch independently of it.
    Output length is `round(frames / ratio)`, give or take a frame."""
    if ratio == 1.0 and semitones == 0.0:
        return audio
    if not audio.any():  # a stem that is silent in this crop stays silent: skip the expensive part
        return np.zeros((int(round(len(audio) / ratio)), audio.shape[1]), dtype=np.float32)
    pb = effects._pedalboard()
    out = pb.time_stretch(
        np.ascontiguousarray(audio.T, dtype=np.float32),
        sample_rate,
        stretch_factor=ratio,
        pitch_shift_in_semitones=semitones,
        high_quality=high_quality,
    )
    return out.T.astype(np.float32)


def fit(audio: np.ndarray, length: int) -> np.ndarray:
    """Trims or zero-pads to exactly `length` frames."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, ((0, length - len(audio)), (0, 0)))
