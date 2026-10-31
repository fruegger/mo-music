"""Training targets derived from a label file."""

import numpy as np

from .vocab import CLASSES


def onset_target(labels: dict, fps: int = 100, by_class: bool = False) -> np.ndarray:
    """Binary onset targets on a frame grid: `(n_frames,)` for any onset, or
    `(len(CLASSES), n_frames)` with `by_class`. A frame is 1 if an onset falls
    inside it. Drum onsets are dense, so train with tolerance (for example by
    widening the targets by one frame) rather than a hard 1-frame target."""
    n_frames = int(round(labels["seconds"] * fps))
    times = labels["times"]
    if not by_class:
        out = np.zeros(n_frames, dtype=np.float32)
        for o in times["onsets"]:
            out[min(int(o * fps), n_frames - 1)] = 1.0
        return out
    out = np.zeros((len(CLASSES), n_frames), dtype=np.float32)
    for name, onsets in times["onsets_by_class"].items():
        for o in onsets:
            out[CLASSES.index(name), min(int(o * fps), n_frames - 1)] = 1.0
    return out
