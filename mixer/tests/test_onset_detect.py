from pathlib import Path

import numpy as np
import pytest

from mixer.onset_detect import detect_onsets
from mixer.sources import SAMPLE_RATE, iter_tracks

SLAKH = Path(r"C:/workarea/slakh2100")


def notes(times, seconds=12.0, freq=220.0, decay=6.0):
    """Plucked tones starting at `times`: a sharp attack and an exponential decay."""
    n = int(seconds * SAMPLE_RATE)
    out = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / SAMPLE_RATE
    for start in times:
        i = int(start * SAMPLE_RATE)
        env = np.exp(-decay * t[: n - i])
        out[i:] += (0.5 * env * np.sin(2 * np.pi * freq * t[: n - i])).astype(np.float32)
    return np.stack([out, out], axis=1)


def match(ref, est, tol=0.05):
    est = list(est)
    hits = 0
    for r in ref:
        close = [e for e in est if abs(e - r) <= tol]
        if close:
            est.remove(min(close, key=lambda e: abs(e - r)))
            hits += 1
    return hits


def test_finds_isolated_notes_in_time():
    times = [0.8, 2.1, 3.0, 4.6, 6.2, 7.0, 8.9, 10.3]
    est = detect_onsets(notes(times), SAMPLE_RATE)
    assert match(times, est) == len(times)
    assert len(est) <= len(times) + 1  # no storm of false onsets


def test_silence_has_no_onsets():
    assert len(detect_onsets(np.zeros((SAMPLE_RATE * 5, 2), dtype=np.float32), SAMPLE_RATE)) == 0


def test_a_steady_tone_has_no_onsets():
    t = np.arange(SAMPLE_RATE * 6) / SAMPLE_RATE
    steady = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    assert len(detect_onsets(np.stack([steady, steady], axis=1), SAMPLE_RATE)) == 0


def test_is_deterministic_and_sorted():
    audio = notes([1.0, 2.5, 4.0, 5.5])
    a, b = detect_onsets(audio, SAMPLE_RATE), detect_onsets(audio, SAMPLE_RATE)
    assert np.array_equal(a, b) and np.all(np.diff(a) > 0)


@pytest.mark.skipif(not (SLAKH / "validation").exists(), reason="Slakh2100 not available")
def test_accuracy_against_slakh_midi_on_held_out_stems():
    """Bass and drums of the first validation track, +-50 ms: documents the
    accuracy of the estimates (see README for the larger measurement)."""
    track = next(iter_tracks("slakh2100", SLAKH, "validation"))
    tp = n_ref = n_est = 0
    for stem in track.stems:
        if stem.label not in ("Bass", "Drums") or stem.onsets is None:
            continue
        ref = stem.onsets()
        est = detect_onsets(stem.load(0, track.n_frames), track.sample_rate)
        tp += match(ref.tolist(), est.tolist())
        n_ref += len(ref)
        n_est += len(est)
    assert n_ref > 100
    precision, recall = tp / max(n_est, 1), tp / n_ref
    assert 2 * precision * recall / max(precision + recall, 1e-9) > 0.6
