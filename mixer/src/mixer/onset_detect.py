"""Note-onset estimation from audio energy, for stems that come without MIDI.

Spectral flux on a log-compressed magnitude spectrogram with a max-filtered
reference frame ("SuperFlux", which ignores vibrato), picked against an
adaptive threshold. Onsets are *estimates*: `tests/test_onset_detect.py` and the
README give the accuracy measured against Slakh2100's MIDI.
"""

import numpy as np

N_FFT = 2048
HOP_S = 0.010  # frame hop in seconds
MAX_HZ = 10000.0
COMPRESSION = 1000.0  # log(1 + COMPRESSION * magnitude)
MEDIAN_SPAN_S = 0.5  # half-width of the moving median the threshold follows
DELTA = 1.5  # threshold = median + DELTA * (global 90th percentile of the flux); tuned on Slakh train
MIN_GAP_S = 0.03
FLOOR_FRACTION = 0.05  # never accept a peak below this share of the strongest flux (sparse signals have p90 = 0)
# Spectral flux peaks when the analysis window overlaps the attack; this shift
# (measured on Slakh drums) puts the estimate on the attack itself.
TIMING_OFFSET_S = -0.006


def _log_spectrogram(mono: np.ndarray, sample_rate: int, hop: int) -> np.ndarray:
    window = np.hanning(N_FFT).astype(np.float32)
    n_bins = int(MAX_HZ / (sample_rate / N_FFT))
    padded = np.pad(mono, (N_FFT // 2, N_FFT // 2), mode="reflect")  # zeros would make the signal ends look like onsets
    n_frames = 1 + (len(padded) - N_FFT) // hop
    out = np.empty((n_frames, n_bins), dtype=np.float32)
    block = 2048
    for lo in range(0, n_frames, block):
        hi = min(n_frames, lo + block)
        idx = (np.arange(lo, hi) * hop)[:, None] + np.arange(N_FFT)[None, :]
        mag = np.abs(np.fft.rfft(padded[idx] * window, axis=1))[:, :n_bins]
        out[lo:hi] = np.log1p(COMPRESSION * mag / N_FFT)
    return out


def _moving_median(x: np.ndarray, half: int) -> np.ndarray:
    padded = np.pad(x, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(windows, axis=1)


def spectral_flux(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Returns (flux per frame, hop in samples)."""
    hop = int(round(HOP_S * sample_rate))
    spec = _log_spectrogram(mono.astype(np.float32), sample_rate, hop)
    ref = spec[:-1]
    ref = np.maximum(np.maximum(ref, np.pad(ref, ((0, 0), (1, 0)))[:, :-1]), np.pad(ref, ((0, 0), (0, 1)))[:, 1:])
    flux = np.maximum(spec[1:] - ref, 0.0).sum(axis=1)
    return np.concatenate([[0.0], flux]), hop


def pick_onsets(flux: np.ndarray, hop: int, sample_rate: int, delta: float = DELTA) -> np.ndarray:
    """Peaks of the flux above the adaptive threshold that are the largest
    within `MIN_GAP_S`, as times in seconds."""
    if not flux.any():
        return np.zeros(0)
    half = max(1, int(MEDIAN_SPAN_S * sample_rate / hop))
    threshold = np.maximum(
        _moving_median(flux, half) + delta * np.percentile(flux, 90), FLOOR_FRACTION * flux.max()
    )
    gap = max(1, int(MIN_GAP_S * sample_rate / hop))
    padded = np.pad(flux, gap, constant_values=0.0)
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * gap + 1).max(axis=1)
    frames = np.flatnonzero((flux >= local_max) & (flux > threshold))
    times = frames.astype(np.float64) * hop / sample_rate + TIMING_OFFSET_S
    return times[times >= 0.0]


def detect_onsets(audio: np.ndarray, sample_rate: int, delta: float = DELTA) -> np.ndarray:
    """Onset times in seconds of `(frames, channels)` audio, sorted."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    flux, hop = spectral_flux(mono, sample_rate)
    times = pick_onsets(flux, hop, sample_rate, delta)
    return times[times < len(mono) / sample_rate - MIN_GAP_S]  # the very end is padding, not a note
