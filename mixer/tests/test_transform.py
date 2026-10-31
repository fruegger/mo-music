from pathlib import Path

import numpy as np
import pytest

from mixer import effects, transform
from mixer.mix import MixConfig, render_snippet
from mixer.sources import SAMPLE_RATE, Stem, Track, iter_tracks

pytest.importorskip("pedalboard")

SLAKH = Path(r"C:/workarea/slakh2100")


def tone(freq, seconds, amp=0.4):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, x], axis=1)


def peak_hz(audio):
    mono = audio.mean(axis=1)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return np.fft.rfftfreq(len(mono), 1 / SAMPLE_RATE)[spectrum.argmax()]


def click_stem(name, label, times, is_drum=False, seconds=30):
    audio = np.zeros((SAMPLE_RATE * seconds, 2), dtype=np.float32)
    for t in times:
        i = int(t * SAMPLE_RATE)
        audio[i : i + 2000] = 0.5
    return Stem(
        name, label, label, lambda s, k, a=audio: a[s : s + k], is_drum, "", lambda: np.array(sorted(times))
    )


def first_rises(audio, window_s=0.5):
    """Start times (s) of bursts in a mono click track: first sample above 20% of the peak after a quiet gap."""
    x = np.abs(audio).max(axis=1)
    above = x > 0.2 * x.max()
    gap = int(window_s * SAMPLE_RATE)
    starts, last = [], -gap
    for i in np.flatnonzero(above):
        if i - last >= gap:
            starts.append(i / SAMPLE_RATE)
        last = i
    return np.array(starts)


def test_stretch_changes_length_and_pitch_independently():
    audio = tone(440.0, 4.0)
    faster = transform.stretch(audio, SAMPLE_RATE, 1.25, 0.0, True)
    assert abs(len(faster) - len(audio) / 1.25) <= 2 and abs(peak_hz(faster) - 440.0) < 3
    shifted = transform.stretch(audio, SAMPLE_RATE, 1.0, 3.0, True)
    assert abs(len(shifted) - len(audio)) <= 2 and abs(peak_hz(shifted) - 440.0 * 2 ** (3 / 12)) < 4
    assert np.array_equal(transform.stretch(audio, SAMPLE_RATE, 1.0, 0.0, True), audio)


def test_fast_mode_is_within_its_documented_pitch_error():
    audio = tone(440.0, 4.0)
    for ratio, semitones in [(0.85, 0.0), (1.25, 0.0), (1.0, -3.0), (1.15, 2.0)]:
        out = transform.stretch(audio, SAMPLE_RATE, ratio, semitones, False)
        cents = 1200 * np.log2(peak_hz(out) / (440.0 * 2 ** (semitones / 12)))
        assert abs(cents) < 40, (ratio, semitones, cents)


def test_fit_trims_and_pads():
    audio = tone(100.0, 1.0)
    assert len(transform.fit(audio, 1000)) == 1000
    padded = transform.fit(audio[:500], 800)
    assert len(padded) == 800 and not padded[500:].any()


def test_tempo_changes_the_snippet_length_in_the_source_and_scales_the_onsets():
    times = [float(t) for t in range(1, 29, 2)]
    track = Track("synthetic", "TrackT", SAMPLE_RATE, SAMPLE_RATE * 30, [click_stem("S00", "Piano", times)])
    config = MixConfig(
        seconds=6.0, p_stem_dropped=0.0, effects=False, gain_db_range=(0.0, 0.0),
        p_tempo=1.0, tempo_ratio_range=(1.25, 1.25),
    )
    snippet = render_snippet(track, 4, config)
    labels = snippet.labels
    assert labels["transform"]["tempo_ratio"] == pytest.approx(1.25)
    assert labels["frames"] == 6 * SAMPLE_RATE and snippet.mix.shape[0] == 6 * SAMPLE_RATE
    assert labels["source_seconds"] == pytest.approx(7.5)

    start_s = labels["start_seconds"]
    expected = [round((t - start_s) / 1.25, 4) for t in times if start_s <= t < start_s + 7.5]
    assert labels["times"]["onsets"] == expected and expected
    # ... and the labels agree with where the clicks really are in the rendered audio.
    heard = first_rises(snippet.stems[0])
    assert len(heard) == len(expected)
    assert np.abs(heard - np.array(expected)).max() < 0.015


def test_all_stems_are_stretched_together():
    times = [float(t) for t in range(1, 29, 3)]
    track = Track(
        "synthetic", "TrackU", SAMPLE_RATE, SAMPLE_RATE * 30,
        [click_stem("S00", "Piano", times), click_stem("S01", "Bass", times)],
    )
    config = MixConfig(
        seconds=5.0, p_stem_dropped=0.0, effects=False, gain_db_range=(0.0, 0.0),
        p_tempo=1.0, tempo_ratio_range=(0.8, 0.8),
    )
    snippet = render_snippet(track, 1, config)
    a, b = (first_rises(s) for s in snippet.stems)
    assert len(a) == len(b) > 0 and np.abs(a - b).max() < 0.005


def test_pitch_skips_drums_unless_asked():
    track = Track(
        "synthetic", "TrackV", SAMPLE_RATE, SAMPLE_RATE * 30,
        [click_stem("S00", "Piano", [1.0]), click_stem("S01", "Drums", [1.0], is_drum=True)],
    )
    base = dict(seconds=2.0, p_stem_dropped=0.0, effects=False, p_pitch=1.0, pitch_semitones_range=(3.0, 3.0))
    by_stem = {s["stem"]: s["pitch_semitones"] for s in render_snippet(track, 0, MixConfig(**base)).labels["stems"]}
    assert by_stem == {"S00": 3.0, "S01": 0.0}
    with_drums = render_snippet(track, 0, MixConfig(pitch_drums=True, **base)).labels["stems"]
    assert {s["stem"]: s["pitch_semitones"] for s in with_drums} == {"S00": 3.0, "S01": 3.0}


def test_pitch_shift_moves_the_tone():
    track = Track("synthetic", "TrackW", SAMPLE_RATE, SAMPLE_RATE * 30, [Stem("S00", "Piano", "Piano", lambda s, k: tone(440.0, 30)[s : s + k])])
    config = MixConfig(
        seconds=3.0, p_stem_dropped=0.0, effects=False, gain_db_range=(0.0, 0.0),
        p_pitch=1.0, pitch_semitones_range=(-5.0, -5.0),
    )
    snippet = render_snippet(track, 2, config)
    assert snippet.labels["transform"]["pitch_semitones"] == -5.0
    assert abs(peak_hz(snippet.mix) - 440.0 * 2 ** (-5 / 12)) < 4


def test_transforms_leave_every_other_random_choice_alone():
    stems = [click_stem(f"S{i}", c, [1.0 + i]) for i, c in enumerate(["Piano", "Bass", "Guitar", "Organ"])]
    track = Track("synthetic", "TrackX", SAMPLE_RATE, SAMPLE_RATE * 30, stems)
    plain = MixConfig(seconds=2.0, p_stem_dropped=0.4)
    moved = MixConfig(seconds=2.0, p_stem_dropped=0.4, p_tempo=1.0, tempo_ratio_range=(1.2, 1.2), p_pitch=1.0)
    for seed in range(6):
        a, b = render_snippet(track, seed, plain).labels, render_snippet(track, seed, moved).labels
        assert [(s["stem"], s["gain_db"], s["effects"]) for s in a["stems"]] == [
            (s["stem"], s["gain_db"], s["effects"]) for s in b["stems"]
        ]
        assert b["transform"]["tempo_ratio"] == pytest.approx(1.2)


def test_label_file_reproduces_a_transformed_snippet():
    track = Track("synthetic", "TrackY", SAMPLE_RATE, SAMPLE_RATE * 30, [click_stem("S00", "Piano", [2.0, 5.0, 9.0])])
    config = MixConfig(seconds=3.0, p_tempo=1.0, p_pitch=1.0)
    first = render_snippet(track, 3, config)
    again = render_snippet(track, 3, MixConfig.from_dict(first.labels["config"]))
    assert np.array_equal(first.mix, again.mix) and first.labels == again.labels


def test_missing_backend_is_an_error_not_a_silent_skip(monkeypatch):
    track = Track("synthetic", "TrackZ", SAMPLE_RATE, SAMPLE_RATE * 30, [click_stem("S00", "Piano", [1.0])])
    monkeypatch.setattr(effects, "_pedalboard", lambda: None)
    with pytest.raises(RuntimeError, match="pedalboard"):
        render_snippet(track, 0, MixConfig(seconds=2.0, p_tempo=1.0))
    render_snippet(track, 0, MixConfig(seconds=2.0))  # transforms off: fine without it


@pytest.mark.skipif(not (SLAKH / "train").exists(), reason="Slakh2100 not available")
def test_slakh_drum_onsets_still_line_up_after_a_tempo_change():
    track = next(iter_tracks("slakh2100", SLAKH, "train"))
    config = MixConfig(
        seconds=10.0, effects=False, p_stem_dropped=0.0, gain_db_range=(0.0, 0.0),
        p_tempo=1.0, tempo_ratio_range=(0.85, 0.85),
    )
    snippet = render_snippet(track, 0, config)
    drum_idx = next(i for i, s in enumerate(snippet.labels["stems"]) if s["is_drum"])
    labelled = np.array(snippet.labels["stems"][drum_idx]["onsets"])
    audio = np.abs(snippet.stems[drum_idx]).max(axis=1)
    offsets = []
    for o in labelled:
        i0, i1 = int((o - 0.02) * SAMPLE_RATE), int((o + 0.2) * SAMPLE_RATE)
        if i0 < 0 or i1 >= len(audio):
            continue
        seg = audio[i0:i1]
        if seg.max() < 0.02 or np.median(seg[: int(0.01 * SAMPLE_RATE)]) > 0.1 * seg.max():
            continue
        offsets.append((i0 + int(np.argmax(seg > 0.2 * seg.max()))) / SAMPLE_RATE - o)
    assert len(offsets) > 20
    assert abs(float(np.median(offsets))) < 0.015
