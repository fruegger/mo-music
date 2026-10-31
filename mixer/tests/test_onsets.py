from pathlib import Path

import mido
import numpy as np
import pytest

from mixer import CLASSES
from mixer.mix import MixConfig, render_snippet
from mixer.sources import SAMPLE_RATE, Stem, Track, iter_tracks, midi_onsets
from mixer.targets import onset_target

SLAKH = Path(r"C:/workarea/slakh2100")


def click_stem(name, label, times, seconds=20, onsets=True):
    """A silent stem with a short loud burst at each time in `times`."""
    n = SAMPLE_RATE * seconds
    audio = np.zeros((n, 2), dtype=np.float32)
    for t in times:
        i = int(t * SAMPLE_RATE)
        audio[i : i + 2000] = 0.5
    return Stem(
        name, label, label or "other", lambda s, k, a=audio: a[s : s + k], False, "",
        (lambda: np.array(sorted(times))) if onsets else None,
    )


def render(stems, seed=0, **config):
    track = Track("synthetic", "TrackOnsets", SAMPLE_RATE, SAMPLE_RATE * 20, stems)
    cfg = MixConfig(seconds=4.0, p_stem_dropped=0.0, effects=False, gain_db_range=(0.0, 0.0), **config)
    return render_snippet(track, seed, cfg)


def test_midi_onsets_follow_the_tempo_map(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))  # 120 bpm: a beat is 0.5 s
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))  # 0.0 s
    track.append(mido.Message("note_on", note=60, velocity=0, time=480))  # note-off, not an onset
    track.append(mido.Message("note_on", note=62, velocity=100, time=480))  # 1.0 s
    track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))  # 60 bpm: a beat is 1 s
    track.append(mido.Message("note_on", note=64, velocity=100, time=480))  # 2.0 s
    path = tmp_path / "t.mid"
    mid.save(path)
    assert np.allclose(midi_onsets(path), [0.0, 1.0, 2.0])


def test_onsets_are_relative_to_the_snippet_start_and_inside_the_crop():
    stem = click_stem("S00", "Piano", [0.5, 3.0, 7.25, 9.0, 12.5, 16.0])
    snippet = render([stem])
    start_s = snippet.labels["start_seconds"]
    expected = [round(t - start_s, 4) for t in [0.5, 3.0, 7.25, 9.0, 12.5, 16.0] if start_s <= t < start_s + 4.0]
    assert snippet.labels["times"]["onsets"] == expected
    assert snippet.labels["stems"][0]["onsets"] == expected
    assert snippet.labels["times"]["onsets_by_class"] == ({"Piano": expected} if expected else {})
    assert all(0.0 <= o < 4.0 for o in expected)


def test_seeds_that_cover_an_onset_find_it():
    stem = click_stem("S00", "Piano", [float(t) for t in range(0, 20)])  # one per second
    found = {len(render([stem], seed).labels["times"]["onsets"]) for seed in range(8)}
    assert found <= {3, 4} and found  # 4 s crop of 1 s spacing


def test_inaudible_onsets_are_dropped():
    # Onsets are listed (MIDI says a note starts) but the stem is silent there.
    n = SAMPLE_RATE * 20
    silent = Stem("S00", "Bass", "Bass", lambda s, k: np.zeros((k, 2), dtype=np.float32), False, "",
                  lambda: np.arange(0.0, 20.0, 0.5))
    loud = click_stem("S01", "Piano", [float(t) for t in range(20)])
    snippet = render([silent, loud])
    bass = next(s for s in snippet.labels["stems"] if s["stem"] == "S00")
    assert bass["onsets"] == []
    assert "Bass" not in snippet.labels["times"]["onsets_by_class"]
    assert snippet.labels["times"]["onsets"] == next(s for s in snippet.labels["stems"] if s["stem"] == "S01")["onsets"]


def test_missing_onset_data_marks_the_snippet_incomplete():
    with_onsets = click_stem("S00", "Piano", [1.0, 5.0, 9.0, 13.0, 17.0])
    without = click_stem("S01", "Guitar", [2.0, 6.0, 10.0, 14.0, 18.0], onsets=False)
    labels = render([with_onsets, without]).labels
    assert labels["times"]["onsets_complete"] is False
    assert next(s for s in labels["stems"] if s["stem"] == "S01")["onsets"] is None
    assert render([with_onsets]).labels["times"]["onsets_complete"] is True


def test_onset_target_frames():
    labels = {"seconds": 4.0, "times": {"onsets": [0.0, 1.234, 3.999], "onsets_by_class": {"Piano": [1.234], "Drums": [0.0, 3.999]}}}
    flat = onset_target(labels, fps=100)
    assert flat.shape == (400,) and set(np.flatnonzero(flat)) == {0, 123, 399}
    by_class = onset_target(labels, fps=100, by_class=True)
    assert by_class.shape == (len(CLASSES), 400)
    assert set(np.flatnonzero(by_class[CLASSES.index("Piano")])) == {123}
    assert set(np.flatnonzero(by_class[CLASSES.index("Drums")])) == {0, 399}


@pytest.mark.skipif(not (SLAKH / "train").exists(), reason="Slakh2100 not available")
def test_slakh_midi_onsets_line_up_with_the_audio():
    """Isolated drum hits in the first Slakh track start within 5 ms of their MIDI time."""
    track = next(iter_tracks("slakh2100", SLAKH, "train"))
    drums = next(s for s in track.stems if s.is_drum)
    onsets = drums.onsets()
    audio = np.abs(drums.load(0, track.n_frames)).max(axis=1)
    offsets = []
    for o in onsets[:300]:
        i0, i1 = int((o - 0.02) * SAMPLE_RATE), int((o + 0.2) * SAMPLE_RATE)
        if i0 < 0 or i1 >= len(audio):
            continue
        seg = audio[i0:i1]
        if seg.max() < 0.02 or np.median(seg[: int(0.01 * SAMPLE_RATE)]) > 0.1 * seg.max():
            continue  # silent, or another hit is still ringing
        offsets.append((i0 + int(np.argmax(seg > 0.2 * seg.max()))) / SAMPLE_RATE - o)
    assert len(offsets) > 100
    assert abs(float(np.median(offsets))) < 0.005


@pytest.mark.skipif(not (SLAKH / "train").exists(), reason="Slakh2100 not available")
def test_slakh_snippet_has_onsets_for_active_stems():
    track = next(iter_tracks("slakh2100", SLAKH, "train"))
    labels = render_snippet(track, 0, MixConfig(effects=False)).labels
    assert labels["times"]["onsets_complete"] is True
    assert len(labels["times"]["onsets"]) > 10
    assert all(0.0 <= o < labels["seconds"] for o in labels["times"]["onsets"])
    for info in labels["stems"]:
        if info["active"] and info["class"]:
            assert info["class"] in labels["times"]["onsets_by_class"] or info["onsets"] == []
