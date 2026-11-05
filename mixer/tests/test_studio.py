import json

import numpy as np
import pytest
import soundfile as sf
from mixer.mix import MixConfig, render_snippet
from mixer.sources import SAMPLE_RATE, iter_tracks, prepare_studio, studio_label


def tone(freq, seconds, channels):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return x if channels == 1 else np.stack([x, 0.5 * x], axis=1)


@pytest.fixture
def source(tmp_path):
    """Two songs of bounced stems: mono and stereo files, slightly different lengths, one stem to exclude."""
    for song, extra in (("Song A", ["piano"]), ("Song B", ["keys"])):
        folder = tmp_path / "src" / song / "tracks"
        folder.mkdir(parents=True)
        sf.write(folder / "bass.wav", tone(80, 3.0, 1), SAMPLE_RATE)
        sf.write(folder / "drums.wav", tone(150, 2.9, 2), SAMPLE_RATE)
        sf.write(folder / "guit1.flac", tone(330, 3.0, 2), SAMPLE_RATE)
        sf.write(folder / "voc1.wav", tone(440, 3.0, 1), SAMPLE_RATE)
        sf.write(folder / "original_timewarp.wav", tone(100, 3.0, 2), SAMPLE_RATE)
        for name in extra:
            sf.write(folder / f"{name}.wav", tone(550, 3.0, 2), SAMPLE_RATE)
    (tmp_path / "src" / "Notes").mkdir()  # a folder without tracks/ is not a song
    return tmp_path / "src"


def test_labels_come_from_the_stem_names():
    assert studio_label("bass") == ("Bass", "bass", False)
    assert studio_label("robo_drum")[0] == "Drums" and studio_label("robo_drum")[2] is True
    names = ("guit1", "guit", "robo_guit", "piano", "organ", "voc2", "keys", "original_timewarp")
    assert [studio_label(n)[0] for n in names] == ["Guitar", "Guitar", "Guitar", "Piano", "Organ", "Vocals", None, None]


def test_prepare_writes_aligned_stereo_stems_with_classes(source, tmp_path):
    dst = tmp_path / "dst"
    assert prepare_studio(source, dst, test_songs=("Song B",)) == 2
    assert (dst / "train" / "Song A").is_dir() and (dst / "test" / "Song B").is_dir()
    meta = json.loads((dst / "train" / "Song A" / "stems.json").read_text())
    assert [s["name"] for s in meta["stems"]] == ["bass", "drums", "guit1", "piano", "voc1"]  # the timewarp stem is excluded
    assert [s["label"] for s in meta["stems"]] == ["Bass", "Drums", "Guitar", "Piano", "Vocals"]
    assert meta["frames"] == int(2.9 * SAMPLE_RATE)  # cut to the shortest stem
    for s in meta["stems"]:
        data, rate = sf.read(dst / "train" / "Song A" / s["file"])
        assert rate == SAMPLE_RATE and data.shape == (meta["frames"], 2)  # mono duplicated, all the same length
    keys = json.loads((dst / "test" / "Song B" / "stems.json").read_text())
    assert {s["name"]: s["label"] for s in keys["stems"]}["keys"] is None  # a mixture of keyboard sounds has no class


def test_a_finished_song_is_skipped_unless_forced_and_the_rate_is_checked(source, tmp_path):
    dst = tmp_path / "dst"
    assert prepare_studio(source, dst) == 2
    assert prepare_studio(source, dst) == 0
    assert prepare_studio(source, dst, force=True) == 2
    sf.write(source / "Song A" / "tracks" / "bass.wav", tone(80, 1.0, 1), 22050)
    with pytest.raises(ValueError, match="sample rate 22050"):
        prepare_studio(source, tmp_path / "other")


def test_the_loader_yields_tracks_the_mixer_can_render(source, tmp_path):
    dst = tmp_path / "dst"
    prepare_studio(source, dst, test_songs=("Song B",))
    tracks = list(iter_tracks("studio", dst, "train"))
    assert [t.name for t in tracks] == ["Song A"] and tracks[0].dataset == "studio"
    assert [list(iter_tracks("studio", dst, "test"))[0].name] == ["Song B"]
    snippet = render_snippet(tracks[0], 0, MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False))
    assert snippet.mix.shape == (SAMPLE_RATE, 2) and snippet.labels["classes"] == ["Bass", "Drums", "Guitar", "Piano", "Vocals"]
    assert np.allclose(snippet.stems.sum(axis=0), snippet.mix, atol=1e-5)  # the stems add up to the mix
