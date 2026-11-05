import json

import numpy as np
import pytest
import soundfile as sf
from mixer.mix import MixConfig, render_snippet
from mixer.sources import SAMPLE_RATE, iter_tracks, prepare_studio, studio_label, studio_mixes


def tone(freq, seconds, channels):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return x if channels == 1 else np.stack([x, 0.5 * x], axis=1)


@pytest.fixture
def source(tmp_path):
    """Two songs of bounced stems: mono and stereo files, slightly different lengths, one demo stem to exclude, and mixes."""
    for song, extra, mixes in (
        ("Song A", ["piano"], {"master": 3.2, "mixdown_no_piano": 3.0}),
        ("Song B", ["keys"], {"karaoke": 3.0, "mixdown no keys": 3.0}),
    ):
        folder = tmp_path / "src" / song / "tracks"
        folder.mkdir(parents=True)
        sf.write(folder / "bass.wav", tone(80, 3.0, 1), SAMPLE_RATE)
        sf.write(folder / "drums.wav", tone(150, 2.9, 2), SAMPLE_RATE)
        sf.write(folder / "guit1.flac", tone(330, 3.0, 2), SAMPLE_RATE)
        sf.write(folder / "voc1.wav", tone(440, 3.0, 1), SAMPLE_RATE)
        sf.write(folder / "original_timewarp.wav", tone(100, 3.0, 2), SAMPLE_RATE)  # a demo take: vocals and guitar together
        for name in extra:
            sf.write(folder / f"{name}.wav", tone(550, 3.0, 2), SAMPLE_RATE)
        for name, seconds in mixes.items():
            (folder.parent / "mixes").mkdir(exist_ok=True)
            sf.write(folder.parent / "mixes" / f"{name}.wav", tone(200, seconds, 1 if name == "karaoke" else 2), SAMPLE_RATE)
    (tmp_path / "src" / "Notes").mkdir()  # a folder without tracks/ is not a song
    return tmp_path / "src"


def stems_json(dst, split, song):
    return json.loads((dst / split / song / "stems.json").read_text())


def test_labels_come_from_the_stem_names():
    assert studio_label("bass") == ("Bass", "bass", False)
    assert studio_label("robo_drum")[0] == "Drums" and studio_label("robo_drum")[2] is True
    names = ("guit1", "guit", "robo_guit", "piano", "organ", "voc2", "keys", "original_timewarp")
    assert [studio_label(n)[0] for n in names] == ["Guitar", "Guitar", "Guitar", "Piano", "Organ", "Vocals", None, None]


def test_prepare_writes_aligned_stereo_stems_with_classes(source, tmp_path):
    dst = tmp_path / "dst"
    assert prepare_studio(source, dst, test_songs=("Song B",), exclude=("original_timewarp",)) == 2
    assert (dst / "train" / "Song A").is_dir() and (dst / "test" / "Song B").is_dir()
    meta = stems_json(dst, "train", "Song A")
    assert [s["name"] for s in meta["stems"]] == ["bass", "drums", "guit1", "piano", "voc1"]  # the demo take is left out
    assert meta["excluded"] == ["original_timewarp"]  # and the file says so
    assert [s["label"] for s in meta["stems"]] == ["Bass", "Drums", "Guitar", "Piano", "Vocals"]
    assert meta["frames"] == int(2.9 * SAMPLE_RATE)  # cut to the shortest stem
    for s in meta["stems"]:
        data, rate = sf.read(dst / "train" / "Song A" / s["file"])
        assert rate == SAMPLE_RATE and data.shape == (meta["frames"], 2)  # mono duplicated, all the same length
    keys = stems_json(dst, "test", "Song B")
    assert {s["name"]: s["label"] for s in keys["stems"]}["keys"] is None  # a mixture of keyboard sounds has no class


def test_nothing_is_excluded_unless_asked(source, tmp_path):
    prepare_studio(source, tmp_path / "dst")
    meta = stems_json(tmp_path / "dst", "train", "Song A")
    assert "original_timewarp" in [s["name"] for s in meta["stems"]] and meta["excluded"] == []


def test_mixes_are_copied_as_stereo_uncut_and_know_what_they_lack(source, tmp_path):
    dst = tmp_path / "dst"
    prepare_studio(source, dst, test_songs=("Song B",), exclude=("original_timewarp",))
    a = {m["name"]: m for m in stems_json(dst, "train", "Song A")["mixes"]}
    assert a["master"]["without"] is None  # unknown unless the name says so
    assert a["mixdown_no_piano"]["without"] == ["piano"]
    assert a["master"]["frames"] == int(3.2 * SAMPLE_RATE)  # not cut to the stems' length
    b = {m["name"]: m for m in stems_json(dst, "test", "Song B")["mixes"]}
    assert b["mixdown no keys"]["without"] == ["keys"] and b["karaoke"]["without"] is None
    data, rate = sf.read(dst / "test" / "Song B" / b["karaoke"]["file"])
    assert rate == SAMPLE_RATE and data.shape[1] == 2  # the mono karaoke mix is stereo now
    listed = [(song, info["name"], path.exists()) for song, info, path in studio_mixes(dst, "train")]
    assert listed == [("Song A", "master", True), ("Song A", "mixdown_no_piano", True)]


def test_labels_can_be_overridden_and_must_be_spelt_right(source, tmp_path):
    dst = tmp_path / "dst"
    prepare_studio(source, dst, labels={"Song B/keys": "Strings", "Song A/voc1": None})
    assert {s["name"]: s["label"] for s in stems_json(dst, "train", "Song B")["stems"]}["keys"] == "Strings"
    assert {s["name"]: s["label"] for s in stems_json(dst, "train", "Song A")["stems"]}["voc1"] is None
    with pytest.raises(ValueError, match="unknown class 'strings'"):
        prepare_studio(source, tmp_path / "other", labels={"Song B/keys": "strings"})
    assert not (tmp_path / "other").exists()  # refused before anything was written


def test_a_hand_edited_misspelt_class_is_refused_by_the_loader(source, tmp_path):
    dst = tmp_path / "dst"
    prepare_studio(source, dst)
    path = dst / "train" / "Song A" / "stems.json"
    meta = json.loads(path.read_text())
    meta["stems"][1]["label"] = "drums"  # should be "Drums": it would have been read as "other" silently
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unknown class 'drums'"):
        list(iter_tracks("studio", dst, "train"))


def test_a_finished_song_is_skipped_unless_forced_and_the_rate_is_checked(source, tmp_path):
    dst = tmp_path / "dst"
    assert prepare_studio(source, dst) == 2
    assert prepare_studio(source, dst) == 0
    assert prepare_studio(source, dst, force=True) == 2
    sf.write(source / "Song A" / "tracks" / "bass.wav", tone(80, 1.0, 1), 22050)
    with pytest.raises(ValueError, match="sample rate 22050"):
        prepare_studio(source, tmp_path / "other")


def test_an_older_preparation_gets_its_mixes_added_and_keeps_the_hand_edits(source, tmp_path):
    dst = tmp_path / "dst"
    prepare_studio(source, dst)
    path = dst / "train" / "Song B" / "stems.json"
    meta = json.loads(path.read_text())
    del meta["mixes"]  # how the first version of the dataset looked
    meta["stems"][-1]["label"] = "Piano"  # a hand edit
    path.write_text(json.dumps(meta))
    assert prepare_studio(source, dst) == 0  # no song is redone ...
    after = json.loads(path.read_text())
    assert [m["name"] for m in after["mixes"]] == ["karaoke", "mixdown no keys"]  # ... but the mixes are added
    assert after["stems"][-1]["label"] == "Piano"  # and the edit is still there


def test_the_loader_yields_tracks_the_mixer_can_render(source, tmp_path):
    dst = tmp_path / "dst"
    prepare_studio(source, dst, test_songs=("Song B",), exclude=("original_timewarp",))
    tracks = list(iter_tracks("studio", dst, "train"))
    assert [t.name for t in tracks] == ["Song A"] and tracks[0].dataset == "studio"
    assert [list(iter_tracks("studio", dst, "test"))[0].name] == ["Song B"]
    snippet = render_snippet(tracks[0], 0, MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False))
    assert snippet.mix.shape == (SAMPLE_RATE, 2) and snippet.labels["classes"] == ["Bass", "Drums", "Guitar", "Piano", "Vocals"]
    assert np.allclose(snippet.stems.sum(axis=0), snippet.mix, atol=1e-5)  # the stems add up to the mix
