import numpy as np

from mixer import effects
from mixer.mix import SCHEMA_VERSION, MixConfig, render_snippet
from mixer.registry import load_registry
from mixer.sources import SAMPLE_RATE, Stem, Track
from mixer.vocab import CLASSES, MUSDB_CLASSES, SLAKH_CLASSES


def tone(freq, n, amp=0.3):
    t = amp * np.sin(2 * np.pi * freq * np.arange(n) / SAMPLE_RATE).astype(np.float32)
    return np.stack([t, t], axis=1)


def make_stem(name, label, audio, source_class=None, is_drum=False):
    return Stem(name, label, source_class or (label or "other"), lambda s, k, a=audio: a[s : s + k], is_drum)


def synthetic_track(n_stems=4, seconds=20):
    n = SAMPLE_RATE * seconds
    labels = ["Bass", "Piano", "Drums", "Guitar"]
    stems = [make_stem(f"S{i:02d}", labels[i], tone(110 * (i + 1), n), is_drum=i == 2) for i in range(n_stems)]
    return Track("synthetic", "TrackA", SAMPLE_RATE, n, stems)


FAST = MixConfig(seconds=2.0)


def test_same_seed_is_bit_identical():
    track = synthetic_track()
    a, b = render_snippet(track, 7, FAST), render_snippet(track, 7, FAST)
    assert np.array_equal(a.mix, b.mix)
    assert a.labels == b.labels


def test_different_seeds_differ():
    track = synthetic_track()
    assert not np.array_equal(render_snippet(track, 1, FAST).mix, render_snippet(track, 2, FAST).mix)


def test_label_file_alone_reproduces_the_snippet():
    track = synthetic_track()
    first = render_snippet(track, 11, FAST)
    config = MixConfig.from_dict(first.labels["config"])
    again = render_snippet(track, first.labels["seed"], config)
    assert np.array_equal(first.mix, again.mix)


def test_mix_is_sum_of_stems_and_does_not_clip():
    snippet = render_snippet(synthetic_track(), 3, MixConfig(seconds=2.0, effects=False))
    assert np.allclose(snippet.mix, snippet.stems.sum(axis=0), atol=1e-6)
    assert np.abs(snippet.mix).max() <= 0.95 + 1e-6
    assert snippet.mix.shape == (2 * SAMPLE_RATE, 2)


def test_schema_fields():
    labels = render_snippet(synthetic_track(), 5, FAST).labels
    assert labels["schema"] == SCHEMA_VERSION
    assert labels["sample_rate"] == SAMPLE_RATE and labels["channels"] == 2
    assert labels["frames"] == 2 * SAMPLE_RATE and labels["seconds"] == 2.0
    assert labels["activity"]["n_frames"] == 20 and labels["activity"]["fps"] == 10
    assert labels["times"]["boundaries"] == [] and labels["times"]["onsets_complete"] is False  # synthetic stems have no onset data
    for info in labels["stems"]:
        assert set(info) >= {"stem", "class", "source_class", "detail", "ambiguous", "gain_db", "rms_db", "active", "effects"}
        assert len(labels["activity"]["per_stem"][info["stem"]]) == 20


def test_silent_stem_is_not_labelled_present():
    n = SAMPLE_RATE * 20
    track = Track(
        "synthetic", "TrackB", SAMPLE_RATE, n,
        [
            make_stem("S00", "Bass", tone(110, n)),
            make_stem("S01", "Guitar", np.zeros((n, 2), dtype=np.float32)),
        ],
    )
    labels = render_snippet(track, 0, MixConfig(seconds=2.0, p_stem_dropped=0.0)).labels
    assert labels["classes"] == ["Bass"]
    guitar = next(s for s in labels["stems"] if s["stem"] == "S01")
    assert guitar["active"] is False and set(labels["activity"]["per_stem"]["S01"]) == {"0"}


def test_partly_silent_stem_is_active_only_where_audible():
    n = SAMPLE_RATE * 4
    audio = tone(220, n)
    audio[SAMPLE_RATE * 2 :] = 0.0
    track = Track("synthetic", "TrackC", SAMPLE_RATE, n, [make_stem("S00", "Piano", audio)])
    config = MixConfig(seconds=4.0, p_stem_dropped=0.0, effects=False, gain_db_range=(0.0, 0.0))
    row = render_snippet(track, 0, config).labels["activity"]["per_stem"]["S00"]
    assert row == "1" * 20 + "0" * 20


def test_unknown_mixture_stem_makes_the_snippet_ambiguous_not_a_class():
    n = SAMPLE_RATE * 20
    track = Track(
        "synthetic", "TrackD", SAMPLE_RATE, n,
        [make_stem("other", None, tone(330, n), source_class="other"), make_stem("bass", "Bass", tone(80, n), source_class="bass")],
    )
    labels = render_snippet(track, 0, MixConfig(seconds=2.0, p_stem_dropped=0.0)).labels
    assert labels["ambiguous"] is True
    assert labels["classes"] == ["Bass"]


def test_effect_records_say_whether_they_ran(monkeypatch):
    monkeypatch.setattr(effects, "_pedalboard", lambda: None)
    labels = render_snippet(synthetic_track(), 4, MixConfig(seconds=1.0)).labels
    records = [r for s in labels["stems"] for r in s["effects"]]
    assert [r["name"] for r in records[: len(effects.DEFAULT_CHAIN)]] == [e.name for e in effects.DEFAULT_CHAIN]
    assert any(r["fired"] for r in records) and not any(r["applied"] for r in records)
    assert labels["effects_backend"] is None


def test_vocabulary_is_shared_across_datasets():
    assert MUSDB_CLASSES["drums"] == SLAKH_CLASSES["Drums"] == "Drums"
    assert MUSDB_CLASSES["bass"] == "Bass" and MUSDB_CLASSES["other"] is None
    assert all(c in CLASSES for c in list(SLAKH_CLASSES.values()) + ["Vocals"])


def test_registry_flags_non_commercial(tmp_path):
    log = tmp_path / "sources.log"
    log.write_text(
        "name | load url | local path | date | license\n"
        "a | http://a | C:\\a | 1 | CC BY 4.0\n"
        "b | http://b | C:\\b | 2 | !! NON COMMERCIAL\n",
        encoding="utf-8",
    )
    reg = load_registry(log)
    assert not reg["a"].non_commercial and reg["b"].non_commercial
