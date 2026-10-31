import numpy as np

from mixer.mix import MixConfig, render_snippet
from mixer.registry import load_registry
from mixer.sources import SAMPLE_RATE, Stem, Track


def synthetic_track(n_stems=4, seconds=20):
    n = SAMPLE_RATE * seconds
    stems = []
    for i in range(n_stems):
        tone = 0.3 * np.sin(2 * np.pi * (110 * (i + 1)) * np.arange(n) / SAMPLE_RATE).astype(np.float32)
        audio = np.stack([tone, tone], axis=1)
        stems.append(
            Stem(f"S{i:02d}", ["Bass", "Piano", "Drums", "Guitar"][i], lambda s, k, a=audio: a[s : s + k], i == 2)
        )
    return Track("synthetic", "TrackA", SAMPLE_RATE, n, stems)


def test_same_seed_is_bit_identical():
    track, config = synthetic_track(), MixConfig(seconds=2.0)
    a, b = render_snippet(track, 7, config), render_snippet(track, 7, config)
    assert np.array_equal(a.mix, b.mix)
    assert a.labels == b.labels


def test_different_seeds_differ():
    track, config = synthetic_track(), MixConfig(seconds=2.0)
    assert not np.array_equal(render_snippet(track, 1, config).mix, render_snippet(track, 2, config).mix)


def test_mix_is_sum_of_stems_and_does_not_clip():
    snippet = render_snippet(synthetic_track(), 3, MixConfig(seconds=2.0, effects=False))
    assert np.allclose(snippet.mix, snippet.stems.sum(axis=0), atol=1e-6)
    assert np.abs(snippet.mix).max() <= 0.95 + 1e-6
    assert snippet.mix.shape == (2 * SAMPLE_RATE, 2)


def test_labels_match_the_stems_in_the_mix():
    snippet = render_snippet(synthetic_track(), 5, MixConfig(seconds=1.0, p_stem_dropped=0.5))
    assert len(snippet.labels["stems"]) == snippet.stems.shape[0]
    assert set(snippet.labels["classes"]) == {s["class"] for s in snippet.labels["stems"]}


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
