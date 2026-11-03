import numpy as np
import pytest

from mixer import effects
from mixer.mix import MixConfig, render_snippet
from mixer.sources import SAMPLE_RATE, Stem, Track

pytest.importorskip("pedalboard")

SECONDS = 2


def make_signal():
    """A decaying plucked-ish tone plus a little noise: gives compressors,
    delays and reverbs something to act on."""
    rng = np.random.default_rng(0)
    t = np.arange(SAMPLE_RATE * SECONDS) / SAMPLE_RATE
    mono = 0.5 * np.sin(2 * np.pi * 220 * t) * np.exp(-3 * (t % 0.5)) + 0.01 * rng.standard_normal(len(t))
    return np.stack([mono, 0.9 * mono], axis=1).astype(np.float32)


@pytest.mark.parametrize("effect", effects.DEFAULT_CHAIN, ids=lambda e: e.name)
def test_each_effect_runs_and_changes_the_audio(effect):
    audio = make_signal()
    for seed in range(5):  # a spread of random parameters, not one lucky draw
        params = effect.sample(np.random.default_rng(seed))
        out = effect.apply(audio, SAMPLE_RATE, params)
        assert out.shape == audio.shape and out.dtype == np.float32
        assert np.isfinite(out).all()
        assert not np.allclose(out, audio, atol=1e-5), f"{effect.name} {params} left the audio unchanged"
        assert np.abs(out).max() < 50.0, f"{effect.name} {params} exploded"


@pytest.mark.parametrize("effect", effects.DEFAULT_CHAIN, ids=lambda e: e.name)
def test_effect_is_deterministic(effect):
    audio = make_signal()
    params = effect.sample(np.random.default_rng(1))
    assert np.array_equal(effect.apply(audio, SAMPLE_RATE, params), effect.apply(audio, SAMPLE_RATE, params))


def test_chain_reports_applied_and_matches_the_rng_without_the_backend(monkeypatch):
    audio = make_signal()
    out, with_backend = effects.apply_chain(audio, SAMPLE_RATE, np.random.default_rng(3))
    assert [r["name"] for r in with_backend] == [e.name for e in effects.DEFAULT_CHAIN]
    assert all(r["applied"] == r["fired"] for r in with_backend)
    assert any(r["applied"] for r in with_backend) and not np.array_equal(out, audio)

    monkeypatch.setattr(effects, "_pedalboard", lambda: None)
    _, without = effects.apply_chain(audio, SAMPLE_RATE, np.random.default_rng(3))
    # Same draws and parameters whether or not the effects ran.
    assert [(r["fired"], r["params"]) for r in with_backend] == [(r["fired"], r["params"]) for r in without]
    assert not any(r["applied"] for r in without)


def test_missing_backend_warns_and_leaves_the_audio_clean(monkeypatch):
    audio = make_signal()
    monkeypatch.setattr(effects, "_pedalboard", lambda: None)
    with pytest.warns(RuntimeWarning, match="pedalboard is not installed"):
        out, _ = effects.apply_chain(audio, SAMPLE_RATE, np.random.default_rng(3))
    assert np.array_equal(out, audio)


def test_snippet_with_effects_is_deterministic_finite_and_unclipped():
    n = SAMPLE_RATE * 10
    t = np.arange(n) / SAMPLE_RATE
    stems = [
        Stem(f"S{i}", c, c, lambda s, k, a=np.stack([x, x], axis=1).astype(np.float32): a[s : s + k])
        for i, (c, x) in enumerate([("Bass", 0.4 * np.sin(2 * np.pi * 80 * t)), ("Piano", 0.4 * np.sin(2 * np.pi * 440 * t))])
    ]
    track = Track("synthetic", "TrackE", SAMPLE_RATE, n, stems)
    config = MixConfig(seconds=2.0, p_stem_dropped=0.0)
    applied_any = False
    for seed in range(20):
        a, b = render_snippet(track, seed, config), render_snippet(track, seed, config)
        assert np.array_equal(a.mix, b.mix)
        assert np.isfinite(a.mix).all() and np.abs(a.mix).max() <= config.peak_target + 1e-6
        applied_any |= any(r["applied"] for s in a.labels["stems"] for r in s["effects"])
    assert applied_any
