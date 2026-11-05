from types import SimpleNamespace

import numpy as np
import pytest
import torch
from helpers import clicks, make_stem, tone
from mixer import MixConfig
from mixer.sources import SAMPLE_RATE, Track

from trainer.data import RemixDataset, SnippetDataset, VocalMixDataset
from trainer.train import build_train_set

CONFIG = MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False)
N_TRACKS = 4


def song(k: int) -> Track:
    """A song whose stems carry its number: bass at 80+10k Hz, piano at 440+20k Hz (drums and guitar as noise/other)."""
    seconds = 10
    stems = [
        make_stem("S0", "Bass", tone(80 + 10 * k, seconds)),
        make_stem("S1", "Drums", clicks(0.5, seconds), is_drum=True),
        make_stem("S2", "Piano", tone(440 + 20 * k, seconds)),
    ]
    return Track("synthetic", f"Song{k}", SAMPLE_RATE, SAMPLE_RATE * seconds, stems)


def base_set(length=8, tasks=("separation",), source_set="musdb4"):
    return SnippetDataset([song(k) for k in range(N_TRACKS)], CONFIG, length, tasks=tasks, source_set=source_set)


def peak_frequency(x: torch.Tensor, low: float, high: float) -> float:
    """Frequency of the strongest component of channel 0 between `low` and `high` Hz."""
    spectrum = torch.fft.rfft(x[0]).abs()
    freqs = torch.fft.rfftfreq(x.shape[-1], 1.0 / SAMPLE_RATE)
    band = (freqs >= low) & (freqs <= high)
    return float(freqs[band][spectrum[band].argmax()])


def test_p_zero_returns_the_plain_items():
    base = base_set()
    ds = RemixDataset(base, p=0.0)
    for i in range(4):
        assert torch.equal(ds[i]["mix"], base[i]["mix"])


def test_a_remixed_item_has_the_shapes_of_a_plain_one_and_its_sources_add_up_to_the_mix():
    base = base_set()
    item, plain = RemixDataset(base, p=1.0)[3], base[3]
    assert set(item) == {"mix", "sources", "sources_mask"}
    assert item["mix"].shape == plain["mix"].shape and item["sources"].shape == plain["sources"].shape
    assert item["sources_mask"].shape == plain["sources_mask"].shape
    assert torch.allclose(item["sources"].sum(dim=0), item["mix"], atol=1e-5)


def test_the_sources_of_one_item_come_from_different_songs():
    ds = RemixDataset(base_set(length=16), p=1.0)
    names = base_set().source_set  # musdb4: drums, bass, vocals, other
    assert names == "musdb4"
    mixed = 0
    for i in range(16):
        sources = ds[i]["sources"]  # (drums, bass, vocals, other)
        bass_song = round((peak_frequency(sources[1], 60, 130) - 80) / 10)
        piano_song = round((peak_frequency(sources[3], 400, 540) - 440) / 20)
        assert 0 <= bass_song < N_TRACKS and 0 <= piano_song < N_TRACKS
        mixed += bass_song != piano_song
    # With 4 songs, a plain item always has bass and piano from the same one; a remix differs 3 times in 4.
    assert mixed >= 6


def test_deterministic_in_index_and_epoch_and_new_mixes_each_epoch():
    ds = RemixDataset(base_set(), p=1.0)
    assert torch.equal(ds[2]["mix"], ds[2]["mix"])
    first = ds[2]["mix"]
    ds.set_epoch(1)
    assert ds.base.epoch == 1 and not torch.equal(first, ds[2]["mix"])


def test_p_is_the_share_of_remixed_items():
    base = base_set(length=40)
    ds = RemixDataset(base, p=0.5)
    remixed = sum(not torch.equal(ds[i]["mix"], base[i]["mix"]) for i in range(40))
    assert 8 <= remixed <= 32


def test_the_mix_is_limited_to_the_peak():
    ds = RemixDataset(base_set(), p=1.0, peak=0.2)
    item = ds[0]
    assert float(item["mix"].abs().max()) <= 0.2 + 1e-6
    assert torch.allclose(item["sources"].sum(dim=0), item["mix"], atol=1e-5)


def test_only_the_separation_task_with_the_musdb4_sources_can_be_remixed():
    with pytest.raises(ValueError, match="separation task only"):
        RemixDataset(base_set(tasks=("instruments",)))
    with pytest.raises(ValueError, match="musdb4"):
        RemixDataset(base_set(source_set="classes"))


def test_build_train_set_wraps_the_dataset_when_the_config_asks_for_remix():
    ds = SimpleNamespace(name="musdb18-prepared", path="unused")
    kwargs = {"tasks": ("separation",), "source_set": "musdb4"}
    entry = {"name": "musdb18-prepared", "train_split": "train"}
    plain = build_train_set({}, ds, entry, CONFIG, 8, kwargs)
    assert isinstance(plain, SnippetDataset)
    remix = build_train_set({}, ds, {**entry, "remix": {"p": 0.7}}, CONFIG, 8, kwargs)
    assert isinstance(remix, RemixDataset) and remix.p == 0.7 and len(remix) == 8
    assert not isinstance(remix, VocalMixDataset)
    with pytest.raises(ValueError, match="not both"):
        build_train_set({}, ds, {**entry, "remix": {}, "vocal_mix": {"from": "x"}}, CONFIG, 8, kwargs)


def test_remix_donors_skip_tempo_and_pitch_unless_asked():
    ds = SimpleNamespace(name="musdb18-prepared", path="unused")
    kwargs = {"tasks": ("separation",), "source_set": "musdb4"}
    entry = {"name": "musdb18-prepared", "train_split": "train"}
    augmented = MixConfig(seconds=1.0, p_tempo=0.5, p_pitch=0.5, effects=False)
    default = build_train_set({}, ds, {**entry, "remix": {"p": 1.0}}, augmented, 8, kwargs)
    assert default.base.config.p_tempo == 0.5  # the plain items keep the augmentation
    assert default.donors.config.p_tempo == 0.0 and default.donors.config.p_pitch == 0.0
    assert default.donors.config.seconds == 1.0 and default.donors.source == default.base.source
    both = build_train_set({}, ds, {**entry, "remix": {"tempo_pitch": True}}, augmented, 8, kwargs)
    assert both.donors is both.base


def test_donors_render_the_remixed_sources_and_must_match_the_base():
    base = base_set()
    donors = SnippetDataset([song(k) for k in range(N_TRACKS)], MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False,
                            gain_db_range=(0.0, 0.0)), 8, tasks=("separation",), source_set="musdb4")
    ds = RemixDataset(base, p=1.0, donors=donors)
    assert ds[0]["sources"].shape == base[0]["sources"].shape
    ds.set_epoch(2)
    assert base.epoch == donors.epoch == 2
    other = SnippetDataset([song(0)], CONFIG, 8, tasks=("separation",), source_set="classes")
    with pytest.raises(ValueError, match="same source set"):
        RemixDataset(base, donors=other)

