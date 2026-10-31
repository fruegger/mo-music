import torch
from helpers import synthetic_track
from mixer import MixConfig

from trainer.data import SnippetDataset, VocalMixDataset

CONFIG = MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False)


def make(p):
    base = SnippetDataset([synthetic_track()], CONFIG, 4, tasks=("separation",), source_set="musdb4")
    # The synthetic track has no vocals; a second one stands in, with its bass as the "vocals" source.
    voc = SnippetDataset([synthetic_track()], CONFIG, 4, 99, tasks=("separation",), source_set="musdb4")
    ds = VocalMixDataset(base, voc, p, (0.0, 0.0), peak=10.0)
    ds.vocal_index = 1  # bass
    return ds, base, voc


def test_p_one_adds_the_vocals_to_the_mix_and_the_target():
    ds, base, voc = make(1.0)
    item, plain = ds[2], base[2]
    assert torch.allclose(item["mix"], plain["mix"] + voc[2]["sources"][1], atol=1e-6)
    assert torch.allclose(item["sources"].sum(dim=0), item["mix"], atol=1e-5)


def test_p_zero_leaves_items_alone():
    ds, base, _ = make(0.0)
    assert torch.equal(ds[1]["mix"], base[1]["mix"])


def test_the_mix_is_limited_to_the_peak():
    ds, _, _ = make(1.0)
    ds.peak = 0.5
    item = ds[0]
    assert float(item["mix"].abs().max()) <= 0.5 + 1e-6
    assert torch.allclose(item["sources"].sum(dim=0), item["mix"], atol=1e-5)


def test_deterministic_and_epoch_reaches_both():
    ds, _, _ = make(0.5)
    a = ds[1]["mix"]
    assert torch.equal(a, ds[1]["mix"])
    ds.set_epoch(3)
    assert ds.base.epoch == ds.vocals.epoch == 3
