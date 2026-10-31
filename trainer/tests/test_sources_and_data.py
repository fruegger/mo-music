import numpy as np
import pytest
import torch
from helpers import synthetic_track
from mixer import CLASSES, MixConfig, render_snippet

from trainer.data import SnippetDataset
from trainer.sources import MUSDB4, group_sources, source_names

CONFIG = MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False)


def test_musdb4_sources_sum_to_the_mix():
    snippet = render_snippet(synthetic_track(), 0, CONFIG)
    sources, mask = group_sources(snippet.stems, snippet.labels, "musdb4")
    assert sources.shape == (4, *snippet.mix.shape) and mask.tolist() == [1, 1, 1, 1]
    assert np.allclose(sources.sum(axis=0), snippet.mix, atol=1e-5)
    # Bass and drums have their own source, vocals are silent, the piano and guitar land in "other".
    by_name = dict(zip(MUSDB4, sources))
    assert np.abs(by_name["vocals"]).max() == 0
    assert np.abs(by_name["bass"]).max() > 0 and np.abs(by_name["drums"]).max() > 0
    stems = {s["stem"]: a for s, a in zip(snippet.labels["stems"], snippet.stems)}
    assert np.allclose(by_name["other"], stems["S2"] + stems["S3"], atol=1e-6)


def test_musdb4_puts_an_unknown_mixture_in_other():
    snippet = render_snippet(synthetic_track(with_unknown=True), 0, CONFIG)
    sources, mask = group_sources(snippet.stems, snippet.labels, "musdb4")
    assert mask.all() and np.allclose(sources.sum(axis=0), snippet.mix, atol=1e-5)


def test_class_sources_follow_the_vocabulary():
    snippet = render_snippet(synthetic_track(), 0, CONFIG)
    sources, mask = group_sources(snippet.stems, snippet.labels, "classes")
    assert sources.shape[0] == len(CLASSES) == len(source_names("classes")) and mask.all()
    assert np.allclose(sources.sum(axis=0), snippet.mix, atol=1e-5)
    silent = [c for c, s in zip(CLASSES, sources) if not np.abs(s).max()]
    assert set(CLASSES) - set(silent) == {"Bass", "Drums", "Piano", "Guitar"}


def test_class_sources_mask_what_an_unknown_mixture_could_hide():
    snippet = render_snippet(synthetic_track(with_unknown=True), 0, CONFIG)
    assert snippet.labels["ambiguous"]
    sources, mask = group_sources(snippet.stems, snippet.labels, "classes")
    known = {c for c, m in zip(CLASSES, mask) if m}
    assert known == {"Bass", "Drums", "Vocals"}
    # The unknown stem belongs to no class, so the classes no longer add up to the mix.
    assert not np.allclose(sources.sum(axis=0), snippet.mix, atol=1e-3)


def test_dataset_items_for_both_tasks():
    ds = SnippetDataset([synthetic_track()], CONFIG, 3, tasks=("instruments", "separation"), source_set="musdb4")
    item = ds[1]
    n = 44100
    assert item["mono"].shape == (n,) and item["instruments"].shape == (len(CLASSES),)
    assert item["mix"].shape == (2, n) and item["sources"].shape == (4, 2, n) and item["sources_mask"].shape == (4,)
    assert all(t.dtype == torch.float32 for t in item.values())
    assert torch.allclose(item["sources"].sum(dim=0), item["mix"], atol=1e-5)
    assert set(SnippetDataset([synthetic_track()], CONFIG, 1)[0]) == {"mono", "instruments", "instruments_mask"}


def test_mono_material_is_served_as_stereo():
    """Slakh2100 is mono, MUSDB18 stereo; the separation model always sees two channels."""
    track = synthetic_track()
    for stem in track.stems:
        load = stem.load
        stem.load = lambda s, k, f=load: f(s, k)[:, :1]
    ds = SnippetDataset([track], CONFIG, 1, tasks=("separation",))
    item = ds[0]
    assert item["mix"].shape == (2, 44100) and item["sources"].shape == (4, 2, 44100)
    assert torch.equal(item["mix"][0], item["mix"][1])
    assert torch.allclose(item["sources"].sum(dim=0), item["mix"], atol=1e-5)


def test_dataset_seeds_change_with_the_epoch_and_not_between_runs():
    ds = SnippetDataset([synthetic_track()], MixConfig(seconds=1.0), 2, tasks=("separation",))
    first, again = ds[0]["mix"], ds[0]["mix"]
    assert torch.equal(first, again)
    ds.set_epoch(1)
    assert not torch.equal(first, ds[0]["mix"])


def test_a_batch_of_items_collates():
    ds = SnippetDataset([synthetic_track()], CONFIG, 4, tasks=("instruments", "separation"))
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
    assert batch["mix"].shape == (2, 2, 44100) and batch["sources"].shape == (2, 4, 2, 44100)
    assert batch["instruments"].shape == (2, len(CLASSES))


def test_unknown_source_set_is_an_error():
    snippet = render_snippet(synthetic_track(), 0, CONFIG)
    with pytest.raises(KeyError):
        group_sources(snippet.stems, snippet.labels, "nope")


SLAKH = __import__("pathlib").Path(r"C:/workarea/slakh2100")


@pytest.mark.skipif(not (SLAKH / "validation").exists(), reason="Slakh2100 not available")
def test_worker_processes_build_their_own_tracks_and_see_the_epoch():
    """Windows starts workers by pickling the dataset: it must pickle, and a
    set_epoch in the main process must reach persistent workers."""
    import pickle

    from mixer import Dataset

    dataset = Dataset("slakh2100", "", SLAKH, "", "CC BY 4.0")
    ds = SnippetDataset.from_dataset(
        dataset, "validation", MixConfig(seconds=0.5, effects=False), 4, tasks=("separation",)
    )
    assert pickle.loads(pickle.dumps(ds)).source == ds.source
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2, persistent_workers=True)
    epoch0 = [b["mix"].clone() for b in loader]
    again = [b["mix"].clone() for b in loader]
    assert all(torch.equal(a, b) for a, b in zip(epoch0, again))  # same epoch, same mixes
    ds.set_epoch(1)
    epoch1 = [b["mix"].clone() for b in loader]
    assert not any(torch.equal(a, b) for a, b in zip(epoch0, epoch1))
