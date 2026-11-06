import torch
from helpers import synthetic_track
from mixer import MixConfig
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.separation import separation_loss
from trainer.train import SeparationTask


def test_a_leak_into_a_silent_source_counts_silent_weight_times_as_much():
    sources = torch.zeros(1, 2, 2, 100)
    sources[0, 0] = 0.3  # the first source is audible, the second must be silent
    estimates = torch.zeros_like(sources)
    estimates[0, 1] = 0.05  # a leak into the silent one
    mask = torch.ones(1, 2)
    assert torch.isclose(separation_loss(estimates, sources, mask), torch.tensor(0.175))  # (0.30 + 0.05) / 2
    heavy = separation_loss(estimates, sources, mask, silent_weight=10.0)
    assert torch.isclose(heavy, torch.tensor(0.4))  # (0.30 + 10 * 0.05) / 2: the divisor is unchanged


def test_errors_on_audible_sources_and_masked_sources_are_not_reweighted():
    sources = torch.zeros(1, 3, 2, 100)
    sources[0, 0] = 0.3
    sources[0, 1] = 0.2
    estimates = sources.clone()
    estimates[0, 0] += 0.1  # an error on an audible source
    estimates[0, 2] = 0.5  # a leak into a silent source whose target is unknown (masked)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    plain = separation_loss(estimates, sources, mask)
    assert torch.isclose(separation_loss(estimates, sources, mask, silent_weight=10.0), plain)
    assert torch.isclose(plain, torch.tensor(0.1 / 2))


def test_the_default_changes_nothing_and_the_task_reads_the_weight_from_the_config():
    sep = {"source_set": "musdb4", "n_fft": 256, "hop": 64, "base_channels": 4, "depth": 2}
    ds = SnippetDataset([synthetic_track()], MixConfig(seconds=1.0, p_stem_dropped=0.5, effects=False), 8,
                        tasks=("separation",), source_set="musdb4")
    batch = next(iter(DataLoader(ds, batch_size=8)))
    assert (batch["sources"].abs().amax(dim=(-1, -2)) == 0).any()  # some source is silent in this batch (no vocals at all)
    torch.manual_seed(0)
    base_task = SeparationTask({"separation": sep})
    model = base_task.build()
    heavy_task = SeparationTask({"separation": {**sep, "silent_weight": 20.0}})
    with torch.no_grad():
        base, heavy = base_task.loss(model, batch, "cpu"), heavy_task.loss(model, batch, "cpu")
    assert heavy > base  # the untrained model leaks into the silent sources
    assert torch.isclose(SeparationTask({"separation": {**sep, "silent_weight": 1.0}}).loss(model, batch, "cpu"), base)
