"""`present_weights`: a source's error counts more only where its target is audible."""

import torch

from trainer.separation import separation_loss
from trainer.train import SeparationTask


def _case():
    sources = torch.zeros(2, 2, 2, 100)
    sources[0, 1] = 0.5  # item 0: source 1 audible; item 1: both silent
    estimates = torch.zeros_like(sources)  # the model outputs silence everywhere
    mask = torch.ones(2, 2)
    return estimates, sources, mask


def test_the_weight_applies_to_audible_targets_only():
    estimates, sources, mask = _case()
    plain = separation_loss(estimates, sources, mask)
    heavy = separation_loss(estimates, sources, mask, present_weights=torch.tensor([1.0, 4.0]))
    assert torch.isclose(heavy, plain * 4)  # the only error is the audible source 1 of item 0
    other = separation_loss(estimates, sources, mask, present_weights=torch.tensor([4.0, 1.0]))
    assert torch.isclose(other, plain)  # source 0 is silent everywhere: nothing to weigh


def test_without_the_option_nothing_changes():
    estimates, sources, mask = _case()
    assert torch.isclose(separation_loss(estimates, sources, mask), separation_loss(estimates, sources, mask, present_weights=None))


def test_the_task_reads_the_weights_and_rejects_unknown_names():
    cfg = {"separation": {"source_set": "musdb6", "present_weights": {"keys": 4.0}}}
    assert SeparationTask(cfg).present_weights.tolist() == [1.0, 1.0, 1.0, 1.0, 4.0, 1.0]
    assert SeparationTask({"separation": {"source_set": "musdb6"}}).present_weights is None
    try:
        SeparationTask({"separation": {"source_set": "musdb6", "present_weights": {"piano": 2.0}}})
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass
