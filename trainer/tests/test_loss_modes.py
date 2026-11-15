"""The loss modes of `separation_loss` (l1, norm_l1, sisdr) stay finite, also on digitally silent snippets."""

import pytest
import torch

from trainer.separation import Separator, separation_loss


@pytest.mark.parametrize("loss_type", ["l1", "norm_l1", "sisdr"])
def test_loss_is_finite_with_a_silent_snippet_and_a_silent_source(loss_type):
    torch.manual_seed(0)
    mix = torch.randn(3, 2, 8192) * 0.1
    sources = torch.randn(3, 4, 2, 8192) * 0.05
    sources[0] = 0.0  # a digitally silent snippet: the mixture is silent too
    mix[0] = 0.0
    sources[1, 2] = 0.0  # one silent source in an audible snippet
    estimates = (torch.randn(3, 4, 2, 8192) * 0.01).requires_grad_()
    mask = torch.ones(3, 4)
    loss = separation_loss(estimates, sources, mask, loss_type=loss_type, mix_rms=(mix**2).mean(dim=(-1, -2)).sqrt())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(estimates.grad).all()


def test_unknown_loss_type_is_refused():
    with pytest.raises(ValueError, match="loss_type"):
        separation_loss(torch.zeros(1, 2, 2, 64), torch.zeros(1, 2, 2, 64), torch.ones(1, 2), loss_type="mse")


def test_lstm_context_keeps_the_output_shape_and_masks_add_up():
    model = Separator(3, 2, 256, 128, base=4, depth=2, context="lstm", context_reduce=2, context_hidden=8)
    mix = torch.randn(2, 2, 4096) * 0.1
    out = model(mix)
    assert out.shape == (2, 3, 2, 4096)
    _, masks = model.analyse(mix)
    assert torch.allclose(masks.sum(dim=1), torch.ones_like(masks.sum(dim=1)), atol=1e-5)


def test_silent_weight_raises_the_loss_of_a_leak_into_a_silent_source_in_sisdr_mode():
    torch.manual_seed(0)
    sources = torch.randn(2, 3, 2, 8192) * 0.05
    sources[:, 1] = 0.0  # a silent source
    estimates = sources + torch.randn(2, 3, 2, 8192) * 0.01  # the silent source leaks
    mask = torch.ones(2, 3)
    mix_rms = torch.full((2,), 0.1)
    base = separation_loss(estimates, sources, mask, loss_type="sisdr", mix_rms=mix_rms)
    weighted = separation_loss(estimates, sources, mask, loss_type="sisdr", mix_rms=mix_rms, silent_weight=10.0)
    assert torch.isfinite(weighted) and weighted > base
