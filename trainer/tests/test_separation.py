import numpy as np
import pytest
import torch
from helpers import synthetic_track
from mixer import MixConfig

from trainer.data import SnippetDataset
from trainer.separation import (
    Separator, SeparationNet, evaluate_separation, separation_loss, si_sdr,
)

TINY = dict(n_fft=256, hop=64, base=4, depth=2)


def test_net_outputs_one_mask_per_source_and_channel():
    net = SeparationNet(n_sources=4, n_channels=2, base=4, depth=2)
    masks = net(torch.rand(3, 2, 32, 16))
    assert masks.shape == (3, 4, 2, 32, 16)
    assert masks.min() >= 0.0 and masks.max() <= 1.0


def test_separator_returns_waveforms_of_the_input_length():
    model = Separator(4, **TINY)
    for samples in (8000, 8001, 12345):  # lengths that are not multiples of the hop
        out = model(torch.randn(2, 2, samples))
        assert out.shape == (2, 4, 2, samples) and torch.isfinite(out).all()


def test_separator_with_unit_masks_reconstructs_the_input(monkeypatch):
    model = Separator(2, **TINY)
    monkeypatch.setattr(model.net, "forward", lambda m: torch.ones(m.shape[0], 2, 2, *m.shape[2:]))
    t = torch.arange(16000) / 44100
    mix = torch.stack([torch.sin(2 * torch.pi * 300 * t), torch.sin(2 * torch.pi * 500 * t)])[None]
    out = model(mix)
    # every source passes the whole mix, apart from the dropped Nyquist bin
    assert torch.allclose(out[:, 0], mix, atol=1e-3) and torch.allclose(out[:, 1], mix, atol=1e-3)


def test_gradients_reach_every_parameter():
    model = Separator(4, **TINY)
    loss = separation_loss(model(torch.randn(2, 2, 6000)), torch.randn(2, 4, 2, 6000), torch.ones(4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_si_sdr_known_values():
    ref = torch.randn(1, 2, 20000)
    assert si_sdr(ref, ref).item() > 60
    assert si_sdr(0.1 * ref, ref).item() > 60  # scale invariant
    noise = torch.randn_like(ref)
    noise = noise - (noise * ref).sum() / (ref * ref).sum() * ref  # orthogonal to the reference
    snr = 10.0
    est = ref + noise * (ref.pow(2).sum() / noise.pow(2).sum() / 10 ** (snr / 10)).sqrt()
    assert si_sdr(est, ref).item() == pytest.approx(snr, abs=0.3)
    assert si_sdr(noise, ref).item() < -30
    assert si_sdr(torch.randn(5, 4, 2, 100), torch.randn(5, 4, 2, 100)).shape == (5, 4)


def test_loss_ignores_unknown_sources_and_supervises_silence():
    target = torch.zeros(1, 2, 2, 100)
    est = torch.zeros(1, 2, 2, 100)
    est[0, 1] = 1.0  # wrong, but its target is masked out
    assert separation_loss(est, target, torch.tensor([1.0, 0.0])).item() == 0.0
    assert separation_loss(est, target, torch.tensor([1.0, 1.0])).item() == pytest.approx(0.5)
    est[0, 0] = 0.3  # a silent target is still a target
    assert separation_loss(est, target, torch.tensor([1.0, 0.0])).item() == pytest.approx(0.3)


def test_evaluation_skips_silent_and_masked_sources():
    ds = SnippetDataset([synthetic_track()], MixConfig(seconds=0.5, p_stem_dropped=0.0, effects=False), 4, tasks=("separation",))
    model = Separator(4, **TINY)
    scores = evaluate_separation(model, torch.utils.data.DataLoader(ds, batch_size=2), "cpu")
    # drums, bass and other are present in every snippet, vocals never
    assert scores["examples"] == [4.0, 4.0, 0.0, 4.0]
    assert all(np.isfinite(scores["si_sdr"]))


def test_a_tiny_model_learns_to_separate_two_sources():
    torch.manual_seed(0)
    config = MixConfig(seconds=0.5, p_stem_dropped=0.0, effects=False, gain_db_range=(-3.0, 3.0))
    ds = SnippetDataset([synthetic_track()], config, 24, tasks=("separation",))
    loader = torch.utils.data.DataLoader(ds, batch_size=6, shuffle=True)
    model = Separator(4, **TINY)
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses = []
    for _ in range(12):
        for batch in loader:
            loss = separation_loss(model(batch["mix"]), batch["sources"], batch["sources_mask"])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
    assert sum(losses[-4:]) / 4 < 0.5 * sum(losses[:4]) / 4
    scores = evaluate_separation(model, torch.utils.data.DataLoader(ds, batch_size=6), "cpu")
    assert scores["si_sdr_improvement"][1] > 0  # bass: better than leaving the mix as it is


def test_softmax_masks_add_up_to_one_across_sources():
    net = SeparationNet(n_sources=4, n_channels=2, base=4, depth=2, mask="softmax")
    masks = net(torch.rand(3, 2, 32, 16))
    assert masks.shape == (3, 4, 2, 32, 16)
    assert torch.allclose(masks.sum(dim=1), torch.ones(3, 2, 32, 16), atol=1e-5)
    # the default sigmoid masks do not
    plain = SeparationNet(n_sources=4, n_channels=2, base=4, depth=2)(torch.rand(3, 2, 32, 16))
    assert not torch.allclose(plain.sum(dim=1), torch.ones(3, 2, 32, 16), atol=1e-2)


def test_an_unknown_mask_mode_is_rejected():
    with pytest.raises(ValueError):
        SeparationNet(n_sources=4, mask="relu")


def test_softmax_separator_estimates_add_up_to_the_mixture():
    torch.manual_seed(0)
    model = Separator(4, mask="softmax", **TINY).eval()
    mix = torch.randn(1, 2, 8000)
    with torch.no_grad():
        total = model(mix).sum(dim=1)
    # the masks add up to 1 except in the dropped Nyquist bin, so the sum is the mixture bar one frequency bin
    assert (total - mix).pow(2).mean() < 0.02 * mix.pow(2).mean()
