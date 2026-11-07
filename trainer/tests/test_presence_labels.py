import pytest
import torch
from torch.utils.data import DataLoader

from trainer.model import InstrumentCnn
from trainer.train import PresenceTask

REL = {"presence": {"source_set": "musdb4", "n_mels": 16, "present_rel_db": -40.0, "absent_rel_db": -60.0}}


def batch_with_levels(levels_db, mix_db=0.0, samples=20000):
    """One item whose sources have the given RMS (dB, 4 values) and whose mix has `mix_db`."""
    sources = torch.stack([torch.full((2, samples), 10 ** (db / 20)) for db in levels_db]).unsqueeze(0)
    return {"sources": sources, "mix": torch.full((1, 2, samples), 10 ** (mix_db / 20))}


def test_relative_labels_call_a_source_present_absent_or_leave_it_out():
    task = PresenceTask(REL)
    # sources at -10, -45, -70 dB re a mix at 0 dB, and one that is exactly silent
    labels, weights = task.targets(batch_with_levels([-10.0, -45.0, -70.0, -120.0]))
    assert labels.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert weights.tolist() == [[1.0, 0.0, 1.0, 1.0]]  # -45 dB: neither clearly there nor clearly not
    # a silent mix has no present source
    silent = batch_with_levels([-120.0] * 4, mix_db=-120.0)
    assert task.targets(silent)[0].sum() == 0 and task.targets(silent)[1].sum() == 4


def test_the_absolute_definition_is_unchanged_and_weighs_everything():
    task = PresenceTask({"presence": {"source_set": "musdb4", "n_mels": 16}})
    labels, weights = task.targets(batch_with_levels([-20.0, -45.0, -55.0, -120.0]))
    assert labels.tolist() == [[1.0, 1.0, 0.0, 0.0]] and weights.tolist() == [[1.0] * 4]
    assert torch.equal(task.labels(batch_with_levels([-20.0, -45.0, -55.0, -120.0])), labels)


def test_a_left_out_label_does_not_change_the_loss():
    task = PresenceTask(REL)
    torch.manual_seed(0)
    model = task.build()
    a = batch_with_levels([-10.0, -45.0, -70.0, -120.0])
    b = {"sources": a["sources"].clone(), "mix": a["mix"]}
    b["sources"][0, 1] *= 0.5  # still in the ignored zone (-51 dB): the target is unchanged anyway
    c = {"sources": a["sources"].clone(), "mix": a["mix"]}
    c["sources"][0, 2] = 10 ** (-10 / 20)  # the "absent" source becomes present: the loss must change
    with torch.no_grad():
        assert torch.isclose(task.loss(model, a, "cpu"), task.loss(model, b, "cpu"))
        assert not torch.isclose(task.loss(model, a, "cpu"), task.loss(model, c, "cpu"))


def test_evaluation_ignores_the_left_out_zone():
    task = PresenceTask(REL)
    model = task.build().eval()
    batch = batch_with_levels([-10.0, -45.0, -70.0, -120.0])
    batch = {k: v.repeat(2, *([1] * (v.dim() - 1))) for k, v in batch.items()}
    scores = task.evaluate(model, [batch], "cpu")
    assert scores["present_share"]["bass"] == 0.0  # the ignored source counts for nothing, not as absent or present
    assert scores["present_share"]["drums"] == 1.0


@pytest.mark.parametrize("pool, width", [("mean", 128), ("maxmean", 256)])
def test_pooling_options_change_only_the_head(pool, width):
    model = InstrumentCnn(n_mels=16, n_classes=4, pool=pool).eval()
    assert model.head.in_features == width
    assert model(torch.randn(2, 8000)).shape == (2, 4)
    task = PresenceTask({"presence": {"source_set": "musdb4", "n_mels": 16, "pool": pool}})
    assert task.build().pool == pool


def test_an_unknown_pooling_is_refused():
    with pytest.raises(ValueError, match="pool must be"):
        InstrumentCnn(pool="min")


def test_maxmean_feeds_the_head_the_average_and_the_maximum_over_time():
    torch.manual_seed(0)
    model = InstrumentCnn(n_mels=16, n_classes=4, pool="maxmean").eval()
    audio = torch.randn(2, 20000)
    with torch.no_grad():
        feats = model.body(torch.log1p(model.mel(audio)).unsqueeze(1))
        expected = model.head(torch.cat([feats.mean(dim=(2, 3)), feats.amax(dim=(2, 3))], dim=1))
        assert torch.allclose(model(audio), expected, atol=1e-6)
        assert (feats.amax(dim=(2, 3)) >= feats.mean(dim=(2, 3))).all()  # the maximum is at least the mean
