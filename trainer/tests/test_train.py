import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import yaml
from helpers import synthetic_track
from mixer import MixConfig
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.train import InstrumentTask, SeparationTask, fit, main

SLAKH = Path(r"C:/workarea/slakh2100")
SEP_CFG = {"separation": {"source_set": "musdb4", "n_fft": 256, "hop": 64, "base_channels": 4, "depth": 2}}


def loaders(task, n=8, batch=4):
    config = MixConfig(seconds=0.5, effects=False)
    ds = SnippetDataset([synthetic_track()], config, n, tasks=(task.name,), **task.dataset_kwargs)
    return DataLoader(ds, batch_size=batch, shuffle=True), DataLoader(ds, batch_size=batch)


@pytest.mark.parametrize("task", [InstrumentTask({}), SeparationTask(SEP_CFG)], ids=lambda t: t.name)
def test_fit_runs_and_reports(task):
    torch.manual_seed(0)
    train, val = loaders(task)
    epochs_seen = []
    history = fit(task, task.build(), train, val, epochs=2, steps_per_epoch=2, lr=1e-3, device="cpu", on_epoch=epochs_seen.append)
    assert epochs_seen == [0, 1] and [h["epoch"] for h in history] == [0, 1]
    assert all(np.isfinite(h["train_loss"]) for h in history)
    json.dumps(history)  # the history ends up in run.json


def test_separation_export_matches_torch(tmp_path):
    torch.manual_seed(0)
    task = SeparationTask(SEP_CFG)
    model = task.build()
    model.eval()
    task.export(model, tmp_path / "sep.onnx", MixConfig())
    meta = json.loads((tmp_path / "sep.json").read_text())
    assert meta["sources"] == ["drums", "bass", "vocals", "other"] and meta["n_fft"] == 256 and meta["frames_multiple_of"] == 4

    session = ort.InferenceSession(str(tmp_path / "sep.onnx"))
    for batch, frames in [(1, 16), (2, 40)]:  # other sizes than the one it was exported with
        magnitude = torch.rand(batch, 2, 128, frames)
        expected = model.net(magnitude).detach().numpy()
        (got,) = session.run(None, {"magnitude": magnitude.numpy()})
        assert got.shape == expected.shape == (batch, 4, 2, 128, frames)
        assert np.allclose(got, expected, atol=1e-4)


def test_instrument_export_runs_in_onnx(tmp_path):
    task = InstrumentTask({})
    model = task.build()
    model.eval()
    task.export(model, tmp_path / "inst.onnx", MixConfig(seconds=1.0))
    session = ort.InferenceSession(str(tmp_path / "inst.onnx"))
    audio = torch.randn(2, 30000)
    (got,) = session.run(None, {"audio": audio.numpy()})
    assert np.allclose(got, model(audio).detach().numpy(), atol=1e-3)


@pytest.mark.skipif(not (SLAKH / "train").exists(), reason="Slakh2100 not available")
@pytest.mark.parametrize("task_name, extra", [("instruments", {}), ("separation", SEP_CFG)])
def test_main_end_to_end_on_slakh(tmp_path, task_name, extra):
    cfg = {
        "task": task_name,
        "sources": str(Path(__file__).resolve().parents[2] / "repo" / "prototype" / "sources.log"),
        "out_dir": str(tmp_path / "run"),
        "datasets": [{"name": "slakh2100", "train_split": "train", "val_split": "validation"}],
        "mix": {"seconds": 1.0, "effects": False, "p_stem_dropped": 0.15, "gain_db_range": [-6.0, 3.0]},
        "train": {"epochs": 1, "steps_per_epoch": 2, "batch_size": 2, "lr": 1e-3, "num_workers": 0,
                  "val_snippets": 2, "val_seed_base": 1000000, "seed": 0},
        **extra,
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    main(["--config", str(config_path)])
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["task"] == task_name and run["licenses"]["slakh2100"] and run["non_commercial"] is False
    assert (tmp_path / "run" / "model.pt").exists() and (tmp_path / "run" / "model.onnx").exists()


def test_resume_continues_where_a_checkpoint_stopped(tmp_path):
    from trainer.train import load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    task = SeparationTask(SEP_CFG)
    train, val = loaders(task)
    model = task.build()
    saved = []

    def save(epoch, history, optimiser):
        save_checkpoint(tmp_path / "checkpoint.pt", model, optimiser, epoch + 1, history)
        saved.append(epoch)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    fit(task, model, train, val, epochs=1, steps_per_epoch=2, lr=1e-3, device="cpu", optimiser=optimiser, on_epoch_end=save)
    assert saved == [0] and not (tmp_path / "checkpoint.tmp").exists()

    fresh = task.build()
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    done, history = load_checkpoint(tmp_path / "checkpoint.pt", fresh, fresh_opt)
    assert done == 1 and [h["epoch"] for h in history] == [0]
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), fresh.state_dict().values()))
    assert len(fresh_opt.state_dict()["state"]) == len(optimiser.state_dict()["state"]) > 0

    seen = []
    history = fit(task, fresh, train, val, epochs=3, steps_per_epoch=2, lr=1e-3, device="cpu",
                  on_epoch=seen.append, optimiser=fresh_opt, start_epoch=done, history=history)
    assert seen == [1, 2] and [h["epoch"] for h in history] == [0, 1, 2]


def test_resume_from_bare_weights_needs_the_epoch_count(tmp_path):
    from trainer.train import load_checkpoint

    task = SeparationTask(SEP_CFG)
    model = task.build()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    other = task.build()
    with pytest.raises(ValueError, match="start-epoch"):
        load_checkpoint(tmp_path / "model.pt", other)
    done, history = load_checkpoint(tmp_path / "model.pt", other, epochs_done=2)
    assert done == 2 and history == []
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), other.state_dict().values()))


def test_source_weights_scale_the_loss_per_source():
    task = SeparationTask({"separation": {"source_set": "musdb4", "source_weights": {"vocals": 3.0}}})
    assert task.source_weights.tolist() == [1.0, 1.0, 3.0, 1.0]
    with pytest.raises(ValueError):
        SeparationTask({"separation": {"source_set": "musdb4", "source_weights": {"piano": 2.0}}})


def test_lr_schedule_warms_up_then_decays_to_the_final_fraction():
    from trainer.train import lr_at

    assert lr_at(0, 100, 1.0) == lr_at(99, 100, 1.0) == 1.0  # defaults: constant
    assert lr_at(0, 100, 1.0, warmup_steps=10) == pytest.approx(0.1)
    assert lr_at(9, 100, 1.0, warmup_steps=10) == pytest.approx(1.0)
    assert lr_at(10, 100, 1.0, warmup_steps=10, final_fraction=0.1) == pytest.approx(1.0)
    assert lr_at(55, 100, 1.0, warmup_steps=10, final_fraction=0.1) == pytest.approx(0.55)  # halfway down the cosine
    assert lr_at(100, 100, 1.0, warmup_steps=10, final_fraction=0.1) == pytest.approx(0.1)


def test_fit_applies_the_schedule_and_clips_the_gradient(monkeypatch):
    task = SeparationTask(SEP_CFG)
    train, val = loaders(task)
    model = task.build()
    optimiser = torch.optim.Adam(model.parameters(), lr=1.0)  # fit overrides this with the schedule
    seen = []
    real_clip = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda params, max_norm: (seen.append(max_norm), real_clip(params, max_norm))[1])
    fit(task, model, train, val, epochs=2, steps_per_epoch=2, lr=1e-3, device="cpu", optimiser=optimiser,
        warmup_steps=2, lr_final_fraction=0.1, grad_clip=0.5)
    assert seen == [0.5] * 4
    # Last step is global step 3 of 4: halfway down the cosine, after the 2 warm-up steps.
    assert optimiser.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.55)


def test_ema_tracks_the_weights_and_swaps_in_and_out():
    from trainer.train import Ema

    model = torch.nn.Linear(2, 2)
    ema = Ema(model, decay=0.5)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)  # the first update uses a ramped-up decay of 2 / 11
    d = (1 + 1) / (10 + 1)
    assert torch.allclose(ema.shadow["weight"], d * before["weight"] + (1 - d) * model.weight, atol=1e-6)
    own = model.weight.detach().clone()
    with ema.applied(model):
        assert torch.allclose(model.weight, ema.shadow["weight"])
    assert torch.equal(model.weight, own)  # the model's own weights come back


def test_fit_with_ema_evaluates_the_average_and_checkpoints_it(tmp_path):
    from trainer.train import Ema, load_checkpoint, save_checkpoint

    torch.manual_seed(0)
    task = SeparationTask(SEP_CFG)
    train, val = loaders(task, n=12)  # three batches of 4
    model = task.build()
    ema = Ema(model, decay=0.9)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    fit(task, model, train, val, epochs=1, steps_per_epoch=3, lr=1e-2, device="cpu", optimiser=optimiser, ema=ema)
    assert ema.updates == 3
    assert not torch.equal(ema.shadow["net.head.weight"], model.state_dict()["net.head.weight"])  # average differs from raw
    save_checkpoint(tmp_path / "c.pt", model, optimiser, 1, [], ema)
    other, other_ema = task.build(), Ema(task.build(), decay=0.9)
    load_checkpoint(tmp_path / "c.pt", other, None, None, other_ema)
    assert other_ema.updates == 3 and torch.equal(other_ema.shadow["net.head.weight"], ema.shadow["net.head.weight"])


@pytest.mark.parametrize("mask", ["sigmoid", "softmax"])
def test_export_with_either_mask_mode_matches_torch(tmp_path, mask):
    torch.manual_seed(0)
    cfg = {"separation": {**SEP_CFG["separation"], "mask": mask}}
    task = SeparationTask(cfg)
    model = task.build()
    model.eval()
    task.export(model, tmp_path / "sep.onnx", MixConfig())
    assert json.loads((tmp_path / "sep.json").read_text())["mask"] == mask
    session = ort.InferenceSession(str(tmp_path / "sep.onnx"))
    magnitude = torch.rand(1, 2, 128, 16)
    (got,) = session.run(None, {"magnitude": magnitude.numpy()})
    assert np.allclose(got, model.net(magnitude).detach().numpy(), atol=1e-4)


@pytest.mark.skipif(not (SLAKH / "train").exists(), reason="Slakh2100 not available")
def test_main_with_ema_and_softmax_masks_saves_the_average_and_resumes(tmp_path):
    cfg = {
        "task": "separation",
        "sources": str(Path(__file__).resolve().parents[2] / "repo" / "prototype" / "sources.log"),
        "out_dir": str(tmp_path / "run"),
        "datasets": [{"name": "slakh2100", "train_split": "train", "val_split": "validation"}],
        "separation": {**SEP_CFG["separation"], "mask": "softmax"},
        "mix": {"seconds": 1.0, "effects": False, "p_stem_dropped": 0.15, "gain_db_range": [-6.0, 3.0]},
        "train": {"epochs": 1, "steps_per_epoch": 2, "batch_size": 2, "lr": 1e-2, "num_workers": 0,
                  "val_snippets": 2, "val_seed_base": 1000000, "seed": 0, "ema_decay": 0.9},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    main(["--config", str(config_path)])
    out = tmp_path / "run"
    saved, checkpoint = torch.load(out / "model.pt"), torch.load(out / "checkpoint.pt")
    assert all(torch.equal(saved[k], checkpoint["ema"]["shadow"][k]) for k in saved)  # model.pt is the average
    assert not all(torch.equal(saved[k], checkpoint["model"][k]) for k in saved)  # which differs from the raw weights
    assert json.loads((out / "model.json").read_text())["mask"] == "softmax"

    cfg["train"]["epochs"] = 2
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    main(["--config", str(config_path), "--resume", str(out / "checkpoint.pt")])
    run = json.loads((out / "run.json").read_text())
    assert [h["epoch"] for h in run["history"]] == [0, 1] and run["epochs_before_resume"] == 1
