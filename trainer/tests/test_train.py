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
