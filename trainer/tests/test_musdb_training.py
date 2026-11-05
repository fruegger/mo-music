import json
from pathlib import Path

import av
import numpy as np
import pytest
import torch
import yaml
from helpers import tone
from mixer import MixConfig
from mixer.sources import MUSDB_STEMS, SAMPLE_RATE, prepare_musdb
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.train import SeparationTask, evaluate_all, main, split_lengths

SLAKH = Path(r"C:/workarea/slakh2100")
SEP = {"separation": {"source_set": "musdb4", "n_fft": 256, "hop": 64, "base_channels": 4, "depth": 2}}


def stem_audio(freq, seconds=6):
    """A decaying tone re-struck every second: audible throughout, with onsets."""
    n = SAMPLE_RATE * seconds
    t = np.arange(n) / SAMPLE_RATE
    x = (0.4 * np.exp(-3 * (t % 1.0)) * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x, 0.9 * x])  # (channels, samples)


def write_stem_mp4(path: Path):
    """A MUSDB18-shaped file: mixture, drums, bass, other, vocals as five AAC streams."""
    freqs = {"drums": 150.0, "bass": 70.0, "other": 330.0, "vocals": 520.0}
    streams = {"mixture": sum(stem_audio(f) for f in freqs.values())}
    streams.update({name: stem_audio(freqs[name]) for name in MUSDB_STEMS})
    with av.open(str(path), "w") as container:
        outputs = {name: container.add_stream("aac", rate=SAMPLE_RATE) for name in streams}
        for stream in outputs.values():
            stream.layout = "stereo"
        for name, audio in streams.items():
            for lo in range(0, audio.shape[1], 1024):
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio[:, lo : lo + 1024]), format="fltp", layout="stereo")
                frame.sample_rate = SAMPLE_RATE
                frame.pts = lo
                for packet in outputs[name].encode(frame):
                    container.mux(packet)
            for packet in outputs[name].encode(None):
                container.mux(packet)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """A tiny prepared MUSDB18: two tracks in train and in test."""
    raw, out = tmp_path_factory.mktemp("raw"), tmp_path_factory.mktemp("prepared")
    for split in ("train", "test"):
        (raw / split).mkdir()
        for i in range(2):
            write_stem_mp4(raw / split / f"Artist {i} - Song {i}.stem.mp4")
    assert prepare_musdb(raw, out) == 4
    return out


def write_config(tmp_path, prepared, datasets, **train):
    sources = tmp_path / "sources.log"
    sources.write_text(
        "name | load url | local path | date | license\n"
        f"MUSDB18-prepared | derived | {prepared} | 1 | !! NON COMMERCIAL\n"
        f"slakh2100 | zenodo | {SLAKH} | 1 | CC BY 4.0\n",
        encoding="utf-8",
    )
    cfg = {
        "task": "separation",
        "sources": str(sources),
        "out_dir": str(tmp_path / "run"),
        "datasets": datasets,
        "mix": {"seconds": 1.0, "effects": False, "p_stem_dropped": 0.0, "gain_db_range": [-3.0, 3.0]},
        "train": {"epochs": 2, "steps_per_epoch": 2, "batch_size": 2, "lr": 1e-3, "num_workers": 0,
                  "val_snippets": 4, "val_seed_base": 1000000, "seed": 0, **train},
        **SEP,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_split_lengths_follow_the_weights():
    assert split_lengths(100, [1.0, 1.0]) == [100, 100]
    assert split_lengths(100, [1.0, 0.25]) == [100, 25]
    assert split_lengths(10, [0.0]) == [1]  # never an empty dataset


def test_evaluate_all_nests_scores_per_dataset_only_when_there_are_several():
    task = SeparationTask(SEP)
    model = task.build()
    from helpers import synthetic_track

    ds = SnippetDataset([synthetic_track()], MixConfig(seconds=0.5, effects=False), 2, tasks=("separation",))
    loader = DataLoader(ds, batch_size=2)
    flat = evaluate_all(task, model, loader, "cpu")
    assert "mean_si_sdr_improvement" in flat and "validation" not in flat
    assert set(evaluate_all(task, model, {"only": loader}, "cpu")) == set(flat)
    nested = evaluate_all(task, model, {"a": loader, "b": loader}, "cpu")
    assert set(nested) == {"validation"} and set(nested["validation"]) == {"a", "b"}


def test_training_on_prepared_musdb_sees_vocals_and_flags_the_licence(tmp_path, prepared):
    cfg = write_config(tmp_path, prepared, [{"name": "musdb18-prepared", "train_split": "train", "val_split": "test"}])
    main(["--config", str(cfg)])
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["finished"] and run["non_commercial"] is True
    assert "NON COMMERCIAL" in run["licenses"]["MUSDB18-prepared"]
    scores = run["history"][-1]
    assert scores["examples"]["vocals"] > 0  # unlike Slakh, MUSDB18 gives the vocals a target
    assert all(scores["examples"][name] > 0 for name in ("drums", "bass", "vocals", "other"))
    assert (tmp_path / "run" / "model.onnx").exists() and (tmp_path / "run" / "checkpoint.pt").exists()


@pytest.mark.skipif(not (SLAKH / "validation").exists(), reason="Slakh2100 not available")
def test_training_on_slakh_and_musdb_together_scores_each_dataset(tmp_path, prepared):
    datasets = [
        {"name": "slakh2100", "train_split": "validation", "val_split": "validation", "weight": 1.0},
        {"name": "musdb18-prepared", "train_split": "train", "val_split": "test", "weight": 0.5},
    ]
    main(["--config", str(write_config(tmp_path, prepared, datasets))])
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    validation = run["history"][-1]["validation"]
    assert set(validation) == {"slakh2100", "MUSDB18-prepared"}
    assert validation["slakh2100"]["examples"]["vocals"] == 0  # Slakh has no vocals
    assert validation["MUSDB18-prepared"]["examples"]["vocals"] > 0
    assert run["non_commercial"] is True


def test_shipped_configs_name_datasets_that_the_log_knows():
    """configs/*.yaml against the sources log each one names (relative to the trainer folder, where configs are run;
    names only: the data need not exist)."""
    from mixer import load_registry

    trainer = Path(__file__).resolve().parents[1]
    for path in sorted((trainer / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        registry = load_registry(trainer / cfg["sources"])
        MixConfig.from_dict(cfg["mix"])
        for entry in cfg["datasets"]:
            assert entry["name"].lower() in registry, f"{path.name}: {entry['name']} is not in {cfg['sources']}"
