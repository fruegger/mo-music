import json

import numpy as np
import soundfile as sf
import torch
import yaml
from helpers import make_stem, synthetic_track, tone
from mixer import MixConfig
from mixer.sources import MUSDB_STEMS, ONSETS_FILE, SAMPLE_RATE, prepare_studio
from mixer.sources import Track as MixerTrack
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.separation import evaluate_separation, separation_loss
from trainer.sources import GROUPS, group_sources, source_names
from trainer.train import main

CONFIG = MixConfig(seconds=1.0, p_stem_dropped=0.0, effects=False)
NAMES = ["drums", "bass", "vocals", "guitar", "keys", "other"]


def labels(*classes, ambiguous=False):
    return {"stems": [{"class": c} for c in classes], "ambiguous": ambiguous}


def test_the_set_has_six_sources_and_the_last_three_form_the_group():
    assert source_names("musdb6") == NAMES and GROUPS["musdb6"] == [3, 4, 5]


def test_stems_go_to_their_source_and_everything_unclassified_goes_to_other():
    classes = ["Drums", "Bass", "Vocals", "Guitar", "Piano", "Organ", "Strings", "Synth Lead", "Brass", None]
    stems = np.stack([np.full((4, 2), i + 1.0, dtype=np.float32) for i in range(len(classes))])
    sources, mask = group_sources(stems, labels(*classes), "musdb6")
    assert sources[:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5 + 6 + 7 + 8, 9 + 10]  # keys: piano, organ, strings, synth
    assert mask.tolist() == [1.0] * 6  # nothing ambiguous: every target is known
    assert np.allclose(sources.sum(axis=0), stems.sum(axis=0))  # the sources add up to the mix


def test_an_audible_unlabelled_mixture_masks_guitar_keys_and_other_but_not_the_rest():
    stems = np.ones((3, 4, 2), dtype=np.float32)
    sources, mask = group_sources(stems, labels("Drums", "Vocals", None, ambiguous=True), "musdb6")
    assert mask.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert sources[5, 0, 0] == 1.0  # the mixture is carried in "other" so that the group sum is right


def test_items_carry_the_group_sum_only_when_individual_targets_are_unknown():
    known = SnippetDataset([synthetic_track()], CONFIG, 2, tasks=("separation",), source_set="musdb6")[0]
    assert known["sources"].shape == (6, 2, SAMPLE_RATE) and known["sources_mask"].tolist() == [1.0] * 6
    assert known["group_mask"].tolist() == [0.0] * 6 and known["group_target"].shape == (2, SAMPLE_RATE)
    ambiguous = SnippetDataset([synthetic_track(with_unknown=True)], CONFIG, 2, tasks=("separation",), source_set="musdb6")[0]
    assert ambiguous["sources_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert ambiguous["group_mask"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert torch.allclose(ambiguous["group_target"], ambiguous["sources"][3:].sum(dim=0), atol=1e-6)
    assert "group_mask" not in SnippetDataset([synthetic_track()], CONFIG, 2, tasks=("separation",), source_set="musdb4")[0]


def test_loss_supervises_the_group_sum_where_individual_targets_are_masked():
    sources = torch.rand(2, 6, 2, 50)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0] * 6])
    group_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0] * 6])
    group_target = sources[0, 3:].sum(dim=0).unsqueeze(0).repeat(2, 1, 1)
    estimates = sources.clone()
    estimates[0, 3] += 0.3  # the first item gets the split inside the group wrong ...
    estimates[0, 4] -= 0.3  # ... but the sum right
    assert float(separation_loss(estimates, sources, mask, group_target, group_mask)) < 1e-6
    assert float(separation_loss(estimates, sources, mask)) < 1e-6  # (masked out either way)
    estimates[0, 5] += 0.2  # now the sum is wrong too
    assert float(separation_loss(estimates, sources, mask, group_target, group_mask)) > 1e-3
    assert float(separation_loss(estimates, sources, mask)) < 1e-6  # without the group term the error is invisible
    plain = separation_loss(estimates, sources, torch.ones(2, 6))  # the old behaviour is unchanged
    assert torch.isclose(plain, (estimates - sources).abs().mean(dim=(-1, -2)).mean())


class Perfect(torch.nn.Module):
    """Returns the true sources of the batch it was last given (a stand-in for a perfect model)."""

    def __init__(self, n):
        super().__init__()
        self.net = type("Net", (), {"n_sources": n})()
        self.sources = None

    def forward(self, mix):
        return self.sources


def test_evaluation_scores_the_group_sum_as_one_source():
    ds = SnippetDataset([synthetic_track(with_unknown=True)], CONFIG, 4, tasks=("separation",), source_set="musdb6")
    batch = next(iter(DataLoader(ds, batch_size=4)))
    model = Perfect(6)
    model.sources = batch["sources"]
    scores = evaluate_separation(model, [batch], "cpu")
    assert scores["examples"][3:] == [0.0, 0.0, 0.0]  # the individual targets are unknown
    assert scores["group"]["examples"] == 4 and scores["group"]["si_sdr_improvement"] > 30  # a perfect sum
    plain = SnippetDataset([synthetic_track()], CONFIG, 4, tasks=("separation",), source_set="musdb4")
    batch4 = next(iter(DataLoader(plain, batch_size=4)))
    model4 = Perfect(4)
    model4.sources = batch4["sources"]
    assert "group" not in evaluate_separation(model4, [batch4], "cpu")  # the four-source sets have no group


def write_prepared_musdb(root):
    """MUSDB18 as `prepare_musdb` leaves it: train/test, two songs each, stems drums/bass/other/vocals as FLAC."""
    freqs = {"drums": 150.0, "bass": 70.0, "other": 330.0, "vocals": 520.0}
    for split in ("train", "test"):
        for i in range(2):
            folder = root / split / f"Artist {i} - Song {i}"
            folder.mkdir(parents=True)
            for name in MUSDB_STEMS:
                sf.write(folder / f"{name}.flac", tone(freqs[name], 4.0), SAMPLE_RATE)
            (folder / ONSETS_FILE).write_text(json.dumps({"sample_rate": SAMPLE_RATE, "onsets": {n: [] for n in MUSDB_STEMS}}))


def write_studio(root, tmp_path):
    src = tmp_path / "studio_src"
    for split, song in (("train", "Song A"), ("test", "Song B")):
        folder = src / song / "tracks"
        folder.mkdir(parents=True)
        for name, freq in (("bass", 80), ("drums", 150), ("guit1", 330), ("piano", 440), ("voc1", 520), ("keys", 660)):
            sf.write(folder / f"{name}.wav", tone(freq, 4.0), SAMPLE_RATE)
    prepare_studio(src, root, test_songs=("Song B",), labels={"Song A/keys": "Strings", "Song B/keys": "Strings"})


def test_training_on_musdb_and_studio_with_the_six_sources_scores_each_dataset(tmp_path):
    musdb, studio = tmp_path / "musdb", tmp_path / "studio"
    write_prepared_musdb(musdb)
    write_studio(studio, tmp_path)
    sources = tmp_path / "sources.log"
    sources.write_text(
        "name | load url | local path | date | license\n"
        f"MUSDB18-prepared | derived | {musdb} | 1 | !! NON COMMERCIAL\n"
        f"studio | own | {studio} | 1 | Original material of the user\n",
        encoding="utf-8",
    )
    cfg = {
        "task": "separation", "sources": str(sources), "out_dir": str(tmp_path / "run"),
        "datasets": [{"name": "musdb18-prepared", "train_split": "train", "val_split": "test", "weight": 1.0},
                     {"name": "studio", "train_split": "train", "val_split": "test", "weight": 0.5}],
        "separation": {"source_set": "musdb6", "n_fft": 256, "hop": 64, "base_channels": 4, "depth": 2, "mask": "softmax"},
        "mix": {"seconds": 1.0, "effects": False, "p_stem_dropped": 0.0, "gain_db_range": [-3.0, 3.0]},
        "train": {"epochs": 2, "steps_per_epoch": 2, "batch_size": 2, "lr": 1e-3, "num_workers": 0,
                  "val_snippets": 4, "val_seed_base": 1000000, "seed": 0},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    main(["--config", str(path)])
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    last = run["history"][-1]["validation"]
    assert set(last) == {"MUSDB18-prepared", "studio"}
    assert "group" in last["MUSDB18-prepared"] and last["MUSDB18-prepared"]["examples"]["guitar"] == 0  # MUSDB18: the sum
    assert last["studio"]["examples"]["guitar"] > 0 and last["studio"]["examples"]["keys"] > 0  # the studio song: each source
    assert run["non_commercial"] is True
    assert json.loads((tmp_path / "run" / "model.json").read_text())["sources"] == NAMES
