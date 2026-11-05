import json

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf
import torch
from helpers import synthetic_track
from mixer import MixConfig
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.recognise import load_recogniser, song_presence, window_probabilities
from trainer.recognise import main as recognise_main
from trainer.separate import main as separate_main
from trainer.train import PresenceTask, fit

RATE = 44100
CFG = {"presence": {"source_set": "musdb4", "n_mels": 16}, "mix": {"seconds": 1.0}}
SEP_CFG = {"separation": {"source_set": "musdb4", "n_fft": 256, "hop": 64, "base_channels": 4, "depth": 2, "mask": "softmax"}}


def loaders(n=8, batch=4, dropped=0.5):
    config = MixConfig(seconds=1.0, effects=False, p_stem_dropped=dropped)
    ds = SnippetDataset([synthetic_track()], config, n, tasks=("separation",), source_set="musdb4")
    return DataLoader(ds, batch_size=batch, shuffle=True), DataLoader(ds, batch_size=batch)


def test_labels_are_the_sources_above_the_activity_threshold():
    task = PresenceTask(CFG)
    sources = torch.zeros(2, 4, 2, 1000)
    sources[0, 0] = 0.1  # drums audible (-20 dB)
    sources[0, 1] = 1e-4  # bass at -80 dB: absent
    sources[1, 3] = 0.01  # other at -40 dB: audible
    labels = task.labels({"sources": sources})
    assert labels.tolist() == [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def test_the_task_trains_and_reports_per_source_scores():
    task = PresenceTask(CFG)
    model = task.build()
    train, val = loaders()
    history = fit(task, model, train, val, 1, 2, 1e-2, "cpu")
    scores = history[0]
    assert set(scores) >= {"precision", "recall", "f1", "present_share", "mean_f1", "train_loss"}
    assert set(scores["f1"]) == {"drums", "bass", "vocals", "other"} and 0.0 <= scores["mean_f1"] <= 1.0
    assert scores["present_share"]["vocals"] == 0.0  # the synthetic song has no vocals


def test_export_matches_torch_and_lists_the_sources(tmp_path):
    task = PresenceTask(CFG)
    model = task.build().eval()
    task.export(model, tmp_path / "model.onnx", MixConfig(seconds=1.0))
    audio = torch.randn(2, RATE) * 0.1
    got = ort.InferenceSession(str(tmp_path / "model.onnx")).run(None, {"audio": audio.numpy()})[0]
    assert np.allclose(got, model(audio).detach().numpy(), atol=1e-3)
    assert json.loads((tmp_path / "model.json").read_text())["sources"] == ["drums", "bass", "vocals", "other"]


@pytest.fixture(scope="module")
def run_dirs(tmp_path_factory):
    torch.manual_seed(0)
    from trainer.train import SeparationTask

    pres = tmp_path_factory.mktemp("presence")
    torch.save(PresenceTask(CFG).build().state_dict(), pres / "model.pt")
    (pres / "run.json").write_text(json.dumps({"config": CFG}), encoding="utf-8")
    sep = tmp_path_factory.mktemp("separation")
    torch.save(SeparationTask(SEP_CFG).build().state_dict(), sep / "model.pt")
    (sep / "run.json").write_text(json.dumps({"config": SEP_CFG}), encoding="utf-8")
    return pres, sep


@pytest.mark.parametrize("seconds, windows", [(0.4, 1), (1.0, 1), (2.0, 3), (2.3, 4)])  # padded, exact, hop 0.5 s, last one shifted
def test_window_probabilities_cover_the_signal(run_dirs, seconds, windows):
    model, names, window_s = load_recogniser(run_dirs[0])
    probs = window_probabilities(model, torch.randn(2, int(seconds * RATE)) * 0.1, RATE, window_s, 0.5)
    assert names == ["drums", "bass", "vocals", "other"] and window_s == 1.0
    assert probs.shape == (windows, 4) and probs.min() >= 0 and probs.max() <= 1


def test_song_presence_keeps_a_source_that_is_audible_in_a_few_windows():
    probs = torch.zeros(20, 3)
    probs[:, 0] = 0.9  # always
    probs[:2, 1] = 0.8  # in 10% of the windows
    probs[0, 2] = 0.8  # in 5%: exactly the default minimum
    assert song_presence(probs).tolist() == [1.0, 1.0, 1.0]
    assert song_presence(probs, min_fraction=0.2).tolist() == [1.0, 0.0, 0.0]
    assert song_presence(probs, threshold=0.95).tolist() == [0.0, 0.0, 0.0]


def test_the_commands_run_end_to_end(run_dirs, tmp_path, capsys):
    pres, sep = run_dirs
    t = torch.arange(int(1.5 * RATE)) / RATE
    sf.write(tmp_path / "song.wav", (0.2 * torch.sin(2 * torch.pi * 220 * t)).repeat(2, 1).T.numpy(), RATE)
    recognise_main(["--run", str(pres), str(tmp_path / "song.wav")])
    out = capsys.readouterr().out
    assert "song.wav" in out and all(n in out for n in ("drums", "bass", "vocals", "other"))
    # With a recogniser the separator gates by itself; the untrained one decides something, and the files follow it.
    separate_main(["--run", str(sep), "--out", str(tmp_path / "stems"), "--recogniser", str(pres), str(tmp_path / "song.wav")])
    written = {p.stem for p in (tmp_path / "stems" / "song").iterdir()}
    assert "remainder" in written and written - {"remainder"} <= {"drums", "bass", "vocals", "other"} and len(written) >= 2
    # --present wins over the recogniser.
    separate_main(["--run", str(sep), "--out", str(tmp_path / "forced"), "--present", "bass", "--recogniser", str(pres), str(tmp_path / "song.wav")])
    assert {p.stem for p in (tmp_path / "forced" / "song").iterdir()} == {"bass", "remainder"}
