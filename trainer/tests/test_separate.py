import json

import numpy as np
import pytest
import soundfile as sf
import torch

from trainer.separate import MODEL_RATE, load_model, main, read_audio, separate, write_stems
from trainer.train import SeparationTask

CFG = {"separation": {"source_set": "musdb4", "n_fft": 256, "hop": 64, "base_channels": 4, "depth": 2, "mask": "softmax"}}


def tonal(seconds: float, rate: int = MODEL_RATE) -> torch.Tensor:
    """Low-frequency stereo signal: nothing in the top (Nyquist) bin, which the separator's masks drop."""
    t = torch.arange(int(seconds * rate)) / rate
    left = 0.3 * torch.sin(2 * torch.pi * 110 * t) + 0.2 * torch.sin(2 * torch.pi * 330 * t)
    return torch.stack([left, 0.7 * left])


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    torch.manual_seed(0)
    torch.save(SeparationTask(CFG).build().state_dict(), run / "model.pt")
    (run / "run.json").write_text(json.dumps({"config": CFG}), encoding="utf-8")
    return run


def test_load_model_rebuilds_the_separator_and_names_its_sources(run_dir):
    model, names = load_model(run_dir)
    assert names == ["drums", "bass", "vocals", "other"] and not model.training


@pytest.mark.parametrize("seconds", [0.3, 2.0, 3.37])  # shorter than a chunk, a few chunks, a length that is no multiple
def test_chunked_separation_keeps_the_length_and_the_sources_add_up_to_the_mix(run_dir, seconds):
    model, _ = load_model(run_dir)
    audio = tonal(seconds)
    stems = separate(model, audio, MODEL_RATE, chunk_s=1.0, overlap_s=0.25)
    assert stems.shape == (4, 2, audio.shape[-1])
    error = (stems.sum(dim=0) - audio).pow(2).mean().sqrt() / audio.pow(2).mean().sqrt()
    assert float(error) < 0.02  # softmax masks sum to one, so the estimates sum to the mix


def test_chunking_does_not_depend_much_on_the_chunk_length(run_dir):
    model, _ = load_model(run_dir)
    audio = tonal(2.5)
    whole = separate(model, audio, MODEL_RATE, chunk_s=30.0)
    chunked = separate(model, audio, MODEL_RATE, chunk_s=1.0, overlap_s=0.5)
    # The untrained net's masks differ near chunk borders, but the cross-fade keeps the result in the same range.
    assert float((whole - chunked).abs().max()) < 3 * float(audio.abs().max())
    assert torch.isfinite(chunked).all()


def test_write_stems_scales_everything_by_one_gain_when_something_would_clip(tmp_path):
    audio = torch.full((2, 1000), 0.5)
    stems = torch.stack([torch.full((2, 1000), 1.5), torch.full((2, 1000), 0.1)])
    gain = write_stems(tmp_path, ["a", "b"], audio, stems, MODEL_RATE)
    assert 0 < gain < 1
    a, _ = sf.read(tmp_path / "a.wav")
    b, _ = sf.read(tmp_path / "b.wav")
    assert a.max() <= 1.0 and np.isclose(a.max() / b.max(), 15.0, rtol=0.01)  # the balance is kept
    assert (tmp_path / "remainder.wav").exists()


def test_main_writes_the_stems_of_stereo_mono_and_resampled_files(run_dir, tmp_path):
    stereo, mono, slow = tmp_path / "stereo.wav", tmp_path / "mono.flac", tmp_path / "slow.wav"
    sf.write(stereo, tonal(1.5).T.numpy(), MODEL_RATE)
    sf.write(mono, tonal(1.0)[0].numpy(), MODEL_RATE)
    sf.write(slow, tonal(1.0, 22050).T.numpy(), 22050)
    out = tmp_path / "out"
    main(["--run", str(run_dir), "--out", str(out), "--chunk", "1.0", "--overlap", "0.25", str(stereo), str(mono), str(slow)])
    for name in ("stereo", "mono", "slow"):
        for stem in ("drums", "bass", "vocals", "other", "remainder"):
            data, rate = sf.read(out / name / f"{stem}.wav")
            assert rate == MODEL_RATE and data.ndim == 2 and data.shape[1] == 2
    assert sf.info(out / "stereo" / "drums.wav").frames == int(1.5 * MODEL_RATE)
    assert abs(sf.info(out / "slow" / "bass.wav").frames - MODEL_RATE) <= 2  # 22.05 kHz resampled up


def test_read_audio_duplicates_mono_and_drops_extra_channels(tmp_path):
    sf.write(tmp_path / "m.wav", np.zeros(100, dtype=np.float32), MODEL_RATE)
    sf.write(tmp_path / "s.wav", np.zeros((100, 4), dtype=np.float32), MODEL_RATE)
    assert read_audio(tmp_path / "m.wav")[0].shape == (2, 100)
    assert read_audio(tmp_path / "s.wav")[0].shape == (2, 100)
