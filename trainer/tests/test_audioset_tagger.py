"""The tagger port runs and gives probabilities (random weights: no download in the tests)."""

import numpy as np
import torch

from trainer import audioset_tagger as tagger


def test_a_window_of_audio_gives_527_probabilities():
    torch.manual_seed(0)
    model = tagger.Cnn14().eval()
    probs = model(torch.randn(2, int(5 * tagger.RATE)) * 0.1)
    assert probs.shape == (2, 527) and float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_tag_audio_windows_a_stereo_signal_at_another_rate():
    torch.manual_seed(0)
    model = tagger.Cnn14().eval()
    audio = np.random.default_rng(0).normal(0, 0.1, size=(44100 * 8, 2)).astype(np.float32)  # 8 s at 44.1 kHz
    starts, probs = tagger.tag_audio(model, audio, 44100, window=5.0, hop=2.5)
    assert starts.tolist() == [0.0, 2.5] and probs.shape == (2, 527)


def test_a_short_signal_is_padded_to_one_window():
    model = tagger.Cnn14().eval()
    starts, probs = tagger.tag_audio(model, np.zeros(16000, dtype=np.float32), 16000, window=5.0, hop=2.5)
    assert starts.tolist() == [0.0] and probs.shape == (1, 527)
