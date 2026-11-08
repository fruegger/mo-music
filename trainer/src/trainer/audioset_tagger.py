"""AudioSet tagger: PANNs Cnn14 (Kong et al. 2020, weights CC BY 4.0), a port that needs nothing beyond torch and torchaudio.

Used as a listening aid, never as training data: it ranks the windows of a stem by how much of an instrument it hears
(`scripts/tag_stems.py`, `scripts/segment_overview.py`). The checkpoint's own filter bank is used for the features, and
the file is loaded with `weights_only=True`, so nothing in it can run code.

    model = load()                                  # downloads the weights (327 MB) to WEIGHTS_DIR on first use
    starts, probs = tag_file(model, "other.flac")   # probs (windows, 527) in [0, 1]; LABELS[i] names column i
"""

import csv
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

RATE = 32000
WEIGHTS_DIR = Path(r"C:\workarea\audioset")  # outside the repo: the checkpoint is 327 MB
WEIGHTS_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
WEIGHTS_FILE, LABELS_FILE = "Cnn14.pth", "class_labels_indices.csv"


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor, pool: tuple[int, int]) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool) if pool != (1, 1) else x


class Cnn14(nn.Module):
    """Mono audio at 32 kHz `(batch, samples)` -> probabilities of the 527 AudioSet classes `(batch, 527)`."""

    def __init__(self):
        super().__init__()
        self.stft_real = nn.Conv1d(1, 513, 1024, stride=320, bias=False)  # the DFT as a convolution (hop 320, 1024 window)
        self.stft_imag = nn.Conv1d(1, 513, 1024, stride=320, bias=False)
        self.register_buffer("melW", torch.zeros(513, 64))
        self.bn0 = nn.BatchNorm2d(64)
        channels = [1, 64, 128, 256, 512, 1024, 2048]
        for k in range(6):
            setattr(self, f"conv_block{k + 1}", _Block(channels[k], channels[k + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = F.pad(wav[:, None], (512, 512), mode="reflect")
        power = self.stft_real(x) ** 2 + self.stft_imag(x) ** 2  # (batch, 513, frames)
        x = (10.0 * torch.log10(torch.clamp(power.transpose(1, 2) @ self.melW, min=1e-10)))[:, None]  # (batch, 1, frames, 64)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for k in range(1, 7):
            x = getattr(self, f"conv_block{k}")(x, (2, 2) if k < 6 else (1, 1))
        x = x.mean(dim=3)
        x = F.relu(self.fc1(x.max(dim=2).values + x.mean(dim=2)))
        return torch.sigmoid(self.fc_audioset(x))


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def labels(weights_dir: Path = WEIGHTS_DIR) -> list[str]:
    """The 527 class names, in the order of the model's outputs (downloaded on first use)."""
    path = Path(weights_dir) / LABELS_FILE
    if not path.exists():
        _fetch(LABELS_URL, path)
    return [row[2] for row in list(csv.reader(path.open(encoding="utf-8")))[1:]]


def load(weights_dir: Path = WEIGHTS_DIR) -> Cnn14:
    """The model in eval mode with the PANNs weights (downloaded to `weights_dir` on first use)."""
    path = Path(weights_dir) / WEIGHTS_FILE
    if not path.exists():
        _fetch(WEIGHTS_URL, path)
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    rename = {"spectrogram_extractor.stft.conv_real.weight": "stft_real.weight",
              "spectrogram_extractor.stft.conv_imag.weight": "stft_imag.weight", "logmel_extractor.melW": "melW"}
    model = Cnn14()
    model.load_state_dict({rename.get(k, k): v for k, v in state.items()}, strict=True)
    return model.eval()


@torch.no_grad()
def tag_audio(model: Cnn14, audio: np.ndarray, rate: int, window: float = 5.0, hop: float = 2.5, batch: int = 8):
    """`audio` `(samples,)` or `(samples, channels)` -> `(starts in s, probabilities (windows, 527))`."""
    mono = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    mono = mono.mean(dim=1) if mono.ndim == 2 else mono
    mono = torchaudio.functional.resample(mono, rate, RATE)
    size = int(window * RATE)
    if len(mono) < size:
        mono = F.pad(mono, (0, size - len(mono)))
    starts = list(range(0, len(mono) - size + 1, max(1, int(hop * RATE))))
    probs = torch.cat([model(torch.stack([mono[s : s + size] for s in starts[i : i + batch]])) for i in range(0, len(starts), batch)])
    return np.array(starts) / RATE, probs.numpy()


def tag_file(model: Cnn14, path, window: float = 5.0, hop: float = 2.5):
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return tag_audio(model, audio, rate, window, hop)
