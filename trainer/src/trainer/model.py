"""A small log-mel CNN with a multi-label instrument head."""

import torch
import torchaudio
from torch import nn

from mixer import CLASSES


class InstrumentCnn(nn.Module):
    def __init__(self, sample_rate: int = 44100, n_mels: int = 96, n_classes: int = len(CLASSES), pool: str = "mean"):
        """`pool`: how the feature map is reduced over mel and time: "mean" (global average) or "maxmean" (average and
        maximum side by side, so a class that is only briefly present is not averaged away)."""
        super().__init__()
        if pool not in ("mean", "maxmean"):
            raise ValueError(f"pool must be 'mean' or 'maxmean', not {pool!r}")
        self.pool = pool
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=2048, hop_length=512, n_mels=n_mels)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)
            )

        self.body = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Linear(128 * (2 if pool == "maxmean" else 1), n_classes)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (batch, samples) -> logits (batch, n_classes)."""
        x = torch.log1p(self.mel(audio)).unsqueeze(1)
        x = self.body(x)
        x = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1) if self.pool == "maxmean" else x.mean(dim=(2, 3))
        return self.head(x)
