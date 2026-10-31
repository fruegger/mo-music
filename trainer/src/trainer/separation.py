"""Source separation: a mask-estimating U-Net on magnitude spectrograms.

`SeparationNet` works in the spectrogram domain (magnitudes in, masks out), so
it can be exported to ONNX; `Separator` wraps it with the STFT / inverse STFT
for training and evaluation. A runtime that loads the exported net has to do
the STFT, the mask multiplication and the inverse STFT itself.
"""

import torch
import torch.nn.functional as F
from torch import nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
    )


class SeparationNet(nn.Module):
    """`(batch, channels, freq, time)` magnitudes -> `(batch, sources, channels, freq, time)` masks in [0, 1].

    `freq` and `time` must be divisible by `2 ** depth` (the `Separator` pads)."""

    def __init__(self, n_sources: int, n_channels: int = 2, base: int = 16, depth: int = 3):
        super().__init__()
        self.n_sources, self.n_channels, self.depth = n_sources, n_channels, depth
        widths = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([conv_block(n_channels if i == 0 else widths[i - 1], widths[i]) for i in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(depth)])
        self.merge = nn.ModuleList([conv_block(2 * widths[i], widths[i]) for i in range(depth)])
        self.head = nn.Conv2d(widths[0], n_sources * n_channels, 1)

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(100.0 * magnitude)
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for i in reversed(range(self.depth)):
            x = self.merge[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        masks = torch.sigmoid(self.head(x))
        b, _, f, t = masks.shape
        return masks.view(b, self.n_sources, self.n_channels, f, t)


class Separator(nn.Module):
    """STFT -> `SeparationNet` masks -> inverse STFT. `forward(mix)` takes
    `(batch, channels, samples)` and returns `(batch, sources, channels, samples)`."""

    def __init__(self, n_sources: int, n_channels: int = 2, n_fft: int = 2048, hop: int = 512, base: int = 16, depth: int = 3):
        super().__init__()
        self.net = SeparationNet(n_sources, n_channels, base, depth)
        self.n_fft, self.hop, self.depth = n_fft, hop, depth
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """`(batch, channels, samples)` -> complex `(batch, channels, freq = n_fft // 2 + 1, frames)`."""
        b, c, t = audio.shape
        spec = torch.stft(audio.reshape(b * c, t), self.n_fft, self.hop, window=self.window, return_complex=True)
        return spec.reshape(b, c, *spec.shape[1:])

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        b, c, t = mix.shape
        spec = self.stft(mix)  # (b, c, F+1, frames)
        freq, frames = spec.shape[2] - 1, spec.shape[3]  # drop the Nyquist bin: keeps freq a power-of-two multiple
        pad = (-frames) % 2**self.depth
        magnitude = F.pad(spec[:, :, :freq].abs(), (0, pad))
        masks = self.net(magnitude)[..., :frames]  # (b, s, c, freq, frames)
        masks = F.pad(masks, (0, 0, 0, 1))  # Nyquist bin: mask 0
        estimates = masks * spec.unsqueeze(1)
        s = estimates.shape[1]
        out = torch.istft(
            estimates.reshape(b * s * c, *estimates.shape[3:]), self.n_fft, self.hop, window=self.window, length=t
        )
        return out.reshape(b, s, c, t)


def separation_loss(estimates: torch.Tensor, sources: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 on the waveforms, averaged over the sources whose target is known.
    A silent source is a valid target (the model should output silence)."""
    per_source = (estimates - sources).abs().mean(dim=(-1, -2))  # (batch, sources)
    return (per_source * mask).sum() / mask.sum().clamp(min=1.0)


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB over the last two dims `(channels, samples)`; any leading dims."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    ref_energy = (reference**2).sum(dim=(-1, -2)).clamp(min=eps)
    alpha = (estimate * reference).sum(dim=(-1, -2)) / ref_energy
    target = alpha[..., None, None] * reference
    noise = estimate - target
    return 10.0 * torch.log10(((target**2).sum(dim=(-1, -2)) + eps) / ((noise**2).sum(dim=(-1, -2)) + eps))


def evaluate_separation(model: Separator, loader, device, active_db: float = -50.0) -> dict:
    """Mean SI-SDR per source, over the examples where that source is audible
    (RMS above `active_db`) and its target is known, plus the improvement over
    using the mixture itself as the estimate."""
    model.eval()
    sums = torch.zeros(model.net.n_sources)
    base_sums = torch.zeros(model.net.n_sources)
    counts = torch.zeros(model.net.n_sources)
    with torch.no_grad():
        for batch in loader:
            mix, sources, mask = batch["mix"].to(device), batch["sources"].to(device), batch["sources_mask"]
            est = model(mix).cpu()
            sources = sources.cpu()
            rms_db = 10.0 * torch.log10((sources**2).mean(dim=(-1, -2)) + 1e-12)
            valid = (rms_db > active_db) & (mask > 0)
            score = si_sdr(est, sources)
            baseline = si_sdr(mix.cpu().unsqueeze(1).expand_as(sources), sources)
            sums += (score * valid).sum(dim=0)
            base_sums += (baseline * valid).sum(dim=0)
            counts += valid.sum(dim=0)
    mean = sums / counts.clamp(min=1)
    base = base_sums / counts.clamp(min=1)
    return {"si_sdr": mean.tolist(), "si_sdr_improvement": (mean - base).tolist(), "examples": counts.tolist()}
