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

    `mask` is `"sigmoid"` (each source's mask on its own) or `"softmax"` (the masks of the sources add up to 1
    in every time-frequency bin, so the estimates add up to the mixture).
    `freq` and `time` must be divisible by `2 ** depth` (the `Separator` pads)."""

    def __init__(self, n_sources: int, n_channels: int = 2, base: int = 16, depth: int = 3, mask: str = "softmax"):
        super().__init__()
        if mask not in ("sigmoid", "softmax"):
            raise ValueError(f"mask must be 'sigmoid' or 'softmax', not {mask!r}")
        self.n_sources, self.n_channels, self.depth, self.mask = n_sources, n_channels, depth, mask
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
        logits = self.head(x)
        b, _, f, t = logits.shape
        logits = logits.view(b, self.n_sources, self.n_channels, f, t)
        return torch.softmax(logits, dim=1) if self.mask == "softmax" else torch.sigmoid(logits)


def gate_masks(masks: torch.Tensor, presence: torch.Tensor, mode: str = "softmax") -> torch.Tensor:
    """Removes the masks of the sources that are not in the song. `masks` `(batch, sources, ...)`, `presence`
    `(batch, sources)` (1 = present, 0 = absent). Softmax masks add up to one in every time-frequency bin, so
    after dropping the absent sources the rest are divided by their sum, `m_k g_k / sum_j(m_j g_j)`: a softmax over
    the present sources only, which hands all the energy to sources that exist. Sigmoid masks are independent
    and are only multiplied. If no source is present the result is silent."""
    g = presence.to(masks.dtype).reshape(*presence.shape, *([1] * (masks.dim() - 2)))
    gated = masks * g
    if mode == "softmax":
        gated = gated / gated.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return gated


class Separator(nn.Module):
    """STFT -> `SeparationNet` masks -> inverse STFT. `forward(mix)` takes
    `(batch, channels, samples)` and returns `(batch, sources, channels, samples)`.
    With `presence` `(batch, sources)` (1 = the source is in the song, 0 = it is not) the masks of the absent
    sources are removed and, for softmax masks, the others renormalised (see `gate_masks`)."""

    def __init__(self, n_sources: int, n_channels: int = 2, n_fft: int = 2048, hop: int = 512, base: int = 16, depth: int = 3,
                 mask: str = "softmax"):
        super().__init__()
        self.net = SeparationNet(n_sources, n_channels, base, depth, mask)
        self.n_fft, self.hop, self.depth = n_fft, hop, depth
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """`(batch, channels, samples)` -> complex `(batch, channels, freq = n_fft // 2 + 1, frames)`."""
        b, c, t = audio.shape
        spec = torch.stft(audio.reshape(b * c, t), self.n_fft, self.hop, window=self.window, return_complex=True)
        return spec.reshape(b, c, *spec.shape[1:])

    def analyse(self, mix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(batch, channels, samples)` -> the complex STFT `(b, c, F+1, frames)` and the network's masks
        `(b, sources, c, F, frames)` (without the Nyquist bin)."""
        spec = self.stft(mix)
        freq, frames = spec.shape[2] - 1, spec.shape[3]  # drop the Nyquist bin: keeps freq a power-of-two multiple
        pad = (-frames) % 2**self.depth
        magnitude = F.pad(spec[:, :, :freq].abs(), (0, pad))
        return spec, self.net(magnitude)[..., :frames]

    def synthesize(self, spec: torch.Tensor, masks: torch.Tensor, length: int) -> torch.Tensor:
        """Masks (from `analyse`, possibly gated) applied to the STFT, then the inverse STFT: `(b, sources, c, length)`."""
        b, s, c = masks.shape[:3]
        masks = F.pad(masks, (0, 0, 0, 1))  # Nyquist bin: mask 0
        estimates = masks * spec.unsqueeze(1)
        out = torch.istft(
            estimates.reshape(b * s * c, *estimates.shape[3:]), self.n_fft, self.hop, window=self.window, length=length
        )
        return out.reshape(b, s, c, length)

    def forward(self, mix: torch.Tensor, presence: torch.Tensor | None = None) -> torch.Tensor:
        spec, masks = self.analyse(mix)
        if presence is not None:
            masks = gate_masks(masks, presence, self.net.mask)
        return self.synthesize(spec, masks, mix.shape[-1])


def separation_loss(estimates: torch.Tensor, sources: torch.Tensor, mask: torch.Tensor,
                    group_target: torch.Tensor | None = None, group_mask: torch.Tensor | None = None) -> torch.Tensor:
    """L1 on the waveforms, averaged over the sources whose target is known.
    A silent source is a valid target (the model should output silence).
    With `group_target` `(batch, channels, samples)` and `group_mask` `(batch, sources)`, an item whose group mask is
    set is also supervised on the sum of the grouped estimates (its individual targets are masked out): the
    sum counts as one more known target, so every supervised target weighs the same."""
    per_source = (estimates - sources).abs().mean(dim=(-1, -2))  # (batch, sources)
    total, count = (per_source * mask).sum(), mask.sum()
    if group_target is not None:
        grouped = (estimates * group_mask[:, :, None, None]).sum(dim=1)  # (batch, channels, samples)
        has = group_mask.amax(dim=1)  # (batch,)
        total = total + ((grouped - group_target).abs().mean(dim=(-1, -2)) * has).sum()
        count = count + has.sum()
    return total / count.clamp(min=1.0)


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
    group = torch.zeros(3)  # sum of SI-SDR, sum of the mixture's SI-SDR, examples: for the items scored on a group sum
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
            if "group_mask" in batch:  # items whose individual targets are unknown: score the sum of the grouped sources
                gm, ref = batch["group_mask"], batch["group_target"]
                ok = (gm.amax(dim=1) > 0) & (10.0 * torch.log10((ref**2).mean(dim=(-1, -2)) + 1e-12) > active_db)
                if ok.any():
                    group += torch.stack([(si_sdr((est * gm[:, :, None, None]).sum(dim=1), ref) * ok).sum(),
                                          (si_sdr(mix.cpu(), ref) * ok).sum(), ok.sum().float()])
    mean = sums / counts.clamp(min=1)
    base = base_sums / counts.clamp(min=1)
    out = {"si_sdr": mean.tolist(), "si_sdr_improvement": (mean - base).tolist(), "examples": counts.tolist()}
    if group[2] > 0:
        out["group"] = {"si_sdr": float(group[0] / group[2]), "si_sdr_improvement": float((group[0] - group[1]) / group[2]),
                        "examples": float(group[2])}
    return out
