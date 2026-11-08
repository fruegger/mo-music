"""Training loop: `trainer --config configs/default.yaml` (instrument recognition)
or `trainer --config configs/separation.yaml` (source separation); the config's
`task` picks one."""

import argparse
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from mixer import MixConfig, load_registry
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .data import RemixDataset, SnippetDataset, VocalMixDataset
from .model import InstrumentCnn
from .separation import Separator, evaluate_separation, separation_loss
from .sources import GROUPS, source_names
from .tags import load_other_tags


def level_db(x: torch.Tensor) -> torch.Tensor:
    """RMS level in dB of `(..., channels, samples)`, over the last two dimensions."""
    return 10.0 * torch.log10((x**2).mean(dim=(-1, -2)) + 1e-12)


class InstrumentTask:
    name = "instruments"
    dataset_kwargs: dict = {}

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def build(self) -> nn.Module:
        return InstrumentCnn()

    def loss(self, model, batch, device):
        """BCE over the classes whose target is known (see `data.loss_mask`)."""
        mask = batch["instruments_mask"].to(device)
        loss = self.bce(model(batch["mono"].to(device)), batch["instruments"].to(device))
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def evaluate(self, model, loader, device) -> dict:
        model.eval()
        tp = fp = fn = 0.0
        with torch.no_grad():
            for batch in loader:
                pred = (model(batch["mono"].to(device)).sigmoid() > 0.5).float().cpu()
                target, mask = batch["instruments"], batch["instruments_mask"]
                tp += float((pred * target * mask).sum())
                fp += float((pred * (1 - target) * mask).sum())
                fn += float(((1 - pred) * target * mask).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-9)}

    def export(self, model, path: Path, mix: MixConfig, sample_rate: int = 44100) -> None:
        model.eval()
        dummy = torch.zeros(1, int(mix.seconds * sample_rate))
        torch.onnx.export(
            model.cpu(), dummy, path, input_names=["audio"], output_names=["logits"],
            dynamic_axes={"audio": {0: "batch", 1: "samples"}},
        )


class PresenceTask:
    """Which sources are audible in a mix: multi-label, one output per source of the set (`musdb4`).
    A source counts as present when its target is above `active_db` (-50 dB RMS, the evaluator's definition).
    With `present_rel_db` (say -40) the label is relative to the mix instead: present if the source is at most that
    many dB below the mix, absent if it is more than `absent_rel_db` (-60) below it, and the zone in between is
    ignored (weight 0): a source 45 dB under the mix is neither clearly there nor clearly not, and an inaudible
    one costs nothing to drop. `pool` ("mean" / "maxmean") is the model's pooling over time.
    Trains on the separation items (mix and sources), so remixing and the other dataset wrappers apply; the
    model listens to the mono mix. Used to tell the separator which masks to keep (`Separator(..., presence=)`).

    With a source set that has a group (`musdb6`: guitar, keys, other) MUSDB18 items do not know the three sources
    individually, only their sum (`group_mask`). Then, per item and by the audibility of the SUM (the same relative
    rule): inaudible = all three absent (known); in the ignored zone = left out; audible = the individual labels come
    from the hand-made `source_tags` if the song has some (untagged instrument: absent; tagged: present, with
    weight `tag_weight` because the instrument may be silent in this snippet; only in `unsure`: left out), and
    otherwise from one "at least one of the three is present" term on the noisy-OR of their probabilities.
    Config: `presence.tags` (the CSV of `trainer.tags`) and `presence.tag_weight` (0.5). With `presence.promote_single`
    a song whose tags name exactly one of the three (guitar only, keys only, ...) is not ambiguous at all: its
    unlabelled stem is moved to that source and all six targets are known (the dataset does it, see `SnippetDataset`)."""

    name = "presence"
    dataset_task = "separation"  # the data it trains on

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pres = {"source_set": "musdb4", "active_db": -50.0, "n_mels": 96, "present_rel_db": None,
                     "absent_rel_db": -60.0, "pool": "mean", "tags": None, "tag_weight": 0.5,
                     "promote_single": False, **cfg.get("presence", {})}
        self.names = source_names(self.pres["source_set"])
        self.group = GROUPS.get(self.pres["source_set"])
        self.bce = nn.BCEWithLogitsLoss()

    @property
    def dataset_kwargs(self) -> dict:
        """Read when the data is built, not in `__init__`: a recogniser loaded for inference (`load_recogniser`)
        must not need the tag file."""
        kwargs = {"source_set": self.pres["source_set"]}
        if self.pres["tags"]:
            kwargs["source_tags"] = load_other_tags(Path(self.pres["tags"]), self.pres["source_set"])
            kwargs["promote_single"] = bool(self.pres["promote_single"])
        return kwargs

    def build(self) -> nn.Module:
        return InstrumentCnn(n_mels=self.pres["n_mels"], n_classes=len(self.names), pool=self.pres["pool"])

    def group_state(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """`(audible, inaudible)`, both `(batch,)` bool: whether the SUM of the grouped sources is clearly there or
        clearly not (neither: the ignored zone). Same definition as the single sources."""
        level = level_db(batch["group_target"])
        if self.pres["present_rel_db"] is None:
            audible = level > self.pres["active_db"]
            return audible, ~audible
        relative = level - level_db(batch["mix"])
        audible = (relative >= self.pres["present_rel_db"]) & (level > -80.0)
        return audible, (relative < self.pres["absent_rel_db"]) | (level <= -80.0)

    def targets(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """`(labels, weights)`, both `(batch, sources)`: 1.0 where the source is audible, and 0.0 weight where the
        label is left out (the zone between present and absent of the relative definition, or an unknown source)."""
        level = level_db(batch["sources"])
        if self.pres["present_rel_db"] is None:
            labels, weights = (level > self.pres["active_db"]).float(), torch.ones_like(level)
        else:
            relative = level - level_db(batch["mix"]).unsqueeze(1)
            present = (relative >= self.pres["present_rel_db"]) & (level > -80.0)  # a silent mix has no present source
            absent = ~present & ((relative < self.pres["absent_rel_db"]) | (level <= -80.0))
            labels, weights = present.float(), (present | absent).float()
        if self.group is not None and "group_mask" in batch:
            unknown = batch["group_mask"] > 0  # the item only knows the sum of these sources
            audible, inaudible = self.group_state(batch)
            labels, weights = labels * ~unknown, weights * ~unknown
            weights = weights + (unknown & inaudible[:, None]).float()  # nothing audible in the sum: all absent
            if "source_tags" in batch:
                tags = batch["source_tags"]
                tagged = unknown & audible[:, None] & (tags >= 0)
                labels = labels + (tagged & (tags == 1)).float()
                weights = weights + (tagged & (tags == 0)).float() + (tagged & (tags == 1)).float() * self.pres["tag_weight"]
        return labels, weights

    def group_weights(self, batch) -> torch.Tensor | None:
        """`(batch,)` weight of the "at least one grouped source is present" term: 1 for an item that only knows
        the sum, where the sum is audible and no tags say what is in it."""
        if self.group is None or "group_mask" not in batch:
            return None
        weight = ((batch["group_mask"].amax(dim=1) > 0) & self.group_state(batch)[0]).float()
        if "source_tags" in batch:
            weight = weight * (batch["source_tags"][:, self.group].amax(dim=1) < 0)
        return weight

    def group_labels(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """`(labels, weights)`, both `(batch,)`, for scoring the group as one source: for an item that only knows
        the sum, present if the sum is audible; otherwise present if any of its sources is."""
        labels, weights = self.targets(batch)
        idx = self.group
        audible, inaudible = self.group_state(batch)
        unknown = batch["group_mask"].amax(dim=1) > 0
        known_labels, known_weights = labels[:, idx].amax(dim=1), weights[:, idx].amin(dim=1)
        return torch.where(unknown, audible.float(), known_labels), torch.where(unknown, (audible | inaudible).float(), known_weights)

    def labels(self, batch) -> torch.Tensor:
        """`(batch, sources)` 1.0 where the source is audible."""
        return self.targets(batch)[0]

    def loss(self, model, batch, device):
        labels, weights = self.targets(batch)
        logits = model(batch["mix"].mean(dim=1).to(device))
        per = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="none")
        total, count = (per * weights.to(device)).sum(), weights.sum().to(device)
        group = self.group_weights(batch)
        if group is not None:
            at_least_one = 1.0 - (1.0 - logits[:, self.group].sigmoid()).prod(dim=1)  # noisy-OR
            total = total - (at_least_one.clamp(min=1e-6).log() * group.to(device)).sum()
            count = count + group.sum().to(device)
        return total / count.clamp(min=1.0)

    def evaluate(self, model, loader, device) -> dict:
        """Per-source precision, recall and F1 at probability 0.5, and the share of snippets where the source is present."""
        model.eval()
        tp = fp = fn = n = present = 0.0
        gtp = gfp = gfn = 0.0  # the group (guitar, keys, other) scored as one source, if the set has one
        with torch.no_grad():
            for batch in loader:
                prob = model(batch["mix"].mean(dim=1).to(device)).sigmoid().cpu()
                pred = (prob > 0.5).float()
                target, weight = self.targets(batch)
                tp, fp = tp + (pred * target * weight).sum(0), fp + (pred * (1 - target) * weight).sum(0)
                fn, present, n = fn + ((1 - pred) * target * weight).sum(0), present + (target * weight).sum(0), n + weight.sum(0)
                if self.group is not None and "group_mask" in batch:
                    g_target, g_weight = self.group_labels(batch)
                    g_pred = (1.0 - (1.0 - prob[:, self.group]).prod(dim=1) > 0.5).float()
                    gtp, gfp = gtp + float((g_pred * g_target * g_weight).sum()), gfp + float((g_pred * (1 - g_target) * g_weight).sum())
                    gfn += float(((1 - g_pred) * g_target * g_weight).sum())
        precision, recall = tp / (tp + fp).clamp(min=1.0), tp / (tp + fn).clamp(min=1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
        per = lambda v: dict(zip(self.names, [round(float(x), 4) for x in v]))
        scores = {"precision": per(precision), "recall": per(recall), "f1": per(f1),
                  "present_share": per(present / n.clamp(min=1.0) if torch.is_tensor(n) else present / max(n, 1)), "mean_f1": float(f1.mean())}
        if gtp + gfp + gfn > 0:
            g_precision, g_recall = gtp / max(gtp + gfp, 1.0), gtp / max(gtp + gfn, 1.0)
            scores["group"] = {"precision": round(g_precision, 4), "recall": round(g_recall, 4),
                               "f1": round(2 * g_precision * g_recall / max(g_precision + g_recall, 1e-9), 4)}
        return scores

    def export(self, model, path: Path, mix: MixConfig, sample_rate: int = 44100) -> None:
        """Mono audio in, one logit per source out; `<path>.json` lists the sources."""
        model.eval()
        torch.onnx.export(
            model.cpu(), torch.zeros(1, int(mix.seconds * sample_rate)), path, input_names=["audio"], output_names=["logits"],
            dynamic_axes={"audio": {0: "batch", 1: "samples"}},
        )
        path.with_suffix(".json").write_text(
            json.dumps({"sources": self.names, "sample_rate": sample_rate, "seconds": mix.seconds,
                        "active_db": self.pres["active_db"], "note": "sigmoid(logits) = probability the source is audible"}, indent=1),
            encoding="utf-8",
        )


class Ema:
    """Exponential moving average of a model's weights (`decay` close to 1, e.g. 0.999). The average is smoother and
    usually better than the raw weights, so it is what gets evaluated and saved. Early on the decay is ramped up
    so the average does not drag the random initial weights along. Non-float state (batch counters) is copied."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay, self.updates = decay, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            else:
                self.shadow[k].copy_(v)

    @contextmanager
    def applied(self, model: nn.Module):
        """Inside the block the model holds the averaged weights; afterwards its own weights come back."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(backup)

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.shadow, self.updates = state["shadow"], state["updates"]


class SeparationTask:
    name = "separation"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sep = {"source_set": "musdb4", "n_fft": 2048, "hop": 512, "base_channels": 16, "depth": 3, "mask": "softmax", **cfg.get("separation", {})}
        # Optional per-source loss weights ({name: weight}, default 1), e.g. to favour vocals.
        names = source_names(self.sep["source_set"])
        weights = self.sep.get("source_weights", {})
        unknown = set(weights) - set(names)
        if unknown:
            raise ValueError(f"source_weights names unknown sources {sorted(unknown)}; the set has {names}")
        self.source_weights = torch.tensor([float(weights.get(n, 1.0)) for n in names])
        present = self.sep.get("present_weights", {})  # {name: weight} on the audible items of a source only
        if set(present) - set(names):
            raise ValueError(f"present_weights names unknown sources {sorted(set(present) - set(names))}; the set has {names}")
        self.present_weights = torch.tensor([float(present.get(n, 1.0)) for n in names]) if present else None

    @property
    def dataset_kwargs(self) -> dict:
        """Read when the data is built (so a model loaded for inference never needs the tag file). With `separation.tags`
        (the CSV of `trainer.tags`) a MUSDB18 `other` stem tagged with exactly one of the group sources (guitar only, ...)
        is that source when `separation.promote_single` is on: its targets are then all known instead of only their sum."""
        kwargs = {"source_set": self.sep["source_set"]}
        if self.sep.get("tags"):
            kwargs["source_tags"] = load_other_tags(Path(self.sep["tags"]), self.sep["source_set"])
            kwargs["promote_single"] = bool(self.sep.get("promote_single", False))
        return kwargs

    def build(self) -> nn.Module:
        s = self.sep
        return Separator(len(source_names(s["source_set"])), 2, s["n_fft"], s["hop"], s["base_channels"], s["depth"], s["mask"])

    def loss(self, model, batch, device):
        estimates = model(batch["mix"].to(device))
        mask = batch["sources_mask"].to(device) * self.source_weights.to(device)
        group = (batch["group_target"].to(device), batch["group_mask"].to(device)) if "group_mask" in batch else (None, None)
        return separation_loss(estimates, batch["sources"].to(device), mask, *group, silent_weight=float(self.sep.get("silent_weight", 1.0)),
                               present_weights=None if self.present_weights is None else self.present_weights.to(device))

    def evaluate(self, model, loader, device) -> dict:
        scores = evaluate_separation(model, loader, device)
        names = source_names(self.sep["source_set"])
        valid = [v for v, n in zip(scores["si_sdr_improvement"], scores["examples"]) if n > 0]
        out = {
            "si_sdr": dict(zip(names, scores["si_sdr"])),
            "si_sdr_improvement": dict(zip(names, scores["si_sdr_improvement"])),
            "examples": dict(zip(names, scores["examples"])),
        }
        if "group" in scores:  # MUSDB18 items: guitar + keys + other scored as one source
            out["group"] = scores["group"]
            valid.append(scores["group"]["si_sdr_improvement"])
        out["mean_si_sdr_improvement"] = sum(valid) / max(len(valid), 1)
        return out

    def export(self, model, path: Path, mix: MixConfig, sample_rate: int = 44100) -> None:
        """Exports the mask network (magnitude spectrogram in, masks out). The
        STFT, the mask multiplication and the inverse STFT are left to the
        runtime; `<path>.json` records the parameters it needs."""
        model.eval()
        net, sep = model.net.cpu(), self.sep
        freq, frames = sep["n_fft"] // 2, 2 ** sep["depth"] * 4
        dummy = torch.rand(1, 2, freq, frames)
        torch.onnx.export(
            net, dummy, path, input_names=["magnitude"], output_names=["masks"],
            dynamic_axes={"magnitude": {0: "batch", 3: "frames"}, "masks": {0: "batch", 4: "frames"}},
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "sources": source_names(sep["source_set"]),
                    "n_fft": sep["n_fft"],
                    "hop": sep["hop"],
                    "sample_rate": sample_rate,
                    "channels": 2,
                    "frames_multiple_of": 2 ** sep["depth"],
                    "mask": sep["mask"],
                    "note": "input: |STFT| without the Nyquist bin, (batch, 2, n_fft/2, frames); output masks in [0,1], "
                    "(batch, sources, 2, n_fft/2, frames), applied to the complex STFT (Nyquist bin: mask 0)",
                },
                indent=1,
            ),
            encoding="utf-8",
        )


TASKS = {"instruments": InstrumentTask, "separation": SeparationTask, "presence": PresenceTask}


def evaluate_all(task, model, val_loader, device) -> dict:
    """One loader: its scores as they are. A dict of loaders (one per dataset): the
    scores of each under "validation", so Slakh2100 and MUSDB18 are not averaged
    into one number."""
    if not isinstance(val_loader, dict):
        return task.evaluate(model, val_loader, device)
    if len(val_loader) == 1:
        return task.evaluate(model, next(iter(val_loader.values())), device)
    return {"validation": {name: task.evaluate(model, loader, device) for name, loader in val_loader.items()}}


def validation_mix(cfg: dict) -> MixConfig:
    """The mix the validation snippets are rendered with: the config's `mix` with the optional `val_mix`
    entries on top (e.g. `val_mix: {effects: false}` trains with effects but scores on clean mixes, like
    real recordings). Without `val_mix` validation uses the training mix."""
    return MixConfig.from_dict({**cfg["mix"], **cfg.get("val_mix", {})})


def build_train_set(registry, ds, d: dict, mix: MixConfig, length: int, kwargs: dict):
    """One dataset's training snippets. An optional `vocal_mix: {from, split, p, gain_db_range}` entry
    adds the vocals of another dataset (MUSDB18) to a share `p` of them, for datasets with no vocals.
    An optional `remix: {p, tempo_pitch}` entry builds a share `p` of them from different songs per source
    (see `RemixDataset`); the songs are rendered without the config's tempo and pitch changes (they are about
    20 times slower than everything else) unless `tempo_pitch: true`. It cannot be combined with `vocal_mix`."""
    base = SnippetDataset.from_dataset(ds, d["train_split"], mix, length, **kwargs)
    vm, rm = d.get("vocal_mix"), d.get("remix")
    if vm and rm is not None:
        raise ValueError("a dataset can have vocal_mix or remix, not both")
    if rm is not None:  # `remix: {}` means all defaults
        rm = rm or {}
        donor_mix = mix if rm.get("tempo_pitch", False) else replace(mix, p_tempo=0.0, p_pitch=0.0)
        donors = base if donor_mix == mix else SnippetDataset.from_dataset(ds, d["train_split"], donor_mix, length, **kwargs)
        return RemixDataset(base, rm.get("p", 0.5), seed_base=20_000_000, donors=donors)
    if not vm:
        return base
    src = registry[vm["from"].lower()]
    vocals = SnippetDataset.from_dataset(src, vm.get("split", "train"), mix, length, seed_base=10_000_000, **kwargs)
    return VocalMixDataset(base, vocals, vm.get("p", 0.5), vm.get("gain_db_range", (-6.0, 6.0)),
                           source_set=kwargs["source_set"])


def split_lengths(total: int, weights: list[float]) -> list[int]:
    """Sizes of the per-dataset training sets, in proportion to `weights`. The loader shuffles over
    all of them, so a dataset's share of the batches follows its weight."""
    return [max(1, round(total * w)) for w in weights]


def lr_at(step: int, total_steps: int, lr: float, warmup_steps: int = 0, final_fraction: float = 1.0) -> float:
    """Learning rate at a global step: a linear warm-up over `warmup_steps`, then a cosine decay to
    `final_fraction * lr` at the last step. The defaults (no warm-up, fraction 1) keep the rate constant.
    A pure function of the step, so a resumed run continues the schedule without saving any state."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return lr * (final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def fit(task, model, train_loader, val_loader, epochs: int, steps_per_epoch: int, lr: float, device: str,
        on_epoch=None, optimiser=None, start_epoch: int = 0, history=None, on_epoch_end=None,
        warmup_steps: int = 0, lr_final_fraction: float = 1.0, grad_clip: float | None = None,
        ema: Ema | None = None) -> list[dict]:
    """The training loop, separated from the data setup so it can be tested on small loaders.
    Runs epochs `start_epoch .. epochs - 1`; `history` is what earlier epochs reported.
    `warmup_steps` / `lr_final_fraction` shape the learning rate (see `lr_at`); `grad_clip` limits the
    gradient norm. With an `ema`, the averaged weights are updated after every step and used for the evaluation."""
    optimiser = optimiser or torch.optim.Adam(model.parameters(), lr=lr)
    history = list(history or [])
    for epoch in range(start_epoch, epochs):
        if on_epoch is not None:
            on_epoch(epoch)
        model.train()
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            for group in optimiser.param_groups:
                group["lr"] = lr_at(epoch * steps_per_epoch + step, epochs * steps_per_epoch, lr, warmup_steps, lr_final_fraction)
            loss = task.loss(model, batch, device)
            optimiser.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            if ema is not None:
                ema.update(model)
        with ema.applied(model) if ema is not None else nullcontext():
            scores = evaluate_all(task, model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), **scores})
        print(json.dumps(history[-1]), flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch, history, optimiser)
    return history


def save_checkpoint(path: Path, model, optimiser, epochs_done: int, history: list[dict], ema: Ema | None = None) -> None:
    """Everything needed to continue: weights, optimiser state, finished epochs, their results, the EMA if any."""
    tmp = path.with_suffix(".tmp")
    state = {"model": model.state_dict(), "optimizer": optimiser.state_dict(), "epoch": epochs_done, "history": history}
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint behind if the process is killed


def load_checkpoint(path: Path, model, optimiser=None, epochs_done: int | None = None, ema: Ema | None = None) -> tuple[int, list[dict]]:
    """Loads a `save_checkpoint` file, or a bare `model.pt` state dict (weights only: the optimiser
    starts fresh and `epochs_done` says how many epochs it stands for). Returns `(epochs done, history)`."""
    data = torch.load(path, map_location="cpu")
    if "model" in data:
        model.load_state_dict(data["model"])
        if optimiser is not None:
            optimiser.load_state_dict(data["optimizer"])
        if ema is not None and "ema" in data:
            ema.load_state_dict(data["ema"])
        return data["epoch"], data["history"]
    if epochs_done is None:
        raise ValueError(f"{path} holds weights only: say how many epochs it stands for (--start-epoch)")
    model.load_state_dict(data)
    if ema is not None:
        ema.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}  # weights only: start the average here
    return epochs_done, []


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainer")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--resume", type=Path, help="checkpoint.pt to continue from (or a bare model.pt, with --start-epoch)")
    parser.add_argument("--start-epoch", type=int, help="epochs already done when resuming from a bare model.pt")
    args = parser.parse_args(argv)
    # torch.onnx prints emoji progress marks, which a cp1252 Windows console cannot encode.
    sys.stdout.reconfigure(errors="replace")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = TASKS[cfg.get("task", "instruments")](cfg)

    registry = load_registry(cfg["sources"])
    mix = MixConfig.from_dict(cfg["mix"])
    val_mix = validation_mix(cfg)
    datasets = [(registry[d["name"].lower()], d) for d in cfg["datasets"]]
    # Datasets only used for their vocals still count for the licences.
    used = [ds for ds, _ in datasets] + [registry[d["vocal_mix"]["from"].lower()] for d in cfg["datasets"] if d.get("vocal_mix")]
    train_len = t["steps_per_epoch"] * t["batch_size"]
    kwargs = {"tasks": (getattr(task, "dataset_task", task.name),), **task.dataset_kwargs}

    # A dataset's optional `weight` (default 1) sets its share of the training batches.
    lengths = split_lengths(train_len, [float(d.get("weight", 1.0)) for _, d in datasets])
    train_set = ConcatDataset([build_train_set(registry, ds, d, mix, n, kwargs) for (ds, d), n in zip(datasets, lengths)])
    # Persistent workers: each lists the tracks once (a minute for Slakh's train split), not every epoch.
    workers = {"num_workers": t["num_workers"], "persistent_workers": t["num_workers"] > 0}
    if t["num_workers"] > 0 and "prefetch_factor" in t:
        workers["prefetch_factor"] = t["prefetch_factor"]
    train_loader = DataLoader(train_set, t["batch_size"], shuffle=True, drop_last=True, **workers)
    # One validation loader per dataset, in the main process (`val_workers` to change): extra
    # worker pools would cost memory, and 48 snippets are quick.
    val_loader = {
        ds.name: DataLoader(
            SnippetDataset.from_dataset(ds, d["val_split"], val_mix, t["val_snippets"], t["val_seed_base"], **kwargs),
            t["batch_size"],
            num_workers=t.get("val_workers", 0),
        )
        for ds, d in datasets
    }

    model = task.build().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=t["lr"])
    ema = Ema(model, t["ema_decay"]) if t.get("ema_decay") else None  # evaluate and save the averaged weights
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    start_epoch, history = 0, []
    if args.resume:
        start_epoch, history = load_checkpoint(args.resume, model, optimiser, args.start_epoch, ema)
        print(f"resumed from {args.resume}: {start_epoch} epochs done, continuing to {t['epochs']}", flush=True)

    def run_record(finished: bool, history: list[dict]) -> dict:
        return {
            "task": task.name,
            "finished": finished,
            "resumed_from": str(args.resume) if args.resume else None,
            "epochs_before_resume": start_epoch,
            "config": cfg,
            "history": history,
            "licenses": {ds.name: ds.license for ds in used},
            "non_commercial": any(ds.non_commercial for ds in used),
        }

    def on_epoch(epoch):
        for d in train_set.datasets:
            d.set_epoch(epoch)

    def on_epoch_end(epoch, history, optimiser):
        # After every epoch: a killed run loses at most one epoch.
        save_checkpoint(out / "checkpoint.pt", model, optimiser, epoch + 1, history, ema)
        torch.save(ema.shadow if ema is not None else model.state_dict(), out / "model.pt")
        (out / "run.json").write_text(json.dumps(run_record(False, history), indent=1), encoding="utf-8")

    history = fit(task, model, train_loader, val_loader, t["epochs"], t["steps_per_epoch"], t["lr"], device,
                  on_epoch, optimiser, start_epoch, history, on_epoch_end,
                  warmup_steps=t.get("warmup_steps", 0), lr_final_fraction=t.get("lr_final_fraction", 1.0),
                  grad_clip=t.get("grad_clip"), ema=ema)
    with ema.applied(model) if ema is not None else nullcontext():
        task.export(model, out / "model.onnx", mix)
    (out / "run.json").write_text(json.dumps(run_record(True, history), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
