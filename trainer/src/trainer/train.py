"""Training loop: `trainer --config configs/default.yaml` (instrument recognition)
or `trainer --config configs/separation.yaml` (source separation); the config's
`task` picks one."""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from mixer import MixConfig, load_registry
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .data import SnippetDataset, VocalMixDataset
from .model import InstrumentCnn
from .separation import Separator, evaluate_separation, separation_loss
from .sources import source_names


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


class SeparationTask:
    name = "separation"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sep = {"source_set": "musdb4", "n_fft": 2048, "hop": 512, "base_channels": 16, "depth": 3, **cfg.get("separation", {})}
        self.dataset_kwargs = {"source_set": self.sep["source_set"]}
        # Optional per-source loss weights ({name: weight}, default 1), e.g. to favour vocals.
        names = source_names(self.sep["source_set"])
        weights = self.sep.get("source_weights", {})
        unknown = set(weights) - set(names)
        if unknown:
            raise ValueError(f"source_weights names unknown sources {sorted(unknown)}; the set has {names}")
        self.source_weights = torch.tensor([float(weights.get(n, 1.0)) for n in names])

    def build(self) -> nn.Module:
        s = self.sep
        return Separator(len(source_names(s["source_set"])), 2, s["n_fft"], s["hop"], s["base_channels"], s["depth"])

    def loss(self, model, batch, device):
        estimates = model(batch["mix"].to(device))
        mask = batch["sources_mask"].to(device) * self.source_weights.to(device)
        return separation_loss(estimates, batch["sources"].to(device), mask)

    def evaluate(self, model, loader, device) -> dict:
        scores = evaluate_separation(model, loader, device)
        names = source_names(self.sep["source_set"])
        valid = [v for v, n in zip(scores["si_sdr_improvement"], scores["examples"]) if n > 0]
        return {
            "si_sdr": dict(zip(names, scores["si_sdr"])),
            "si_sdr_improvement": dict(zip(names, scores["si_sdr_improvement"])),
            "examples": dict(zip(names, scores["examples"])),
            "mean_si_sdr_improvement": sum(valid) / max(len(valid), 1),
        }

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
                    "note": "input: |STFT| without the Nyquist bin, (batch, 2, n_fft/2, frames); output masks in [0,1], "
                    "(batch, sources, 2, n_fft/2, frames), applied to the complex STFT (Nyquist bin: mask 0)",
                },
                indent=1,
            ),
            encoding="utf-8",
        )


TASKS = {"instruments": InstrumentTask, "separation": SeparationTask}


def evaluate_all(task, model, val_loader, device) -> dict:
    """One loader: its scores as they are. A dict of loaders (one per dataset): the
    scores of each under "validation", so Slakh2100 and MUSDB18 are not averaged
    into one number."""
    if not isinstance(val_loader, dict):
        return task.evaluate(model, val_loader, device)
    if len(val_loader) == 1:
        return task.evaluate(model, next(iter(val_loader.values())), device)
    return {"validation": {name: task.evaluate(model, loader, device) for name, loader in val_loader.items()}}


def build_train_set(registry, ds, d: dict, mix: MixConfig, length: int, kwargs: dict):
    """One dataset's training snippets. An optional `vocal_mix: {from, split, p, gain_db_range}` entry
    adds the vocals of another dataset (MUSDB18) to a share `p` of them, for datasets with no vocals."""
    base = SnippetDataset.from_dataset(ds, d["train_split"], mix, length, **kwargs)
    vm = d.get("vocal_mix")
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


def fit(task, model, train_loader, val_loader, epochs: int, steps_per_epoch: int, lr: float, device: str,
        on_epoch=None, optimiser=None, start_epoch: int = 0, history=None, on_epoch_end=None) -> list[dict]:
    """The training loop, separated from the data setup so it can be tested on small loaders.
    Runs epochs `start_epoch .. epochs - 1`; `history` is what earlier epochs reported."""
    optimiser = optimiser or torch.optim.Adam(model.parameters(), lr=lr)
    history = list(history or [])
    for epoch in range(start_epoch, epochs):
        if on_epoch is not None:
            on_epoch(epoch)
        model.train()
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            loss = task.loss(model, batch, device)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), **evaluate_all(task, model, val_loader, device)})
        print(json.dumps(history[-1]), flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch, history, optimiser)
    return history


def save_checkpoint(path: Path, model, optimiser, epochs_done: int, history: list[dict]) -> None:
    """Everything needed to continue: weights, optimiser state, finished epochs, their results."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimiser.state_dict(), "epoch": epochs_done, "history": history}, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint behind if the process is killed


def load_checkpoint(path: Path, model, optimiser=None, epochs_done: int | None = None) -> tuple[int, list[dict]]:
    """Loads a `save_checkpoint` file, or a bare `model.pt` state dict (weights only: the optimiser
    starts fresh and `epochs_done` says how many epochs it stands for). Returns `(epochs done, history)`."""
    data = torch.load(path, map_location="cpu")
    if "model" in data:
        model.load_state_dict(data["model"])
        if optimiser is not None:
            optimiser.load_state_dict(data["optimizer"])
        return data["epoch"], data["history"]
    if epochs_done is None:
        raise ValueError(f"{path} holds weights only: say how many epochs it stands for (--start-epoch)")
    model.load_state_dict(data)
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
    datasets = [(registry[d["name"].lower()], d) for d in cfg["datasets"]]
    # Datasets only used for their vocals still count for the licences.
    used = [ds for ds, _ in datasets] + [registry[d["vocal_mix"]["from"].lower()] for d in cfg["datasets"] if d.get("vocal_mix")]
    train_len = t["steps_per_epoch"] * t["batch_size"]
    kwargs = {"tasks": (task.name,), **task.dataset_kwargs}

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
            SnippetDataset.from_dataset(ds, d["val_split"], mix, t["val_snippets"], t["val_seed_base"], **kwargs),
            t["batch_size"],
            num_workers=t.get("val_workers", 0),
        )
        for ds, d in datasets
    }

    model = task.build().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=t["lr"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    start_epoch, history = 0, []
    if args.resume:
        start_epoch, history = load_checkpoint(args.resume, model, optimiser, args.start_epoch)
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
        save_checkpoint(out / "checkpoint.pt", model, optimiser, epoch + 1, history)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "run.json").write_text(json.dumps(run_record(False, history), indent=1), encoding="utf-8")

    history = fit(task, model, train_loader, val_loader, t["epochs"], t["steps_per_epoch"], t["lr"], device,
                  on_epoch, optimiser, start_epoch, history, on_epoch_end)
    task.export(model, out / "model.onnx", mix)
    (out / "run.json").write_text(json.dumps(run_record(True, history), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
