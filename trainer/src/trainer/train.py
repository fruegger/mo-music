"""Training loop: `trainer --config configs/default.yaml` (instrument recognition)
or `trainer --config configs/separation.yaml` (source separation); the config's
`task` picks one."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from mixer import MixConfig, load_registry
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .data import SnippetDataset
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

    def build(self) -> nn.Module:
        s = self.sep
        return Separator(len(source_names(s["source_set"])), 2, s["n_fft"], s["hop"], s["base_channels"], s["depth"])

    def loss(self, model, batch, device):
        estimates = model(batch["mix"].to(device))
        return separation_loss(estimates, batch["sources"].to(device), batch["sources_mask"].to(device))

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


def fit(task, model, train_loader, val_loader, epochs: int, steps_per_epoch: int, lr: float, device: str,
        on_epoch=None) -> list[dict]:
    """The training loop, separated from the data setup so it can be tested on small loaders."""
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
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
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), **task.evaluate(model, val_loader, device)})
        print(json.dumps(history[-1]))
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainer")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = TASKS[cfg.get("task", "instruments")](cfg)

    registry = load_registry(cfg["sources"])
    mix = MixConfig.from_dict(cfg["mix"])
    datasets = [(registry[d["name"].lower()], d) for d in cfg["datasets"]]
    train_len = t["steps_per_epoch"] * t["batch_size"]
    kwargs = {"tasks": (task.name,), **task.dataset_kwargs}

    train_set = ConcatDataset(
        [SnippetDataset.from_dataset(ds, d["train_split"], mix, train_len, **kwargs) for ds, d in datasets]
    )
    val_set = ConcatDataset(
        [
            SnippetDataset.from_dataset(ds, d["val_split"], mix, t["val_snippets"], t["val_seed_base"], **kwargs)
            for ds, d in datasets
        ]
    )
    train_loader = DataLoader(train_set, t["batch_size"], shuffle=True, num_workers=t["num_workers"], drop_last=True)
    val_loader = DataLoader(val_set, t["batch_size"], num_workers=t["num_workers"])

    model = task.build().to(device)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    def on_epoch(epoch):
        for d in train_set.datasets:
            d.set_epoch(epoch)
        if epoch:
            torch.save(model.state_dict(), out / "model.pt")

    history = fit(task, model, train_loader, val_loader, t["epochs"], t["steps_per_epoch"], t["lr"], device, on_epoch)
    torch.save(model.state_dict(), out / "model.pt")
    task.export(model, out / "model.onnx", mix)
    run = {
        "task": task.name,
        "config": cfg,
        "history": history,
        "licenses": {ds.name: ds.license for ds, _ in datasets},
        "non_commercial": any(ds.non_commercial for ds, _ in datasets),
    }
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
