"""Training loop: `trainer --config configs/default.yaml`."""

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


def evaluate(model, loader, device) -> dict:
    model.eval()
    tp = fp = fn = 0.0
    with torch.no_grad():
        for audio, target in loader:
            pred = (model(audio.to(device)).sigmoid() > 0.5).float().cpu()
            tp += float((pred * target).sum())
            fp += float((pred * (1 - target)).sum())
            fn += float(((1 - pred) * target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-9)}


def export_onnx(model, path: Path, seconds: float, sample_rate: int = 44100) -> None:
    model.eval()
    dummy = torch.zeros(1, int(seconds * sample_rate))
    torch.onnx.export(
        model.cpu(), dummy, path, input_names=["audio"], output_names=["logits"],
        dynamic_axes={"audio": {0: "batch", 1: "samples"}},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trainer")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    registry = load_registry(cfg["sources"])
    mix = MixConfig(**{**cfg["mix"], "gain_db_range": tuple(cfg["mix"]["gain_db_range"])})
    datasets = [(registry[d["name"].lower()], d) for d in cfg["datasets"]]
    train_len = t["steps_per_epoch"] * t["batch_size"]

    train_set = ConcatDataset(
        [SnippetDataset.from_dataset(ds, d["train_split"], mix, train_len) for ds, d in datasets]
    )
    val_set = ConcatDataset(
        [
            SnippetDataset.from_dataset(ds, d["val_split"], mix, t["val_snippets"], t["val_seed_base"])
            for ds, d in datasets
        ]
    )
    train_loader = DataLoader(train_set, t["batch_size"], shuffle=True, num_workers=t["num_workers"], drop_last=True)
    val_loader = DataLoader(val_set, t["batch_size"], num_workers=t["num_workers"])

    model = InstrumentCnn().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=t["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(t["epochs"]):
        for d in train_set.datasets:
            d.set_epoch(epoch)
        model.train()
        for step, (audio, target) in enumerate(train_loader):
            if step >= t["steps_per_epoch"]:
                break
            loss = loss_fn(model(audio.to(device)), target.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scores = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, **scores})
        print(json.dumps(history[-1]))
        torch.save(model.state_dict(), out / "model.pt")

    export_onnx(model, out / "model.onnx", mix.seconds)
    run = {
        "config": cfg,
        "history": history,
        "licenses": {ds.name: ds.license for ds, _ in datasets},
        "non_commercial": any(ds.non_commercial for ds, _ in datasets),
    }
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
