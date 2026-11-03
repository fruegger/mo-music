"""Where does a training step spend its time: rendering the batch, or the model?

    python scripts/bench_pipeline.py --config configs/runpod-grid-base.yaml [--workers N] [--steps 20]

Prints the loader alone (s/batch with the config's workers), the model step alone on one fixed batch
(on cuda if there is one), and a per-step estimate: if the loader is clearly faster than the model step
the run is model-bound and pre-rendering the data cannot speed it up. Takes about a minute; run it on
a pod before paying for hours there.
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from mixer import MixConfig
from mixer.effects import _pedalboard
from mixer.registry import load_registry
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.train import TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="loader workers (default: the config's num_workers)")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t = cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = TASKS[cfg.get("task", "instruments")](cfg)
    ds = load_registry(cfg["sources"])[cfg["datasets"][0]["name"].lower()]
    mix = MixConfig.from_dict(cfg["mix"])
    data = SnippetDataset.from_dataset(ds, cfg["datasets"][0]["train_split"], mix, 50 * t["batch_size"],
                                       tasks=(task.name,), **task.dataset_kwargs)
    workers = args.workers if args.workers is not None else t["num_workers"]
    print(f"device={device} cpu threads={torch.get_num_threads()} workers={workers} "
          f"effects configured={mix.effects} pedalboard installed={_pedalboard() is not None}")
    if mix.effects and _pedalboard() is None:
        print("WARNING: effects are configured but pedalboard is missing: they are silently skipped")

    loader = DataLoader(data, t["batch_size"], shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, **({"prefetch_factor": t["prefetch_factor"]} if workers and "prefetch_factor" in t else {}))
    it = iter(loader)
    batch = next(it)  # warm-up: workers started, tracks listed
    start = time.perf_counter()
    for _ in range(args.steps):
        next(it)
    loader_s = (time.perf_counter() - start) / args.steps

    model = task.build().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for i in range(args.steps + 2):
        if i == 2:
            sync()
            start = time.perf_counter()
        loss = task.loss(model, batch, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync()
    model_s = (time.perf_counter() - start) / args.steps

    print(f"loader alone: {loader_s:.3f} s/batch   model step alone: {model_s:.3f} s/step")
    print("=> " + ("DATA-bound: pre-rendering can help" if loader_s > model_s else "MODEL-bound: pre-rendering cannot speed this up")
          + f" (a run takes about max of the two: {max(loader_s, model_s):.3f} s/step)")


if __name__ == "__main__":
    main()
