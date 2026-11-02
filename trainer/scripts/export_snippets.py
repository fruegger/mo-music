#!/usr/bin/env python3
"""Write a few training snippets as WAV files to listen to: the mix and each target source, exactly as the model sees them.

    python scripts/export_snippets.py --config configs/separation-cpu-vocalmix-lr.yaml --out DIR --n 6

Makes three groups: `slakh+vocals` (Slakh with a MUSDB18 vocal mixed in, always, whatever the config's p), `slakh` (plain)
and `musdb` (plain). Needs the datasets in the config's sources log. A `levels.txt` lists the level of each source."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from mixer import MixConfig, load_registry

from trainer.sources import source_names
from trainer.train import build_train_set


def db(x: torch.Tensor) -> float:
    return float(10 * torch.log10((x**2).mean() + 1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    registry = load_registry(cfg["sources"])
    mix = MixConfig.from_dict(cfg["mix"])
    source_set = cfg.get("separation", {}).get("source_set", "musdb4")
    names = source_names(source_set)
    kwargs = {"tasks": ("separation",), "source_set": source_set}

    groups = {}
    for d in cfg["datasets"]:
        if d["name"].lower().startswith("slakh") and d.get("vocal_mix"):
            groups["slakh+vocals"] = {**d, "vocal_mix": {**d["vocal_mix"], "p": 1.0}}
            groups["slakh"] = {k: v for k, v in d.items() if k != "vocal_mix"}
        elif d["name"].lower().startswith("musdb"):
            groups["musdb"] = d

    lines = []
    for group, d in groups.items():
        ds = build_train_set(registry, registry[d["name"].lower()], d, mix, args.n, kwargs)
        for i in range(args.n):
            item = ds[i]
            folder = args.out / f"{group}_{i}"
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / "mix.wav", item["mix"].numpy().T, 44100)
            levels = [f"mix {db(item['mix']):.0f} dB"]
            for name, audio in zip(names, item["sources"]):
                sf.write(folder / f"{name}.wav", audio.numpy().T, 44100)
                levels.append(f"{name} {db(audio):.0f} dB")
            lines.append(f"{group}_{i}: " + ", ".join(levels))
            print(lines[-1], flush=True)
    (args.out / "levels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("written to", args.out)


if __name__ == "__main__":
    main()
