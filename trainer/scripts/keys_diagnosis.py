"""Where does the energy of a source go? For a six-source separator: how the estimate of each source is made up of the TRUE sources.

    python scripts/keys_diagnosis.py --run runs/local-musdb6-long-promote-seed0 [--dataset studio] [--seconds 10] [--snippets 64]

For every test snippet in which the chosen source (default `keys`) is audible, each estimate E_j is regressed on the true
sources S_i of the same snippet (least squares, one gain per source). Reported, energy-weighted over the snippets:

  made up of   per output j, the share of its energy that comes from each true source (rows sum to ~100 %, the rest is
               unexplained: artefacts and sources not in the snippet)
  goes to      per true source i, the share of ITS energy that ends up in each output j (gain^2 * |S_i|^2 / |S_i|^2 summed)
  level        the chosen source's estimate level relative to its target, and its SI-SDR improvement

Needs exact targets: the studio test songs (all six sources known), or MUSDB18 songs promoted by the hand tags.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mixer import MixConfig, load_registry
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.separate import load_model
from trainer.separation import si_sdr
from trainer.train import SeparationTask


def level_db(x: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10((x**2).mean(dim=(-1, -2)) + 1e-12)


def decompose(estimates: torch.Tensor, sources: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """`(sources, channels, samples)` each -> coef `(outputs, true sources)`: the least-squares gain of every true source in
    every estimate, and the energy of every estimate and of every true source (for the weighting)."""
    e = estimates.reshape(estimates.shape[0], -1).double()
    s = sources.reshape(sources.shape[0], -1).double()
    coef = torch.linalg.lstsq(s.T, e.T).solution.T  # (outputs, true sources)
    return coef.numpy(), s.pow(2).sum(dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", default="studio")
    parser.add_argument("--source", default="keys")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--snippets", type=int, default=64)
    parser.add_argument("--sources", type=Path, default=Path("configs/studio-sources.log"))
    args = parser.parse_args()

    cfg = json.loads((args.run / "run.json").read_text(encoding="utf-8"))["config"]
    model, names = load_model(args.run)
    k = names.index(args.source)
    kwargs = SeparationTask(cfg).dataset_kwargs  # includes the hand tags and promotion if the run used them
    mix_cfg = MixConfig.from_dict({**cfg["mix"], "effects": False, "p_tempo": 0.0, "p_pitch": 0.0, "seconds": args.seconds,
                                   "p_stem_dropped": 0.0})
    dataset = load_registry(str(args.sources))[args.dataset if args.dataset != "musdb18" else "musdb18-prepared"]
    data = SnippetDataset.from_dataset(dataset, "test", mix_cfg, args.snippets, 1_000_000, tasks=("separation",), **kwargs)
    print(f"{args.run.name}: {args.dataset} test, {len(data.tracks)} songs, {args.snippets} snippets of {args.seconds:.0f} s, source={args.source}")

    n = len(names)
    made, goes, weight_e, weight_s = np.zeros((n, n)), np.zeros((n, n)), np.zeros(n), np.zeros(n)
    gains, imps, used, per_song = [], [], 0, {}
    with torch.no_grad():
        for i in range(args.snippets):
            item = data[i]
            mix, src, mask = item["mix"][None], item["sources"], item["sources_mask"]
            if mask.min() == 0 or level_db(src[k]) - level_db(item["mix"]) < -40.0:
                continue  # unknown targets, or the chosen source is not audible
            spec, masks = model.analyse(mix)
            est = model.synthesize(spec, masks, mix.shape[-1])[0]
            coef, s_energy = decompose(est, src)
            e_energy = est.double().reshape(n, -1).pow(2).sum(dim=1).numpy()
            made += (coef**2 * s_energy[None, :])  # energy of output j explained by true source i
            weight_e += e_energy
            goes += coef**2 * s_energy[None, :]    # same numbers, normalised per true source below
            weight_s += s_energy
            gains.append(float(level_db(est[k]) - level_db(src[k])))
            imps.append(float(si_sdr(est[k][None], src[k][None]) - si_sdr(mix, src[k][None])))
            per_song.setdefault(data.tracks[i % len(data.tracks)].name, []).append(imps[-1])
            used += 1
    print(f"{used} snippets with an audible, exactly known {args.source}")
    if not used:
        return
    print(f"\n{args.source}: estimate level vs target {np.mean(gains):+.1f} dB, SI-SDR improvement {np.mean(imps):+.1f} dB "
          f"({', '.join(f'{s}: {np.mean(v):+.1f}' for s, v in per_song.items())})")
    pct = lambda x: " ".join(f"{100 * v:5.0f}%" for v in x)
    print(f"\nmade up of (rows: output, columns: true {' / '.join(names)}), % of the output's energy")
    for j in range(n):
        print(f"  output {names[j]:7s} {pct(made[j] / np.maximum(weight_e[j], 1e-12))}")
    print(f"\ngoes to (rows: true source, columns: output {' / '.join(names)}), % of the source's energy")
    for i in range(n):
        print(f"  true {names[i]:7s} {pct(goes[:, i] / np.maximum(weight_s[i], 1e-12))}")


if __name__ == "__main__":
    main()
