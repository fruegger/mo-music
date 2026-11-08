"""Compare our separator with reference separators on the same clean excerpts.

Three steps, so that the reference models can live in their own environment (demucs pulls in a lot) and so that results of
services you run by hand (commercial APIs) can be scored the same way:

  1. python scripts/reference_compare.py export --out C:/workarea/reference-test --run C:/workarea/batch2_results/b2-d-control-s0
         writes `<out>/excerpts/<set>/<nnn>/mix.wav` (stereo 44.1 kHz) and `truth.pt` (the six targets and which are known) for
         `--snippets` clean 10 s excerpts (natural stem levels, no effects) of every test set: MUSDB18 test and the studio test.
  2. In the reference environment: python scripts/reference_run_demucs.py --out C:/workarea/reference-test --model htdemucs_6s
         writes `<out>/refs/<model>/<set>/<nnn>/<stem>.wav`. Any other system works the same way: put its stems for an excerpt
         into `<out>/refs/<name>/<set>/<nnn>/` (names: drums, bass, vocals, guitar, piano or keys, other; any other file name is
         added to `other`).
  3. python scripts/reference_compare.py score --out C:/workarea/reference-test --run RUN [RUN ...] [--refs htdemucs_6s ...]
         prints the SI-SDR improvement (dB over the unprocessed mix, the usual metric) per source for our run(s) and every system under refs/.

Units: drums, bass, vocals and guitar / keys where the true target is known (the studio songs; MUSDB18 only knows their sum),
and `group` = guitar + keys + other as one unit (always known): a four-stem model's `other` is its group estimate, a six-stem
model's group is the sum of its guitar, piano and other. A source counts only where its target is audible (> -50 dB), n is shown.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SETS = ("musdb18-prepared", "studio")
SOURCES = ["drums", "bass", "vocals", "guitar", "keys", "other"]
GROUP = [3, 4, 5]
RATE = 44100
ACTIVE_DB = -50.0
ALIASES = {"piano": "keys", "keys": "keys", "guitar": "guitar", "drums": "drums", "bass": "bass", "vocals": "vocals"}


def level_db(x: torch.Tensor) -> float:
    return float(10.0 * torch.log10((x**2).mean() + 1e-12))


def export(args) -> None:
    from mixer import MixConfig, load_registry

    from trainer.data import SnippetDataset

    config = json.loads((args.run / "run.json").read_text(encoding="utf-8"))["config"]
    mix_cfg = MixConfig.from_dict({**config["mix"], "effects": False, "p_tempo": 0.0, "p_pitch": 0.0, "p_stem_dropped": 0.0,
                                   "gain_db_range": [0.0, 0.0], "seconds": args.seconds})
    registry = load_registry(str(args.sources))
    for name in SETS:
        data = SnippetDataset.from_dataset(registry[name], "test", mix_cfg, args.snippets, 1_000_000, tasks=("separation",), source_set="musdb6")
        for i in range(args.snippets):
            item = data[i]
            folder = args.out / "excerpts" / name / f"{i:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / "mix.wav", item["mix"].T.numpy(), RATE, subtype="FLOAT")
            torch.save({"sources": item["sources"], "mask": item["sources_mask"]}, folder / "truth.pt")
        print(f"{name}: {args.snippets} excerpts of {args.seconds:.0f} s -> {args.out / 'excerpts' / name}")


def read_stems(folder: Path, length: int) -> dict[str, torch.Tensor]:
    """A system's stems for one excerpt -> `{source: (2, length)}` with `group` = everything that is not drums/bass/vocals."""
    out, group = {}, torch.zeros(2, length)
    for f in sorted(folder.glob("*.wav")):
        data, rate = sf.read(f, dtype="float32", always_2d=True)
        x = torch.from_numpy(np.ascontiguousarray(data.T))
        x = x.repeat(2, 1) if x.shape[0] == 1 else x[:2]
        if rate != RATE:
            import torchaudio

            x = torchaudio.functional.resample(x, rate, RATE)
        x = torch.nn.functional.pad(x, (0, max(0, length - x.shape[-1])))[:, :length]
        key = ALIASES.get(f.stem.lower(), "other")
        out[key] = out.get(key, 0) + x
        if key not in ("drums", "bass", "vocals"):
            group = group + x
    out["group"] = group
    return out


def our_stems(model, names: list[str], mix: torch.Tensor) -> dict[str, torch.Tensor]:
    from trainer.separate import separate

    est = separate(model, mix)
    out = {n: est[i] for i, n in enumerate(names)}
    out["group"] = est[GROUP].sum(dim=0)
    return out


def score(args) -> None:
    from trainer.separate import load_model
    from trainer.separation import si_sdr

    systems = {}
    for run in args.run:
        model, names = load_model(run)
        systems[run.name] = ("run", model, names)
    refs = args.refs or sorted(p.name for p in (args.out / "refs").iterdir() if p.is_dir())
    for r in refs:
        systems[r] = ("ref", r, None)

    units = ["drums", "bass", "vocals", "guitar", "keys", "group"]
    table = {}
    for name in SETS:
        folders = sorted((args.out / "excerpts" / name).iterdir())
        for sysname, (kind, obj, names) in systems.items():
            sums, counts = {u: 0.0 for u in units}, {u: 0 for u in units}
            for folder in folders:
                data, _ = sf.read(folder / "mix.wav", dtype="float32", always_2d=True)
                mix = torch.from_numpy(np.ascontiguousarray(data.T))
                truth = torch.load(folder / "truth.pt")
                sources, mask = truth["sources"], truth["mask"]
                targets = {s: sources[i] for i, s in enumerate(SOURCES)}
                targets["group"] = sources[GROUP].sum(dim=0)
                known = {s: bool(mask[i]) for i, s in enumerate(SOURCES)} | {"group": True}
                if kind == "run":
                    est = our_stems(obj, names, mix)
                else:
                    ref_dir = args.out / "refs" / obj / name / folder.name
                    if not ref_dir.exists():
                        continue
                    est = read_stems(ref_dir, mix.shape[-1])
                for u in units:
                    if u not in est or not known[u] or level_db(targets[u]) < ACTIVE_DB:
                        continue
                    gain = float(si_sdr(est[u], targets[u]) - si_sdr(mix, targets[u]))
                    sums[u] += gain
                    counts[u] += 1
            table[(name, sysname)] = {u: (sums[u] / counts[u] if counts[u] else None, counts[u]) for u in units}

    for name in SETS:
        print(f"\n{name} test, {len(folders)} excerpts: SI-SDR improvement in dB (n)")
        print(f"{'system':28s}" + "".join(f"{u:>12s}" for u in units))
        for sysname in systems:
            row = table[(name, sysname)]
            cells = "".join(f"{(f'{v:+.1f} ({n})' if v is not None else 'n/a'):>12s}" for v, n in (row[u] for u in units))
            print(f"{sysname:28s}{cells}")
    if args.json:
        args.json.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in table.items()}, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--run", type=Path, required=True, help="one of our runs (its mix settings are used)")
    e.add_argument("--snippets", type=int, default=12, help="excerpts per test set")
    e.add_argument("--seconds", type=float, default=10.0)
    e.add_argument("--sources", type=Path, default=Path("configs/studio-sources-swap.log"), help="dataset log with musdb18-prepared and studio")
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--run", type=Path, nargs="*", default=[], help="our runs (run folders with run.json and model.pt)")
    s.add_argument("--refs", nargs="*", help="systems under refs/ to score (default: all)")
    s.add_argument("--json", type=Path)
    args = parser.parse_args()
    export(args) if args.cmd == "export" else score(args)


if __name__ == "__main__":
    main()
