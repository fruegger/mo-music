"""Per-window scores of finished runs on the studio test songs, with bootstrap confidence intervals.

    python scripts/studio_window_scores.py --sources LOG --groups D=run1,run2 E=run3,run4 ... [--snippets 120] [--out FILE]

The sources log must register `studio` at the split the runs were validated on (the swapped one: test = In My Belief,
Just Me Alone). Renders `--snippets` clean 10 s windows (no effects, no tempo/pitch, no stem left out) at fixed seeds,
so every run scores the very same windows, and records the SI-SDR improvement of each of the six sources in each window
where that source is audible and its target known. Prints, per group and source: the mean over windows with a 95%
bootstrap interval (windows resampled within a song, songs kept), the per-run means (the seed spread), and the
difference to the first group with its interval. The interval covers only the sampling of test windows, NOT the training
seed: compare it with the per-run spread.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from mixer import MixConfig, iter_tracks, load_registry

from trainer.data import SnippetDataset
from trainer.separate import load_model
from trainer.separation import si_sdr


def bootstrap(per_song: dict, draws: int, rng: random.Random) -> list[float]:
    """Means of `draws` resamples: windows resampled with replacement inside each song, songs pooled."""
    out = []
    for _ in range(draws):
        vals = []
        for xs in per_song.values():
            vals += [xs[rng.randrange(len(xs))] for _ in xs]
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def interval(xs: list[float]) -> tuple[float, float]:
    xs = sorted(x for x in xs if x == x)
    return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--groups", nargs="+", required=True, help="NAME=runfolder,runfolder ...; the first group is the reference")
    ap.add_argument("--dataset", default="studio")
    ap.add_argument("--split", default="test")
    ap.add_argument("--snippets", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    ds = load_registry(args.sources)[args.dataset.lower()]
    tracks = list(iter_tracks(ds.name, ds.path, args.split))
    names = [t.name for t in tracks]
    mix = MixConfig(seconds=args.seconds, gain_db_range=(0.0, 0.0), p_stem_dropped=0.0, effects=False, p_tempo=0.0, p_pitch=0.0)
    data = SnippetDataset(tracks, mix, args.snippets, 1_000_000, tasks=("separation",), source_set="musdb6")
    print(f"{ds.name}/{args.split}: {names}, {args.snippets} windows of {args.seconds:.0f} s")

    groups: dict[str, list[Path]] = {}
    for g in args.groups:
        name, runs = g.split("=", 1)
        groups[name] = [Path(r) for r in runs.split(",")]
    models = {}
    sources = None
    for runs in groups.values():
        for r in runs:
            models[r], sources = load_model(r)
    # scores[run][song][source] = [improvement per window where audible]
    scores = {r: {n: defaultdict(list) for n in names} for r in models}
    for i in range(args.snippets):
        item = data[i]
        song = names[i % len(names)]
        m, tgt, mask = item["mix"][None], item["sources"][None], item["sources_mask"][None]
        rms_db = 10.0 * torch.log10((tgt**2).mean(dim=(-1, -2)) + 1e-12)
        valid = (rms_db > -50.0) & (mask > 0)
        base = si_sdr(m.unsqueeze(1).expand_as(tgt), tgt)
        for r, model in models.items():
            with torch.no_grad():
                imp = si_sdr(model(m), tgt) - base
            for k, s in enumerate(sources):
                if valid[0, k]:
                    scores[r][song][s].append(float(imp[0, k]))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.snippets} windows", flush=True)
    if args.out:
        args.out.write_text(json.dumps({"sources": sources, "songs": names, "scores": {str(r): {n: dict(d) for n, d in v.items()} for r, v in scores.items()}}), encoding="utf-8")

    rng = random.Random(0)

    def pooled(runs: list[Path], s: str) -> dict:
        """per_song[song] = windows' scores of source s, the runs of the group averaged window by window."""
        out = {}
        for n in names:
            lists = [scores[r][n][s] for r in runs]
            out[n] = [sum(x) / len(x) for x in zip(*lists)] if lists and lists[0] else []
        return {n: v for n, v in out.items() if v}

    ref_name = next(iter(groups))
    for s in sources:
        print(f"\n== {s}   (windows per song: " + ", ".join(f"{n} {len(scores[next(iter(models))][n][s])}" for n in names) + ")")
        ref = pooled(groups[ref_name], s)
        for g, runs in groups.items():
            ps = pooled(runs, s)
            if not ps:
                print(f"  {g:4s} no audible windows")
                continue
            allv = [v for xs in ps.values() for v in xs]
            lo, hi = interval(bootstrap(ps, args.draws, rng))
            per_run = "  ".join(f"{sum(sum(scores[r][n][s]) for n in names) / max(sum(len(scores[r][n][s]) for n in names), 1):+5.1f}" for r in runs)
            line = f"  {g:4s} {sum(allv) / len(allv):+5.1f} [{lo:+5.1f}, {hi:+5.1f}]   runs: {per_run}"
            if g != ref_name and ref:
                common = {n: [a - b for a, b in zip(ps[n], ref[n])] for n in ps if n in ref}
                dl, dh = interval(bootstrap(common, args.draws, rng))
                d = [v for xs in common.values() for v in xs]
                line += f"   diff vs {ref_name} {sum(d) / len(d):+5.1f} [{dl:+5.1f}, {dh:+5.1f}]"
            print(line)
            if len(names) > 1:
                print("        by song: " + "  ".join(f"{n} {sum(v) / len(v):+5.1f} (n={len(v)})" for n, v in ps.items()))


if __name__ == "__main__":
    main()
