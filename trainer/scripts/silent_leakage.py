"""Does a model put sound into a source that is not there? Leakage into silent sources, per run.

    python scripts/silent_leakage.py --sources LOG --dataset musdb18-prepared --split test --tags ../labels/musdb18-other-tags.csv --promote-single \
        --groups L1=run1,run2 SISDR=run3,run4 [--snippets 600] [--seconds 6]

Clean windows (same renderer as `oracle_ceiling.py`). A window counts for a source when that source's target is KNOWN and
silent (RMS below -60 dB) while the mixture is audible (above -50 dB). For every such window the level of the model's
estimate of that source is taken relative to the mixture's level, in dB: -inf is perfect silence, -40 dB is a faint
trace, -20 dB is clearly audible. Printed per group (runs averaged) and source: the number of windows, the mean level
and the median (dB re mixture) and the share of the windows where the leak is above -40 dB and above -30 dB.
The metric the training reports (SI-SDR improvement) is only defined where the source is audible, so this is the other half.
"""

import argparse
import statistics
from pathlib import Path

import torch
from mixer import MixConfig, iter_tracks, load_registry

from trainer.data import SnippetDataset
from trainer.separate import load_model
from trainer.tags import load_other_tags

SILENT_DB = -60.0
AUDIBLE_DB = -50.0


def level_db(x: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10((x**2).mean(dim=(-1, -2)) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=Path("configs/studio-sources.log"))
    ap.add_argument("--dataset", default="musdb18-prepared")
    ap.add_argument("--split", default="test")
    ap.add_argument("--snippets", type=int, default=600)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--tags", type=Path)
    ap.add_argument("--promote-single", action="store_true")
    ap.add_argument("--groups", nargs="+", required=True, help="NAME=run1,run2 (the runs of a group are averaged)")
    args = ap.parse_args()

    ds = load_registry(args.sources)[args.dataset.lower()]
    tracks = list(iter_tracks(ds.name, ds.path, args.split))
    mix_cfg = MixConfig(seconds=args.seconds, gain_db_range=(0.0, 0.0), p_stem_dropped=0.0, effects=False, p_tempo=0.0, p_pitch=0.0)
    kwargs = {}
    if args.tags:
        kwargs = {"source_tags": load_other_tags(args.tags, "musdb6"), "promote_single": args.promote_single}
    data = SnippetDataset(tracks, mix_cfg, args.snippets, 1_000_000, tasks=("separation",), source_set="musdb6", **kwargs)

    groups, names = {}, None
    for g in args.groups:
        label, runs = g.split("=", 1)
        groups[label] = []
        for r in runs.split(","):
            model, names = load_model(Path(r))
            groups[label].append(model)
    leaks = {label: {n: [] for n in names} for label in groups}  # label -> source -> leak levels (dB re mixture)
    for i in range(args.snippets):
        item = data[i]
        mix, sources, mask = item["mix"], item["sources"], item["sources_mask"]
        silent = (level_db(sources) < SILENT_DB) & (mask > 0)
        if not silent.any() or level_db(mix) < AUDIBLE_DB:
            continue
        mix_db = float(level_db(mix))
        for label, models in groups.items():
            for model in models:
                with torch.no_grad():
                    est = model(mix[None])[0]
                rel = level_db(est) - mix_db
                for k, n in enumerate(names):
                    if silent[k]:
                        leaks[label][n].append(float(rel[k]))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.snippets} windows", flush=True)

    print(f"\n{ds.name}/{args.split}: leak level of the estimate of a silent source, dB re the mixture (lower is better)")
    print(f"{'group':8s}{'source':>8s}{'n':>7s}{'mean':>8s}{'median':>8s}{'>-40dB':>8s}{'>-30dB':>8s}")
    for label, per in leaks.items():
        n_models = len(groups[label])
        for s, vals in per.items():
            if not vals:
                continue
            print(f"{label:8s}{s:>8s}{len(vals) // n_models:7d}{statistics.fmean(vals):8.1f}{statistics.median(vals):8.1f}"
                  f"{sum(v > -40 for v in vals) / len(vals):8.0%}{sum(v > -30 for v in vals) / len(vals):8.0%}")


if __name__ == "__main__":
    main()
