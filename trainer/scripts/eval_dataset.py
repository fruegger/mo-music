"""How well do trained separators do on a prepared dataset (e.g. your own songs)?

    python scripts/eval_dataset.py --runs runs/local-soft-aug-clean-seed0 runs/local-soft-clean-seed0 \
        --dataset studio --split train --sources configs/studio-sources.log [--snippets 40] [--seconds 10]

Renders fixed snippets (clean: no effects, no tempo/pitch, stems at their own level, no stems left out) from the
dataset's songs, grouped into the model's sources (drums, bass, vocals, other), and prints per run and per song
the mean SI-SDR improvement over the mixture, and per source. The mix is the exact sum of the stems.
A six-source model (drums, bass, vocals, guitar, keys, other) is scored twice: as the four-source models are (its
guitar, keys and other outputs added up and compared with the four-source "other"), so the two kinds compare, and on
its own six sources.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from mixer import MixConfig, iter_tracks, load_registry
from torch import nn
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.separate import load_model
from trainer.separation import evaluate_separation


class Merged(nn.Module):
    """A six-source separator seen as a four-source one: guitar, keys and other are added up into "other"."""

    def __init__(self, model):
        super().__init__()
        self.model, self.net = model, type("Net", (), {"n_sources": 4})()

    def forward(self, mix):
        est = self.model(mix)
        return torch.cat([est[:, :3], est[:, 3:].sum(dim=1, keepdim=True)], dim=1)


def report(model, data, per_song, names, label):
    totals = defaultdict(lambda: [0.0, 0.0])
    print(f"  [{label}]")
    for song, idx in per_song.items():
        loader = DataLoader(torch.utils.data.Subset(data, idx), batch_size=4)
        with torch.no_grad():
            scores = evaluate_separation(model, loader, "cpu")
        valid = [v for v, n in zip(scores["si_sdr_improvement"], scores["examples"]) if n > 0]
        detail = "  ".join(f"{s} {v:+5.1f}" if n > 0 else f"{s}   -  " for s, v, n in zip(names, scores["si_sdr_improvement"], scores["examples"]))
        print(f"    {song:18s} mean {sum(valid) / max(len(valid), 1):+5.2f}   {detail}")
        for k, (v, n) in enumerate(zip(scores["si_sdr_improvement"], scores["examples"])):
            totals[names[k]][0] += v * n
            totals[names[k]][1] += n
    mean = {s: t[0] / t[1] for s, t in totals.items() if t[1] > 0}
    print(f"    {'all songs':18s} mean {sum(mean.values()) / len(mean):+5.2f}   " + "  ".join(f"{s} {v:+5.1f}" for s, v in mean.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sources", type=Path, default=Path("configs/studio-sources.log"))
    parser.add_argument("--snippets", type=int, default=40, help="snippets in total, spread over the songs")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    dataset = load_registry(args.sources)[args.dataset.lower()]
    mix = MixConfig(seconds=args.seconds, gain_db_range=(0.0, 0.0), p_stem_dropped=0.0, effects=False, p_tempo=0.0, p_pitch=0.0)
    tracks = list(iter_tracks(dataset.name, dataset.path, args.split))
    names = [t.name for t in tracks]
    print(f"{dataset.name}/{args.split}: {len(tracks)} songs, {args.snippets} snippets of {args.seconds:.0f} s")
    data4 = SnippetDataset(tracks, mix, args.snippets, 1_000_000, tasks=("separation",), source_set="musdb4")
    data6 = SnippetDataset(tracks, mix, args.snippets, 1_000_000, tasks=("separation",), source_set="musdb6")
    # Snippet i comes from song i % n_songs: evaluate each song on its own to report them separately.
    per_song = defaultdict(list)
    for i in range(args.snippets):
        per_song[names[i % len(names)]].append(i)
    for run in args.runs:
        model, sources = load_model(run)
        print()
        print(run.name)
        if len(sources) == 6:
            report(Merged(model), data4, per_song, ["drums", "bass", "vocals", "other"], "as four sources: other = guitar + keys + other")
            report(model, data6, per_song, sources, "its six sources")
        else:
            report(model, data4, per_song, sources, "four sources")


if __name__ == "__main__":
    main()
