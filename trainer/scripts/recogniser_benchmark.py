"""How well does a presence recogniser tell, for a whole song, which sources are in it?

    python scripts/recogniser_benchmark.py --recognisers runs/presence-cpu-musdb-seed0 [more runs] [--seconds 60]

Takes an excerpt from the middle of every MUSDB18 test song and makes four versions: the song as it is, and with the
vocals, the drums or the bass muted. A source counts as present when its stem is above -50 dB RMS in the excerpt (the
training label). For each recogniser the 3 s window probabilities of an excerpt are reduced to one number per source
in several ways (the share of windows >= 0.5 or >= 0.2, mean, median, 90th percentile, maximum) and each is scored by

  AUC          how well it ranks present sources above absent ones (1.0 = perfectly, 0.5 = chance)
  found@2%     the share of absent sources found when the threshold removes 2 % of the present sources
  found@5%     the same at 5 % (a removed source comes out silent, so this is the cost that matters)

per source and over all. No separator is involved: this is the recogniser alone, so changes to it can be compared fast.

A six-source recogniser (musdb6: drums, bass, vocals, guitar, keys, other) is scored on the same four columns: drums, bass
and vocals as before, and `other` = the group guitar + keys + other (noisy-OR of their window probabilities) against the
MUSDB18 `other` stem. With `--tags labels/musdb18-other-tags.csv` it also gets one column for each group source, scored
on the `full` version only against the hand tags of the song (tagged = present, no tag for it = absent, `unsure` and
untagged songs left out). Those truths are song-level and weak: a tagged instrument may be silent in the excerpt.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from trainer.recognise import load_recogniser, window_probabilities
from trainer.sources import GROUPS, SOURCE_SETS
from trainer.tags import ABSENT, PRESENT, load_other_tags

RATE = 44100
STEMS = ["drums", "bass", "vocals", "other"]
VERSIONS = {"full": None, "no-vocals": "vocals", "no-drums": "drums", "no-bass": "bass"}
ACTIVE_DB = -50.0
STATS = {
    "share>=0.5": lambda p: (p >= 0.5).float().mean(dim=0),
    "share>=0.2": lambda p: (p >= 0.2).float().mean(dim=0),
    "mean": lambda p: p.mean(dim=0),
    "median": lambda p: p.median(dim=0).values,
    "p90": lambda p: p.quantile(0.9, dim=0),
    "max": lambda p: p.max(dim=0).values,
}


def level_db(x: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10((x**2).mean(dim=(-1, -2)) + 1e-12)


def load_excerpt(song: Path, seconds: float) -> torch.Tensor:
    """`(4, 2, samples)`: the stems of the middle `seconds` of the song."""
    total = sf.info(song / "drums.flac").frames
    length = min(total, int(seconds * RATE))
    start = max(0, total // 2 - length // 2)
    return torch.from_numpy(np.stack([sf.read(song / f"{n}.flac", start=start, frames=length, dtype="float32", always_2d=True)[0].T for n in STEMS]))


def source_set_of(names: list[str]) -> str:
    return next(s for s, n in SOURCE_SETS.items() if n == names)


def columns(probs: torch.Tensor, names: list[str]) -> tuple[torch.Tensor, list[str]]:
    """The window probabilities as `(windows, 4 + group sources)`: drums, bass, vocals, other (for a set with a group, the
    noisy-OR of the group sources), then the group sources one by one except `other` (scored against the tags only)."""
    if names == STEMS:
        return probs, STEMS
    group = GROUPS[source_set_of(names)]
    exact = [names.index(n) for n in STEMS[:3]]
    both = 1.0 - (1.0 - probs[:, group]).prod(dim=1, keepdim=True)
    extra = [i for i in group if names[i] != "other"]
    return torch.cat([probs[:, exact], both, probs[:, extra]], dim=1), STEMS[:3] + ["other"] + [names[i] for i in extra]


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    return float((positive[:, None] > negative[None, :]).mean() + 0.5 * (positive[:, None] == negative[None, :]).mean())


def found_at(positive: np.ndarray, negative: np.ndarray, budget: float) -> float:
    """Share of the absent sources whose statistic is below the threshold that removes `budget` of the present ones."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    return float((negative < np.quantile(positive, budget)).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recognisers", type=Path, nargs="+", required=True)
    parser.add_argument("--musdb", type=Path, default=Path(r"C:\workarea\MUSDB18_prepared"))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--songs", type=int, default=50)
    parser.add_argument("--truth", choices=["abs", "rel"], default="abs",
                        help="abs: present if the stem is above -50 dB; rel: present if at most 40 dB under the mix, absent if "
                             "more than 60 dB under it, the cases in between are left out (audibility, not a cut-off)")
    parser.add_argument("--tags", type=Path, default=None,
                        help="hand tags of the MUSDB18 other stems (labels/musdb18-other-tags.csv): scores guitar and keys of "
                             "a six-source recogniser against them")
    args = parser.parse_args()

    songs = sorted(p for p in (args.musdb / "test").iterdir() if p.is_dir())[: args.songs]
    recs = [load_recogniser(r) for r in args.recognisers]
    tags = load_other_tags(args.tags, "musdb6") if args.tags else {}
    labels = {r.name: columns(torch.zeros(1, len(names)), names)[1] for r, (_, names, _) in zip(args.recognisers, recs)}
    rows = {r.name: [] for r in args.recognisers}  # per recogniser: (source index, present, {stat: value})
    with torch.no_grad():
        for n, song in enumerate(songs, 1):
            stems = load_excerpt(song, args.seconds)
            for version, muted in VERSIONS.items():
                heard = stems.clone()
                if muted:
                    heard[STEMS.index(muted)] = 0
                mix = heard.sum(dim=0)
                level = level_db(heard)
                if args.truth == "abs":
                    present, counted = level > ACTIVE_DB, torch.ones(len(STEMS), dtype=torch.bool)
                else:
                    relative = level - level_db(mix)
                    present = (relative >= -40.0) & (level > -80.0)
                    counted = present | (relative < -60.0) | (level <= -80.0)
                for run, (model, names, window_s) in zip(args.recognisers, recs):
                    probs, cols = columns(window_probabilities(model, mix, RATE, window_s, window_s / 2), names)
                    values = {name: fn(probs) for name, fn in STATS.items()}
                    for k in range(len(STEMS)):
                        if counted[k]:
                            rows[run.name].append((k, bool(present[k]), {name: float(v[k]) for name, v in values.items()}))
                    state = tags.get(song.name)
                    if state is not None and version == "full":  # weak, song-level truth for guitar and keys
                        for k in range(len(STEMS), len(cols)):
                            truth = state[names.index(cols[k])]
                            if truth in (PRESENT, ABSENT):
                                rows[run.name].append((k, truth == PRESENT, {name: float(v[k]) for name, v in values.items()}))
            if n % 10 == 0:
                print(f"  {n}/{len(songs)} songs", flush=True)

    for run in args.recognisers:
        data = rows[run.name]
        print(f"\n{run.name} (truth: {args.truth}): {len(songs)} songs x 4 versions, {args.seconds:.0f} s excerpts, "
              f"{sum(p for _, p, _ in data)} present / {sum(not p for _, p, _ in data)} absent source-excerpts")
        cols = labels[run.name]
        print(f"  {'statistic':11s} {'AUC':>6s} " + " ".join(f"{s:>6s}" for s in cols) + f" | {'found@2%':>8s} {'found@5%':>8s}")
        for stat in STATS:
            def split(k=None):
                sel = [(p, v[stat]) for kk, p, v in data if (kk < len(STEMS) if k is None else kk == k)]
                return np.array([v for p, v in sel if p]), np.array([v for p, v in sel if not p])
            pos, neg = split()  # over the four stem columns only: the tag columns are weak truths
            per = " ".join(f"{auc(*split(k)):6.3f}" for k in range(len(cols)))
            print(f"  {stat:11s} {auc(pos, neg):6.3f} {per} | {100 * found_at(pos, neg, 0.02):7.0f}% {100 * found_at(pos, neg, 0.05):7.0f}%")


if __name__ == "__main__":
    main()
