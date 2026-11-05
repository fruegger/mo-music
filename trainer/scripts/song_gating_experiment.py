"""The gating feature as it is used: the recogniser decides for a whole song, then the separator drops what is absent.

    python scripts/song_gating_experiment.py --runs runs/local-soft-aug-clean-seed0 \
        --recogniser runs/presence-cpu-musdb-seed0 [--seconds 20] [--songs 50]

Takes an excerpt from the middle of every MUSDB18 test song and makes four versions: the song as it is ("full"),
and the song with the vocals, the drums or the bass muted ("no-vocals", "no-drums", "no-bass": an instrumental,
a song without a drummer, ...). The targets are always the original stems. For each version it separates with

  none        no gating
  oracle      the true presence (a source is present if it is above -50 dB in the excerpt)
  rec<t>/<f>  the recogniser: a source is in the song if at least a share <f> of its 3 s windows reach probability <t>

and reports, per version and condition: the mean SI-SDR improvement over the sources that are present and kept,
how many present sources were wrongly removed (they come out silent), and the leakage into absent sources (their
estimate's level relative to the mix, dB). Also how often the recogniser finds the muted source and keeps the others.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from trainer.recognise import load_recogniser, song_presence, window_probabilities
from trainer.separate import load_model
from trainer.separation import gate_masks, si_sdr

RATE = 44100
STEMS = ["drums", "bass", "vocals", "other"]
ACTIVE_DB = -50.0
RULES = {"rec0.5/0.05": (0.5, 0.05), "rec0.35/0.05": (0.35, 0.05), "rec0.5/0.2": (0.5, 0.2)}
VERSIONS = {"full": None, "no-vocals": "vocals", "no-drums": "drums", "no-bass": "bass"}


def level_db(x: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10((x**2).mean(dim=(-1, -2)) + 1e-12)


def load_excerpt(song: Path, seconds: float) -> torch.Tensor:
    """`(4, 2, samples)`: the stems of the middle `seconds` of the song."""
    total = sf.info(song / "drums.flac").frames
    length = min(total, int(seconds * RATE))
    start = max(0, total // 2 - length // 2)
    stems = [sf.read(song / f"{n}.flac", start=start, frames=length, dtype="float32", always_2d=True)[0].T for n in STEMS]
    return torch.from_numpy(np.stack(stems))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--recogniser", type=Path, required=True)
    parser.add_argument("--musdb", type=Path, default=Path(r"C:\workarea\MUSDB18_prepared"))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--songs", type=int, default=50)
    args = parser.parse_args()

    songs = sorted(p for p in (args.musdb / "test").iterdir() if p.is_dir())[: args.songs]
    models = [load_model(r)[0] for r in args.runs]
    recogniser, _, window_s = load_recogniser(args.recogniser)
    conditions = ["none", "oracle", *RULES]
    # stats[version][condition] = [gain sum, kept count, removed count, present count, leak sum, absent count]
    stats = {v: {c: torch.zeros(6) for c in conditions} for v in VERSIONS}
    detection = {v: torch.zeros(4) for v in VERSIONS}  # absent found, absent total, present kept, present total (rec0.5/0.05)

    with torch.no_grad():
        for n, song in enumerate(songs, 1):
            stems = load_excerpt(song, args.seconds)
            for version, muted in VERSIONS.items():
                heard = stems.clone()
                if muted:
                    heard[STEMS.index(muted)] = 0
                mix = heard.sum(dim=0)
                present = level_db(heard) > ACTIVE_DB  # truth, per source
                baseline = si_sdr(mix.unsqueeze(0).expand_as(stems), stems)
                probs = window_probabilities(recogniser, mix, RATE, window_s, window_s / 2)
                decisions = {"oracle": present.float()}
                for name, (thr, frac) in RULES.items():
                    d = song_presence(probs, thr, frac)
                    decisions[name] = d if d.sum() > 0 else torch.ones_like(d)
                d0 = decisions["rec0.5/0.05"]
                detection[version] += torch.tensor([float(((d0 == 0) & ~present).sum()), float((~present).sum()),
                                                    float(((d0 > 0) & present).sum()), float(present.sum())])
                for model in models:
                    spec, masks = model.analyse(mix.unsqueeze(0))
                    for cond in conditions:
                        g = None if cond == "none" else decisions[cond]
                        m = masks if g is None else gate_masks(masks, g.unsqueeze(0), model.net.mask)
                        est = model.synthesize(spec, m, mix.shape[-1])[0]
                        kept = present if g is None else present & (g > 0)
                        gain = si_sdr(est, stems) - baseline
                        leak = level_db(est) - level_db(mix)
                        s = stats[version][cond]
                        s += torch.tensor([float((gain * kept).sum()), float(kept.sum()), float((present & ~kept).sum()),
                                           float(present.sum()), float((leak * ~present).sum()), float((~present).sum())])
            if n % 10 == 0:
                print(f"  {n}/{len(songs)} songs", flush=True)

    print(f"\n{len(songs)} songs, {args.seconds:.0f} s excerpts, {len(models)} model(s), recogniser {args.recogniser.name}")
    for version in VERSIONS:
        det = detection[version]
        found = f"{100 * det[0] / det[1]:.0f}% of the absent sources found" if det[1] > 0 else "no absent source"
        print(f"\n{version}: {found}; {100 * det[2] / det[3]:.0f}% of the present sources kept  [rec0.5/0.05]")
        print(f"  {'condition':14s} {'improvement dB':>15s} {'present removed %':>18s} {'leakage into absent dB':>23s}")
        for cond in conditions:
            gain, kept, removed, present, leak, absent = stats[version][cond].tolist()
            leak_s = f"{leak / absent:23.1f}" if absent else f"{'-':>23s}"
            print(f"  {cond:14s} {gain / max(kept, 1):+15.2f} {100 * removed / max(present, 1):18.1f} {leak_s}")


if __name__ == "__main__":
    main()
