"""Which sources are in a song? Uses a run of the `presence` task.

    python -m trainer.recognise --run runs/presence-cpu-musdb song.mp3 [more files]

prints, per source, how often it is audible (the share of 3 s windows with probability >= `--threshold`) and the
decision "in the song" (that share >= `--min-fraction`). The decision is what `trainer.separate --recogniser`
hands to the separator. A missed source is silent in the result while a false alarm only leaves some leakage, so
the song-level rule is generous: a source counts as present as soon as a few windows say so.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from .train import PresenceTask


def load_recogniser(run: Path) -> tuple[torch.nn.Module, list[str], float]:
    """The presence model of a run folder (eval mode), its source names and the window length in seconds."""
    cfg = json.loads((run / "run.json").read_text(encoding="utf-8"))["config"]
    task = PresenceTask(cfg)
    model = task.build()
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.eval(), task.names, float(cfg["mix"].get("seconds", 3.0))


@torch.no_grad()
def window_probabilities(model, audio: torch.Tensor, rate: int, window_s: float = 3.0, hop_s: float = 1.5,
                         batch: int = 16) -> torch.Tensor:
    """`audio` `(2, samples)` -> `(windows, sources)`: the probability that each source is audible in each
    window of `window_s` seconds (hop `hop_s`). A signal shorter than a window is zero-padded to one."""
    mono = audio.mean(dim=0)
    length = max(1, int(window_s * rate))
    if mono.shape[0] < length:
        mono = torch.nn.functional.pad(mono, (0, length - mono.shape[0]))
    starts = list(range(0, mono.shape[0] - length + 1, max(1, int(hop_s * rate))))
    if starts[-1] + length < mono.shape[0]:
        starts.append(mono.shape[0] - length)  # the last window ends exactly at the end
    windows = torch.stack([mono[s : s + length] for s in starts])
    return torch.cat([model(windows[i : i + batch]).sigmoid() for i in range(0, len(windows), batch)])


def song_presence(probabilities: torch.Tensor, threshold: float = 0.5, min_fraction: float = 0.05) -> torch.Tensor:
    """`(windows, sources)` probabilities -> `(sources,)` float, 1 where the source is in the song: at least
    `min_fraction` of the windows reach `threshold`."""
    share = (probabilities >= threshold).float().mean(dim=0)
    return (share >= min_fraction).float()


def main(argv=None) -> None:
    from .separate import read_audio

    parser = argparse.ArgumentParser(prog="trainer.recognise", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="a presence run folder with run.json and model.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-fraction", type=float, default=0.05)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    model, names, window_s = load_recogniser(args.run)
    for path in args.files:
        audio, rate = read_audio(path)
        probs = window_probabilities(model, audio, rate, window_s, window_s / 2)
        share = (probs >= args.threshold).float().mean(dim=0)
        present = song_presence(probs, args.threshold, args.min_fraction)
        print(f"{path.name}: {audio.shape[-1] / rate:.1f} s, {len(probs)} windows")
        for name, frac, mean, here in zip(names, share, probs.mean(dim=0), present):
            print(f"  {name:7s} audible in {100 * float(frac):5.1f}% of windows (mean p {float(mean):.2f})  -> {'IN THE SONG' if here else 'absent'}")


if __name__ == "__main__":
    main()
