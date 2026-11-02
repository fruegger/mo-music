#!/usr/bin/env python3
"""One line per epoch from a run.json: improvement over the raw mix (dB) per source, per validation dataset,
and the average of the last few epochs (a single epoch is noisy: judge a run by that average).

    python scripts/summarize_run.py /workspace/runs/vocalmix-5k/run.json [more run.json ...] [--last 3] [--compact]"""

import argparse
import json

SOURCES = ["drums", "bass", "vocals", "other"]


def row(name: str, v: dict) -> str:
    imp = v["si_sdr_improvement"]
    return f"{name[:6]:>6} " + " ".join(f"{imp[s]:+6.1f}" for s in SOURCES) + f" | mean {v['mean_si_sdr_improvement']:+5.1f}"


def average(parts: list[dict]) -> dict:
    """The mean of several epochs' results for one dataset, in the shape `row` takes."""
    n = len(parts)
    return {
        "si_sdr_improvement": {s: sum(p["si_sdr_improvement"][s] for p in parts) / n for s in SOURCES},
        "mean_si_sdr_improvement": sum(p["mean_si_sdr_improvement"] for p in parts) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--last", type=int, default=3, help="average this many final epochs")
    parser.add_argument("--compact", action="store_true", help="only the last-epochs average, one line per run")
    args = parser.parse_args()

    for path in args.runs:
        run = json.load(open(path, encoding="utf-8"))
        per_epoch = [h["validation"] if "validation" in h else {"val": h} for h in run["history"]]
        tail = per_epoch[-args.last:]
        avg = {name: average([p[name] for p in tail]) for name in tail[0]}
        line = "   ".join(row(n, v) for n, v in avg.items())
        if args.compact:
            name = path.replace("\\", "/").rsplit("/", 2)[-2]
            print(f"{name:<28} ep {len(per_epoch):>2}{'' if run['finished'] else '*'}  {line}")
            continue
        print(f"== {path}  (finished: {run['finished']})")
        for h, parts in zip(run["history"], per_epoch):
            print(f"ep {h['epoch']:>2} loss {h['train_loss']:.4f}  " + "   ".join(row(n, v) for n, v in parts.items()))
        print(f"avg of the last {len(tail)} epochs:   " + line)
        print("columns: " + " ".join(f"{s:>6}" for s in SOURCES) + "  (SI-SDR improvement in dB)")


if __name__ == "__main__":
    main()
