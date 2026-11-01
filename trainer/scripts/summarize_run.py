#!/usr/bin/env python3
"""One line per epoch from a run.json: improvement over the raw mix (dB) per source, per validation dataset.

    python scripts/summarize_run.py /workspace/runs/vocalmix-5k/run.json [more run.json ...]"""

import json
import sys

SOURCES = ["drums", "bass", "vocals", "other"]


def main() -> None:
    for path in sys.argv[1:]:
        run = json.load(open(path, encoding="utf-8"))
        print(f"== {path}  (finished: {run['finished']})")
        for h in run["history"]:
            parts = h["validation"] if "validation" in h else {"val": h}
            cells = []
            for name, v in parts.items():
                imp = v["si_sdr_improvement"]
                cells.append(f"{name[:6]:>6} " + " ".join(f"{imp[s]:+6.1f}" for s in SOURCES) + f" | mean {v['mean_si_sdr_improvement']:+5.1f}")
            print(f"ep {h['epoch']:>2} loss {h['train_loss']:.4f}  " + "   ".join(cells))
        print("columns: " + " ".join(f"{s:>6}" for s in SOURCES) + "  (SI-SDR improvement in dB)")


if __name__ == "__main__":
    main()
