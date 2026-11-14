"""Are the studio stems at the level they have in the finished mixes?

    python scripts/stem_levels.py [--root C:/workarea/studio_prepared]

For every prepared song: each stem's level (RMS in dBFS over its whole length, and over the frames where it is
audible) and peak. Then, for every finished mix of the song that is explained by the stems, the gain each stem
needs to match the mix (least squares on the whole song): about 1.0 = the stem is at its level in the mix, below 1
= the exported stem is louder than it is in the mix, above 1 = quieter (the mix boosted it); about 0 = the stem is
not in that mix. Mixes the stems explain badly (mastered versions, below `--min-fit` dB) are skipped.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def db(x: float) -> float:
    return 10.0 * np.log10(x + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"C:\workarea\studio_prepared"))
    parser.add_argument("--min-fit", type=float, default=10.0, help="skip mixes the stems explain worse than this (dB)")
    args = parser.parse_args()
    for split in ("train", "test"):
        for song in sorted(p for p in (args.root / split).glob("*") if (p / "stems.json").exists()):
            meta = json.loads((song / "stems.json").read_text(encoding="utf-8"))
            stems = {s["name"]: sf.read(song / s["file"], dtype="float32", always_2d=True)[0] for s in meta["stems"]}
            print(f"\n== {song.name} ({split})")
            print(f"   {'stem':12s} {'RMS dB':>8s} {'RMS while audible':>18s} {'peak dB':>8s}")
            for name, a in stems.items():
                frame = a.reshape(-1, 2)[: len(a) // 4410 * 4410].reshape(-1, 4410, 2)
                level = (frame**2).mean(axis=(1, 2))
                audible = level > 10 ** (-50 / 10)
                print(f"   {name:12s} {db((a**2).mean()):8.1f} {db(level[audible].mean()) if audible.any() else float('nan'):18.1f} {db(np.abs(a).max() ** 2):8.1f}")
            for mix in meta.get("mixes", []):
                m = sf.read(song / mix["file"], dtype="float32", always_2d=True)[0]
                n = min(len(m), min(len(a) for a in stems.values()))
                names = list(stems)
                a = np.stack([np.concatenate([stems[k][:n:4, c] for c in (0, 1)]) for k in names], axis=1)
                b = np.concatenate([m[:n:4, c] for c in (0, 1)])
                g, *_ = np.linalg.lstsq(a, b, rcond=None)
                fit = db((b**2).sum() / (((b - a @ g) ** 2).sum() + 1e-12))
                if fit < args.min_fit:
                    print(f"   mix {mix['name']:22s} not explained by the stems (fit {fit:4.1f} dB): skipped")
                    continue
                print(f"   mix {mix['name']:22s} fit {fit:4.1f} dB; gains: " + ", ".join(f"{k} {v:.2f}" for k, v in zip(names, g)))


if __name__ == "__main__":
    main()
