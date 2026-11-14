"""What gain does each stem of a song have inside other full-length mixes of that song?

    python scripts/find_stem_gains.py --song "Just Me Alone" --mixes FILE [FILE ...]

Aligns each mix with the sum of the song's prepared stems (cross-correlation of the signals at 5.5 kHz, +-15 s),
resamples it to 44.1 kHz if needed, then fits one gain per stem by least squares on a stretch from the middle of the
song. Prints the offset, how well the stems explain the mix (dB; above ~10 is a real match) and the gains, also in dB.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import json

RATE = 44100


def db(x: float) -> float:
    return 10.0 * np.log10(x + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", required=True)
    parser.add_argument("--root", type=Path, default=Path(r"C:\workarea\studio_prepared"))
    parser.add_argument("--mixes", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    folder = next(p for split in ("train", "test") for p in [args.root / split / args.song] if p.exists())
    meta = json.loads((folder / "stems.json").read_text(encoding="utf-8"))
    names = [s["name"] for s in meta["stems"]]
    stems = np.stack([sf.read(folder / s["file"], dtype="float32", always_2d=True)[0] for s in meta["stems"]])  # (n, T, 2)
    total = stems.sum(axis=0)
    print(f"{args.song}: stems {names}, {total.shape[0] / RATE:.1f} s")
    for path in args.mixes:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        data = data[:, :2] if data.shape[1] > 1 else np.repeat(data, 2, axis=1)
        if sr != RATE:
            data = torchaudio.functional.resample(torch.from_numpy(data.T.copy()), sr, RATE).numpy().T
        d = 8
        a, b = total.mean(axis=1)[::d], data.mean(axis=1)[::d]
        n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
        cc = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
        reach = 15 * RATE // d
        lags = np.concatenate([np.arange(0, reach), np.arange(-reach, 0)])
        idx = np.concatenate([np.arange(0, reach), np.arange(n - reach, n)])
        lag = int(lags[np.argmax(cc[idx])]) * d  # mix[t + lag] ~ stems[t]
        start = len(total) // 4
        length = min(60 * RATE, len(total) - start - abs(lag), len(data) - start - abs(lag))
        s0 = start
        cols = np.stack([np.concatenate([stems[k][s0 : s0 + length, c] for c in (0, 1)]) for k in range(len(names))], axis=1)
        target = np.concatenate([data[s0 + lag : s0 + lag + length, c] for c in (0, 1)])
        g, *_ = np.linalg.lstsq(cols, target, rcond=None)
        fit = db((target**2).sum() / (((target - cols @ g) ** 2).sum() + 1e-12))
        gains = ", ".join(f"{k} {v:.2f} ({20 * np.log10(abs(v) + 1e-9):+.0f} dB)" for k, v in zip(names, g))
        print(f"  {path.name}: {len(data) / RATE:.1f} s @ {sr}, offset {lag / RATE:+.2f} s, fit {fit:.1f} dB\n      {gains}")


if __name__ == "__main__":
    main()
