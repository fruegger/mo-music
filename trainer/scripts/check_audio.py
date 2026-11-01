#!/usr/bin/env python3
"""Find unreadable audio files in the training data.

    python scripts/check_audio.py                 # a random sample of 400 files from Slakh2100 and MUSDB18_prepared
    python scripts/check_audio.py --sample 0      # every file (slow: reads ~60 GB)
    python scripts/check_audio.py DIR [DIR ...]   # other roots

Each file is decoded completely, so a truncated or corrupted file shows up. Prints one BAD line per failure."""

import argparse
import glob
import random

import soundfile as sf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="*", default=["/workspace/slakh2100", "/workspace/MUSDB18_prepared"])
    p.add_argument("--sample", type=int, default=400, help="check this many random files (0 = all)")
    args = p.parse_args()

    files = [f for root in args.roots for f in glob.glob(f"{root}/**/*.flac", recursive=True) if not f.endswith("mix.flac")]
    random.seed(0)
    random.shuffle(files)
    if args.sample:
        files = files[: args.sample]
    bad = 0
    for i, f in enumerate(files, 1):
        try:
            sf.read(f, always_2d=True)
        except Exception as e:  # noqa: BLE001 - report any failure
            bad += 1
            print("BAD", f, repr(e)[:120], flush=True)
        if i % 100 == 0:
            print(f"{i}/{len(files)} checked, {bad} bad", flush=True)
    print(f"checked {len(files)} files, bad: {bad}")


if __name__ == "__main__":
    main()
