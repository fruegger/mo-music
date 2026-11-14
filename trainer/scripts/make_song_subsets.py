"""Nested subsets of the MUSDB18 train songs, as folders the loader reads like a prepared dataset.

    python scripts/make_song_subsets.py --src /root/data/MUSDB18_prepared --dst /root/data/b6 --sizes 3 10 30 100 \
        --first "Actions - Devil's Words" "BigTroubles - Phantom" "ANiMAL - Clinic A" [--seed 0] [--copy]

Writes `<dst>/n<size>/train/<song>` for the first `size` songs of one fixed order (the `--first` songs, then the other train songs in a
seeded shuffle, so every subset contains the smaller ones) and `<dst>/n<size>/test/<song>` for ALL test songs. Songs are symlinks
(`--copy` copies them instead, for a system without symlinks: it needs the disk space). Training on `n<size>` then validates on the
real test songs, and the songs it trains on are `n<size>/train`: score both splits with `oracle_ceiling.py`.
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--first", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    train = sorted(p.name for p in (args.src / "train").iterdir() if p.is_dir())
    test = sorted(p.name for p in (args.src / "test").iterdir() if p.is_dir())
    missing = [s for s in args.first if s not in train]
    assert not missing, f"not in {args.src / 'train'}: {missing}"
    rest = [s for s in train if s not in args.first]
    random.Random(args.seed).shuffle(rest)
    order = list(args.first) + rest
    assert max(args.sizes) <= len(order), f"{len(order)} train songs, {max(args.sizes)} asked"

    def put(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            return
        if args.copy:
            shutil.copytree(src, dst)
        else:
            os.symlink(src.resolve(), dst, target_is_directory=True)

    for size in args.sizes:
        root = args.dst / f"n{size}"
        for song in order[:size]:
            put(args.src / "train" / song, root / "train" / song)
        for song in test:
            put(args.src / "test" / song, root / "test" / song)
        print(f"{root}: {size} train songs, {len(test)} test songs")


if __name__ == "__main__":
    main()
