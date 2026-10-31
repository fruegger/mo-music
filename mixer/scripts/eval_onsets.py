"""Scores mixer.onset_detect against Slakh2100's MIDI onsets, per instrument class.

    python scripts/eval_onsets.py --split validation --tracks 12 [--delta 0.35]

F-measure with a +-50 ms tolerance and one-to-one matching, as in mir_eval.
Used to tune the detector on the train split and to report on validation.
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from mixer.onset_detect import DELTA, pick_onsets, spectral_flux
from mixer.sources import iter_tracks


def match(ref, est, tol=0.05):
    i = j = hits = 0
    offsets = []
    while i < len(ref) and j < len(est):
        d = est[j] - ref[i]
        if abs(d) <= tol:
            hits += 1
            offsets.append(d)
            i, j = i + 1, j + 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return hits, offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("C:/workarea/slakh2100"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--tracks", type=int, default=12)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--deltas", type=float, nargs="+", default=[DELTA])
    args = parser.parse_args()

    results = {d: ([collections.Counter() for _ in range(3)], collections.defaultdict(list)) for d in args.deltas}
    tracks = list(iter_tracks("slakh2100", args.root, args.split))[args.first : args.first + args.tracks]
    for track in tracks:
        for stem in track.stems:
            if stem.onsets is None:
                continue
            ref = stem.onsets()
            if len(ref) < 10:
                continue
            flux, hop = spectral_flux(stem.load(0, track.n_frames).mean(axis=1), track.sample_rate)
            for delta in args.deltas:
                (tp, n_ref, n_est), offsets = results[delta]
                est = pick_onsets(flux, hop, track.sample_rate, delta)
                hits, offs = match(ref, est)
                tp[stem.label] += hits
                n_ref[stem.label] += len(ref)
                n_est[stem.label] += len(est)
                offsets[stem.label] += offs
    for delta in args.deltas:
        (tp, n_ref, n_est), offsets = results[delta]
        print(f"\ndelta = {delta}")
        print(f"{'class':22s} {'ref':>6s} {'est':>6s} {'P':>5s} {'R':>5s} {'F':>5s}  median offset")
        for label in sorted(n_ref):
            p, r = tp[label] / max(n_est[label], 1), tp[label] / n_ref[label]
            print(f"{label:22s} {n_ref[label]:6d} {n_est[label]:6d} {p:5.2f} {r:5.2f} {2*p*r/max(p+r,1e-9):5.2f}  {np.median(offsets[label])*1000:+6.1f} ms")
        p, r = sum(tp.values()) / max(sum(n_est.values()), 1), sum(tp.values()) / sum(n_ref.values())
        print(f"{'ALL':22s} {sum(n_ref.values()):6d} {sum(n_est.values()):6d} {p:5.2f} {r:5.2f} {2*p*r/max(p+r,1e-9):5.2f}")


if __name__ == "__main__":
    main()
