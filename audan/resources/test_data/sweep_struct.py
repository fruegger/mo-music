"""Grid-sweep audan struct's segmentation knobs against the MIDI markers.

Runs `audan struct --format lab --seg-...` for every parameter combination
on every annotated song (audio from --library-dir, files.txt names) and
ranks the combinations (downbeat snapping on) by mean boundary F-measure at 3 s, with 0.5 s,
pairwise F and the estimated/reference section count ratio alongside. The
untouched defaults are scored as the baseline. Warm audan cache makes a run
~0.2 s, so use a persistent --cache-dir.

    python sweep_struct.py --library-dir ../songs --cache-dir <dir> [--top 15]
"""

import argparse
import itertools
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compare_beats_midi import HERE, load_entries
from compare_struct_midi import boundaries, f_measure, pairwise_f, reference_sections

KERNELS = [8, 12, 16, 24, 32, 48]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]
MIN_DISTANCES = [8, 16, 24, 43]


def run_struct(exe, cache, audio, params):
    cmd = [str(exe), "struct", str(audio), "--format", "lab", "--cache-dir", str(cache)]
    for flag, value in params.items():
        cmd += [flag] if value == "" else [flag, str(value)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200])
    return [(float(a), float(b), l) for a, b, l in (ln.split() for ln in out.stdout.splitlines() if ln.strip())]


def evaluate(exe, cache, songs, params):
    rows = []
    for audio, ref, end in songs:
        est = run_struct(exe, cache, audio, params)
        rb, eb = boundaries(ref), boundaries(est)
        rows.append((f_measure(rb, eb, 3.0), f_measure(rb, eb, 0.5), pairwise_f(ref, est, end), len(est) / len(ref)))
    return tuple(statistics.mean(col) for col in zip(*rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--exe", type=Path, default=HERE.parent.parent / "target" / "release" / "audan.exe")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--config", type=Path, help="analysis.toml, e.g. one setting [beats] model to use a model's beat grid")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    songs = []
    for entry in load_entries():
        audio = args.library_dir / entry["audio"]
        if audio.exists():
            ref, end = reference_sections(entry["midi"])
            songs.append((audio, ref, end))
        else:
            print(f"skipping #{entry['nr']}: {audio} not found", file=sys.stderr)
    print(f"{len(songs)} songs", file=sys.stderr)

    grid = [
        {"--seg-kernel-half-size": k, "--seg-peak-threshold-k": t, "--seg-min-boundary-distance": d}
        for k, t, d in itertools.product(KERNELS, THRESHOLDS, MIN_DISTANCES)
    ]
    extra = {"--config": args.config, "--accept-model-license": ""} if args.config else {}
    baseline = evaluate(args.exe, args.cache_dir, songs, extra)
    unsnapped = evaluate(args.exe, args.cache_dir, songs, {**extra, "--seg-snap-downbeats": "false"})
    with ThreadPoolExecutor(args.jobs) as pool:
        scores = list(pool.map(lambda p: evaluate(args.exe, args.cache_dir, songs, {**extra, **p}), grid))

    def show(label, s):
        print(f"{label:34} F@3s {s[0]:.3f}  F@0.5s {s[1]:.3f}  pairwise {s[2]:.3f}  sections x{s[3]:.2f}")

    show("defaults", baseline)
    show("defaults, no downbeat snapping", unsnapped)
    print()
    print("best by boundary F @3 s (kernel / threshold k / min distance):")
    for p, s in sorted(zip(grid, scores), key=lambda x: -x[1][0])[: args.top]:
        show("/".join(str(v) for v in p.values()), s)


if __name__ == "__main__":
    main()
