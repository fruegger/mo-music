"""`mixer render`: write fixed, seeded snippets (mix .wav + labels .json)."""

import argparse
import json
from pathlib import Path

import soundfile as sf

from .mix import MixConfig, render_snippet
from .registry import load_registry
from .sources import iter_tracks, prepare_musdb, prepare_studio


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mixer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    render = sub.add_parser("render", help="render seeded snippets to disk")
    render.add_argument("--sources", type=Path, required=True, help="dataset log (repo/prototype/sources.log)")
    render.add_argument("--dataset", required=True)
    render.add_argument("--split", default="train")
    render.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 3,5,8")
    render.add_argument("--limit", type=int, help="only the first N tracks")
    render.add_argument("--seconds", type=float, default=MixConfig.seconds)
    render.add_argument("--p-tempo", type=float, default=0.0, help="chance of a tempo change (needs pedalboard)")
    render.add_argument("--p-pitch", type=float, default=0.0, help="chance of a pitch shift (needs pedalboard)")
    render.add_argument("--fast-stretch", action="store_true", help="faster, less accurate tempo/pitch (up to ~30 cents off)")
    render.add_argument("--stems", action="store_true", help="also write each processed stem (for separation training)")
    render.add_argument("--out", type=Path, required=True)
    prepare = sub.add_parser("prepare-musdb", help="decode MUSDB18 once into FLAC stems plus estimated onsets")
    prepare.add_argument("--src", type=Path, required=True, help="MUSDB18 folder with train/ and test/")
    prepare.add_argument("--dst", type=Path, required=True)
    prepare.add_argument("--splits", nargs="+", default=["train", "test"])
    prepare.add_argument("--limit", type=int, help="decode at most N tracks in this run")
    prepare.add_argument("--force", action="store_true", help="redo tracks that are already prepared")
    studio = sub.add_parser("prepare-studio", help="turn bounced stems (<song>/tracks/*.mp3|wav|flac) into aligned FLAC stems with classes")
    studio.add_argument("--src", type=Path, required=True, help="folder with one subfolder per song, each with tracks/")
    studio.add_argument("--dst", type=Path, required=True)
    studio.add_argument("--test", nargs="*", default=[], help="songs that go to the test split (default: all train)")
    studio.add_argument("--exclude", nargs="*", default=["original_timewarp"], help="stem names to leave out")
    studio.add_argument("--force", action="store_true", help="redo songs that are already prepared")
    args = parser.parse_args(argv)

    if args.cmd == "prepare-studio":
        n = prepare_studio(args.src, args.dst, tuple(args.test), tuple(args.exclude), args.force, verbose=True)
        print(f"prepared {n} songs in {args.dst}")
        return

    if args.cmd == "prepare-musdb":
        n = prepare_musdb(args.src, args.dst, tuple(args.splits), args.limit, args.force, verbose=True)
        print(f"prepared {n} tracks in {args.dst}")
        return

    dataset = load_registry(args.sources)[args.dataset.lower()]
    config = MixConfig(
        seconds=args.seconds,
        p_tempo=args.p_tempo,
        p_pitch=args.p_pitch,
        stretch_high_quality=not args.fast_stretch,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for i, track in enumerate(iter_tracks(dataset.name, dataset.path, args.split)):
        if args.limit is not None and i >= args.limit:
            break
        for seed in parse_seeds(args.seeds):
            snippet = render_snippet(track, seed, config)
            snippet.labels["license"] = dataset.license
            name = f"{track.name}_{seed}"
            sf.write(args.out / f"{name}.wav", snippet.mix, snippet.sample_rate)
            (args.out / f"{name}.json").write_text(json.dumps(snippet.labels, indent=1), encoding="utf-8")
            if args.stems:
                stem_dir = args.out / f"{name}.stems"
                stem_dir.mkdir(exist_ok=True)
                for info, samples in zip(snippet.labels["stems"], snippet.stems):
                    sf.write(stem_dir / f"{info['stem']}.wav", samples, snippet.sample_rate)
            print(name)


if __name__ == "__main__":
    main()
