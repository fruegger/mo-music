"""Run songs through Demucs and write a local listening page: the mix and every stem, one player each.

    python scripts/demucs_listen.py --songs-dir C:/workarea/mo-music/audan/resources/songs --out C:/workarea/demucs-listen \
        --files "01 Enter Sandman.mp3" "03 Take Five.mp3" ... [--model htdemucs_6s] [--seconds 90] [--device cpu]

Needs the environment of `reference_run_demucs.py` (demucs, soundfile; not the trainer's). For each song an excerpt of
`--seconds` seconds from the middle (0 = the whole song) is separated; mix and stems are written as FLAC under
`<out>/<song>/` and `<out>/index.html` lists them with audio players (open it in a browser). The songs are the user's own copies and
stay on this machine: nothing is uploaded. Output is never committed.
"""

import argparse
import html
import re
import sys
from pathlib import Path

import julius
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--model", default="htdemucs_6s")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = get_model(args.model)
    model.eval()
    print(f"{args.model} on {args.device}: {len(args.files)} songs, stems {model.sources}", flush=True)
    done = []
    for k, name in enumerate(args.files):
        path = args.songs_dir / name
        info = sf.info(path)
        if args.seconds > 0 and info.duration > args.seconds:
            start = int(max(0.0, 0.4 * info.duration - args.seconds / 2) * info.samplerate)
            frames = int(args.seconds * info.samplerate)
        else:
            start, frames = 0, -1
        data, rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
        start_s = start / rate
        if rate != model.samplerate:  # e.g. 48 kHz files: resample the excerpt to the model's rate
            data = julius.resample_frac(torch.from_numpy(np.ascontiguousarray(data.T)), rate, model.samplerate).T.numpy()
            rate = model.samplerate
        song = re.sub(r"[^A-Za-z0-9\- ]+", "", path.stem).strip()[:60]  # ASCII only: safe for the console and for URLs
        folder = args.out / song
        folder.mkdir(parents=True, exist_ok=True)
        if not all((folder / f"{s}.flac").exists() for s in ["mix", *model.sources]):  # a rerun skips finished songs
            wav = torch.from_numpy(np.ascontiguousarray(data.T))
            ref = wav.mean(0)
            norm = (wav - ref.mean()) / (ref.std() + 1e-8)  # the normalisation demucs applies itself
            with torch.no_grad():
                out = apply_model(model, norm[None], device=args.device, split=True, overlap=0.25, progress=False)[0]
            out = out * (ref.std() + 1e-8) + ref.mean()
            sf.write(folder / "mix.flac", data, rate, subtype="PCM_16")
            for s, stem in zip(model.sources, out):
                sf.write(folder / f"{s}.flac", stem.T.cpu().numpy(), rate, subtype="PCM_16")
        done.append((song, start_s, data.shape[0] / rate))
        print(f"  {k + 1}/{len(args.files)} {song} ({start_s:.0f} s + {data.shape[0] / rate:.0f} s)", flush=True)

    rows = []
    for song, start, length in done:
        players = "".join(
            f'<div class="p"><span>{n}</span><audio controls preload="none" src="{html.escape(song)}/{n}.flac"></audio></div>'
            for n in ["mix", *model.sources]
        )
        rows.append(f"<section><h2>{html.escape(song)} <small>{start:.0f} s + {length:.0f} s</small></h2>{players}</section>")
    page = (
        '<!doctype html><meta charset="utf-8"><title>Demucs stems</title>'
        "<style>body{font:15px sans-serif;margin:1.5em;max-width:60em}section{margin-bottom:1.5em}small{color:#777;font-weight:normal}"
        ".p{display:flex;gap:1em;align-items:center}.p span{width:6em}audio{height:32px}</style>"
        f"<h1>{html.escape(args.model)}: mix and stems</h1>" + "".join(rows)
    )
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"written {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
