"""Separate a song into stems you can listen to, with a model trained by this package.

    python -m trainer.separate --run runs/local-soft-aug-clean-seed0 song.mp3 [more files] --out stems

writes `stems/<song>/drums.wav`, `bass.wav`, `vocals.wav`, `other.wav` (the names of the model's source set), plus
`remainder.wav` (the song minus the stems: what the model lost or added; silent for a perfect split).
`--present drums,bass,other` says which sources are in the song (say, an instrumental with no vocals): the absent
ones are removed and their share goes to the others instead of leaking into a stem that should be empty
(see `gate_masks`); only the present stems are written. Without it every source is assumed present.
`--run` is a run folder with `run.json` and `model.pt`. The audio is cut into overlapping chunks that are
cross-faded, so a whole song needs little memory; stereo 44.1 kHz is expected (other rates are resampled, mono is
duplicated). The stems come out at the song's own level; if any would clip, all are scaled down by the same factor.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .sources import source_names
from .train import SeparationTask

MODEL_RATE = 44100


def load_model(run: Path) -> tuple[torch.nn.Module, list[str]]:
    """The separator of a finished (or interrupted) run folder, in eval mode, and its source names."""
    cfg = json.loads((run / "run.json").read_text(encoding="utf-8"))["config"]
    task = SeparationTask(cfg)
    model = task.build()
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return model.eval(), source_names(task.sep["source_set"])


@torch.no_grad()
def separate(model, audio: torch.Tensor, rate: int = MODEL_RATE, chunk_s: float = 20.0, overlap_s: float = 2.0,
             presence: torch.Tensor | None = None) -> torch.Tensor:
    """`audio` `(2, samples)` -> `(sources, 2, samples)`. The signal is processed in chunks of `chunk_s` seconds that
    overlap by about `overlap_s`; each chunk is weighted by a smooth window that never reaches zero and the weighted
    outputs are divided by the summed weights, so the chunks cross-fade and the ends of the signal stay exact.
    `presence` `(sources,)` (1 = in the song, 0 = not) gates the masks of every chunk, see `Separator.forward`."""
    total = audio.shape[-1]
    chunk = max(1, min(total, int(chunk_s * rate)))
    step = max(1, chunk - int(overlap_s * rate))
    starts = list(range(0, max(total - chunk, 0), step))
    if not starts or starts[-1] + chunk < total:
        starts.append(max(total - chunk, 0))  # the last chunk ends exactly at the end
    window = torch.sin(torch.pi * (torch.arange(chunk) + 0.5) / chunk) ** 2
    out = weights = None
    for start in starts:
        piece = audio[:, start : start + chunk]
        est = model(piece.unsqueeze(0), presence=None if presence is None else presence.unsqueeze(0))[0]  # (sources, 2, chunk)
        if out is None:
            out = torch.zeros(est.shape[0], 2, total)
            weights = torch.zeros(total)
        w = window[: est.shape[-1]]
        out[..., start : start + est.shape[-1]] += est * w
        weights[start : start + est.shape[-1]] += w
    return out / weights.clamp(min=1e-8)


def read_audio(path: Path) -> tuple[torch.Tensor, int]:
    """`(2, samples)` float32 at `MODEL_RATE`, from any file soundfile reads: mono is duplicated, extra channels
    dropped, other sample rates resampled."""
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(np.ascontiguousarray(data.T))
    audio = audio.repeat(2, 1) if audio.shape[0] == 1 else audio[:2]
    if rate != MODEL_RATE:
        import torchaudio

        audio = torchaudio.functional.resample(audio, rate, MODEL_RATE)
    return audio, MODEL_RATE


def write_stems(out_dir: Path, names: list[str], audio: torch.Tensor, stems: torch.Tensor, rate: int) -> float:
    """Writes one WAV per source and the remainder (`audio` minus the stems). Returns the gain applied to all of
    them to avoid clipping (1.0 when none was needed)."""
    tracks = {**dict(zip(names, stems)), "remainder": audio - stems.sum(dim=0)}
    peak = max(float(t.abs().max()) for t in tracks.values())
    gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, track in tracks.items():
        sf.write(out_dir / f"{name}.wav", (track * gain).T.numpy(), rate, subtype="PCM_16")
    return gain


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="trainer.separate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="run folder with run.json and model.pt")
    parser.add_argument("--out", type=Path, default=Path("stems"), help="output folder (one subfolder per song)")
    parser.add_argument("--chunk", type=float, default=20.0, help="chunk length in seconds")
    parser.add_argument("--overlap", type=float, default=2.0, help="overlap between chunks in seconds")
    parser.add_argument("--present", help="comma-separated sources that are in the song, e.g. drums,bass,other; "
                        "the others are removed (default: all)")
    parser.add_argument("files", nargs="+", type=Path, help="audio files to separate")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    model, names = load_model(args.run)
    presence = None
    if args.present is not None:
        wanted = [n.strip() for n in args.present.split(",") if n.strip()]
        unknown = sorted(set(wanted) - set(names))
        if unknown or not wanted:
            parser.error(f"--present needs source names from {', '.join(names)}, not {args.present!r}")
        presence = torch.tensor([1.0 if n in wanted else 0.0 for n in names])
    for path in args.files:
        audio, rate = read_audio(path)
        print(f"{path.name}: {audio.shape[-1] / rate:.1f} s, separating into {', '.join(names)} ...", flush=True)
        stems = separate(model, audio, rate, args.chunk, args.overlap, presence)
        out_dir = args.out / path.stem
        keep = [i for i, n in enumerate(names) if presence is None or presence[i] > 0]
        gain = write_stems(out_dir, [names[i] for i in keep], audio, stems[keep], rate)
        note = f" (scaled by {gain:.2f} to avoid clipping)" if gain < 1.0 else ""
        print(f"  wrote {out_dir}{note}", flush=True)


if __name__ == "__main__":
    main()
