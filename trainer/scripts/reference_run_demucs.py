"""Run a Demucs model over the excerpts that `reference_compare.py export` wrote.

    python scripts/reference_run_demucs.py --out C:/workarea/reference-test --model htdemucs_6s [--device cuda]

Needs `pip install demucs soundfile` in an environment of its own (it is not a dependency of this package: do not install it
into the trainer's environment). Models: htdemucs (4 stems), htdemucs_ft (4 stems, fine-tuned, 4x slower), htdemucs_6s (adds
guitar and piano: piano is weak), mdx_extra. The weights are downloaded on first use. Stems are written to
`<out>/refs/<model>/<set>/<nnn>/<stem>.wav`; `reference_compare.py score` reads them from there.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="htdemucs_6s")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = get_model(args.model)
    model.eval()
    mixes = sorted((args.out / "excerpts").glob("*/*/mix.wav"))
    print(f"{args.model} on {args.device}: {len(mixes)} excerpts, stems {model.sources}")
    for k, mix_file in enumerate(mixes):
        folder = args.out / "refs" / args.model / mix_file.parent.parent.name / mix_file.parent.name
        if all((folder / f"{s}.wav").exists() for s in model.sources):
            continue
        data, rate = sf.read(mix_file, dtype="float32", always_2d=True)
        assert rate == model.samplerate, f"{mix_file}: {rate} Hz, the model wants {model.samplerate}"
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)  # the normalisation demucs applies itself
        with torch.no_grad():
            out = apply_model(model, wav[None], device=args.device, split=True, overlap=0.25, progress=False)[0]
        out = out * (ref.std() + 1e-8) + ref.mean()
        folder.mkdir(parents=True, exist_ok=True)
        for s, stem in zip(model.sources, out):
            sf.write(folder / f"{s}.wav", stem.T.cpu().numpy(), rate, subtype="FLOAT")
        print(f"  {k + 1}/{len(mixes)} {mix_file.parent.parent.name}/{mix_file.parent.name}")


if __name__ == "__main__":
    main()
