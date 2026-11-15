"""How far could a mask on this spectrogram go? And do the trained runs score as well on training songs as on test songs?

    python scripts/oracle_ceiling.py --sources LOG --dataset musdb18-prepared --split test --tags ../labels/musdb18-other-tags.csv --promote-single \
        --runs NAME=run1 NAME=run2 ... [--snippets 200] [--seconds 6]

For each clean window (same renderer and seeds as `studio_window_scores.py`), prints the SI-SDR improvement of
  * the ORACLE ratio mask: |S_k|^p / sum_j |S_j|^p computed from the true stems and applied to the mixture's STFT (so
    the mixture's phase is kept, as in our separator), p = 1 and p = 2, with the STFT of our model (n_fft 2048, hop 1024).
    This is the best any mask on this STFT could do with the mixture's phase: a ceiling for the architecture, not for
    training (the network never sees the true stems);
  * every given run (`NAME=folder`), scored on the same windows.
Where a window's individual guitar/keys/other targets are unknown (only their sum is), the sum counts as one source in the
oracle's denominator and only drums, bass and vocals are scored. Run it once with `--split test` and once with
`--split train` (the songs the runs were trained on): a run far better on train than on test is data-limited.
"""

import argparse
from pathlib import Path

import torch
from mixer import MixConfig, iter_tracks, load_registry

from trainer.data import SnippetDataset
from trainer.separate import load_model
from trainer.separation import Separator, si_sdr
from trainer.tags import load_other_tags

ACTIVE_DB = -50.0


def oracle(stft: Separator, mix: torch.Tensor, sources: torch.Tensor, known: torch.Tensor, group: torch.Tensor | None, power: float) -> torch.Tensor:
    """`(S, 2, T)` ratio-mask estimates of the known sources (the others stay 0). `known` `(S,)` bool; `group` `(2, T)` or None:
    the sum of the unknown (grouped) sources, one extra source in the denominator."""
    n_fft, hop, window = stft.n_fft, stft.hop, stft.window
    length = mix.shape[-1]
    spec = stft.stft(mix[None])[0]  # (2, F+1, frames)
    parts = [sources[k] for k in range(sources.shape[0]) if known[k]]
    if group is not None:
        parts.append(group)
    mags = torch.stack([stft.stft(p[None])[0].abs() ** power for p in parts])  # (P, 2, F+1, frames)
    ratio = mags / mags.sum(dim=0, keepdim=True).clamp(min=1e-12)
    est = torch.zeros_like(sources)
    j = 0
    for k in range(sources.shape[0]):
        if known[k]:
            est[k] = torch.istft(ratio[j] * spec, n_fft, hop, window=window, length=length)
            j += 1
    return est


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=Path("configs/studio-sources.log"))
    ap.add_argument("--dataset", default="musdb18-prepared")
    ap.add_argument("--split", default="test")
    ap.add_argument("--snippets", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--tags", type=Path)
    ap.add_argument("--promote-single", action="store_true")
    ap.add_argument("--runs", nargs="*", default=[], help="NAME=run folder")
    ap.add_argument("--n-fft", type=int, default=2048, help="STFT size of the ORACLE (the runs use their own)")
    ap.add_argument("--hop", type=int, default=1024, help="STFT hop of the ORACLE (the runs use their own)")
    args = ap.parse_args()

    ds = load_registry(args.sources)[args.dataset.lower()]
    tracks = list(iter_tracks(ds.name, ds.path, args.split))
    mix_cfg = MixConfig(seconds=args.seconds, gain_db_range=(0.0, 0.0), p_stem_dropped=0.0, effects=False, p_tempo=0.0, p_pitch=0.0)
    kwargs = {}
    if args.tags:
        kwargs = {"source_tags": load_other_tags(args.tags, "musdb6"), "promote_single": args.promote_single}
    data = SnippetDataset(tracks, mix_cfg, args.snippets, 1_000_000, tasks=("separation",), source_set="musdb6", **kwargs)
    print(f"{ds.name}/{args.split}: {len(tracks)} songs, {args.snippets} windows of {args.seconds:.0f} s, oracle STFT n_fft {args.n_fft} hop {args.hop}")

    models, names = {}, None
    for r in args.runs:
        label, folder = r.split("=", 1)
        models[label], names = load_model(Path(folder))
    names = names or ["drums", "bass", "vocals", "guitar", "keys", "other"]
    stft = Separator(len(names), 2, args.n_fft, args.hop, 4, 3)
    rows = {"oracle p=1": {}, "oracle p=2": {}, **{k: {} for k in models}}  # label -> source -> [improvements]

    for i in range(args.snippets):
        item = data[i]
        mix, sources, mask = item["mix"], item["sources"], item["sources_mask"]
        level = 10.0 * torch.log10((sources**2).mean(dim=(-1, -2)) + 1e-12)
        valid = (level > ACTIVE_DB) & (mask > 0)
        if not valid.any():
            continue
        base = si_sdr(mix.unsqueeze(0).expand_as(sources), sources)
        group = item["group_target"] if "group_mask" in item and float(item["group_mask"].max()) > 0 else None
        known = mask > 0
        scores = {}
        for p, label in ((1.0, "oracle p=1"), (2.0, "oracle p=2")):
            scores[label] = si_sdr(oracle(stft, mix, sources, known, group, p), sources) - base
        for label, model in models.items():
            with torch.no_grad():
                scores[label] = si_sdr(model(mix[None])[0], sources) - base
        for label, imp in scores.items():
            for k, s in enumerate(names):
                if valid[k]:
                    rows[label].setdefault(s, []).append(float(imp[k]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.snippets} windows", flush=True)

    print(f"\n{'':14s}" + "".join(f"{s:>9s}" for s in names))
    for label, per in rows.items():
        print(f"{label:14s}" + "".join(f"{(sum(per[s]) / len(per[s])):+9.1f}" if s in per else f"{'-':>9s}" for s in names))
    print(f"{'windows':14s}" + "".join(f"{len(rows['oracle p=1'].get(s, [])):9d}" for s in names))


if __name__ == "__main__":
    main()
