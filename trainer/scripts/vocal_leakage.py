"""Where does the vocal signal go, and what leaks into the vocal output? Compare runs on MUSDB18 test windows.

    python scripts/vocal_leakage.py --sources LOG --groups D=run1,run2 E=run3,run4 ... [--snippets 100] [--out FILE]

Clean 10 s windows of MUSDB18 test (prepared), scored as four sources (a six-source model's guitar, keys and other
outputs are added up into "other"). For every run and window:
  * SI-SDR improvement of the vocal output (the number the decision rules use);
  * a 4 x 4 table: the energy of each output that lies along each true source (the output projected on the true
    source, divided by the output's energy), so the vocal row shows what leaks into the vocal output (drums, bass,
    other columns), and the vocals column of the other rows shows vocals that went to the wrong output; the rest of
    a row is unexplained (distortion, mixed-source residue, overlap between sources).
Windows where the vocals are audible only (RMS above -50 dB). Prints group means (runs averaged) and the share of windows
where the vocal improvement is below 0 dB, which tells whether the loss is spread evenly or comes from a few windows.
"""

import argparse
from pathlib import Path

import torch
from mixer import MixConfig, iter_tracks, load_registry

from trainer.data import SnippetDataset
from trainer.separate import load_model
from trainer.separation import si_sdr

NAMES = ["drums", "bass", "vocals", "other"]


def four(model, sources, mix):
    est = model(mix)
    if len(sources) == 6:  # drums, bass, vocals, guitar, keys, other
        est = torch.cat([est[:, :3], est[:, 3:].sum(dim=1, keepdim=True)], dim=1)
    return est


def shares(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """`(4, 4)`: [i, j] = energy of estimate i along true source j, divided by the energy of estimate i (est: (4, 2, T), ref: (4, 2, T))."""
    e, r = est.reshape(4, -1), ref.reshape(4, -1)
    dot = e @ r.T
    along = dot**2 / (r.pow(2).sum(dim=1)[None, :] + 1e-12)
    return along / (e.pow(2).sum(dim=1)[:, None] + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=Path("configs/studio-sources.log"))
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--dataset", default="musdb18-prepared")
    ap.add_argument("--split", default="test")
    ap.add_argument("--snippets", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    ds = load_registry(args.sources)[args.dataset.lower()]
    tracks = list(iter_tracks(ds.name, ds.path, args.split))
    mix = MixConfig(seconds=args.seconds, gain_db_range=(0.0, 0.0), p_stem_dropped=0.0, effects=False, p_tempo=0.0, p_pitch=0.0)
    data = SnippetDataset(tracks, mix, args.snippets, 1_000_000, tasks=("separation",), source_set="musdb4")
    print(f"{ds.name}/{args.split}: {len(tracks)} songs, {args.snippets} windows of {args.seconds:.0f} s")

    groups = {}
    models = {}
    for g in args.groups:
        name, runs = g.split("=", 1)
        groups[name] = [Path(r) for r in runs.split(",")]
        for r in groups[name]:
            models[r] = load_model(r)
    rec = {r: {"imp": [], "share": []} for r in models}
    for i in range(args.snippets):
        item = data[i]
        m, tgt = item["mix"][None], item["sources"][None]
        if float(10.0 * torch.log10((tgt[0, 2] ** 2).mean() + 1e-12)) <= -50.0:
            continue  # vocals not audible here
        base = si_sdr(m.unsqueeze(1).expand_as(tgt), tgt)[0, 2]
        for r, (model, sources) in models.items():
            with torch.no_grad():
                est = four(model, sources, m)
            rec[r]["imp"].append(float(si_sdr(est, tgt)[0, 2] - base))
            rec[r]["share"].append(shares(est[0], tgt[0]))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.snippets} windows", flush=True)

    def pct(x):
        return f"{100 * x:5.1f}%"

    for g, runs in groups.items():
        imp = torch.tensor([rec[r]["imp"] for r in runs]).mean(dim=0)
        sh = torch.stack([torch.stack(rec[r]["share"]) for r in runs]).mean(dim=0)  # (windows, est, true)
        print(f"\n== {g}   windows {len(imp)}   vocal SI-SDR improvement {imp.mean():+5.2f}  "
              f"(per run: " + ", ".join(f"{sum(rec[r]['imp']) / len(rec[r]['imp']):+5.2f}" for r in runs) + f")   windows below 0 dB: {pct((imp < 0).float().mean())}")
        s = sh.mean(dim=0)
        v = NAMES.index("vocals")
        print("   energy of each output along each true source (rows = output, columns = true source):")
        print("                 " + "  ".join(f"{n:>7s}" for n in NAMES) + "   unexplained")
        for k, n in enumerate(NAMES):
            print(f"   {n:>7s} out   " + "  ".join(f"{pct(s[k, j]):>7s}" for j in range(4)) + f"   {pct(1 - s[k].sum().clamp(max=1.0)):>7s}")
        print(f"   vocal output: interference (drums+bass+other along) {pct(s[v, [0, 1, 3]].sum())}")


if __name__ == "__main__":
    main()
