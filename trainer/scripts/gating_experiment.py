"""Does knowing which sources are in the song help the separator?

    python scripts/gating_experiment.py --runs runs/local-soft-clean-seed0 runs/local-soft-aug-clean-seed0 \
        [--snippets 200] [--dropout 0.05 0.4] [--sources ../repo/prototype/sources.log]

Scores MUSDB18 test snippets (clean mixes, 3 s) with each run, with the masks gated by a presence vector
(`Separator(mix, presence=...)`: absent sources removed, the rest renormalised) taken from:

  none        no gating (the plain model)
  oracle      the truth: a source is present if its target is above -50 dB (an upper bound for any recogniser)
  noisy       the truth with every entry flipped with probability 0.1 (a recogniser that is wrong 1 time in 10)
  self<dB>    two passes, no recogniser: a source counts as present if its first-pass estimate is above <dB>
              relative to the mix

Reported per dropout setting (`--dropout` = chance that a source is left out of a snippet, i.e. how often a source
is absent) and condition, averaged over the runs: the mean SI-SDR improvement over the sources that are present
and were kept (the usual metric), how many present sources were wrongly removed (per 100 present), and the
leakage into absent sources (their estimate's level relative to the mix, dB; lower is better).
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from mixer import MixConfig, load_registry
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.separate import load_model
from trainer.separation import si_sdr

ACTIVE_DB = -50.0
SELF_THRESHOLDS = (-20.0, -30.0, -40.0)  # dB relative to the mix
FLIP = 0.1


def level_db(x: torch.Tensor) -> torch.Tensor:
    """Mean power in dB over the last two dims."""
    return 10.0 * torch.log10((x**2).mean(dim=(-1, -2)) + 1e-12)


def presences(first_pass: torch.Tensor, mix: torch.Tensor, truth: torch.Tensor, generator) -> dict[str, torch.Tensor | None]:
    """The presence vectors `(batch, sources)` of every condition for one batch."""
    oracle = (level_db(truth) > ACTIVE_DB).float()
    flips = (torch.rand(oracle.shape, generator=generator) < FLIP).float()
    noisy = (oracle - flips).abs()
    noisy[noisy.sum(dim=1) == 0] = 1.0  # a recogniser always finds something
    out = {"none": None, "oracle": oracle, "noisy": noisy}
    rel = level_db(first_pass) - level_db(mix).unsqueeze(1)
    for thr in SELF_THRESHOLDS:
        present = (rel > thr).float()
        present[present.sum(dim=1) == 0] = 1.0
        out[f"self{thr:.0f}"] = present
    return out


def evaluate(model, batches, generator) -> dict[str, dict[str, float]]:
    sums, kept, removed, present_total, leak_sum, leak_n = {}, {}, {}, {}, {}, {}
    n_sources = None
    for mix, sources in batches:
        spec, masks = model.analyse(mix)
        n_sources = masks.shape[1]
        first = model.synthesize(spec, masks, mix.shape[-1])
        active = level_db(sources) > ACTIVE_DB  # (batch, sources)
        baseline = si_sdr(mix.unsqueeze(1).expand_as(sources), sources)
        for name, presence in presences(first, mix, sources, generator).items():
            est = first if presence is None else model.synthesize(spec, _gate(model, masks, presence), mix.shape[-1])
            kept_mask = active if presence is None else active & (presence > 0)
            gain = si_sdr(est, sources) - baseline  # (batch, sources)
            sums.setdefault(name, torch.zeros(n_sources)).add_((gain * kept_mask).sum(dim=0))
            kept.setdefault(name, torch.zeros(n_sources)).add_(kept_mask.sum(dim=0))
            present_total.setdefault(name, torch.zeros(n_sources)).add_(active.sum(dim=0))
            removed.setdefault(name, torch.zeros(n_sources)).add_((active & ~kept_mask).sum(dim=0))
            absent = ~active
            rel = level_db(est) - level_db(mix).unsqueeze(1)
            leak_sum.setdefault(name, torch.zeros(1)).add_((rel * absent).sum())
            leak_n.setdefault(name, torch.zeros(1)).add_(absent.sum())
    result = {}
    for name in sums:
        per_source = sums[name] / kept[name].clamp(min=1)
        result[name] = {
            "improvement": float(per_source[kept[name] > 0].mean()),
            "removed_per_100": float(100 * removed[name].sum() / present_total[name].sum().clamp(min=1)),
            "leakage_db": float(leak_sum[name] / leak_n[name].clamp(min=1)) if leak_n[name] > 0 else float("nan"),
            "absent_cases": int(leak_n[name]),
        }
    return result


def _gate(model, masks, presence):
    from trainer.separation import gate_masks

    return gate_masks(masks, presence, model.net.mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--snippets", type=int, default=200)
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.05, 0.4])
    parser.add_argument("--sources", type=Path, default=Path("../repo/prototype/sources.log"))
    parser.add_argument("--out", type=Path, help="also write the numbers as JSON")
    args = parser.parse_args()

    registry = load_registry(str(args.sources))
    dataset = registry["musdb18-prepared"]
    base_mix = json.loads((args.runs[0] / "run.json").read_text(encoding="utf-8"))["config"]["mix"]
    models = [load_model(run)[0] for run in args.runs]
    torch.manual_seed(0)
    report = {}
    for dropout in args.dropout:
        mix_cfg = MixConfig.from_dict({**base_mix, "effects": False, "p_tempo": 0.0, "p_pitch": 0.0, "p_stem_dropped": dropout})
        data = SnippetDataset.from_dataset(dataset, "test", mix_cfg, args.snippets, 1_000_000, tasks=("separation",), source_set="musdb4")
        with torch.no_grad():
            batches = [(b["mix"], b["sources"]) for b in DataLoader(data, 8)]
            per_model = [evaluate(m, batches, torch.Generator().manual_seed(1)) for m in models]
        report[str(dropout)] = per_model
        print(f"\nsource dropout {dropout}: {args.snippets} snippets, {len(models)} run(s), "
              f"{per_model[0]['none']['absent_cases']} absent-source cases")
        print(f"{'condition':10s} {'improvement dB':>15s} {'present removed /100':>21s} {'leakage into absent dB':>23s}")
        for name in per_model[0]:
            vals = {k: [r[name][k] for r in per_model] for k in ("improvement", "removed_per_100", "leakage_db")}
            mean = {k: sum(v) / len(v) for k, v in vals.items()}
            spread = max(vals["improvement"]) - min(vals["improvement"])
            print(f"{name:10s} {mean['improvement']:+9.2f} (+/-{spread / 2:.2f}) {mean['removed_per_100']:21.1f} {mean['leakage_db']:23.1f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
