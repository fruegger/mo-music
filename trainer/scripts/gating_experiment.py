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
  pred<p>     with `--recogniser RUN` (a `presence` run): the recogniser's prediction on the 3 s mono mix, a source
              is present if its probability is at least <p>

Reported per dropout setting (`--dropout` = chance that a source is left out of a snippet, i.e. how often a source
is absent) and condition, averaged over the runs: the mean SI-SDR improvement over the sources that are present
and were kept (the usual metric), how many present sources were wrongly removed (per 100 present), and the
leakage into absent sources (their estimate's level relative to the mix, dB; lower is better).

A six-source separator (musdb6: drums, bass, vocals, guitar, keys, other) is scored as four units, as MUSDB18 only
knows the sum of guitar, keys and other: the estimates and targets of the group are added up into one `other` unit.
The gate still acts on the six masks. A unit is present when its target is above -50 dB, kept when at least one of its
sources is let through. The oracle opens all three group sources when the group is audible (the truth is only known
for the sum); `--recogniser` takes a six-source recogniser then.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from mixer import MixConfig, load_registry
from torch.utils.data import DataLoader

from trainer.data import SnippetDataset
from trainer.recognise import load_recogniser
from trainer.separate import load_model
from trainer.separation import si_sdr
from trainer.sources import GROUPS

ACTIVE_DB = -50.0
SELF_THRESHOLDS = (-20.0, -30.0, -40.0)  # dB relative to the mix
PRED_THRESHOLDS = (0.2, 0.35, 0.5)
FLIP = 0.1


def level_db(x: torch.Tensor) -> torch.Tensor:
    """Mean power in dB over the last two dims."""
    return 10.0 * torch.log10((x**2).mean(dim=(-1, -2)) + 1e-12)


def fold(x: torch.Tensor, group: list[int] | None) -> torch.Tensor:
    """`(batch, sources, ...)` -> `(batch, units, ...)`: the group sources added up into one unit (the last one)."""
    if group is None:
        return x
    rest = [i for i in range(x.shape[1]) if i not in group]
    return torch.cat([x[:, rest], x[:, group].sum(dim=1, keepdim=True)], dim=1)


def unit_presence(presence: torch.Tensor, group: list[int] | None) -> torch.Tensor:
    """`(batch, sources)` -> `(batch, units)`: a unit is open when any of its sources is."""
    if group is None:
        return presence
    rest = [i for i in range(presence.shape[1]) if i not in group]
    return torch.cat([presence[:, rest], presence[:, group].amax(dim=1, keepdim=True)], dim=1)


def expand(units: torch.Tensor, n_sources: int, group: list[int] | None) -> torch.Tensor:
    """`(batch, units)` -> `(batch, sources)`: every source of a unit gets the unit's value."""
    if group is None:
        return units
    rest = [i for i in range(n_sources) if i not in group]
    out = torch.zeros(units.shape[0], n_sources)
    out[:, rest] = units[:, :-1]
    out[:, group] = units[:, -1:]
    return out


def presences(first_pass: torch.Tensor, mix: torch.Tensor, truth: torch.Tensor, generator, recogniser=None,
              group: list[int] | None = None) -> dict[str, torch.Tensor | None]:
    """The presence vectors `(batch, sources)` of every condition for one batch."""
    oracle = (level_db(fold(truth, group)) > ACTIVE_DB).float()  # per unit
    flips = (torch.rand(oracle.shape, generator=generator) < FLIP).float()
    noisy = (oracle - flips).abs()
    noisy[noisy.sum(dim=1) == 0] = 1.0  # a recogniser always finds something
    n = first_pass.shape[1]
    out = {"none": None, "oracle": expand(oracle, n, group), "noisy": expand(noisy, n, group)}
    rel = level_db(first_pass) - level_db(mix).unsqueeze(1)
    for thr in SELF_THRESHOLDS:
        present = (rel > thr).float()
        present[present.sum(dim=1) == 0] = 1.0
        out[f"self{thr:.0f}"] = present
    if recogniser is not None:
        probs = recogniser(mix.mean(dim=1)).sigmoid()
        for thr in PRED_THRESHOLDS:
            present = (probs >= thr).float()
            present[present.sum(dim=1) == 0] = 1.0
            out[f"pred{thr:.2f}"] = present
    return out


def evaluate(model, batches, generator, recogniser=None, group: list[int] | None = None) -> dict[str, dict[str, float]]:
    sums, kept, removed, present_total, leak_sum, leak_n = {}, {}, {}, {}, {}, {}
    n_sources = None
    for mix, sources in batches:
        spec, masks = model.analyse(mix)
        first = model.synthesize(spec, masks, mix.shape[-1])
        sources_full, sources = sources, fold(sources, group)
        n_sources = sources.shape[1]
        active = level_db(sources) > ACTIVE_DB  # (batch, units)
        baseline = si_sdr(mix.unsqueeze(1).expand_as(sources), sources)
        for name, presence in presences(first, mix, sources_full, generator, recogniser, group).items():
            est = first if presence is None else model.synthesize(spec, _gate(model, masks, presence), mix.shape[-1])
            est = fold(est, group)
            kept_mask = active if presence is None else active & (unit_presence(presence, group) > 0)
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
    parser.add_argument("--recogniser", type=Path, help="a presence run folder: adds the pred<p> conditions")
    parser.add_argument("--out", type=Path, help="also write the numbers as JSON")
    args = parser.parse_args()

    registry = load_registry(str(args.sources))
    dataset = registry["musdb18-prepared"]
    config = json.loads((args.runs[0] / "run.json").read_text(encoding="utf-8"))["config"]
    base_mix = config["mix"]
    source_set = config.get("separation", {}).get("source_set", "musdb4")
    group = GROUPS.get(source_set)
    models = [load_model(run)[0] for run in args.runs]
    recogniser = load_recogniser(args.recogniser)[0] if args.recogniser else None
    torch.manual_seed(0)
    report = {}
    for dropout in args.dropout:
        mix_cfg = MixConfig.from_dict({**base_mix, "effects": False, "p_tempo": 0.0, "p_pitch": 0.0, "p_stem_dropped": dropout})
        data = SnippetDataset.from_dataset(dataset, "test", mix_cfg, args.snippets, 1_000_000, tasks=("separation",), source_set=source_set)
        with torch.no_grad():
            batches = [(b["mix"], b["sources"]) for b in DataLoader(data, 8)]
            per_model = [evaluate(m, batches, torch.Generator().manual_seed(1), recogniser, group) for m in models]
            if recogniser is not None:
                print(f"\nrecogniser alone, dropout {dropout} (probability >= 0.5), per source: precision / recall / present share")
                probs = torch.cat([recogniser(mix.mean(dim=1)).sigmoid() for mix, _ in batches])
                if group is not None:  # the group as one unit: noisy-OR of its sources
                    rest = [i for i in range(probs.shape[1]) if i not in group]
                    probs = torch.cat([probs[:, rest], 1.0 - (1.0 - probs[:, group]).prod(dim=1, keepdim=True)], dim=1)
                truth = torch.cat([(level_db(fold(src, group)) > ACTIVE_DB).float() for _, src in batches])
                for k, name in enumerate(["drums", "bass", "vocals", "other"]):
                    pred = probs[:, k] >= 0.5
                    tp = float((pred & (truth[:, k] > 0)).sum())
                    prec, rec = tp / max(float(pred.sum()), 1), tp / max(float(truth[:, k].sum()), 1)
                    print(f"  {name:7s} {prec:5.2f} / {rec:5.2f} / {float(truth[:, k].mean()):5.2f}")
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
