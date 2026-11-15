# Pod batch 8 (draft, rules only): fix the vocal bleed of the SI-SDR loss, and does size stack on it?

Facts (`RUNPOD-BATCH7.md`): the SI-SDR loss (negative SI-SDR x0.01 on audible sources, plain L1 on silent ones) on the clean recipe gives Fit 9.6, Test 8.4 on both seeds (clean L1: 8.8 / 8.1), and it fixes the keys collapse (keys Fit +6.4, Test +3.7, from -11..-15) with equal weights.
Its cost, measured on the laptop (`scripts/silent_leakage.py`): in windows where a source is silent the estimate of that source is louder than with L1, most for **vocals** (MUSDB18 -29.0 -> -23.5 dB re the mixture, share above -30 dB 57% -> 82%; studio -38.8 -> -26.5 dB, 18% -> 80%), a little for bass and drums. Arm I of batch 6 (1.9 M, L1) gained Fit +0.8, Test +0.4 over the 0.27 M clean net.
Two arms, each ONE change from `n100-sisdr` (same data, 100 MUSDB18 songs, seed 0, 10,000 steps of batch 8 x 3 s, lr 0.002, no augmentation, equal source weights):

| arm | change | config | parameters | question |
|---|---|---|---|---|
| `sisdr_sw10` | `silent_weight` 10: the L1 error of every source whose target is below -60 dB counts 10 times as much | `configs/runpod-b8-sisdr_sw10.yaml` | 0.27 M | does a stronger silent-source term remove the vocal bleed without costing the scores? (in the L1 regime weights 1, 3, 10 lowered leakage but wrecked the audible sources; here the audible sources have their own SI-SDR term) |
| `sisdr_i` | `base_channels` 32 (depth 3) | `configs/runpod-b8-sisdr_i.yaml` | 1.9 M | do the size gain of arm I and the loss gain stack? |

Why 10 and not 3: in the sisdr loss the L1 term of a silent source is small next to the SI-SDR terms (a leak at -25 dB re the mixture is an L1 of a few 1e-3, an SI-SDR term is 0.01 x the dB), so a weight of 3 might not move anything; 10 is the top of the old range. If 10 shows no effect, the term is too weak at any tried weight and the lever is a dedicated term or presence gating, not this weight.
Code: nothing new (`separation.silent_weight` already exists; a test checks that it raises the sisdr loss of a leak, `tests/test_loss_modes.py`). Runner `scripts/runpod_batch8.sh` (from the batch 7 one; console log in `/workspace/runs/batch8-console.log`).

## Scoring
Pod: as before, `oracle_ceiling.py` Fit (own 100 songs) and Test (50 songs), 240 clean windows, mean of drums, bass, vocals, keys read separately. References (same windows): **`n100-sisdr` Fit 9.6 (9.63 / 9.63), Test 8.4 (8.43 / 8.43), keys Fit +6.4, Test +3.6 / +3.7**; arm I of batch 6 (L1) Fit 9.6, Test 8.5.
Laptop, after the run: copy `/workspace/batch8-results.tar.gz` and the console log off the volume, then `scripts/silent_leakage.py` on MUSDB18 test (600 windows) and the studio test (400 windows) with the same group syntax (`SISDR=...b7-n100-sisdr-s0,...s1` as reference, the new arm as its own group): the leakage numbers are decided there, not on the pod.
Reference leakage of `sisdr` (mean dB re the mixture; share above -30 dB): MUSDB18 vocals -23.5 (82%), bass -23.3, drums -24.9; studio vocals -26.5 (80%); keys -48 (silent windows).

## Data on the pod, cost
As in batch 7 (same local copy, subset `b6/n100`, venv, `git pull`). Run (whole line, with the redirect): `HOURS_PER_RUN=1.5 nohup bash scripts/runpod_batch8.sh > /workspace/batch8.log 2>&1 &`; the arms run in the order sisdr_i, sisdr_sw10; `AUTO_STOP` stays off.
Estimate (measured on the same GPU): `sisdr_sw10` 12-13 min, `sisdr_i` about 22 min (arm I of batch 6), scoring about 5 min each: about 45 min, roughly $0.5 at $0.6/h. Check the time after the first epoch of `sisdr_i`: over 3 min per epoch, stop and think.
Copy the bundle and the console log off the volume before the pod is stopped, then stop it.

## Decision rules (fixed before the run)
Seed noise is about 0.1 dB on the scored means (batch 7: two seeds of `sisdr` agreed to 0.1); 0.5 dB or more is read as real. Leakage differences of 3 dB or more are read as real (the two models of each group differ by about 1-2 dB).
**sisdr_sw10** (the vocal bleed):
1. **Fixes it:** vocal leakage on MUSDB18 at least 3 dB lower than `sisdr` (-26.5 or lower) and on the studio songs at least 3 dB lower (-29.5 or lower), **and** the scores hold: Fit at least 9.1 (within 0.5 of 9.6), Test at least 8.1 (within 0.3 of 8.4), keys on Test still above 0. Adopt `silent_weight` 10 in the recipe.
2. **Trades:** leakage down by 3 dB or more but a score below the bounds: the weight is too strong; the next step is 3 (one more arm, not a batch of its own), or the dedicated term below.
3. **No effect:** leakage within 1.5 dB of `sisdr`: the silent-source L1 term is too weak at weight 10 as well; the bleed needs another lever (a dedicated silent-source penalty on the estimate's level, or presence gating with the recogniser, which gave +2.3 dB at 40% missing sources), and the weight is dropped.
4. **Worse scores with no leakage gain** (Fit below 9.1 and leakage within 1.5 dB): dropped, noted.

**sisdr_i** (does size stack):
5. **Stacks:** Fit at least 10.1 (0.5 above 9.6) and Test at least 8.7 (0.3 above 8.4): the 1.9 M network becomes the base for the next experiments (the data side: Slakh keys, studio songs), if its vocal leakage is not worse than `sisdr` by 3 dB or more.
6. **Does not stack:** Fit below 10.1 or Test below 8.7: the gains overlap; stay at 0.27 M (cheaper, 12 min) for the next experiments, size remains a small lever.
7. **Keys:** watch it in both arms (Test above 0 is the keep line); a keys regression in `sisdr_i` counts against rule 5.

Not decided here: guitar (leaks in every model; a guess is that `other` is almost always off), a second seed (only for an arm the rules put on a boundary), the data side, augmentation, combining the two arms (only if both pass their rules: one run with silent_weight 10 on the 1.9 M network).

Limits: one seed per arm; the leakage rests on few windows for drums, bass, keys and guitar (vocals on MUSDB18 and the studio songs are solid, n 128 and 62); `silent_weight` multiplies the L1 term of the silent sources only, it cannot move audible-source scores except through the shared network; the 6 s windows of the check are clean mixes.
