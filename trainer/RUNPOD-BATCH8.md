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

## Results (2026-11-15; mean of drums / bass / vocals; reference `n100-sisdr` Fit 9.63, Test 8.43, keys +6.4 / +3.7)
| arm | time | Fit | Test | keys Fit / Test | rule |
|---|---|---|---|---|---|
| sisdr_i (1.9 M) | 23 min | **10.4** (10.1 / 10.1 / 11.1) | **8.9** (8.1 / 8.5 / 10.0) | +7.4 / +3.6 | rule 5: stacks (Fit >= 10.1, Test >= 8.7), keys held; its vocal leakage is 3-4 dB **lower** than sisdr (below), so it becomes the base |
| sisdr_sw10 | 13 min | 8.7 (8.6 / 8.5 / 9.0) | 7.9 (7.5 / 7.8 / 8.4) | +5.1 / +3.4 | rule 1 fails on the scores (Fit 8.7 < 9.1, Test 7.9 < 8.1); leakage down by 5.8 / 11 dB: **rule 2, a trade** (weight too strong) |
The gains of size and of the loss are additive: size gives +0.8 Fit / +0.4 Test over L1 (batch 6) and +0.8 / +0.44 over sisdr. Against the clean-L1 base: Fit +1.6, Test +0.8; the test oracle is about 13.4. The Fit/Test gap grows (0.7 clean, 1.2 sisdr, 1.6 sisdr_i): the first sign of a gap that regularisation could work on.
The sisdr_sw10 validation curve ends with vocals at 8.0 (sisdr 9.0) and guitar below 0.

Leakage into silent sources (`scripts/silent_leakage.py`, dB re the mixture, lower is better; SISDR = batch 7 seeds 0 and 1):
| source | MUSDB18 SISDR / sisdr_i / sw10 | studio SISDR / sisdr_i / sw10 |
|---|---|---|
| vocals | -23.5 / -26.4 / **-29.3** (n 128) | -26.5 / -30.4 / **-37.6** (n 62) |
| bass | -23.3 / -24.0 / -23.6 (n 91) | -29.3 / -28.6 / -29.9 (n 7) |
| drums | -24.9 / -26.4 / -23.4 (n 43) | n 1 |
| guitar | -11.4 / -10.3 / -15.5 (n 133) | -1.0 / -0.7 / -1.1 (n 7) |
| keys | -48.1 / -52.3 / -71.7 (n 183) | -48.5 / -52.1 / -74.6 (n 6) |
| other | -89.6 / -93.0 / -97.8 | -88.8 / -91.2 / -96.1 |
For reference the L1 models had vocals -29.0 (MUSDB18) / -38.8 (studio). `silent_weight` 10 brings the vocal bleed back to the L1 level but costs 0.9 Fit and 0.5 Test; the 1.9 M network cuts it by 3-4 dB for free.

## Conclusion of batch 8 and next
- Base from now on: `sisdr` loss + the 1.9 M network (`n100-sisdr_i`: Fit 10.4, Test 8.9, keys +7.4 / +3.6), one seed so far.
- `silent_weight` between 1 and 10: 10 is too strong on the 0.27 M net; next arm `silent_weight` 3 on the 1.9 M network (rule 2), plus a seed-1 rerun of `sisdr_i` (it is the base now and rests on one seed).
- Then: the data side (Slakh keys and the studio songs, now that keys no longer collapses), and gentler augmentation, since a Fit/Test gap has started to open.
