# Pod batch 9 (draft, rules only): does a longer STFT window help drums and bass?

Facts: on the 2048 / 1024 STFT our best small model (`n100-sisdr`, 0.27 M, SI-SDR loss) gets drums 7.7 and bass 8.2 on test (SI-SDR improvement, dB); the 1.9 M version (`n100-sisdr_i`) 8.1 and 8.5; Demucs `htdemucs_6s` is about 7-8 dB further ahead on drums and bass on the reference excerpts (MUSDB18: 13.6 / 13.3 against our 6.9 / 7.3; studio 18.1 / 13.7 against 10.4 / 6.6).
The oracle ceiling (`scripts/oracle_ceiling.py --n-fft --hop`: the best a ratio mask on a given STFT can do with the mixture's phase; 240 MUSDB18 test windows, `p=1`, dB; hand tags with promotion):

| n_fft / hop | drums | bass | vocals |
|---|---|---|---|
| 512 / 128 | 11.3 | 10.7 | 12.9 |
| 1024 / 256 | 11.9 | 11.2 | 14.3 |
| **2048 / 1024 (ours)** | 12.4 | 12.1 | 15.4 |
| 2048 / 512 | 12.6 | 12.2 | 15.6 |
| **4096 / 1024** | **13.1** | **13.2** | 16.3 |
| 4096 / 2048 | 12.8 | 13.0 | 16.1 |
| 8192 / 2048 | 13.0 | 13.5 | 16.3 |

A longer window lifts the ceiling by about 0.7 dB for drums and 1.1 dB for bass (4096 / 1024), 1.4 for bass at 8192; shorter windows are worse for both. Our models sit about 4 dB below the ceiling, so the representation is only a small part of the gap to Demucs; this batch asks whether the network can use it.
One arm, ONE change from `n100-sisdr`: `n_fft` 4096, `hop` 1024 (same network of 0.27 M parameters, same data: 100 MUSDB18 songs, seed 0, 10,000 steps of batch 8 x 3 s, lr 0.002, no augmentation, equal source weights, SI-SDR loss).

| arm | change | config | parameters |
|---|---|---|---|
| `sisdr_fft4096` | `separation.n_fft` 4096 (hop stays 1024) | `configs/runpod-b9-sisdr_fft4096.yaml` | 0.27 M |
Code: nothing new (`n_fft` and `hop` are config keys; the frequency axis becomes 2048 bins, divisible by 2^depth). Runner `scripts/runpod_batch9.sh` (from the batch 8 one; console log in `/workspace/runs/batch9-console.log`).
Not run: 8192 / 2048 (bass ceiling 13.5, drums slightly lower), the 1.9 M network at 4096, 4096 / 512, a multi-resolution input; they follow the result.

## Scoring
`oracle_ceiling.py` on the pod, as before: Fit on the arm's own 100 songs, Test on the 50 test songs, 240 clean windows each, mean of drums, bass, vocals; keys read separately. The oracle rows in these score files use the default 2048 / 1024 STFT (a fixed reference), not the arm's own.
References (same windows): **`n100-sisdr` (two seeds): Fit 9.6 (drums 9.35, bass 9.35, vocals 10.15), Test 8.4 (drums 7.7, bass 8.15, vocals 9.45), keys Test +3.6 / +3.7**; `n100-sisdr_i` (1.9 M): Fit 10.4, Test 8.9 (drums 8.1, bass 8.5).
The decisive numbers are drums and bass on **Test**; their mean in `n100-sisdr` is 7.93.

## Data on the pod, cost
As in batch 7/8 (same local copy, subset `b6/n100`, venv, `git pull`). Run (whole line, with the redirect): `HOURS_PER_RUN=1.5 nohup bash scripts/runpod_batch9.sh > /workspace/batch9.log 2>&1 &`; `AUTO_STOP` is off.
Estimate (a CPU step measurement: one 4096 step costs 2.05x a 2048 step, 6.6 s against 3.2 s; the 2048 arm took 12 min on the pod): **about 15-25 min** of training plus about 5 min of scoring, so about $0.2-0.3 at $0.6/h. Balance 2026-11-15: $1.50. Check the time after the first epoch: over 3 min per epoch (1 h per run), stop and think.
Copy `/workspace/batch9-results.tar.gz` and `/workspace/runs/batch9-console.log` off the volume, then stop the pod (it bills while idle).

## Decision rules (fixed before the run)
Seed noise is about 0.1 dB on the scored means (batch 7); 0.5 dB or more is read as real, 0.2 to 0.5 as grey.
1. **A lever:** the mean of drums and bass on Test at least 0.5 above `n100-sisdr` (8.4 or more), **and** the mean of drums, bass, vocals on Test no lower than 8.1 (0.3 below 8.4), **and** keys on Test still above 0. Then 4096 / 1024 becomes the STFT of the recipe; the next arm is the 1.9 M network at 4096 (expected about 45 min, about $0.5), and 8192 / 2048 for bass.
2. **Grey:** drums and bass mean gains 0.2 to 0.5 on Test: the longer window gives a small gain at twice the cost per step; it is not adopted for the cheap experiments, and reconsidered for the final model (where the cost per step matters less than the score). A seed-1 rerun only if the money allows.
3. **No effect:** drums and bass mean within 0.2 of 7.93, or lower: the ceiling lift is not realised by this network; the STFT is dropped as a lever, and the remaining gap to Demucs lies in what the network learns (capacity with the SI-SDR loss, gentler augmentation on the 1.9 M network now that Fit and Test differ by 1.6, more data).
4. **Fit versus Test:** if Fit rises by 0.5 or more but Test does not, the longer window helps the network memorise the 100 songs, not generalise: counted as rule 3, noted.
5. **Sanity:** the training runs to epoch 19 without NaN; the validation curve is read next to the scores (flat tail or still rising).
Not decided here: vocals and keys beyond the bounds in rule 1, guitar, the 1.9 M network, augmentation, data, silent-source leakage of the new model (check on the laptop with `scripts/silent_leakage.py` only if the arm becomes the base).

Limits: one seed; the oracle ceiling is for a mask with the mixture's phase on 6 s windows and says what the representation allows, not what a network learns; the CPU step ratio (2.05x) is an estimate of the GPU time; the arm changes only the STFT (the frequency axis doubles, the number of frames stays), so the network sees twice the bins and half the relative time resolution.
