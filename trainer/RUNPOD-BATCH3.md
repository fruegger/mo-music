# Pod batch 3 (draft): does Slakh2100 at a small weight help?

Question: Slakh2100 (synthesised, CC BY 4.0, exact stems for guitar, piano, organ, strings, synths) is the only licensable source of exact
keys and piano data. Earlier pod runs with Slakh at equal weight scored about 3 dB worse on real music than MUSDB18-only (before the
warm-up/clip/EMA fixes, softmax masks and the keys weight). Does a small share of Slakh now help keys/guitar on real music without hurting the rest?

| arm | what |
|---|---|
| D (batch 2, **not rerun**) | the recipe: six sources, hand tags + promotion, keys x4, swapped studio split, no listening labels |
| E `e-slakh10` | D + Slakh at weight 0.1 (about 7 % of the training items), MUSDB vocals mixed into 70 % of Slakh items |
| F `f-slakh30` | D + Slakh at weight 0.3 (about 19 %) |

Two seeds each (0 and 1, the seeds of D), 10,000 steps, 4 runs. Slakh is also validated as its own set (in-domain sanity check, not a decision criterion).
Configs `configs/runpod-b3-*.yaml`, runner `scripts/runpod_batch3.sh`; a local 3-step smoke test of arm F ran through (Slakh + vocal mix + swapped studio).

## Cost (estimates, not measured)
| item | estimate |
|---|---|
| network volume, 150 GB at $0.07 per GB and month | about $10.5 a month, **$2.5 for a week**; billed whether or not a pod is attached, delete when done |
| one-time Slakh download and unpack (`runpod_setup.sh data`, 12 connections, about 25 MB/s) | about 1.5 h on the cheapest pod, a GPU is not needed |
| 4 runs x 10,000 steps | unknown with Slakh: guess 0.15-0.3 s/step = 25-50 min per run, 2-3.5 h in all, about $1.5-2.5 at $0.69/h |
| total for batch 3 | about $5-6 including a week of volume |

Slakh is read as many-stem FLAC and may make the loader slower than batch 2: after the first epoch of the first run, check the time per epoch
(batch 2: roughly 1.5 min). If an epoch takes over 4 min (more than 80 min per run), stop and think before paying for the rest.

## Steps
1. Create a network volume (150 GB) in a data centre that has GPUs for the cards you can use; with the volume attached, start the cheapest pod.
2. `git clone` the repo, then `bash scripts/runpod_setup.sh data` (installs, downloads MUSDB18 and Slakh; Slakh writes `.complete` only after the track counts check out).
3. Upload `studio_prepared.zip` and unzip to `/workspace/studio_prepared`, then swap the split (Shimmering Star into train, In My Belief into test; batch 2's script does it).
4. Stop that pod, start a GPU pod on the same volume, `nvidia-smi` must list a GPU (`runpod_setup.sh train` now refuses to train without one, `ALLOW_CPU=1` overrides).
5. `HOURS_PER_RUN=2 AUTO_STOP=1 bash scripts/runpod_batch3.sh 2>&1 | tee /workspace/batch3.log`; the bundle should say 12 files.
6. Download `/workspace/batch3-results.tar.gz`, summarise with `summarize_run.py ... --last 3 --compact`. Keep the volume if Slakh is adopted, otherwise delete it.

## Decision rules (fixed before the run)
Compare each Slakh arm with D (batch 2, mean of its two seeds), per source, MUSDB18 and studio. Seed noise is about 1 dB (more on single studio
numbers), so nothing under 1 dB counts as a gain; losses are judged at 0.5 dB.
- **Slakh helps:** keys (or guitar) is at least 1 dB above D on MUSDB18 and studio on both seeds, and drums/bass/vocals stay within 0.5 dB of D. Adopt
  the largest weight that meets this (E or F); the piano class (Slakh gives exact piano) becomes worth building.
- **Dose problem:** the higher weight gains on keys/guitar but costs more than 0.5 dB on another source while the lower one does not gain: use the lower weight, no further tuning.
- **No difference** (within 1 dB everywhere): Slakh is not worth its size and complexity at this recipe; delete the volume, the piano class has no data source yet.
- **Slakh hurts** (another source more than 0.5 dB below D at both weights): drop Slakh for good (as the earlier runs suggested) and delete the volume.
- Limits of the test: MUSDB18 keys is 10 items and studio keys one song (Just Me Alone), so a gain in keys may be hard to see, while harm to drums/bass/vocals/guitar shows
  clearly. The Slakh validation score only shows that the model learns the synthetic data; it does not count.
