# Overfit test (2026-11-15, run for the small network): can the network fit three songs when nothing makes the task hard?

Why: `GAP-DIAGNOSTICS.md` found the oracle mask on our STFT at +12..+16 dB, our runs at +6..+7, and train scores equal to test scores: the runs **underfit**. This test separates two readings:
(a) the network can fit a tiny set but not the whole of MUSDB18 with augmentation, so the problem is the task's difficulty, learning rate or loss weights under diversity;
(b) the network cannot even fit three songs, so optimisation or the architecture limits it before data, size or augmentation matter.

## Setup (all in place, smoke-tested with 3 steps)
- Data: three MUSDB18 train songs, copied into **both** splits of `C:\workarea\musdb18_overfit` (`Actions - Devil's Words` and `BigTroubles - Phantom`: `other` = guitar only; `ANiMAL - Clinic A`: `other` = synth only, i.e. keys),
  so the validation scores songs the run trains on. With `promote_single` drums, bass, vocals and guitar or keys are all exact targets. The sources log is `configs/overfit-sources.log` (the loader needs the dataset name `MUSDB18-prepared`, only the path differs).
- Config `configs/overfit-3songs.yaml`: six sources, softmax masks, 1,500 steps (6 epochs of 250, batch 8, 3 s), lr 0.002 with the usual warm-up, cosine decay, clip and EMA; **no augmentation** (no effects, tempo, pitch, random stem gains or dropped stems)
  and **equal source weights** (no vocals x4, no keys x4). Validation: 24 clean 3 s snippets of the same songs.
- Two arms, same everything else: `base_channels` 12 (D's network, out_dir `runs/overfit-3songs-d`) and `base_channels` 32 (I's network, change `out_dir` to `runs/overfit-3songs-i`).

## Run
    cd C:\workarea\mo-music\trainer
    .venv\Scripts\python -m trainer.train --config configs\overfit-3songs.yaml
(for the second arm copy the config, set `base_channels: 32` and the out_dir). The time per epoch has not been measured on the laptop: read it from the first epoch and stop if one run would take over about 90 min.
Then score the run and the oracle on the same windows:

    .venv\Scripts\python scripts\oracle_ceiling.py --sources configs\overfit-sources.log --split train --snippets 60 --seconds 6 --tags ..\labels\musdb18-other-tags.csv --promote-single --runs D3=runs\overfit-3songs-d I3=runs\overfit-3songs-i

## Decision rules (fixed before the run)
Metric: the mean SI-SDR improvement of drums, bass and vocals on the three songs from `oracle_ceiling.py` (train split = the same songs), last epoch; the oracle's number on the same windows is the ceiling (expected about +12..+15).
- **The network can fit** (at least +10 dB on the mean): reading (a). The failure on the full set comes from what training the whole set asks for (diversity, augmentation, loss weights, learning rate). Next: a recipe batch on the pod,
  changing one thing at a time from this recipe (add the augmentation back, add the weights back, more songs) to find what breaks the fit.
- **The network cannot fit** (at most +8 dB): reading (b). Optimisation or the network is the limit even in the easiest case: look at the learning rate (0.0005, 0.001, 0.004), the input (magnitude only), the mask head and the loss before any data, size or augmentation experiment.
- **Between 8 and 10 dB:** not decided; run again with 3,000 steps and read the curve (still rising: it is a speed problem, flat: a limit).
- If I (32 channels) fits clearly better than D (12) on three songs: capacity limits the fit even here. If both are equal: not capacity.
- Guitar and keys are read but not part of the decision (one song each, a handful of windows).

Limits: one seed; three songs are not representative (two guitar songs, one synth song); equal weights change the recipe on purpose, so the numbers are not comparable with batches 1-5; clean windows at natural levels.

## Result (arm base_channels 12, 1,500 steps, 15.4 min per epoch of 250 steps on the laptop CPU, 93 min in all)
Validation (24 clean 3 s snippets of the same songs), last epoch: drums 9.9, bass 9.9, vocals 10.9, guitar 10.3, keys 8.3; mean of drums/bass/vocals **10.2**. The curve rose every epoch (7.7, 8.9, 9.4, 9.9, 10.1, 10.2) with the gains shrinking.
Scored with `oracle_ceiling.py` on 60 windows of 6 s of the same songs (the rule's measurement):

| | drums | bass | vocals | guitar | keys |
|---|---|---|---|---|---|
| oracle p=1 / p=2 | 11.6 / 12.8 | 10.8 / 11.6 | 15.2 / 16.0 | 12.8 / 13.8 | 11.7 / 12.6 |
| run (small network) | **10.1** | **9.8** | **11.6** | 10.8 | 3.1 |
| share of the p=1 oracle | 87% | 91% | 76% | 84% | 26% |
| windows | 58 | 58 | 47 | 40 | 20 |

**Decision by the rules: the network can fit** (mean of drums, bass and vocals 10.5, bar +10): reading (a). On the whole dataset 10,000 and 30,000 steps stay near +6, here 1,500 steps give +10.5 on three songs, so
the network is not the limit; what the full task asks of it is (diversity, augmentation, random stem gains, loss weights, learning rate). Fitting the songs it trains on is partly memorisation: this shows the capacity to fit, not generalisation.
- Keys are the exception: 3.1 against an oracle of 11.7 on one synth song (ANiMAL - Clinic A, a rapid sequencer), 20 windows; the validation said 8.3, so the number is noisy.
- The 32-channel arm was **not run**: the rule needed it only to compare capacity, and the small network already fits.
- Next (proposal): a song-count ladder from this recipe, 3 -> 10 -> 30 -> 100 songs without augmentation and with equal weights, scored on the training songs, then the augmentation and the weights added back one at a time. Too slow on the laptop (see the
  time per step above), a pod batch.
