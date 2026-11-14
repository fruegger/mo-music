# Overfit test (draft, not run): can the network fit three songs when nothing makes the task hard?

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
