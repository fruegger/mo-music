# Pod batch 5 (draft, rules only): is the separator too small and too briefly trained?

Reference comparison (2026-11-10, 12 clean 10 s excerpts per set, `scripts/reference_compare.py`): Demucs `htdemucs_6s` is 8-10 dB ahead of our
recipe on drums, bass, guitar and the group (MUSDB18 drums +13.6 vs +5.4, studio drums +18.1 vs +7.8, studio guitar +12.5 vs 4-5.7), about 8 dB on
vocals. Every recipe effect measured so far (keys weight, labels, Slakh) was 1-2 dB. Our network has **0.27 M parameters** (`base_channels` 12,
depth 3, 10,000 steps, batch 8 of 3 s); Demucs is tens of millions. Question: how much of the gap does capacity or training length close,
with the data and recipe unchanged (D of batch 2: MUSDB18 + studio, six sources, softmax, vocals x4, keys x4, swapped studio split, no Slakh)?

| arm | change from D | parameters | steps |
|---|---|---|---|
| D (batch 2, **not rerun**) | the recipe | 0.27 M | 10,000 |
| I `i-wide` | `base_channels` 32, depth 3 | 1.9 M | 10,000 |
| L `l-large` | `base_channels` 48, depth 4 | 17.5 M | 10,000 |
| J `j-long` | D's network, 30,000 steps (60 epochs of 500) | 0.27 M | 30,000 |

Seed 0 only at first (3 runs). A second seed (seed 1) is run for an arm only when the rules below put it in the grey zone. Everything else as D:
same sources, same learning rate (not tuned for the bigger nets, see the limits), no Slakh, `val_snippets` 48 clean validation.
Parameter counts are measured (`Separator(6, 2, 2048, 1024, base, depth)`); the speed on the pod is not: a CPU forward-backward of one batch is
about 3x slower for I and about 7x for L than for D, the GPU factor is unknown because the earlier runs were partly data-bound.

## Cost (estimates, not measured)
D took about 25 min per run (A4500, 1.1 min per epoch with 12 workers; 2 min on the Blackwell card). Guess: I about 30-40 min, L about 1-1.5 h,
J about 75 min: 3-3.5 h in all, about $2-3 at $0.7/h. The network volume (about $0.35/day) still holds MUSDB18_prepared and the swapped studio
split, so it is kept until this batch is done (no Slakh needed). **After the first epoch of L, check the time: over 6 min per epoch (2 h per
run), stop and think.**

## Decision rules (fixed before the run)
Metric: the mean of the SI-SDR improvement of drums, bass and vocals on MUSDB18 (D: 5.7 / 6.1 / 5.5, mean 5.8) and on the studio test
(D: 7.75 / 3.2 / 9.45, mean 6.8), last 3 epochs, as in batches 1-4. Seed noise is about 1 dB per number, a bit less on a mean of three, so a gain of 2 dB
is clear, 1-2 dB is a grey zone. Guitar and keys are read afterwards, not part of the decision.
- **Capacity is a main lever:** I or L gains at least 2 dB on the MUSDB18 mean and at least 1.5 dB on the studio mean, and none of drums/bass/vocals
  is lower than D on MUSDB18. If L is at least 0.5 dB above I, size keeps helping: the next batch scales further (and then data matters again).
- **Training length is a lever:** J gains at least 1 dB on the MUSDB18 mean (the long laptop run suggested more steps help). Then longer
  training is added to whichever size wins in the next batch.
- **Grey zone** (best arm between +1 and +2 dB on the MUSDB18 mean): run seed 1 of that arm only, then decide with the two-seed mean by the rules above.
- **Nothing helps** (I, L and J all within 1 dB of D on both sets): capacity and length are not the bottleneck at this scale. The gap is then
  the architecture (a mask on a spectrogram against a hybrid waveform + spectrogram model) or the amount of data; no more tuning of this network, the next question is a
  different architecture.
- **A bigger net is worse** (more than 1 dB below D): compare the training loss with D's. A lower training loss with a worse validation is
  overfitting on about 86 MUSDB18 songs (data is the limit); a higher or NaN loss is the learning rate (rerun that arm once at lr 0.001, this is
  the same arm, not a new one).
- Not decided here: whether any of this is worth deploying; the table of 12 excerpts per set is for judging by ear and by dB.

Limits: one seed per arm at first; the learning rate is D's (0.002) and not tuned for the wider nets; the studio numbers are 12 excerpts of 5 songs
(vocals 6 and keys 4 audible), which is why the decision uses MUSDB18 first and the studio only as a check; the reference model has more data and
a different architecture, so closing the gap fully is not expected from size alone.
