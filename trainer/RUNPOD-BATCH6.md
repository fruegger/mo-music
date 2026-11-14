# Pod batch 6 (draft, rules only): what stops the network from fitting the whole dataset?

Facts so far (`GAP-DIAGNOSTICS.md`, `OVERFIT-TEST.md`): an ideal mask on our STFT reaches +12..+16 dB on drums, bass and vocals; our runs sit at +6..+7 on train and test songs alike (they underfit);
size (0.27 M to 17.5 M parameters) and length (3x steps) did not move that. The same small network, with no augmentation, no random stem gains, no dropped stems and equal source weights, fits **three** songs to +10.5 dB in 1,500 steps on the laptop.
So the network can learn the task; something about the whole task stops it. Candidates: the number of songs (diversity), the augmentation, the source weights (vocals x4, keys x4). This batch changes one at a time.

## Arms (all: MUSDB18 only, six sources, softmax masks, D's network of 0.27 M parameters, 10,000 steps of batch 8 x 3 s, lr 0.002, seed 0)
| arm | songs trained on | recipe | question |
|---|---|---|---|
| `n3-clean` | 3 (the laptop overfit songs) | clean | does the laptop result hold at 10,000 steps? (sanity) |
| `n10-clean` | 10 | clean | the ladder |
| `n30-clean` | 30 | clean | the ladder |
| `n100-clean` | all 100 | clean | does fitting get harder with more songs? |
| `n100-aug` | all 100 | clean + augmentation (effects, stem gains -12..+6 dB, 5% dropped stems; no tempo/pitch, as on the pod before) | does the augmentation stop the fit? |
| `n100-w` | all 100 | clean + source weights (vocals x4, keys x4 where audible) | do the weights stop the fit? |

"Clean" = `configs/runpod-b6-clean.yaml`: no augmentation, equal source weights. Subsets are nested (each contains the smaller ones; the first three are the overfit songs, then a seeded shuffle, `scripts/make_song_subsets.py`).
The studio songs are left out on purpose (D had them at weight 0.25), so these arms are not comparable with batches 1-5 except through the same scoring (see below). Not run here: the combination aug + weights (that is the old recipe), a second seed
(only for an arm the rules put in the grey zone), the larger networks (they come after this, with the recipe this batch finds).

## Scoring
After each arm the runner scores the run with `scripts/oracle_ceiling.py` (240 clean 6 s windows, stems at their own level, hand tags with `promote_single`, SI-SDR improvement in dB, last weights):
- **Fit**: on the arm's own training songs (`n<N>/train`), `score-train.txt`;
- **Test**: on the 50 real test songs, `score-test.txt`;
and prints the oracle's ceiling next to it. The metric is the mean of drums, bass and vocals; guitar and keys are read afterwards (a handful of songs, noisy).
References from the same script and windows: D (batch 2, with studio, augmentation and weights): test 6.1 / 6.0 / 6.3 (mean 6.1), train 6.4 / 6.7 / 6.3 (mean 6.5); oracle p=1 about 12.4 / 12.1 / 15.4 on test; the laptop overfit run on 3 songs: 10.1 / 9.8 / 11.6 (mean 10.5, 1,500 steps).

## Data on the pod, cost
Same idea as batch 5: the master copy of `MUSDB18_prepared` (4.7 GB) on a network volume, copied once to the pod's local disk (`/root/data`; the runner refuses with under 15 GB free), subsets as symlinks, results on the volume (`/workspace/runs/b6-*`, bundle
`/workspace/batch6-results.tar.gz` with `run.json`, `model.pt` and the two score files per arm). The batch 5 volume may be gone: a **20 GB network volume** is enough (about $0.05 a day); one-time setup about 15 min (pod on the volume,
`bash scripts/runpod_setup.sh data-musdb`, Python 3.11 venv on `/workspace/venv`, activate it); the studio upload of batch 5 is not needed. Run: `HOURS_PER_RUN=1.5 AUTO_STOP=1 nohup bash scripts/runpod_batch6.sh > /workspace/batch6.log 2>&1 &`.
Estimate (not measured): the small network took about 1.3 min per epoch of 500 steps on the RTX 2000 Ada of batch 5 (26 min per run), clean arms render faster, so about 25-30 min per arm, six arms about 3 h, scoring about 5 min per arm, about 3.5 h in all; roughly $2-3 at $0.7/h.
**After the first epoch of `n3-clean`, check the time: over 3 min per epoch (1 h per run), stop and think.**

## Decision rules (fixed before the run)
Seed noise is about 1 dB per number on one seed, a bit less on a mean of three sources; 1.5 dB is treated as a real difference, 0.5-1.5 dB as grey.
1. **Does fitting get harder with more songs?** Fit of `n100-clean`:
   - **at least 9.0:** the number of songs is not the limit. Go to rule 2.
   - **at most 7.5:** fitting falls with the number of songs (read the ladder n3 -> n100: a steady fall per tripling of the songs means the network's budget per song is the limit). Then capacity and length matter in the clean recipe
     (batch 5 tested them with the augmentation and the weights on, which this batch shows to be a separate issue): the next batch is I (1.9 M) and J (30,000 steps) on `n100-clean`.
   - **between 7.5 and 9.0:** grey: read the ladder and the training curve (still rising at 10,000 steps: a speed problem); rerun `n100-clean` with seed 1 before deciding.
2. **What stops the fit** (only if rule 1 says the songs are not the limit): compare Fit of `n100-aug` and `n100-w` with `n100-clean`. A factor that lowers the Fit by at least 1.5 dB is a culprit (both can be). Neither: grey, the combination
   and the studio mix are the next suspects; a culprit: soften or remove it (weaker augmentation, smaller weights) in a follow-up batch, one change at a time again.
3. **Does the clean recipe also generalise?** Test of `n100-clean` against D's 6.1: at least +1.5 dB better means the old recipe's augmentation, weights or studio mix cost the real test score and the clean recipe replaces it; within 1 dB means they were not the lever for the test score
   (augmentation is expected to help the test score a little: compare Test of `n100-aug` with `n100-clean`).
4. **Sanity:** `n3-clean` should reach at least +10 on its own songs (the laptop did in 1,500 steps); if it does not, something in the pod setup differs from the laptop and the other arms are not trusted until that is understood.
5. Not decided here: whether to bring Slakh back or add songs; that waits for this batch.

Limits: one seed per arm; a clean recipe without the studio songs is not the recipe that produced batches 1-5; Fit is measured on the songs trained on (memorisation counts), Test on 50 songs and 240 windows; the Fit of `n3` and `n100` rests on different numbers of windows
per song; guitar and keys are too noisy to decide anything; ten thousand steps may be too few for the larger subsets to reach their limit (read the training curve).
