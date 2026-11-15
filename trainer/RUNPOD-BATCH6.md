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

**Run order (conditional, decided 2026-11-15):** the arms are not run as one block. `n100-clean` goes first and alone (`ARMS="100:clean"`, `AUTO_STOP` off, so the pod stays up while we read it); the rest follows from rule 1:
- Fit at least 9.0: run `100:aug 100:w`. The ladder (`3`, `10`, `30`) and the sanity arm are skipped: a good fit cannot come from a pod-setup bug.
- Fit at most 7.5: run `3:clean` first (sanity: bug or real limit), then `10:clean` and `30:clean` for the ladder.
- Grey (7.5-9.0): run `3:clean`, `10:clean`, `30:clean` and `100:clean` with `SEEDS="1"`.

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
`bash scripts/runpod_setup.sh data-musdb`, Python 3.11 venv on `/workspace/venv`, activate it); the studio upload of batch 5 is not needed. Run, step 1: `ARMS="100:clean" HOURS_PER_RUN=1.5 nohup bash scripts/runpod_batch6.sh > /workspace/batch6.log 2>&1 &` (no `AUTO_STOP`; the pod idles at about $0.7/h while we read the result). Later steps: the same with the `ARMS` of the rule that applies, `AUTO_STOP=1` on the last one.
Estimate (not measured): the small network took about 1.3 min per epoch of 500 steps on the RTX 2000 Ada of batch 5 (26 min per run), clean arms render faster, so about 25-30 min per arm plus about 5 min scoring. Best case (fit at least 9.0): `n100-clean`, `n100-aug`, `n100-w`, about 1.5-2 h in all with setup, roughly $1-1.5. Worst case (all six arms, a seed-1 rerun not counted): about 3.5 h, $2-3 at $0.7/h.
**After the first epoch of `n100-clean`, check the time: over 3 min per epoch (1 h per run), stop and think.**

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
4. **Sanity (only run when `n100-clean` fits at 9.0 or below):** `n3-clean` should reach at least +10 on its own songs (the laptop did in 1,500 steps); if it does not, something in the pod setup differs from the laptop and the other arms are not trusted until that is understood. A Fit of at least 9.0 on `n100-clean` already shows the setup works, so the sanity arm is skipped then.
5. Not decided here: whether to bring Slakh back or add songs; that waits for this batch.

Limits: one seed per arm; a clean recipe without the studio songs is not the recipe that produced batches 1-5; Fit is measured on the songs trained on (memorisation counts), Test on 50 songs and 240 windows; the Fit of `n3` and `n100` rests on different numbers of windows
per song; guitar and keys are too noisy to decide anything; ten thousand steps may be too few for the larger subsets to reach their limit (read the training curve).

## Results so far (2026-11-15, RTX PRO 4000 Blackwell, $0.6/h; mean of drums / bass / vocals, SI-SDR improvement in dB)
Time: 12 min training per arm (about 15 min with scoring), a lot faster than the estimate above. Seed 1 of `n100-clean` trained to the same validation curve as seed 0 (7.54 vs 7.52, the two curves within 0.02 dB);
its scores: Fit 8.5 / 8.7 / 9.3 (mean 8.83), Test 7.6 / 8.0 / 8.8 (mean 8.13), against seed 0 Fit 8.83, Test 8.07: seed noise on the scored means is about 0.1 dB, not the 1 dB assumed in the rules. Rule 1 therefore stays in the grey zone for good (Fit 8.8 on both seeds), and the ladder and curve reading below decides: go to arm I.

| arm | Fit (own songs) | Test (50 songs) | gap |
|---|---|---|---|
| n3-clean (laptop, 1,500 steps) | 10.5 | - | |
| n10-clean | 11.0 | 5.8 | 5.1 |
| n30-clean | 10.3 | 7.1 | 3.3 |
| n100-clean seed 0 | 8.8 | 8.1 | 0.7 |
Reference D (old recipe, studio + augmentation + weights): Fit 6.5, Test 6.1; oracle on the test windows about 13.4.

Reading: rule 3 passes (Test 8.1 vs D 6.1, +2.0): the clean recipe replaces the old one. Rule 1 stays in the grey zone (Fit 8.8, between 7.5 and 9.0), but the ladder is not a steady fall: 10 -> 30 songs costs 0.7 dB of Fit, 30 -> 100 costs 1.5 dB, so the 0.27 M network's budget starts to bind between 30 and 100 songs.
Test still rises by about 1 dB per tripling of the songs (more data helps). The n100 validation curve is flat from about epoch 12 (the last 10 epochs add about 0.15 dB; the learning rate decays to 5% over the run, so this partly reflects the schedule), so more steps alone (J) is not the first thing to try.
Keys (equal weight, 11-52 windows) slides steadily downward during training (-2.5 to -12 on validation): the old keys collapse, not decided here.

## Arm I (next): does a larger network fit the 100 songs?
Config `configs/runpod-b6-i.yaml` = `runpod-b6-clean.yaml` with `base_channels` 32 (depth 3, 1.9 M parameters, the network of batch 5 arm I); everything else the same: MUSDB18 only, no augmentation, equal weights, 10,000 steps of batch 8 x 3 s, lr 0.002, seed 0.
Run (arm name `n100-i`, results in `/workspace/runs/b6-n100-i-s0`), same scoring as before (Fit on the 100 training songs, Test on the 50 test songs, same 240 windows as every other arm, so the numbers compare directly):
`ARMS="100:i" HOURS_PER_RUN=1.5 AUTO_STOP=1 nohup bash scripts/runpod_batch6.sh > /workspace/batch6-i.log 2>&1 &`   (spell out the whole line including the log redirect; a cut line sends the output to `nohup.out`)
Estimate (not measured): batch 5 took 47 min for I against about 25 min for D on its GPU, so about 2x D's 12 min here: about 25 min plus scoring, about $0.3.
`AUTO_STOP=1` stops the pod after the scoring; the files stay on the volume. Read the scores first if you want them at once.
Check the time after the first epoch: over 3 min per epoch, stop and think.

### Decision rules for I (fixed before the run)
Reference: `n100-clean` (Fit 8.8, Test 8.1, the same on both seeds). Seed noise is about 0.1 dB on the scored means; the thresholds below (1.0 dB real, 0.5 to 1.0 grey) are kept on purpose, since arm I differs in more than a seed.
1. **Capacity limits the fit:** Fit of `n100-i` at least 1.0 dB above `n100-clean`.
   - and Test also at least 1.0 dB above: capacity is a lever in the clean recipe. Next: L (`base_channels` 48, depth 4, 17.5 M, from batch 5) on `n100-clean`, to see if size keeps helping; then the added data (Slakh, studio) with the bigger network.
   - Fit up but Test not (less than 0.5 dB): the bigger network memorises the 100 songs; more data or augmentation is needed before size helps (then `n100-aug` with I).
2. **Capacity is not the limit:** Fit less than 0.5 dB above `n100-clean`. Then the limit is not parameters: batch 5 already found size flat with the old recipe, this repeats it for the clean one. Look at optimisation and the data side (learning rate, the loss, the masks/STFT resolution, `n100-aug` and `n100-w` from the original plan) rather than size.
3. **Grey (0.5 to 1.0 dB on Fit):** read the validation curve (still rising at step 10,000 for I: run I at 30,000 steps, the J arm on I); otherwise treat as rule 2 with a note.
4. **Keys:** not decided here (noisy); only noted, and compared with the `n100-clean` slide.
Not decided here: the remaining arms of the original plan (`n100-aug`, `n100-w`) and Slakh or studio songs; those wait for this result.

Limits: one seed; Fit measured on songs trained on; the 1.9 M network is 7x larger, so it also takes about 2x longer per step; batch 5 results (size flat) were measured with the old recipe and studio songs, so this is a real new test, not a repeat.

## Result of arm I (2026-11-15, 22 min training, about $0.3)
`n100-i` (1.9 M, clean, 100 songs, seed 0): Fit 9.2 / 9.5 / 10.2 (mean 9.63), Test 7.9 / 8.3 / 9.4 (mean 8.53); guitar Fit 5.5, Test 4.3; keys Fit -8.3, Test -11.1 (noise).
Against `n100-clean` (Fit 8.8, Test 8.1): **Fit +0.8, Test +0.4**; the Fit/Test gap grows from 0.7 to 1.1.
Rule 3 (grey, +0.5 to +1.0 on Fit): the validation curve was not read (the pod was terminated before the log was copied), so the "still rising" check for the 30,000-step arm is open. Treated as rule 2: 7x the parameters bought +0.8 dB of Fit and +0.4 dB of Test, the same small size effect as batch 5 (I +0.6, L +0.1), now confirmed in the clean recipe. Size is not the main lever.
L (17.5 M) is not run: the gain per parameter is too small. The remaining levers are the data (Test rises about 1 dB per tripling of the songs), augmentation as extra diversity, and the loss/mask design.
Lesson: copy the batch log (`/workspace/batch6-*.log`) off the pod, or into `/workspace/runs/`, before the pod ends; the epoch curves live only there.

## Arms aug and w (next): what do the old recipe's pieces do on the clean base?
Two arms on all 100 songs, small network (0.27 M), 10,000 steps, seed 0, each one change from `n100-clean`:
- **`n100-aug`** (`configs/runpod-b6-aug.yaml`): effects on, random stem gains -12..+6 dB, 5% dropped stems; no tempo/pitch (as on the pod before). Equal source weights. Validation and scoring stay on clean mixes.
- **`n100-w`** (`configs/runpod-b6-w.yaml`): `source_weights` vocals x4, `present_weights` keys x4 (where audible). No augmentation.
Not run: aug and w together (the old recipe), a second seed, the 1.9 M network.
Run: `ARMS="100:aug 100:w" HOURS_PER_RUN=1.5 AUTO_STOP=1 nohup bash scripts/runpod_batch6.sh > /workspace/batch6-aw.log 2>&1 &` (whole line, with the redirect). Estimate (not measured): the augmentation renders effects, so `aug` may be slower than the 12 min of a clean arm (guess 15-25 min); `w` about 12 min; scoring 5 min each; about 45 min, about $0.5.
Before the pod ends: copy `/workspace/batch6-aw.log` (the epoch curves live only there) and the score files; with `AUTO_STOP=1` the pod stops after the bundle, so copy the log into `/workspace/runs/` straight after the last arm, or run the last arm without it.
Check the time after the first epoch of `aug`: over 3 min per epoch, stop and think.

### Decision rules (fixed before the run)
Reference: `n100-clean`, Fit 8.8, Test 8.1 (mean of drums, bass, vocals, scored on the same 240 clean windows; two seeds agreed within 0.1). A difference of 0.5 dB or more on the scored means is read as real, below that as none; 1.0 dB or more as clearly real.
**aug** (the question: does augmentation help generalisation? Fit is expected to fall, since the training mix is harder than the clean windows it is scored on):
1. Test at least +0.5 over `n100-clean` (8.6 or more): augmentation helps the test score. Keep it in the recipe; a lower Fit does not matter then. Next: aug with the 1.9 M network, and as the base for added data.
2. Test within 0.5 of `n100-clean` (7.6 to 8.6): no gain at this budget. If Fit also fell by 1.0 or more, leave it out for now (it costs fit and buys nothing); if Fit is unchanged, it is harmless and stays an option for later and larger runs.
3. Test at least 0.5 below (7.6 or less): the augmentation hurts here; D's 6.1 was partly paid for by it. Leave it out; weaken it (smaller gain range, fewer effects) in a follow-up only if there is a reason to.
**w** (the question: do the weights fix keys without costing the audible sources?):
4. Keys on Test: the collapse counts as fixed if keys is above 0 dB (clean: -11 to -15, 52 windows, noisy, so only a jump of several dB counts). Drums, bass and vocals mean on Test:
   - keys fixed and the mean within 0.5 of `n100-clean` (7.6 or more): keep the weights. Next: the weights in the recipe from now on.
   - keys fixed but the mean down by 1.0 or more: the weights buy keys at the price of the audible sources. Soften them (vocals x2, keys x2) in a follow-up, or gate instead (the recogniser).
   - keys not fixed (still below 0): the weights do not do here what they did in batch 1; check with the keys Fit and the training curve before using them.
   - mean down 0.5 to 1.0: grey; read vocals alone (x4 weight, expected to rise or stay) and the curve.
**Together:** the next recipe takes what rules 1-4 keep (augmentation if rule 1, weights if rule 4 says keep); if both are kept, one run with both on the clean base (the old recipe minus the studio songs) shows whether they add up, before anything bigger is built on it. If neither is kept, the next lever is data (Slakh, studio songs) or the loss and mask design, on the clean recipe.
Not decided here: guitar and keys beyond the keys check of rule 4 (too few windows); the studio songs, Slakh; any size above 1.9 M.
Limits: one seed per arm (seed noise on the means about 0.1 dB for the clean arm, unknown for these); Fit and Test measured on clean windows, so augmentation is not scored on what it trains on; the clean base has no studio songs, so this is not a repeat of D.
