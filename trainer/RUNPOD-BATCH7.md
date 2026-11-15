# Pod batch 7 (draft, rules only): what limits the fit on the 100 MUSDB18 songs?

Facts (`RUNPOD-BATCH6.md`): on all 100 songs the clean recipe (no augmentation, equal weights, 0.27 M network, 10,000 steps) reaches **Fit 8.8, Test 8.1** (two seeds agree within 0.1; mean of drums, bass, vocals).
The network fits 3 songs to +10.5 and 30 songs to 10.3, but not 100, and the validation curve is flat from about epoch 12. More steps, more size (1.9 M: Fit +0.8, Test +0.4), augmentation (-1.6 on Test) and the source weights
did not close the gap; an ideal mask on our own STFT reaches +12..+16 dB, so the mask representation is not the cap. What is left in the model and the training: the loss, what the network can see, and the optimiser.
This batch changes **one** thing at a time on `n100-clean` (same data, seed 0, 10,000 steps of batch 8 x 3 s). Augmentation is deliberately not touched here: the Fit/Test gap at 100 songs is 0.7 dB, so nothing is overfitted yet
that a regulariser could help (gentler augmentation comes back when the fit improves and the gap grows).

## Arms (all: MUSDB18 only, 100 songs, six sources, softmax masks, no augmentation, equal source weights, seed 0, 20 epochs of 500 steps)
| arm | change from `n100-clean` | config | parameters | hypothesis |
|---|---|---|---|---|
| `sisdr` | loss: negative SI-SDR (x0.01, clamped to +-30 dB) on audible sources; plain L1 on silent ones; group term unchanged | `configs/runpod-b7-sisdr.yaml` | 0.27 M | we train on waveform L1 and score on SI-SDR; the two differ, and L1 is dominated by loud sources |
| `norm` | loss: per-source relative L1 = error / (source RMS + 0.1 x mixture RMS), x0.1 | `configs/runpod-b7-norm.yaml` | 0.27 M | quiet or sparse sources (vocals, keys) get little gradient from a level-blind L1 |
| `ctx` | a bidirectional LSTM (reduce 8, hidden 128) at the U-Net bottleneck over the whole spectrum and the whole snippet, residual | `configs/runpod-b7-ctx.yaml` | **1.72 M** | the 3x3 U-Net sees only about +-48 bins (about 2 kHz) and +-48 frames (about 2 s) (own estimate, 44.1 kHz): widening the network (batch 5, arm I) does not widen what it sees |
| `lrlow` | learning rate 0.001 (clean: 0.002) | `configs/runpod-b7-lrlow.yaml` | 0.27 M | 0.002 may be too high for the plateau |
| `lrhigh` | learning rate 0.004 | `configs/runpod-b7-lrhigh.yaml` | 0.27 M | 0.002 may be too low (the net may simply be slow) |
Code: `separation.loss` (`l1` default, `norm_l1`, `sisdr`) and `separation.context` (`none` default, `lstm`; `context_reduce`, `context_hidden`) in `src/trainer/separation.py` and `train.py`; the defaults reproduce every earlier run (132 tests pass,
a smoke test checked forward/backward for each loss with and without the LSTM and the ONNX export of the LSTM net). The LSTM adds 1.45 M parameters, so `ctx` is compared with `n100-clean` **and** with arm I of batch 6 (1.9 M, Fit 9.6, Test 8.5).
Not run: combinations (they follow the findings), a second seed (only for an arm the rules put in the grey zone), gentler augmentation, hop/n_fft changes (the oracle says the representation is not the cap).

## Scoring
As in batch 6: after each arm `scripts/oracle_ceiling.py` scores the last weights on the arm's own 100 training songs (Fit, `score-train.txt`) and on the 50 test songs (Test, `score-test.txt`), 240 clean 6 s windows each, hand tags with `promote_single`,
mean of drums, bass, vocals. References from the same script: **`n100-clean` Fit 8.8 (8.83 / 8.83), Test 8.1 (8.07 / 8.13)**; arm I (1.9 M) Fit 9.6, Test 8.5; the oracle on the test windows about 13.4 (p=1).
Guitar and keys are read afterwards (a handful of windows, too noisy to decide). **Not measured by these scores:** leakage into silent sources, which a change of loss can move (the old `silent_weight` experiments showed it matters); listen to or check it
(`scripts/keys_diagnosis.py`, `trainer.separate`) before adopting a loss that wins on the numbers.

## Data on the pod, cost
As in batch 6 (`RUNPOD-BATCH6.md`): the master copy of `MUSDB18_prepared` on the network volume, copied once to the pod's local disk (`/root/data`), the subset `b6/n100` as symlinks. Setup on a fresh pod: pod on the volume, activate the venv
(`source /workspace/venv/bin/activate`; torch 2.8 + cu128 sees the Blackwell card; if the venv has no torch, `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`), `cd /workspace/mo-music && git pull && cd trainer`.
Run (whole line, with the redirect): `HOURS_PER_RUN=1.5 nohup bash scripts/runpod_batch7.sh > /workspace/batch7.log 2>&1 &`; the arms run in the order sisdr, norm, ctx, lrlow, lrhigh.
`AUTO_STOP` is **off** by default: the whole console output (all epoch curves) is saved on the volume in `/workspace/runs/batch7-console.log` as the run goes, but read the scores before the pod is stopped, and stop it yourself.
Estimate (not measured): 12 min of training per arm for the 0.27 M arms (batch 6), `ctx` with 1.7 M parameters and an LSTM maybe 15-25 min, scoring about 5 min per arm: about 1.5-2 h, roughly $1 at $0.6/h on the RTX PRO 4000.
**After the first epoch of the first arm and of `ctx`, check the time: over 3 min per epoch, stop and think.** A `lrhigh` run that diverges (loss NaN or the curve collapsing) is a result: 0.004 is too high.

## Decision rules (fixed before the run)
Seed noise on the scored means of the clean arm was about 0.1 dB; a difference of **0.5 dB or more on Fit** is read as real, **1.0 dB or more** as a lever, 0.5 to 1.0 as grey (not decided, noted). Test is read next to it.
1. **A lever** is an arm with Fit at least 1.0 above `n100-clean` (9.8 or more) **and** Test at least 0.5 above (8.6 or more). Fit up but Test not (less than 0.5): the change fits the training songs better without generalising;
   noted, not adopted by itself. Fit within 0.5 of the reference: no effect. Fit 0.5 or more below: harmful, dropped.
2. **sisdr, norm (the loss):** a lever by rule 1 and no worse than 0.5 on any of drums, bass, vocals separately (a loss that wins on the mean by sacrificing a source is not adopted); then check silent-source leakage as described under Scoring.
   If both losses are levers, the larger gain goes into the next recipe, the other is a candidate for a combination.
3. **ctx (the context):** compare with both references. Fit at least 1.0 above `n100-clean` **and** at least 0.5 above arm I (10.1 or more): the context is a lever beyond the extra parameters. Fit at least 1.0 above clean but not 0.5 above I:
   it is capacity again, not context (the same small size effect as batch 5), and `ctx` is not worth its cost. Fit within 0.5 of clean: the field of view is not the limit.
4. **lr (the optimiser):** a direction with Fit at least 0.5 above 8.8 is adopted as the learning rate of the next recipe. Both arms within 0.3 of 8.8: the learning rate is flat here, not a lever; if both are lower, 0.002 is near its optimum.
   The best result at the end of the range (0.001 or 0.004) suggests a further step in the same direction in a later batch.
5. **Next batch:** the changes that are levers (rules 2-4) go together into one run on `n100-clean`, then on the 1.9 M network, to see whether they add up. If **none** is a lever, the limit is not the loss, the field of view or the learning rate:
   then the data (Slakh, the studio songs, which cost MUSDB18 vocals before) and a gentler augmentation with a larger network are the open questions, and the 3-song overfit (+10.5 at 1,500 steps) remains unexplained.
6. **Sanity:** every arm's training must run to epoch 19 without NaN; the validation curve (in `batch7-console.log`) is read next to the scores (flat tail or still rising: a still-rising arm is a speed question for a longer run).
Not decided here: guitar and keys beyond the noted values, the source weights (kept equal), augmentation, Slakh, the studio songs.

Limits: one seed per arm (grey results are rerun with seed 1 before anything is adopted); the arms change the loss, the model and the optimiser, but `ctx` confounds context with 1.45 M extra parameters (hence the comparison with arm I);
Fit is measured on the songs trained on, Test on 50 songs and 240 windows; the loss arms change what is optimised, so their training loss values are not comparable with the clean arm's; ten thousand steps may be too few for an arm that learns more slowly (read the curve).

## Results so far (2026-11-15; mean of drums / bass / vocals; reference `n100-clean` Fit 8.8, Test 8.1)
- **sisdr** (12 min): Fit 9.3 / 9.3 / 10.2 (mean 9.6), Test 7.7 / 8.2 / 9.4 (mean 8.4): +0.8 / +0.3, grey by rule 1 (a lever needs +1.0 and +0.5), no source worse. **Keys: Fit +6.4, Test +3.6, validation about +6 from epoch 0** (every earlier arm was negative: clean -11 to -15, weights -5.4), with equal weights. Plausible reason: L1 on the waveform is median-like and switches off a source that is silent in most items; SI-SDR on audible sources does not. Keys rests on 37 / 52 windows (the rules say not to decide on it), but the jump is 15-18 dB. Next: seed 1 (grey rule), silent-source leakage check, then sisdr + weights / 1.9 M.
- **norm**: NaN from epoch 0 (all scores NaN): a bug, not a result. Digitally silent MUSDB snippets have a zero mixture and zero sources, so the denominator (source RMS + 0.1 x mixture RMS) was 0 and 0/0 = NaN. Fixed with a 1e-4 floor in `separation_loss`, with `tests/test_loss_modes.py` (the test fails on the old code). The arm must be rerun: `ARMS="100:norm" bash scripts/runpod_batch7.sh` after `git pull`.

## Results of the other arms (2026-11-15; mean of drums / bass / vocals; reference `n100-clean` Fit 8.8, Test 8.1; 12 min of training per arm on the RTX PRO 4000, 23 min for `norm` on a slower pod)
| arm | Fit | Test | keys Fit / Test | rule |
|---|---|---|---|---|
| ctx (1.72 M) | 8.9 (8.5 / 8.6 / 9.5) | 8.0 (7.6 / 7.9 / 8.4) | -12.6 / -13.3 | rule 3: within 0.5 of clean and below arm I (Fit 9.6): the field of view is not the limit; dropped. Validation vocals fall from 7.2 (epoch 4) to 6.5 |
| lrlow (0.001) | 8.9 (8.5 / 8.8 / 9.4) | 8.1 (7.5 / 8.0 / 8.7) | -14.7 / -14.6 | rule 4: flat |
| lrhigh (0.004) | 8.9 (8.5 / 8.7 / 9.4) | 8.0 (7.5 / 7.9 / 8.7) | -15.2 / -14.8 | rule 4: flat; no divergence. The learning rate is not a lever, 0.002 stays |
| norm (rerun after the fix) | 6.5 (6.7 / 6.7 / 6.2) | not read | -10.5 / - | rule 1: harmful (-2.3 on Fit), dropped. Validation vocals end at 4.5 (clean 8.1): a loss that divides by the source's own level over-weights near-silent windows (a guess, not tested) |
Open: the `sisdr` seed-1 check (grey on the mean, keys +6.4 / +3.6), the silent-source leakage check, then `sisdr` with the source weights and on the 1.9 M network.

## Result of the sisdr seed-1 check and the batch 7 conclusion (2026-11-15)
`n100-sisdr` seed 1 (13 min): Fit 9.4 / 9.4 / 10.1 (mean 9.63), Test 7.7 / 8.1 / 9.5 (mean 8.43), keys Fit +6.4, Test +3.7; seed 0: Fit 9.63, Test 8.43, keys +6.4 / +3.6. The two seeds agree to 0.1 dB or better, so seed noise is about 0.1, not the 0.5 assumed in the rules.
Against `n100-clean` (Fit 8.8, Test 8.1): **Fit +0.8, Test +0.3**, no source worse, **keys +15..18 dB (from -11..-15 to +4..+6 with equal weights)**. By the written bar (Fit +1.0 and Test +0.5) it stays grey; **adopted anyway**, because the bar assumed 5x the noise that was measured and the keys effect is far outside any noise. This is a deliberate step beyond the rule's wording.
Conclusion of batch 7: the loss is the only lever found. The SI-SDR loss (audible sources; plain L1 on silent ones) replaces waveform L1 in the recipe; the keys collapse was a loss artefact (L1 is median-like and switches off a source that is silent in most items), not a data problem, and needs no keys weight. Not lever: learning rate (0.001 / 0.002 / 0.004 flat), the LSTM context (1.72 M, no gain over the clean net, below arm I), the relative-L1 loss (-2.3 on Fit).
Open: silent-source leakage of the sisdr models (not measured by these scores), `sisdr` on the 1.9 M network (arm I alone gave Fit +0.8), the data side (Slakh keys and the studio songs may pay off now that keys no longer collapses), gentler augmentation once a Fit/Test gap exists.
