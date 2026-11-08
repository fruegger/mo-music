# Pod batch 1: does longer training hold up, and is the keys output fixable?

Two arms x two seeds of the six-source separator (`configs/runpod-b1-a-promote.yaml`, `configs/runpod-b1-b-keys4.yaml`),
10,000 steps each, run one after the other by `scripts/runpod_batch1.sh`.

| arm | what |
|---|---|
| A `a-promote` | the laptop long recipe: hand tags + single-source promotion, vocals weight 4, warm-up/cosine/clip/EMA (control) |
| B `b-keys4` | A with `present_weights: {keys: 4}` (keys error x4 where keys are audible) |

Differences from the laptop long run (5,000 steps, +5.8 / +5.2 / +4.0 / +4.7 dB drums/bass/vocals/guitar on MUSDB18; keys -2.4 MUSDB18, -5.1 studio):
twice the steps, and **no tempo/pitch augmentation** (about 20x slower per item; it made the laptop run data-bound at 8 s/step). So
compare A and B with each other, and with the laptop run only roughly. The listening labels are NOT used (batch 2), so the
batch also stays a clean baseline. Studio split as before (test = Shimmering Star piano + Just Me Alone organ).

## Cost (an estimate, check it with the benchmark first)
Earlier pod runs without tempo/pitch took 0.47-0.56 s/step (data-bound). 4 x 10,000 steps = about 5.6 h at $0.69/h = about $4, plus
~0.3 h of setup. Hard cap: `HOURS_PER_RUN` (default 2) per run = at most 8 h = $5.5, and the pod stops itself after the last run
(`AUTO_STOP=1`). If `scripts/bench_pipeline.py --config configs/runpod-b1-a-promote.yaml` on the pod says more than ~0.7 s/step,
stop and rethink (fewer steps or one seed) before paying for hours.

## Before renting the pod (needs the user)
1. The code must be on GitHub `main` (the pod clones it): merge and push `keys-diagnosis`, `piano-labels` and `pod-batch-1`
   (present_weights, tag/label support, configs, scripts). Nothing is pushed yet.
2. Zip `C:\workarea\studio_prepared` (579 MB; not in the repo) and upload it to `/workspace/studio_prepared` on the pod
   (`runpodctl send` / `receive`, or scp). Rights: own material.
3. Pod: 8+ vCPU, 30 GB volume (MUSDB18 only, no Slakh), PyTorch template with Python 3.11+. Many vCPUs per dollar.

## On the pod
```
cd /workspace && git clone https://github.com/fruegger/mo-music.git && cd mo-music/trainer
bash scripts/runpod_setup.sh data-musdb
python scripts/bench_pipeline.py --config configs/runpod-b1-a-promote.yaml     # s/step: decide here
HOURS_PER_RUN=2 AUTO_STOP=1 bash scripts/runpod_batch1.sh 2>&1 | tee /workspace/batch1.log
```
**Results are not lost if you cannot download right away.** The runs are written to `/workspace` (the volume disk) and bundled into
`/workspace/batch1-results.tar.gz`. With `AUTO_STOP=1` the pod *stops* itself after the last run: compute billing ends, the volume
is kept (a stopped pod's volume costs about $0.20 per GB and month, so roughly $6 a month for 30 GB, i.e. cents per day). To
get the files later, start the pod again (CPU-only is enough if no GPU is free) and download the bundle through the web terminal or
`runpodctl send`; then **remove the pod** so the storage fee ends. Do not use `AUTO_TERMINATE` unless the files are downloaded:
it deletes the pod and its volume. Summarise with
`python scripts/summarize_run.py runs/b1-*/run.json --last 3 --compact`.

## Decision rules (fixed before the run)
- **Keys:** if A reproduces the collapse (studio keys < 0 dB) and B keeps it above 0 dB on both seeds, with drums/bass/vocals/guitar
  within 0.5 dB of A: use B's setting from now on. If A does not collapse here, the weight is not needed: stay with A.
  If B also fails: the keys problem is data (piano class, Slakh), not the loss; do not tune the weight further.
- **Longer training:** the MUSDB18 drums/bass/vocals/guitar means should be at least the laptop numbers minus 0.5 dB (no augmentation here).
  Clearly higher means 10,000 steps is worth it; equal means the limit is data.
- **Seeds:** differences under about 0.5-1 dB between arms are noise.
- Batch 2 (not before these answers): add the listening labels (`separation.stem_labels`, `separation.solo_windows`), then Slakh at a small weight.


## Results (2026-11-08; mean of the last 3 epochs, SI-SDR improvement in dB; MUSDB18 drums/bass/vocals/guitar/keys, then studio)
| run | MUSDB18 | studio |
|---|---|---|
| A seed 0 | 6.0 / 5.9 / 5.6 / 6.9 / **-7.9** | 7.4 / 4.5 / 8.4 / 5.4 / **-8.1** |
| A seed 1 | 6.0 / 6.1 / 3.4 / 5.8 / **-4.7** | 6.9 / 4.6 / 7.9 / 5.7 / **-6.9** |
| B seed 0 | 5.8 / 6.5 / 5.4 / 6.8 / **+3.7** | 7.2 / 5.2 / 9.5 / 5.8 / **+4.7** |
| B seed 1 | 6.0 / 5.9 / 4.1 / 6.0 / **+3.8** | 6.7 / 4.7 / 5.8 / 5.3 / **+4.8** |

All three keys rules are met: A collapses (studio keys below -6.9 dB), B keeps keys at about +4.7 dB (studio) and +3.7 dB (MUSDB18) on both
seeds, and B is within 0.5 dB of A on drums/bass/vocals/guitar (largest gap studio vocals -0.5, well inside the seed spread). **B's setting
(`present_weights: {keys: 4}`) is adopted.** 10,000 steps without tempo/pitch also beats the laptop long run (5,000 steps) on drums/bass/guitar
(+6.0/+5.9-6.5/+6.0-6.9 vs +5.8/+5.2/+4.7); vocals is still rising at epoch 19 (not converged). Note for the bundle: `tar` does not expand
wildcards when creating an archive, the first bundle command made an empty file (fixed with `find`).
