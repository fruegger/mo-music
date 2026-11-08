# Pod batch 2: do the listening labels help keys?

Batch 1 (see `RUNPOD-BATCH1.md`) found that the keys weight (`present_weights: {keys: 4}`) fixes the keys collapse. Batch 2 asks whether
the listening labels (`labels/stems.csv`, `labels/segments.csv`) add anything on top, two seeds each, 10,000 steps, same recipe.

| arm | what |
|---|---|
| D `d-control` | batch 1's arm B (keys x4) on the **swapped** studio split, no listening labels |
| C `c-labels` | D + `stem_labels`, `solo_windows`, `p_solo: 0.25` (exact keys targets for 4 whole stems, solo windows in 14 songs) |

The studio split is swapped (train: Down Lizard Lane, Shimmering Star, Summer Smiled; test: In My Belief, Just Me Alone), so batch 1's
studio numbers are **not** comparable; MUSDB18 numbers are (D should match batch 1's B within noise on MUSDB18: a sanity check).
`runpod_batch2.sh` swaps the folders on the pod itself (nothing to upload). Caveat: In My Belief's `keys` is unknown (a piano +
vibraphone + xylophone mixture), so studio keys is scored on Just Me Alone (organ) only: noisy. MUSDB18 keys is 10 test items: also noisy.

## On the reused pod
Batch 1's pod was stopped, not removed: `/workspace` (MUSDB18_prepared, studio_prepared, the batch 1 runs) is kept; the container disk
(the installed code) is not, `runpod_batch2.sh` reinstalls it (about a minute). The code must be on GitHub `main` first (merge + push `pod-batch-2`).
```
cd /workspace/mo-music && git pull && cd trainer
HOURS_PER_RUN=2 AUTO_STOP=1 bash scripts/runpod_batch2.sh 2>&1 | tee /workspace/batch2.log
```
Cost: like batch 1 (4 runs x 10,000 steps, data-bound at about 0.13 s/step on the RTX 4000 Ada). Results: `/workspace/batch2-results.tar.gz`
(12 files; the script prints the file count: it should say 12, not 0). The pod stops itself at the end. Summarise with
`python scripts/summarize_run.py <folder>/b2-*/run.json --last 3 --compact`.

## Decision rules (fixed before the run)
Compare C with D, seed by seed and as a mean of the two seeds. Seed noise is about 1 dB on keys and up to 2-4 dB on single studio numbers
(batch 1), so nothing under 1 dB counts.
- **Labels help:** C's MUSDB18 keys is at least 1 dB above D's on both seeds, studio keys is not more than 0.5 dB lower, and
  drums/bass/vocals/guitar stay within 0.5 dB of D (mean over seeds). Then the labels become part of the recipe and the piano class
  (more labelling, `INSTRUMENT_SOURCE` entry for a seventh source) is worth building.
- **No difference** (keys within 1 dB): 26 stems and 215 s of solo windows are too little to matter at this weight. Do not label more
  for the separator; the piano class waits for more exact piano data (Slakh at a small weight).
- **Labels hurt** (keys more than 1 dB lower, or another source more than 0.5 dB lower): drop them; investigate `p_solo` only if a hint
  shows where the harm comes from.
- **Sanity:** D's MUSDB18 numbers should match batch 1's arm B within about 0.5-1 dB; if not, something else changed and the batch is read with care.


## Results (2026-11-08; mean of the last 3 epochs, SI-SDR improvement in dB; drums/bass/vocals/guitar/keys)
| run | MUSDB18 | studio (swapped split) |
|---|---|---|
| C labels, seed 0 | 5.6 / 5.8 / 5.3 / 6.6 / 4.2 | 8.0 / 3.0 / 10.4 / 3.3 / 4.2 |
| C labels, seed 1 | 5.5 / 5.8 / 4.7 / 6.2 / 4.9 | 7.5 / 2.8 / 8.1 / 1.7 / 7.5 |
| D control, seed 0 | 6.1 / 6.3 / 5.7 / 6.8 / 4.5 | 7.7 / 3.0 / 10.5 / 3.2 / 4.7 |
| D control, seed 1 | 5.3 / 5.9 / 5.3 / 6.4 / 4.5 | 7.8 / 3.4 / 8.4 / 3.7 / 8.1 |

**Outcome by the rules: no difference.** C minus D on keys: MUSDB18 -0.3 / +0.4 dB (seeds 0/1), studio -0.5 / -0.6 dB; the rule needed
+1 dB on both seeds. The other sources are 0.15-0.5 dB lower for C on MUSDB18 (vocals -0.5) and studio guitar about 1 dB lower (noise: same
arm differs by 2 dB between seeds), so the labels also do not hurt keys. Sanity check passed: D's MUSDB18 numbers are within ~0.75 dB of
batch 1's arm B. Caveats: studio keys is scored on one song (Just Me Alone; D alone gives 4.7 and 8.1), MUSDB18 keys on 10 items, and
the labels are small (4 whole stems, 14 songs with solo windows). Consequence: do not label more for the separator; the piano class
waits for more exact piano data (Slakh2100 at a small weight, needs the full 104 GB download, so a larger volume).
