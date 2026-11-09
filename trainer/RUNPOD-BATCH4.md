# Pod batch 4: can Slakh keep its keys gain without the vocal loss?

Batch 3 (`RUNPOD-BATCH3.md`): Slakh at weight 0.1 (E) and 0.3 (F) raised studio keys by about 6 dB (one song) but lowered MUSDB18 vocals
by about 1.6 dB at both weights, both seeds. Same loss at both weights means it is probably not simple dilution of MUSDB18. Two guesses,
one arm each. Nothing else changes; D (batch 2) and E (batch 3) are the comparison and are **not rerun** (same seeds 0 and 1).

| arm | change from E (Slakh 0.1) | guess it tests |
|---|---|---|
| G `g-vocal-always` | `vocal_mix.p` 0.7 -> 1.0 | In 30 % of Slakh items the vocals target is silent, which teaches "vocals are rarely there"; with p 1.0 every Slakh item has vocals |
| H `h-slakh03` | Slakh weight 0.1 -> 0.03 (about 2 % of the items), `vocal_mix.p` 0.7 | any Slakh at all disturbs vocals (a threshold, not a dose); a tiny dose may keep the keys gain |

Two seeds each, 10,000 steps, 4 runs, about 25 min each (batch 3 measured 1.1 min per epoch). Cost: about $1.5 of GPU time plus the volume
(about $0.35 per day). The network volume with Slakh must still exist; the studio split is already the swapped one on it.

Configs `configs/runpod-b4-g-vocal-always.yaml` and `runpod-b4-h-slakh03.yaml` (copies of batch 3's arm E with only the one setting changed), runner
`scripts/runpod_batch4.sh`.

## Steps
1. Start a GPU pod on the existing network volume (same data centre); `nvidia-smi` must list a GPU; activate the Python 3.11 venv (`/workspace/venv`).
2. `cd /workspace/mo-music && git pull && cd trainer` (the branch must be merged first), then
   `HOURS_PER_RUN=2 AUTO_STOP=1 bash scripts/runpod_batch4.sh 2>&1 | tee /workspace/batch4.log`; start it with `nohup` if the terminal may drop.
3. About 1.7 h later the bundle says 12 files; the pod stops itself. Download `/workspace/batch4-results.tar.gz` (also possible from any pod on the volume).
4. Summarise with `python scripts/summarize_run.py <folder>/b4-*/run.json --last 3 --compact`, apply the rules below, then delete the volume unless an arm is adopted.

## Decision rules (fixed before the run)
Numbers are means over the two seeds of the last 3 epochs, as in batches 1-3; D is batch 2's control. Seed noise is about 1 dB.

An arm is **good** when all of these hold:
1. MUSDB18 vocals is not more than 1.0 dB below D (D is 5.5, so at least 4.5; E was 3.9) and each of its two seeds is above the E mean (3.9).
2. MUSDB18 and studio drums, bass and guitar are each not more than 0.5 dB below D.
3. Studio keys is at least 3 dB above D (D 6.4, so at least 9.4) on both seeds; MUSDB18 keys is not more than 0.5 dB below D.

Outcomes:
- **One arm is good:** adopt it as the recipe (Slakh stays in the training mix at that setting). The piano class (Slakh gives exact piano) then becomes worth building, with a
  check by ear on the real keys items, because the keys evidence is one studio song.
- **Both good:** the vocal loss was avoidable in two ways; take H (smaller weight) unless G's worst loss over the sources of rules 1 and 2 is more than 0.5 dB smaller.
- **Vocals recover but studio keys falls under +3 dB** (the gain needs what hurts vocals): drop Slakh as training data, no more tuning.
- **Vocals do not recover in either arm** (rule 1 fails twice): drop Slakh as training data, no more tuning. The cause is then not one of the two guesses; do not guess a third without a new reason.
- Either way the network volume is deleted afterwards (unless an arm is adopted and a longer run follows).

Limits: studio keys is one song (Just Me Alone), MUSDB18 keys 10 items; a keys gain on studio and none on MUSDB18 may only mean that Slakh's
synthesised keys resemble the studio instruments. Seed noise on single studio numbers is 2 to 4 dB, hence "both seeds" in rule 3.
This test does not say whether the gain survives on other real recordings; only listening can.

## Results (2026-11-10; mean of the last 3 epochs, SI-SDR improvement in dB; drums/bass/vocals/guitar/keys)
| run | MUSDB18 | studio (swapped split) |
|---|---|---|
| G vocal p 1.0, seed 0 | 5.6 / 6.1 / 4.3 / 6.4 / 2.8 | 8.2 / 3.1 / 9.1 / 3.1 / 7.9 |
| G vocal p 1.0, seed 1 | 5.5 / 6.0 / 4.2 / 6.4 / 4.6 | 7.4 / 2.4 / 8.4 / 2.6 / 9.4 |
| H weight 0.03, seed 0 | 5.8 / 5.6 / 4.0 / 6.3 / 5.2 | 7.6 / 3.4 / 8.7 / 3.5 / 7.1 |
| H weight 0.03, seed 1 | 5.9 / 5.5 / 4.5 / 6.0 / 4.6 | 8.0 / 2.6 / 9.7 / 3.2 / 6.7 |

Means: G MUSDB18 5.55 / 6.05 / **4.25** / 6.4 / **3.7**, studio 7.8 / 2.75 / 8.75 / **2.85** / 8.65; H MUSDB18 5.85 / **5.55** / **4.25** / 6.15 / 4.9,
studio 7.8 / 3.0 / 9.2 / 3.35 / **6.9**. (D: MUSDB18 5.7 / 6.1 / 5.5 / 6.6 / 4.5, studio 7.75 / 3.2 / 9.45 / 3.45 / 6.4; E: vocals 3.9, studio keys 12.8.)

**Outcome by the rules: neither arm is good, Slakh is dropped as training data, no further tuning.**
- Rule 1 (MUSDB18 vocals at least 4.5): fails for both, 4.25 each (D 5.5, E 3.9). The two seeds are above E's mean, but the recovery is 0.35 dB, within noise.
- Rule 2: G studio guitar is 0.6 below D, H MUSDB18 bass 0.55 below D.
- Rule 3 (studio keys at least 9.4 on both seeds): fails for both. G 7.9 / 9.4, H 7.1 / 6.7. The +6 dB of batch 3 did not come back: G +2.2 over D, H +0.5. MUSDB18 keys of G is 0.8 below D.
- Reading: the vocal loss is about 1.25 dB at weights 0.03, 0.1 and 0.3 and with vocals in every Slakh item, so neither guess explains it and it is not a dose. The studio keys gain shrank from +6 to +2 dB with vocals in every item, but one song with a seed spread of 3 dB cannot separate that from noise.
- Caveat: D was run in batch 2, E/F in batch 3 and G/H in batch 4, on different pods and cards; D was not rerun on the same hardware.
- Speed: about 2 min per epoch on an RTX PRO 4000 Blackwell (1.1 min on the earlier card); the first epoch of a run was very slow because every loader worker scanned Slakh over the network volume (fixed by `.mixer-index.json`, branch `slakh-index-cache`).
