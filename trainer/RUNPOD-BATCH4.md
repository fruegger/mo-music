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
