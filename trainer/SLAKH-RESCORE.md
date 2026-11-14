# Slakh rescored (2026-11-14): is dropping it still justified?

Batches 3 and 4 dropped Slakh because MUSDB18 vocals fell by 1.3-1.7 dB at every weight while the studio keys gain (+6 dB) rested on one song and a 3 dB seed spread.
This note re-scores the finished checkpoints of D (batch 2), E/F (batch 3: Slakh weight 0.1 / 0.3), G (batch 4: vocals mixed into every Slakh item) and H (batch 4: weight 0.03),
two seeds each, on many more windows, with the same windows for every run (paired). No training, no pod: `scripts/studio_window_scores.py` and `scripts/vocal_leakage.py`
on the `model.pt` of each run (final weights, not the mean of the last 3 epochs the batch rules used).

## What the studio test really is
The swapped split tests on In My Belief and Just Me Alone. In My Belief has no vocal stems (the demo take is excluded), its keys are unlabelled and it has almost no
audible guitar window, so **every studio vocals, keys and guitar score is Just Me Alone**; its keys are one organ stem at about -49 dB.

## Studio (200 clean 10 s windows, SI-SDR improvement in dB; per seed in brackets)
| arm | keys (43 windows) | vs D | vocals (86 windows) | vs D | drums vs D |
|---|---|---|---|---|---|
| D | 4.8 (2.9 / 6.8) | - | 10.4 | - | - |
| E 0.1 | **13.8** (13.9 / 13.8) | +9.0 | 9.8 | -0.6 | -0.2 |
| F 0.3 | **12.1** (12.5 / 11.6) | +7.2 | 9.7 | -0.7 | -0.4 |
| G | 9.4 (7.7 / 11.1) | +4.5 | 9.7 | -0.6 | -0.4 |
| H 0.03 | 6.3 (6.0 / 6.5) | +1.4 | 9.9 | -0.5 | -0.3 |

The keys gain is larger and steadier than batch 3 reported (both E seeds and both F seeds above 11.5, D's seeds 2.9 and 6.8) and grows with the weight. The bootstrap intervals
printed by the script ([+8.0, +9.8] for E) only cover the sampling of windows, and the windows overlap heavily on one song: trust the per-seed values.

## MUSDB18 vocals (vocal SI-SDR improvement, windows with audible vocals)
| arm | 10 s, 69 windows | per seed | 3 s, 184 windows | per seed | 3 s windows below 0 dB |
|---|---|---|---|---|---|
| D | 6.19 | 6.21 / 6.17 | 6.24 | 6.19 / 6.29 | 7.6% |
| E 0.1 | 5.42 (-0.8) | 6.06 / 4.79 | 5.44 (-0.8) | 5.65 / 5.24 | 12.0% |
| F 0.3 | 5.63 (-0.6) | 5.31 / 5.95 | 5.34 (-0.9) | 5.19 / 5.49 | 12.5% |
| G | 5.97 (-0.2) | 6.17 / 5.76 | 5.86 (-0.4) | 5.85 / 5.87 | 11.4% |
| H 0.03 | 5.99 (-0.2) | 5.75 / 6.24 | 6.02 (-0.2) | 5.87 / 6.18 | 8.2% |

- The vocal cost is **0.2-0.9 dB, not 1.3-1.7**, and it is the same at 3 s and 10 s, so the window length is not what separates this from the batch numbers (D 5.5, E/F about 3.9, G/H 4.25).
  Not established why: the batch numbers were the mean of the last 3 epochs on 48 validation snippets (about one window per song), these are the final weights on 184 windows.
  The paired comparison on identical windows here is the cleaner one.
- The cost now looks **dose-like** (H -0.2, E -0.8, F -0.9), unlike the flat 1.25 dB of the batch rules.
- Where the vocal energy goes (3 s): the vocal output's interference from drums, bass and other is the same in every arm (about 5.5%); the share of the output along the true vocals drops
  from 56.9% (D) to 54.0-54.2% (E, F) and the share unexplained rises by about 3 points. So Slakh does not put other instruments into the vocal output, it makes the vocal output slightly less faithful.
  The projection is crude ("unexplained" lumps distortion and overlap of the true sources).

## MUSDB18 keys and guitar (500 clean 6 s windows over the 50 test songs, stems at their own level)
Guitar and keys are scored on the test songs whose `other` stem the hand tags say is a single source (`labels/musdb18-other-tags.csv` with `promote_single`):
15 guitar songs (148 windows) and 11 keys songs (105 windows). Per seed in brackets; the interval resamples songs, then windows (1,000 draws).

| arm | guitar | vs D | keys | vs D |
|---|---|---|---|---|
| D | 1.3 (1.2 / 1.4) | - | 1.2 (1.4 / 1.0) | - |
| E 0.1 | 1.1 (1.2 / 0.9) | -0.3 | 0.6 (1.0 / 0.2) | -0.6 [-1.6, +0.7] |
| F 0.3 | 0.9 (0.9 / 0.9) | -0.4 | 1.8 (1.8 / 1.9) | +0.6 [-0.5, +2.1] |
| G | 1.2 (1.2 / 1.3) | -0.1 | -0.2 (-0.5 / +0.1) | **-1.4** [-2.1, -0.8] |
| H 0.03 | 1.2 (1.1 / 1.3) | -0.1 | 2.0 (2.7 / 1.4) | +0.8 [0.0, +2.1] |

Other sources on the same windows, vs D: drums -0.3..-0.1, bass -0.3..-0.1, vocals -0.7 (E), -0.7 (F), -0.3 (G), -0.1 (H).
- **The studio keys gain does not show up on MUSDB18.** No arm gains on guitar; on keys E is below D, F and H are +0.6 / +0.8 with intervals that include 0, G is clearly worse.
  The +7..+9 dB on the studio organ is therefore specific to that song's timbre (Slakh has organs), not evidence that Slakh helps keys in general.
- Guitar and keys on MUSDB18 are weak in absolute terms: about +1 dB at natural levels, against +5.8 for drums. They are the real gap, not just the distance to Demucs on drums, bass and vocals.
- **The batch tables' MUSDB18 guitar and keys columns are a small, lucky sample.** The in-training validation uses 48 snippets (about 13 guitar and 10 keys windows) with random stem gains of -12 to +6 dB (`val_mix` only switches off effects, tempo and pitch).
  Scoring exactly those 48 windows reproduces the tables (D: guitar 5.9, keys 4.4; table 6.8 / 4.5), but 300 windows of the same kind give guitar 3.6, keys 2.1, and at natural levels 2.5 / 1.9 (500 windows of 6 s: 1.3 / 1.2).
  Song-to-song spread is large (guitar interval [-0.7, +6.4]). Drums, bass and vocals are far steadier (intervals about +-1 dB) and the batch 1-5 decisions rested mainly on those; but every keys or guitar claim from the in-training validation (the keys x4 fix, batch 3/4 keys thresholds) carries an error of several dB.

## Reading
- The reason for dropping Slakh was a vocal loss of about 1.5 dB at every weight. Re-measured it is about 0.2-0.9 dB, grows with the weight, and is smallest at weight 0.03 (H) and with vocals mixed in (G).
- The keys gain on the one studio organ song is consistent across seeds and grows with the weight, but it does **not** carry over to MUSDB18 keys (11 songs, see above): it looks specific to the organ timbre.
- No weight is good on both counts: E/F have the big keys gain and the 0.8-0.9 dB vocal cost, H has almost no vocal cost and a small keys gain (+1.4).
- Not yet decided: whether Slakh comes back. The batch 3/4 verdict was made by fixed rules and is not overturned by this note; it shows that the rules' vocal number was inflated by the measuring, and that the keys evidence is
  one song. A batch 6 would need a keys test that is not one organ (for example held-out Slakh tracks, or a second studio keys song).

Limits: final weights, not the last-3-epoch mean; two seeds per arm; MUSDB18 vocals only (drums, bass and guitar on MUSDB18 not rescored); windows from the same songs overlap; D came from another pod than E-H (hardware does not change the numbers here, the scoring ran locally on CPU).
