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

## Reading
- The reason for dropping Slakh was a vocal loss of about 1.5 dB at every weight. Re-measured it is about 0.2-0.9 dB, grows with the weight, and is smallest at weight 0.03 (H) and with vocals mixed in (G).
- The keys gain on the one studio organ song is real in the sense that it is consistent across seeds, and it also grows with the weight. Whether it carries over to keys other than an organ is **not** shown:
  Slakh contains organ timbres, and the studio test has one organ.
- No weight is good on both counts: E/F have the big keys gain and the 0.8-0.9 dB vocal cost, H has almost no vocal cost and a small keys gain (+1.4).
- Not yet decided: whether Slakh comes back. The batch 3/4 verdict was made by fixed rules and is not overturned by this note; it shows that the rules' vocal number was inflated by the measuring, and that the keys evidence is
  one song. A batch 6 would need a keys test that is not one organ (for example held-out Slakh tracks, or a second studio keys song).

Limits: final weights, not the last-3-epoch mean; two seeds per arm; MUSDB18 vocals only (drums, bass and guitar on MUSDB18 not rescored); windows from the same songs overlap; D came from another pod than E-H (hardware does not change the numbers here, the scoring ran locally on CPU).
