# Why is our separator 6-8 dB behind Demucs? Two cheap diagnostics (2026-11-14)

Batch 5 showed that size (0.27 M to 17.5 M parameters) and training length (3x) move MUSDB18 drums/bass/vocals by at most 0.8 dB. Two local checks, no pod, with `scripts/oracle_ceiling.py`
(240 clean 6 s windows each, stems at their own level, hand tags with `promote_single` for guitar and keys; SI-SDR improvement over the mixture in dB):

1. **Oracle ceiling:** the ideal ratio mask |S_k|^p / sum_j |S_j|^p computed from the true stems and applied to the mixture's STFT (n_fft 2048, hop 1024, the mixture's phase), as our separator would. It needs the true stems, so no
   network reaches it; it is a ceiling for "a mask on this STFT with the mixture's phase".
2. **Train versus test:** the same runs scored on the songs they were trained on (MUSDB18 train, 100 songs) and on test (50 songs).

| | drums | bass | vocals | guitar | keys |
|---|---|---|---|---|---|
| oracle p=1 / p=2, test | 12.4 / 13.2 | 12.1 / 13.1 | 15.4 / 16.3 | 10.6 / 11.5 | 14.2 / 15.3 |
| oracle p=1 / p=2, train | 12.9 / 13.9 | 12.0 / 12.9 | 14.3 / 15.1 | 11.3 / 12.3 | 11.5 / 12.5 |
| D (0.27 M), test | 6.1 | 6.0 | 6.3 | 1.9 | 1.6 |
| I `i-wide` (1.9 M), test | 6.5 | 6.4 | 7.0 | 3.0 | 1.9 |
| J `j-long` (0.27 M, 3x steps), test | 6.7 | 6.8 | 7.8 | 2.6 | -0.6 |
| L `l-large` (17.5 M), test | 6.4 | 6.4 | 6.8 | 2.0 | 2.2 |
| D, train | 6.4 | 6.7 | 6.3 | 1.7 | 1.3 |
| I, train | 6.9 | 7.0 | 6.5 | 2.3 | 0.6 |
| J, train | 7.3 | 7.5 | 7.0 | 2.3 | 0.1 |
| L, train | 6.8 | 6.8 | 6.0 | 1.9 | 0.2 |
| windows (test) | 220 | 203 | 193 | 69 | 52 |
| windows (train) | 219 | 210 | 192 | 111 | 37 |

Guitar and keys rest on few windows (and few songs): rough numbers only. Drums, bass and vocals are steady to about +-1 dB.

## Reading
- **The mask architecture is not the ceiling.** The oracle reaches +12..+16 dB on drums, bass and vocals, about twice what any run scores and about where Demucs sits (+13..+15 in `reference_compare.py`; different windows,
  so only the order of magnitude counts). A mask on this spectrogram has room above 6 dB.
- **Not overfitting.** Every run scores nearly the same on its training songs as on unseen ones (D 6.4 / 6.7 / 6.3 on train against 6.1 / 6.0 / 6.3 on test; J's gap is under 0.7 dB; J's vocals are better on test).
  The earlier reading of batch 5 (plateau from fitting about 86 songs) is not supported: **the runs underfit**, they do not even fit the songs they were trained on.
- Hence more data is probably not the first lever, and capacity is not either (17.5 M does not beat 1.9 M, also on train). What is left is how well the network *learns* the task: learning rate (L ran at 0.002, untuned), what the
  network sees (the magnitude of the mixture only), how hard training is made (effects, random stem gains of -12..+6 dB, 3 s snippets), the loss weighting (vocals x4, keys x4, silent targets), EMA and warm-up.
- The loss is already L1 on the output waveforms (through the inverse STFT), so a "time-domain loss" is not a new lever.
- Not measured: how the training loss compares with the oracle, and whether a clean, un-augmented run fits better. The train scores here are on clean windows, the training itself used effects and random gains.

## Next (proposals, not run)
1. **Tiny overfit test** (local): train on 2-3 songs without augmentation until the loss stops falling and score on those songs. If the model cannot approach the oracle there, the problem is optimisation or the network, not data or size.
2. **Recipe batch on the pod** with `i-wide`: learning rate (0.0005, 0.001, 0.004), 10 s snippets, and a run with less augmentation; judged on drums, bass and vocals first.
3. Slakh or more data only after these.

Limits: one seed per run; 240 windows per cell, about 10 s of the test songs per window position; oracle masks use the exact stems, not the tag-based promotion, for the unknown group (the grouped sum is one source in the denominator).
