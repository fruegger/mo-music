# trainer

Trains audio-analysis models on snippets from [`../mixer`](../mixer). The
`task` in the config picks what is learned:

| task | config | model | status |
|---|---|---|---|
| `instruments` | `configs/default.yaml` | log-mel CNN, multi-label (which classes are in the mix) | runs end to end on Slakh |
| `separation` | `configs/separation.yaml` (Slakh2100 + MUSDB18), `configs/separation-cpu.yaml` (Slakh only, CPU-sized) | spectrogram U-Net predicting one mask per source | runs end to end; see the first run below |

```
pip install -e ../mixer[effects]
pip install -e .[dev]
trainer --config configs/separation.yaml
```

## Layout

| module | role |
|---|---|
| `data.py` | `SnippetDataset`: a torch `Dataset` that calls `mixer.render_snippet` on the fly; items are dicts of tensors per task; fixed seeds for validation |
| `sources.py` | groups the mixer's stems into separation sources (`musdb4` or `classes`) with a mask for unknown targets |
| `model.py` | instrument-recognition CNN |
| `separation.py` | `SeparationNet` (masks from magnitudes, exportable), `Separator` (STFT, masks, inverse STFT), loss, SI-SDR, evaluation |
| `train.py` | `InstrumentTask` / `SeparationTask`, the `fit` loop, ONNX export, `run.json` |

## Separation targets

The mixer returns each stem as it is heard in the mix, so the targets add up to
the mix. `separation.source_set` chooses the sources:

- `musdb4`: drums, bass, vocals, other. Every stem has a place (an unknown
  mixture such as MUSDB18's `other` goes to `other`), so the sources sum to the
  mix and all targets are known. Slakh has no vocals: that source is silent.
- `classes`: one source per instrument class. A stem without a class belongs
  to no source, so while one is audible the classes it could contain are
  masked out of the loss (`sources_mask`), as in the instrument task.

Slakh2100 is mono and MUSDB18 stereo; the dataset always serves two channels
(a mono track is duplicated).

The loss is L1 on the waveforms, averaged over the sources with a known target;
a silent source is a valid target. Evaluation reports SI-SDR per source over the
examples where that source is audible, and the improvement over using the
mixture as the estimate.

## Exports

`model.onnx` plus a JSON next to it (`model.json` for separation).
Instruments: audio in, class logits out. Separation: only the mask network is
exported (magnitude spectrogram `(batch, 2, n_fft/2, frames)` in, masks
`(batch, sources, 2, n_fft/2, frames)` out; `frames` a multiple of
`2^depth`). The STFT, the mask multiplication and the inverse STFT are left to
the runtime, and the JSON lists the sources and STFT parameters. Both exports
are tested against PyTorch with onnxruntime. The exporter needs `onnxscript`.

## Contract with the mixer

- training: seed = epoch-dependent, so each epoch sees new mixes
- validation: fixed seeds (`val_seed_base + index`), so scores are comparable
- validation mix: the config's `mix`, with the optional `val_mix` entries on top. `val_mix: {effects: false}`
  trains with effects (augmentation) but scores on clean mixes, which is what real recordings are; scores
  with and without it are not comparable
  between runs
- each run writes `run.json` with the dataset licences from `sources.log`; a
  run that used MUSDB18 is marked `non_commercial`

## Training on MUSDB18 as well

`configs/separation.yaml` lists Slakh2100 and `musdb18-prepared`. MUSDB18 has to
be decoded once first (it is 150 five-stream AAC files; decoding a track takes
seconds, too slow to repeat in training):

```
mixer prepare-musdb --src C:/workarea/MUSDB18 --dst C:/workarea/MUSDB18_prepared
```

(`MUSDB18-prepared` in `repo/prototype/sources.log` points at that folder.) Each
dataset has a `weight`, its share of the training batches. MUSDB18 has no
validation split, so its `test` split is used for validation. Validation is
scored per dataset (`history[].validation.<name>`) and never averaged across
them: Slakh2100 (mono, synthesised, no vocals) and MUSDB18 (stereo, real
recordings, with vocals) are very different. Vocals only get a target from
MUSDB18; Slakh snippets have a silent vocals target.

Anything trained on MUSDB18 is non-commercial: `run.json` says
`non_commercial: true`, and the licence string of every dataset used.

## First separation run (CPU)

`configs/separation-cpu.yaml`: 3-second snippets with effects (no tempo/pitch),
`musdb4` sources, a 0.27M-parameter separator (n_fft 2048, hop 1024), batch 8,
150 steps per epoch. Trained on Slakh2100 train, scored on 48 fixed validation
snippets. It ran on an 8-core CPU without a GPU and was stopped by the system
twice for low memory, so only 4 of the planned 10 epochs (600 steps) finished.

SI-SDR improvement over using the mixture as the estimate (dB):

| epoch | drums | bass | other | mean |
|---|---|---|---|---|
| 0 | +4.0 | +6.8 | -1.5 | +3.1 |
| 1 | +10.1 | +8.4 | -0.5 | +6.0 |
| 2 | +9.6 | +7.7 | -1.1 | +5.4 |
| 3 | +10.6 | +8.0 | -1.0 | +5.9 |

Vocals are not scored (Slakh has none). Bass is still at -3 dB SI-SDR
(improved, not separated); `other` is not better than the mixture. The run had
levelled off at about +6 dB, and 48 validation snippets are noisy, so read the
epochs 1-3 differences as noise. This shows that the pipeline learns on real
Slakh material; it is not a useful separator. That needs a GPU, more steps,
a larger model, and MUSDB18 in the mix.

## Resuming and long runs

`trainer --config ... --resume runs/<run>/checkpoint.pt` continues exactly
(weights, optimiser, epoch count, results so far). A checkpoint and
`run.json` (`finished: false` until the end) are written after every epoch, so a
killed run loses at most one epoch. A bare `model.pt` also resumes, with
`--start-epoch N` for the epochs it stands for; the optimiser then starts fresh.

On Windows the DataLoader workers pickle the dataset, so a dataset made with
`from_dataset` lists its own tracks inside each worker (about a minute for
Slakh's train split per worker) and `set_epoch` reaches persistent workers
through shared memory. Each worker costs memory; on a 16 GB machine with a
busy desktop, 2-3 workers ran it out of memory.

## Status and notes

- Building the track list reads every track's metadata: about a minute for
  Slakh's train split at start-up, once per worker.
- Separation quality beyond the run above has not been measured; the model, loss
  and metric are also verified on synthetic material (a tiny model learns two
  synthetic sources and improves SI-SDR over the mixture).
- Open: onset head, an STFT-free export for runtimes without an FFT, MUSDB18
  in the separation config (needs `mixer prepare-musdb` first), tempo/pitch
  augmentation in training (too slow on CPU).
