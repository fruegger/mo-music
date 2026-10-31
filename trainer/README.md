# trainer

Trains audio-analysis models on snippets from [`../mixer`](../mixer). The
`task` in the config picks what is learned:

| task | config | model | status |
|---|---|---|---|
| `instruments` | `configs/default.yaml` | log-mel CNN, multi-label (which classes are in the mix) | runs end to end on Slakh |
| `separation` | `configs/separation.yaml` | spectrogram U-Net predicting one mask per source | runs end to end on Slakh; not trained beyond a few steps |

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
  between runs
- each run writes `run.json` with the dataset licences from `sources.log`; a
  run that used MUSDB18 is marked `non_commercial`

## Status and notes

- Building the track list reads every track's metadata: about a minute for
  Slakh's train split at start-up.
- Only a few training steps have been run (in tests). No separation quality
  has been measured: the model, loss and metric are verified on synthetic
  material (a tiny model learns two synthetic sources and improves SI-SDR over
  the mixture), not on real music.
- Open: onset head, an STFT-free export for runtimes without an FFT, MUSDB18
  in the separation config (needs `mixer prepare-musdb` first).
