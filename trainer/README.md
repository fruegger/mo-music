# trainer

Trains audio-analysis models on snippets from [`../mixer`](../mixer). The
first task is instrument recognition (which instrument classes are in a mix);
source separation, onsets and structure come later as further heads/tasks.

```
pip install -e ../mixer[effects,musdb]
pip install -e .
trainer --config configs/default.yaml
```

## Layout

| module | role |
|---|---|
| `data.py` | `SnippetDataset`: a torch `Dataset` that calls `mixer.render_snippet` on the fly; fixed seeds for validation |
| `model.py` | small log-mel CNN, multi-label instrument head |
| `train.py` | training loop, validation, checkpointing, ONNX export |
| `configs/default.yaml` | datasets, sizes, seeds, optimiser |

## Contract with the mixer

- training: seed = epoch-dependent, so each epoch sees new mixes
- validation: fixed seeds (`val_seed_base + index`), so scores are comparable
  between runs
- each run writes `run.json` with the dataset licences from `sources.log`; a
  run that used MUSDB18 is marked `non_commercial`

## Status

Scaffold only. Nothing here has been run: torch is not installed on this
machine yet, so the code is untested. The class vocabulary comes from
`mixer.CLASSES`; the loss ignores classes whose target is unknown (snippets
with an audible MUSDB18 `other` stem). Open: onset/separation heads, ONNX
export check against `audan`.
