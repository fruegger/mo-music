# mixer

Turns multitrack stems (Slakh2100, MUSDB18) into randomised training snippets:
a mix plus the labels and the exact parameters that produced it.

Everything is a pure function of `(track, seed, MixConfig)`, so a snippet can
be regenerated bit-for-bit, validation sets are fixed by their seeds, and the
trainer can generate on the fly in dataloader workers.

```
mixer render --sources ../repo/prototype/sources.log --dataset slakh2100 \
             --split train --seeds 0-9 --out out/
```

writes `out/<track>_<seed>.wav` (the mix) and `.json` (labels + parameters).

## Layout

| module | role |
|---|---|
| `registry.py` | parses `repo/prototype/sources.log` (name, url, path, date, license) |
| `sources.py` | `Track`/`Stem`, loaders for Slakh2100 and MUSDB18 |
| `effects.py` | per-stem effect chain with seeded parameter sampling |
| `mix.py` | `MixConfig`, `render_snippet(track, seed, config)` |
| `cli.py` | `mixer render` |

## Labels

Per snippet: the instrument classes present, and per stem its class, gain and
effect parameters, plus the crop offset. Slakh gives exact classes
(`inst_class`); MUSDB18 only `drums`, `bass`, `other`, `vocals`.

## Not done yet

- onset labels (Slakh: from `MIDI/Sxx.mid`; MUSDB18: from stem energy)
- pitch shift and tempo change (they move onsets, so labels must be re-timed)
- mix-bus effects and loudness normalisation of the final mix
- the effect chain needs `pip install -e ".[effects]"` (pedalboard); without it
  only gain is applied. It is untested in this scaffold.
- MUSDB18 loading is untested (needs `stempeg` and ffmpeg)

## Licences

`mixer render` copies each dataset's licence string into the label file, and
`Dataset.non_commercial` flags it, so anything trained on MUSDB18 can be marked
non-commercial.
