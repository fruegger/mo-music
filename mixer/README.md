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

## Labels (schema 1)

One JSON object per snippet; the full field list is the docstring of
`mix.py`. The important points:

- **Reproducible:** `dataset`, `track`, `seed`, `config` (the whole
  `MixConfig`) plus `mixer_version` re-render the snippet:
  `render_snippet(track, labels["seed"], MixConfig.from_dict(labels["config"]))`.
- **Presence is measured, not assumed.** A stem is `active` only where its
  frame RMS in the final mix exceeds `activity_threshold_db`; `activity`
  holds the per-frame 0/1 strings and `classes` lists the classes with an
  active stem. A stem that is silent in the crop is not a positive.
- **One vocabulary** (`vocab.py`): Slakh2100's 13 classes plus `Vocals`.
  MUSDB18 `drums`/`bass`/`vocals` map onto it; its `other` stem is a mixture of
  unknown instruments, has `class: null` and sets `ambiguous: true` while
  audible. The trainer masks the classes it could contain out of the loss.
- **Finer identity:** `source_class` is the dataset's own name; `detail` is
  Slakh's MIDI program name (for example "Distortion Guitar").
- **Effects** are an ordered list per stem: `fired` (the random draw) and
  `applied` (whether it ran; false when pedalboard is missing).
- **Onsets** (`times`, seconds from the snippet start): `onsets` (all),
  `onsets_by_class`, and per stem under `stems[].onsets`. From Slakh's
  `MIDI/Sxx.mid` note-ons, following each file's tempo map. A note counts only
  if its stem is audible in the `onset_window_s` (50 ms) after it, so notes
  that the gain or a mute make inaudible are dropped. Slakh's MIDI times match
  its audio to about 1 ms on isolated drum hits (checked in
  `tests/test_onsets.py`). MUSDB18 has no MIDI: its stems have `onsets: null`
  and the snippet gets `onsets_complete: false` (do not use it for onset
  negatives). `mixer.targets.onset_target` turns the labels into frame targets.
- **Tempo and pitch** (`transform`, off unless `p_tempo` / `p_pitch` > 0): the
  whole snippet is sped up by `tempo_ratio` (log-uniform, 0.85-1.15 by
  default) so all stems stay in sync, and pitch-shifted by `pitch_semitones`
  (uniform, -3..3). Drums are not pitch-shifted unless `pitch_drums`. The
  crop is `seconds * tempo_ratio` long in the original track (`source_seconds`),
  and all `times` are in the new, output time. Rubber Band (through
  pedalboard) does the work; asking for it without pedalboard is an error, not
  a silent skip. The tempo/pitch draws come from their own random stream, so
  enabling them leaves gains, dropped stems and effects unchanged.
  Measured: onsets stay within ~3 ms (tempo) to ~10 ms (with pitch shift) of
  their label; pitch is exact in high quality mode. The fast mode
  (`--fast-stretch`) is ~2x quicker here but detunes by up to ~30 cents.
  **Cost:** ~1 s per stem, about 7 s per 8-stem snippet when every snippet
  is transformed (0.3 s without), single core.
- Seeds rendered before mixer 0.4.0 give different crops: the crop-start draw changed so tempo changes do not disturb the other random choices.
- `times.boundaries` is reserved for structure labels, empty for now.

`mixer render --stems` also writes each processed stem to `<name>.stems/` for
separation training.

## Not done yet

- onset labels for MUSDB18 (no MIDI; would have to come from stem energy)
- mix-bus effects and loudness normalisation of the final mix
- the effect chain needs `pip install -e ".[effects]"` (pedalboard 0.9.26 was
  tested); without it only gain is applied and the label records say so.
  Its tests (`tests/test_effects.py`) are skipped when pedalboard is missing.
- MUSDB18 loading is untested (needs `stempeg` and ffmpeg)

## Licences

`mixer render` copies each dataset's licence string into the label file, and
`Dataset.non_commercial` flags it, so anything trained on MUSDB18 can be marked
non-commercial.
