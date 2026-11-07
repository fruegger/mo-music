# MUSDB18 `other` stem tags

`musdb18-other-tags.csv`: one row per song (100 train, 50 test), tagged by listening to the song's `other.flac` stem
(`C:\workarea\MUSDB18_prepared\<split>\<song>\other.flac`). Done 2026-11-07; typos fixed afterwards.

## Columns

- `instruments`: every instrument heard, separated by `+` (no spaces around it), lower case, e.g. `guitar+synth+piano`.
- `dominant`: the one that carries the stem (optional, one name from `instruments`; filled for only a few songs).
- `unsure`: instruments that MAY be present but are not certain (e.g. `trumpet+sax`). Treat them as "no label either
  way" (weight 0 in the loss), not as present and not as absent.
- `notes`: free text (flavour such as `distorted`, `epiano`, `djent`, or timing such as `trumpet in second half`).

## Vocabulary used (counts over the 150 songs)

guitar 118, synth 47, piano 34, organ 12, violin 5, trumpet 3, vocals 3, vibraphone 3, strings 3, harp 2, clarinet 2,
and one each of synth pad, trombone, percussion, banjo, flute, accordion, sax, brass, bells. Only guitar, synth and
piano have enough songs for an output of their own; organ is borderline, the rest has to be grouped.

## Mapping to model classes (proposal, decide when training)

`mixer.vocab.CLASSES` names: violin and strings -> Strings; trumpet, trombone, brass -> Brass; sax, clarinet, oboe -> Reed;
flute -> Pipe; vibraphone and bells -> Chromatic Percussion; synth pad -> synth. Open: harp, banjo, accordion, percussion.
`vocals` in an `other` stem (ANiMAL - Rockshow samples, Lushlife - Toynbee Suite, Port St Willow - Stay Even leak) is not
an `other` class: ignore it.

## Known gaps

- Empty `instruments`: Grants - PunchDrunk (note: silence and noises, read as nothing present), Side Effects Project -
  Sing With Me (no note: unfinished?), Triviul feat. The Fiend - Widow (note `weird truff`, unclear).
- ANiMAL - Easy Tiger: the note mentions mellotron, tagged only `synth`.
- A stem label says an instrument is in the song somewhere, not in every 10 s snippet (weak, song-level labels).
